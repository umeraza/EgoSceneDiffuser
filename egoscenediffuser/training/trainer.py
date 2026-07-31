from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from egoscenediffuser.config import save_config
from egoscenediffuser.losses import CompositeLoss
from egoscenediffuser.metrics import MetricAccumulator, evaluate_batch
from egoscenediffuser.utils.checkpoint import save_checkpoint


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def configure_trainable_modules(model: nn.Module, module_names: list[str]) -> None:
    train_all = "all" in module_names
    aliases = {
        "sparse_encoder": "sparse_temporal.sparse_encoder",
        "history_encoder": "sparse_temporal.history_encoder",
        "trajectory_encoder": "sparse_temporal.trajectory_encoder",
        "temporal_encoder": "sparse_temporal.temporal_encoder",
        "prior": "prior",
    }
    prefixes = [aliases.get(name, name) for name in module_names]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(train_all or any(name.startswith(prefix) for prefix in prefixes))


def build_optimizer(model: nn.Module, cfg: Any) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters selected")
    return AdamW(parameters, lr=float(cfg.training.learning_rate), weight_decay=float(cfg.training.weight_decay))


class Trainer:
    def __init__(self, model: nn.Module, cfg: Any, device: torch.device) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.criterion = CompositeLoss(cfg)
        self.output_dir = Path(cfg.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.output_dir / "resolved_config.yaml")

    def _autocast(self):
        enabled = bool(self.cfg.training.amp) and self.device.type == "cuda"
        return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()

    def run_stage(self, stage: str, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, float]:
        stage_cfg = self.cfg.training.stages[stage]
        configure_trainable_modules(self.model, list(stage_cfg.train_modules))
        optimizer = build_optimizer(self.model, self.cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(self.cfg.training.amp) and self.device.type == "cuda")
        epochs = int(stage_cfg.epochs)
        best = float("inf")
        final_metrics: dict[str, float] = {}
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, optimizer, scaler, stage, epoch)
            val_metrics = self.evaluate(val_loader, stage) if val_loader is not None else {}
            final_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            final_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            monitor = val_metrics.get("loss", train_metrics["loss"])
            payload = {"stage": stage, "epoch": epoch, **final_metrics}
            print(json.dumps(payload, sort_keys=True))
            save_checkpoint(
                self.output_dir / f"{stage}_last.pt", self.model, optimizer, epoch, stage, final_metrics, self.cfg.to_dict()
            )
            if monitor < best:
                best = monitor
                save_checkpoint(
                    self.output_dir / f"{stage}_best.pt", self.model, optimizer, epoch, stage, final_metrics, self.cfg.to_dict()
                )
        return final_metrics

    def train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
        stage: str, epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        accumulator = MetricAccumulator()
        start = time.perf_counter()
        for step, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, self.device)
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = self.model.forward_prior(batch) if stage == "stage1" else self.model(batch, sample=False)
                losses = self.criterion(outputs, batch, stage)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.training.gradient_clip))
            scaler.step(optimizer)
            scaler.update()
            accumulator.update({"loss": losses["total"], **{k: v for k, v in losses.items() if k != "total"}})
            if step % int(self.cfg.training.log_every) == 0:
                elapsed = time.perf_counter() - start
                print(json.dumps({"stage": stage, "epoch": epoch, "step": step, "steps_per_s": step / elapsed}))
        return accumulator.compute()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, stage: str = "stage3") -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        accumulator = MetricAccumulator()
        for raw_batch in loader:
            batch = move_batch(raw_batch, self.device)
            if stage == "stage1":
                outputs = self.model.forward_prior(batch)
                losses = self.criterion(outputs, batch, stage)
                accumulator.update({"loss": losses["total"], **losses})
            else:
                outputs = self.model(batch, sample=True)
                losses = self.criterion(outputs, batch, stage)
                accumulator.update({"loss": losses["total"], **evaluate_batch(outputs, batch, self.cfg)})
        return accumulator.compute()
