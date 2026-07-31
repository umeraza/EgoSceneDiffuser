#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataloader
from egoscenediffuser.models import EgoSceneDiffuser
from egoscenediffuser.training import Trainer
from egoscenediffuser.utils import resolve_device, seed_everything
from egoscenediffuser.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EgoSceneDiffuser")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default=None, choices=["stage1", "stage2", "stage3", "all"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--override", action="append", default=[], help="Dotted key=value override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    model = EgoSceneDiffuser(cfg)
    if args.resume:
        load_checkpoint(args.resume, model, map_location=device, strict=True)
    trainer = Trainer(model, cfg, device)
    requested = args.stage or str(cfg.training.stage)
    stages = ["stage1", "stage2", "stage3"] if requested == "all" else [requested]
    for stage in stages:
        stage_cfg = cfg.training.stages[stage]
        train_loader = build_dataloader(cfg, "train", batch_size=int(stage_cfg.batch_size), shuffle=True)
        try:
            val_loader = build_dataloader(cfg, "val", batch_size=int(cfg.evaluation.batch_size), shuffle=False)
        except (RuntimeError, FileNotFoundError):
            val_loader = None
        trainer.run_stage(stage, train_loader, val_loader)


if __name__ == "__main__":
    main()
