#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json

import torch

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataloader
from egoscenediffuser.metrics import MetricAccumulator, evaluate_batch
from egoscenediffuser.models import EgoSceneDiffuser
from egoscenediffuser.training import move_batch
from egoscenediffuser.utils import resolve_device, seed_everything
from egoscenediffuser.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EgoSceneDiffuser")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.experiment.seed))
    device = resolve_device(str(cfg.experiment.device))
    model = EgoSceneDiffuser(cfg).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    try:
        loader = build_dataloader(cfg, args.split, batch_size=int(cfg.evaluation.batch_size), shuffle=False)
    except RuntimeError:
        # Synthetic CI data has equivalent generated samples for every split.
        loader = build_dataloader(cfg, "test", batch_size=int(cfg.evaluation.batch_size), shuffle=False)
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(batch, sample=True)
            accumulator.update(evaluate_batch(outputs, batch, cfg))
    metrics = accumulator.compute()
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    output = Path(args.output) if args.output else Path(cfg.experiment.output_dir) / f"metrics_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
