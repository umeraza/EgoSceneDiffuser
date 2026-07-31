#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time

import torch

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataloader
from egoscenediffuser.models import EgoSceneDiffuser
from egoscenediffuser.training import move_batch
from egoscenediffuser.utils import resolve_device
from egoscenediffuser.utils.checkpoint import load_checkpoint


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    device = resolve_device(str(cfg.experiment.device))
    model = EgoSceneDiffuser(cfg).to(device).eval()
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    batch = move_batch(next(iter(build_dataloader(cfg, "test", batch_size=1, shuffle=False))), device)
    warmup = int(cfg.evaluation.runtime_warmup)
    iterations = int(cfg.evaluation.runtime_iterations)
    with torch.no_grad():
        for _ in range(warmup):
            model(batch, sample=True)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(batch, sample=True)
        synchronize(device)
    seconds = time.perf_counter() - start
    params = sum(parameter.numel() for parameter in model.parameters())
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    print(json.dumps({
        "latency_ms": 1000.0 * seconds / iterations,
        "sequences_per_second": iterations / seconds,
        "motion_frames_per_second": iterations * int(cfg.data.observed_frames + cfg.data.future_frames) / seconds,
        "parameters": params,
        "peak_gpu_memory_bytes": peak_memory,
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
