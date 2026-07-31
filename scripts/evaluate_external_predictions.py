#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataset
from egoscenediffuser.metrics import MetricAccumulator, evaluate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate externally generated baseline predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.split)
    prediction_root = Path(args.predictions)
    accumulator = MetricAccumulator()
    missing: list[str] = []
    for sample in dataset:
        sequence_id = str(sample["sequence_id"])
        path = prediction_root / f"{sequence_id}.npz"
        if not path.exists():
            missing.append(sequence_id)
            continue
        with np.load(path, allow_pickle=False) as archive:
            motion = torch.from_numpy(archive["motion"]).float().unsqueeze(0)
            if motion.shape[1:] != sample["motion"].shape:
                raise ValueError(f"{path}: motion shape {tuple(motion.shape[1:])} != {tuple(sample['motion'].shape)}")
            if "contact_probability" in archive:
                probability = torch.from_numpy(archive["contact_probability"]).float().clamp(1e-6, 1 - 1e-6)
                contact_logits = torch.logit(probability).unsqueeze(0)
            else:
                contact_logits = torch.zeros(1, motion.shape[1], len(cfg.data.contact_joints))
            sigma = (
                torch.from_numpy(archive["uncertainty"]).float().unsqueeze(0)
                if "uncertainty" in archive
                else torch.ones_like(motion) * 0.1
            )
        batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        outputs = {"motion": motion, "contact_logits": contact_logits, "prior_sigma": sigma}
        accumulator.update(evaluate_batch(outputs, batch, cfg))
    if missing:
        raise SystemExit(f"Missing {len(missing)} predictions; first IDs: {missing[:10]}")
    metrics = accumulator.compute()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
