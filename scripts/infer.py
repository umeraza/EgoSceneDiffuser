#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import numpy as np
import torch

from egoscenediffuser.config import load_config
from egoscenediffuser.data.common import default_value, ensure_float_tensor, normalize_images, pad_or_trim_time
from egoscenediffuser.models import EgoSceneDiffuser
from egoscenediffuser.utils import resolve_device
from egoscenediffuser.utils.checkpoint import load_checkpoint


def load_window(path: Path, cfg) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        sample = {key: ensure_float_tensor(archive[key]) for key in archive.files}
    obs = int(cfg.data.observed_frames)
    total = obs + int(cfg.data.future_frames)
    lengths = {"sparse": obs, "history": obs, "trajectory": obs, "images": obs, "motion": total,
               "observed_motion": obs, "contacts": total, "sparse_positions": obs, "valid_mask": total}
    for key, length in lengths.items():
        sample[key] = pad_or_trim_time(sample.get(key, default_value(key, cfg)), length)
    sample["scene_points"] = pad_or_trim_time(sample.get("scene_points", default_value("scene_points", cfg)), int(cfg.data.scene_points))
    sample["images"] = normalize_images(sample["images"])
    return {key: value.unsqueeze(0) for key, value in sample.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EgoSceneDiffuser on one standardized NPZ window")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    device = resolve_device(str(cfg.experiment.device))
    model = EgoSceneDiffuser(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    batch = {key: value.to(device) for key, value in load_window(Path(args.input), cfg).items()}
    with torch.no_grad():
        outputs = model(batch, sample=True)
    np.savez_compressed(
        args.output,
        motion=outputs["motion"][0].cpu().numpy(),
        contact_probability=torch.sigmoid(outputs["contact_logits"])[0].cpu().numpy(),
        prior_mean=outputs["prior_mean"][0].cpu().numpy(),
        prior_sigma=outputs["prior_sigma"][0].cpu().numpy(),
    )


if __name__ == "__main__":
    main()
