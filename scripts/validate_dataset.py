#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate standardized dataset tensor shapes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.split)
    checked = min(args.samples, len(dataset))
    for index in range(checked):
        sample = dataset[index]
        assert sample["sparse"].shape == (cfg.data.observed_frames, cfg.data.sparse_dim)
        assert sample["history"].shape[:2] == (cfg.data.observed_frames, cfg.data.num_joints)
        assert sample["motion"].shape[:2] == (cfg.data.observed_frames + cfg.data.future_frames, cfg.data.num_joints)
        assert sample["scene_points"].shape == (cfg.data.scene_points, 3)
    print(json.dumps({"dataset_size": len(dataset), "checked": checked, "status": "ok"}))


if __name__ == "__main__":
    main()
