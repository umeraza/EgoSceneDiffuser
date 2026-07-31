#!/usr/bin/env python3
"""Create one synthetic NPZ illustrating the data contract."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from egoscenediffuser.config import load_config
from egoscenediffuser.data.synthetic import SyntheticMotionDataset


def main() -> None:
    cfg = load_config("configs/smoke.yaml")
    sample = SyntheticMotionDataset(cfg)[0]
    output = Path("examples/standardized_sample.npz")
    arrays = {key: value.numpy() for key, value in sample.items() if hasattr(value, "numpy")}
    np.savez_compressed(output, **arrays)
    print(output)


if __name__ == "__main__":
    main()
