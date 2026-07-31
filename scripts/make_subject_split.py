#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse

import pandas as pd

from egoscenediffuser.data.splits import subject_disjoint_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible non-official subject split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    if "subject_id" not in frame:
        raise SystemExit("Manifest must contain subject_id")
    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("Split ratios must sum to 1")
    mapping = subject_disjoint_split(frame["subject_id"].astype(str).tolist(), ratios, args.seed)
    frame["split"] = frame["subject_id"].astype(str).map(mapping)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote generated subject-disjoint split to {output}; this is not an official dataset split.")


if __name__ == "__main__":
    main()
