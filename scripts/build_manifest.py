#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"sparse", "history", "trajectory", "scene_points", "motion"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manifest from standardized NPZ windows")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--default-split", default="train")
    parser.add_argument("--split-csv", help="CSV with sequence_id,split columns")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    split_map = {}
    if args.split_csv:
        split_frame = pd.read_csv(args.split_csv)
        split_map = dict(zip(split_frame["sequence_id"].astype(str), split_frame["split"].astype(str)))
    rows = []
    errors = []
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                missing = REQUIRED.difference(archive.files)
            if missing:
                errors.append(f"{path}: missing {sorted(missing)}")
                continue
        except Exception as exc:  # corrupted archive
            errors.append(f"{path}: {exc}")
            continue
        sequence_id = path.stem
        rows.append({
            "path": str(path.relative_to(root)),
            "sequence_id": sequence_id,
            "dataset": args.dataset,
            "split": split_map.get(sequence_id, args.default_split),
        })
    if errors:
        raise SystemExit("Invalid files:\n" + "\n".join(errors[:50]))
    if not rows:
        raise SystemExit(f"No standardized NPZ windows found under {root}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
