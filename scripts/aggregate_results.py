#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation JSON files into CSV")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for raw in args.inputs:
        path = Path(raw)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": path.parent.name, "file": str(path), **metrics})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
