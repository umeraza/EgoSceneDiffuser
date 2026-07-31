#!/usr/bin/env python3
"""Index author-preprocessed EgoBody windows without redistributing licensed data.

This script intentionally does not fabricate a raw EgoBody-to-SMPL-X conversion. The
manuscript omits the target identity selection, calibration chain, head/hand tracker
construction, joint subset, and scene-collision representation needed for that conversion.
Create standardized NPZ windows following docs/DATA.md, then use this script to validate
and index them. Official sequence splits may be supplied as a CSV.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--official-split-csv")
    args = parser.parse_args()
    command = [sys.executable, "scripts/build_manifest.py", "--root", args.processed_root,
               "--output", args.manifest, "--dataset", "egobody"]
    if args.official_split_csv:
        command += ["--split-csv", args.official_split_csv]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
