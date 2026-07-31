#!/usr/bin/env python3
"""Validate and index author-preprocessed GIMO windows.

The paper's proxy HMD/controller extraction and head-centric joint mapping are not defined,
so this command only indexes windows that already follow docs/DATA.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-csv")
    args = parser.parse_args()
    command = [sys.executable, "scripts/build_manifest.py", "--root", args.processed_root,
               "--output", args.manifest, "--dataset", "gimo"]
    if args.split_csv:
        command += ["--split-csv", args.split_csv]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
