#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or more ablation configurations")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--stage", default="all", choices=["stage1", "stage2", "stage3", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for config in args.configs:
        command = [sys.executable, "scripts/train.py", "--config", config, "--stage", args.stage,
                   "--override", f"experiment.output_dir=outputs/ablations/{Path(config).stem}"]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
