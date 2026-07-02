#!/usr/bin/env python3
"""Generate submission CSV using rule-based scoring (wrapper around rank.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rule-based submission generator")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reports", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "rank.py"),
        "--candidates", args.candidates,
        "--out", args.out,
        "--mode", "rule",
        "--top-k", str(args.top_k),
        "--quiet",
    ]
    if args.reports:
        cmd.extend(["--reports", args.reports])

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
