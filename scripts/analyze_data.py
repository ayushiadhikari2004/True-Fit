#!/usr/bin/env python3
"""Analyze candidate pool data distributions."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def analyze(path: Path, sample_size: int = 500) -> None:
    candidates = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= sample_size:
                break
            candidates.append(json.loads(line))

    print(f"Sample size: {len(candidates)}")
    print()

    locations = Counter(c["profile"]["country"] for c in candidates)
    print("Countries:", locations.most_common(10))

    india_locs = [c["profile"]["location"] for c in candidates if c["profile"]["country"] == "India"]
    print("Indian cities (sample):", Counter(india_locs).most_common(15))

    exps = [c["profile"]["years_of_experience"] for c in candidates]
    print(f"Exp: min={min(exps)}, max={max(exps)}, mean={statistics.mean(exps):.1f}, median={statistics.median(exps):.1f}")

    titles = Counter(c["profile"]["current_title"] for c in candidates)
    print("Top titles:", titles.most_common(20))

    # AI-relevant title share in sample
    ai_titles = sum(
        1
        for c in candidates
        if any(
            t in c["profile"]["current_title"].lower()
            for t in ("ai", "ml", "machine learning", "data scien", "nlp", "search")
        )
    )
    print(f"\nAI/ML-ish titles in sample: {ai_titles}/{len(candidates)} ({100*ai_titles/len(candidates):.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze candidate data distribution")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()
    analyze(Path(args.candidates), args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
