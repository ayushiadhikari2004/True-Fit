#!/usr/bin/env python3
"""Validate submission CSV format for the Redrob hackathon."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path


def load_candidate_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    is_gz = path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
    opener = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"

    # Plain .json (not .jsonl) is a JSON array, not newline-delimited records —
    # candidate_loader.py already handles this distinction; mirror it here so
    # validation works against both candidates.jsonl and sample_candidates.json.
    if path.suffix.lower() == ".json" and not is_gz:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            ids.add(record["candidate_id"])
        return ids

    with opener(path, mode, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ids.add(record["candidate_id"])
    return ids


def validate_submission(csv_path: Path, candidates_path: Path) -> list[str]:
    errors: list[str] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["candidate_id", "rank", "score", "reasoning"]
        if reader.fieldnames != expected:
            errors.append(f"Header must be exactly: {','.join(expected)}; got {reader.fieldnames}")
            return errors
        rows = list(reader)

    if len(rows) != 100:
        errors.append(f"Expected exactly 100 data rows, found {len(rows)}")

    ranks = []
    candidate_ids = []
    scores = []
    for idx, row in enumerate(rows, start=2):
        try:
            rank = int(row["rank"])
            score = float(row["score"])
        except ValueError:
            errors.append(f"Line {idx}: rank/score must be numeric")
            continue
        ranks.append(rank)
        candidate_ids.append(row["candidate_id"])
        scores.append(score)

    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be integers 1 through 100 exactly once")

    if len(set(candidate_ids)) != len(candidate_ids):
        errors.append("Duplicate candidate_id values found")

    for idx in range(1, len(scores)):
        if scores[idx] > scores[idx - 1] + 1e-9:
            errors.append(f"Scores must be non-increasing by rank; violation near rank {idx + 1}")

    pool_ids = load_candidate_ids(candidates_path)
    unknown = [cid for cid in candidate_ids if cid not in pool_ids]
    if unknown:
        errors.append(f"{len(unknown)} candidate_id values not found in candidate pool")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Redrob submission CSV")
    parser.add_argument("submission_csv")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    args = parser.parse_args()

    errors = validate_submission(Path(args.submission_csv), Path(args.candidates))
    if errors:
        print("INVALID submission:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("VALID submission format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
