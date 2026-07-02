"""Candidate loading utilities."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidates from JSONL/JSON/gzipped JSONL."""
    path = Path(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                yield item
            return
        yield payload
        return

    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".jsonl.gz") else open
    mode = "rt" if path.suffix in {".gz"} or path.name.endswith(".jsonl.gz") else "r"
    with opener(path, mode, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates(path: str | Path, limit: int | None = None) -> list[dict]:
    """Load candidates into memory (use limit for sandbox samples)."""
    results: list[dict] = []
    for idx, candidate in enumerate(iter_candidates(path)):
        results.append(candidate)
        if limit is not None and idx + 1 >= limit:
            break
    return results
