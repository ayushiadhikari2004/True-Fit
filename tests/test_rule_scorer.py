"""Tests for rule-based scorer."""

from __future__ import annotations

import json
from pathlib import Path

from src.rule_scorer import is_honeypot, normalize_scores, score_candidate

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"


def test_score_returns_result():
    candidates = json.loads(SAMPLE.read_text(encoding="utf-8"))
    result = score_candidate(candidates[0])
    assert isinstance(result.raw_score, float)
    assert isinstance(result.flags, list)


def test_normalize_scores_monotonic():
    fake = [(100 - i, {}, []) for i in range(100)]
    norms = normalize_scores(fake, top_n=100)
    assert norms[0] >= norms[-1]
    assert all(0 < s < 1 for s in norms)


def test_irrelevant_title_penalized():
    candidates = json.loads(SAMPLE.read_text(encoding="utf-8"))
    marketing = next(c for c in candidates if "marketing" in c["profile"]["current_title"].lower())
    engineer = next(
        c for c in candidates if "engineer" in c["profile"]["current_title"].lower()
    )
    m_score = score_candidate(marketing).raw_score
    e_score = score_candidate(engineer).raw_score
    assert e_score > m_score
