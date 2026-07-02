"""Integration tests for rule-based ranker."""

from __future__ import annotations

import json
from pathlib import Path

from src.rule_ranker import RuleBasedRanker, RuleRankerConfig

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"


def test_rule_ranker_produces_submission_shape():
    candidates = json.loads(SAMPLE.read_text(encoding="utf-8"))
    ranker = RuleBasedRanker(config=RuleRankerConfig(top_k=10, show_progress=False))
    result = ranker.rank_candidates_with_reports(candidates)
    assert len(result.dataframe) == 10
    assert list(result.dataframe.columns) == ["candidate_id", "rank", "score", "reasoning"]
    assert result.dataframe["score"].is_monotonic_decreasing
    assert len(result.explanations) == 10
    assert result.stats["mode"] == "rule"
