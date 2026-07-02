"""Tests for candidate card format."""

from __future__ import annotations

import json
from pathlib import Path

from src.candidate_card import build_candidate_card
from src.card_renderer import render_shortlist_html
from src.rule_scorer import score_candidate

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"


def test_candidate_card_has_required_fields():
    candidates = json.loads(SAMPLE.read_text(encoding="utf-8"))
    candidate = candidates[0]
    result = score_candidate(candidate)
    card = build_candidate_card(
        rank=1,
        candidate=candidate,
        flags=result.flags,
        norm_score=0.95,
        factor_scores={"Final Score": 95, "Technical Fit": 90},
        strengths=["test"],
        gaps=[],
    )
    assert card.name
    assert card.title
    assert card.score == 0.95
    assert card.active_label
    assert card.summary.endswith(".")


def test_html_renderer():
    candidates = json.loads(SAMPLE.read_text(encoding="utf-8"))
    candidate = candidates[0]
    result = score_candidate(candidate)
    card = build_candidate_card(1, candidate, result.flags, 0.9, {"Final Score": 90}, [], [])
    html = render_shortlist_html([card])
    assert "candidate-card" in html
    assert card.name in html
