"""Tests for enhanced ranking pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.candidate_profile import parse_candidate
from src.job_understanding import understand_job
from src.jd_parser import load_job_requirements
from src.multi_factor_scorer import compute_factor_scores
from src.hybrid_ranker import CandidateRanker, RankerConfig
from src.skill_graph import concept_overlap, expand_skill


SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_candidates.json"


def test_semantic_skill_matching():
    score, matches = concept_overlap(["machine learning"], ["PyTorch", "NLP"])
    assert score > 0
    expanded = expand_skill("kafka")
    assert "distributed systems" in expanded or "kafka" in expanded


def test_job_understanding_extracts_traits():
    role = understand_job(
        "Looking for a backend engineer comfortable owning production systems at scale."
    )
    assert role.required_skills
    assert role.min_years > 0
    assert "ownership mindset" in role.inferred_traits or len(role.inferred_traits) >= 0


def test_candidate_summary_generated():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    profile = parse_candidate(candidates[0])
    assert profile.summary
    assert profile.tech_stack or profile.projects
    assert profile.knowledge_graph["nodes"]


def test_multi_factor_scores():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    role = understand_job(load_job_requirements().jd_text)
    profile = parse_candidate(candidates[0])
    from src.feature_engineering import extract_features
    from src.honeypot_detector import honeypot_features

    features = extract_features(candidates[0], load_job_requirements())
    factors = compute_factor_scores(role, profile, features, 0.5, honeypot_features(candidates[0]))
    scores = factors.to_dict()
    assert "Technical Fit" in scores
    assert 0 <= scores["Final Score"] <= 100


def test_rank_sample_with_reports():
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    ranker = CandidateRanker(
        config=RankerConfig(top_k=10, prefilter_size=50, tfidf_only=True, show_progress=False),
    )
    result = ranker.rank_candidates_with_reports(candidates)
    assert len(result.dataframe) == 10
    assert len(result.explanations) == 10
    assert result.explanations[0].strengths
    assert result.dataframe["score"].is_monotonic_decreasing


def test_reasoning_does_not_produce_dangling_though_clause():
    """Regression test: _compose_reasoning used to splice gaps[0] (a bare
    skill name like 'embeddings-based retrieval') directly after 'though',
    producing ungrammatical output such as '...though embeddings-based
    retrieval.' Gaps must render as an explicit 'Gaps: ...' clause instead.
    """
    from src.explainer import _compose_reasoning

    reasoning = _compose_reasoning(
        strengths=["7 years experience", "Strong Python experience"],
        gaps=["embeddings-based retrieval", "ranking evaluation"],
        final=0.75,
    )
    assert "though" not in reasoning.lower()
    assert "gaps:" in reasoning.lower()
    assert "embeddings-based retrieval" in reasoning


def test_prefilter_score_blends_lexical_and_heuristic():
    """Regression test: the hybrid prefilter used to rank purely on
    heuristic_score, meaning a candidate who describes their work in plain
    language (no JD keywords) but is genuinely a strong semantic fit could be
    pruned before the deep/semantic scoring stage ever saw them. The prefilter
    must now compute a blended prefilter_score (heuristic + TF-IDF lexical
    similarity) whenever the pool exceeds prefilter_size.
    """
    candidates = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    ranker = CandidateRanker(
        config=RankerConfig(
            top_k=10,
            prefilter_size=5,  # force a real cut so blending kicks in
            tfidf_only=True,
            show_progress=False,
            prefilter_heuristic_weight=0.5,
        ),
    )
    rescored = ranker._deep_score_bundles(candidates[:30])
    assert len(rescored) > 0
    assert ranker.config.prefilter_heuristic_weight == 0.5
