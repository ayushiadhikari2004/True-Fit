"""Hybrid scoring that combines semantic and structured signals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .candidate_profile import StructuredCandidateProfile, parse_candidate
from .feature_engineering import extract_features
from .honeypot_detector import honeypot_features
from .job_understanding import StructuredRoleProfile
from .jd_parser import JobRequirements
from .multi_factor_scorer import FactorScores, compute_factor_scores


def heuristic_score(features: dict[str, Any], honeypot: dict[str, Any]) -> float:
    """Fast structured score for pre-filtering the full candidate pool."""
    consulting_penalty = 1.0 - min(features["consulting_ratio"] * 0.55, 0.55)

    score = (
        features["title_relevance"] * 0.24
        + features["years_score"] * 0.1
        + features["skill_score"] * 0.22
        + features["production_signals"] * 0.14
        + features["location_score"] * 0.06
        + features["behavioral_score"] * 0.12
        + features["assessment_score"] * 0.06
        + min(features["matched_ranking_skills"] / 5.0, 1.0) * 0.06
    )

    score *= consulting_penalty
    score *= features["research_penalty"]
    score *= features["job_hopping_penalty"]
    score *= features["keyword_stuffer_penalty"]
    score *= features["cv_only_penalty"]
    score *= honeypot["honeypot_penalty"]
    return float(np.clip(score, 0.0, 1.0))


def pseudo_rerank_boost(
    factors: FactorScores,
    semantic_similarity: float,
) -> float:
    """
    Simulates LLM re-ranking: cross-checks factor alignment with semantic fit.
    Lightweight, CPU-only, no API calls.
    """
    alignment = (
        factors.technical_fit * 0.3
        + factors.project_relevance * 0.3
        + factors.domain_experience * 0.2
        + semantic_similarity * 0.2
    )
    return float(np.clip(factors.final_score * 0.85 + alignment * 0.15, 0.0, 1.0))


def score_candidate_bundle(
    candidate: dict,
    requirements: JobRequirements,
    role: StructuredRoleProfile,
    semantic_similarity: float = 0.0,
) -> dict[str, Any]:
    """Full scoring bundle with multi-factor breakdown."""
    profile = parse_candidate(candidate)
    features = extract_features(candidate, requirements)
    features["text"] = profile.summary  # embedding uses recruiter summary
    honeypot = honeypot_features(candidate)
    features["honeypot"] = honeypot
    features["heuristic_score"] = heuristic_score(features, honeypot)

    factors = compute_factor_scores(role, profile, features, semantic_similarity, honeypot)
    final = pseudo_rerank_boost(factors, semantic_similarity)

    return {
        **features,
        "candidate": candidate,
        "structured_profile": profile,
        "factor_scores": factors,
        "semantic_similarity": semantic_similarity,
        "final_score": final,
    }


def build_feature_bundle(candidate: dict, requirements: JobRequirements, role: StructuredRoleProfile | None = None) -> dict[str, Any]:
    """Backward-compatible feature bundle for pass-1 prefilter."""
    from .job_understanding import understand_job

    if role is None:
        role = understand_job(requirements.jd_text)
    bundle = score_candidate_bundle(candidate, requirements, role, semantic_similarity=0.0)
    return bundle
