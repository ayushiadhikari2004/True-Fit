"""Generate recruiter-trustworthy reasoning strings (delegates to explainer)."""

from __future__ import annotations

from typing import Any

from .explainer import build_explanation
from .job_understanding import understand_job
from .multi_factor_scorer import FactorScores


def build_reasoning(features: dict[str, Any], final: float) -> str:
    """Backward-compatible reasoning from feature bundle."""
    if "structured_profile" in features and "factor_scores" in features:
        role = understand_job(features.get("jd_text", ""))
        return build_explanation(
            rank=features.get("rank", 0),
            role=role,
            profile=features["structured_profile"],
            factors=features["factor_scores"],
            features=features,
        ).reasoning

    # Fallback for lightweight bundles
    title = features.get("title", "")
    years = features.get("years", 0)
    return f"{title} with {years:.1f} yrs; score {final:.2f}."
