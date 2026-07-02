"""Multi-factor transparent scoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidate_profile import StructuredCandidateProfile
from .job_understanding import StructuredRoleProfile
from .skill_graph import concept_overlap, expand_skills


@dataclass
class FactorScores:
    technical_fit: float
    domain_experience: float
    leadership: float
    communication: float
    learning_ability: float
    culture_fit: float
    project_relevance: float
    behavioral: float
    final_score: float

    def to_dict(self) -> dict[str, int]:
        """Return 0-100 integer scores for recruiter dashboard."""
        return {
            "Technical Fit": round(self.technical_fit * 100),
            "Domain Experience": round(self.domain_experience * 100),
            "Leadership": round(self.leadership * 100),
            "Communication": round(self.communication * 100),
            "Learning Ability": round(self.learning_ability * 100),
            "Culture Fit": round(self.culture_fit * 100),
            "Project Relevance": round(self.project_relevance * 100),
            "Behavioral Signals": round(self.behavioral * 100),
            "Final Score": round(self.final_score * 100),
        }


FACTOR_WEIGHTS = {
    "technical_fit": 0.22,
    "domain_experience": 0.14,
    "leadership": 0.08,
    "communication": 0.07,
    "learning_ability": 0.10,
    "culture_fit": 0.10,
    "project_relevance": 0.17,
    "behavioral": 0.12,
}


def _scale(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def compute_factor_scores(
    role: StructuredRoleProfile,
    profile: StructuredCandidateProfile,
    features: dict[str, Any],
    semantic_similarity: float,
    honeypot: dict[str, Any],
) -> FactorScores:
    """Generate transparent multi-dimensional scores."""

    required_overlap, _ = concept_overlap(role.required_skills, profile.tech_stack)
    nice_overlap, _ = concept_overlap(role.nice_to_have_skills, profile.tech_stack)

    technical_fit = _scale(
        required_overlap * 0.45
        + nice_overlap * 0.15
        + features.get("skill_score", 0) * 0.25
        + semantic_similarity * 0.15
    )

    # Domain: product ML / recruiting / search industry alignment
    domain_text = f"{profile.industry} {' '.join(profile.projects)}".lower()
    domain_hits = sum(
        1
        for term in ("product", "recruiting", "search", "recommendation", "talent", "ml", "ai")
        if term in domain_text
    )
    domain_experience = _scale(
        domain_hits / 4.0 * 0.4
        + features.get("production_signals", 0) * 0.35
        + (1.0 - features.get("consulting_ratio", 0) * 0.6) * 0.25
    )

    leadership = _scale(
        min(len(profile.leadership_signals) / 3.0, 1.0) * 0.5
        + profile.career_growth_score * 0.3
        + (1.0 if role.leadership_requirements else 0.5) * 0.2
    )

    communication = _scale(
        min(len(profile.communication_signals) / 2.0, 1.0) * 0.4
        + features.get("response_rate", 0) * 0.35
        + (1.0 if len(profile.summary) > 80 else 0.4) * 0.25
    )

    # Learning: career growth, diverse stack, assessments, github
    learning_ability = _scale(
        profile.career_growth_score * 0.35
        + profile.open_source_score * 0.25
        + features.get("assessment_score", 0.5) * 0.25
        + min(len(profile.tech_stack) / 10.0, 1.0) * 0.15
    )

    trait_text = f"{profile.summary} {' '.join(profile.projects)}".lower()
    culture_hits = sum(1 for trait in role.culture_traits if any(w in trait_text for w in trait.lower().split()[:2]))
    inferred_hits = sum(1 for trait in role.inferred_traits if trait.split()[0] in trait_text)
    culture_fit = _scale(culture_hits / max(len(role.culture_traits), 1) * 0.5 + inferred_hits / max(len(role.inferred_traits), 1) * 0.5)

    project_text = " ".join(profile.projects + profile.achievements).lower()
    project_hits = sum(
        1
        for resp in role.responsibilities
        if any(w in project_text for w in resp.lower().split()[:3])
    )
    project_relevance = _scale(
        project_hits / max(len(role.responsibilities), 1) * 0.45
        + features.get("production_signals", 0) * 0.35
        + semantic_similarity * 0.20
    )

    behavioral = _scale(features.get("behavioral_score", 0.5))

    # Compose weighted final score
    raw_final = (
        technical_fit * FACTOR_WEIGHTS["technical_fit"]
        + domain_experience * FACTOR_WEIGHTS["domain_experience"]
        + leadership * FACTOR_WEIGHTS["leadership"]
        + communication * FACTOR_WEIGHTS["communication"]
        + learning_ability * FACTOR_WEIGHTS["learning_ability"]
        + culture_fit * FACTOR_WEIGHTS["culture_fit"]
        + project_relevance * FACTOR_WEIGHTS["project_relevance"]
        + behavioral * FACTOR_WEIGHTS["behavioral"]
    )

    # Title relevance and trap penalties
    raw_final *= features.get("title_relevance", 0.5) * 0.4 + 0.6
    raw_final *= features.get("keyword_stuffer_penalty", 1.0)
    raw_final *= features.get("research_penalty", 1.0)
    raw_final *= features.get("cv_only_penalty", 1.0)
    raw_final *= features.get("job_hopping_penalty", 1.0)
    raw_final *= honeypot.get("honeypot_penalty", 1.0)
    if honeypot.get("is_likely_honeypot"):
        raw_final *= 0.05

    years = profile.experience_years
    if role.min_years <= years <= role.max_years:
        raw_final *= 1.05
    elif years < role.min_years - 1:
        raw_final *= 0.7

    final_score = _scale(raw_final)

    return FactorScores(
        technical_fit=technical_fit,
        domain_experience=domain_experience,
        leadership=leadership,
        communication=communication,
        learning_ability=learning_ability,
        culture_fit=culture_fit,
        project_relevance=project_relevance,
        behavioral=behavioral,
        final_score=final_score,
    )
