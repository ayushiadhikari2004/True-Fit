"""Explainable AI reports for recruiter trust."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_profile import StructuredCandidateProfile
from .job_understanding import StructuredRoleProfile
from .multi_factor_scorer import FactorScores
from .skill_graph import concept_overlap, find_missing_skills


@dataclass
class ExplanationReport:
    rank: int
    candidate_id: str
    candidate_name: str
    summary: str
    strengths: list[str]
    gaps: list[str]
    factor_scores: dict[str, int]
    reasoning: str

    def to_markdown(self) -> str:
        lines = [
            f"### Why Ranked #{self.rank} — {self.candidate_name}",
            "",
            self.summary,
            "",
            "**Strengths**",
        ]
        for s in self.strengths:
            lines.append(f"- ✓ {s}")
        if self.gaps:
            lines.extend(["", "**Missing / Concerns**"])
            for g in self.gaps:
                lines.append(f"- • {g}")
        lines.extend(["", "**Factor Scores**"])
        for name, score in self.factor_scores.items():
            if name != "Final Score":
                lines.append(f"- {name}: {score}")
        lines.append(f"\n**Final Score: {self.factor_scores.get('Final Score', 0)}**")
        return "\n".join(lines)


def build_explanation(
    rank: int,
    role: StructuredRoleProfile,
    profile: StructuredCandidateProfile,
    factors: FactorScores,
    features: dict[str, Any],
    semantic_matches: list[tuple[str, str]] | None = None,
) -> ExplanationReport:
    """Build transparent why-ranked report with strengths and gaps."""
    strengths: list[str] = []
    gaps: list[str] = []

    years = profile.experience_years
    if role.min_years <= years <= role.max_years:
        strengths.append(f"{years:.1f} years experience in JD range ({role.min_years:.0f}-{role.max_years:.0f})")
    elif years >= role.min_years:
        strengths.append(f"{years:.1f} years experience")

    if features.get("title_relevance", 0) >= 0.8:
        strengths.append(f"{profile.current_title} aligns with {role.role_title}")
    elif features.get("title_relevance", 0) < 0.3:
        gaps.append(f"Title ({profile.current_title}) weakly aligned with role")

    _, skill_matches = concept_overlap(role.required_skills, profile.tech_stack)
    for req, matched in skill_matches[:4]:
        if req != matched:
            strengths.append(f"Semantic match: {matched} → {req}")
        else:
            strengths.append(f"Strong {req} experience")

    for achievement in profile.achievements[:2]:
        strengths.append(achievement.rstrip(".")[:100])

    for project in profile.projects[:1]:
        if any(w in project.lower() for w in ("ranking", "retrieval", "search", "recommendation", "embedding")):
            strengths.append(f"Built similar system: {project.split(':')[-1].strip()[:90]}")

    if profile.leadership_signals:
        strengths.append(f"Leadership: {profile.leadership_signals[0]}")

    if features.get("open_to_work"):
        strengths.append("Open to work on platform")
    if features.get("response_rate", 0) >= 0.6:
        strengths.append(f"Recruiter response rate {features['response_rate']:.0%}")

    top_skills = [s for s in profile.tech_stack if s][:3]
    if top_skills:
        strengths.append(f"Tech stack: {', '.join(top_skills)}")

    if features.get("production_signals", 0) >= 0.5:
        strengths.append("Production deployment experience in career history")

    # Gaps
    missing = find_missing_skills(role.required_skills, profile.tech_stack, limit=4)
    for skill in missing:
        gaps.append(skill)

    if features.get("consulting_ratio", 0) > 0.7:
        gaps.append("Career heavily weighted toward consulting/services")
    if features.get("notice_period_days", 90) > 60:
        gaps.append(f"{features['notice_period_days']}-day notice period")
    if features.get("response_rate", 0) < 0.25:
        gaps.append(f"Low recruiter response rate ({features['response_rate']:.0%})")
    if features.get("honeypot", {}).get("is_likely_honeypot"):
        gaps.append("Profile consistency issues detected")
    if features.get("keyword_stuffer_penalty", 1.0) < 0.2:
        gaps.append("Skills appear keyword-stuffed vs actual role")

    factor_dict = factors.to_dict()
    reasoning = _compose_reasoning(strengths, gaps, factors.final_score)

    return ExplanationReport(
        rank=rank,
        candidate_id=profile.candidate_id,
        candidate_name=profile.name,
        summary=profile.summary,
        strengths=strengths[:6],
        gaps=gaps[:4],
        factor_scores=factor_dict,
        reasoning=reasoning,
    )


def _compose_reasoning(strengths: list[str], gaps: list[str], final: float) -> str:
    """Short CSV-friendly reasoning string.

    `gaps` is a list of bare nouns/skill names (e.g. "embeddings-based
    retrieval"), not full clauses, so it can't be spliced directly after
    "though" the way `strengths` entries (already full phrases) can. Build an
    explicit "Gaps:" clause instead so the sentence stays grammatical
    regardless of what's in the gap list.
    """
    top = "; ".join(strengths[:3]) if strengths else "Partial fit"
    if not gaps:
        return f"{top}."
    gap_text = "; ".join(gaps[:2])
    if final < 0.6:
        return f"{top}. Gaps: {gap_text}."
    return f"{top}. Minor gaps: {gap_text}."
