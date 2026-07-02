"""Candidate text representation and structured feature extraction."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from .jd_parser import JobRequirements

PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.8,
    "expert": 1.0,
}

CV_ONLY_SKILLS = {
    "image classification",
    "object detection",
    "speech recognition",
    "tts",
    "computer vision",
    "robotics",
    "yolo",
    "opencv",
}

RANKING_SKILLS = {
    "ndcg",
    "map",
    "mrr",
    "learning to rank",
    "ranking",
    "recommendation",
    "information retrieval",
    "hybrid search",
    "bm25",
    "retrieval",
    "vector search",
    "embeddings",
    "faiss",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "rag",
    "sentence-transformers",
    "bge",
    "e5",
}

AI_CORE_SKILLS = RANKING_SKILLS | {
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "fine-tuning",
    "fine-tuning llms",
    "lora",
    "qlora",
    "pytorch",
    "transformers",
    "xgboost",
    "python",
}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def candidate_document(candidate: dict) -> str:
    """Build a rich text document for semantic matching."""
    profile = candidate["profile"]
    chunks = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_company", ""),
        profile.get("current_industry", ""),
        profile.get("location", ""),
    ]

    for role in candidate.get("career_history", []):
        chunks.extend(
            [
                role.get("title", ""),
                role.get("company", ""),
                role.get("industry", ""),
                role.get("description", ""),
            ]
        )

    for edu in candidate.get("education", []):
        chunks.extend(
            [
                edu.get("degree", ""),
                edu.get("field_of_study", ""),
                edu.get("institution", ""),
            ]
        )

    for skill in candidate.get("skills", []):
        chunks.append(
            f"{skill.get('name', '')} ({skill.get('proficiency', '')}, "
            f"{skill.get('duration_months', 0)} months)"
        )

    for cert in candidate.get("certifications", []):
        chunks.append(f"{cert.get('name', '')} from {cert.get('issuer', '')}")

    return "\n".join(part for part in chunks if part).strip()


def _skill_names(candidate: dict) -> list[str]:
    return [s.get("name", "").strip() for s in candidate.get("skills", []) if s.get("name")]


def _normalized_skills(candidate: dict) -> set[str]:
    return {name.lower() for name in _skill_names(candidate)}


def _title_relevance(title: str, requirements: JobRequirements) -> float:
    lowered = title.lower().strip()
    for bad in requirements.irrelevant_titles:
        if bad in lowered:
            return 0.05
    for good in requirements.relevant_titles:
        if good in lowered:
            return 1.0
    if any(token in lowered for token in ("engineer", "scientist", "developer", "architect")):
        return 0.45
    return 0.2


def _years_score(years: float, requirements: JobRequirements) -> float:
    if years < requirements.min_years:
        return max(0.1, years / requirements.min_years * 0.5)
    if requirements.ideal_years_min <= years <= requirements.ideal_years_max:
        distance = abs(years - requirements.sweet_spot_years)
        return 1.0 - min(distance / 4.0, 0.35)
    if years > requirements.ideal_years_max:
        return max(0.55, 1.0 - (years - requirements.ideal_years_max) * 0.05)
    return 0.65


def _consulting_ratio(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 0.0
    consulting_months = 0
    total_months = 0
    for role in history:
        months = int(role.get("duration_months") or 0)
        total_months += months
        blob = f"{role.get('company', '')} {role.get('industry', '')}".lower()
        if any(term in blob for term in ("consult", "it services", "outsourc", "staffing")):
            consulting_months += months
    if total_months == 0:
        return 0.0
    return consulting_months / total_months


def _production_signals(text: str) -> float:
    markers = [
        "production",
        "deployed",
        "shipped",
        "serving",
        "users",
        "a/b test",
        "ab test",
        "online",
        "latency",
        "index refresh",
        "embedding drift",
        "retrieval quality",
        "recommendation",
        "ranking system",
        "search system",
        "vector search",
        "hybrid retrieval",
    ]
    hits = sum(1 for marker in markers if marker in text.lower())
    return min(hits / 6.0, 1.0)


def _research_only_penalty(text: str, title: str) -> float:
    lowered = text.lower()
    research_hits = sum(
        1
        for term in ("published paper", "phd", "thesis", "academic lab", "university research")
        if term in lowered
    )
    production_hits = sum(
        1 for term in ("production", "deployed", "shipped", "product") if term in lowered
    )
    if research_hits >= 2 and production_hits == 0:
        return 0.35
    if "research scientist" in title.lower() and production_hits == 0:
        return 0.4
    return 1.0


def _job_hopping_penalty(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    short_stints = sum(1 for role in history if int(role.get("duration_months") or 0) < 18)
    if len(history) >= 4 and short_stints >= 3:
        return 0.55
    if short_stints >= 2 and len(history) >= 3:
        return 0.75
    return 1.0


def _skill_match_score(candidate: dict) -> tuple[float, int, int]:
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, 0, 0

    matched_core = 0
    matched_ranking = 0
    weighted = 0.0
    total_weight = 0.0

    for skill in skills:
        name = skill.get("name", "").lower()
        prof = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "intermediate"), 0.5)
        months = int(skill.get("duration_months") or 0)
        duration_factor = min(months / 24.0, 1.0) if months else 0.15
        weight = prof * (0.35 + 0.65 * duration_factor)
        total_weight += weight

        if any(core in name or name in core for core in AI_CORE_SKILLS):
            matched_core += 1
            weighted += weight
        if any(rank in name or name in rank for rank in RANKING_SKILLS):
            matched_ranking += 1
            weighted += weight * 1.2

    if total_weight == 0:
        return 0.0, matched_core, matched_ranking
    density = weighted / total_weight
    breadth = min(matched_core / 8.0, 1.0)
    ranking_bonus = min(matched_ranking / 4.0, 1.0) * 0.25
    return min(density * 0.55 + breadth * 0.45 + ranking_bonus, 1.0), matched_core, matched_ranking


def _keyword_stuffer_penalty(candidate: dict, matched_core: int) -> float:
    title = candidate["profile"].get("current_title", "").lower()
    if matched_core >= 7 and any(bad in title for bad in ("marketing", "hr", "sales", "accountant", "designer")):
        return 0.08
    skills = candidate.get("skills", [])
    if len(skills) >= 10 and matched_core >= 8:
        expert_zero = sum(
            1
            for s in skills
            if s.get("proficiency") == "expert" and int(s.get("duration_months") or 0) < 3
        )
        if expert_zero >= 3:
            return 0.05
    return 1.0


def _cv_only_penalty(candidate: dict) -> float:
    skills = _normalized_skills(candidate)
    cv_hits = len(skills & CV_ONLY_SKILLS)
    ir_hits = len(skills & RANKING_SKILLS)
    if cv_hits >= 3 and ir_hits <= 1:
        return 0.25
    return 1.0


def _location_score(candidate: dict, requirements: JobRequirements) -> float:
    blob = (
        f"{candidate['profile'].get('location', '')} "
        f"{candidate['profile'].get('country', '')}"
    ).lower()
    if any(loc in blob for loc in requirements.preferred_locations):
        return 1.0
    if candidate["profile"].get("country", "").lower() not in {"india", "in"}:
        return 0.35
    return 0.55


def _behavioral_score(signals: dict) -> float:
    if not signals:
        return 0.5

    response = float(signals.get("recruiter_response_rate") or 0)
    interview = float(signals.get("interview_completion_rate") or 0)
    completeness = float(signals.get("profile_completeness_score") or 0) / 100.0
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.55

    last_active = _parse_date(signals.get("last_active_date"))
    recency = 0.5
    if last_active:
        days = (date(2025, 5, 28) - last_active).days
        if days <= 14:
            recency = 1.0
        elif days <= 60:
            recency = 0.85
        elif days <= 180:
            recency = 0.55
        else:
            recency = 0.2

    notice = int(signals.get("notice_period_days") or 90)
    notice_score = 1.0 if notice <= 30 else 0.75 if notice <= 60 else 0.5 if notice <= 90 else 0.35

    saved = min(int(signals.get("saved_by_recruiters_30d") or 0) / 5.0, 1.0)
    github = signals.get("github_activity_score", -1)
    github_score = 0.5 if github == -1 else min(float(github) / 100.0, 1.0)

    verified = sum(
        1
        for key in ("verified_email", "verified_phone", "linkedin_connected")
        if signals.get(key)
    ) / 3.0

    return (
        response * 0.22
        + recency * 0.18
        + open_to_work * 0.12
        + interview * 0.1
        + notice_score * 0.12
        + completeness * 0.08
        + saved * 0.08
        + github_score * 0.05
        + verified * 0.05
    )


def _assessment_score(signals: dict, candidate: dict) -> float:
    assessments = signals.get("skill_assessment_scores") or {}
    if not assessments:
        return 0.5
    relevant = []
    skill_names = _normalized_skills(candidate)
    for skill_name, score in assessments.items():
        lowered = skill_name.lower()
        if any(token in lowered for token in skill_names) or any(
            theme in lowered for theme in ("python", "ml", "nlp", "retrieval", "ranking", "embedding")
        ):
            relevant.append(float(score) / 100.0)
    if not relevant:
        return sum(float(v) for v in assessments.values()) / (100.0 * len(assessments))
    return sum(relevant) / len(relevant)


def extract_features(candidate: dict, requirements: JobRequirements) -> dict[str, Any]:
    """Extract interpretable features used by the hybrid ranker."""
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    text = candidate_document(candidate)
    title = profile.get("current_title", "")
    years = float(profile.get("years_of_experience") or 0)

    skill_score, matched_core, matched_ranking = _skill_match_score(candidate)

    return {
        "candidate_id": candidate["candidate_id"],
        "text": text,
        "title": title,
        "years": years,
        "title_relevance": _title_relevance(title, requirements),
        "years_score": _years_score(years, requirements),
        "skill_score": skill_score,
        "matched_core_skills": matched_core,
        "matched_ranking_skills": matched_ranking,
        "consulting_ratio": _consulting_ratio(candidate),
        "production_signals": _production_signals(text),
        "research_penalty": _research_only_penalty(text, title),
        "job_hopping_penalty": _job_hopping_penalty(candidate),
        "keyword_stuffer_penalty": _keyword_stuffer_penalty(candidate, matched_core),
        "cv_only_penalty": _cv_only_penalty(candidate),
        "location_score": _location_score(candidate, requirements),
        "behavioral_score": _behavioral_score(signals),
        "assessment_score": _assessment_score(signals, candidate),
        "notice_period_days": int(signals.get("notice_period_days") or 90),
        "response_rate": float(signals.get("recruiter_response_rate") or 0),
        "open_to_work": bool(signals.get("open_to_work_flag")),
        "current_company": profile.get("current_company", ""),
        "location": profile.get("location", ""),
        "summary_snippet": profile.get("summary", "")[:180],
    }
