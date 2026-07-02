"""Rich recruiter candidate card format (Claude-style shortlist UI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .rule_scorer import CONSULTING_FIRMS, MUST_HAVE_SKILLS, get_text_blob

REFERENCE_DATE = date(2026, 5, 28)

DISPLAY_SKILL_TAGS = [
    ("embedding", "Embeddings"),
    ("sentence-transformer", "Embeddings"),
    ("bge", "BGE Embeddings"),
    ("e5", "E5 Embeddings"),
    ("faiss", "FAISS"),
    ("bm25", "BM25"),
    ("learning to rank", "LTR"),
    ("lambdarank", "LTR"),
    ("pinecone", "Pinecone"),
    ("weaviate", "Weaviate"),
    ("qdrant", "Qdrant"),
    ("milvus", "Milvus"),
    ("rag", "RAG"),
    ("hybrid search", "Hybrid Search"),
    ("ndcg", "NDCG"),
    ("vector search", "Vector Search"),
    ("elasticsearch", "Elasticsearch"),
    ("opensearch", "OpenSearch"),
    ("python", "Python"),
    ("fine-tuning", "Fine-tuning"),
    ("lora", "LoRA"),
]


@dataclass
class CandidateCard:
    rank: int
    candidate_id: str
    name: str
    title: str
    years: float
    company: str
    score: float
    location: str
    country: str
    active_label: str
    open_to_work: bool
    notice_days: int
    all_must_haves: bool
    skill_tags: list[str] = field(default_factory=list)
    summary: str = ""
    factor_scores: dict[str, int] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    # Enhanced signal fields
    badges: list[Any] = field(default_factory=list)
    salary_explanation: dict[str, Any] = field(default_factory=dict)
    acceptance_explanation: dict[str, Any] = field(default_factory=dict)
    verification_explanation: dict[str, Any] = field(default_factory=dict)
    completeness_explanation: dict[str, Any] = field(default_factory=dict)
    enhanced_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "name": self.name,
            "title": self.title,
            "years": self.years,
            "company": self.company,
            "score": self.score,
            "location": self.location,
            "country": self.country,
            "active_label": self.active_label,
            "open_to_work": self.open_to_work,
            "notice_days": self.notice_days,
            "all_must_haves": self.all_must_haves,
            "skill_tags": self.skill_tags,
            "summary": self.summary,
            "factor_scores": self.factor_scores,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "badges": [b.key for b in self.badges] if self.badges else [],
            "salary_explanation": self.salary_explanation,
            "acceptance_explanation": self.acceptance_explanation,
            "verification_explanation": self.verification_explanation,
            "completeness_explanation": self.completeness_explanation,
            "enhanced_detail": self.enhanced_detail,
        }


def _days_since_active(last_active: str | None) -> int | None:
    if not last_active:
        return None
    try:
        active_date = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - active_date).days
    except ValueError:
        return None


def _active_label(last_active: str | None) -> str:
    days = _days_since_active(last_active)
    if days is None:
        return "Activity unknown"
    if days <= 14:
        return f"Active {days}d ago"
    if days <= 30:
        return f"Active {days}d ago"
    if days <= 90:
        return f"Active {days}d ago"
    return f"Inactive {days}d"


def _is_product_company(company: str) -> bool:
    lowered = company.lower()
    return not any(cf in lowered for cf in CONSULTING_FIRMS)


def _extract_skill_tags(text: str, limit: int = 8) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for keyword, label in DISPLAY_SKILL_TAGS:
        if keyword in text and label not in seen:
            tags.append(label)
            seen.add(label)
        if len(tags) >= limit:
            break
    return tags


def _short_title(title: str) -> str:
    mapping = {
        "machine learning engineer": "MLE",
        "staff machine learning engineer": "Staff MLE",
        "senior machine learning engineer": "Senior MLE",
        "applied ml engineer": "Applied MLE",
        "ai engineer": "AI Engineer",
        "lead ai engineer": "Lead AI Engineer",
        "search engineer": "Search Engineer",
        "nlp engineer": "NLP Engineer",
        "recommendation systems engineer": "RecSys Engineer",
    }
    lowered = title.lower()
    for key, short in mapping.items():
        if key in lowered:
            return short
    return title


def build_card_summary(candidate: dict, flags: list[str], skill_tags: list[str]) -> str:
    """One-paragraph recruiter summary like the reference UI."""
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    exp = float(profile["years_of_experience"] or 0)
    title = _short_title(profile["current_title"])
    company = profile["current_company"]
    company_note = "(product co)" if _is_product_company(company) else ""

    parts: list[str] = [f"{exp:.0f}yr {title} at {company} {company_note}".strip()]

    text = get_text_blob(candidate)
    shipped: list[str] = []
    if any(k in text for k in ("ranking", "retrieval", "recommendation")):
        shipped.append("ranking + retrieval stack")
    if "faiss" in text and "bm25" in text:
        shipped.append("FAISS + BM25 hybrid")
    elif "faiss" in text:
        shipped.append("FAISS vector index")
    elif "bm25" in text:
        shipped.append("BM25 hybrid retrieval")
    if any(k in text for k in ("ndcg", "map", "mrr", "evaluation framework")):
        shipped.append("owns NDCG eval framework")
    if "rag" in text:
        shipped.append("RAG pipeline")
    if shipped:
        parts.append(f"shipped {'; '.join(shipped[:2])}")

    if "all_must_haves" in flags:
        parts.append("all must-haves covered")
    elif "preferred_location" in flags:
        parts.append("JD-preferred city")

    rr = float(signals.get("recruiter_response_rate") or 0)
    if rr >= 0.6:
        parts.append("high response rate")

    notice = int(signals.get("notice_period_days") or 90)
    if notice <= 30:
        parts.append("sub-30d notice")

    if skill_tags:
        parts.append("skills: " + ", ".join(skill_tags[:4]))

    return "; ".join(parts) + "."


def build_candidate_card(
    rank: int,
    candidate: dict,
    flags: list[str],
    norm_score: float,
    factor_scores: dict[str, int],
    strengths: list[str],
    gaps: list[str],
    badges: list[Any] | None = None,
    enhanced_result: Any | None = None,
) -> CandidateCard:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    text = get_text_blob(candidate)
    skill_tags = _extract_skill_tags(text)

    # Build enhanced detail explanation referencing actual data
    enhanced_detail = ""
    salary_exp = {}
    acceptance_exp = {}
    verification_exp = {}
    completeness_exp = {}

    if enhanced_result is not None:
        salary_exp = enhanced_result.salary_explanation
        acceptance_exp = enhanced_result.acceptance_explanation
        verification_exp = enhanced_result.verification_explanation
        completeness_exp = enhanced_result.completeness_explanation

        detail_parts = []
        if salary_exp.get("reason"):
            detail_parts.append(f" Salary: {salary_exp['reason']}")
        if acceptance_exp.get("reason"):
            detail_parts.append(f" Join: {acceptance_exp['reason']}")
        if verification_exp.get("reason"):
            detail_parts.append(f" Identity: {verification_exp['reason']}")
        if completeness_exp.get("reason"):
            detail_parts.append(f" Profile: {completeness_exp['reason']}")
        enhanced_detail = " · ".join(detail_parts)

    return CandidateCard(
        rank=rank,
        candidate_id=candidate["candidate_id"],
        name=profile.get("anonymized_name", "Candidate"),
        title=profile.get("current_title", ""),
        years=float(profile.get("years_of_experience") or 0),
        company=profile.get("current_company", ""),
        score=norm_score,
        location=profile.get("location", ""),
        country=profile.get("country", ""),
        active_label=_active_label(signals.get("last_active_date")),
        open_to_work=bool(signals.get("open_to_work_flag")),
        notice_days=int(signals.get("notice_period_days") or 90),
        all_must_haves="all_must_haves" in flags,
        skill_tags=skill_tags,
        summary=build_card_summary(candidate, flags, skill_tags),
        factor_scores=factor_scores,
        strengths=strengths,
        gaps=gaps,
        badges=badges or [],
        salary_explanation=salary_exp,
        acceptance_explanation=acceptance_exp,
        verification_explanation=verification_exp,
        completeness_explanation=completeness_exp,
        enhanced_detail=enhanced_detail,
    )
