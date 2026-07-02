"""Rule-based JD scorer (explicit signals, distribution-tested)."""

from __future__ import annotations

from dataclasses import dataclass, field

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "ltimindtree", "lti", "mphasis",
}

MUST_HAVE_SKILLS = {
    "embeddings": [
        "embedding", "sentence-transformer", "sentence transformer", "openai embeddings",
        "bge", "e5", "word2vec", "dense retrieval",
    ],
    "vector_db": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "faiss", "vector db", "vector database", "vector search",
    ],
    "retrieval": [
        "retrieval", "semantic search", "bm25", "hybrid search", "information retrieval",
        " ir ", "rag", "ranking system", "ranking", "recommendation",
    ],
    "eval_frameworks": [
        "ndcg", "mrr", "map", "a/b test", "ab test", "evaluation framework",
        "offline eval", "online eval", "benchmark",
    ],
    "python": ["python"],
}

NICE_TO_HAVE = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning"],
    "ltr": ["learning to rank", "ltr", "lambdarank", "ranknet", "xgboost rank"],
    "distributed": ["distributed", "large-scale inference", "inference optimization", "serving", "triton"],
    "open_source": ["open-source contribution", "open source contribution", "github"],
}

IRRELEVANT_TITLES = {
    "marketing manager", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "content writer", "customer support", "graphic designer",
    "sales executive", "operations manager", "business development",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

AI_ML_INDICATORS = [
    "ml engineer", "ai engineer", "machine learning engineer", "applied scientist",
    "nlp engineer", "search engineer", "ranking engineer", "recommendation",
    "data scientist", "applied ml", "research engineer", "retrieval",
    "information retrieval", "senior machine learning", "staff machine learning",
]

PRODUCT_WORK_KEYWORDS = [
    "ranking", "search", "retrieval", "embedding", "recommendation", "llm",
    "model serving", "inference", "vector", "hybrid search", "ndcg",
]


def get_text_blob(candidate: dict) -> str:
    parts = [
        candidate["profile"]["headline"].lower(),
        candidate["profile"]["summary"].lower(),
        candidate["profile"]["current_title"].lower(),
    ]
    for job in candidate["career_history"]:
        parts.extend([
            job["title"].lower(),
            job["description"].lower(),
            job["company"].lower(),
        ])
    for skill in candidate["skills"]:
        parts.append(skill["name"].lower())
    for cert in candidate.get("certifications", []):
        if isinstance(cert, dict):
            parts.append(cert.get("name", "").lower())
    return " ".join(parts)


def is_honeypot(candidate: dict) -> bool:
    zero_duration_expert = sum(
        1
        for s in candidate["skills"]
        if s.get("proficiency") == "expert" and int(s.get("duration_months") or 0) == 0
    )
    if zero_duration_expert >= 3:
        return True

    exp_years = float(candidate["profile"]["years_of_experience"] or 0)
    for job in candidate["career_history"]:
        duration_years = int(job.get("duration_months") or 0) / 12.0
        if duration_years > exp_years + 2:
            return True

    # Timeline inconsistency: role months exceed date span
    for job in candidate["career_history"]:
        start = job.get("start_date")
        end = job.get("end_date")
        months = int(job.get("duration_months") or 0)
        if start and end and months:
            sy, sm = int(start[:4]), int(start[5:7])
            ey, em = int(end[:4]), int(end[5:7])
            actual = (ey - sy) * 12 + (em - sm)
            if actual >= 0 and months > actual + 3:
                return True

    return False


@dataclass
class RuleScoreResult:
    raw_score: float
    flags: list[str] = field(default_factory=list)
    is_honeypot: bool = False


def score_candidate(candidate: dict) -> RuleScoreResult:
    """Score candidate using explicit JD-aligned rules."""
    if is_honeypot(candidate):
        return RuleScoreResult(raw_score=-999.0, flags=["HONEYPOT"], is_honeypot=True)

    score = 0.0
    flags: list[str] = []
    text = get_text_blob(candidate)
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    current_title = profile["current_title"].lower()
    if any(t in current_title for t in IRRELEVANT_TITLES):
        score -= 40
        flags.append("irrelevant_title")

    exp = float(profile["years_of_experience"] or 0)
    if 6 <= exp <= 8:
        score += 20
        flags.append(f"{exp:.1f}yr_exp_sweet_spot")
    elif 5 <= exp <= 9:
        score += 15
        flags.append(f"{exp:.1f}yr_exp_good")
    elif 4 <= exp < 5 or 9 < exp <= 11:
        score += 8
    else:
        score += 2

    title_hits = sum(1 for t in AI_ML_INDICATORS if t in text)
    score += min(title_hits * 3, 15)

    product_company_roles = 0
    all_consulting = True
    for job in candidate["career_history"]:
        company = job["company"].lower()
        if not any(cf in company for cf in CONSULTING_FIRMS):
            all_consulting = False
            desc = job["description"].lower()
            if any(kw in desc for kw in PRODUCT_WORK_KEYWORDS):
                product_company_roles += 1

    if all_consulting and len(candidate["career_history"]) >= 2:
        score -= 20
        flags.append("all_consulting")
    score += min(product_company_roles * 5, 20)

    must_have_found = 0
    for category, keywords in MUST_HAVE_SKILLS.items():
        if any(kw in text for kw in keywords):
            must_have_found += 1
            weights = {"retrieval": 8, "embeddings": 7, "vector_db": 6, "eval_frameworks": 5, "python": 4}
            score += weights[category]
    if must_have_found == 5:
        score += 5
        flags.append("all_must_haves")

    for _category, keywords in NICE_TO_HAVE.items():
        if any(kw in text for kw in keywords):
            score += 2.5

    last_active = signals.get("last_active_date", "2020-01-01")
    if last_active >= "2026-05-01":
        score += 6
    elif last_active >= "2026-01-01":
        score += 4
    elif last_active >= "2025-07-01":
        score += 1
    else:
        score -= 5
        flags.append("inactive_6mo")

    if signals.get("open_to_work_flag"):
        score += 4

    rr = float(signals.get("recruiter_response_rate") or 0)
    score += rr * 5

    icr = float(signals.get("interview_completion_rate") or 0)
    score += icr * 3

    location = profile["location"].lower()
    country = profile["country"]
    if country == "India":
        score += 4
        if any(city in location for city in PREFERRED_LOCATIONS):
            score += 8
            flags.append("preferred_location")
        elif signals.get("willing_to_relocate"):
            score += 5
    elif not signals.get("willing_to_relocate"):
        score -= 5

    notice = int(signals.get("notice_period_days") or 90)
    if notice <= 30:
        score += 5
    elif notice <= 60:
        score += 3
    elif notice <= 90:
        score += 1
    else:
        score -= 2

    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            score += 5
            break
        if tier == "tier_2":
            score += 3
            break
        if tier == "tier_3":
            score += 1
            break

    assessments = signals.get("skill_assessment_scores") or {}
    relevant = [
        float(v)
        for k, v in assessments.items()
        if any(x in k.lower() for x in ["nlp", "machine learning", "python", "deep learning", "retrieval", "search"])
    ]
    if relevant:
        score += (sum(relevant) / len(relevant)) / 100 * 5

    github = signals.get("github_activity_score", 0)
    if github != -1:
        score += min(float(github) / 100 * 3, 3)

    return RuleScoreResult(raw_score=score, flags=flags)


def score_candidate_enhanced(
    candidate: dict,
    enhanced_config: "EnhancedSignalConfig | None" = None,
) -> "EnhancedRuleScoreResult":
    """Score candidate with base rule scoring + enhanced signals.

    This wraps the base ``score_candidate`` and adds salary fit, offer
    acceptance rate, verified identity, and profile completeness signals.
    The enhanced composite score is added on top of the raw score using
    configurable weights.
    """
    from .enhanced_signals import (
        EnhancedSignalConfig,
        EnhancedSignalResult,
        compute_enhanced_signals,
    )

    if enhanced_config is None:
        enhanced_config = EnhancedSignalConfig()

    base = score_candidate(candidate)
    if base.is_honeypot:
        return EnhancedRuleScoreResult(
            raw_score=base.raw_score,
            flags=base.flags,
            is_honeypot=True,
            enhanced=EnhancedSignalResult(),
        )

    enhanced = compute_enhanced_signals(candidate, enhanced_config)

    # Add enhanced composite (scaled to the same order of magnitude as base scoring)
    enhanced_bonus = enhanced.composite_score(enhanced_config) * 30.0
    total_score = base.raw_score + enhanced_bonus

    return EnhancedRuleScoreResult(
        raw_score=total_score,
        flags=base.flags,
        is_honeypot=False,
        enhanced=enhanced,
    )


@dataclass
class EnhancedRuleScoreResult:
    """Score result including both base and enhanced signal data."""
    raw_score: float
    flags: list[str] = field(default_factory=list)
    is_honeypot: bool = False
    enhanced: "EnhancedSignalResult | None" = None


def find_missing_must_haves(candidate: dict) -> list[str]:
    """Return must-have categories not semantically present in candidate profile."""
    text = get_text_blob(candidate)
    missing: list[str] = []
    labels = {
        "embeddings": "embeddings/retrieval models",
        "vector_db": "vector database",
        "retrieval": "retrieval/search",
        "eval_frameworks": "ranking evaluation (NDCG/MAP)",
        "python": "Python",
    }
    for category, keywords in MUST_HAVE_SKILLS.items():
        if not any(kw in text for kw in keywords):
            missing.append(labels.get(category, category))
    return missing


def build_reasoning(candidate: dict, flags: list[str]) -> str:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    exp = profile["years_of_experience"]
    title = profile["current_title"]
    location = profile["location"]
    country = profile["country"]
    active = signals.get("last_active_date", "N/A")
    notice = signals.get("notice_period_days", "N/A")
    rr = float(signals.get("recruiter_response_rate") or 0)

    if country == "India":
        reasoning = f"{exp:.0f}yr exp as {title} in {location}, India; "
    else:
        reasoning = f"{exp:.0f}yr exp as {title} in {location}; "

    if "all_must_haves" in flags:
        reasoning += "covers all core must-haves (embeddings, vector DB, retrieval, eval, Python); "
    elif "preferred_location" in flags:
        reasoning += "based in JD-preferred city; "

    reasoning += f"last active {active}, {rr * 100:.0f}% recruiter response rate, {notice}d notice."
    return reasoning[:220]


def normalize_scores(scored: list[tuple[float, dict, list[str]]], top_n: int = 100) -> list[float]:
    """Map raw scores to monotonic [0.001, 0.999] for submission."""
    top = scored[:top_n]
    max_score = top[0][0]
    min_score = top[-1][0]
    score_range = max_score - min_score if max_score != min_score else 1.0

    normalized: list[float] = []
    for raw, _, _ in top:
        norm = round(0.5 + 0.499 * (raw - min_score) / score_range, 4)
        normalized.append(max(0.001, min(0.999, norm)))
    return normalized
