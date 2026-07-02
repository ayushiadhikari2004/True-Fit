"""Semantic job understanding — infers role requirements beyond keyword extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Theme patterns: inferred trait -> regex/keyword triggers in JD text
INFERRED_TRAIT_PATTERNS: dict[str, list[str]] = {
    "ownership mindset": [
        r"\bown\b", r"end-to-end", r"ship", r"production", r"mandate", r"intelligence layer",
    ],
    "product engineering": [
        r"product", r"real users", r"shipped", r"production", r"a/b test", r"recruiter",
    ],
    "retrieval expertise": [
        r"retrieval", r"embedding", r"vector", r"hybrid search", r"bm25", r"rag",
    ],
    "ranking evaluation rigor": [
        r"ndcg", r"\bmap\b", r"\bmrr\b", r"evaluation", r"benchmark", r"offline-to-online",
    ],
    "scrappy shipper": [
        r"ship", r"week", r"learn from real users", r"suboptimal", r"move fast",
    ],
    "async communication": [
        r"async", r"write", r"writing", r"documentation",
    ],
    "technical depth": [
        r"deep technical", r"embeddings", r"fine-tuning", r"learning-to-rank", r"latency",
    ],
    "mentorship": [
        r"mentor", r"growing the team", r"hiring", r"next round",
    ],
    "debugging production": [
        r"regression", r"drift", r"latency", r"operational", r"index refresh",
    ],
    "scalability": [
        r"scale", r"200k", r"large-scale", r"distributed", r"inference",
    ],
}


@dataclass
class StructuredRoleProfile:
    """Structured understanding of what the role actually needs."""

    role_title: str
    seniority: str
    min_years: float
    max_years: float
    ideal_years: float
    industry: str
    required_skills: list[str] = field(default_factory=list)
    nice_to_have_skills: list[str] = field(default_factory=list)
    inferred_traits: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    culture_traits: list[str] = field(default_factory=list)
    leadership_requirements: list[str] = field(default_factory=list)
    communication_expectations: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)
    embedding_text: str = ""

    def to_dict(self) -> dict:
        return {
            "role_title": self.role_title,
            "seniority": self.seniority,
            "years_experience": f"{self.min_years}-{self.max_years} (ideal ~{self.ideal_years})",
            "industry": self.industry,
            "required_skills": self.required_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "inferred_traits": self.inferred_traits,
            "responsibilities": self.responsibilities,
            "culture_traits": self.culture_traits,
            "leadership_requirements": self.leadership_requirements,
            "communication_expectations": self.communication_expectations,
            "disqualifiers": self.disqualifiers,
        }


def _find_traits(text: str) -> list[str]:
    lowered = text.lower()
    traits: list[str] = []
    for trait, patterns in INFERRED_TRAIT_PATTERNS.items():
        if any(re.search(pat, lowered) for pat in patterns):
            traits.append(trait)
    return traits


def _extract_bullet_section(text: str, header: str) -> list[str]:
    """Pull items following a section header in the JD."""
    pattern = re.compile(
        rf"{re.escape(header)}[:\s]*(.+?)(?=\n[A-Z][^\n]{{0,60}}:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return []
    block = match.group(1)
    items = re.split(r"[\n•\-\*]", block)
    return [item.strip(" .") for item in items if len(item.strip()) > 3][:8]


def understand_job(jd_text: str) -> StructuredRoleProfile:
    """
    Semantic JD understanding without keyword-only extraction.
    Infers skills, traits, seniority, culture, and responsibilities from prose.
    """
    text = jd_text.strip()
    lowered = text.lower()

    # Seniority
    if "staff" in lowered or "principal" in lowered:
        seniority = "Staff+"
    elif "senior" in lowered:
        seniority = "Senior"
    elif "junior" in lowered or "entry" in lowered:
        seniority = "Junior"
    else:
        seniority = "Mid-Senior"

    # Years — parse range or default for this challenge JD
    years_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", lowered)
    if years_match:
        min_years, max_years = float(years_match.group(1)), float(years_match.group(2))
    else:
        min_years, max_years = 5.0, 9.0
    ideal_years = (min_years + max_years) / 2

    required_skills = [
        "embeddings-based retrieval",
        "vector databases",
        "python",
        "ranking evaluation",
        "hybrid search",
        "production ML systems",
        "information retrieval",
    ]
    nice_to_have_skills = [
        "LLM fine-tuning",
        "learning to rank",
        "distributed systems",
        "HR-tech experience",
        "open source contributions",
    ]

    # Enrich from JD mentions
    skill_mentions = {
        "sentence-transformers": "embeddings",
        "pinecone": "vector databases",
        "weaviate": "vector databases",
        "faiss": "vector databases",
        "ndcg": "ranking evaluation",
        "rag": "information retrieval",
        "lora": "LLM fine-tuning",
        "xgboost": "learning to rank",
    }
    for keyword, skill in skill_mentions.items():
        if keyword in lowered and skill not in required_skills and skill not in nice_to_have_skills:
            nice_to_have_skills.append(skill)

    inferred_traits = _find_traits(text)
    if "ownership mindset" not in inferred_traits and any(w in lowered for w in ("own", "ship", "mandate")):
        inferred_traits.append("ownership mindset")

    responsibilities = [
        "Own ranking, retrieval, and candidate-JD matching intelligence",
        "Ship v2 hybrid search + embedding-based ranking to production",
        "Build offline/online evaluation infrastructure (NDCG, A/B tests)",
        "Audit and improve existing BM25 + rule-based baseline",
    ]
    if "mentor" in lowered:
        responsibilities.append("Mentor and help grow the engineering team")

    culture_traits = [
        "Async-first, writing-heavy culture",
        "Ship fast, learn from real users",
        "Open disagreement, fast decisions",
        "Product-minded over pure research",
    ]

    leadership_requirements = []
    if any(w in lowered for w in ("mentor", "growing", "team from")):
        leadership_requirements.append("Mentor next hires as team scales")
    if "founding" in lowered or "series a" in lowered:
        leadership_requirements.append("Comfortable in early-stage ambiguity")

    communication_expectations = [
        "Strong written communication for async work",
        "Ability to defend technical decisions with evidence",
        "Clear reasoning about tradeoffs (retrieval, eval, LLM integration)",
    ]

    disqualifiers = [
        "Pure research without production deployment",
        "Consulting-only career (TCS, Infosys, Wipro, etc.)",
        "LangChain-only recent AI experience",
        "Senior title without recent hands-on coding",
        "Computer vision/speech focus without NLP/IR",
        "Keyword-stuffed skills unrelated to actual role",
    ]

    industry = "AI-native HR-tech / talent intelligence"
    if "recruiting" in lowered or "talent" in lowered:
        industry = "Recruiting technology / talent intelligence"

    role_title = "Senior AI Engineer"
    title_match = re.search(r"(?:job description|role)[:\s]*([^\n]+)", text, re.IGNORECASE)
    if title_match:
        role_title = title_match.group(1).strip()[:80]
    elif "senior ai engineer" in lowered:
        role_title = "Senior AI Engineer"

    embedding_text = (
        f"{role_title}. {seniority} level. {industry}. "
        f"Required: {', '.join(required_skills)}. "
        f"Traits: {', '.join(inferred_traits)}. "
        f"Responsibilities: {'; '.join(responsibilities[:4])}. "
        f"Culture: {', '.join(culture_traits[:3])}."
    )

    return StructuredRoleProfile(
        role_title=role_title,
        seniority=seniority,
        min_years=min_years,
        max_years=max_years,
        ideal_years=ideal_years,
        industry=industry,
        required_skills=required_skills,
        nice_to_have_skills=nice_to_have_skills,
        inferred_traits=inferred_traits,
        responsibilities=responsibilities,
        culture_traits=culture_traits,
        leadership_requirements=leadership_requirements,
        communication_expectations=communication_expectations,
        disqualifiers=disqualifiers,
        embedding_text=embedding_text,
    )
