"""Structured job description requirements for Senior AI Engineer role."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobRequirements:
    """Parsed requirements derived from the released JD."""

    role_title: str = "Senior AI Engineer"
    company: str = "Redrob AI"
    min_years: float = 4.0
    ideal_years_min: float = 5.0
    ideal_years_max: float = 9.0
    sweet_spot_years: float = 7.0

    # Core must-have skill themes (semantic + keyword)
    core_skill_themes: tuple[str, ...] = (
        "embeddings",
        "sentence-transformers",
        "vector search",
        "vector database",
        "retrieval",
        "hybrid search",
        "ranking",
        "recommendation system",
        "information retrieval",
        "python",
        "ndcg",
        "map",
        "mrr",
        "evaluation framework",
        "a/b testing",
        "rag",
        "llm",
        "fine-tuning",
        "machine learning",
        "nlp",
        "search",
        "pinecone",
        "weaviate",
        "faiss",
        "elasticsearch",
        "opensearch",
        "milvus",
        "qdrant",
        "learning to rank",
        "bm25",
    )

    relevant_titles: tuple[str, ...] = (
        "ai engineer",
        "ml engineer",
        "machine learning engineer",
        "senior machine learning engineer",
        "junior ml engineer",
        "applied scientist",
        "applied ml engineer",
        "data scientist",
        "senior data scientist",
        "nlp engineer",
        "research engineer",
        "software engineer",
        "backend engineer",
        "full stack engineer",
        "platform engineer",
        "search engineer",
        "recommendation engineer",
        "staff engineer",
        "principal engineer",
        "tech lead",
    )

    irrelevant_titles: tuple[str, ...] = (
        "marketing manager",
        "hr manager",
        "content writer",
        "graphic designer",
        "accountant",
        "sales executive",
        "customer support",
        "civil engineer",
        "mechanical engineer",
        "operations manager",
    )

    consulting_companies: tuple[str, ...] = (
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
        "hcl",
        "tech mahindra",
        "mindtree",
        "ltimindtree",
        "lti",
        "mphasis",
        "persistent",
        "cyient",
    )

    preferred_locations: tuple[str, ...] = (
        "pune",
        "noida",
        "delhi",
        "gurgaon",
        "gurugram",
        "ncr",
        "mumbai",
        "hyderabad",
        "bangalore",
        "bengaluru",
        "india",
    )

    jd_text: str = field(default="", repr=False)


JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI.
Own the intelligence layer: ranking, retrieval, and matching systems for recruiters and candidates.
Ship embeddings-based retrieval, hybrid search, and LLM re-ranking at production scale.
Strong Python. Experience with vector databases, sentence-transformers, BGE, E5, or OpenAI embeddings.
Design evaluation frameworks: NDCG, MAP, MRR, offline-to-online correlation, A/B testing.
Applied ML/AI at product companies (not pure research or consulting-only careers).
5-9 years experience; shipper mindset over pure researcher.
Disqualify: pure academic research without production, LangChain-only recent AI, senior without coding,
consulting-only careers, computer vision/speech/robotics without NLP/IR, title-chasing job hoppers.
Location Pune/Noida preferred; India (Hyderabad, Mumbai, Delhi NCR). Sub-30 day notice preferred.
Active on platform, responds to recruiters, open to work.
Look for end-to-end ranking, search, or recommendation systems shipped to real users.
"""


def load_job_requirements(jd_path: str | None = None) -> JobRequirements:
    """Load JD requirements. Optionally read an external JD file."""
    from .job_understanding import understand_job

    text = JD_TEXT
    if jd_path:
        from pathlib import Path

        path = Path(jd_path)
        if path.suffix.lower() == ".docx":
            text = _read_docx(path) or JD_TEXT
        elif path.exists():
            text = path.read_text(encoding="utf-8")

    role = understand_job(text.strip())
    return JobRequirements(
        role_title=role.role_title,
        min_years=role.min_years,
        ideal_years_min=role.min_years,
        ideal_years_max=role.max_years,
        sweet_spot_years=role.ideal_years,
        jd_text=text.strip(),
    )


def _read_docx(path) -> str | None:
    import zipfile
    import xml.etree.ElementTree as ET

    try:
        with zipfile.ZipFile(path) as zf:
            root = ET.fromstring(zf.read("word/document.xml"))
        parts: list[str] = []
        for paragraph in root.iter(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
        ):
            texts = [
                node.text
                for node in paragraph.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                )
                if node.text
            ]
            if texts:
                parts.append("".join(texts))
        return "\n".join(parts)
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        # zipfile.BadZipFile: not a real docx (e.g. truncated/renamed file)
        # KeyError: valid zip but missing word/document.xml (not a Word doc)
        # ET.ParseError: corrupted document.xml
        import warnings

        warnings.warn(
            f"Could not parse JD docx at {path}: {exc}. Falling back to the built-in JD summary.",
            stacklevel=2,
        )
        return None
