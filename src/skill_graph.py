"""Semantic skill hierarchy for embedding-style matching beyond exact keywords."""

from __future__ import annotations

# Parent concept -> related skills/technologies (bidirectional inference)
SKILL_CLUSTERS: dict[str, set[str]] = {
    "machine learning": {
        "machine learning", "ml", "deep learning", "pytorch", "tensorflow", "sklearn",
        "scikit-learn", "xgboost", "neural networks", "transformers", "hugging face",
        "fine-tuning", "fine-tuning llms", "lora", "qlora", "peft", "model training",
    },
    "nlp": {
        "nlp", "natural language processing", "llm", "rag", "text classification",
        "named entity recognition", "sentiment analysis", "prompt engineering",
        "information extraction", "language models",
    },
    "information retrieval": {
        "information retrieval", "retrieval", "search", "vector search", "hybrid search",
        "bm25", "elasticsearch", "opensearch", "semantic search", "ranking",
        "learning to rank", "recommendation", "recommendation system",
    },
    "embeddings": {
        "embeddings", "sentence-transformers", "vector embeddings", "bge", "e5",
        "openai embeddings", "embedding drift", "dense retrieval",
    },
    "vector databases": {
        "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "chroma", "annoy", "hnsw",
    },
    "distributed systems": {
        "distributed systems", "kafka", "redis", "spark", "apache beam", "airflow",
        "microservices", "scalability", "high availability", "load balancing",
    },
    "backend engineering": {
        "backend", "api", "rest", "graphql", "python", "java", "spring", "flask",
        "fastapi", "django", "node.js", "go", "system design",
    },
    "devops": {
        "devops", "docker", "kubernetes", "ci/cd", "aws", "gcp", "azure", "terraform",
        "monitoring", "observability", "production deployment",
    },
    "data engineering": {
        "data engineering", "etl", "data pipeline", "sql", "spark", "dbt", "snowflake",
        "data warehouse", "streaming",
    },
    "evaluation": {
        "evaluation", "ndcg", "map", "mrr", "precision", "recall", "a/b testing",
        "offline evaluation", "online evaluation", "benchmark",
    },
    "ownership": {
        "ownership", "end-to-end", "shipped", "production", "deployed", "on-call",
        "incident response", "sla", "maintained", "owned",
    },
    "leadership": {
        "leadership", "mentored", "led team", "tech lead", "architect", "managed",
        "hiring", "cross-functional", "stakeholder",
    },
    "communication": {
        "communication", "documentation", "technical writing", "presented", "blog",
        "async", "collaboration", "pair programming",
    },
}

# Reverse index: skill -> parent concepts
_SKILL_TO_CONCEPTS: dict[str, set[str]] = {}
for concept, members in SKILL_CLUSTERS.items():
    for member in members:
        key = member.lower()
        _SKILL_TO_CONCEPTS.setdefault(key, set()).add(concept)
        _SKILL_TO_CONCEPTS[key].add(key)


def normalize_skill(name: str) -> str:
    return name.strip().lower()


def expand_skill(skill: str) -> set[str]:
    """Expand a skill to its semantic cluster."""
    key = normalize_skill(skill)
    concepts = _SKILL_TO_CONCEPTS.get(key, {key})
    expanded: set[str] = {key}
    for concept in concepts:
        expanded.update(SKILL_CLUSTERS.get(concept, set()))
        expanded.add(concept)
    return expanded


def expand_skills(skills: list[str]) -> set[str]:
    result: set[str] = set()
    for skill in skills:
        result.update(expand_skill(skill))
    return result


def concept_overlap(required: list[str], candidate_skills: list[str]) -> tuple[float, list[tuple[str, str]]]:
    """
    Semantic skill match score in [0, 1] plus matched pairs.
    Returns (score, [(required_concept, matched_candidate_skill), ...])
    """
    if not required:
        return 0.5, []

    candidate_expanded = expand_skills(candidate_skills)
    matches: list[tuple[str, str]] = []
    hit_count = 0

    for req in required:
        req_key = normalize_skill(req)
        req_expanded = expand_skill(req)
        overlap = req_expanded & candidate_expanded
        if overlap:
            hit_count += 1
            best_match = next(iter(overlap))
            matches.append((req, best_match))

    score = hit_count / len(required)
    return min(score, 1.0), matches


def find_missing_skills(required: list[str], candidate_skills: list[str], limit: int = 5) -> list[str]:
    """Skills/concepts required by JD with no semantic match in candidate profile."""
    candidate_expanded = expand_skills(candidate_skills)
    missing: list[str] = []
    for req in required:
        if not (expand_skill(req) & candidate_expanded):
            missing.append(req)
        if len(missing) >= limit:
            break
    return missing
