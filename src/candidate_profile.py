"""Structured candidate profile parsing and recruiter-style summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuredCandidateProfile:
    """Knowledge-graph style structured representation of a candidate."""

    candidate_id: str
    name: str
    current_title: str
    experience_years: float
    industry: str
    location: str
    projects: list[str] = field(default_factory=list)
    tech_stack: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    leadership_signals: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    open_source_score: float = 0.0
    communication_signals: list[str] = field(default_factory=list)
    career_growth_score: float = 0.0
    summary: str = ""
    knowledge_graph: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "current_title": self.current_title,
            "experience_years": self.experience_years,
            "industry": self.industry,
            "location": self.location,
            "projects": self.projects,
            "tech_stack": self.tech_stack,
            "achievements": self.achievements,
            "education": self.education,
            "leadership_signals": self.leadership_signals,
            "certifications": self.certifications,
            "open_source_score": round(self.open_source_score, 2),
            "communication_signals": self.communication_signals,
            "career_growth_score": round(self.career_growth_score, 2),
            "summary": self.summary,
        }


def _extract_achievements(text: str) -> list[str]:
    patterns = [
        r"(?:built|shipped|deployed|designed|led|owned|scaled|improved|reduced|increased)[^.]{10,120}\.",
        r"(?:serving|processing|handled)\s+[\d,]+[KMB]?\s*(?:users|requests|events|records)[^.]*\.?",
    ]
    achievements: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            snippet = match.group(0).strip()
            if len(snippet) > 15 and snippet not in achievements:
                achievements.append(snippet[:140])
    return achievements[:5]


def _extract_projects(candidate: dict) -> list[str]:
    projects: list[str] = []
    for role in candidate.get("career_history", []):
        desc = role.get("description", "")
        title = role.get("title", "")
        company = role.get("company", "")
        if desc:
            first_sentence = desc.split(".")[0].strip()
            if len(first_sentence) > 20:
                projects.append(f"{title} at {company}: {first_sentence[:120]}")
    return projects[:5]


def _leadership_signals(candidate: dict, text: str) -> list[str]:
    signals: list[str] = []
    lowered = text.lower()
    for phrase in ("led team", "mentored", "managed", "tech lead", "hiring", "cross-functional", "architect"):
        if phrase in lowered:
            signals.append(phrase.title())
    title = candidate["profile"].get("current_title", "").lower()
    if any(t in title for t in ("lead", "staff", "principal", "manager", "head")):
        signals.append(f"Title: {candidate['profile']['current_title']}")
    return list(dict.fromkeys(signals))[:5]


def _communication_signals(text: str, candidate: dict) -> list[str]:
    signals: list[str] = []
    lowered = text.lower()
    for phrase in ("documented", "blog", "presented", "wrote", "published", "technical writing"):
        if phrase in lowered:
            signals.append(phrase)
    summary_len = len(candidate["profile"].get("summary", ""))
    if summary_len > 200:
        signals.append("detailed professional summary")
    return signals[:4]


def _career_growth_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return 0.5
    seniority_markers = ("senior", "lead", "staff", "principal", "manager", "head")
    scores: list[float] = []
    for idx, role in enumerate(history):
        title = role.get("title", "").lower()
        weight = (idx + 1) / len(history)
        if any(m in title for m in seniority_markers):
            scores.append(weight)
    return min(sum(scores) / max(len(history) * 0.5, 1), 1.0) if scores else 0.4


def _build_summary(profile: StructuredCandidateProfile) -> str:
    """Recruiter-style one-liner used as the embedding document."""
    parts = [f"{profile.current_title} with {profile.experience_years:.1f} years in {profile.industry}"]

    if profile.tech_stack:
        top_skills = ", ".join(profile.tech_stack[:5])
        parts.append(f"Strong in {top_skills}")

    if profile.achievements:
        parts.append(profile.achievements[0].rstrip("."))

    if profile.leadership_signals:
        parts.append(f"Leadership: {profile.leadership_signals[0]}")

    if profile.open_source_score > 0.5:
        parts.append("active open-source contributor")

    # Ownership / production signals from projects
    ownership = [p for p in profile.projects if any(w in p.lower() for w in ("built", "owned", "shipped", "designed"))]
    if ownership:
        snippet = ownership[0].split(":")[-1].strip()[:80]
        parts.append(f"Ownership: {snippet}")

    return ". ".join(parts) + "."


def parse_candidate(candidate: dict) -> StructuredCandidateProfile:
    """Convert raw candidate JSON into a structured profile + summary."""
    profile_data = candidate["profile"]
    signals = candidate.get("redrob_signals", {})

    tech_stack = [
        s.get("name", "")
        for s in candidate.get("skills", [])
        if s.get("name") and s.get("proficiency") in {"advanced", "expert", "intermediate"}
    ][:15]

    education = [
        f"{e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution', '')}"
        for e in candidate.get("education", [])
    ]

    certifications = [
        f"{c.get('name', '')} ({c.get('issuer', '')})"
        for c in candidate.get("certifications", [])
    ]

    full_text = " ".join(
        [
            profile_data.get("summary", ""),
            " ".join(r.get("description", "") for r in candidate.get("career_history", [])),
        ]
    )

    github = signals.get("github_activity_score", -1)
    open_source_score = 0.0 if github == -1 else min(float(github) / 100.0, 1.0)

    structured = StructuredCandidateProfile(
        candidate_id=candidate["candidate_id"],
        name=profile_data.get("anonymized_name", "Candidate"),
        current_title=profile_data.get("current_title", ""),
        experience_years=float(profile_data.get("years_of_experience") or 0),
        industry=profile_data.get("current_industry", "Technology"),
        location=profile_data.get("location", ""),
        projects=_extract_projects(candidate),
        tech_stack=tech_stack,
        achievements=_extract_achievements(full_text),
        education=education,
        leadership_signals=_leadership_signals(candidate, full_text),
        certifications=certifications,
        open_source_score=open_source_score,
        communication_signals=_communication_signals(full_text, candidate),
        career_growth_score=_career_growth_score(candidate),
    )

    structured.summary = _build_summary(structured)

    # Knowledge graph: nodes (skills, companies, projects) + edges (used_at, worked_on)
    nodes = [{"id": s, "type": "skill"} for s in tech_stack[:10]]
    nodes += [
        {"id": r.get("company", ""), "type": "company"}
        for r in candidate.get("career_history", [])[:5]
    ]
    edges = [
        {"from": structured.current_title, "to": r.get("company", ""), "relation": "worked_at"}
        for r in candidate.get("career_history", [])[:5]
    ]
    structured.knowledge_graph = {"nodes": nodes, "edges": edges}

    return structured
