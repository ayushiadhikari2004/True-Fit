"""Rule-based recruiter chat assistant over ranked results."""

from __future__ import annotations

import re
from typing import Any

from .explainer import ExplanationReport


def answer_recruiter_query(
    query: str,
    ranked_reports: list[ExplanationReport],
    role_summary: str,
) -> str:
    """Simple chat assistant — no external LLM, works offline."""
    q = query.strip().lower()
    if not ranked_reports:
        return "No ranked candidates yet. Run the ranker first."

    if any(w in q for w in ("top", "best", "shortlist", "#1", "rank 1")):
        r = ranked_reports[0]
        return (
            f"Top pick: **{r.candidate_name}** ({r.candidate_id}) — Final Score {r.factor_scores.get('Final Score', 0)}.\n\n"
            f"{r.summary}\n\n"
            f"Key strengths: {'; '.join(r.strengths[:3])}"
        )

    # Match candidate ID
    id_match = re.search(r"cand_\d{7}", q)
    if id_match:
        cid = id_match.group(0).upper()
        for r in ranked_reports:
            if r.candidate_id == cid:
                return r.to_markdown()
        return f"Candidate {cid} is not in the current shortlist."

    if "why" in q and ("rank" in q or "shortlist" in q or "selected" in q):
        r = ranked_reports[0]
        return r.to_markdown()

    if "missing" in q or "gap" in q or "concern" in q:
        r = ranked_reports[0]
        if r.gaps:
            return "Top candidate gaps:\n" + "\n".join(f"- • {g}" for g in r.gaps)
        return "No major gaps flagged for the top candidate."

    if "score" in q or "factor" in q or "breakdown" in q:
        r = ranked_reports[0]
        lines = [f"**{k}**: {v}" for k, v in r.factor_scores.items()]
        return "Factor breakdown for #1:\n" + "\n".join(f"- {line}" for line in lines)

    if "role" in q or "job" in q or "jd" in q:
        return f"Role understanding:\n{role_summary}"

    if "compare" in q:
        if len(ranked_reports) >= 2:
            a, b = ranked_reports[0], ranked_reports[1]
            return (
                f"**{a.candidate_name}** (Score {a.factor_scores.get('Final Score', 0)}) vs "
                f"**{b.candidate_name}** (Score {b.factor_scores.get('Final Score', 0)}).\n\n"
                f"#1 strengths: {'; '.join(a.strengths[:2])}\n"
                f"#2 strengths: {'; '.join(b.strengths[:2])}"
            )

    # Default: summarize shortlist
    top3 = ranked_reports[:3]
    lines = ["Here's the current shortlist:"]
    for r in top3:
        lines.append(
            f"- #{r.rank} {r.candidate_name} — Score {r.factor_scores.get('Final Score', 0)}: {r.summary[:100]}"
        )
    lines.append("\nAsk me: 'Why is CAND_XXXXXXX ranked?' or 'Show factor breakdown'")
    return "\n".join(lines)
