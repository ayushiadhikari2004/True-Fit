"""Honeypot and impossible-profile detection."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def detect_honeypot_issues(candidate: dict) -> list[str]:
    """Return human-readable issues for subtly impossible profiles."""
    issues: list[str] = []

    profile = candidate.get("profile", {})
    years = float(profile.get("years_of_experience") or 0)
    history = candidate.get("career_history", [])

    total_role_months = sum(int(role.get("duration_months") or 0) for role in history)
    if years > 0 and total_role_months > 0:
        implied_years = total_role_months / 12.0
        if implied_years + 1.5 < years * 0.5 or implied_years > years * 2.5 + 2:
            issues.append("experience timeline inconsistent with role durations")

    for role in history:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date"))
        months = int(role.get("duration_months") or 0)
        if start and end and months:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if actual_months >= 0 and months > actual_months + 3:
                issues.append(f"role duration exceeds date range at {role.get('company', '')}")

    expert_zero = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "")
        months = int(skill.get("duration_months") or 0)
        endorsements = int(skill.get("endorsements") or 0)
        if prof == "expert" and months < 6:
            expert_zero += 1
        if prof == "expert" and months == 0:
            issues.append(f"expert skill with zero usage: {skill.get('name', '')}")
        if prof in {"advanced", "expert"} and months < 3 and endorsements > 20:
            issues.append(f"high endorsements with minimal usage: {skill.get('name', '')}")

    if expert_zero >= 4:
        issues.append("multiple expert skills with very low usage duration")

    signals = candidate.get("redrob_signals", {})
    if signals:
        notice = int(signals.get("notice_period_days") or 0)
        if notice > 180:
            issues.append("notice period exceeds allowed maximum")

    return issues


def honeypot_penalty(candidate: dict) -> float:
    """Return multiplier in [0, 1]; lower means more likely honeypot."""
    issues = detect_honeypot_issues(candidate)
    if not issues:
        return 1.0
    if len(issues) >= 3:
        return 0.01
    if len(issues) == 2:
        return 0.08
    return 0.25


def honeypot_features(candidate: dict) -> dict[str, Any]:
    issues = detect_honeypot_issues(candidate)
    return {
        "honeypot_penalty": honeypot_penalty(candidate),
        "honeypot_issues": issues,
        "is_likely_honeypot": len(issues) >= 2,
    }
