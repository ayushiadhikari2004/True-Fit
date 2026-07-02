"""Recruiter-friendly badge system — visual indicators for candidate quality.

Each badge is earned by meeting specific data-driven thresholds.  Badges
reference the actual underlying data so recruiters can trust them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enhanced_signals import EnhancedSignalConfig, EnhancedSignalResult
from .rule_scorer import find_missing_must_haves


# ── Badge definitions ─────────────────────────────────────────────────────────

@dataclass
class Badge:
    """A single recruiter badge with display metadata."""

    key: str           # machine-readable key
    label: str         # display label
    icon: str          # emoji icon
    color: str         # CSS colour for background
    text_color: str    # CSS colour for text
    border_color: str  # CSS colour for border
    tooltip: str       # hover explanation with actual data
    category: str      # grouping: fit / trust / risk / quality


# Standard badge templates
BADGE_TEMPLATES = {
    "salary_fit": {
        "label": "Salary Fit",
        "icon": "💰",
        "color": "#0f2e1f",
        "text_color": "#4ade80",
        "border_color": "#166534",
        "category": "fit",
    },
    "verified": {
        "label": "Verified",
        "icon": "✅",
        "color": "#1a2744",
        "text_color": "#60a5fa",
        "border_color": "#1d4ed8",
        "category": "trust",
    },
    "high_join_probability": {
        "label": "High Join Probability",
        "icon": "🤝",
        "color": "#1f2937",
        "text_color": "#a78bfa",
        "border_color": "#6d28d9",
        "category": "trust",
    },
    "top_match": {
        "label": "Top Match",
        "icon": "⭐",
        "color": "#422006",
        "text_color": "#fbbf24",
        "border_color": "#b45309",
        "category": "fit",
    },
    "missing_skills": {
        "label": "Missing Skills",
        "icon": "⚠️",
        "color": "#2d1515",
        "text_color": "#f87171",
        "border_color": "#7f1d1d",
        "category": "risk",
    },
    "complete_profile": {
        "label": "Complete Profile",
        "icon": "📋",
        "color": "#1a3a3a",
        "text_color": "#5eead4",
        "border_color": "#115e59",
        "category": "quality",
    },
}


def compute_badges(
    candidate: dict,
    enhanced: EnhancedSignalResult,
    norm_score: float,
    flags: list[str],
    config: EnhancedSignalConfig | None = None,
) -> list[Badge]:
    """Compute which badges a candidate earns based on their actual data."""
    if config is None:
        config = EnhancedSignalConfig()

    badges: list[Badge] = []

    # ── Salary Fit ────────────────────────────────────────────────────────
    if enhanced.salary_fit >= config.salary_fit_badge_threshold:
        sal = enhanced.salary_explanation
        badges.append(Badge(
            key="salary_fit",
            tooltip=sal.get("reason", "Salary within budget"),
            **BADGE_TEMPLATES["salary_fit"],
        ))

    # ── Verified Identity ─────────────────────────────────────────────────
    if enhanced.identity_verification >= config.verification_badge_threshold:
        ver = enhanced.verification_explanation
        count = ver.get("verified_count", 0)
        total = ver.get("total_checks", 4)
        badges.append(Badge(
            key="verified",
            tooltip=f"{count}/{total} identity checks passed — {ver.get('reason', '')}",
            **BADGE_TEMPLATES["verified"],
        ))

    # ── High Join Probability ─────────────────────────────────────────────
    if enhanced.offer_acceptance >= config.join_probability_badge_threshold:
        acc = enhanced.acceptance_explanation
        rate = acc.get("acceptance_rate")
        tip = f"{rate:.0%} historical acceptance" if rate else "Strong join signals"
        badges.append(Badge(
            key="high_join_probability",
            tooltip=tip,
            **BADGE_TEMPLATES["high_join_probability"],
        ))

    # ── Top Match ─────────────────────────────────────────────────────────
    if norm_score >= config.top_match_badge_threshold:
        badges.append(Badge(
            key="top_match",
            tooltip=f"Overall score {norm_score:.0%} — exceeds {config.top_match_badge_threshold:.0%} threshold",
            **BADGE_TEMPLATES["top_match"],
        ))

    # ── Complete Profile ──────────────────────────────────────────────────
    if enhanced.profile_completeness >= config.completeness_badge_threshold:
        cmp = enhanced.completeness_explanation
        badges.append(Badge(
            key="complete_profile",
            tooltip=f"Profile {cmp.get('completeness_score', 0):.0%} complete — {cmp.get('reason', '')}",
            **BADGE_TEMPLATES["complete_profile"],
        ))

    # ── Missing Skills (risk badge) ───────────────────────────────────────
    missing = find_missing_must_haves(candidate)
    if len(missing) >= 2:
        badges.append(Badge(
            key="missing_skills",
            tooltip=f"Missing: {', '.join(missing[:4])}",
            **BADGE_TEMPLATES["missing_skills"],
        ))

    return badges


def render_badge_html(badge: Badge) -> str:
    """Render a single badge as an HTML span with tooltip."""
    return (
        f'<span class="recruiter-badge" '
        f'style="background:{badge.color};color:{badge.text_color};'
        f'border:1px solid {badge.border_color};padding:4px 10px;'
        f'border-radius:999px;font-size:12px;white-space:nowrap;'
        f'cursor:help;display:inline-flex;align-items:center;gap:4px" '
        f'title="{badge.tooltip}">'
        f'{badge.icon} {badge.label}'
        f'</span>'
    )


def render_badges_html(badges: list[Badge]) -> str:
    """Render a row of badges."""
    if not badges:
        return ""
    badge_html = " ".join(render_badge_html(b) for b in badges)
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0">{badge_html}</div>'
