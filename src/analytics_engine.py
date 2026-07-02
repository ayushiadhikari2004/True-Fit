"""Analytics engine — aggregates hiring insights from ranked candidate pool.

Generates stats for the analytics dashboard: salary distributions, profile
completeness histograms, verification breakdowns, acceptance rate distribution,
and top missing skills across the shortlist.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .enhanced_signals import (
    EnhancedSignalConfig,
    compute_enhanced_signals,
    EnhancedSignalResult,
)
from .rule_scorer import find_missing_must_haves


@dataclass
class HiringAnalytics:
    """Aggregated analytics across the ranked candidate pool."""

    total_candidates: int = 0
    shortlisted: int = 0

    # Salary distribution
    salary_candidates_count: int = 0
    salary_min: float = 0.0
    salary_max: float = 0.0
    salary_avg: float = 0.0
    salary_median: float = 0.0
    salary_in_budget_count: int = 0
    salary_over_budget_count: int = 0
    salary_under_budget_count: int = 0
    salary_distribution: list[dict[str, Any]] = field(default_factory=list)

    # Profile completeness
    completeness_avg: float = 0.0
    completeness_above_85: int = 0
    completeness_below_50: int = 0
    completeness_distribution: list[dict[str, Any]] = field(default_factory=list)

    # Verification stats
    verified_email_count: int = 0
    verified_phone_count: int = 0
    linkedin_connected_count: int = 0
    github_active_count: int = 0
    fully_verified_count: int = 0
    verification_distribution: list[dict[str, Any]] = field(default_factory=list)

    # Acceptance rates
    acceptance_avg: float = 0.0
    high_acceptance_count: int = 0  # >= 0.7
    low_acceptance_count: int = 0   # < 0.4
    acceptance_distribution: list[dict[str, Any]] = field(default_factory=list)

    # Top missing skills
    top_missing_skills: list[tuple[str, int]] = field(default_factory=list)

    # Score distribution
    score_avg: float = 0.0
    score_distribution: list[dict[str, Any]] = field(default_factory=list)

    # Badge stats
    badge_counts: dict[str, int] = field(default_factory=dict)


def _salary_bucket(lpa: float) -> str:
    if lpa < 10:
        return "< 10L"
    elif lpa < 20:
        return "10-20L"
    elif lpa < 30:
        return "20-30L"
    elif lpa < 40:
        return "30-40L"
    elif lpa < 50:
        return "40-50L"
    else:
        return "50L+"


def compute_analytics(
    candidates: list[dict],
    enhanced_results: list[EnhancedSignalResult],
    scores: list[float],
    badges_per_candidate: list[list[Any]],
    config: EnhancedSignalConfig | None = None,
) -> HiringAnalytics:
    """Compute aggregate analytics from the ranked shortlist."""
    if config is None:
        config = EnhancedSignalConfig()

    analytics = HiringAnalytics(
        total_candidates=len(candidates),
        shortlisted=len(candidates),
    )

    # ── Salary distribution ───────────────────────────────────────────────
    salary_values: list[float] = []
    salary_buckets: Counter = Counter()

    for candidate in candidates:
        signals = candidate.get("redrob_signals", {})
        sal_range = signals.get("expected_salary_range_inr_lpa")
        if sal_range and isinstance(sal_range, dict):
            mid = (float(sal_range.get("min", 0)) + float(sal_range.get("max", 0))) / 2
            salary_values.append(mid)
            salary_buckets[_salary_bucket(mid)] += 1

            cand_min = float(sal_range.get("min", 0))
            cand_max = float(sal_range.get("max", 0))
            if cand_min >= config.salary_budget_min_lpa and cand_max <= config.salary_budget_max_lpa:
                analytics.salary_in_budget_count += 1
            elif cand_min > config.salary_budget_max_lpa:
                analytics.salary_over_budget_count += 1
            elif cand_max < config.salary_budget_min_lpa:
                analytics.salary_under_budget_count += 1
            else:
                analytics.salary_in_budget_count += 1  # partial overlap counts as in-budget

    if salary_values:
        analytics.salary_candidates_count = len(salary_values)
        analytics.salary_min = min(salary_values)
        analytics.salary_max = max(salary_values)
        analytics.salary_avg = sum(salary_values) / len(salary_values)
        sorted_sal = sorted(salary_values)
        n = len(sorted_sal)
        analytics.salary_median = sorted_sal[n // 2] if n % 2 else (sorted_sal[n // 2 - 1] + sorted_sal[n // 2]) / 2

    bucket_order = ["< 10L", "10-20L", "20-30L", "30-40L", "40-50L", "50L+"]
    analytics.salary_distribution = [
        {"bucket": b, "count": salary_buckets.get(b, 0)} for b in bucket_order
    ]

    # ── Profile completeness ──────────────────────────────────────────────
    completeness_values: list[float] = []
    completeness_buckets: Counter = Counter()

    for er in enhanced_results:
        c = er.profile_completeness
        completeness_values.append(c)
        if c >= 0.85:
            analytics.completeness_above_85 += 1
        if c < 0.50:
            analytics.completeness_below_50 += 1

        if c >= 0.9:
            completeness_buckets["90-100%"] += 1
        elif c >= 0.8:
            completeness_buckets["80-90%"] += 1
        elif c >= 0.7:
            completeness_buckets["70-80%"] += 1
        elif c >= 0.6:
            completeness_buckets["60-70%"] += 1
        else:
            completeness_buckets["< 60%"] += 1

    if completeness_values:
        analytics.completeness_avg = sum(completeness_values) / len(completeness_values)

    comp_order = ["< 60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    analytics.completeness_distribution = [
        {"bucket": b, "count": completeness_buckets.get(b, 0)} for b in comp_order
    ]

    # ── Verification stats ────────────────────────────────────────────────
    for er in enhanced_results:
        v = er.verification_explanation
        if v.get("verified_email"):
            analytics.verified_email_count += 1
        if v.get("verified_phone"):
            analytics.verified_phone_count += 1
        if v.get("linkedin_connected"):
            analytics.linkedin_connected_count += 1
        if v.get("github_active"):
            analytics.github_active_count += 1
        if v.get("verified_count", 0) == v.get("total_checks", 4):
            analytics.fully_verified_count += 1

    analytics.verification_distribution = [
        {"channel": "Email", "count": analytics.verified_email_count},
        {"channel": "Phone", "count": analytics.verified_phone_count},
        {"channel": "LinkedIn", "count": analytics.linkedin_connected_count},
        {"channel": "GitHub", "count": analytics.github_active_count},
    ]

    # ── Acceptance rates ──────────────────────────────────────────────────
    acceptance_values: list[float] = []
    acceptance_buckets: Counter = Counter()

    for er in enhanced_results:
        a = er.acceptance_explanation
        rate = a.get("acceptance_rate")
        if rate is not None:
            acceptance_values.append(rate)
            if rate >= 0.7:
                analytics.high_acceptance_count += 1
            if rate < 0.4:
                analytics.low_acceptance_count += 1

            if rate >= 0.8:
                acceptance_buckets["80-100%"] += 1
            elif rate >= 0.6:
                acceptance_buckets["60-80%"] += 1
            elif rate >= 0.4:
                acceptance_buckets["40-60%"] += 1
            else:
                acceptance_buckets["< 40%"] += 1

    if acceptance_values:
        analytics.acceptance_avg = sum(acceptance_values) / len(acceptance_values)

    acc_order = ["< 40%", "40-60%", "60-80%", "80-100%"]
    analytics.acceptance_distribution = [
        {"bucket": b, "count": acceptance_buckets.get(b, 0)} for b in acc_order
    ]

    # ── Top missing skills ────────────────────────────────────────────────
    skill_counter: Counter = Counter()
    for candidate in candidates:
        missing = find_missing_must_haves(candidate)
        for skill in missing:
            skill_counter[skill] += 1

    analytics.top_missing_skills = skill_counter.most_common(8)

    # ── Score distribution ────────────────────────────────────────────────
    if scores:
        analytics.score_avg = sum(scores) / len(scores)
        score_buckets: Counter = Counter()
        for s in scores:
            pct = s * 100
            if pct >= 90:
                score_buckets["90-100"] += 1
            elif pct >= 80:
                score_buckets["80-90"] += 1
            elif pct >= 70:
                score_buckets["70-80"] += 1
            elif pct >= 60:
                score_buckets["60-70"] += 1
            else:
                score_buckets["< 60"] += 1

        score_order = ["< 60", "60-70", "70-80", "80-90", "90-100"]
        analytics.score_distribution = [
            {"bucket": b, "count": score_buckets.get(b, 0)} for b in score_order
        ]

    # ── Badge stats ───────────────────────────────────────────────────────
    badge_counter: Counter = Counter()
    for badge_list in badges_per_candidate:
        for badge in badge_list:
            badge_counter[badge.key] += 1
    analytics.badge_counts = dict(badge_counter)

    return analytics
