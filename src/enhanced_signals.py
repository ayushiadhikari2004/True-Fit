"""Enhanced ranking signals — salary fit, offer acceptance, identity verification, profile completeness.

Each signal function returns a float in [0, 1] and an explanation dict referencing
the actual candidate data used to compute the score.  The explanation dict is
consumed by the badge system and the per-candidate detail panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class EnhancedSignalConfig:
    """Recruiter-configurable parameters for all enhanced signals."""

    # Salary budget (INR LPA) — set by recruiter in sidebar
    salary_budget_min_lpa: float = 15.0
    salary_budget_max_lpa: float = 45.0

    # Weights for the final ranking formula (new signals)
    weight_salary_fit: float = 0.08
    weight_offer_acceptance: float = 0.06
    weight_identity_verification: float = 0.04
    weight_profile_completeness: float = 0.04

    # Thresholds for badge eligibility
    salary_fit_badge_threshold: float = 0.7
    verification_badge_threshold: float = 0.75
    join_probability_badge_threshold: float = 0.65
    top_match_badge_threshold: float = 0.80
    completeness_badge_threshold: float = 0.85
    completeness_penalty_threshold: float = 0.50


# ── Salary Budget Compatibility ───────────────────────────────────────────────

def salary_fit_score(
    candidate: dict,
    config: EnhancedSignalConfig,
) -> tuple[float, dict[str, Any]]:
    """Score how well the candidate's expected salary fits the recruiter's budget.

    Returns (score, explanation) where score ∈ [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    salary_range = signals.get("expected_salary_range_inr_lpa")

    if not salary_range or not isinstance(salary_range, dict):
        return 0.5, {
            "salary_fit": 0.5,
            "reason": "Salary expectation not provided",
            "candidate_min": None,
            "candidate_max": None,
            "budget_min": config.salary_budget_min_lpa,
            "budget_max": config.salary_budget_max_lpa,
        }

    cand_min = float(salary_range.get("min", 0))
    cand_max = float(salary_range.get("max", 0))
    budget_min = config.salary_budget_min_lpa
    budget_max = config.salary_budget_max_lpa

    explanation: dict[str, Any] = {
        "candidate_min": cand_min,
        "candidate_max": cand_max,
        "budget_min": budget_min,
        "budget_max": budget_max,
    }

    # Perfect overlap: candidate range fully within budget
    if cand_min >= budget_min and cand_max <= budget_max:
        score = 1.0
        explanation["reason"] = f"₹ {cand_min:.1f}–{cand_max:.1f}L fully within budget ₹ {budget_min:.0f}–{budget_max:.0f}L"
    # Partial overlap
    elif cand_min <= budget_max and cand_max >= budget_min:
        overlap_start = max(cand_min, budget_min)
        overlap_end = min(cand_max, budget_max)
        overlap = max(0, overlap_end - overlap_start)
        candidate_span = max(cand_max - cand_min, 1.0)
        score = 0.4 + 0.6 * (overlap / candidate_span)
        explanation["reason"] = f"₹ {cand_min:.1f}–{cand_max:.1f}L partially overlaps budget ₹ {budget_min:.0f}–{budget_max:.0f}L"
    # No overlap — candidate wants more
    elif cand_min > budget_max:
        gap = cand_min - budget_max
        score = max(0.05, 0.4 - gap / budget_max * 0.5)
        explanation["reason"] = f"₹ {cand_min:.1f}–{cand_max:.1f}L exceeds budget cap ₹ {budget_max:.0f}L by ₹ {gap:.1f}L"
    # No overlap — candidate is below budget
    else:
        score = 0.7  # Under-budget is OK, mildly positive
        explanation["reason"] = f"₹ {cand_min:.1f}–{cand_max:.1f}L is below budget floor ₹ {budget_min:.0f}L"

    explanation["salary_fit"] = round(score, 3)
    return score, explanation


# ── Offer Acceptance Rate (Join Probability) ──────────────────────────────────

def offer_acceptance_score(candidate: dict) -> tuple[float, dict[str, Any]]:
    """Score based on historical offer acceptance rate — predicts join probability.

    Returns (score, explanation) where score ∈ [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    rate = signals.get("offer_acceptance_rate")

    if rate is None:
        return 0.5, {
            "acceptance_rate": None,
            "score": 0.5,
            "reason": "No offer acceptance history available",
        }

    rate = max(0.0, min(1.0, float(rate)))
    # Non-linear: strong signal when high, weak penalty when low
    if rate >= 0.85:
        score = 1.0
    elif rate >= 0.7:
        score = 0.8 + (rate - 0.7) / 0.15 * 0.2
    elif rate >= 0.5:
        score = 0.5 + (rate - 0.5) / 0.2 * 0.3
    elif rate >= 0.3:
        score = 0.3 + (rate - 0.3) / 0.2 * 0.2
    else:
        score = max(0.1, rate)

    reason = f"{rate:.0%} offer acceptance rate"
    if rate >= 0.75:
        reason += " — high join probability"
    elif rate < 0.4:
        reason += " — risk of offer decline"

    return score, {
        "acceptance_rate": rate,
        "score": round(score, 3),
        "reason": reason,
    }


# ── Verified Identity Scoring ─────────────────────────────────────────────────

def identity_verification_score(candidate: dict) -> tuple[float, dict[str, Any]]:
    """Score based on verified email, phone, LinkedIn, and GitHub activity.

    Returns (score, explanation) where score ∈ [0, 1].
    """
    signals = candidate.get("redrob_signals", {})

    checks = {
        "email": bool(signals.get("verified_email")),
        "phone": bool(signals.get("verified_phone")),
        "linkedin": bool(signals.get("linkedin_connected")),
    }

    github_score_raw = signals.get("github_activity_score", -1)
    has_github = github_score_raw not in (-1, None) and float(github_score_raw) > 0
    checks["github"] = has_github

    weights = {"email": 0.30, "phone": 0.25, "linkedin": 0.30, "github": 0.15}
    score = sum(weights[k] for k, v in checks.items() if v)

    verified_list = [k.title() for k, v in checks.items() if v]
    unverified_list = [k.title() for k, v in checks.items() if not v]

    reason_parts = []
    if verified_list:
        reason_parts.append(f"Verified: {', '.join(verified_list)}")
    if unverified_list:
        reason_parts.append(f"Missing: {', '.join(unverified_list)}")

    return score, {
        "verified_email": checks["email"],
        "verified_phone": checks["phone"],
        "linkedin_connected": checks["linkedin"],
        "github_active": checks["github"],
        "github_score": float(github_score_raw) if github_score_raw not in (-1, None) else None,
        "verification_score": round(score, 3),
        "verified_count": sum(1 for v in checks.values() if v),
        "total_checks": len(checks),
        "reason": "; ".join(reason_parts),
    }


# ── Profile Completeness Scoring ──────────────────────────────────────────────

def profile_completeness_score(candidate: dict) -> tuple[float, dict[str, Any]]:
    """Score profile completeness with penalties for critical missing fields.

    Returns (score, explanation) where score ∈ [0, 1].
    """
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Platform-provided completeness as baseline
    platform_score = float(signals.get("profile_completeness_score", 0)) / 100.0

    # Check critical fields
    missing_fields: list[str] = []
    field_checks = {
        "summary": bool(profile.get("summary", "").strip()),
        "headline": bool(profile.get("headline", "").strip()),
        "current_title": bool(profile.get("current_title", "").strip()),
        "location": bool(profile.get("location", "").strip()),
        "experience": float(profile.get("years_of_experience", 0)) > 0,
        "skills": len(candidate.get("skills", [])) >= 3,
        "career_history": len(candidate.get("career_history", [])) >= 1,
        "education": len(candidate.get("education", [])) >= 1,
    }

    for field_name, present in field_checks.items():
        if not present:
            missing_fields.append(field_name)

    # Count filled optional fields
    optional_present = 0
    optional_total = 4
    if candidate.get("certifications"):
        optional_present += 1
    if candidate.get("languages"):
        optional_present += 1
    if signals.get("expected_salary_range_inr_lpa"):
        optional_present += 1
    if signals.get("github_activity_score", -1) != -1:
        optional_present += 1

    # Composite score
    critical_filled = sum(1 for v in field_checks.values() if v)
    critical_ratio = critical_filled / len(field_checks)
    optional_ratio = optional_present / optional_total

    score = platform_score * 0.4 + critical_ratio * 0.45 + optional_ratio * 0.15

    # Penalty for critical gaps
    if len(missing_fields) >= 3:
        score *= 0.6
    elif len(missing_fields) >= 2:
        score *= 0.8

    score = max(0.0, min(1.0, score))

    reason_parts = []
    if missing_fields:
        reason_parts.append(f"Missing: {', '.join(missing_fields)}")
    else:
        reason_parts.append("All critical fields present")
    reason_parts.append(f"Platform score: {platform_score:.0%}")

    return score, {
        "platform_score": round(platform_score, 3),
        "critical_filled": critical_filled,
        "critical_total": len(field_checks),
        "missing_fields": missing_fields,
        "optional_filled": optional_present,
        "optional_total": optional_total,
        "completeness_score": round(score, 3),
        "reason": "; ".join(reason_parts),
    }


# ── Composite Enhanced Score ──────────────────────────────────────────────────

@dataclass
class EnhancedSignalResult:
    """All enhanced signal scores and explanations for a single candidate."""

    salary_fit: float = 0.5
    salary_explanation: dict[str, Any] = field(default_factory=dict)

    offer_acceptance: float = 0.5
    acceptance_explanation: dict[str, Any] = field(default_factory=dict)

    identity_verification: float = 0.0
    verification_explanation: dict[str, Any] = field(default_factory=dict)

    profile_completeness: float = 0.5
    completeness_explanation: dict[str, Any] = field(default_factory=dict)

    def composite_score(self, config: EnhancedSignalConfig) -> float:
        """Weighted composite of all enhanced signals (additive to existing score)."""
        return (
            self.salary_fit * config.weight_salary_fit
            + self.offer_acceptance * config.weight_offer_acceptance
            + self.identity_verification * config.weight_identity_verification
            + self.profile_completeness * config.weight_profile_completeness
        )


def compute_enhanced_signals(
    candidate: dict,
    config: EnhancedSignalConfig | None = None,
) -> EnhancedSignalResult:
    """Compute all enhanced signals for a candidate."""
    if config is None:
        config = EnhancedSignalConfig()

    sal_score, sal_exp = salary_fit_score(candidate, config)
    acc_score, acc_exp = offer_acceptance_score(candidate)
    ver_score, ver_exp = identity_verification_score(candidate)
    cmp_score, cmp_exp = profile_completeness_score(candidate)

    return EnhancedSignalResult(
        salary_fit=sal_score,
        salary_explanation=sal_exp,
        offer_acceptance=acc_score,
        acceptance_explanation=acc_exp,
        identity_verification=ver_score,
        verification_explanation=ver_exp,
        profile_completeness=cmp_score,
        completeness_explanation=cmp_exp,
    )
