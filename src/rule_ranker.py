"""Rule-based ranker integrated with explainability reports."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .candidate_loader import iter_candidates, load_candidates
from .candidate_card import CandidateCard, build_candidate_card
from .candidate_profile import parse_candidate
from .card_renderer import render_shortlist_html
from .explainer import ExplanationReport
from .job_understanding import StructuredRoleProfile, understand_job
from .jd_parser import JobRequirements, load_job_requirements
from .rule_scorer import (
    MUST_HAVE_SKILLS,
    build_reasoning,
    find_missing_must_haves,
    normalize_scores,
    score_candidate,
    score_candidate_enhanced,
    EnhancedRuleScoreResult,
)
from .enhanced_signals import EnhancedSignalConfig, EnhancedSignalResult, compute_enhanced_signals
from .badge_system import Badge, compute_badges, render_badges_html
from .analytics_engine import HiringAnalytics, compute_analytics


@dataclass
class RuleRankerConfig:
    top_k: int = 100
    show_progress: bool = True
    exclude_honeypots: bool = True
    enhanced_config: EnhancedSignalConfig | None = None
    enable_enhanced_signals: bool = True


@dataclass
class RuleRankingResult:
    dataframe: pd.DataFrame
    role_profile: StructuredRoleProfile
    explanations: list[ExplanationReport] = field(default_factory=list)
    cards: list[CandidateCard] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    analytics: HiringAnalytics | None = None

    def save_reports(self, path: str | Path) -> None:
        payload = {
            "mode": "rule",
            "stats": self.stats,
            "role_profile": self.role_profile.to_dict(),
            "candidates": [card.to_dict() for card in self.cards],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_html(self, path: str | Path) -> None:
        Path(path).write_text(render_shortlist_html(self.cards), encoding="utf-8")


def _rule_explanation(
    rank: int,
    candidate: dict,
    flags: list[str],
    norm_score: float,
    role: StructuredRoleProfile,
    enhanced: EnhancedSignalResult | None = None,
) -> ExplanationReport:
    """Build dashboard-compatible explanation from rule scorer output."""
    profile = parse_candidate(candidate)
    strengths: list[str] = []
    gaps: list[str] = []

    if "all_must_haves" in flags:
        strengths.append("Covers all core must-haves (embeddings, vector DB, retrieval, eval, Python)")
    for flag in flags:
        if flag.endswith("yr_exp_sweet_spot"):
            strengths.append(f"Experience in JD sweet spot ({profile.experience_years:.1f} years)")
        elif flag.endswith("yr_exp_good"):
            strengths.append(f"Experience within JD range ({profile.experience_years:.1f} years)")
        elif flag == "preferred_location":
            strengths.append(f"Based in preferred location ({profile.location})")
        elif flag == "open_to_work":
            strengths.append("Marked open to work")

    if any(t in profile.current_title.lower() for t in ("ai", "ml", "search", "nlp", "recommendation")):
        strengths.append(f"Relevant title: {profile.current_title}")

    for achievement in profile.achievements[:2]:
        strengths.append(achievement.rstrip(".")[:100])

    # Enhanced signal-based strengths/gaps referencing actual data
    if enhanced is not None:
        sal = enhanced.salary_explanation
        if enhanced.salary_fit >= 0.7 and sal.get("reason"):
            strengths.append(f"Salary fit: {sal['reason']}")
        elif enhanced.salary_fit < 0.3 and sal.get("reason"):
            gaps.append(f"Salary: {sal['reason']}")

        acc = enhanced.acceptance_explanation
        rate = acc.get("acceptance_rate")
        if rate is not None and rate >= 0.7:
            strengths.append(f"High join probability ({rate:.0%} acceptance rate)")
        elif rate is not None and rate < 0.4:
            gaps.append(f"Low offer acceptance ({rate:.0%})")

        ver = enhanced.verification_explanation
        if enhanced.identity_verification >= 0.75:
            strengths.append(f"Identity verified: {ver.get('reason', '')}")

        cmp = enhanced.completeness_explanation
        if enhanced.profile_completeness < 0.5:
            missing = cmp.get('missing_fields', [])
            gaps.append(f"Incomplete profile — missing: {', '.join(missing[:3])}")

    if "irrelevant_title" in flags:
        gaps.append(f"Title ({profile.current_title}) may be weakly aligned")
    if "all_consulting" in flags:
        gaps.append("Career heavily weighted toward consulting/services")
    if "inactive_6mo" in flags:
        gaps.append("Inactive on platform for 6+ months")

    missing = find_missing_must_haves(candidate)
    for skill in missing[:3]:
        gaps.append(f"Missing: {skill}")

    notice = candidate["redrob_signals"].get("notice_period_days", 90)
    if notice > 60:
        gaps.append(f"{notice}-day notice period")

    factor_scores = {
        "Technical Fit": min(100, int(norm_score * 100)),
        "Domain Experience": 75 if "all_must_haves" in flags else 55,
        "Leadership": min(100, len(profile.leadership_signals) * 25),
        "Communication": min(100, int(candidate["redrob_signals"].get("recruiter_response_rate", 0) * 100)),
        "Learning Ability": min(100, int(profile.career_growth_score * 100)),
        "Culture Fit": 70 if profile.industry.lower() != "it services" else 45,
        "Project Relevance": 85 if any("ranking" in p.lower() or "retrieval" in p.lower() for p in profile.projects) else 60,
        "Behavioral Signals": min(100, int(float(candidate["redrob_signals"].get("recruiter_response_rate", 0)) * 100)),
        "Final Score": int(norm_score * 100),
    }

    # Add enhanced signal scores to the breakdown
    if enhanced is not None:
        factor_scores["Salary Fit"] = int(enhanced.salary_fit * 100)
        factor_scores["Join Probability"] = int(enhanced.offer_acceptance * 100)
        factor_scores["Identity Verification"] = int(enhanced.identity_verification * 100)
        factor_scores["Profile Completeness"] = int(enhanced.profile_completeness * 100)

    return ExplanationReport(
        rank=rank,
        candidate_id=candidate["candidate_id"],
        candidate_name=profile.name,
        summary=profile.summary,
        strengths=strengths[:8],
        gaps=gaps[:6],
        factor_scores=factor_scores,
        reasoning=build_reasoning(candidate, flags),
    )


class RuleBasedRanker:
    """Production rule-based ranker — fast, explicit JD signals, honeypot filtering."""

    def __init__(
        self,
        requirements: JobRequirements | None = None,
        role_profile: StructuredRoleProfile | None = None,
        config: RuleRankerConfig | None = None,
    ):
        self.requirements = requirements or load_job_requirements()
        self.role_profile = role_profile or understand_job(self.requirements.jd_text)
        self.config = config or RuleRankerConfig()
        self.enhanced_config = self.config.enhanced_config or EnhancedSignalConfig()
        self._last_result: RuleRankingResult | None = None

    def _score_all(
        self,
        candidates_path: str,
        limit: int | None = None,
    ) -> list[tuple[float, dict, list[str], EnhancedSignalResult | None]]:
        scored: list[tuple[float, dict, list[str], EnhancedSignalResult | None]] = []
        honeypots_filtered = 0

        iterator = iter_candidates(candidates_path)
        if limit:
            from itertools import islice
            iterator = islice(iterator, limit)

        if self.config.show_progress and not limit:
            iterator = tqdm(iterator, total=100_000, desc="Rule scoring", unit="cand")

        for candidate in iterator:
            if self.config.enable_enhanced_signals:
                result = score_candidate_enhanced(candidate, self.enhanced_config)
                if result.is_honeypot:
                    honeypots_filtered += 1
                    if self.config.exclude_honeypots:
                        continue
                scored.append((result.raw_score, candidate, result.flags, result.enhanced))
            else:
                result = score_candidate(candidate)
                if result.is_honeypot:
                    honeypots_filtered += 1
                    if self.config.exclude_honeypots:
                        continue
                scored.append((result.raw_score, candidate, result.flags, None))

        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        self._honeypots_filtered = honeypots_filtered
        return scored

    def rank_from_scored(
        self,
        scored: list[tuple[float, dict, list[str], EnhancedSignalResult | None]],
    ) -> RuleRankingResult:
        top_k = min(self.config.top_k, len(scored))
        top = scored[:top_k]
        # Normalize using the same signature (strip enhanced from tuples)
        scored_for_norm = [(s, c, f) for s, c, f, _ in scored]
        norm_scores = normalize_scores(scored_for_norm, top_n=top_k)

        explanations: list[ExplanationReport] = []
        cards: list[CandidateCard] = []
        all_enhanced: list[EnhancedSignalResult] = []
        all_badges: list[list[Badge]] = []
        all_candidates: list[dict] = []
        all_norm_scores: list[float] = []
        rows = []

        for rank, ((raw, candidate, flags, enhanced), norm_score) in enumerate(zip(top, norm_scores), start=1):
            explanation = _rule_explanation(
                rank, candidate, flags, norm_score, self.role_profile, enhanced
            )
            explanations.append(explanation)

            # Compute badges
            if enhanced is None:
                enhanced = compute_enhanced_signals(candidate, self.enhanced_config)
            badges = compute_badges(
                candidate, enhanced, norm_score, flags, self.enhanced_config
            )

            card = build_candidate_card(
                rank=rank,
                candidate=candidate,
                flags=flags,
                norm_score=norm_score,
                factor_scores=explanation.factor_scores,
                strengths=explanation.strengths,
                gaps=explanation.gaps,
                badges=badges,
                enhanced_result=enhanced,
            )
            cards.append(card)
            all_enhanced.append(enhanced)
            all_badges.append(badges)
            all_candidates.append(candidate)
            all_norm_scores.append(norm_score)

            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": rank,
                    "score": norm_score,
                    "reasoning": card.summary,
                }
            )

        df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

        # Compute analytics
        analytics = compute_analytics(
            all_candidates, all_enhanced, all_norm_scores, all_badges, self.enhanced_config
        )

        self._last_result = RuleRankingResult(
            dataframe=df,
            role_profile=self.role_profile,
            explanations=explanations,
            cards=cards,
            stats={
                "mode": "rule",
                "total_scored": len(scored),
                "honeypots_filtered": getattr(self, "_honeypots_filtered", 0),
                "raw_score_range": [top[0][0], top[-1][0]] if top else [0, 0],
            },
            analytics=analytics,
        )
        return self._last_result

    def rank_file(self, candidates_path: str, limit: int | None = None) -> pd.DataFrame:
        scored = self._score_all(candidates_path, limit=limit)
        return self.rank_from_scored(scored).dataframe

    def rank_file_with_reports(self, candidates_path: str, limit: int | None = None) -> RuleRankingResult:
        start = time.time()
        scored = self._score_all(candidates_path, limit=limit)
        result = self.rank_from_scored(scored)
        result.stats["elapsed_seconds"] = round(time.time() - start, 1)
        if self.config.show_progress:
            print(f"Rule ranking completed in {result.stats['elapsed_seconds']}s")
            print(f"Scored: {result.stats['total_scored']}, honeypots filtered: {result.stats['honeypots_filtered']}")
        return result

    def rank_candidates(self, candidates: list[dict]) -> pd.DataFrame:
        scored: list[tuple[float, dict, list[str], EnhancedSignalResult | None]] = []
        for candidate in candidates:
            if self.config.enable_enhanced_signals:
                result = score_candidate_enhanced(candidate, self.enhanced_config)
                if result.is_honeypot and self.config.exclude_honeypots:
                    continue
                scored.append((result.raw_score, candidate, result.flags, result.enhanced))
            else:
                result = score_candidate(candidate)
                if result.is_honeypot and self.config.exclude_honeypots:
                    continue
                scored.append((result.raw_score, candidate, result.flags, None))
        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        return self.rank_from_scored(scored).dataframe

    def rank_candidates_with_reports(self, candidates: list[dict]) -> RuleRankingResult:
        scored: list[tuple[float, dict, list[str], EnhancedSignalResult | None]] = []
        for candidate in candidates:
            if self.config.enable_enhanced_signals:
                result = score_candidate_enhanced(candidate, self.enhanced_config)
                if result.is_honeypot and self.config.exclude_honeypots:
                    continue
                scored.append((result.raw_score, candidate, result.flags, result.enhanced))
            else:
                result = score_candidate(candidate)
                if result.is_honeypot and self.config.exclude_honeypots:
                    continue
                scored.append((result.raw_score, candidate, result.flags, None))
        scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
        return self.rank_from_scored(scored)
