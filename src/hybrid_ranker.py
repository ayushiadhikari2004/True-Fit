"""End-to-end ranking pipeline with semantic understanding and explainability."""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .candidate_loader import iter_candidates, load_candidates
from .embeddings import HybridEmbedder
from .explainer import ExplanationReport, build_explanation
from .job_understanding import StructuredRoleProfile, understand_job
from .jd_parser import JobRequirements, load_job_requirements
from .hybrid_scorer import build_feature_bundle, score_candidate_bundle


@dataclass
class RankerConfig:
    top_k: int = 100
    prefilter_size: int = 2500
    rerank_pool: int = 200
    use_transformer: bool = True
    tfidf_only: bool = False
    show_progress: bool = True
    # Weight on the structured heuristic score vs. cheap TF-IDF lexical-semantic
    # similarity when ranking candidates for the prefilter cut. Pure 1.0 here
    # reproduces the old keyword-only prefilter bug; 0.6/0.4 lets candidates who
    # describe their work in plain language still survive to deep scoring.
    prefilter_heuristic_weight: float = 0.6


@dataclass
class RankingResult:
    dataframe: pd.DataFrame
    role_profile: StructuredRoleProfile
    explanations: list[ExplanationReport] = field(default_factory=list)

    def save_reports(self, path: str | Path) -> None:
        payload = {
            "role_profile": self.role_profile.to_dict(),
            "candidates": [
                {
                    "rank": e.rank,
                    "candidate_id": e.candidate_id,
                    "name": e.candidate_name,
                    "summary": e.summary,
                    "strengths": e.strengths,
                    "gaps": e.gaps,
                    "factor_scores": e.factor_scores,
                    "reasoning": e.reasoning,
                }
                for e in self.explanations
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CandidateRanker:
    """Hybrid semantic + multi-factor recruiter-style ranker."""

    def __init__(
        self,
        requirements: JobRequirements | None = None,
        role_profile: StructuredRoleProfile | None = None,
        config: RankerConfig | None = None,
    ):
        self.requirements = requirements or load_job_requirements()
        self.role_profile = role_profile or understand_job(self.requirements.jd_text)
        self.config = config or RankerConfig()
        self.embedder = HybridEmbedder(
            use_transformer=self.config.use_transformer and not self.config.tfidf_only
        )
        self._last_result: RankingResult | None = None

    def _semantic_scores(self, documents: list[str]) -> list[float]:
        jd_text = self.role_profile.embedding_text
        if self.config.tfidf_only or not self.config.use_transformer:
            scores = self.embedder.tfidf_similarity(jd_text, documents)
        else:
            scores = self.embedder.hybrid_similarity(jd_text, documents)
        return [float(s) for s in scores]

    def _finalize_shortlist(self, bundles: list[dict[str, Any]]) -> RankingResult:
        bundles.sort(key=lambda item: (-item["final_score"], item["candidate_id"]))
        top = bundles[: self.config.top_k]

        explanations: list[ExplanationReport] = []
        rows = []
        for rank, item in enumerate(top, start=1):
            explanation = build_explanation(
                rank=rank,
                role=self.role_profile,
                profile=item["structured_profile"],
                factors=item["factor_scores"],
                features=item,
            )
            explanations.append(explanation)
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "rank": rank,
                    "score": round(float(item["final_score"]), 4),
                    "reasoning": explanation.reasoning,
                }
            )

        df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
        self._last_result = RankingResult(
            dataframe=df,
            role_profile=self.role_profile,
            explanations=explanations,
        )
        return self._last_result

    def _deep_score_bundles(self, candidates: list[dict]) -> list[dict[str, Any]]:
        bundles: list[dict[str, Any]] = []
        for candidate in candidates:
            bundle = build_feature_bundle(candidate, self.requirements, self.role_profile)
            bundles.append(bundle)

        # Same fix as the streaming path: blend in a cheap TF-IDF lexical-semantic
        # score before the prefilter cut so plain-language fits aren't pruned
        # before the deep semantic re-rank ever sees them.
        if len(bundles) > self.config.prefilter_size:
            all_documents = [item["structured_profile"].summary for item in bundles]
            lexical_scores = self.embedder.tfidf_similarity(self.role_profile.embedding_text, all_documents)
            h_weight = self.config.prefilter_heuristic_weight
            l_weight = 1.0 - h_weight
            for item, l_score in zip(bundles, lexical_scores):
                item["prefilter_score"] = h_weight * item["heuristic_score"] + l_weight * float(l_score)
        else:
            for item in bundles:
                item["prefilter_score"] = item["heuristic_score"]

        bundles.sort(key=lambda item: item["prefilter_score"], reverse=True)
        shortlist = bundles[: self.config.prefilter_size]

        documents = [item["structured_profile"].summary for item in shortlist]
        semantic_scores = self._semantic_scores(documents)

        rescored: list[dict[str, Any]] = []
        for idx, candidate in enumerate([b["candidate"] for b in shortlist]):
            bundle = score_candidate_bundle(
                candidate,
                self.requirements,
                self.role_profile,
                semantic_similarity=semantic_scores[idx],
            )
            rescored.append(bundle)

        # Pseudo LLM re-rank: resort top pool with full factor scores
        rescored.sort(key=lambda item: (-item["final_score"], item["candidate_id"]))
        return rescored[: max(self.config.rerank_pool, self.config.top_k)]

    def rank_candidates(self, candidates: list[dict]) -> pd.DataFrame:
        rescored = self._deep_score_bundles(candidates)
        return self._finalize_shortlist(rescored).dataframe

    def rank_candidates_with_reports(self, candidates: list[dict]) -> RankingResult:
        rescored = self._deep_score_bundles(candidates)
        return self._finalize_shortlist(rescored)

    def rank_file(self, candidates_path: str, limit: int | None = None) -> pd.DataFrame:
        candidates = load_candidates(candidates_path, limit=limit)
        return self.rank_candidates(candidates)

    def rank_file_streaming(self, candidates_path: str) -> pd.DataFrame:
        """Two-pass approach for the full 100K pool within memory limits.

        Pass 1 computes a *blended* prefilter score: the structured heuristic
        score (title/skills/behavioral/etc.) plus a cheap TF-IDF lexical-semantic
        similarity against the JD, computed once in a single vectorized batch
        over the whole pool. Without the TF-IDF term, candidates who describe
        their work in plain language (no JD-matching keywords) but are
        genuinely a strong semantic fit get pruned here and never reach the
        transformer re-rank in Pass 2 — which defeats the purpose of having a
        semantic stage at all. TF-IDF over 100k short documents is sklearn-
        vectorized and takes low single-digit seconds, so this stays well
        within the CPU/time budget.
        """
        start = time.time()
        prefilter_size = self.config.prefilter_size
        heuristic_weight = self.config.prefilter_heuristic_weight
        lexical_weight = 1.0 - heuristic_weight

        candidate_ids: list[str] = []
        heuristic_scores: list[float] = []
        documents: list[str] = []

        iterator = iter_candidates(candidates_path)
        if self.config.show_progress:
            iterator = tqdm(iterator, total=100_000, desc="Pass 1: heuristic scoring", unit="cand")

        for candidate in iterator:
            bundle = build_feature_bundle(candidate, self.requirements, self.role_profile)
            candidate_ids.append(candidate["candidate_id"])
            heuristic_scores.append(bundle["heuristic_score"])
            documents.append(bundle["structured_profile"].summary)

        if self.config.show_progress:
            print("Pass 1b: lexical-semantic prefilter scoring (TF-IDF, vectorized)...")

        lexical_scores = self.embedder.tfidf_similarity(self.role_profile.embedding_text, documents)

        heap: list[tuple[float, str]] = []
        for candidate_id, h_score, l_score in zip(candidate_ids, heuristic_scores, lexical_scores):
            blended = heuristic_weight * h_score + lexical_weight * float(l_score)
            if len(heap) < prefilter_size:
                heapq.heappush(heap, (blended, candidate_id))
            elif blended > heap[0][0]:
                heapq.heapreplace(heap, (blended, candidate_id))

        shortlisted_ids = {candidate_id for _, candidate_id in heap}

        candidates: list[dict] = []
        iterator = iter_candidates(candidates_path)
        if self.config.show_progress:
            iterator = tqdm(iterator, total=100_000, desc="Pass 2: load shortlist", unit="cand")

        for candidate in iterator:
            if candidate["candidate_id"] in shortlisted_ids:
                candidates.append(candidate)

        rescored = self._deep_score_bundles(candidates)
        result = self._finalize_shortlist(rescored)

        if self.config.show_progress:
            print(f"Ranking completed in {time.time() - start:.1f}s")

        return result.dataframe
