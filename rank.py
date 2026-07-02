#!/usr/bin/env python3
"""CLI entrypoint for Redrob candidate ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.candidate_loader import load_candidates
from src.jd_parser import load_job_requirements
from src.job_understanding import understand_job
from src.hybrid_ranker import CandidateRanker, RankerConfig
from src.rule_ranker import RuleBasedRanker, RuleRankerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Senior AI Engineer JD")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--jd", default=None, help="Optional job description file (.docx/.md/.txt)")
    parser.add_argument("--reports", default=None, help="Optional JSON path for explainability reports")
    parser.add_argument("--html", default=None, help="Optional HTML shortlist path (card UI format)")
    parser.add_argument(
        "--mode",
        choices=["rule", "hybrid", "compare"],
        default="rule",
        help=(
            "Ranking mode: rule (default, fast JD-aligned), hybrid (semantic "
            "multi-factor), or compare (runs both, writes rule mode as --out and "
            "hybrid mode as <out>_hybrid.csv, and prints a top-10/top-100 "
            "agreement report so you can see how much the two modes diverge "
            "before choosing what to submit)"
        ),
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to rank")
    parser.add_argument("--prefilter", type=int, default=2500, help="Hybrid mode: candidates to deep-score")
    parser.add_argument("--limit", type=int, default=None, help="Optional candidate limit for sandbox demos")
    parser.add_argument("--tfidf-only", action="store_true", help="Hybrid mode: skip transformer model")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    return parser.parse_args()


def _run_rule(args: argparse.Namespace, requirements, role, candidates_path: str):
    ranker = RuleBasedRanker(
        requirements=requirements,
        role_profile=role,
        config=RuleRankerConfig(top_k=args.top_k, show_progress=not args.quiet),
    )
    return ranker.rank_file_with_reports(candidates_path, limit=args.limit)


def _run_hybrid(args: argparse.Namespace, requirements, role, candidates_path: str):
    config = RankerConfig(
        top_k=args.top_k,
        prefilter_size=args.prefilter,
        use_transformer=not args.tfidf_only,
        tfidf_only=args.tfidf_only,
        show_progress=not args.quiet,
    )
    ranker = CandidateRanker(requirements=requirements, role_profile=role, config=config)
    if args.limit:
        return ranker.rank_candidates_with_reports(load_candidates(candidates_path, limit=args.limit))
    elif Path(candidates_path).stat().st_size > 50_000_000:
        ranker.rank_file_streaming(candidates_path)
        return ranker._last_result
    else:
        return ranker.rank_candidates_with_reports(load_candidates(candidates_path))


def _report_agreement(rule_result, hybrid_result) -> None:
    """Surface how much rule and hybrid mode agree, rather than letting two
    silently divergent rankings ship without comment. A judge re-running with
    a different --mode flag should see *why* the lists differ, not just get
    a different unexplained answer.
    """
    rule_ids = list(rule_result.dataframe["candidate_id"])
    hybrid_ids = list(hybrid_result.dataframe["candidate_id"])
    rule_set, hybrid_set = set(rule_ids), set(hybrid_ids)
    overlap = rule_set & hybrid_set
    top10_overlap = set(rule_ids[:10]) & set(hybrid_ids[:10])

    print("\n--- Mode agreement report ---")
    print(f"  Top-100 overlap : {len(overlap)}/100 candidates appear in both shortlists")
    print(f"  Top-10 overlap  : {len(top10_overlap)}/10 candidates appear in both top-10s")
    if len(top10_overlap) < 5:
        print(
            "  WARNING: low top-10 agreement between modes. This usually means the rule\n"
            "  scorer's weights and the hybrid multi-factor scorer's weights are not\n"
            "  aligned on what 'fit' means for this JD. Treat one mode as primary before\n"
            "  submitting, and investigate the divergence rather than picking either\n"
            "  output blind."
        )
    print("------------------------------\n")


def main() -> int:
    args = parse_args()
    requirements = load_job_requirements(args.jd)
    role = understand_job(requirements.jd_text)
    candidates_path = str(Path(args.candidates))

    if args.mode == "compare":
        print("Running rule mode...")
        rule_result = _run_rule(args, requirements, role, candidates_path)
        print("Running hybrid mode...")
        hybrid_result = _run_hybrid(args, requirements, role, candidates_path)
        _report_agreement(rule_result, hybrid_result)
        # Primary output is rule mode (the documented default); hybrid is
        # written alongside with a suffix so both are inspectable.
        result = rule_result
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hybrid_path = out_path.with_name(out_path.stem + "_hybrid" + out_path.suffix)
        hybrid_result.dataframe.to_csv(hybrid_path, index=False)
        print(f"Wrote hybrid-mode comparison output to {hybrid_path}")
    elif args.mode == "rule":
        result = _run_rule(args, requirements, role, candidates_path)
    else:
        result = _run_hybrid(args, requirements, role, candidates_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(out_path, index=False)
    print(f"Wrote {len(result.dataframe)} ranked candidates to {out_path} (mode={args.mode})")

    if args.reports:
        result.save_reports(args.reports)
        print(f"Wrote explainability reports to {args.reports}")

    if args.html and hasattr(result, "save_html"):
        result.save_html(args.html)
        print(f"Wrote HTML shortlist to {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
