# True Fit – AI Intelligent Candidate Ranker

An explainable AI-powered recruiting system that ranks candidates the way experienced recruiters do—not by matching keywords, but by understanding job requirements, career history, technical depth, and behavioral signals.

Built for the [Redrob Intelligent Candidate Discovery & Ranking Challenge](https://github.com/) dataset (`candidates.jsonl`, 100K profiles).

## Architecture

Job Description
        │
        ▼
Semantic Job Understanding (skills, traits, culture, seniority)
        │
        ▼
Structured Role Profile
        │
        ▼
Candidate Profile Parser → Knowledge Graph + Recruiter Summary
        │
        ▼
Embeddings + Vector Search (summary-based, not keyword tags)
        │
        ▼
Hybrid Multi-Factor Scoring Engine
    ├── Technical Fit
    ├── Domain Experience
    ├── Leadership
    ├── Communication
    ├── Learning Ability
    ├── Culture Fit
    ├── Project Relevance
    └── Behavioral Signals
        │
        ▼
Pseudo Re-Ranking (cross-factor alignment, CPU-only)
        │
        ▼
Explainable AI Reports + Recruiter Dashboard + Chat Assistant
```

## What it does

1. **Semantic Job Understanding** — infers required/nice-to-have skills, seniority, culture, leadership, and traits like *ownership mindset* (not just keyword tags).
2. **Candidate Understanding** — parses each profile into structured experience, projects, tech stack, achievements, and a recruiter summary used for embeddings.
3. **Semantic Skill Matching** — maps PyTorch→ML, Kafka→Distributed Systems via skill hierarchy graph.
4. **Multi-Factor Scoring** — transparent 0–100 scores across 8 dimensions plus final score.
5. **Explainable AI** — per-candidate strengths (✓), gaps (•), and factor breakdown.
6. **Recruiter Dashboard** — Streamlit UI with rankings, JD view, candidate drill-down, and a keyword-matched chat assistant over the ranked results (intent-pattern matching, not an LLM — fully offline, no API calls).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Rank the full 100K pool (rule mode, default — ~30s CPU)
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --jd data/job_description.docx \
  --out submission.csv

# Hybrid semantic mode (optional, ~2 min CPU with --tfidf-only on the full pool)
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --out submission.csv \
  --mode hybrid \
  --tfidf-only

# Compare both modes and see how much they agree before picking one (recommended)
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --out submission.csv \
  --mode compare \
  --tfidf-only

# Validate format before upload
python validate_submission.py submission.csv --candidates /path/to/candidates.jsonl

# Pre-flight check before zipping for the portal
python3 scripts/check_submission_ready.py

# Generate explainability reports (JSON)
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --out submission.csv \
  --reports output/reports.json
```

### Sandbox demo (Streamlit)

```bash
streamlit run app.py
```

Deploy to [Streamlit Cloud](https://streamlit.io/cloud) for the required sandbox link. The app ranks the bundled 50-candidate sample or an uploaded JSON/JSONL file.

### Fast local demo (TF-IDF only, no model download)

```bash
python rank.py \
  --candidates data/sample_candidates.json \
  --out output/demo.csv \
  --limit 50 \
  --tfidf-only
```

## Compute constraints

The ranking step is designed to meet hackathon limits:

| Constraint | Design choice |
|---|---|
| ≤ 5 min CPU | Two-pass streaming; embeddings only on top 2,500 |
| ≤ 16 GB RAM | Heap-based pre-filter; no full-pool embedding matrix |
| No network | Local `sentence-transformers` model, cached after first run |
| No GPU | CPU inference with MiniLM |

Pre-computation (first model download) may happen once during setup; the `rank.py` ranking step itself makes no external API calls.

## Project layout

```
redrob-recruiter-ranker/
├── rank.py                  # Main CLI — reproduce submission
├── app.py                   # Streamlit sandbox
├── validate_submission.py   # CSV format validator
├── submission_metadata.yaml # Portal metadata — fill in before submitting
├── requirements.txt
├── scripts/
│   └── check_submission_ready.py  # Pre-flight check: run before zipping
├── data/
│   ├── sample_candidates.json
│   └── job_description.docx
└── src/
    ├── job_understanding.py   # Semantic JD parser
    ├── candidate_profile.py   # Structured profile + summary
    ├── skill_graph.py         # Semantic skill hierarchy
    ├── multi_factor_scorer.py # 8-factor transparent scoring (hybrid mode)
    ├── explainer.py           # Why-ranked reports (hybrid mode)
    ├── chat_assistant.py      # Offline keyword-based recruiter chat — no LLM
    ├── jd_parser.py           # JD loader, with a baked-in JD fallback
    ├── feature_engineering.py # Profile + signal features
    ├── honeypot_detector.py   # Trap detection
    ├── embeddings.py          # TF-IDF + sentence-transformer similarity
    ├── hybrid_scorer.py       # Hybrid mode scoring orchestration
    ├── hybrid_ranker.py       # Hybrid mode end-to-end pipeline
    ├── rule_scorer.py         # Rule mode scoring — the documented default
    └── rule_ranker.py         # Rule mode end-to-end pipeline
```

**Rule mode vs. hybrid mode are two independently-weighted scoring
philosophies**, not a base/upgrade pair — running both on the same pool
typically produces well under half top-10 agreement. Use `--mode compare`
to see exactly how much they diverge before deciding which to submit; see
"Mode agreement" below.

| Signal | Weight / role |
|---|---|
| Title relevance | Penalizes Marketing/HR/etc.; boosts AI/ML/Search engineers |
| Career production markers | "shipped", "production", "A/B test", "ranking system" |
| Skill depth | Proficiency × duration; retrieval/ranking skills weighted higher |
| Consulting ratio | Down-ranks services-heavy careers per JD disqualifiers |
| Behavioral signals | Response rate, recency, notice period, open-to-work |
| Semantic similarity | Catches plain-language fits without keyword stuffing |
| Honeypot penalty | Timeline/skill inconsistencies → near-zero score |

## Mode agreement

Rule mode (default) and hybrid mode use different weighting schemes and
disagree more than you'd expect — on the full 100k pool, top-10 overlap is
typically around 2/10. Before picking which one to submit, run:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --mode compare --tfidf-only
```

This writes rule mode's output to `submission.csv`, hybrid mode's output to
`submission_hybrid.csv`, and prints a top-10/top-100 overlap report. Treat
low agreement as a signal to actually read both shortlists and decide which
one's reasoning you trust more for this JD, not as something to average away.

The hybrid prefilter (the step that cuts the pool to ~2,500 candidates
before deep/semantic scoring) blends the structured heuristic score with a
cheap TF-IDF lexical-semantic score against the JD (`prefilter_heuristic_weight`
in `RankerConfig`, default 0.6/0.4). A pure-heuristic prefilter would prune
candidates who describe their work in plain language before the semantic
re-rank ever sees them — exactly the keyword-filter trap this system exists
to avoid — so don't set that weight to 1.0 without re-checking what gets cut.

## Submission

Reproduce the CSV with a single command:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Before zipping for the portal, run the pre-flight check:

```bash
python3 scripts/check_submission_ready.py
```

It catches placeholder values left in `submission_metadata.yaml`, symlinks
that point outside the repo (these work locally but break for anyone else
who unzips the project), and junk directories (`.venv/`, `__pycache__/`,
`output/`) that bloat the zip and shouldn't ship. Fill in
`submission_metadata.yaml` with your real team details — every placeholder
is marked with a `TODO_` prefix so it can't be missed.

## Methodology

Keyword filters miss the point of this JD. The released job description explicitly warns that Marketing Managers with AI keywords in their skills section are **not** fits, while product engineers who built recommendation/search systems in plain language **are**.

This system encodes that judgment through:
- **Structured disqualifiers** tied to JD language
- **Skill duration/proficiency** instead of skill-name counting
- **Behavioral availability** as a multiplier
- **Semantic embeddings** over full career narratives, not skill tags alone

## License

MIT
