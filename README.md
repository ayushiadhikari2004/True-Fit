True Fit – AI Intelligent Candidate Ranker

An explainable AI-powered recruiting system that ranks candidates the way experienced recruiters do—not by matching keywords, but by understanding job requirements, career history, technical depth, and behavioral signals.

Built for the [Redrob Intelligent Candidate Discovery & Ranking Challenge](https://github.com/) dataset (`candidates.jsonl`, 100K profiles).

Architecture

                    Job Description
                           │
                           ▼
            Semantic Job Understanding
      (skills, culture, seniority, traits)
                           │
                           ▼
              Structured Role Profile
                           │
                           ▼
          Candidate Profile Understanding
     (experience, projects, achievements)
                           │
                           ▼
      Recruiter Summary + Knowledge Graph
                           │
                           ▼
     Semantic Search + Hybrid Candidate Retrieval
                           │
                           ▼
        Multi-Factor AI Scoring Engine
      ├── Technical Fit
      ├── Domain Expertise
      ├── Leadership
      ├── Communication
      ├── Learning Ability
      ├── Project Relevance
      ├── Culture Fit
      └── Behavioral Signals
                           │
                           ▼
                Explainable Re-ranking
                           │
                           ▼
    Recruiter Dashboard • Analytics • AI Assistant

Features:

1. Semantic Job Understanding – Extracts required skills, seniority, leadership expectations, culture fit, and implicit traits from the JD.

2. Deep Candidate Understanding – Builds structured candidate profiles from experience, projects, skills, and achievements.

3. Semantic Matching – Matches related skills (e.g., PyTorch → Machine Learning, Kafka → Distributed Systems) instead of relying on exact keywords.

4. Multi-Factor Ranking Engine – Scores candidates across multiple recruiter-relevant dimensions.

5. Explainable AI – Every ranking includes strengths, gaps, score breakdown, and recruiter-friendly reasoning.

6. Interactive Recruiter Dashboard – Streamlit interface with rankings, analytics, filters, candidate comparison, and AI-assisted insights.

7. CPU-Optimized Pipeline – Processes large candidate pools efficiently without requiring GPUs or external APIs.

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

The hybrid prefilter (the step that cuts the pool to ~2,500 candidates
before deep/semantic scoring) blends the structured heuristic score with a
cheap TF-IDF lexical-semantic score against the JD (`prefilter_heuristic_weight`
in `RankerConfig`, default 0.6/0.4). A pure-heuristic prefilter would prune
candidates who describe their work in plain language before the semantic
re-rank ever sees them — exactly the keyword-filter trap this system exists
to avoid — so don't set that weight to 1.0 without re-checking what gets cut.


## Methodology

Keyword filters miss the point of this JD. The released job description explicitly warns that Marketing Managers with AI keywords in their skills section are **not** fits, while product engineers who built recommendation/search systems in plain language **are**.

This system encodes that judgment through:
- **Structured disqualifiers** tied to JD language
- **Skill duration/proficiency** instead of skill-name counting
- **Behavioral availability** as a multiplier
- **Semantic embeddings** over full career narratives, not skill tags alone

## License

MIT
