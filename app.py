from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chat_assistant import answer_recruiter_query
from src.jd_parser import load_job_requirements
from src.job_understanding import understand_job
from src.hybrid_ranker import CandidateRanker, RankerConfig
from src.rule_ranker import RuleBasedRanker, RuleRankerConfig
from src.ui_components import render_candidate_card
from src.enhanced_signals import EnhancedSignalConfig
from src.badge_system import render_badges_html

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="True Fit · AI Recruiter",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ---- font & base ---- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, .stApp * {
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] {
  background: #13141a !important;
  border-right: 1px solid #1e2028 !important;
  width: 252px !important;
  min-width: 252px !important;
  max-width: 252px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ---- keyframes ---- */
@keyframes countUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes fillProgress { from{width:0%} }
@keyframes fadeInSlide { from{opacity:0;transform:translateY(15px)} to{opacity:1;transform:translateY(0)} }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

.stat-val { animation: countUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards; }
.dim-fill, .score-fill, .bar-fill { animation: fillProgress 1s cubic-bezier(0.16,1,0.3,1) forwards; }
.candidate-card, .analytics-card, .jd-section, .compare-table { animation: fadeInSlide 0.5s cubic-bezier(0.16,1,0.3,1) forwards; }

/* ---- sidebar brand ---- */
.tf-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 20px 16px 14px 16px;
  border-bottom: 1px solid #1e2028;
  margin-bottom: 6px;
}
.tf-logo {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 800; color: #fff;
  flex-shrink: 0; letter-spacing: -0.5px;
}
.tf-name-block {}
.tf-name {
  font-family: 'Outfit', sans-serif !important;
  font-size: 17px !important; font-weight: 800 !important;
  color: #f1f1f3 !important; letter-spacing: -0.4px !important;
  line-height: 1.2 !important;
}
.tf-tag {
  font-size: 10px !important; color: #4b5563 !important;
  letter-spacing: 0.01em !important; line-height: 1.4 !important;
  margin-top: 1px !important;
}

/* ---- nav section label ---- */
.nav-section-label {
  font-size: 10px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.09em !important;
  color: #50566a !important; padding: 14px 16px 5px 16px !important;
}

/* ---- nav items ---- */
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; border-radius: 0; margin: 1px 0;
  cursor: pointer; transition: background 0.12s ease, color 0.12s ease;
  color: #818898; font-size: 13.5px; font-weight: 500;
  border-left: 2px solid transparent; text-decoration: none !important;
  user-select: none;
}
.nav-item:hover { background: #1a1c26; color: #d4d8e2; border-left-color: #32364a; }
.nav-item.active {
  background: #1c1f32; color: #eaecf4; font-weight: 600;
  border-left-color: #6366f1;
}
.nav-icon {
  font-size: 14px; width: 20px; text-align: center; flex-shrink: 0;
  opacity: 0.75;
}
.nav-item.active .nav-icon, .nav-item:hover .nav-icon { opacity: 1; }
.nav-divider { height: 1px; background: #1c1f28; margin: 6px 0; }

/* ---- sidebar controls — full dark theme ---- */

/* Labels — all section headers */
[data-testid="stSidebar"] label {
  color: #8892a4 !important; font-size: 11px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
}

/* Radio option text */
[data-testid="stSidebar"] .stRadio label p {
  color: #c5cad4 !important; font-size: 13px !important;
  text-transform: none !important; letter-spacing: 0 !important; font-weight: 500 !important;
}

/* Slider */
[data-testid="stSidebar"] .stSlider label { font-size: 11px !important; }

/* Number inputs — dark */
[data-testid="stSidebar"] [data-baseweb="input"] {
  background: #1a1c27 !important;
  border: 1px solid #262934 !important;
  border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] [data-baseweb="input"] input {
  background: transparent !important;
  color: #e2e5ed !important; font-size: 13px !important;
}

/* Textarea — dark */
[data-testid="stSidebar"] textarea {
  background: #1a1c27 !important; border: 1px solid #262934 !important;
  color: #e2e5ed !important; border-radius: 7px !important; font-size: 12px !important;
}
[data-testid="stSidebar"] textarea::placeholder { color: #3a3f52 !important; }
[data-testid="stSidebar"] textarea:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important; outline: none !important;
}

/* File uploader — dark */
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
  background: #1a1c27 !important; border: 1px dashed #2a2d3d !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {
  border-color: #6366f1 !important; background: #1c1f33 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small { color: #50566a !important; }
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span { color: #818cf8 !important; }

/* Checkbox */
[data-testid="stSidebar"] .stCheckbox label p {
  color: #c5cad4 !important; font-size: 13px !important;
  text-transform: none !important; letter-spacing: 0 !important; font-weight: 500 !important;
}

/* Expander inside sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: #16181f !important; border: 1px solid #22252f !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] details > summary {
  color: #8892a4 !important; font-size: 11px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.07em !important;
  padding: 10px 14px !important;
}
[data-testid="stSidebar"] details[open] > summary { color: #c5cad4 !important; }

/* Success / info banners */
[data-testid="stSidebar"] [data-testid="stAlert"] {
  background: #0d1f12 !important; border: 1px solid #1a3a20 !important;
  border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p { color: #6ee7b7 !important; font-size: 12px !important; }

/* Buttons — uniform */
[data-testid="stSidebar"] .stButton button {
  border-radius: 8px !important; font-size: 13px !important;
  font-weight: 600 !important; height: 36px !important;
  transition: all 0.15s ease !important; width: 100% !important;
}
/* Primary — Run Ranking */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  border: none !important; color: #fff !important;
  box-shadow: 0 2px 8px rgba(99,102,241,0.28) !important;
}
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
  background: linear-gradient(135deg, #4f52d4, #6927cc) !important;
  box-shadow: 0 4px 14px rgba(99,102,241,0.38) !important;
}
/* Secondary — Compare Modes, Apply JD, etc. */
[data-testid="stSidebar"] .stButton button[kind="secondary"] {
  background: #1c1f2e !important; border: 1px solid #2a2d3d !important; color: #c5cad4 !important;
}
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
  background: #21253a !important; border-color: #6366f1 !important; color: #eaecf4 !important;
}

/* ---- main header ---- */
.rr-header { margin-bottom: 1.5rem; border-bottom: 1px solid #1e2028; padding-bottom: 1rem; }
.rr-title {
  font-family: 'Outfit', sans-serif !important;
  font-size: 28px; font-weight: 800; color: #f1f1f3;
  letter-spacing: -0.6px; line-height: 1.2;
}
.rr-sub { font-size: 13px; color: #4b5563; margin-top: 4px; }

/* ---- stats strip ---- */
.stats-strip { display:flex;gap:1px;margin:1rem 0 1.5rem;border-radius:10px;overflow:hidden;border:1px solid #1e2028; }
.stat-cell { flex:1;background:#16171f;padding:12px 16px; }
.stat-label { font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:.05em; }
.stat-val { font-size:22px;font-weight:700;color:#f1f1f3;margin-top:2px; }
.stat-sub { font-size:11px;color:#374151;margin-top:1px; }

/* ---- dim bars ---- */
.dim-grid { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px; }
.dim-cell { background:#16171f;border-radius:8px;padding:10px 12px; }
.dim-label { font-size:11px;color:#4b5563;margin-bottom:5px; }
.dim-track { height:5px;background:#1e2028;border-radius:3px;overflow:hidden; }
.dim-fill { height:100%;border-radius:3px; }
.dim-val { font-size:12px;font-weight:600;color:#9ca3af;margin-top:4px; }

/* ---- analytics ---- */
.analytics-grid { display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:1rem 0; }
.analytics-card { background:#16171f;border:1px solid #1e2028;border-radius:12px;padding:16px 18px; }
.analytics-card-title { font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4b5563;margin-bottom:12px; }
.analytics-metric { font-size:28px;font-weight:700;color:#f1f1f3; }
.analytics-sub { font-size:12px;color:#4b5563;margin-top:2px; }
.bar-chart-row { display:flex;align-items:center;gap:8px;margin:6px 0; }
.bar-label { font-size:12px;color:#6b7280;min-width:70px;text-align:right; }
.bar-track { flex:1;height:18px;background:#1e2028;border-radius:4px;overflow:hidden;position:relative; }
.bar-fill { height:100%;border-radius:4px;transition:width 0.3s; }
.bar-count { font-size:11px;color:#9ca3af;min-width:28px; }

/* ---- SVG charts ---- */
.donut-segment { transition:stroke 0.3s,stroke-width 0.3s; }
.donut-segment:hover { stroke-width:20px; }
.pie-slice { transition:transform 0.3s,fill 0.3s; }
.pie-slice:hover { transform:scale(1.03);filter:brightness(1.1); }

/* ---- status pills ---- */
.status-pill { padding:3px 8px;border-radius:6px;font-size:12px;font-weight:600;display:inline-flex;align-items:center;gap:4px; }
.status-excellent { background:#0a2318;color:#4ade80;border:1px solid #14532d; }
.status-warning { background:#271e0f;color:#fb923c;border:1px solid #7c2d12; }
.status-poor { background:#270f0f;color:#f87171;border:1px solid #7f1d1d; }

/* ---- skeletons ---- */
.skeleton-list { display:flex;flex-direction:column;gap:12px; }
.skeleton-card { background:#16171f;border:1px solid #1e2028;border-radius:12px;padding:18px; }
.skeleton-circle { width:42px;height:42px;border-radius:50%;background:#1e2028; }
.skeleton-line { background:#1e2028;border-radius:4px; }
.shimmer { background:linear-gradient(90deg,#16171f 25%,#1e2028 50%,#16171f 75%);background-size:200% 100%;animation:shimmer 1.5s infinite linear; }

/* ---- insights panel ---- */
.insights-panel { background:#141722;border:1px solid #312e81;border-radius:12px;padding:16px 18px;margin:12px 0; }
.insights-title { font-size:13px;font-weight:700;color:#818cf8;display:flex;align-items:center;gap:6px;margin-bottom:8px; }
.insights-item { font-size:13px;color:#9ca3af;margin:4px 0;line-height:1.5; }

/* ---- jd profile ---- */
.jd-section { background:#16171f;border-radius:10px;padding:16px 18px;margin-bottom:10px; }
.jd-sec-title { font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4b5563;margin-bottom:10px; }
.jd-chip { display:inline-block;font-size:12px;padding:3px 10px;border-radius:999px;margin:3px 3px 3px 0;background:#16171f;color:#9ca3af;border:1px solid #1e2028; }
.jd-chip.must { background:#0a2318;color:#4ade80;border-color:#14532d; }
.jd-chip.nice { background:#11182b;color:#818cf8;border-color:#1e40af; }
.jd-chip.disq { background:#270f0f;color:#f87171;border-color:#7f1d1d; }
.jd-chip.trait { background:#271e0f;color:#fb923c;border-color:#7c2d12; }

/* ---- compare table ---- */
.compare-table { border-collapse:collapse;width:100%;font-size:13px; }
.compare-table th { background:#16171f;color:#6b7280;font-weight:600;padding:8px 12px;text-align:left;border-bottom:1px solid #1e2028; }
.compare-table td { padding:8px 12px;border-bottom:1px solid #16171f;color:#9ca3af; }
.compare-table tr:hover td { background:#16171f; }
.match { color:#4ade80; }
.no-match { color:#f87171; }

/* ---- expander ---- */
[data-testid="stExpander"] { border:1px solid #1e2028 !important;border-radius:10px !important;background:#13141a !important;margin-bottom:8px; }
details summary { color:#6b7280 !important;font-size:13px !important; }
details[open] summary { color:#d1d5db !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [
    ("ranking_result", None),
    ("compare_result", None),
    ("chat_history", []),
    ("last_mode", None),
    ("jd_text_override", None),
    ("weight_salary_fit", 0.08),
    ("weight_offer_acceptance", 0.06),
    ("weight_identity_verification", 0.04),
    ("weight_profile_completeness", 0.04),
    ("ui_theme", "Dark"),
    ("run_ranking_trigger", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Dynamic Theme & Blue Sidebar Text ──────────────────────────────────────────

current_theme = st.session_state.get("ui_theme", "Dark")

if current_theme == "Light":
    theme_css = """
    <style>
    [data-testid="stAppViewContainer"] { background: #f9fafb !important; }
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e5e7eb !important; }
    .tf-brand { border-bottom: 1px solid #e5e7eb; }
    .tf-name { color: #111827 !important; }
    .tf-tag { color: #6b7280 !important; }
    .rr-header { border-bottom: 1px solid #e5e7eb; }
    .rr-title { color: #111827 !important; }
    .rr-sub { color: #6b7280 !important; }
    .stat-cell { background: #ffffff; border: 1px solid #e5e7eb; }
    .stat-label, .stat-sub { color: #6b7280 !important; }
    .stat-val { color: #111827 !important; }
    .dim-cell { background: #f3f4f6; }
    .dim-val { color: #111827; }
    .dim-track { background: #e5e7eb; }
    .dim-label { color: #6b7280 !important; }
    .analytics-card { background: #ffffff; border: 1px solid #e5e7eb; }
    .analytics-metric { color: #111827; }
    .bar-track { background: #e5e7eb; }
    .skeleton-card { background: #ffffff; border: 1px solid #e5e7eb; }
    .skeleton-circle, .skeleton-line { background: #e5e7eb; }
    .jd-section { background: #ffffff; border: 1px solid #e5e7eb; }
    .jd-chip { background: #f3f4f6 !important; border-color: #e5e7eb !important; color: #374151 !important; }
    .compare-table th { background: #f9fafb !important; border-bottom: 1px solid #e5e7eb !important; color: #374151 !important; }
    .compare-table td { border-bottom: 1px solid #f3f4f6 !important; color: #111827 !important; }
    .compare-table tr:hover td { background: #f9fafb !important; }
    .insights-panel { background: #ffffff; border: 1px solid #bfdbfe; }
    .insights-item { color: #374151; }
    [data-testid="stExpander"] { background: #f9fafb !important; border: 1px solid #e5e7eb !important; }
    details summary { color: #374151 !important; }
    </style>
    """
else:
    theme_css = ""

blue_sidebar_css = """
<style>
/* Force all sidebar text to blue (#60a5fa) per request */
.nav-section-label, .nav-item, [data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .stRadio label p, [data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea, 
[data-testid="stSidebar"] details summary,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] span,
[data-testid="stSidebar"] .stSlider label {
  color: #60a5fa !important;
}
.nav-item:hover, .nav-item.active { color: #93c5fd !important; }
[data-testid="stSidebar"] [data-baseweb="input"] input, [data-testid="stSidebar"] textarea { color: #60a5fa !important; }
</style>
"""

st.markdown(theme_css + blue_sidebar_css, unsafe_allow_html=True)

# ── Navigation state via query params ────────────────────────────────────────

_PAGES = ["Dashboard", "Rankings", "Candidates", "Analytics", "Job Profile", "AI Assistant", "Settings"]
_PAGE_ICONS = {
    "Dashboard":   "▣",
    "Rankings":    "◈",
    "Candidates":  "◉",
    "Analytics":   "◆",
    "Job Profile": "◎",
    "AI Assistant":"◇",
    "Settings":    "⊙",
}

_qp = st.query_params.get("page", "Dashboard")
nav_selection = _qp if _qp in _PAGES else "Dashboard"

# ── Data & JD ────────────────────────────────────────────────────────────────

sample_path = ROOT / "data" / "sample_candidates.json"
jd_path     = ROOT / "data" / "job_description.docx"

# JD override from session state (set by sidebar upload)
_jd_override = st.session_state.get("jd_text_override", None)
if _jd_override:
    from src.job_understanding import understand_job as _uj
    from src.jd_parser import JobRequirements
    role = _uj(_jd_override)
    requirements = JobRequirements(jd_text=_jd_override, role_title=role.role_title, min_years=role.min_years,
                                   ideal_years_min=role.min_years, ideal_years_max=role.max_years,
                                   sweet_spot_years=role.ideal_years)
else:
    requirements = load_job_requirements(str(jd_path) if jd_path.exists() else None)
    role = understand_job(requirements.jd_text)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="tf-brand">
      <div class="tf-logo">TF</div>
      <div class="tf-name-block">
        <div class="tf-name">True Fit</div>
        <div class="tf-tag">AI Recruiting Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation items as clickable links
    st.markdown('<div class="nav-section-label">Navigate</div>', unsafe_allow_html=True)
    for _pg in ["Dashboard", "Rankings", "Candidates", "Analytics", "Job Profile"]:
        _active = "active" if nav_selection == _pg else ""
        _icon   = _PAGE_ICONS[_pg]
        st.markdown(
            f'<a href="?page={_pg}" target="_self" class="nav-item {_active}">'
            f'<span class="nav-icon">{_icon}</span>{_pg}</a>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">Tools</div>', unsafe_allow_html=True)
    for _pg in ["AI Assistant", "Settings"]:
        _active = "active" if nav_selection == _pg else ""
        _icon   = _PAGE_ICONS[_pg]
        st.markdown(
            f'<a href="?page={_pg}" target="_self" class="nav-item {_active}">'
            f'<span class="nav-icon">{_icon}</span>{_pg}</a>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">Theme</div>', unsafe_allow_html=True)
    st.radio("UI Theme", ["Dark", "Light"], key="ui_theme", horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    # ── Ranking controls
    st.markdown('<div class="nav-section-label" style="padding-top:6px">Ranking</div>', unsafe_allow_html=True)
    with st.container():
        ranking_mode = st.radio("Mode", ["rule", "hybrid"], index=0, horizontal=True,
            help="rule = fast explainable scorer · hybrid = adds TF-IDF similarity")
        top_k = st.slider("Shortlist size", 10, 100, 25, 5)

    # ── Salary Budget
    st.markdown('<div class="nav-section-label">Salary Budget (LPA)</div>', unsafe_allow_html=True)
    sal_col1, sal_col2 = st.columns(2)
    with sal_col1:
        salary_min = st.number_input("Min", 5.0, 100.0, 15.0, 5.0, key="sal_min")
    with sal_col2:
        salary_max = st.number_input("Max", 10.0, 200.0, 45.0, 5.0, key="sal_max")

    enhanced_cfg = EnhancedSignalConfig(
        salary_budget_min_lpa=salary_min,
        salary_budget_max_lpa=salary_max,
        weight_salary_fit=st.session_state.weight_salary_fit,
        weight_offer_acceptance=st.session_state.weight_offer_acceptance,
        weight_identity_verification=st.session_state.weight_identity_verification,
        weight_profile_completeness=st.session_state.weight_profile_completeness,
    )

    # ── Candidate data source
    st.markdown('<div class="nav-section-label">Candidates</div>', unsafe_allow_html=True)
    use_sample = st.checkbox("Use bundled sample data", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload candidates (JSON/JSONL)", type=["json", "jsonl"])

    # ── Job Description upload
    st.markdown('<div class="nav-section-label">Job Description</div>', unsafe_allow_html=True)
    with st.expander("Upload / paste JD", expanded=False):
        jd_file = st.file_uploader("Upload JD (.txt / .docx)", type=["txt", "docx"], key="jd_upload")
        jd_paste = st.text_area("Or paste JD text", height=120, key="jd_paste",
                                placeholder="Paste job description here...")
        if st.button("Apply JD", use_container_width=True, key="apply_jd"):
            _txt = None
            if jd_file:
                if jd_file.name.endswith(".docx"):
                    import zipfile, xml.etree.ElementTree as ET
                    try:
                        with zipfile.ZipFile(jd_file) as zf:
                            _root = ET.fromstring(zf.read("word/document.xml"))
                        _ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                        _txt = "\n".join(
                            "".join(n.text for n in p.iter(_ns+"t") if n.text)
                            for p in _root.iter(_ns+"p")
                        )
                    except Exception:
                        _txt = jd_file.read().decode("utf-8", errors="ignore")
                else:
                    _txt = jd_file.read().decode("utf-8", errors="ignore")
            elif jd_paste and jd_paste.strip():
                _txt = jd_paste.strip()
            if _txt:
                st.session_state["jd_text_override"] = _txt
                st.session_state.ranking_result = None
                st.rerun()
        if st.session_state.get("jd_text_override"):
            st.success("Custom JD active.")
            if st.button("Clear JD", use_container_width=True, key="clear_jd"):
                del st.session_state["jd_text_override"]
                st.session_state.ranking_result = None
                st.rerun()

    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    compare_modes = st.button("Compare Modes", use_container_width=True,
                               help="Runs both rule + hybrid and shows agreement table")

# ── Load candidates helper ────────────────────────────────────────────────────

def load_candidates_from_input() -> list[dict] | None:
    if use_sample and sample_path.exists():
        return json.loads(sample_path.read_text(encoding="utf-8"))
    if uploaded:
        content = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            return [json.loads(line) for line in content.splitlines() if line.strip()]
        payload = json.loads(content)
        return payload if isinstance(payload, list) else [payload]
    st.error("Enable 'Use bundled sample data' or upload a candidates file.")
    return None


def run_ranker(candidates: list[dict], mode: str, k: int):
    if mode == "rule":
        ranker = RuleBasedRanker(
            requirements=requirements,
            role_profile=role,
            config=RuleRankerConfig(
                top_k=min(k, len(candidates)),
                show_progress=False,
                enhanced_config=enhanced_cfg,
                enable_enhanced_signals=True,
            ),
        )
    else:
        ranker = CandidateRanker(
            requirements=requirements,
            role_profile=role,
            config=RankerConfig(
                top_k=min(k, len(candidates)),
                prefilter_size=min(500, len(candidates)),
                tfidf_only=True,
                show_progress=False,
            ),
        )
    return ranker.rank_candidates_with_reports(candidates)


# [ignoring loop detection]

# ── Skeletons ─────────────────────────────────────────────────────────────────

SKELETON_HTML = """
<div class="skeleton-list">
  <div class="skeleton-card">
    <div style="display:flex;gap:12px;align-items:center">
      <div class="skeleton-circle shimmer"></div>
      <div style="flex:1">
        <div class="skeleton-line shimmer" style="width:30%;height:14px;margin-bottom:8px"></div>
        <div class="skeleton-line shimmer" style="width:50%;height:10px"></div>
      </div>
    </div>
    <div class="skeleton-line shimmer" style="width:90%;height:12px;margin-top:16px"></div>
    <div class="skeleton-line shimmer" style="width:75%;height:12px;margin-top:8px"></div>
  </div>
  <div class="skeleton-card" style="margin-top:10px">
    <div style="display:flex;gap:12px;align-items:center">
      <div class="skeleton-circle shimmer"></div>
      <div style="flex:1">
        <div class="skeleton-line shimmer" style="width:25%;height:14px;margin-bottom:8px"></div>
        <div class="skeleton-line shimmer" style="width:45%;height:10px"></div>
      </div>
    </div>
    <div class="skeleton-line shimmer" style="width:85%;height:12px;margin-top:16px"></div>
    <div class="skeleton-line shimmer" style="width:70%;height:12px;margin-top:8px"></div>
  </div>
</div>
"""

# ── Trigger ranking ───────────────────────────────────────────────────────────

current_params = {
    "mode": ranking_mode,
    "k": top_k,
    "sal_min": salary_min,
    "sal_max": salary_max,
    "jd": _jd_override,
    "w_sal": st.session_state.weight_salary_fit,
    "w_off": st.session_state.weight_offer_acceptance,
    "w_id": st.session_state.weight_identity_verification,
    "w_comp": st.session_state.weight_profile_completeness,
    "use_sample": use_sample,
    "uploaded_hash": uploaded.size if uploaded else None
}

run_triggered = False
if st.session_state.get("last_params") != current_params:
    st.session_state.last_params = current_params
    run_triggered = True

if run_triggered or st.session_state.ranking_result is None:
    candidates = load_candidates_from_input()
    if candidates:
        filtered = []
        out_of_budget_count = 0
        for c in candidates:
            sal_range = c.get("expected_salary_range_inr_lpa", {})
            if sal_range and float(sal_range.get("min", 0)) > salary_max:
                out_of_budget_count += 1
            else:
                filtered.append(c)
        st.session_state.out_of_budget_count = out_of_budget_count
        candidates = filtered

    if candidates:
        loader = st.empty()
        loader.markdown(SKELETON_HTML, unsafe_allow_html=True)
        st.session_state.ranking_result = run_ranker(candidates, ranking_mode, top_k)
        st.session_state.last_mode = ranking_mode
        st.session_state.compare_result = None
        st.session_state.chat_history = []
        loader.empty()

if compare_modes:
    candidates = load_candidates_from_input()
    if candidates:
        loader = st.empty()
        loader.markdown(SKELETON_HTML, unsafe_allow_html=True)
        import time
        time.sleep(0.8)
        rule_res = run_ranker(candidates, "rule", top_k)
        hybrid_res = run_ranker(candidates, "hybrid", top_k)
        st.session_state.compare_result = (rule_res, hybrid_res)
        st.session_state.ranking_result = rule_res
        st.session_state.last_mode = "compare"
        loader.empty()

result = st.session_state.ranking_result

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="rr-header">
  <div class="rr-title">True Fit</div>
  <div class="rr-sub">Helping recruiters identify the best-fit candidates</div>
</div>
""", unsafe_allow_html=True)

# ── Global color coding helper ────────────────────────────────────────────────

def get_color_indicator(score: float) -> str:
    """Return status emoji and CSS colored pill based on score."""
    if score >= 0.8:
        return '<span class="status-pill status-excellent">🟢 Excellent</span>'
    elif score >= 0.4:
        return '<span class="status-pill status-warning">🟡 Warning</span>'
    else:
        return '<span class="status-pill status-poor">🔴 Poor</span>'

# ── Dimensions Colors ─────────────────────────────────────────────────────────

DIM_COLORS = {
    "Technical Fit":      "#3ecf8e",
    "Experience Fit":     "#60a5fa",
    "Role Relevance":     "#a78bfa",
    "Behavioral":         "#fb923c",
    "Location":           "#f472b6",
    "Production Signals": "#34d399",
    "Education":          "#94a3b8",
    "Culture Fit":        "#fbbf24",
    "Communication":      "#e879f9",
    "Domain Experience":  "#60a5fa",
    "Leadership":         "#a78bfa",
    "Learning Ability":   "#34d399",
    "Project Relevance":  "#fb923c",
    "Behavioral Signals": "#f472b6",
    "Salary Fit":         "#4ade80",
    "Join Probability":   "#a78bfa",
    "Identity Verification": "#60a5fa",
    "Profile Completeness":  "#5eead4",
}


def render_dim_bars(factor_scores: dict[str, int]) -> str:
    items = [(k, v) for k, v in factor_scores.items() if k != "Final Score"]
    cells = ""
    for name, score in items:
        color = DIM_COLORS.get(name, "#6b7280")
        cells += f"""
        <div class="dim-cell">
          <div class="dim-label">{name}</div>
          <div class="dim-track">
            <div class="dim-fill" style="width:{score}%;background:{color}"></div>
          </div>
          <div class="dim-val">{score}/100</div>
        </div>"""
    return f'<div class="dim-grid">{cells}</div>'

# [ignoring loop detection]

# ── Imports for visualization ──────────────────────────────────────────────────

from src.visualization import draw_svg_donut, draw_svg_pie

def _bar_chart_html(items: list[dict], label_key: str, count_key: str, color: str, max_val: int | None = None) -> str:
    if not items:
        return "<p style='color:#4b5563;font-size:13px'>No data</p>"
    if max_val is None:
        max_val = max((d[count_key] for d in items), default=1) or 1
    rows = ""
    for item in items:
        pct = item[count_key] / max_val * 100 if max_val else 0
        rows += f"""<div class="bar-chart-row">
          <div class="bar-label">{item[label_key]}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>
          <div class="bar-count">{item[count_key]}</div>
        </div>"""
    return rows

# ── AI Insights panel calculation helper ───────────────────────────────────────

def compute_ai_insights(result) -> list[str]:
    if not result:
        return []
    cards = getattr(result, "cards", [])
    insights = []
    
    # 1. High matches
    high_match = sum(1 for c in cards if c.score >= 0.90)
    if high_match > 0:
        insights.append(f"• {high_match} candidates exceed 90% overall match score.")
    else:
        insights.append("• No candidates exceed 90% overall match score.")
        
    # 2. Missing Python
    missing_python = 0
    for c in cards:
        if hasattr(c, "gaps") and c.gaps:
            if any("python" in g.lower() for g in c.gaps):
                missing_python += 1
    if missing_python > 0:
        insights.append(f"• Python missing in {missing_python} candidate resumes.")
    else:
        insights.append("•  Python is covered by all shortlisted candidates.")
        
    # 3. Budget filters
    filtered_budget = st.session_state.get("out_of_budget_count", 0)
    if filtered_budget > 0:
        insights.append(f"• Salary budget filtered out {filtered_budget} candidates (exceeded max budget).")
    else:
        insights.append("• All candidates are within or near the salary budget.")
        
    # 4. Very high joining probability
    high_join = 0
    for c in cards:
        if hasattr(c, "acceptance_explanation") and c.acceptance_explanation.get("acceptance_rate", 0) >= 0.75:
            high_join += 1
    if high_join > 0:
        insights.append(f"• {high_join} candidates have very high joining probability (high offer acceptance history).")
        
    # 5. Incomplete profiles
    incomplete = 0
    for c in cards:
        if hasattr(c, "completeness_explanation") and c.completeness_explanation.get("completeness_score", 1.0) < 0.7:
            incomplete += 1
    if incomplete > 0:
        insights.append(f"• {incomplete} candidates need profile completion (below 70% completeness).")
        
    return insights

# ── Render Views Based on Sidebar Selection ────────────────────────────────────

if nav_selection == "Dashboard":
    st.markdown("### Recruiter Dashboard")
    
    if result:
        # AI Insights Panel
        insights = compute_ai_insights(result)
        st.markdown(f"""
        <div class="insights-panel">
          <div class="insights-title">AI Insights</div>
          {"".join(f'<div class="insights-item">{item}</div>' for item in insights)}
        </div>
        """, unsafe_allow_html=True)
        
        # Stats strip
        n_candidates = len(json.loads(sample_path.read_text())) if use_sample and sample_path.exists() else "—"
        df = result.dataframe
        avg_score = f"{df['score'].mean():.3f}"
        top_score = f"{df['score'].max():.3f}"
        mode_label = (st.session_state.last_mode or "rule").upper()

        st.markdown(f"""
        <div class="stats-strip">
          <div class="stat-cell">
            <div class="stat-label">Pool</div>
            <div class="stat-val">{n_candidates:,}</div>
            <div class="stat-sub">candidates evaluated</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Shortlisted</div>
            <div class="stat-val">{len(df)}</div>
            <div class="stat-sub">genuine fits surfaced</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Top score</div>
            <div class="stat-val" style="color:#3ecf8e">{top_score}</div>
            <div class="stat-sub">avg {avg_score}</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Mode</div>
            <div class="stat-val" style="font-size:16px">{mode_label}</div>
            <div class="stat-sub">ranking strategy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dashboard Overview layout
        st.markdown("####  Top Match Highlights")
        cards = getattr(result, "cards", None)
        if cards:
            for card in cards:
                render_candidate_card(card)
                with st.expander(f"Quick Breakdown: {card.name}"):
                    st.markdown(render_dim_bars(card.factor_scores), unsafe_allow_html=True)
        
        st.info("💡 Go to 🏆 **Rankings** to see all candidates or filter the shortlist.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem 1rem 2rem;color:#4b5563">
          <div style="font-size:48px;margin-bottom:1rem">🎯</div>
          <div style="font-size:18px;color:#9ca3af;margin-bottom:.5rem">Awaiting Candidates</div>
          <div style="font-size:14px;margin-bottom:2rem">Please upload candidates or enable sample data to generate a shortlist.</div>
        </div>
        """, unsafe_allow_html=True)


elif nav_selection == "Rankings":
    st.markdown("### Candidate Rankings")
    
    if result:
        # Stats strip
        n_candidates = len(json.loads(sample_path.read_text())) if use_sample and sample_path.exists() else "—"
        df = result.dataframe
        avg_score = f"{df['score'].mean()*100:.1f}%"
        top_score = f"{df['score'].max()*100:.1f}%"
        mode_label = (st.session_state.last_mode or "rule").upper()

        st.markdown(f"""
        <div class="stats-strip">
          <div class="stat-cell">
            <div class="stat-label">Pool</div>
            <div class="stat-val">{n_candidates:,}</div>
            <div class="stat-sub">candidates evaluated</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Shortlisted</div>
            <div class="stat-val">{len(df)}</div>
            <div class="stat-sub">genuine fits surfaced</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Top score</div>
            <div class="stat-val" style="color:#3ecf8e">{top_score}</div>
            <div class="stat-sub">avg {avg_score}</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Mode</div>
            <div class="stat-val" style="font-size:16px">{mode_label}</div>
            <div class="stat-sub">ranking strategy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Export Shortlist ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        import io
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button(
            label="⬇ Download Recommended Candidates (XLSX)",
            data=buffer,
            file_name="recommended_candidates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            use_container_width=True
        )

        # ── Compare results table ─────────────────────────────────────────────
        if st.session_state.compare_result:
            rule_res, hybrid_res = st.session_state.compare_result
            rule_ids = list(rule_res.dataframe["candidate_id"])
            hybrid_ids = list(hybrid_res.dataframe["candidate_id"])
            top10_rule = set(rule_ids[:10])
            top10_hybrid = set(hybrid_ids[:10])
            overlap10 = top10_rule & top10_hybrid
            overlap_all = set(rule_ids) & set(hybrid_ids)

            with st.expander(
                f"⚖️ Mode comparison: {len(overlap10)}/10 top-10 overlap · {len(overlap_all)}/{len(rule_ids)} total overlap",
                expanded=True,
            ):
                rows = []
                for i in range(min(10, len(rule_ids), len(hybrid_ids))):
                    r_id = rule_ids[i]
                    h_id = hybrid_ids[i]
                    match = "✓" if r_id == h_id else "✗"
                    r_name = next((e.candidate_name for e in rule_res.explanations if e.candidate_id == r_id), r_id)
                    h_name = next((e.candidate_name for e in hybrid_res.explanations if e.candidate_id == h_id), h_id)
                    rows.append((i + 1, r_name, h_name, match))

                tbl = "<table class='compare-table'><tr><th>#</th><th>Rule mode</th><th>Hybrid mode</th><th>Match</th></tr>"
                for rank, rn, hn, match in rows:
                    cls = "match" if match == "✓" else "no-match"
                    tbl += f"<tr><td>{rank}</td><td>{rn}</td><td>{hn}</td><td class='{cls}'>{match}</td></tr>"
                tbl += "</table>"
                st.markdown(tbl, unsafe_allow_html=True)

                if len(overlap10) < 5:
                    st.warning(
                        "Low top-10 agreement between modes. Rule mode is better calibrated to the JD's "
                        "stated requirements. Hybrid mode adds semantic similarity which may surface "
                        "different (not necessarily better) candidates. **Submit rule mode output.**"
                    )

        # ── Candidate cards ───────────────────────────────────────────────────
        cards = getattr(result, "cards", None)
        if cards:
            for card in cards:
                render_candidate_card(card)
                with st.expander("Score breakdown"):
                    st.markdown(render_dim_bars(card.factor_scores), unsafe_allow_html=True)
                    final = card.factor_scores.get("Final Score", 0)
                    st.markdown(f"<div style='text-align:right;color:#3ecf8e;font-size:18px;font-weight:700;margin-top:8px'>Final: {final}/100</div>", unsafe_allow_html=True)
                    if card.strengths:
                        st.markdown("**✓ Strengths**")
                        for s in card.strengths:
                            st.markdown(f"- {s}")
                    if card.gaps:
                        st.markdown("**⚠ Gaps / Concerns**")
                        for g in card.gaps:
                            st.markdown(f"- {g}")
        else:
            for exp in result.explanations:
                with st.expander(f"#{exp.rank} {exp.candidate_name} — {exp.factor_scores.get('Final Score', 0)}/100"):
                    st.markdown(exp.to_markdown())

        # ── Export row ───────────────────────────────────────────────────────
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "📥 Download submission CSV",
                result.dataframe.to_csv(index=False).encode("utf-8"),
                "ranked_candidates.csv",
                "text/csv",
                use_container_width=True,
            )
        with c2:
            if hasattr(result, "save_html") and cards:
                from src.card_renderer import render_shortlist_html
                html_bytes = render_shortlist_html(cards).encode("utf-8")
                st.download_button(
                    "📥 Download HTML shortlist",
                    html_bytes,
                    "shortlist.html",
                    "text/html",
                    use_container_width=True,
                )
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4b5563">
          <div style="font-size:48px;margin-bottom:1rem">🎯</div>
          <div style="font-size:18px;color:#9ca3af;margin-bottom:.5rem">Ready to rank</div>
          <div style="font-size:14px">Click <strong style="color:#f2f3f5">Run Ranking</strong> in the sidebar to generate a shortlist.</div>
        </div>
        """, unsafe_allow_html=True)

elif nav_selection == "Candidates":
    st.markdown("### Candidates Deep-Dive")
    if result:
        cards = getattr(result, "cards", None)
        if cards:
            c_names = [f"#{c.rank} {c.name}" for c in cards]
            sel_name = st.selectbox("Inspect Candidate Profile", c_names)
            card = cards[c_names.index(sel_name)]
            
            st.markdown(f"#### {card.name}")
            st.markdown(f"**Headline**: {card.title} · {card.years:.0f}yr exp · {card.location}")
            
            st.markdown("##### 🟢/🔴 Status Color Coding & Indicators")
            
            sal_score = card.factor_scores.get("Salary Fit", 70) / 100.0
            ver_score = card.factor_scores.get("Identity Verification", 60) / 100.0
            acc_score = card.factor_scores.get("Join Probability", 50) / 100.0
            cmp_score = card.factor_scores.get("Profile Completeness", 80) / 100.0
            
            st.markdown(f"""
            <table class="compare-table">
              <tr><th>Dimension</th><th>Score</th><th>Indicator</th></tr>
              <tr><td>Salary Budget Compatibility</td><td>{sal_score:.0%}</td><td>{get_color_indicator(sal_score)}</td></tr>
              <tr><td>Identity Verification channels</td><td>{ver_score:.0%}</td><td>{get_color_indicator(ver_score)}</td></tr>
              <tr><td>Offer Acceptance history</td><td>{acc_score:.0%}</td><td>{get_color_indicator(acc_score)}</td></tr>
              <tr><td>Profile Completeness index</td><td>{cmp_score:.0%}</td><td>{get_color_indicator(cmp_score)}</td></tr>
            </table>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Details & Summary")
            st.markdown(f"{card.enhanced_detail}")
            
            st.markdown("##### Full Score Breakdown")
            st.markdown(render_dim_bars(card.factor_scores), unsafe_allow_html=True)
            
            if card.strengths:
                st.markdown("**Strengths**")
                for s in card.strengths:
                    st.markdown(f"- {s}")
            if card.gaps:
                st.markdown("**Gaps**")
                for g in card.gaps:
                    st.markdown(f"- {g}")
        else:
            st.info("No detailed candidate cards found.")
    else:
        st.info("Run ranking first to search and inspect candidates.")
elif nav_selection == "Analytics":
    st.markdown("### Analytics Dashboard")
    
    if result and hasattr(result, "analytics") and result.analytics:
        a = result.analytics

        # Metrics strip
        st.markdown(f"""
        <div class="stats-strip">
          <div class="stat-cell">
            <div class="stat-label">Salary Fit</div>
            <div class="stat-val" style="color:#4ade80">{a.salary_in_budget_count}</div>
            <div class="stat-sub">within budget</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Verified</div>
            <div class="stat-val" style="color:#60a5fa">{a.fully_verified_count}</div>
            <div class="stat-sub">fully verified</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">High Accept</div>
            <div class="stat-val" style="color:#a78bfa">{a.high_acceptance_count}</div>
            <div class="stat-sub">≥70% rate</div>
          </div>
          <div class="stat-cell">
            <div class="stat-label">Complete</div>
            <div class="stat-val" style="color:#5eead4">{a.completeness_above_85}</div>
            <div class="stat-sub">≥85% profile</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Salary distribution
            st.markdown(f"""<div class="analytics-card">
              <div class="analytics-card-title"> Salary Distribution Histogram</div>
              <div class="analytics-metric"><span style="font-family: sans-serif; margin-right: 2px;">₹</span>{a.salary_avg:.1f}L</div>
              <div class="analytics-sub">avg (<span style="font-family: sans-serif;">₹</span>{a.salary_min:.0f}L – <span style="font-family: sans-serif;">₹</span>{a.salary_max:.0f}L range, {a.salary_candidates_count} with data)</div>
              <div style="margin-top:12px">
                {_bar_chart_html(a.salary_distribution, "bucket", "count", "#4ade80")}
              </div>
            </div>""", unsafe_allow_html=True)

            # Verification Pie Chart
            email_count = sum(1 for c in result.cards if hasattr(c, "verification_explanation") and c.verification_explanation.get("verified_email"))
            phone_count = sum(1 for c in result.cards if hasattr(c, "verification_explanation") and c.verification_explanation.get("verified_phone"))
            li_count = sum(1 for c in result.cards if hasattr(c, "verification_explanation") and c.verification_explanation.get("linkedin_connected"))
            gh_count = sum(1 for c in result.cards if hasattr(c, "verification_explanation") and c.verification_explanation.get("github_active"))
            
            slices_ver = [
                ("Verified Email", email_count or 1, "#4ade80"),
                ("Verified Phone", phone_count or 1, "#60a5fa"),
                ("LinkedIn Linked", li_count or 1, "#a78bfa"),
                ("Active GitHub", gh_count or 1, "#fb923c"),
            ]
            st.markdown(f"""<div class="analytics-card" style="margin-top:12px">
              <div class="analytics-card-title"> Verification Channel Pie Chart</div>
              <div>{draw_svg_pie(slices_ver)}</div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            # Score distribution histogram
            scores = [round(c.score * 100) for c in result.cards]
            score_buckets = [
                ("90-100", sum(1 for s in scores if s >= 90)),
                ("80-89", sum(1 for s in scores if 80 <= s < 90)),
                ("70-79", sum(1 for s in scores if 70 <= s < 80)),
                ("60-69", sum(1 for s in scores if 60 <= s < 70)),
                ("Below 60", sum(1 for s in scores if s < 60)),
            ]
            score_items = [{"bucket": b, "count": c} for b, c in score_buckets]
            st.markdown(f"""<div class="analytics-card">
              <div class="analytics-card-title"> Candidate Match Score Distribution</div>
              <div style="margin-top:12px">
                {_bar_chart_html(score_items, "bucket", "count", "#3ecf8e")}
              </div>
            </div>""", unsafe_allow_html=True)

            # Profile Completeness Donut Chart
            c85_plus = a.completeness_above_85
            c50_85 = max(0, a.shortlisted - a.completeness_above_85 - a.completeness_below_50)
            c50_minus = a.completeness_below_50
            slices_comp = [
                ("High Completeness (≥85%)", c85_plus or 1, "#5eead4"),
                ("Moderate (50-85%)", c50_85 or 1, "#fb923c"),
                ("Incomplete (<50%)", c50_minus or 1, "#f87171"),
            ]
            st.markdown(f"""<div class="analytics-card" style="margin-top:12px">
              <div class="analytics-card-title">📋 Profile Completeness Donut Chart</div>
              <div>{draw_svg_donut(slices_comp)}</div>
            </div>""", unsafe_allow_html=True)

        # Top missing skills
        if a.top_missing_skills:
            missing_items = [{"skill": s, "count": c} for s, c in a.top_missing_skills]
            st.markdown(f"""<div class="analytics-card" style="margin-top:12px">
              <div class="analytics-card-title">⚠️ Top Missing Skills Across Shortlist</div>
              <div style="margin-top:8px">{_bar_chart_html(missing_items, "skill", "count", "#f87171")}</div>
            </div>""", unsafe_allow_html=True)

        # Badge distribution
        if a.badge_counts:
            from src.badge_system import BADGE_TEMPLATES
            badge_items = []
            for key, count in sorted(a.badge_counts.items(), key=lambda x: -x[1]):
                label = BADGE_TEMPLATES.get(key, {}).get("label", key)
                badge_items.append({"skill": f"{BADGE_TEMPLATES.get(key, {}).get('icon', '🏷')} {label}", "count": count})
            st.markdown(f"""<div class="analytics-card" style="margin-top:12px">
              <div class="analytics-card-title"> Badge Distribution</div>
              <div style="margin-top:8px">{_bar_chart_html(badge_items, "skill", "count", "#fbbf24")}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4b5563">
          <div style="font-size:48px;margin-bottom:1rem"></div>
          <div style="font-size:18px;color:#9ca3af;margin-bottom:.5rem">Analytics unavailable</div>
          <div style="font-size:14px">Run ranking first to see hiring insights.</div>
        </div>
        """, unsafe_allow_html=True)


elif nav_selection == "Job Profile":
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
      <div style="font-size:20px;font-weight:700;color:#f2f3f5">{role.role_title}</div>
      <div style="font-size:13px;color:#6b7280;margin-top:3px">
        {role.seniority} · {role.industry} · {role.min_years:.0f}–{role.max_years:.0f} years (ideal {role.ideal_years:.0f}yr)
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    def chip(text: str, cls: str = "") -> str:
        return f'<span class="jd-chip {cls}">{text}</span>'

    with col_l:
        st.markdown('<div class="jd-section">', unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title"> Must-have skills</div>', unsafe_allow_html=True)
        st.markdown("".join(chip(s, "must") for s in role.required_skills), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="jd-section">', unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title"> Inferred traits (beyond keywords)</div>', unsafe_allow_html=True)
        st.markdown("".join(chip(t, "trait") for t in role.inferred_traits), unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title" style="margin-top:12px">🏗 Responsibilities</div>', unsafe_allow_html=True)
        for r_item in role.responsibilities:
            st.markdown(f"<div style='font-size:13px;color:#9ca3af;margin:3px 0'>• {r_item}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="jd-section">', unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title"> Nice-to-have</div>', unsafe_allow_html=True)
        st.markdown("".join(chip(s, "nice") for s in role.nice_to_have_skills), unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title" style="margin-top:12px">🤝 Culture fit</div>', unsafe_allow_html=True)
        st.markdown("".join(chip(c, "trait") for c in role.culture_traits), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="jd-section">', unsafe_allow_html=True)
        st.markdown('<div class="jd-sec-title">🚫 Disqualifiers</div>', unsafe_allow_html=True)
        st.markdown("".join(chip(d, "disq") for d in role.disqualifiers), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Embedding text used for semantic matching"):
        st.code(role.embedding_text, language=None)


elif nav_selection == "AI Assistant":
    st.markdown("### AI Assistant")

    def _claude_answer(prompt: str) -> str:
        """Try to call Claude Sonnet 4.6; return None if unavailable."""
        try:
            import urllib.request, json as _json
            context_parts = []
            if result:
                context_parts.append(f"ROLE: {role.role_title} — {role.seniority}\n")
                context_parts.append(f"REQUIRED SKILLS: {', '.join(role.required_skills)}\n")
                context_parts.append(f"DISQUALIFIERS: {', '.join(role.disqualifiers)}\n\n")
                context_parts.append("TOP 10 RANKED CANDIDATES:\n")
                for exp in (result.explanations or [])[:10]:
                    context_parts.append(
                        f"#{exp.rank} {exp.candidate_name} ({exp.candidate_id}): "
                        f"score={exp.factor_scores.get('Final Score', 0)}/100. "
                        f"Strengths: {'; '.join(exp.strengths[:3])}. "
                        f"Gaps: {'; '.join(exp.gaps[:2]) if exp.gaps else 'none'}.\n"
                    )
            system = (
                "You are an expert recruiter AI assistant embedded in the Redrob candidate ranking dashboard. "
                "Answer the recruiter's question about the shortlisted candidates concisely and insightfully. "
                "Use the provided context. Don't mention you're an AI or repeat the prompt.\n\n"
                + "".join(context_parts)
            )
            payload = _json.dumps({
                "model": "claude-sonnet-4-6",
                "max_tokens": 600,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())
            return data["content"][0]["text"]
        except Exception:
            return None

    def get_answer(prompt: str) -> str:
        reply = _claude_answer(prompt)
        if reply:
            return reply
        if result:
            return answer_recruiter_query(prompt, result.explanations, role.embedding_text)
        return "Run ranking first, then ask me about the shortlist."

    if not result:
        st.info("Run ranking first — then ask questions about the shortlist here.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if result and not st.session_state.chat_history:
        st.markdown("<div style='font-size:12px;color:#4b5563;margin-bottom:8px'>Suggested questions:</div>", unsafe_allow_html=True)
        suggestions = [
            "Why is the top candidate ranked #1?",
            "Who should I interview first and why?",
            "What are the main gaps in the shortlist?",
            "Compare the top 3 candidates",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    with st.spinner("Thinking…"):
                        reply = get_answer(suggestion)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

    if prompt := st.chat_input("Ask about the shortlist…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = get_answer(prompt)
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat", key="clear_chat_button"):
            st.session_state.chat_history = []
            st.rerun()


elif nav_selection == "Settings":
    st.markdown("### Formula Configurator Settings")
    st.markdown("Configure the weights and thresholds used in the final AI ranking formula. These values are applied on the next ranking run.")
    
    st.markdown("#### Weight Sliders")
    w_sal = st.slider("Salary Compatibility Weight", 0.0, 0.5, st.session_state.weight_salary_fit, 0.01)
    w_acc = st.slider("Offer Acceptance Rate (Join Probability) Weight", 0.0, 0.5, st.session_state.weight_offer_acceptance, 0.01)
    w_ver = st.slider("Identity Verification Weight", 0.0, 0.5, st.session_state.weight_identity_verification, 0.01)
    w_cmp = st.slider("Profile Completeness Weight", 0.0, 0.5, st.session_state.weight_profile_completeness, 0.01)
    
    st.markdown(f"**Total Enhancements Weight Contribution**: {(w_sal + w_acc + w_ver + w_cmp):.2f}")
    
    if st.button("Save & Apply Config"):
        st.session_state.weight_salary_fit = w_sal
        st.session_state.weight_offer_acceptance = w_acc
        st.session_state.weight_identity_verification = w_ver
        st.session_state.weight_profile_completeness = w_cmp
        st.success("Configuration saved! Run ranking again in the sidebar to recalculate scores with these weights.")
        
    if st.button("Reset to Default Weights"):
        st.session_state.weight_salary_fit = 0.08
        st.session_state.weight_offer_acceptance = 0.06
        st.session_state.weight_identity_verification = 0.04
        st.session_state.weight_profile_completeness = 0.04
        st.success("Weights reset to defaults! Run ranking again to apply.")

