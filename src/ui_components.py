"""Streamlit components for rich candidate cards."""

from __future__ import annotations

import streamlit as st

from .candidate_card import CandidateCard
from .card_renderer import CARD_CSS, render_card_html


def render_candidate_card(card: CandidateCard) -> None:
    """Render a single candidate card matching the reference shortlist UI."""
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.markdown(render_card_html(card), unsafe_allow_html=True)

    with st.expander("Show score breakdown"):
        cols = st.columns(2)
        factors = [(k, v) for k, v in card.factor_scores.items() if k != "Final Score"]
        mid = (len(factors) + 1) // 2
        with cols[0]:
            for name, score in factors[:mid]:
                st.progress(score / 100, text=f"{name}: {score}")
        with cols[1]:
            for name, score in factors[mid:]:
                st.progress(score / 100, text=f"{name}: {score}")
        st.metric("Final Score", card.factor_scores.get("Final Score", 0))

        if card.strengths:
            st.markdown("**Strengths**")
            for s in card.strengths:
                st.markdown(f"- ✓ {s}")
        if card.gaps:
            st.markdown("**Missing / Concerns**")
            for g in card.gaps:
                st.markdown(f"- • {g}")
