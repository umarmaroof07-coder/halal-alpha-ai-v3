"""
Halal Alpha AI V6 — Streamlit Dashboard

Run with: streamlit run app.py

Navigation: st.sidebar.radio (no fake tabs, no CSS sidebar hiding).
Safety rule: ALL recommendations come from safe_recommendations() only.
Dashboard Top 5 == CLI Top 5 by construction.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Halal Alpha AI V6",
    page_icon="☪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark mode CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card {
        background: #1c2333; border-radius: 8px; padding: 12px 16px;
        border: 1px solid #2d3748; margin-bottom: 8px;
    }
    .badge-pass  { background:#16a34a;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px; }
    .badge-warn  { background:#ca8a04;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px; }
    .badge-block { background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px; }
    .badge-info  { background:#2563eb;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "Overview",
    "Live Recommendations",
    "Final Picks",
    "Stock Screener",
    "Stock Detail",
    "Portfolio Tracker",
    "Data Quality",
    "AI Research",
    "Stress Tests",
    "Factor Monitor",
    "Settings",
]

with st.sidebar:
    st.title("☪ Halal Alpha AI")
    st.caption("V6 Institutional Engine")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Safety: picks only from safe_recommendations()")

# ── Dispatch ──────────────────────────────────────────────────────────────────
if page == "Overview":
    from dashboard.pages.overview import render
    render()

elif page == "Live Recommendations":
    from dashboard.pages.recommendations import render
    render()

elif page == "Final Picks":
    from dashboard.pages.final_picks import render
    render()

elif page == "Stock Screener":
    from dashboard.pages.screener import render
    render()

elif page == "Stock Detail":
    from dashboard.pages.stock_detail import render
    render()

elif page == "Portfolio Tracker":
    from dashboard.pages.portfolio_tracker import render
    render()

elif page == "Data Quality":
    from dashboard.pages.data_quality import render
    render()

elif page == "AI Research":
    from dashboard.pages.ai_research import render
    render()

elif page == "Stress Tests":
    from dashboard.pages.stress_tests import render
    render()

elif page == "Factor Monitor":
    from dashboard.pages.factor_monitor import render
    render()

elif page == "Settings":
    from dashboard.pages.settings import render
    render()
