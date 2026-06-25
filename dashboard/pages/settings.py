"""Settings page — account size, API status, weight display, cache controls."""

from __future__ import annotations

import streamlit as st
from pathlib import Path


def render() -> None:
    st.title("Settings")

    from dashboard.state import clear_recommendations_cache, clear_live_prices_cache

    # ── Account & config ──────────────────────────────────────────────────────
    st.subheader("Account Configuration")
    try:
        from config.settings import ACCOUNT_SIZE
        st.metric("Account Size", f"${ACCOUNT_SIZE:,.0f}")
        st.caption("Edit `config/settings.py` to change ACCOUNT_SIZE.")
    except Exception as e:
        st.warning(f"Could not load config: {e}")

    st.divider()

    # ── Factor weights (display only) ─────────────────────────────────────────
    st.subheader("Current Factor Weights (read-only)")
    try:
        from config.weights import FACTOR_WEIGHTS
        rows = [{"Factor": k.replace("_", " ").title(), "Weight": f"{v*100:.0f}%"}
                for k, v in FACTOR_WEIGHTS.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Weights locked at V5 balanced: Q20% M15% Rev20% Val10% EQ10% Moat10% CA10% Risk5%")
    except Exception as e:
        st.warning(f"Could not load weights: {e}")

    st.divider()

    # ── API status ────────────────────────────────────────────────────────────
    st.subheader("API Key Status")
    import os
    keys = {
        "ANTHROPIC_API_KEY":    "Claude AI (AI Research)",
        "FMP_API_KEY":          "Financial Modeling Prep (Fundamentals)",
        "ALPHA_VANTAGE_KEY":    "Alpha Vantage (optional)",
    }
    for env_key, label in keys.items():
        val = os.environ.get(env_key, "")
        if val:
            masked = val[:4] + "…" + val[-4:] if len(val) >= 8 else "***"
            st.success(f"✓ {label}: `{masked}`")
        else:
            st.error(f"✗ {label}: not set")

    st.divider()

    # ── Cache controls ────────────────────────────────────────────────────────
    st.subheader("Cache Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🗑 Clear Recommendations Cache"):
            clear_recommendations_cache()
            st.success("Recommendation cache cleared.")

    with col2:
        if st.button("🗑 Clear Live Price Cache"):
            clear_live_prices_cache()
            st.success("Live price cache cleared.")

    with col3:
        if st.button("🗑 Clear All Session State"):
            st.session_state.clear()
            st.success("All session state cleared.")
            st.rerun()

    st.divider()

    # ── Snapshot info ─────────────────────────────────────────────────────────
    st.subheader("Point-in-Time Snapshots")
    try:
        from data_layer.snapshot import get_snapshot_summary
        snap = get_snapshot_summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Snapshots", snap["count"])
        c2.metric("First Date", snap["first_date"] or "—")
        c3.metric("Last Date",  snap["last_date"]  or "—")
        st.caption(
            f"Snapshots in data/history/.  "
            f"Need ≥4 for meaningful walk-forward. "
            f"Current: {snap['count']}."
        )
        if snap["count"] > 0:
            with st.expander("All snapshot dates"):
                for d in snap.get("dates", []):
                    st.text(d)
    except Exception as e:
        st.warning(f"Could not load snapshot info: {e}")

    st.divider()

    # ── Data files ────────────────────────────────────────────────────────────
    st.subheader("Data File Status")
    files = {
        "data/cache/scored_universe.json":    "Scored universe (--refresh-data)",
        "data/cache/final_picks_report.json": "Final picks report (--final-picks)",
        "data/cache/walk_forward_results.json": "Walk-forward results (--walk-forward)",
        "data/cache/factor_monitor.json":     "Factor monitor (--factor-monitor)",
        "data/cache/universe_screened.csv":   "Screened universe (--refresh-universe)",
    }
    for path, label in files.items():
        p = Path(path)
        if p.exists():
            import datetime
            mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"✅ **{label}** — last updated {mtime}")
        else:
            st.markdown(f"❌ **{label}** — not found")
