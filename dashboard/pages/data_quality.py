"""Data Quality page — cache health, missing data flags, provider status."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

_CACHE_DB = Path("data/cache/data_cache.db")
_AI_CACHE_DB = Path("data/cache/ai_research_cache.db")


def _db_row_count(db_path: Path, table: str) -> int:
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(db_path)
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def _db_latest(db_path: Path, table: str, col: str = "fetched_at") -> str:
    if not db_path.exists():
        return "—"
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone()
        conn.close()
        return row[0] or "—"
    except Exception:
        return "—"


def render() -> None:
    st.title("Data Quality")

    # ── Provider status ───────────────────────────────────────────────────────
    st.subheader("Provider Configuration")
    from config.settings import FMP_API_KEY, FINNHUB_API_KEY, ANTHROPIC_API_KEY

    col1, col2, col3 = st.columns(3)
    col1.metric("FMP",       "✓ Configured" if FMP_API_KEY else "✗ Missing")
    col2.metric("Finnhub",   "✓ Configured" if FINNHUB_API_KEY else "✗ Missing")
    col3.metric("Anthropic", "✓ Configured" if ANTHROPIC_API_KEY else "✗ Missing")

    st.divider()

    # ── Cache stats ───────────────────────────────────────────────────────────
    st.subheader("Cache Statistics")
    data_rows = _db_row_count(_CACHE_DB, "cache")
    ai_rows   = _db_row_count(_AI_CACHE_DB, "ai_research_cache")
    data_latest = _db_latest(_CACHE_DB, "cache")
    ai_latest   = _db_latest(_AI_CACHE_DB, "ai_research_cache")

    c1, c2 = st.columns(2)
    c1.metric("Data Cache Rows",       str(data_rows))
    c1.caption(f"Last fetch: {data_latest}")
    c2.metric("AI Research Cache Rows", str(ai_rows))
    c2.caption(f"Last fetch: {ai_latest}")

    st.divider()

    # ── Missing data flags ────────────────────────────────────────────────────
    st.subheader("Universe Coverage")
    from dashboard.state import get_universe
    universe = get_universe()
    total = len(universe)
    compliant = sum(1 for s in universe if s.get("shariah_status") == "compliant")
    non_compliant = sum(1 for s in universe if s.get("shariah_status") == "non_compliant")
    unknown = total - compliant - non_compliant
    no_price = sum(1 for s in universe if not s.get("price"))

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Total",         total)
    col_b.metric("Compliant",     compliant)
    col_c.metric("Non-Compliant", non_compliant)
    col_d.metric("Unknown",       unknown)
    col_e.metric("No Price",      no_price)

    if unknown > 0:
        st.warning(f"{unknown} stock(s) have unknown Shariah status and are excluded from recommendations.")

    st.divider()

    # ── Cache management ──────────────────────────────────────────────────────
    st.subheader("Cache Management")
    st.caption("Clearing the cache will force a fresh data fetch on the next run.")
    confirm = st.checkbox("I confirm I want to clear the data cache")
    if st.button("Clear Data Cache", disabled=not confirm):
        if _CACHE_DB.exists():
            try:
                conn = sqlite3.connect(_CACHE_DB)
                conn.execute("DELETE FROM cache")
                conn.commit()
                conn.close()
                st.success("Data cache cleared.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to clear cache: {exc}")
        else:
            st.info("No cache database found.")
