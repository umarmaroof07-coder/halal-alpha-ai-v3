"""
Live Recommendations — one button, factor score breakdown, clean entry text.
"""

from __future__ import annotations
import streamlit as st


# ---------------------------------------------------------------------------
# Live price fragment — isolated; only reruns when button clicked
# ---------------------------------------------------------------------------

@st.fragment
def _price_fragment(buy_now: list, univ_map: dict, budget: float) -> None:
    from dashboard.state import get_live_prices_cached, clear_live_prices_cache
    from portfolio.entry_price import compute_entry_analysis
    import logging

    col_btn, col_ts = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Refresh Prices"):
            clear_live_prices_cache()
            st.rerun(scope="fragment")

    buy_tickers = tuple(sorted(r.ticker for r in buy_now))
    live_data   = get_live_prices_cached(buy_tickers)

    ts_list = [v.get("timestamp", "") for v in live_data.values() if v.get("timestamp")]
    with col_ts:
        st.caption(f"Last price refresh: {ts_list[0][:19].replace('T',' ') if ts_list else '—'} UTC")

    # Allocation cards
    cols = st.columns(len(buy_now))
    for idx, r in enumerate(buy_now):
        w     = r.conviction_weight or 0
        alloc = budget * w
        t     = r.ticker
        ld    = live_data.get(t, {})
        lp    = ld.get("price") or float(univ_map.get(t, {}).get("price") or r.price or 0)
        chg   = ld.get("change_pct")
        sh    = alloc / lp if lp > 0 else 0
        with cols[idx]:
            st.metric(f"#{r.rank} {t}", f"${lp:,.2f}" if lp else "—",
                      delta=f"{chg:+.2f}%" if chg is not None else None)
            st.caption(f"${alloc:,.0f} · {sh:.2f} sh")

    # Entry analysis
    entry_map: dict = {}
    try:
        for r in buy_now:
            t  = r.ticker
            s  = univ_map.get(t, {})
            ld = live_data.get(t, {})
            px = ld.get("price") or float(s.get("price") or r.price or 0)
            if px > 0:
                entry_map[t] = compute_entry_analysis(
                    ticker           = t,
                    current_price    = px,
                    valuation_score  = float(s.get("valuation", 50)),
                    composite_score  = float(s.get("composite", 50)),
                    model_confidence = float(s.get("model_confidence", 65)),
                    risk_score       = float(s.get("risk_adjustment", 50)),
                    momentum_score   = float(s.get("momentum", 50)),
                )
    except Exception as exc:
        logging.getLogger(__name__).warning("Entry analysis: %s", exc)

    # Price + entry table
    rows = []
    for r in buy_now:
        t  = r.ticker
        s  = univ_map.get(t, {})
        ld = live_data.get(t, {})
        ea = entry_map.get(t)
        lp = ld.get("price") or float(s.get("price") or r.price or 0)
        rows.append({
            "Ticker":       t,
            "Live Price":   f"${lp:,.2f}" if lp else "—",
            "Chg%":         f"{ld.get('change_pct', 0):+.2f}%" if ld else "—",
            "Buy Below":    f"${ea.buy_limit:.2f}"        if ea and ea.buy_limit        else "—",
            "Strong Buy":   f"${ea.strong_buy_limit:.2f}" if ea and ea.strong_buy_limit else "—",
            "Entry Score":  f"{ea.entry_score:.0f}/100"   if ea                         else "—",
            "Rating":       ea.entry_rating                if ea                         else "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Entry callouts — plain text, no HTML, no markdown dollar-sign issues
    for r in buy_now:
        ea = entry_map.get(r.ticker)
        if ea and ea.explanation:
            icon = {"Strong Buy": "🟢", "Buy": "🔵", "Watch": "🟡", "Wait": "🔴"}.get(ea.entry_rating, "⚪")
            st.text(f"{icon} {r.ticker} ({ea.entry_rating}): {ea.explanation}")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Recommendations")

    from dashboard.state import (
        get_recommendations, get_universe,
        clear_recommendations_cache, run_refresh_data,
    )
    from dashboard.jobs import poll_status

    # ── ONE button ────────────────────────────────────────────────────────────
    status = poll_status("refresh_data")
    s      = status.get("status", "idle")

    if s == "running":
        started = (status.get("started_at") or "")[:19].replace("T", " ")
        st.button(f"⏳ Refreshing… (started {started} UTC)", disabled=True, type="primary")
        if st.button("Check if done"):
            st.rerun()
    else:
        if st.button("🔄 Refresh All  (scores · AI research · Top 5)", type="primary",
                     help="Fetches new data, recomputes all 8 factor scores, runs AI on Top 20, rebuilds Top 5."):
            run_refresh_data()
            st.rerun()

        if s == "completed":
            if st.session_state.get("_last_refresh") != status.get("finished_at"):
                clear_recommendations_cache()
                st.session_state["_last_refresh"] = status.get("finished_at")
            finished = (status.get("finished_at") or "")[:19].replace("T", " ")
            st.caption(f"Last refresh completed: {finished} UTC")
            with st.expander("Refresh output"):
                st.code("\n".join(status.get("last_lines") or []))
        elif s == "failed":
            st.error("Last refresh failed.")
            with st.expander("Error output", expanded=True):
                st.code("\n".join(status.get("last_lines") or []))

    st.caption(
        "ℹ Scores are frozen between refreshes. "
        "**Re-evaluate positions weekly, not daily** — single-day rank changes are noise."
    )

    # ── Budget ────────────────────────────────────────────────────────────────
    col_b, col_i = st.columns([1, 3])
    with col_b:
        budget = st.number_input("💰 Budget ($)", min_value=100, max_value=10_000_000,
                                 value=int(st.session_state.get("portfolio_budget", 1_500)), step=500)
        st.session_state["portfolio_budget"] = budget
    with col_i:
        st.info(f"Dollar amounts shown for **${budget:,.0f}**.")

    st.divider()

    # ── Load ──────────────────────────────────────────────────────────────────
    recs     = get_recommendations()
    universe = get_universe()
    univ_map = {s["ticker"]: s for s in universe}

    buy_now   = [r for r in recs if r.action == "BUY NOW"]
    watchlist = [r for r in recs if r.action == "WATCHLIST"]
    avoid     = [r for r in recs if r.action == "AVOID"]

    # ── TOP 5 ─────────────────────────────────────────────────────────────────
    st.subheader("📈 Top 5 — BUY NOW")

    if not buy_now:
        st.info("No recommendations yet. Click **Refresh All** to generate Top 5.")
        return

    # Factor score breakdown — one row per stock, one column per factor
    factor_keys = [
        ("Composite",    "composite"),
        ("Quality",      "quality"),
        ("Momentum",     "momentum"),
        ("Revisions",    "earnings_revisions"),
        ("Valuation",    "valuation"),
        ("Earn.Quality", "earnings_quality"),
        ("Moat",         "moat"),
        ("Cap.Alloc",    "capital_allocation"),
        ("Risk",         "risk_adjustment"),
        ("AI",           "ai_research"),
    ]

    score_rows = []
    for r in buy_now:
        t     = r.ticker
        s     = univ_map.get(t, {})
        w     = r.conviction_weight or 0
        alloc = budget * w
        px    = float(s.get("price") or r.price or 0)
        sh    = alloc / px if px > 0 else 0
        sbc   = s.get("sbc_data") or {}
        warns = []
        if sbc.get("sbc_fcf_ratio", 0) > 0.20:
            warns.append("SBC")
        if s.get("value_trap_warning"):
            warns.append("⚠ VALUE TRAP")
        rev_g = s.get("revenue_growth")
        if rev_g is not None and rev_g < 0:
            warns.append(f"Rev {rev_g*100:.1f}%")
        warn = " | ".join(warns)

        row = {
            "Rank":        r.rank,
            "Ticker":      t,
            f"$ ({budget:,.0f})": f"${alloc:,.0f}",
            "Wt%":         f"{w*100:.0f}%",
            "Shares":      f"{sh:.2f}",
            "Shariah":     r.shariah_status.upper(),
            "⚠":           warn,
        }
        for label, key in factor_keys:
            val = s.get(key) or (r.composite_score if key == "composite" else 50.0)
            row[label] = round(float(val), 1)

        score_rows.append(row)

    st.dataframe(score_rows, use_container_width=True, hide_index=True)
    st.caption("Q20% · M15% · Rev20% · Val10% · EQ10% · Moat10% · CA10% · Risk5%  |  Scores 0–100, neutral = 50")

    # Risk banners — value trap and negative growth
    for r in buy_now:
        s = univ_map.get(r.ticker, {})
        if s.get("value_trap_warning"):
            st.error(f"⚠ {r.ticker} — {s['value_trap_warning']}")
        elif s.get("revenue_growth") is not None and s["revenue_growth"] < 0:
            st.warning(f"⚠ {r.ticker} — Negative revenue growth ({s['revenue_growth']*100:.1f}%). Verify thesis before buying.")

    # ── Business Health Cards ──────────────────────────────────────────────
    st.divider()
    st.markdown("**Business Reality Audit**")
    bh_cols = st.columns(len(buy_now))
    _BH_GRADE_COLOR = {
        "Exceptional": "🟢",
        "Strong":      "🟢",
        "Good":        "🔵",
        "Watch":       "🟡",
        "High Risk":   "🔴",
    }
    for idx, r in enumerate(buy_now):
        s  = univ_map.get(r.ticker, {})
        bh = s.get("business_health") or {}
        bh_score = bh.get("score")
        bh_grade = bh.get("grade", "—")
        icon     = _BH_GRADE_COLOR.get(bh_grade, "⚪")
        with bh_cols[idx]:
            st.metric(
                f"{r.ticker} Health",
                f"{bh_score:.0f}/100" if bh_score is not None else "—",
                delta=f"{icon} {bh_grade}",
            )
            # Risk badges
            if bh.get("severe_decline_risk"):
                st.error("Structural Decline")
            if bh.get("value_trap_risk"):
                st.error("Value Trap")
            if bh.get("debt_risk"):
                st.warning("Debt Risk")
            if bh.get("dilution_risk"):
                st.warning("Dilution Risk")
            if bh.get("durability_risk"):
                st.caption(f"Durability: {bh.get('durability_score', 50):.0f}/100")

    # ── Per-stock thesis expanders ─────────────────────────────────────────
    for r in buy_now:
        s    = univ_map.get(r.ticker, {})
        bh   = s.get("business_health") or {}
        if not bh:
            continue
        grade = bh.get("grade", "—")
        score = bh.get("score")
        with st.expander(f"{r.ticker} — {grade} ({score:.0f}/100)  |  Business Health Thesis"):
            th_cols = st.columns(5)
            th_cols[0].metric("Growth",         f"{bh.get('growth_score', 50):.0f}")
            th_cols[1].metric("Balance Sheet",  f"{bh.get('balance_sheet_score', 50):.0f}")
            th_cols[2].metric("Cap Alloc",      f"{bh.get('capital_allocation_score', 50):.0f}")
            th_cols[3].metric("Durability",     f"{bh.get('durability_score', 50):.0f}")
            th_cols[4].metric("Mkt Confirm",    f"{bh.get('market_confirmation_score', 50):.0f}")

            strengths = bh.get("top_strengths") or []
            risks_lst = bh.get("top_risks")     or []
            breakers  = bh.get("thesis_breakers") or []

            if strengths:
                st.markdown("**Top Strengths**")
                for item in strengths:
                    st.markdown(f"- ✅ {item}")
            if risks_lst:
                st.markdown("**Top Risks**")
                for item in risks_lst:
                    st.markdown(f"- ⚠ {item}")
            if breakers:
                st.markdown("**Thesis Breakers**")
                for item in breakers:
                    st.markdown(f"- 🔴 {item}")

    st.divider()
    st.markdown("**Live Prices & Entry Limits**")
    _price_fragment(buy_now, univ_map, budget)

    st.divider()

    # ── WATCHLIST ─────────────────────────────────────────────────────────────
    with st.expander(f"👁 Watchlist ({len(watchlist)} stocks)"):
        if watchlist:
            rows = []
            for r in watchlist:
                t   = r.ticker
                s   = univ_map.get(t, {})
                sbc = s.get("sbc_data") or {}
                rows.append({
                    "Ticker":    t,
                    "Score":     round(r.composite_score, 1),
                    "Momentum":  round(float(s.get("momentum", 50)), 1),
                    "Valuation": round(float(s.get("valuation", 50)), 1),
                    "Price":     f"${r.price:.2f}" if r.price else "—",
                    "Shariah":   r.shariah_status.upper(),
                    "⚠":         "SBC" if sbc.get("sbc_fcf_ratio", 0) > 0.20 else "",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander(f"✗ Avoid ({len(avoid)} stocks)"):
        if avoid:
            rows = []
            for r in avoid:
                rows.append({
                    "Ticker":  r.ticker,
                    "Score":   round(r.composite_score, 1),
                    "Shariah": r.shariah_status.upper(),
                    "Reasons": "; ".join(r.rejection_reasons),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
