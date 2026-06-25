"""Overview page — live prices, action buttons, Top 5 picks, equity curve."""

from __future__ import annotations

import streamlit as st
from datetime import datetime, timezone
from pathlib import Path


def render() -> None:
    st.title("Halal Alpha AI — Overview")

    st.warning(
        "⚠ SURVIVORSHIP BIAS: Backtest tickers are based on today's universe. "
        "Returns are likely overstated."
    )
    st.warning(
        "⚠ LOOK-AHEAD BIAS: Fundamental data is not point-in-time. "
        "Factor scores may reflect information unavailable at the rebalance date."
    )

    from dashboard.state import (
        get_recommendations, get_backtest_result, get_universe,
        get_live_prices_cached, clear_live_prices_cache, clear_recommendations_cache,
    )
    from config.settings import ACCOUNT_SIZE

    # ── Action buttons ────────────────────────────────────────────────────────
    st.subheader("Actions")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        if st.button("🔄 Refresh Prices", help="Re-fetch live prices (bypasses 60s cache)"):
            clear_live_prices_cache()
            st.rerun()

    with col_b2:
        if st.button("📊 Refresh Fundamentals",
                     help="Re-score fundamentals (runs --refresh-data pipeline)"):
            clear_recommendations_cache()
            st.rerun()

    with col_b3:
        if st.button("🤖 Refresh AI Research",
                     help="Re-score Top 20 with Claude AI (requires ANTHROPIC_API_KEY)"):
            clear_recommendations_cache()
            st.rerun()

    with col_b4:
        if st.button("✅ Generate Final Picks",
                     help="Run 7-gate integrity check and show final Top 5"):
            st.switch_page("dashboard/pages/recommendations.py")

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    backtest = get_backtest_result()
    recs     = get_recommendations()
    buy_now  = [r for r in recs if r.action == "BUY NOW"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Account Size",     f"${ACCOUNT_SIZE:,.0f}")
    c2.metric("Active Positions", str(len(buy_now)))
    c3.metric("Today",            datetime.now().strftime("%Y-%m-%d"))
    if backtest:
        c4.metric("Backtest CAGR", f"{backtest.get('cagr', 0):.1f}%")
    else:
        c4.metric("Backtest CAGR", "Run --backtest first")

    st.divider()

    # ── Live price table ──────────────────────────────────────────────────────
    st.subheader("Live Prices — Top Picks")

    universe   = get_universe()
    buy_tickers = [r.ticker for r in buy_now]

    if buy_tickers:
        live_data = get_live_prices_cached(tuple(sorted(buy_tickers)))

        # Staleness check
        max_age_min = 0
        source_set: set[str] = set()
        for ticker, pd in live_data.items():
            try:
                ts   = datetime.fromisoformat(pd.get("timestamp", ""))
                age  = (datetime.now() - ts).total_seconds() / 60
                if age > max_age_min:
                    max_age_min = age
                source_set.add(pd.get("source", "?"))
            except Exception:
                pass

        # Source + staleness badge
        source_str = "/".join(sorted(source_set)) or "unknown"
        badge_color = "green" if max_age_min <= 2 else ("orange" if max_age_min <= 30 else "red")
        freshness   = "FRESH" if max_age_min <= 2 else ("SEMI-LIVE" if max_age_min <= 30 else "STALE")

        col_info, col_badge = st.columns([4, 1])
        with col_info:
            if live_data:
                ts_str = list(live_data.values())[0].get("timestamp", "")
                try:
                    ts_dt = datetime.fromisoformat(ts_str)
                    st.caption(f"Prices last updated: {ts_dt.strftime('%H:%M:%S')}  |  "
                               f"Source: {source_str}  |  Refreshes every 60 s on page interaction")
                except Exception:
                    st.caption(f"Source: {source_str}")
        with col_badge:
            st.markdown(
                f"<span style='background:{badge_color};color:white;"
                f"padding:3px 8px;border-radius:4px;font-size:12px'>"
                f"{freshness}</span>",
                unsafe_allow_html=True,
            )

        if max_age_min > 30:
            st.warning(
                f"⚠ Price data is {max_age_min:.0f} min old. "
                f"Click 'Refresh Prices' to update. BUY NOW labels should not be acted "
                f"on when prices are stale."
            )

        # Build price rows
        univ_map = {s["ticker"]: s for s in universe}
        rows = []
        for r in buy_now:
            t         = r.ticker
            live_pd   = live_data.get(t, {})
            live_price = live_pd.get("price")
            chg_pct    = live_pd.get("change_pct", 0.0)
            stored_price = float(univ_map.get(t, {}).get("price") or 0)

            chg_str = f"{chg_pct:+.2f}%" if live_pd else "—"
            src_str = live_pd.get("source", "—") if live_pd else "stale"
            rows.append({
                "Rank":       f"#{r.rank}",
                "Ticker":     t,
                "Live Price": f"${live_price:.2f}" if live_price else f"${stored_price:.2f} (cached)",
                "Change":     chg_str,
                "Score":      f"{r.composite_score:.1f}",
                "Weight":     f"{(r.conviction_weight or 0)*100:.0f}%",
                "$Amount":    f"${r.dollar_amount or 0:.0f}",
                "Source":     src_str,
                "Shariah":    r.shariah_status.upper(),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Data freshness note
        st.caption(
            "💡 Prices = live/semi-live (60s refresh)  |  "
            "Fundamentals = daily/quarterly refresh  |  "
            "Financial statements = quarterly"
        )
    else:
        st.info("No BUY NOW picks yet. Run: python3 main.py --refresh-data --recommend")

    st.divider()

    # ── Top 5 score breakdown ─────────────────────────────────────────────────
    st.subheader("Top 5 — Factor Score Breakdown")
    if buy_now:
        score_rows = []
        univ_map = {s["ticker"]: s for s in universe}
        for r in buy_now:
            s = univ_map.get(r.ticker, {})
            score_rows.append({
                "Ticker":       r.ticker,
                "Composite":    f"{r.composite_score:.1f}",
                "Quality":      f"{s.get('quality', 50):.1f}",
                "Momentum":     f"{s.get('momentum', 50):.1f}",
                "Revisions":    f"{s.get('earnings_revisions', 50):.1f}",
                "Valuation":    f"{s.get('valuation', 50):.1f}",
                "Earn.Quality": f"{s.get('earnings_quality', 50):.1f}",
                "AI":           f"{s.get('ai_research', 50):.1f}",
                "Shariah":      r.shariah_status.upper(),
            })
        st.dataframe(score_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Weights: Quality 30% | Momentum 25% | Revisions 20% | "
            "Valuation 10% | Earnings Quality 10% | AI Research 5%"
        )

    st.divider()

    # ── Mini equity curve ─────────────────────────────────────────────────────
    charts_dir = Path("data/reports/charts")
    equity_png = charts_dir / "equity_curve.png"
    if equity_png.exists():
        st.subheader("Equity Curve")
        st.image(str(equity_png), use_container_width=True)
    elif backtest:
        st.info("Run --backtest again to regenerate charts.")

    # ── Backtest key stats ────────────────────────────────────────────────────
    if backtest:
        st.subheader("Last Backtest Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CAGR",     f"{backtest.get('cagr', 0):.1f}%")
        col2.metric("Sharpe",   f"{backtest.get('sharpe', 0):.2f}")
        col3.metric("Sortino",  f"{backtest.get('sortino', 0):.2f}")
        col4.metric("Max DD",   f"{backtest.get('max_drawdown', 0):.1f}%")
        col5.metric("Win Rate", f"{backtest.get('win_rate', 0):.1f}%")
