"""Portfolio Tracker page — current holdings vs target, rebalance actions."""

from __future__ import annotations

import io
import csv

import streamlit as st


def render() -> None:
    st.title("Portfolio Tracker")

    from dashboard.state import get_recommendations, get_portfolio
    from portfolio.rebalancer import CurrentPosition, compute_rebalance
    from config.settings import ACCOUNT_SIZE

    portfolio = get_portfolio()
    recs = get_recommendations()
    buy_now = [r for r in recs if r.action == "BUY NOW"]

    # ── Target portfolio ──────────────────────────────────────────────────────
    st.subheader("Target Portfolio")
    if buy_now:
        rows = []
        for r in buy_now:
            rows.append({
                "Rank":     r.rank,
                "Ticker":   r.ticker,
                "Score":    round(r.composite_score, 1),
                "Weight":   f"{(r.conviction_weight or 0)*100:.0f}%",
                "$ Target": f"${r.dollar_amount or 0:.0f}",
                "Price":    f"${r.price:.2f}" if r.price else "—",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.metric("Total Invested", f"${portfolio.total_invested:,.0f}")
        c2.metric("Cash Remaining", f"${portfolio.cash_remaining:,.0f}")
    else:
        st.info("No target positions. Go to Recommendations and click Refresh.")

    st.divider()

    # ── Limit order prices ────────────────────────────────────────────────────
    st.subheader("Limit Order Prices")
    st.warning("Use limit orders only. Fill is not guaranteed. Never use market orders.")

    if buy_now and st.button("💰 Compute Limit Prices"):
        with st.spinner("Fetching live prices…"):
            from portfolio.limit_prices import compute_limit_price
            limit_data: dict = {}
            for r in buy_now:
                limit_data[r.ticker] = compute_limit_price(
                    r.ticker, r.dollar_amount or 0
                ).to_dict()
            st.session_state["pt_limit_results"] = limit_data
        st.rerun()

    pt_limits = st.session_state.get("pt_limit_results", {})
    if pt_limits and buy_now:
        limit_rows = []
        for r in buy_now:
            lp = pt_limits.get(r.ticker, {})
            if not lp or lp.get("error") or lp.get("stale"):
                limit_rows.append({
                    "Ticker":          r.ticker,
                    "Live Price":      "—",
                    "Suggested Limit": "unavailable",
                    "Tier":            "—",
                    "Shares":          "—",
                    "Est. Fill Cost":  "—",
                    "$ Allocation":    f"${r.dollar_amount or 0:,.0f}",
                })
                continue
            limit_rows.append({
                "Ticker":          r.ticker,
                "Live Price":      f"${lp.get('live_price', 0):.2f}",
                "Suggested Limit": f"${lp.get('suggested_limit', 0):.2f}",
                "Tier":            lp.get("suggested_tier", "—"),
                "Cons. Limit":     f"${lp.get('conservative_limit', 0):.2f}",
                "Normal Limit":    f"${lp.get('normal_limit', 0):.2f}",
                "Aggr. Limit":     f"${lp.get('aggressive_limit', 0):.2f}",
                "Shares":          f"{lp.get('shares', 0):.3f}" if lp.get("shares") else "—",
                "Est. Fill Cost":  f"${lp.get('estimated_fill_cost', 0):,.2f}" if lp.get("estimated_fill_cost") else "—",
                "$ Allocation":    f"${r.dollar_amount or 0:,.0f}",
            })
        st.dataframe(limit_rows, use_container_width=True, hide_index=True)

        for r in buy_now:
            lp = pt_limits.get(r.ticker, {})
            if lp.get("warning"):
                st.warning(f"**{r.ticker}:** {lp['warning']}")
            if lp.get("explanation"):
                st.caption(f"{r.ticker}: {lp['explanation']}")

    st.divider()

    # ── Upload current holdings ───────────────────────────────────────────────
    st.subheader("Upload Current Holdings")
    st.caption("CSV format: ticker, shares, price  (one row per position)")

    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    current_positions: list[CurrentPosition] = []

    if uploaded is not None:
        try:
            text = uploaded.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                current_positions.append(CurrentPosition(
                    ticker=row.get("ticker", "").strip().upper(),
                    shares=float(row.get("shares", 0)),
                    price=float(row.get("price", 0)),
                ))
            st.success(f"Loaded {len(current_positions)} current positions.")
        except Exception as exc:
            st.error(f"Failed to parse CSV: {exc}")

    if current_positions or buy_now:
        st.divider()
        st.subheader("Rebalance Actions")
        from datetime import date
        report = compute_rebalance(
            current_positions=current_positions,
            target_portfolio=portfolio,
            as_of_date=date.today().isoformat(),
        )

        if report.needs_rebalance:
            st.warning(f"Rebalance needed — max drift: {report.drift_pct*100:.1f}%")
        else:
            st.success("Portfolio is within tolerance. No rebalance needed.")

        if report.actions:
            action_rows = []
            for a in report.actions:
                action_rows.append({
                    "Ticker":    a.ticker,
                    "Action":    a.action,
                    "Current":   f"{a.current_shares:.2f} sh",
                    "Target":    f"{a.target_shares:.2f} sh",
                    "Delta":     f"{a.delta_shares:+.2f} sh",
                    "$ Change":  f"${a.delta_dollars:+,.0f}",
                    "Reason":    a.reason,
                })
            st.dataframe(action_rows, use_container_width=True, hide_index=True)
