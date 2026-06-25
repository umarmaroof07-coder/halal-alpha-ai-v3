"""Stock Analysis page — factor breakdown, earnings quality, and live price for a single ticker."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Stock Analysis")

    from dashboard.state import get_universe, get_factor_scores, get_recommendations, get_live_prices_cached

    universe  = get_universe()
    compliant = [s["ticker"] for s in universe if s.get("shariah_status") == "compliant"]

    if not compliant:
        st.info("No compliant stocks found. Run --refresh-data first.")
        return

    ticker     = st.selectbox("Select ticker", sorted(compliant))
    stock_data = next((s for s in universe if s["ticker"] == ticker), None)

    # ── Live price ────────────────────────────────────────────────────────────
    live_map   = get_live_prices_cached((ticker,))
    ld         = live_map.get(ticker, {})
    live_price = ld.get("price")
    chg_pct    = ld.get("change_pct", 0)
    source     = ld.get("source", "—")
    ts_str     = ld.get("timestamp", "")

    st.subheader("Live Price")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    display_price = live_price or (stock_data.get("price") if stock_data else None)
    col_p1.metric("Price",   f"${display_price:.2f}" if display_price else "—",
                  delta=f"{chg_pct:+.2f}%" if ld else None)
    col_p2.metric("Source",  source)
    col_p3.metric("Updated", ts_str[11:19] if ts_str else "cached")
    col_p4.metric("Status",  "FRESH" if ts_str else "STALE")
    st.caption(
        "Prices = live/semi-live (60 s TTL)  |  "
        "Fundamentals = daily  |  Financial statements = quarterly"
    )
    st.divider()

    # ── Factor Scores ─────────────────────────────────────────────────────────
    st.subheader("Factor Scores")
    scores    = get_factor_scores()
    score_map = {s.ticker: s for s in scores}
    fs        = score_map.get(ticker)

    if fs:
        cols = st.columns(7)
        cols[0].metric("Composite",    f"{fs.composite:.1f}")
        cols[1].metric("Quality",      f"{fs.quality:.1f}")
        cols[2].metric("Momentum",     f"{fs.momentum:.1f}")
        cols[3].metric("Valuation",    f"{fs.valuation:.1f}")
        cols[4].metric("Revisions",    f"{fs.earnings_revisions:.1f}")
        cols[5].metric("Earn.Quality", f"{fs.earnings_quality:.1f}")
        cols[6].metric("AI Research",  f"{fs.ai_research:.1f}")

        st.bar_chart({
            "Quality":       fs.quality,
            "Momentum":      fs.momentum,
            "Revisions":     fs.earnings_revisions,
            "Valuation":     fs.valuation,
            "Earn.Quality":  fs.earnings_quality,
            "AI Research":   fs.ai_research,
        })
        st.caption("Weights: Q30% M25% Rev20% Val10% EQ10% AI5%")
    else:
        st.info(f"No factor scores available for {ticker}.")

    st.divider()

    if not stock_data:
        st.info(f"No fundamental data available for {ticker}.")
        return

    # ── Key Fundamentals ──────────────────────────────────────────────────────
    st.subheader("Key Fundamentals")
    col1, col2, col3 = st.columns(3)
    col1.metric("Price",   f"${stock_data.get('price', 0):.2f}")
    col2.metric("Mkt Cap", f"${stock_data.get('mkt_cap', 0)/1e9:.1f}B")
    col3.metric("Shariah", stock_data.get("shariah_status", "unknown").upper())
    vol = stock_data.get("avg_volume", 0) or 0
    st.caption(f"Avg Daily Volume: {vol/1e6:.1f}M shares")

    # ── SBC Warning ───────────────────────────────────────────────────────────
    sbc = stock_data.get("sbc_data")
    if sbc:
        ratio = sbc.get("sbc_fcf_ratio", 0)
        if ratio > 0.20:
            adj_fcf = sbc.get("adj_fcf", 0)
            fcf     = sbc.get("fcf", 0)
            tier    = "large" if ratio > 0.50 else ("medium" if ratio > 0.30 else "small")
            st.warning(
                f"**High stock-based compensation ({tier} SBC penalty applied)**  \n"
                f"SBC is {ratio*100:.0f}% of reported FCF  |  "
                f"Reported FCF: ${fcf/1e9:.2f}B  |  SBC-adjusted FCF: ${adj_fcf/1e9:.2f}B  \n"
                f"FCF overstatement risk — see Earnings Quality section."
            )

    st.divider()

    # ── Earnings Quality Detail ───────────────────────────────────────────────
    st.subheader("Earnings Quality")
    eqd = stock_data.get("earnings_quality_detail") or {}
    if eqd:
        eq_score = stock_data.get("earnings_quality", 50)
        penalty  = eqd.get("sbc_penalty", 0)
        tier_str = eqd.get("sbc_penalty_tier", "none")

        c1, c2, c3 = st.columns(3)
        c1.metric("EQ Score",    f"{eq_score:.1f}")
        c2.metric("SBC Penalty", f"−{penalty:.0%}" if penalty else "None",
                  delta=tier_str if tier_str != "none" else None,
                  delta_color="inverse")
        c3.metric("Signals",     str(len(eqd.get("signals_used") or [])))

        rows = []
        for key, label, note in [
            ("fcf_conversion", "FCF Conversion (FCF/NI)",   "≥1.0 excellent, <0.7 weak"),
            ("accrual_ratio",  "Accrual Ratio ((NI−FCF)/Rev)", "Lower = better quality"),
            ("sbc_ratio",      "SBC / FCF",                  ">20% penalty, >30% medium, >50% large"),
            ("share_dilution", "Share Dilution (YoY)",       "Negative = buybacks (good)"),
            ("debt_trend",     "Debt Trend (ΔDebt/Equity)",  "Negative = deleveraging (good)"),
            ("roic_trend",     "ROIC Trend (YoY change)",    "Positive = improving returns"),
        ]:
            val = eqd.get(key)
            rows.append({
                "Signal":  label,
                "Value":   f"{val:.3f}" if val is not None else "—",
                "Meaning": note,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        for w in (eqd.get("warnings") or []):
            st.warning(f"⚠ {w}")
    else:
        st.info("Earnings quality detail not available. Run --refresh-data.")

    st.divider()

    # ── Moat Quality ─────────────────────────────────────────────────────────
    st.subheader("Moat Quality")
    md = stock_data.get("moat_detail") or {}
    if md and md.get("quant_moat_score") is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Quant Moat",   f"{md['quant_moat_score']:.1f}")
        c2.metric("AI Moat",      f"{md['ai_moat_score']:.1f}" if md.get("ai_moat_score") else "—")
        c3.metric("Blended Moat", f"{md['blended_moat_score']:.1f}")
        st.caption("Blended = 70% quantitative + 30% AI (when AI available)")
    else:
        st.info("Moat detail not available. Run --refresh-data.")

    st.divider()

    # ── Quality Factor Detail ─────────────────────────────────────────────────
    st.subheader("Quality Factor Detail")
    qd = stock_data.get("quality_detail") or {}
    if qd:
        rows = []
        for key, label in [
            ("roic",             "ROIC"),
            ("operating_margin", "Operating Margin"),
            ("net_margin",       "Net Margin"),
            ("fcf_margin",       "FCF Margin"),
            ("equity_ratio",     "Equity Ratio"),
        ]:
            v = qd.get(key)
            rows.append({"Signal": label, "Value": f"{v*100:.1f}%" if v is not None else "—"})
        for key, label in [
            ("revenue_growth",  "Revenue Growth"),
            ("earnings_growth", "Earnings Growth"),
            ("fcf_growth",      "FCF Growth"),
        ]:
            sig = qd.get(key) or {}
            val = sig.get("value")
            flg = sig.get("flag", "—")
            rows.append({
                "Signal": label,
                "Value":  f"{val*100:.0f}%" if val is not None else f"excl ({flg})",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()

    # ── Revisions Detail ──────────────────────────────────────────────────────
    rd = stock_data.get("revisions_detail") or {}
    if rd and rd.get("signals_used"):
        st.subheader("Analyst Revisions")
        c1, c2 = st.columns(2)
        c1.metric("Revisions Score", f"{stock_data.get('earnings_revisions', 50):.1f}")
        c2.metric("Confidence",      rd.get("confidence", "—").upper())
        st.info(rd.get("reason", "No analyst data"))
        if rd.get("total_analysts"):
            st.caption(f"Based on {rd['total_analysts']} analysts")
        if rd.get("price_target_upside") is not None:
            st.caption(f"Analyst price target upside: {rd['price_target_upside']*100:+.1f}%")

    st.divider()

    # ── Recommendation ────────────────────────────────────────────────────────
    recs = get_recommendations()
    rec  = next((r for r in recs if r.ticker == ticker), None)
    if rec:
        st.subheader("Recommendation")
        if rec.action == "BUY NOW":
            st.success(
                f"BUY NOW — Rank #{rec.rank}  |  "
                f"${rec.dollar_amount:.0f}  |  "
                f"{(rec.conviction_weight or 0)*100:.0f}% conviction"
            )
        elif rec.action == "WATCHLIST":
            st.info("WATCHLIST — Passes Shariah + constraints but outside Top 5")
        else:
            st.error("AVOID — " + "; ".join(rec.rejection_reasons))
