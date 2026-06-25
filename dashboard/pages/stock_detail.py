"""Stock Detail page — per-ticker factor breakdown, stress test, AI notes."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Stock Detail")

    from dashboard.state import get_universe, get_live_prices_cached

    universe = get_universe()
    if not universe:
        st.warning("No universe data. Run --refresh-data first.")
        return

    tickers = sorted(s["ticker"] for s in universe)

    # Ticker selector — also settable from other pages via session_state
    default_idx = 0
    if "detail_ticker" in st.session_state and st.session_state["detail_ticker"] in tickers:
        default_idx = tickers.index(st.session_state["detail_ticker"])

    ticker = st.selectbox("Select ticker", tickers, index=default_idx)
    st.session_state["detail_ticker"] = ticker

    stock = next((s for s in universe if s["ticker"] == ticker), None)
    if stock is None:
        st.error(f"{ticker} not found in universe.")
        return

    st.divider()

    # ── Live price ────────────────────────────────────────────────────────────
    live = get_live_prices_cached((ticker,))
    live_d = live.get(ticker, {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ticker",    ticker)
    col2.metric("Live Price", f"${live_d.get('price', stock.get('price', 0)):.2f}")
    col3.metric("Change",    f"{live_d.get('change_pct', 0):+.2f}%")
    col4.metric("Market Cap", f"${stock.get('mkt_cap', 0)/1e9:.1f}B" if stock.get("mkt_cap") else "—")
    col5.metric("Shariah",   stock.get("shariah_status", "unknown").upper())

    st.caption(f"Name: {stock.get('name', '—')}  |  Sector: {stock.get('sector', '—')}  |  Industry: {stock.get('industry', '—')}")

    st.divider()

    # ── Factor scores ─────────────────────────────────────────────────────────
    st.subheader("Factor Scores")
    factor_cols = [
        ("Composite",          "composite"),
        ("Quality",            "quality"),
        ("Momentum",           "momentum"),
        ("Earnings Revisions", "earnings_revisions"),
        ("Valuation",          "valuation"),
        ("Earnings Quality",   "earnings_quality"),
        ("Moat",               "moat"),
        ("Capital Allocation", "capital_allocation"),
        ("Risk Adjustment",    "risk_adjustment"),
        ("AI Research",        "ai_research"),
    ]

    score_rows = []
    for label, key in factor_cols:
        val = stock.get(key)
        if val is not None:
            delta = val - 50
            score_rows.append({"Factor": label, "Score": f"{val:.1f}", "vs Neutral": f"{delta:+.1f}"})

    if score_rows:
        st.dataframe(score_rows, use_container_width=True, hide_index=True)

        # Bar chart
        import pandas as pd
        scores_only = {label: float(stock.get(key, 50)) for label, key in factor_cols if key != "composite"}
        chart_df = pd.DataFrame({"Score": scores_only})
        st.bar_chart(chart_df)
    else:
        st.info("Run --refresh-data to see factor scores.")

    st.divider()

    # ── Entry Analysis ────────────────────────────────────────────────────────
    st.subheader("Entry Analysis")
    st.caption("ATR-anchored entry prices. Recalculated live on every load. No hardcoded discounts.")

    price = float(stock.get("price") or 0)
    if price > 0:
        with st.spinner("Computing entry analysis…"):
            try:
                from portfolio.entry_price import compute_entry_analysis
                ea = compute_entry_analysis(
                    ticker           = ticker,
                    current_price    = price,
                    valuation_score  = float(stock.get("valuation", 50)),
                    composite_score  = float(stock.get("composite", 50)),
                    model_confidence = float(stock.get("model_confidence", 65)),
                    risk_score       = float(stock.get("risk_adjustment", 50)),
                    momentum_score   = float(stock.get("momentum", 50)),
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price",    f"${ea.current_price:.2f}")
                c2.metric("Buy Below",         f"${ea.buy_limit:.2f}"         if ea.buy_limit        else "—")
                c3.metric("Strong Buy Below",  f"${ea.strong_buy_limit:.2f}"  if ea.strong_buy_limit else "—")

                c4, c5, c6 = st.columns(3)
                c4.metric("Entry Score",       f"{ea.entry_score:.0f} / 100")
                c5.metric("Entry Rating",       ea.entry_rating)
                c6.metric("ATR-14",            f"${ea.atr14:.2f}" if ea.atr14 else "—")

                c7, c8 = st.columns(2)
                c7.metric("Confidence",        f"{ea.confidence_label} ({ea.model_confidence:.0f})")
                c8.metric("Risk Level",         f"{ea.risk_score:.0f} / 100")

                rating_color = {
                    "Strong Buy": "green", "Buy": "#2563eb",
                    "Watch": "orange", "Wait": "red",
                }.get(ea.entry_rating, "gray")
                st.markdown(
                    f"<p style='color:{rating_color};font-size:15px'><b>{ea.entry_rating}:</b> {ea.explanation}</p>",
                    unsafe_allow_html=True,
                )

                with st.expander("Score component breakdown"):
                    comp_rows = [
                        {"Component": "Valuation (40%)",    "Score": f"{ea.val_component:.1f}",  "Weight": "40%"},
                        {"Component": "Momentum (20%)",     "Score": f"{ea.mom_component:.1f}",  "Weight": "20%"},
                        {"Component": "Confidence (20%)",   "Score": f"{ea.conf_component:.1f}", "Weight": "20%"},
                        {"Component": "Risk (20%)",         "Score": f"{ea.risk_component:.1f}", "Weight": "20%"},
                        {"Component": "Entry Score",        "Score": f"{ea.entry_score:.1f}",    "Weight": "Total"},
                    ]
                    st.dataframe(comp_rows, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Entry analysis error: {e}")
    else:
        st.info("No price data available — run --refresh-data first.")

    st.divider()

    # ── Stress test ───────────────────────────────────────────────────────────
    st.subheader("Stress Test History")
    if st.button("Run Stress Test"):
        with st.spinner(f"Running stress test for {ticker}…"):
            try:
                from factors.stress_test import compute_stress_test
                result = compute_stress_test(ticker)
                d = result.to_dict()
                crisis_rows = []
                for name, data in d.get("crises", {}).items():
                    if isinstance(data, dict) and data.get("has_data"):
                        crisis_rows.append({
                            "Crisis":       name,
                            "Max Drawdown": f"{data.get('max_drawdown', 0):.1f}%",
                            "Total Return": f"{data.get('total_return', 0):.1f}%",
                            "SPY Drawdown": f"{data.get('spy_drawdown', 0):.1f}%",
                            "vs SPY":       f"{data.get('relative_vs_spy', 0):+.1f}%",
                        })
                if crisis_rows:
                    st.dataframe(crisis_rows, use_container_width=True, hide_index=True)
                else:
                    st.warning("Insufficient price history for stress test windows.")
            except Exception as e:
                st.error(f"Stress test error: {e}")

    st.divider()

    # ── Model confidence ──────────────────────────────────────────────────────
    st.subheader("Model Confidence")
    if st.button("Compute Confidence"):
        with st.spinner("Computing model confidence…"):
            try:
                from factors.model_confidence import compute_model_confidence
                result = compute_model_confidence(
                    data_quality_score=float(stock.get("data_quality_score", 70)),
                    ai_score=float(stock.get("ai_research", 50)),
                    factor_stability_score=float(stock.get("factor_stability", 70)),
                    analyst_coverage_score=float(stock.get("analyst_coverage", 50)),
                )
                d = result.to_dict()
                c1, c2 = st.columns(2)
                c1.metric("Confidence Score", f"{d['score']:.0f}/100")
                c2.metric("Label", d["label"])
                with st.expander("Breakdown"):
                    st.json(d)
            except Exception as e:
                st.error(f"Confidence error: {e}")

    st.divider()

    # ── AI Risk Review (from scored_universe ai_detail) ───────────────────────
    st.subheader("AI Risk Review")
    st.caption(
        "Qualitative risk layer. Max effective weight: ~3%. "
        "Confidence < 30% → score clamped to neutral 50. "
        "One-way dampening: AI can only reduce moat, never boost it."
    )

    ai_detail = stock.get("ai_detail") or {}
    ai_score  = stock.get("ai_research", 50)
    ai_conf   = ai_detail.get("confidence", 0.0)

    if ai_detail:
        c1, c2, c3 = st.columns(3)
        c1.metric("AI Score", f"{ai_score:.1f}")
        c2.metric("Confidence", f"{ai_conf*100:.0f}%")
        c3.metric("Effective", f"{ai_score:.1f}" if ai_conf >= 0.30 else "50.0 (clamped)")

        if ai_conf < 0.30:
            st.warning("Confidence < 30% — AI score clamped to neutral 50. No effect on ranking.")

        def _show_list(label: str, items: list, color: str = "warning") -> None:
            if not items:
                return
            st.markdown(f"**{label}**")
            for item in items:
                if color == "error":
                    st.error(f"⚑ {item}")
                else:
                    st.warning(f"⚠ {item}")

        thesis_breakers = ai_detail.get("all_thesis_breakers", [])
        if thesis_breakers:
            st.markdown("**🚨 Thesis Breakers**")
            for tb in thesis_breakers:
                st.error(f"⚑ {tb}")

        col1, col2 = st.columns(2)
        with col1:
            _show_list("Accounting Concerns", ai_detail.get("all_accounting_concerns", []))
            _show_list("Management Concerns", ai_detail.get("all_management_concerns", []))
        with col2:
            _show_list("Thesis Risks",        ai_detail.get("all_thesis_risks", []))
            _show_list("Moat Concerns",       ai_detail.get("all_moat_concerns", []))

        red_flags = ai_detail.get("all_red_flags", [])
        if red_flags:
            with st.expander(f"All red flags ({len(red_flags)})"):
                for f in red_flags:
                    st.error(f"⚑ {f}")

        if not any([
            thesis_breakers,
            ai_detail.get("all_accounting_concerns"),
            ai_detail.get("all_management_concerns"),
            ai_detail.get("all_thesis_risks"),
            ai_detail.get("all_moat_concerns"),
            red_flags,
        ]):
            st.success("No AI risk flags in the stored analysis.")
    else:
        st.info("No AI analysis stored. Run --refresh-ai to generate.")

    st.divider()

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("Raw data"):
        st.json(stock)
