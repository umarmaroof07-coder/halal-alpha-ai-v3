"""
AI Research page — qualitative risk-review layer (V6).

Role: risk identification, not ranking. AI surfaces thesis risks, accounting
concerns, management concerns, moat concerns, and thesis breakers.
AI max effective weight: ~3% (30% of Moat × 10% Moat weight).
Confidence < 30% → score clamped to neutral 50 (no signal).
One-way dampening: AI can only reduce moat, never boost it.
"""

from __future__ import annotations

import streamlit as st


def _risk_section(title: str, items: list[str], color: str = "orange") -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        if color == "red":
            st.error(f"⚑ {item}")
        elif color == "orange":
            st.warning(f"⚠ {item}")
        else:
            st.info(f"ℹ {item}")


def render() -> None:
    st.title("AI Research — Qualitative Risk Review")

    st.info(
        "**Role:** AI is a risk-identification layer, not a ranking factor.  \n"
        "Max effective weight: ~3% (30% of Moat × 10% Moat weight).  \n"
        "Confidence < 30% → score clamped to neutral 50.  \n"
        "One-way dampening: AI can only *reduce* moat scores, never boost them.  \n"
        "AI will never push a weak stock into Top 5."
    )

    from dashboard.state import get_universe
    from config.settings import ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        st.warning(
            "ANTHROPIC_API_KEY is not set. AI Research returns neutral 50. "
            "Add it to your .env file to enable live analysis."
        )

    universe = get_universe()
    tickers  = sorted(s["ticker"] for s in universe)

    if not tickers:
        st.info("No universe data. Run --refresh-data first.")
        return

    col_l, col_r = st.columns([2, 1])
    with col_l:
        ticker = st.selectbox("Select ticker", tickers)
    with col_r:
        as_of_date = st.date_input("As-of date").isoformat()

    st.divider()

    filing_text     = st.text_area(
        "SEC Filing text (10-K / 10-Q MD&A or Risk Factors)", height=140,
        placeholder="Paste the MD&A or risk factors section here…"
    )
    transcript_text = st.text_area(
        "Earnings Call transcript excerpt", height=140,
        placeholder="Paste the earnings call transcript here…"
    )
    profile_text    = st.text_area(
        "Company business description (moat assessment)", height=90,
        placeholder="Paste the business description here…"
    )

    st.caption(
        "Claude only uses the text you provide. It cannot hallucinate — "
        "if a section is empty, that sub-analyzer returns neutral 50 with confidence 0."
    )

    if st.button("🔬 Run AI Risk Review"):
        with st.spinner("Calling Claude API…"):
            from ai_research.composite import run_ai_research
            result = run_ai_research(
                ticker=ticker,
                filing_text=filing_text or None,
                transcript_text=transcript_text or None,
                company_profile=profile_text or None,
                as_of_date=as_of_date,
                is_backtest=False,
            )

        st.divider()
        st.subheader(f"Risk Review — {ticker}")

        # ── Header metrics ────────────────────────────────────────────────────
        conf_pct = result.confidence * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("AI Score (display only)", f"{result.score:.1f} / 100")
        c2.metric("Confidence", f"{conf_pct:.0f}%")
        c3.metric(
            "Effective Score",
            f"{result.score:.1f}" if conf_pct >= 30 else "50.0 (clamped)",
            delta="clamped — conf < 30%" if conf_pct < 30 else None,
        )

        if conf_pct < 30:
            st.warning(
                f"⚠ Confidence {conf_pct:.0f}% is below the 30% threshold. "
                "Score is clamped to neutral 50 and has NO effect on rankings or moat."
            )
        elif result.score < 45:
            st.error(f"AI flags concerns for {ticker}. Review risks below before trading.")
        elif result.score >= 65:
            st.success(f"No major concerns detected. Confidence {conf_pct:.0f}%.")
        else:
            st.info(f"Neutral — insufficient signal or mixed findings.")

        st.caption(
            "⚑ One-way dampening: if the moat sub-score < 50 it reduces the moat factor. "
            "A high AI score (>50) is ignored for ranking — AI cannot boost stocks."
        )

        st.divider()

        # ── Sub-scores ────────────────────────────────────────────────────────
        st.subheader("Sub-Analyzer Scores")
        sub_rows = []
        for label, sub in [
            ("SEC Filing",  result.sec_filing),
            ("Transcript",  result.transcript),
            ("Moat",        result.moat),
            ("Management",  result.management),
        ]:
            if sub:
                conf_flag = "⚠ low confidence" if sub.confidence < 0.30 else ""
                sub_rows.append({
                    "Analyzer":   label,
                    "Score":      f"{sub.score:.1f}",
                    "Confidence": f"{sub.confidence*100:.0f}%",
                    "Flag":       conf_flag,
                    "Rationale":  sub.rationale[:120] + "…" if len(sub.rationale) > 120 else sub.rationale,
                })
        if sub_rows:
            st.dataframe(sub_rows, use_container_width=True, hide_index=True)

        st.divider()

        # ── Thesis Breakers (most severe) ─────────────────────────────────────
        if result.all_thesis_breakers:
            st.subheader("🚨 Thesis Breakers")
            st.error("These findings would normally trigger an immediate review or sell decision.")
            for item in result.all_thesis_breakers:
                st.error(f"⚑ {item}")
            st.divider()

        # ── Risk categories ───────────────────────────────────────────────────
        st.subheader("Risk Categories")

        col_a, col_b = st.columns(2)
        with col_a:
            _risk_section("Accounting Concerns", result.all_accounting_concerns, "orange")
            _risk_section("Management Concerns", result.all_management_concerns, "orange")
        with col_b:
            _risk_section("Thesis Risks",        result.all_thesis_risks,        "orange")
            _risk_section("Moat Concerns",       result.all_moat_concerns,       "orange")

        if not any([
            result.all_thesis_breakers,
            result.all_accounting_concerns,
            result.all_management_concerns,
            result.all_thesis_risks,
            result.all_moat_concerns,
        ]):
            st.success("No specific risk categories flagged in the provided text.")

        st.divider()

        # ── Red flags / positives ─────────────────────────────────────────────
        col_rf, col_pos = st.columns(2)
        with col_rf:
            if result.all_red_flags:
                st.markdown("**Red Flags**")
                for f in result.all_red_flags:
                    st.error(f"⚑ {f}")
        with col_pos:
            if result.all_positives:
                st.markdown("**Positives**")
                for p in result.all_positives:
                    st.success(f"✓ {p}")

        # ── Cache info ────────────────────────────────────────────────────────
        sub_list = [result.sec_filing, result.transcript, result.moat, result.management]
        any_cached = any(r and r.cache_hit for r in sub_list if r)
        st.caption(
            "ℹ Results loaded from cache." if any_cached else
            "ℹ Fresh analysis — results cached for future calls."
        )

        # ── Raw JSON ──────────────────────────────────────────────────────────
        with st.expander("Raw JSON"):
            st.json(result.to_dict())
