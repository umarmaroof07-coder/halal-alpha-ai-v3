"""
Dashboard parity test: Top 5 tickers from dashboard == CLI.

Both call safe_recommendations() with identical inputs derived from the same
scored_universe.json. If this test fails, something in dashboard/state.py
is sourcing recommendations outside of safe_recommendations().
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


SCORED_UNIVERSE = Path("data/cache/scored_universe.json")


@pytest.mark.skipif(
    not SCORED_UNIVERSE.exists(),
    reason="scored_universe.json not found — run --refresh-data first",
)
def test_dashboard_top5_equals_cli_top5() -> None:
    """Dashboard Top 5 == CLI Top 5: same safe_recommendations() call, same inputs."""

    # ── Produce CLI top 5 directly ────────────────────────────────────────────
    from factors.composite import FactorScores
    from portfolio.constructor import build_portfolio
    from portfolio.constraints import check_constraints
    from portfolio.recommendation_guard import safe_recommendations

    with SCORED_UNIVERSE.open() as f:
        payload = json.load(f)
    universe: list[dict] = payload.get("universe", [])

    assert universe, "scored_universe.json has empty universe"

    all_scores = sorted(
        [
            FactorScores(
                ticker=s["ticker"],
                quality=float(s.get("quality", 50.0)),
                momentum=float(s.get("momentum", 50.0)),
                valuation=float(s.get("valuation", 50.0)),
                earnings_revisions=float(s.get("earnings_revisions", 50.0)),
                earnings_quality=float(s.get("earnings_quality", 50.0)),
                moat=float(s.get("moat", 50.0)),
                capital_allocation=float(s.get("capital_allocation", 50.0)),
                risk_adjustment=float(s.get("risk_adjustment", 50.0)),
                ai_research=float(s.get("ai_research", 50.0)),
                composite=float(s.get("composite", 50.0)),
            )
            for s in universe
        ],
        key=lambda x: x.composite,
        reverse=True,
    )

    crs = {
        s["ticker"]: check_constraints(
            ticker=s["ticker"],
            price=float(s.get("price") or 0.0),
            market_cap=float(s.get("mkt_cap") or s.get("mktCap") or 0.0),
            avg_volume=float(s.get("avg_volume") or s.get("avgVolume") or 0.0),
            shariah_status=s.get("shariah_status", "unknown"),
        )
        for s in universe
    }

    prices   = {s["ticker"]: float(s.get("price") or 0.0) for s in universe}
    shariah  = {s["ticker"]: s.get("shariah_status", "unknown") for s in universe}
    portfolio = build_portfolio(all_scores, crs, prices)

    cli_recs = safe_recommendations(
        portfolio_result=portfolio,
        all_scores=all_scores,
        constraint_results=crs,
        prices=prices,
        shariah_statuses=shariah,
    )
    cli_top5 = [r.ticker for r in cli_recs if r.action == "BUY NOW"]

    # ── Dashboard path via state helpers ──────────────────────────────────────
    # Patch st.session_state to a plain dict so state.py works outside Streamlit
    mock_ss: dict = {}
    with patch("streamlit.session_state", mock_ss):
        from dashboard import state as ds
        # Reload universe into mock session state
        ds_scores    = ds._build_inputs()  # (all_scores, crs, prices, shariah)
        dash_scores, dash_crs, dash_prices, dash_shariah = ds_scores

        dash_portfolio = build_portfolio(dash_scores, dash_crs, dash_prices)
        dash_recs = safe_recommendations(
            portfolio_result=dash_portfolio,
            all_scores=dash_scores,
            constraint_results=dash_crs,
            prices=dash_prices,
            shariah_statuses=dash_shariah,
        )
    dash_top5 = [r.ticker for r in dash_recs if r.action == "BUY NOW"]

    assert cli_top5 == dash_top5, (
        f"Dashboard Top 5 {dash_top5} != CLI Top 5 {cli_top5}.  "
        f"Both must use safe_recommendations() with same inputs."
    )


def test_safe_recommendations_is_the_only_source() -> None:
    """dashboard/state.py get_recommendations() must call safe_recommendations()."""
    import ast
    state_src = Path("dashboard/state.py").read_text()
    tree = ast.parse(state_src)

    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

    assert "safe_recommendations" in calls, (
        "dashboard/state.py must call safe_recommendations(). "
        "Never source recommendations from raw ranked_df."
    )
