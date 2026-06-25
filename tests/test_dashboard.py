"""
Tests for Phase 9: Streamlit Dashboard.

Tests verify structural rules without running a live Streamlit server.
No API calls. Streamlit is mocked via unittest.mock where needed.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import py_compile
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DASHBOARD_ROOT = Path(__file__).parent.parent / "dashboard"
APP_PY = Path(__file__).parent.parent / "app.py"

ALL_PAGE_MODULES = [
    "dashboard.pages.overview",
    "dashboard.pages.recommendations",
    "dashboard.pages.screener",
    "dashboard.pages.backtest",
    "dashboard.pages.portfolio_tracker",
    "dashboard.pages.stock_analysis",
    "dashboard.pages.valuation",
    "dashboard.pages.data_quality",
    "dashboard.pages.ai_research",
]


def _src(module_path: str) -> str:
    """Return source code of a module given its dotted path."""
    path = Path(__file__).parent.parent / module_path.replace(".", "/")
    if not path.suffix:
        path = path.with_suffix(".py")
    return path.read_text(encoding="utf-8")


def _app_src() -> str:
    return APP_PY.read_text(encoding="utf-8")


def _state_src() -> str:
    return (DASHBOARD_ROOT / "state.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# py_compile: all dashboard files must parse without syntax errors
# ---------------------------------------------------------------------------

class TestPyCompile:
    @pytest.mark.parametrize("module", ALL_PAGE_MODULES + [
        "dashboard.state",
        "dashboard.components",
        "dashboard.data_access",
        "dashboard.charts",
    ])
    def test_compiles(self, module: str):
        path = Path(__file__).parent.parent / (module.replace(".", "/") + ".py")
        py_compile.compile(str(path), doraise=True)

    def test_app_py_compiles(self):
        py_compile.compile(str(APP_PY), doraise=True)


# ---------------------------------------------------------------------------
# Structural: no fake tabs, no CSS sidebar hiding
# ---------------------------------------------------------------------------

class TestStructuralRules:
    def test_no_st_tabs_in_any_page(self):
        """st.tabs() must not appear in any page module."""
        for mod in ALL_PAGE_MODULES:
            src = _src(mod)
            assert "st.tabs(" not in src, f"st.tabs() found in {mod}"

    def test_no_st_tabs_in_app(self):
        assert "st.tabs(" not in _app_src()

    def test_no_css_sidebar_hiding_in_app(self):
        """No CSS that hides the sidebar (display:none on sidebar elements)."""
        src = _app_src()
        assert "display:none" not in src.lower()
        assert "display: none" not in src.lower()
        assert "[data-testid" not in src or "display" not in src

    def test_no_css_sidebar_hiding_in_pages(self):
        for mod in ALL_PAGE_MODULES:
            src = _src(mod)
            assert "display:none" not in src.lower(), f"CSS sidebar hide found in {mod}"

    def test_sidebar_uses_radio_in_app(self):
        """Navigation must use st.radio, not st.tabs or st.selectbox."""
        src = _app_src()
        assert "st.radio(" in src

    def test_pages_defined_in_app(self):
        """PAGES list must contain the V6 11-page set."""
        src = _app_src()
        assert "PAGES = [" in src or 'PAGES=[' in src
        page_names = [
            "Overview", "Live Recommendations", "Final Picks",
            "Stock Screener", "Stock Detail", "Portfolio Tracker",
            "Data Quality", "AI Research", "Stress Tests",
            "Factor Monitor", "Settings",
        ]
        for name in page_names:
            assert name in src, f"Page '{name}' not found in app.py"


# ---------------------------------------------------------------------------
# Recommendation parity: safe_recommendations() is the single source
# ---------------------------------------------------------------------------

class TestRecommendationParity:
    def test_state_imports_safe_recommendations(self):
        """state.py must import safe_recommendations from portfolio.recommendation_guard."""
        src = _state_src()
        assert "safe_recommendations" in src
        assert "portfolio.recommendation_guard" in src

    def test_recommendations_page_does_not_reimplement_logic(self):
        """recommendations.py must not contain its own BUY NOW / WATCHLIST / AVOID logic."""
        src = _src("dashboard.pages.recommendations")
        # It should reference safe_recommendations via state, not duplicate it
        assert "def _classify" not in src
        assert "if shariah" not in src

    def test_no_duplicate_recommendation_logic_in_pages(self):
        """No page should independently classify stocks without calling safe_recommendations."""
        forbidden = [
            'action = "BUY NOW"',
            'action = "WATCHLIST"',
            'action = "AVOID"',
        ]
        for mod in ALL_PAGE_MODULES:
            src = _src(mod)
            for pattern in forbidden:
                assert pattern not in src, (
                    f"Direct label assignment '{pattern}' found in {mod}. "
                    "Labels must come from safe_recommendations() only."
                )

    def test_state_get_recommendations_calls_safe_recommendations(self):
        """get_recommendations() in state.py must call safe_recommendations()."""
        src = _state_src()
        assert "safe_recommendations(" in src

    def test_app_dispatches_all_pages(self):
        """Every V6 page must have a dispatch branch in app.py."""
        src = _app_src()
        pages = [
            "Overview", "Live Recommendations", "Final Picks",
            "Stock Screener", "Stock Detail", "Portfolio Tracker",
            "Data Quality", "AI Research", "Stress Tests",
            "Factor Monitor", "Settings",
        ]
        for page in pages:
            assert page in src, f"'{page}' not dispatched in app.py"


# ---------------------------------------------------------------------------
# State module: functional tests (mock streamlit)
# ---------------------------------------------------------------------------

class TestStateModule:
    """
    Test the recommendation pipeline that state.py orchestrates.
    We call the underlying functions directly — no streamlit session_state
    needed and no sys.modules manipulation (which breaks numpy).

    The universe loaders are patched to return a realistic set of stocks so
    tests run without needing real CSV files on disk.
    """

    _FAKE_UNIVERSE = [
        {"ticker": "AAPL",  "composite": 88.1, "quality": 85.0, "momentum": 90.0,
         "valuation": 70.0, "earnings_revisions": 80.0, "ai_research": 50.0,
         "price": 189.0, "mkt_cap": 2.9e12, "avg_volume": 55e6,
         "shariah_status": "compliant", "name": "Apple Inc.", "sector": "Technology"},
        {"ticker": "MSFT",  "composite": 85.7, "quality": 83.0, "momentum": 85.0,
         "valuation": 68.0, "earnings_revisions": 78.0, "ai_research": 50.0,
         "price": 415.0, "mkt_cap": 3.1e12, "avg_volume": 22e6,
         "shariah_status": "compliant", "name": "Microsoft Corp", "sector": "Technology"},
        {"ticker": "NVDA",  "composite": 83.2, "quality": 80.0, "momentum": 88.0,
         "valuation": 60.0, "earnings_revisions": 75.0, "ai_research": 50.0,
         "price": 875.0, "mkt_cap": 2.1e12, "avg_volume": 41e6,
         "shariah_status": "compliant", "name": "NVIDIA Corp", "sector": "Technology"},
        {"ticker": "GOOG",  "composite": 79.4, "quality": 77.0, "momentum": 80.0,
         "valuation": 65.0, "earnings_revisions": 70.0, "ai_research": 50.0,
         "price": 174.0, "mkt_cap": 2.2e12, "avg_volume": 23e6,
         "shariah_status": "compliant", "name": "Alphabet Inc.", "sector": "Technology"},
        {"ticker": "META",  "composite": 76.0, "quality": 74.0, "momentum": 76.0,
         "valuation": 62.0, "earnings_revisions": 72.0, "ai_research": 50.0,
         "price": 505.0, "mkt_cap": 1.3e12, "avg_volume": 18e6,
         "shariah_status": "compliant", "name": "Meta Platforms", "sector": "Technology"},
        {"ticker": "AMZN",  "composite": 72.3, "quality": 70.0, "momentum": 74.0,
         "valuation": 58.0, "earnings_revisions": 68.0, "ai_research": 50.0,
         "price": 185.0, "mkt_cap": 1.9e12, "avg_volume": 38e6,
         "shariah_status": "compliant", "name": "Amazon.com Inc.", "sector": "Technology"},
        {"ticker": "BAC",   "composite": 68.0, "quality": 65.0, "momentum": 70.0,
         "valuation": 60.0, "earnings_revisions": 65.0, "ai_research": 50.0,
         "price": 39.0, "mkt_cap": 310e9, "avg_volume": 40e6,
         "shariah_status": "non_compliant", "name": "Bank of America", "sector": "Financials"},
    ]

    def _run_pipeline(self):
        """
        Run _build_inputs → build_portfolio → safe_recommendations
        with patched universe loaders (no real CSV files needed).
        """
        from portfolio.recommendation_guard import safe_recommendations
        from portfolio.constructor import build_portfolio
        import dashboard.state as state_mod

        with patch.object(state_mod, "_load_scored_universe", return_value=self._FAKE_UNIVERSE):
            all_scores, crs, prices, shariah = state_mod._build_inputs()

        portfolio = build_portfolio(all_scores, crs, prices)
        recs = safe_recommendations(
            portfolio_result=portfolio,
            all_scores=all_scores,
            constraint_results=crs,
            prices=prices,
            shariah_statuses=shariah,
        )
        return recs, portfolio, crs, shariah

    def test_get_recommendations_returns_list(self):
        """Pipeline must return a non-empty list of Recommendation objects."""
        recs, _, _, _ = self._run_pipeline()
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_buy_now_tickers_match_portfolio(self):
        """Tickers in BUY NOW must exactly match portfolio positions."""
        recs, portfolio, _, _ = self._run_pipeline()
        buy_now_tickers = {r.ticker for r in recs if r.action == "BUY NOW"}
        portfolio_tickers = {p.ticker for p in portfolio.positions}
        assert buy_now_tickers == portfolio_tickers

    def test_recommendation_ranks_are_sequential(self):
        """BUY NOW ranks must be 1, 2, 3, ... with no gaps."""
        recs, _, _, _ = self._run_pipeline()
        buy_ranks = sorted(r.rank for r in recs if r.action == "BUY NOW")
        assert buy_ranks == list(range(1, len(buy_ranks) + 1))

    def test_watchlist_are_compliant_and_passing(self):
        """WATCHLIST stocks must be Shariah-compliant and pass all constraints."""
        recs, _, crs, shariah = self._run_pipeline()
        for r in recs:
            if r.action == "WATCHLIST":
                assert shariah.get(r.ticker) == "compliant", \
                    f"{r.ticker} on WATCHLIST but shariah={shariah.get(r.ticker)}"
                cr = crs.get(r.ticker)
                assert cr is not None and cr.passed, \
                    f"{r.ticker} on WATCHLIST but constraint failed"

    def test_avoid_has_rejection_reasons(self):
        """AVOID stocks must have at least one rejection reason."""
        recs, _, _, _ = self._run_pipeline()
        for r in recs:
            if r.action == "AVOID":
                assert len(r.rejection_reasons) > 0, \
                    f"{r.ticker} is AVOID but has no rejection reasons"


# ---------------------------------------------------------------------------
# Page render functions: importable and have render()
# ---------------------------------------------------------------------------

class TestPageStructure:
    @pytest.mark.parametrize("mod_path", ALL_PAGE_MODULES)
    def test_each_page_has_render_function(self, mod_path: str):
        """Every page module must expose a render() function."""
        src = _src(mod_path)
        assert "def render()" in src, f"render() not found in {mod_path}"

    def test_app_calls_render(self):
        """app.py must call render() for each dispatched page."""
        src = _app_src()
        assert "render()" in src

    def test_state_has_get_recommendations(self):
        src = _state_src()
        assert "def get_recommendations()" in src

    def test_state_has_clear_cache(self):
        src = _state_src()
        assert "def clear_recommendations_cache()" in src
