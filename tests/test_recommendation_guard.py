"""
Tests for Phase 6: recommendation_guard.py

Verifies that safe_recommendations() is the ONLY source of
BUY NOW / WATCHLIST / AVOID labels and that the classification logic
is correct per the stated rules.
"""

from __future__ import annotations

import pytest
from portfolio.recommendation_guard import safe_recommendations, Recommendation
from portfolio.constructor import PortfolioResult, PortfolioPosition
from portfolio.constraints import ConstraintResult
from factors.composite import FactorScores
from config.settings import CONVICTION_WEIGHTS, CONVICTION_DOLLARS, ACCOUNT_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cr(ticker: str, passed: bool, failures=None) -> ConstraintResult:
    return ConstraintResult(ticker=ticker, passed=passed, failures=failures or [])


def _score(ticker: str, composite: float = 75.0) -> FactorScores:
    return FactorScores(ticker=ticker, composite=composite)


def _make_position(ticker: str, rank: int, price: float = 100.0) -> PortfolioPosition:
    w = CONVICTION_WEIGHTS[rank - 1]
    d = CONVICTION_DOLLARS[rank - 1]
    return PortfolioPosition(
        rank=rank, ticker=ticker, composite_score=80.0,
        conviction_weight=w, dollar_amount=d,
        price=price, shares_to_buy=d / price,
        constraint_result=_cr(ticker, True),
    )


def _make_portfolio(positions: list[PortfolioPosition]) -> PortfolioResult:
    total = sum(p.dollar_amount for p in positions)
    return PortfolioResult(
        positions=positions,
        total_invested=total,
        cash_remaining=ACCOUNT_SIZE - total,
        n_positions=len(positions),
    )


def _run(
    portfolio_positions: list[PortfolioPosition],
    all_scores: list[FactorScores],
    constraint_results: dict[str, ConstraintResult],
    prices: dict[str, float],
    shariah_statuses: dict[str, str],
) -> list[Recommendation]:
    portfolio = _make_portfolio(portfolio_positions)
    return safe_recommendations(
        portfolio_result=portfolio,
        all_scores=all_scores,
        constraint_results=constraint_results,
        prices=prices,
        shariah_statuses=shariah_statuses,
    )


# ---------------------------------------------------------------------------
# BUY NOW
# ---------------------------------------------------------------------------

class TestBuyNow:
    def test_portfolio_ticker_gets_buy_now(self):
        pos = _make_position("AAPL", rank=1, price=150.0)
        recs = _run(
            [pos],
            all_scores=[_score("AAPL", 90.0)],
            constraint_results={"AAPL": _cr("AAPL", True)},
            prices={"AAPL": 150.0},
            shariah_statuses={"AAPL": "compliant"},
        )
        buy = [r for r in recs if r.action == "BUY NOW"]
        assert len(buy) == 1
        assert buy[0].ticker == "AAPL"

    def test_buy_now_has_rank(self):
        pos = _make_position("MSFT", rank=2, price=300.0)
        recs = _run(
            [pos],
            all_scores=[_score("MSFT", 85.0)],
            constraint_results={"MSFT": _cr("MSFT", True)},
            prices={"MSFT": 300.0},
            shariah_statuses={"MSFT": "compliant"},
        )
        buy = [r for r in recs if r.action == "BUY NOW"][0]
        assert buy.rank == 2

    def test_buy_now_has_conviction_weight_and_dollars(self):
        pos = _make_position("AAPL", rank=1, price=150.0)
        recs = _run(
            [pos],
            all_scores=[_score("AAPL", 90.0)],
            constraint_results={"AAPL": _cr("AAPL", True)},
            prices={"AAPL": 150.0},
            shariah_statuses={"AAPL": "compliant"},
        )
        buy = [r for r in recs if r.action == "BUY NOW"][0]
        assert buy.conviction_weight == pytest.approx(CONVICTION_WEIGHTS[0])
        assert buy.dollar_amount == pytest.approx(CONVICTION_DOLLARS[0])

    def test_five_buy_now_in_rank_order(self):
        positions = [_make_position(f"S{i+1}", rank=i+1) for i in range(5)]
        scores = [_score(f"S{i+1}", 90 - i * 5) for i in range(5)]
        crs = {f"S{i+1}": _cr(f"S{i+1}", True) for i in range(5)}
        prices = {f"S{i+1}": 100.0 for i in range(5)}
        shariah = {f"S{i+1}": "compliant" for i in range(5)}
        recs = _run(positions, scores, crs, prices, shariah)
        buy_now = [r for r in recs if r.action == "BUY NOW"]
        assert len(buy_now) == 5
        assert [r.rank for r in buy_now] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# WATCHLIST
# ---------------------------------------------------------------------------

class TestWatchlist:
    def test_compliant_passing_non_top5_gets_watchlist(self):
        # Portfolio has S1. S2 is compliant+passing but not in portfolio.
        pos = _make_position("S1", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("S1", 90.0), _score("S2", 70.0)],
            constraint_results={"S1": _cr("S1", True), "S2": _cr("S2", True)},
            prices={"S1": 100.0, "S2": 100.0},
            shariah_statuses={"S1": "compliant", "S2": "compliant"},
        )
        watchlist = [r for r in recs if r.action == "WATCHLIST"]
        assert len(watchlist) == 1
        assert watchlist[0].ticker == "S2"

    def test_watchlist_has_no_dollar_amount(self):
        pos = _make_position("S1", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("S1", 90.0), _score("S2", 70.0)],
            constraint_results={"S1": _cr("S1", True), "S2": _cr("S2", True)},
            prices={"S1": 100.0, "S2": 100.0},
            shariah_statuses={"S1": "compliant", "S2": "compliant"},
        )
        w = [r for r in recs if r.action == "WATCHLIST"][0]
        assert w.dollar_amount is None
        assert w.conviction_weight is None
        assert w.rank is None

    def test_watchlist_sorted_by_score_descending(self):
        pos = _make_position("S1", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("S1", 90.0), _score("W1", 65.0), _score("W2", 80.0)],
            constraint_results={
                "S1": _cr("S1", True), "W1": _cr("W1", True), "W2": _cr("W2", True),
            },
            prices={"S1": 100.0, "W1": 100.0, "W2": 100.0},
            shariah_statuses={"S1": "compliant", "W1": "compliant", "W2": "compliant"},
        )
        watchlist = [r for r in recs if r.action == "WATCHLIST"]
        assert watchlist[0].ticker == "W2"   # higher score first
        assert watchlist[1].ticker == "W1"

    def test_non_compliant_not_on_watchlist(self):
        pos = _make_position("S1", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("S1", 90.0), _score("BAC", 75.0)],
            constraint_results={"S1": _cr("S1", True), "BAC": _cr("BAC", False, ["Shariah: non_compliant"])},
            prices={"S1": 100.0, "BAC": 40.0},
            shariah_statuses={"S1": "compliant", "BAC": "non_compliant"},
        )
        watchlist = [r for r in recs if r.action == "WATCHLIST"]
        assert not any(r.ticker == "BAC" for r in watchlist)

    def test_constraint_failing_not_on_watchlist(self):
        pos = _make_position("S1", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("S1", 90.0), _score("TINY", 70.0)],
            constraint_results={
                "S1": _cr("S1", True),
                "TINY": _cr("TINY", False, ["Market cap $0.50B < $5B minimum"]),
            },
            prices={"S1": 100.0, "TINY": 5.0},
            shariah_statuses={"S1": "compliant", "TINY": "compliant"},
        )
        watchlist = [r for r in recs if r.action == "WATCHLIST"]
        assert not any(r.ticker == "TINY" for r in watchlist)


# ---------------------------------------------------------------------------
# AVOID
# ---------------------------------------------------------------------------

class TestAvoid:
    def test_non_compliant_gets_avoid(self):
        recs = _run(
            [],
            all_scores=[_score("BAC", 60.0)],
            constraint_results={"BAC": _cr("BAC", False, ["Shariah: non_compliant"])},
            prices={"BAC": 40.0},
            shariah_statuses={"BAC": "non_compliant"},
        )
        avoid = [r for r in recs if r.action == "AVOID"]
        assert len(avoid) == 1
        assert avoid[0].ticker == "BAC"

    def test_avoid_has_rejection_reasons(self):
        recs = _run(
            [],
            all_scores=[_score("BAC", 60.0)],
            constraint_results={"BAC": _cr("BAC", False, ["Shariah: non_compliant"])},
            prices={"BAC": 40.0},
            shariah_statuses={"BAC": "non_compliant"},
        )
        avoid = [r for r in recs if r.action == "AVOID"][0]
        assert len(avoid.rejection_reasons) > 0

    def test_price_violation_gets_avoid_with_reason(self):
        recs = _run(
            [],
            all_scores=[_score("BRK", 70.0)],
            constraint_results={"BRK": _cr("BRK", False, ["Price $1500.00 ≥ $1,000 limit"])},
            prices={"BRK": 1500.0},
            shariah_statuses={"BRK": "compliant"},
        )
        avoid = [r for r in recs if r.action == "AVOID"][0]
        assert any("Price" in r or "price" in r.lower() for r in avoid.rejection_reasons)

    def test_avoid_has_no_conviction_weight(self):
        recs = _run(
            [],
            all_scores=[_score("BAC", 60.0)],
            constraint_results={"BAC": _cr("BAC", False, ["Shariah"])},
            prices={"BAC": 40.0},
            shariah_statuses={"BAC": "non_compliant"},
        )
        avoid = [r for r in recs if r.action == "AVOID"][0]
        assert avoid.conviction_weight is None
        assert avoid.dollar_amount is None
        assert avoid.rank is None

    def test_missing_constraint_result_gets_avoid(self):
        # Ticker has no constraint result at all
        recs = _run(
            [],
            all_scores=[_score("UNKN", 60.0)],
            constraint_results={},   # empty
            prices={"UNKN": 50.0},
            shariah_statuses={"UNKN": "compliant"},
        )
        avoid = [r for r in recs if r.action == "AVOID"]
        assert any(r.ticker == "UNKN" for r in avoid)


# ---------------------------------------------------------------------------
# Output ordering and structure
# ---------------------------------------------------------------------------

class TestOutputOrder:
    def test_order_is_buy_then_watchlist_then_avoid(self):
        pos = _make_position("TOP", rank=1)
        recs = _run(
            [pos],
            all_scores=[_score("TOP", 95.0), _score("WATCH", 70.0), _score("BAD", 50.0)],
            constraint_results={
                "TOP":   _cr("TOP", True),
                "WATCH": _cr("WATCH", True),
                "BAD":   _cr("BAD", False, ["Shariah: non_compliant"]),
            },
            prices={"TOP": 100.0, "WATCH": 100.0, "BAD": 40.0},
            shariah_statuses={"TOP": "compliant", "WATCH": "compliant", "BAD": "non_compliant"},
        )
        actions = [r.action for r in recs]
        buy_idx   = actions.index("BUY NOW")
        watch_idx = actions.index("WATCHLIST")
        avoid_idx = actions.index("AVOID")
        assert buy_idx < watch_idx < avoid_idx

    def test_to_dict_structure(self):
        pos = _make_position("AAPL", rank=1, price=150.0)
        recs = _run(
            [pos],
            all_scores=[_score("AAPL", 90.0)],
            constraint_results={"AAPL": _cr("AAPL", True)},
            prices={"AAPL": 150.0},
            shariah_statuses={"AAPL": "compliant"},
        )
        d = recs[0].to_dict()
        for key in ("ticker", "action", "rank", "composite_score",
                    "conviction_weight", "dollar_amount", "price",
                    "shariah_status", "rejection_reasons"):
            assert key in d

    def test_only_safe_recommendations_produces_labels(self):
        """
        Importing constructor and rebalancer should NOT expose any
        BUY NOW / WATCHLIST / AVOID strings in their public API.
        This is a structural guard test.
        """
        import portfolio.constructor as c
        import portfolio.rebalancer as r
        import portfolio.constraints as cnst

        # None of these modules' public names contain the label strings
        for name in dir(c) + dir(r) + dir(cnst):
            assert "BUY NOW" not in name
            assert "WATCHLIST" not in name
            assert "AVOID" not in name
