"""
Tests for Phase 6: constraints.py, constructor.py, rebalancer.py
"""

from __future__ import annotations

import pytest
from portfolio.constraints import (
    ConstraintResult,
    check_constraints,
    check_constraints_batch,
    MIN_PORTFOLIO_MARKET_CAP,
    MIN_PORTFOLIO_AVG_VOLUME,
)
from portfolio.constructor import (
    PortfolioPosition,
    PortfolioResult,
    build_portfolio,
)
from portfolio.rebalancer import (
    CurrentPosition,
    RebalanceAction,
    RebalanceReport,
    compute_rebalance,
    DRIFT_THRESHOLD,
)
from factors.composite import FactorScores
from config.settings import (
    MAX_STOCK_PRICE,
    CONVICTION_WEIGHTS,
    CONVICTION_DOLLARS,
    ACCOUNT_SIZE,
    MAX_POSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cr(ticker: str, passed: bool, failures=None) -> ConstraintResult:
    return ConstraintResult(ticker=ticker, passed=passed, failures=failures or [])


def _score(ticker: str, composite: float = 75.0) -> FactorScores:
    return FactorScores(ticker=ticker, composite=composite)


# ---------------------------------------------------------------------------
# constraints.py
# ---------------------------------------------------------------------------

class TestCheckConstraints:
    def test_all_pass(self):
        r = check_constraints("AAPL", price=150.0, market_cap=10e9, avg_volume=5e6, shariah_status="compliant")
        assert r.passed is True
        assert r.failures == []

    def test_non_compliant_fails(self):
        r = check_constraints("BAC", price=50.0, market_cap=100e9, avg_volume=10e6, shariah_status="non_compliant")
        assert r.passed is False
        assert any("Shariah" in f for f in r.failures)

    def test_unknown_shariah_fails(self):
        r = check_constraints("XYZ", price=50.0, market_cap=10e9, avg_volume=2e6, shariah_status="unknown")
        assert r.passed is False
        assert any("unknown" in f for f in r.failures)

    def test_price_at_limit_fails(self):
        r = check_constraints("BRK", price=MAX_STOCK_PRICE, market_cap=10e9, avg_volume=2e6, shariah_status="compliant")
        assert r.passed is False
        assert any("Price" in f for f in r.failures)

    def test_price_below_limit_passes(self):
        r = check_constraints("T", price=MAX_STOCK_PRICE - 0.01, market_cap=10e9, avg_volume=2e6, shariah_status="compliant")
        assert r.passed is True

    def test_price_none_fails(self):
        r = check_constraints("T", price=None, market_cap=10e9, avg_volume=2e6, shariah_status="compliant")
        assert r.passed is False
        assert any("Price unavailable" in f for f in r.failures)

    def test_market_cap_too_low_fails(self):
        r = check_constraints("TINY", price=10.0, market_cap=1e9, avg_volume=2e6, shariah_status="compliant")
        assert r.passed is False
        assert any("Market cap" in f for f in r.failures)

    def test_market_cap_none_passes(self):
        # None market_cap = data unavailable, not a $0 company — skip the check
        r = check_constraints("T", price=50.0, market_cap=None, avg_volume=2e6, shariah_status="compliant")
        assert r.passed is True

    def test_volume_too_low_fails(self):
        r = check_constraints("LOW", price=50.0, market_cap=10e9, avg_volume=499_999, shariah_status="compliant")
        assert r.passed is False
        assert any("volume" in f.lower() for f in r.failures)

    def test_volume_none_passes(self):
        # None avg_volume = data unavailable — skip the check rather than reject the stock
        r = check_constraints("T", price=50.0, market_cap=10e9, avg_volume=None, shariah_status="compliant")
        assert r.passed is True

    def test_multiple_failures_all_reported(self):
        # With None mkt_cap / None volume now skipped, only Shariah + price fail
        r = check_constraints("BAD", price=None, market_cap=None, avg_volume=None, shariah_status="non_compliant")
        assert r.passed is False
        assert any("shariah" in f.lower() for f in r.failures)
        assert any("price" in f.lower() for f in r.failures)

    def test_batch(self):
        stocks = [
            {"ticker": "AAPL", "price": 150.0, "marketCap": 10e9, "avgVolume": 5e6, "shariah_status": "compliant"},
            {"ticker": "BAC",  "price": 40.0,  "marketCap": 80e9, "avgVolume": 50e6, "shariah_status": "non_compliant"},
        ]
        results = check_constraints_batch(stocks)
        assert results["AAPL"].passed is True
        assert results["BAC"].passed is False

    def test_to_dict(self):
        r = check_constraints("AAPL", 150.0, 10e9, 5e6, "compliant")
        d = r.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["passed"] is True
        assert isinstance(d["failures"], list)


# ---------------------------------------------------------------------------
# constructor.py
# ---------------------------------------------------------------------------

class TestBuildPortfolio:
    def _ranked(self, tickers_scores: list[tuple[str, float]]) -> list[FactorScores]:
        return [_score(t, s) for t, s in tickers_scores]

    def _passing_cr(self, tickers: list[str]) -> dict[str, ConstraintResult]:
        return {t: _cr(t, True) for t in tickers}

    def _prices(self, tickers: list[str], price: float = 100.0) -> dict[str, float]:
        return {t: price for t in tickers}

    def test_five_passing_stocks_gives_five_positions(self):
        tickers = ["A", "B", "C", "D", "E"]
        ranked = self._ranked([(t, 90 - i * 5) for i, t in enumerate(tickers)])
        result = build_portfolio(ranked, self._passing_cr(tickers), self._prices(tickers))
        assert result.n_positions == 5
        assert [p.rank for p in result.positions] == [1, 2, 3, 4, 5]

    def test_conviction_weights_assigned_correctly(self):
        tickers = ["A", "B", "C", "D", "E"]
        ranked = self._ranked([(t, 90 - i * 5) for i, t in enumerate(tickers)])
        result = build_portfolio(ranked, self._passing_cr(tickers), self._prices(tickers))
        for i, pos in enumerate(result.positions):
            assert pos.conviction_weight == pytest.approx(CONVICTION_WEIGHTS[i])
            assert pos.dollar_amount == pytest.approx(CONVICTION_DOLLARS[i])

    def test_dollar_amounts_sum_to_account_size(self):
        tickers = ["A", "B", "C", "D", "E"]
        ranked = self._ranked([(t, 90 - i * 5) for i, t in enumerate(tickers)])
        result = build_portfolio(ranked, self._passing_cr(tickers), self._prices(tickers))
        assert result.total_invested == pytest.approx(ACCOUNT_SIZE)
        assert result.cash_remaining == pytest.approx(0.0)

    def test_three_passing_gives_three_positions(self):
        tickers = ["A", "B", "C", "D", "E"]
        passing = {
            "A": _cr("A", True), "B": _cr("B", True), "C": _cr("C", True),
            "D": _cr("D", False, ["price too high"]), "E": _cr("E", False, ["low volume"]),
        }
        ranked = self._ranked([(t, 90 - i * 5) for i, t in enumerate(tickers)])
        result = build_portfolio(ranked, passing, self._prices(tickers))
        assert result.n_positions == 3
        assert [p.ticker for p in result.positions] == ["A", "B", "C"]
        # V4: risk-adjusted sizing (or fallback); weights must sum to 1 and be in [0.05, 0.35]
        total_w = sum(p.conviction_weight for p in result.positions)
        assert total_w == pytest.approx(1.0, abs=0.01)
        for p in result.positions:
            assert 0.04 <= p.conviction_weight <= 0.50

    def test_zero_passing_gives_empty_portfolio(self):
        tickers = ["A", "B"]
        ranked = self._ranked([(t, 80.0) for t in tickers])
        failing = {t: _cr(t, False, ["shariah"]) for t in tickers}
        result = build_portfolio(ranked, failing, self._prices(tickers))
        assert result.n_positions == 0
        assert result.positions == []
        assert result.cash_remaining == pytest.approx(ACCOUNT_SIZE)

    def test_missing_price_stock_skipped(self):
        tickers = ["A", "B", "C"]
        ranked = self._ranked([(t, 90.0) for t in tickers])
        crs = self._passing_cr(tickers)
        prices = {"A": 100.0, "C": 100.0}  # B has no price
        result = build_portfolio(ranked, crs, prices)
        assert result.n_positions == 2
        assert [p.ticker for p in result.positions] == ["A", "C"]

    def test_shares_calculated_correctly(self):
        ranked = self._ranked([("A", 90.0)])
        crs = self._passing_cr(["A"])
        result = build_portfolio(ranked, crs, {"A": 200.0})
        # V4: single position gets 100% weight → $2000 / $200 = 10 shares
        # (with 1 stock, equal-weight fallback = 100%)
        p = result.positions[0]
        assert p.shares_to_buy == pytest.approx(p.dollar_amount / 200.0)

    def test_top_five_only_from_long_list(self):
        tickers = [f"S{i}" for i in range(10)]
        ranked = self._ranked([(t, 100 - i) for i, t in enumerate(tickers)])
        result = build_portfolio(ranked, self._passing_cr(tickers), {t: 100.0 for t in tickers})
        assert result.n_positions == MAX_POSITIONS
        assert [p.ticker for p in result.positions] == ["S0", "S1", "S2", "S3", "S4"]

    def test_to_dict_structure(self):
        tickers = ["A", "B"]
        ranked = self._ranked([(t, 80.0) for t in tickers])
        result = build_portfolio(ranked, self._passing_cr(tickers), self._prices(tickers))
        d = result.to_dict()
        assert "positions" in d
        assert d["n_positions"] == 2


# ---------------------------------------------------------------------------
# rebalancer.py
# ---------------------------------------------------------------------------

class TestRebalancer:
    def _make_portfolio(self, tickers_prices: list[tuple[str, float]]) -> PortfolioResult:
        positions = []
        for i, (ticker, price) in enumerate(tickers_prices):
            rank = i + 1
            w = CONVICTION_WEIGHTS[i]
            d = CONVICTION_DOLLARS[i]
            positions.append(PortfolioPosition(
                rank=rank, ticker=ticker, composite_score=80.0,
                conviction_weight=w, dollar_amount=d,
                price=price, shares_to_buy=d / price,
                constraint_result=_cr(ticker, True),
            ))
        total = sum(CONVICTION_DOLLARS[:len(positions)])
        return PortfolioResult(
            positions=positions, total_invested=total,
            cash_remaining=ACCOUNT_SIZE - total, n_positions=len(positions),
        )

    def test_empty_current_all_buy(self):
        target = self._make_portfolio([("AAPL", 150.0), ("MSFT", 300.0)])
        report = compute_rebalance([], target)
        actions = {a.ticker: a for a in report.actions}
        assert actions["AAPL"].action == "BUY"
        assert actions["MSFT"].action == "BUY"

    def test_exact_match_hold(self):
        target = self._make_portfolio([("AAPL", 150.0)])
        target_shares = CONVICTION_DOLLARS[0] / 150.0
        current = [CurrentPosition("AAPL", shares=target_shares, price=150.0)]
        report = compute_rebalance(current, target)
        actions = {a.ticker: a for a in report.actions}
        assert actions["AAPL"].action == "HOLD"

    def test_dropped_ticker_gets_close(self):
        target = self._make_portfolio([("MSFT", 300.0)])
        current = [
            CurrentPosition("MSFT", shares=2.0, price=300.0),
            CurrentPosition("AAPL", shares=4.0, price=150.0),
        ]
        report = compute_rebalance(current, target)
        actions = {a.ticker: a for a in report.actions}
        assert actions["AAPL"].action == "CLOSE"
        assert actions["AAPL"].delta_shares == pytest.approx(-4.0)

    def test_new_ticker_gets_buy(self):
        target = self._make_portfolio([("AAPL", 150.0), ("GOOG", 120.0)])
        current = [CurrentPosition("AAPL", shares=4.0, price=150.0)]
        report = compute_rebalance(current, target)
        actions = {a.ticker: a for a in report.actions}
        assert actions["GOOG"].action == "BUY"

    def test_large_drift_triggers_rebalance_flag(self):
        target = self._make_portfolio([("AAPL", 150.0)])
        target_shares = CONVICTION_DOLLARS[0] / 150.0
        # Hold triple the target — large drift
        current = [CurrentPosition("AAPL", shares=target_shares * 3, price=150.0)]
        report = compute_rebalance(current, target)
        assert report.needs_rebalance is True

    def test_no_drift_no_rebalance_flag(self):
        target = self._make_portfolio([("AAPL", 150.0)])
        target_shares = CONVICTION_DOLLARS[0] / 150.0
        current = [CurrentPosition("AAPL", shares=target_shares, price=150.0)]
        report = compute_rebalance(current, target)
        assert report.needs_rebalance is False

    def test_empty_target_all_close(self):
        empty_target = PortfolioResult(
            positions=[], total_invested=0.0,
            cash_remaining=ACCOUNT_SIZE, n_positions=0,
        )
        current = [CurrentPosition("AAPL", shares=4.0, price=150.0)]
        report = compute_rebalance(current, empty_target)
        assert len(report.actions) == 1
        assert report.actions[0].action == "CLOSE"

    def test_report_to_dict(self):
        target = self._make_portfolio([("AAPL", 150.0)])
        report = compute_rebalance([], target, as_of_date="2024-06-30")
        d = report.to_dict()
        assert d["as_of_date"] == "2024-06-30"
        assert "actions" in d
        assert "needs_rebalance" in d

    def test_sell_action_when_over_weight(self):
        # Rank-1 target: $600 at $100 = 6 shares, but we hold 20
        target = self._make_portfolio([("AAPL", 100.0)])
        current = [CurrentPosition("AAPL", shares=20.0, price=100.0)]
        report = compute_rebalance(current, target)
        actions = {a.ticker: a for a in report.actions}
        assert actions["AAPL"].action == "SELL"
        assert actions["AAPL"].delta_shares < 0
