"""Tests for portfolio/limit_prices.py."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from portfolio.limit_prices import (
    LimitPriceResult,
    _atr,
    _compute_limits,
    _choose_tier,
    _daily_vol,
    _ma,
    compute_limit_price,
    compute_limit_prices_for_picks,
    DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_ma5_basic(self):
        closes = [10.0, 11.0, 12.0, 13.0, 14.0]
        assert _ma(closes, 5) == pytest.approx(12.0)

    def test_ma_insufficient_data(self):
        assert _ma([10.0, 11.0], 5) is None

    def test_ma20_uses_last_20(self):
        closes = list(range(1, 31))   # 1..30
        result = _ma(closes, 20)
        expected = sum(range(11, 31)) / 20
        assert result == pytest.approx(expected)


class TestATR:
    def _make_bars(self, n=20, h=105.0, l=95.0, c=100.0):
        highs  = [h] * n
        lows   = [l] * n
        closes = [c] * (n + 1)
        return highs, lows, closes

    def test_atr_constant_range(self):
        highs, lows, closes = self._make_bars(20, h=105.0, l=95.0, c=100.0)
        result = _atr(highs, lows, closes, n=14)
        assert result is not None
        assert result == pytest.approx(10.0, abs=0.1)

    def test_atr_insufficient_data(self):
        assert _atr([105.0]*5, [95.0]*5, [100.0]*6, n=14) is None

    def test_atr_positive(self):
        import random
        rng = random.Random(42)
        closes = [100.0 + rng.gauss(0, 2) for _ in range(22)]
        highs  = [c + abs(rng.gauss(0, 1)) for c in closes[1:]]
        lows   = [c - abs(rng.gauss(0, 1)) for c in closes[1:]]
        result = _atr(highs, lows, closes, n=14)
        assert result is not None
        assert result > 0


class TestDailyVol:
    def test_constant_prices_zero_vol(self):
        closes = [100.0] * 22
        result = _daily_vol(closes, n=20)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_vol_is_positive(self):
        import random
        rng = random.Random(7)
        closes = [100.0 * math.exp(rng.gauss(0, 0.02)) for _ in range(22)]
        result = _daily_vol(closes, n=20)
        assert result is not None
        assert result > 0

    def test_insufficient_data(self):
        assert _daily_vol([100.0]*5, n=20) is None


class TestComputeLimits:
    def test_conservative_below_normal_below_aggressive(self):
        price = 100.0
        c, n, a = _compute_limits(price, bid=99.80, ask=100.20, atr=2.0)
        assert c <= n <= a

    def test_conservative_uses_atr_when_lower(self):
        price = 100.0
        atr   = 10.0  # large ATR → 0.25 × 10 = 2.5 deduction → 97.5
        pct   = price * 0.995  # 99.5
        c, _, _ = _compute_limits(price, bid=None, ask=None, atr=atr)
        assert c <= pct   # ATR path is lower

    def test_normal_uses_midpoint_when_available(self):
        price = 100.0
        bid, ask = 99.80, 100.00
        midpoint = (bid + ask) / 2   # 99.9
        pct_normal = price * 0.9975  # 99.75
        _, n, _ = _compute_limits(price, bid=bid, ask=ask, atr=None)
        assert n == pytest.approx(min(pct_normal, midpoint))

    def test_aggressive_is_ask_when_available(self):
        price = 100.0
        ask   = 100.50
        _, _, a = _compute_limits(price, bid=99.80, ask=ask, atr=None)
        assert a == pytest.approx(ask)

    def test_aggressive_fallback_to_price_when_no_ask(self):
        price = 100.0
        _, _, a = _compute_limits(price, bid=None, ask=None, atr=None)
        assert a == pytest.approx(price)

    def test_all_limits_below_price_times_1_01(self):
        price = 200.0
        c, n, a = _compute_limits(price, bid=199.50, ask=200.50, atr=3.0)
        assert a <= price * 1.01 + 1e-6


class TestChooseTier:
    def test_high_vol_returns_conservative(self):
        tier, expl, warn = _choose_tier(100.0, 99.9, 100.1, ma5=100.0, daily_vol=0.03, spread_pct=0.001)
        assert tier == "conservative"
        assert warn == ""

    def test_tight_spread_no_fast_market_returns_normal(self):
        tier, expl, warn = _choose_tier(100.0, 99.95, 100.05, ma5=100.0, daily_vol=0.005, spread_pct=0.0005)
        assert tier == "normal"
        assert warn == ""

    def test_fast_market_returns_aggressive(self):
        # price > ma5 * 1.005
        tier, expl, warn = _choose_tier(101.0, 100.90, 101.10, ma5=100.0, daily_vol=0.005, spread_pct=0.001)
        assert tier == "aggressive"
        assert "FAST MARKET" in warn

    def test_no_signal_defaults_to_normal(self):
        tier, expl, warn = _choose_tier(100.0, bid=None, ask=None, ma5=None, daily_vol=None, spread_pct=None)
        assert tier == "normal"
        assert warn == ""


# ---------------------------------------------------------------------------
# Integration tests — compute_limit_price (mocked fetch)
# ---------------------------------------------------------------------------

_FAKE_RAW = {
    "price":      100.0,
    "bid":        99.80,
    "ask":        100.20,
    "prev_close": 99.50,
    "closes":     [98.0 + i * 0.1 for i in range(25)],
    "highs":      [98.5 + i * 0.1 for i in range(24)],
    "lows":       [97.5 + i * 0.1 for i in range(24)],
    "timestamp":  "2099-01-01T12:00:00",   # far future → never stale in tests
}


class TestComputeLimitPrice:
    def test_returns_limit_price_result(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        assert isinstance(result, LimitPriceResult)

    def test_suggested_limit_is_positive(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        assert result.suggested_limit is not None
        assert result.suggested_limit > 0

    def test_shares_equals_allocation_over_limit(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        if result.suggested_limit and result.shares:
            assert result.shares == pytest.approx(600.0 / result.suggested_limit, rel=1e-6)

    def test_estimated_fill_cost_close_to_allocation(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        if result.estimated_fill_cost:
            # should be within a few percent of the allocation
            assert abs(result.estimated_fill_cost - 600.0) / 600.0 < 0.05

    def test_disclaimer_set(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        assert result.disclaimer == DISCLAIMER

    def test_stale_data_returns_stale_flag(self):
        stale_raw = {**_FAKE_RAW, "timestamp": "2000-01-01T00:00:00"}
        result = compute_limit_price("NEM", 600.0, live_price_data=stale_raw)
        assert result.stale is True
        assert result.suggested_limit is None

    def test_no_price_returns_error(self):
        bad_raw = {**_FAKE_RAW, "price": None}
        result = compute_limit_price("NEM", 600.0, live_price_data=bad_raw)
        assert result.suggested_limit is None
        assert result.error != "" or result.stale

    def test_conservative_lte_normal_lte_aggressive(self):
        result = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW)
        if result.conservative_limit and result.normal_limit and result.aggressive_limit:
            assert result.conservative_limit <= result.normal_limit + 1e-6
            assert result.normal_limit <= result.aggressive_limit + 1e-6

    def test_to_dict_has_required_keys(self):
        d = compute_limit_price("NEM", 600.0, live_price_data=_FAKE_RAW).to_dict()
        for key in (
            "ticker", "live_price", "suggested_limit", "suggested_tier",
            "conservative_limit", "normal_limit", "aggressive_limit",
            "shares", "estimated_fill_cost", "explanation", "disclaimer",
        ):
            assert key in d, f"Missing key: {key}"

    def test_zero_allocation_gives_zero_shares(self):
        result = compute_limit_price("NEM", 0.0, live_price_data=_FAKE_RAW)
        assert result.shares is None or result.shares == pytest.approx(0.0, abs=1e-9)


class TestComputeLimitPricesForPicks:
    def test_returns_dict_keyed_by_ticker(self):
        picks = [
            {"ticker": "NEM", "dollar_amount": 600.0},
            {"ticker": "APA", "dollar_amount": 500.0},
        ]
        with patch("portfolio.limit_prices._fetch_price_data", return_value=_FAKE_RAW):
            results = compute_limit_prices_for_picks(picks)
        assert "NEM" in results
        assert "APA" in results

    def test_skips_picks_without_ticker(self):
        picks = [{"ticker": "", "dollar_amount": 600.0}]
        with patch("portfolio.limit_prices._fetch_price_data", return_value=_FAKE_RAW):
            results = compute_limit_prices_for_picks(picks)
        assert results == {}

    def test_allocation_flows_through(self):
        picks = [{"ticker": "NEM", "dollar_amount": 1200.0}]
        with patch("portfolio.limit_prices._fetch_price_data", return_value=_FAKE_RAW):
            results = compute_limit_prices_for_picks(picks)
        assert results["NEM"].allocation == pytest.approx(1200.0)
