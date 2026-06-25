"""Tests for portfolio/entry_price.py — Buy Limit Price Engine."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from portfolio.entry_price import (
    EntryPriceResult,
    _compute_atr14,
    _confidence_label,
    _entry_rating,
    compute_buy_limit,
    compute_strong_buy_limit,
    compute_entry_score,
    compute_entry_analysis,
    compute_entry_analyses_for_picks,
    _CONF_MULTIPLIER_DELTA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ea(**overrides) -> EntryPriceResult:
    """Build a compute_entry_analysis call with sensible defaults."""
    kwargs = dict(
        ticker="NEM",
        current_price=100.0,
        valuation_score=60.0,
        composite_score=65.0,
        model_confidence=70.0,
        risk_score=65.0,
        momentum_score=55.0,
        atr14=2.0,
    )
    kwargs.update(overrides)
    return compute_entry_analysis(**kwargs)


# ---------------------------------------------------------------------------
# Confidence label
# ---------------------------------------------------------------------------

class TestConfidenceLabel:
    def test_very_high(self):
        assert _confidence_label(95) == "Very High"

    def test_high(self):
        assert _confidence_label(85) == "High"

    def test_medium(self):
        assert _confidence_label(70) == "Medium"

    def test_low(self):
        assert _confidence_label(55) == "Low"

    def test_very_low(self):
        assert _confidence_label(30) == "Very Low"

    def test_boundary_90(self):
        assert _confidence_label(90) == "Very High"

    def test_boundary_80(self):
        assert _confidence_label(80) == "High"


# ---------------------------------------------------------------------------
# Entry rating
# ---------------------------------------------------------------------------

class TestEntryRating:
    def test_strong_buy(self):
        assert _entry_rating(80) == "Strong Buy"
        assert _entry_rating(95) == "Strong Buy"

    def test_buy(self):
        assert _entry_rating(60) == "Buy"
        assert _entry_rating(79) == "Buy"

    def test_watch(self):
        assert _entry_rating(40) == "Watch"
        assert _entry_rating(59) == "Watch"

    def test_wait(self):
        assert _entry_rating(0)  == "Wait"
        assert _entry_rating(39) == "Wait"


# ---------------------------------------------------------------------------
# Buy limit formula
# ---------------------------------------------------------------------------

class TestComputeBuyLimit:
    def test_base_case_medium_confidence(self):
        """Base multiplier 0.5, no adjustments for medium / neutral inputs."""
        lim = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        # multiplier = 0.5 + 0 (conf) + 0 (val) + 0 (mom) + 0 (risk) = 0.5
        assert lim == pytest.approx(100.0 - 0.5 * 2.0, abs=0.01)

    def test_very_high_confidence_raises_limit(self):
        """Very High confidence → +0.10 on multiplier → smaller pullback needed."""
        lim_vh = compute_buy_limit(100.0, 2.0, "Very High", 60.0, 55.0, 60.0)
        lim_md = compute_buy_limit(100.0, 2.0, "Medium",    60.0, 55.0, 60.0)
        assert lim_vh > lim_md

    def test_very_low_confidence_lowers_limit(self):
        """Very Low confidence → -0.10 on multiplier → require larger pullback."""
        lim_vl = compute_buy_limit(100.0, 2.0, "Very Low", 60.0, 55.0, 60.0)
        lim_md = compute_buy_limit(100.0, 2.0, "Medium",   60.0, 55.0, 60.0)
        assert lim_vl < lim_md

    def test_high_valuation_raises_limit(self):
        """valuation > 80 → stock is cheap → can buy closer to price."""
        lim_cheap = compute_buy_limit(100.0, 2.0, "Medium", 85.0, 55.0, 60.0)
        lim_fair  = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_cheap > lim_fair

    def test_low_valuation_lowers_limit(self):
        """valuation < 40 → stock is expensive → wait for bigger dip."""
        lim_exp  = compute_buy_limit(100.0, 2.0, "Medium", 35.0, 55.0, 60.0)
        lim_fair = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_exp < lim_fair

    def test_high_momentum_raises_limit(self):
        """momentum > 70 → trending → smaller pullback acceptable."""
        lim_hi = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 75.0, 60.0)
        lim_md = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_hi > lim_md

    def test_low_momentum_lowers_limit(self):
        lim_lo = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 35.0, 60.0)
        lim_md = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_lo < lim_md

    def test_high_risk_raises_limit(self):
        """risk_score > 75 → safe balance sheet → tighter limit OK."""
        lim_safe = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 80.0)
        lim_mid  = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_safe > lim_mid

    def test_low_risk_lowers_limit(self):
        """risk_score < 40 → weak balance sheet → deeper discount."""
        lim_risky = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 35.0)
        lim_mid   = compute_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert lim_risky < lim_mid

    def test_always_below_current_price(self):
        """Buy limit must always be below current price."""
        for conf in ("Very High", "High", "Medium", "Low", "Very Low"):
            lim = compute_buy_limit(200.0, 5.0, conf, 90.0, 80.0, 90.0)
            assert lim < 200.0

    def test_multiplier_floor(self):
        """Extremely favourable inputs still produce a discount ≥ 0.10 × ATR."""
        lim = compute_buy_limit(100.0, 2.0, "Very High", 95.0, 85.0, 90.0)
        assert lim <= 100.0 - 0.10 * 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Strong buy limit formula
# ---------------------------------------------------------------------------

class TestComputeStrongBuyLimit:
    def test_deeper_than_buy_limit(self):
        bl  = compute_buy_limit(      100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        sbl = compute_strong_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        assert sbl < bl

    def test_strong_buy_always_below_current(self):
        for conf in ("Very High", "High", "Medium", "Low", "Very Low"):
            sbl = compute_strong_buy_limit(150.0, 4.0, conf, 60.0, 55.0, 60.0)
            assert sbl < 150.0

    def test_base_case_medium_confidence(self):
        sbl = compute_strong_buy_limit(100.0, 2.0, "Medium", 60.0, 55.0, 60.0)
        # base mult = 1.0, no adjustments → 100 - 1.0*2.0 = 98
        assert sbl == pytest.approx(98.0, abs=0.01)


# ---------------------------------------------------------------------------
# Entry score
# ---------------------------------------------------------------------------

class TestComputeEntryScore:
    def test_all_100_gives_100(self):
        score, *_ = compute_entry_score(100, 100, 100, 100)
        assert score == pytest.approx(100.0)

    def test_all_0_gives_0(self):
        score, *_ = compute_entry_score(0, 0, 0, 0)
        assert score == pytest.approx(0.0)

    def test_weights_sum_to_1(self):
        """Weighted average: 40+20+20+20 = 100%."""
        score, *_ = compute_entry_score(100, 0, 0, 0)
        assert score == pytest.approx(40.0)   # val only

    def test_components_returned(self):
        score, val_c, mom_c, conf_c, risk_c = compute_entry_score(80, 60, 70, 65)
        assert val_c  == pytest.approx(80.0)
        assert mom_c  == pytest.approx(60.0)
        assert conf_c == pytest.approx(70.0)
        assert risk_c == pytest.approx(65.0)

    def test_clamps_inputs(self):
        """Out-of-range inputs are clamped, not rejected."""
        score, *_ = compute_entry_score(110, -10, 150, 200)
        assert 0.0 <= score <= 100.0

    def test_valuation_weight_is_highest(self):
        """Valuation (40%) contributes more than any other single factor."""
        s_val, *_  = compute_entry_score(100,  50,  50,  50)
        s_mom, *_  = compute_entry_score( 50, 100,  50,  50)
        assert s_val > s_mom


# ---------------------------------------------------------------------------
# compute_entry_analysis (integration, ATR supplied)
# ---------------------------------------------------------------------------

class TestComputeEntryAnalysis:
    def test_returns_result_object(self):
        ea = _make_ea()
        assert isinstance(ea, EntryPriceResult)

    def test_ticker_set(self):
        ea = _make_ea(ticker="LRCX")
        assert ea.ticker == "LRCX"

    def test_buy_limit_below_price(self):
        ea = _make_ea(current_price=100.0, atr14=2.0)
        assert ea.buy_limit < 100.0

    def test_strong_buy_below_buy(self):
        ea = _make_ea(current_price=100.0, atr14=2.0)
        assert ea.strong_buy_limit < ea.buy_limit

    def test_entry_score_in_range(self):
        ea = _make_ea()
        assert 0.0 <= ea.entry_score <= 100.0

    def test_entry_rating_valid(self):
        ea = _make_ea()
        assert ea.entry_rating in ("Strong Buy", "Buy", "Watch", "Wait")

    def test_explanation_non_empty(self):
        ea = _make_ea()
        assert ea.explanation != ""

    def test_pct_above_buy_positive_when_above(self):
        ea = _make_ea(current_price=100.0, atr14=2.0)
        # price 100 > buy_limit (should be ~99+), so pct_above_buy should be positive
        if ea.buy_limit and ea.current_price > ea.buy_limit:
            assert ea.pct_above_buy is not None
            assert ea.pct_above_buy > 0

    def test_no_atr_gives_none_limits(self):
        """When ATR cannot be fetched, limits are None and explanation says so."""
        with patch("portfolio.entry_price._compute_atr14", return_value=None):
            ea = compute_entry_analysis(
                ticker="NEM", current_price=100.0,
                valuation_score=60, composite_score=65,
                model_confidence=70, risk_score=65, momentum_score=55,
            )
        assert ea.buy_limit is None
        assert ea.strong_buy_limit is None
        assert "ATR unavailable" in ea.explanation

    def test_to_dict_has_all_keys(self):
        ea = _make_ea()
        d = ea.to_dict()
        for key in (
            "ticker", "current_price", "atr14",
            "buy_limit", "strong_buy_limit",
            "entry_score", "entry_rating",
            "valuation_score", "momentum_score",
            "model_confidence", "confidence_label",
            "risk_score", "composite_score",
            "val_component", "mom_component", "conf_component", "risk_component",
            "explanation", "computed_at",
        ):
            assert key in d, f"Missing key in to_dict(): {key}"

    def test_high_confidence_raises_buy_limit(self):
        ea_vh = _make_ea(model_confidence=92)   # Very High
        ea_vl = _make_ea(model_confidence=30)   # Very Low
        assert ea_vh.buy_limit > ea_vl.buy_limit

    def test_high_valuation_raises_buy_limit(self):
        ea_cheap = _make_ea(valuation_score=85)
        ea_exp   = _make_ea(valuation_score=30)
        assert ea_cheap.buy_limit > ea_exp.buy_limit

    def test_high_risk_score_raises_buy_limit(self):
        ea_safe  = _make_ea(risk_score=80)
        ea_risky = _make_ea(risk_score=30)
        assert ea_safe.buy_limit > ea_risky.buy_limit

    def test_strong_buy_is_deeper_than_buy(self):
        for val in (30, 60, 90):
            ea = _make_ea(valuation_score=val)
            if ea.buy_limit and ea.strong_buy_limit:
                assert ea.strong_buy_limit < ea.buy_limit

    def test_computed_at_is_iso_string(self):
        from datetime import datetime
        ea = _make_ea()
        datetime.fromisoformat(ea.computed_at)   # raises ValueError if malformed


# ---------------------------------------------------------------------------
# compute_entry_analyses_for_picks
# ---------------------------------------------------------------------------

class TestComputeEntryAnalysesForPicks:
    _PICKS = [
        {"ticker": "NEM",  "price": 104.0, "valuation": 70.0, "composite": 65.0,
         "model_confidence": 70.0, "risk_adjustment": 65.0, "momentum": 55.0},
        {"ticker": "LRCX", "price": 390.0, "valuation": 55.0, "composite": 72.0,
         "model_confidence": 80.0, "risk_adjustment": 70.0, "momentum": 60.0},
    ]

    def test_returns_dict_keyed_by_ticker(self):
        with patch("portfolio.entry_price._compute_atr14", return_value=3.0):
            results = compute_entry_analyses_for_picks(self._PICKS)
        assert "NEM"  in results
        assert "LRCX" in results

    def test_skips_zero_price(self):
        picks = [{"ticker": "XXX", "price": 0.0}]
        with patch("portfolio.entry_price._compute_atr14", return_value=3.0):
            results = compute_entry_analyses_for_picks(picks)
        assert "XXX" not in results

    def test_skips_empty_ticker(self):
        picks = [{"ticker": "", "price": 100.0}]
        with patch("portfolio.entry_price._compute_atr14", return_value=3.0):
            results = compute_entry_analyses_for_picks(picks)
        assert results == {}

    def test_buy_limits_differ_by_volatility(self):
        """Different ATRs give different limits even at the same price."""
        with patch("portfolio.entry_price._compute_atr14", side_effect=[2.0, 8.0]):
            results = compute_entry_analyses_for_picks(self._PICKS)
        # NEM (ATR=2) should have buy_limit closer to price than LRCX (ATR=8) adjusted
        nem_gap  = self._PICKS[0]["price"] - (results["NEM"].buy_limit  or 0)
        lrcx_gap = self._PICKS[1]["price"] - (results["LRCX"].buy_limit or 0)
        assert lrcx_gap > nem_gap
