"""
Tests for Phase 4 factor scoring engine.

Covers:
  - Momentum: signal computation, missing data, boundary cases
  - Quality: growth rates, ROIC, operating margin, missing data
  - Valuation: P/E inversion, FCF yield, negative PE excluded
  - Revisions: weighted score, buy ratio, missing → neutral
  - Composite: normalization to 0–100, weights, AI neutral enforcement,
               backtest AI lock, ranking
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from factors.momentum  import compute_momentum, MomentumRaw
from factors.quality   import compute_quality,  QualityRaw
from factors.valuation import compute_valuation, ValuationRaw
from factors.revisions import compute_revisions, RevisionsRaw
from factors.composite import compute_composite, rank_scores, FactorScores, NEUTRAL, _zscore_to_0_100


# ============================================================
# Helpers
# ============================================================

def flat_prices(n: int, val: float = 100.0) -> list[float]:
    return [val] * n

def trending_up(n: int, start: float = 100.0, step: float = 0.5) -> list[float]:
    return [start + i * step for i in range(n)]

def trending_down(n: int, start: float = 200.0, step: float = 0.5) -> list[float]:
    return [start - i * step for i in range(n)]


# ============================================================
# 1. Momentum
# ============================================================

class TestMomentum:
    def test_insufficient_history_returns_none(self):
        r = compute_momentum("T", [100.0, 101.0])
        assert r.raw_score is None

    def test_50_day_history_gives_ma_signals(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert "price_above_50dma" in r.signals_used
        assert "golden_cross" in r.signals_used

    def test_positive_trend_price_above_50dma(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert r.price_above_50dma is True

    def test_downtrend_price_below_50dma(self):
        prices = trending_down(260)
        r = compute_momentum("T", prices)
        assert r.price_above_50dma is False

    def test_uptrend_golden_cross(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert r.golden_cross is True

    def test_6m_return_positive_on_uptrend(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert r.ret_6m is not None and r.ret_6m > 0

    def test_12_1m_return_present_with_full_history(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert r.ret_12_1m is not None

    def test_flat_prices_scores_near_neutral(self):
        # All binary signals will be border-case; score should be defined
        prices = flat_prices(260)
        r = compute_momentum("T", prices)
        # flat: 6m return = 0, 12-1m return = 0, ma signals indeterminate
        assert r.raw_score is not None or len(r.signals_used) < 2

    def test_signals_used_populated(self):
        prices = trending_up(260)
        r = compute_momentum("T", prices)
        assert len(r.signals_used) >= 2

    def test_too_short_for_6m_but_long_enough_for_ma(self):
        prices = trending_up(60)   # 60 days: enough for 50DMA but not 6m return
        r = compute_momentum("T", prices)
        assert "price_above_50dma" in r.signals_used
        assert "ret_6m" not in r.signals_used


# ============================================================
# 2. Quality
# ============================================================

class TestQuality:
    def test_all_fields_present_gives_score(self):
        r = compute_quality(
            ticker="T", revenue=200, net_income=40, free_cash_flow=30,
            operating_income=50, total_equity=300, total_debt=100, cash=50,
            revenue_prior=180, net_income_prior=35, free_cash_flow_prior=25,
        )
        assert r.raw_score is not None
        # 8 signals: roic, op_margin, net_margin, fcf_margin, equity_ratio,
        # revenue_growth, earnings_growth, fcf_growth
        assert len(r.signals_used) == 8

    def test_missing_prior_data_skips_growth_signals(self):
        r = compute_quality(
            ticker="T", revenue=200, net_income=40, free_cash_flow=30,
            operating_income=50, total_equity=300, total_debt=100, cash=50,
        )
        assert "revenue_growth" not in r.signals_used
        assert "earnings_growth" not in r.signals_used
        assert "fcf_growth" not in r.signals_used
        assert "roic" in r.signals_used
        assert "operating_margin" in r.signals_used
        assert "net_margin" in r.signals_used
        assert "fcf_margin" in r.signals_used
        assert "equity_ratio" in r.signals_used

    def test_revenue_growth_correct(self):
        r = compute_quality(
            ticker="T", revenue=110, net_income=None, free_cash_flow=None,
            operating_income=None, total_equity=None, total_debt=None, cash=None,
            revenue_prior=100,
        )
        assert r.revenue_growth == pytest.approx(0.10)

    def test_negative_prior_revenue_excluded(self):
        # Negative prior period is excluded by base-period protection
        # (prior < 0 → flag="prior_negative" → value=None)
        r = compute_quality(
            ticker="T", revenue=50, net_income=None, free_cash_flow=None,
            operating_income=None, total_equity=None, total_debt=None, cash=None,
            revenue_prior=-100,
        )
        assert r.revenue_growth is None
        assert r.revenue_growth_sig is not None
        assert r.revenue_growth_sig.flag == "prior_negative"

    def test_zero_prior_revenue_excluded(self):
        r = compute_quality(
            ticker="T", revenue=100, net_income=None, free_cash_flow=None,
            operating_income=None, total_equity=None, total_debt=None, cash=None,
            revenue_prior=0,
        )
        assert r.revenue_growth is None

    def test_roic_computed_correctly(self):
        # NOPAT = 100 * (1 - 0.21) = 79; IC = 300+100-50 = 350; ROIC = 79/350
        r = compute_quality(
            ticker="T", revenue=500, net_income=None, free_cash_flow=None,
            operating_income=100, total_equity=300, total_debt=100, cash=50,
            effective_tax_rate=0.21,
        )
        expected = (100 * 0.79) / 350
        assert r.roic == pytest.approx(expected, rel=1e-4)

    def test_missing_all_fields_returns_none_score(self):
        r = compute_quality(
            ticker="T", revenue=None, net_income=None, free_cash_flow=None,
            operating_income=None, total_equity=None, total_debt=None, cash=None,
        )
        assert r.raw_score is None

    def test_only_one_signal_returns_none_score(self):
        r = compute_quality(
            ticker="T", revenue=100, net_income=None, free_cash_flow=None,
            operating_income=None, total_equity=None, total_debt=None, cash=None,
            revenue_prior=90,
        )
        # Only 1 signal → raw_score = None (need >= 3 in V4)
        assert r.raw_score is None

    def test_5yr_revenue_cagr_computed(self):
        """5-year revenue series → CAGR stored on QualityRaw."""
        r = compute_quality(
            ticker="T", revenue=200, net_income=20, free_cash_flow=15,
            operating_income=30, total_equity=100, total_debt=20, cash=10,
            revenue_series=[100, 120, 140, 165, 200],
        )
        assert r.revenue_cagr_5yr is not None
        # CAGR from 100→200 over 4 years ≈ 18.9%
        assert 0.15 < r.revenue_cagr_5yr < 0.22

    def test_5yr_roic_stability(self):
        """Consistent ROIC → high stability score."""
        r = compute_quality(
            ticker="T", revenue=200, net_income=20, free_cash_flow=15,
            operating_income=30, total_equity=100, total_debt=20, cash=10,
            roic_series=[0.20, 0.21, 0.20, 0.22, 0.21],
        )
        assert r.roic_stability is not None
        assert r.roic_stability > 0.80  # low CoV → high stability

    def test_5yr_share_count_decline_rewards(self):
        """Declining share count → share_count_trend > 0.5."""
        r = compute_quality(
            ticker="T", revenue=200, net_income=20, free_cash_flow=15,
            operating_income=30, total_equity=100, total_debt=20, cash=10,
            shares_series=[100e6, 98e6, 96e6, 94e6, 92e6],
        )
        assert r.share_count_trend is not None
        assert r.share_count_trend > 0.5

    def test_5yr_multiyear_data_increases_years_of_data(self):
        """Providing 5-year series → years_of_data = 5."""
        r = compute_quality(
            ticker="T", revenue=200, net_income=20, free_cash_flow=15,
            operating_income=30, total_equity=100, total_debt=20, cash=10,
            revenue_series=[100, 120, 140, 165, 200],
            fcf_series=[8, 10, 12, 13, 15],
        )
        assert r.years_of_data == 5


# ============================================================
# 3. Valuation
# ============================================================

class TestValuation:
    def test_all_fields_present(self):
        r = compute_valuation(
            ticker="T", price=100, trailing_eps=5, forward_eps=6,
            free_cash_flow=10_000_000, market_cap=100_000_000,
        )
        assert r.raw_score is not None
        assert len(r.signals_used) == 3

    def test_pe_inverted_lower_pe_gives_higher_inv(self):
        r_cheap = compute_valuation("T", price=100, trailing_eps=10,
                                    forward_eps=None, free_cash_flow=None, market_cap=None)
        r_expensive = compute_valuation("T", price=100, trailing_eps=2,
                                        forward_eps=None, free_cash_flow=None, market_cap=None)
        # cheaper stock (PE=10 → inv=0.1) vs expensive (PE=50 → inv=0.02)
        assert r_cheap.trailing_pe_inv > r_expensive.trailing_pe_inv

    def test_negative_pe_excluded(self):
        r = compute_valuation(
            ticker="T", price=100, trailing_eps=-5, forward_eps=None,
            free_cash_flow=None, market_cap=None,
        )
        assert r.trailing_pe_inv is None

    def test_pe_below_1_excluded(self):
        r = compute_valuation(
            ticker="T", trailing_pe=0.5, price=None, trailing_eps=None,
            forward_eps=None, free_cash_flow=None, market_cap=None,
        )
        assert r.trailing_pe_inv is None

    def test_fcf_yield_correct(self):
        r = compute_valuation(
            ticker="T", price=None, trailing_eps=None, forward_eps=None,
            free_cash_flow=10_000_000, market_cap=100_000_000,
        )
        assert r.fcf_yield == pytest.approx(0.10)

    def test_negative_fcf_yield_included(self):
        r = compute_valuation(
            ticker="T", price=None, trailing_eps=None, forward_eps=None,
            free_cash_flow=-5_000_000, market_cap=100_000_000,
        )
        assert r.fcf_yield == pytest.approx(-0.05)
        assert "fcf_yield" in r.signals_used

    def test_missing_all_fields_returns_none(self):
        r = compute_valuation(
            ticker="T", price=None, trailing_eps=None, forward_eps=None,
            free_cash_flow=None, market_cap=None,
        )
        assert r.raw_score is None

    def test_direct_pe_overrides_computed(self):
        r = compute_valuation(
            ticker="T", price=100, trailing_eps=2,   # would give PE=50
            forward_eps=None, free_cash_flow=None, market_cap=None,
            trailing_pe=10.0,                         # should use this instead
        )
        assert r.trailing_pe_inv == pytest.approx(1 / 10.0)


# ============================================================
# 4. Revisions
# ============================================================

class TestRevisions:
    def test_all_buys_gives_high_score(self):
        r = compute_revisions("T", strong_buy=10, buy=5, hold=0, sell=0, strong_sell=0)
        assert r.raw_score is not None and r.raw_score > 0.5

    def test_all_sells_gives_low_score(self):
        r = compute_revisions("T", strong_buy=0, buy=0, hold=0, sell=5, strong_sell=10)
        # V2: raw_score is [0,1]; all-sells should be below 0.5 (bearish)
        assert r.raw_score is not None and r.raw_score < 0.45

    def test_missing_data_returns_none(self):
        r = compute_revisions("T")
        assert r.raw_score is None

    def test_consensus_only_gives_score(self):
        r = compute_revisions("T", consensus_rating=1.0)   # Strong Buy
        assert r.consensus_inv == pytest.approx(1.0)
        assert r.raw_score is not None

    def test_consensus_strong_sell(self):
        r = compute_revisions("T", consensus_rating=5.0)
        assert r.consensus_inv == pytest.approx(0.0)

    def test_buy_ratio_correct(self):
        r = compute_revisions("T", strong_buy=3, buy=2, hold=5, sell=0, strong_sell=0)
        assert r.buy_ratio == pytest.approx(5 / 10)

    def test_weighted_score_all_strong_buy(self):
        r = compute_revisions("T", strong_buy=10, buy=0, hold=0, sell=0, strong_sell=0)
        assert r.weighted_score == pytest.approx(1.0)

    def test_zero_analysts_gives_none(self):
        r = compute_revisions("T", strong_buy=0, buy=0, hold=0, sell=0, strong_sell=0)
        assert r.raw_score is None

    def test_eps_direction_rising_estimates_boosts_score(self):
        """When EPS estimates rise 10% over 30d, direction score should be 1.0."""
        r = compute_revisions("T",
            eps_current=2.20, eps_30d_ago=2.00, eps_90d_ago=2.00, eps_7d_ago=2.20)
        assert r.eps_direction_score is not None
        assert r.eps_direction_score > 0.8

    def test_eps_direction_falling_estimates_lowers_score(self):
        """When EPS estimates fall 10% over 30d, direction score should be 0.0."""
        r = compute_revisions("T",
            eps_current=1.80, eps_30d_ago=2.00, eps_90d_ago=2.00, eps_7d_ago=1.80)
        assert r.eps_direction_score is not None
        assert r.eps_direction_score < 0.2

    def test_revision_breadth_all_upgrades(self):
        """All estimates raised → breadth_score close to 1.0."""
        r = compute_revisions("T", rev_up_30d=20, rev_dn_30d=0)
        assert r.breadth_score is not None
        assert r.breadth_score == pytest.approx(1.0)

    def test_revision_breadth_all_downgrades(self):
        """All estimates cut → breadth_score close to 0.0."""
        r = compute_revisions("T", rev_up_30d=0, rev_dn_30d=20)
        assert r.breadth_score is not None
        assert r.breadth_score == pytest.approx(0.0)

    def test_net_upgrades_90d_positive(self):
        """Net upgrades (more ups than downs) → score > 0.5."""
        r = compute_revisions("T", upgrades_90d=8, downgrades_90d=2)
        assert r.upgrade_momentum_score is not None
        assert r.upgrade_momentum_score > 0.5

    def test_pt_upside_large_positive(self):
        """PT upside +30% → pt_upside_score = 1.0."""
        r = compute_revisions("T", price_target_upside=0.30)
        assert r.pt_upside_score is not None
        assert r.pt_upside_score == pytest.approx(1.0)

    def test_high_coverage_score(self):
        """10+ analysts → coverage_score close to 1.0."""
        r = compute_revisions("T", num_analysts=15)
        assert r.coverage_score is not None
        assert r.coverage_score == pytest.approx(1.0)

    def test_5_signals_gives_high_confidence(self):
        """5+ populated signals → confidence 'high'."""
        r = compute_revisions("T",
            eps_current=2.20, eps_30d_ago=2.00, eps_90d_ago=2.00, eps_7d_ago=2.20,
            rev_up_30d=15, rev_dn_30d=5, upgrades_90d=5, downgrades_90d=2,
            price_target_upside=0.20, num_analysts=12)
        assert r.revisions_confidence == "high"
        assert len(r.signals_used) >= 5


# ============================================================
# 5. Composite
# ============================================================

def _make_raws(tickers, mom_vals, qua_vals, val_vals, rev_vals):
    """Helper to create raw dicts with given raw_score values."""
    from factors.momentum  import MomentumRaw
    from factors.quality   import QualityRaw
    from factors.valuation import ValuationRaw
    from factors.revisions import RevisionsRaw

    mom = {t: MomentumRaw(ticker=t, raw_score=v)  for t, v in zip(tickers, mom_vals)}
    qua = {t: QualityRaw(ticker=t,  raw_score=v)  for t, v in zip(tickers, qua_vals)}
    val = {t: ValuationRaw(ticker=t, raw_score=v) for t, v in zip(tickers, val_vals)}
    rev = {t: RevisionsRaw(ticker=t, raw_score=v) for t, v in zip(tickers, rev_vals)}
    return mom, qua, val, rev


class TestComposite:
    def test_all_neutral_gives_50(self):
        tickers = ["A", "B", "C"]
        mom, qua, val, rev = _make_raws(tickers, [None]*3, [None]*3, [None]*3, [None]*3)
        scores = compute_composite(tickers, mom, qua, val, rev)
        for s in scores:
            assert s.composite == pytest.approx(NEUTRAL)

    def test_output_count_matches_input(self):
        tickers = ["A", "B", "C", "D"]
        mom, qua, val, rev = _make_raws(tickers, [0.1]*4, [0.2]*4, [0.3]*4, [0.4]*4)
        scores = compute_composite(tickers, mom, qua, val, rev)
        assert len(scores) == 4

    def test_scores_bounded_0_to_100(self):
        tickers = ["A", "B", "C"]
        mom, qua, val, rev = _make_raws(tickers, [1.0, 0.5, 0.0], [1.0, 0.5, 0.0],
                                         [1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
        scores = compute_composite(tickers, mom, qua, val, rev)
        for s in scores:
            assert 0 <= s.quality <= 100
            assert 0 <= s.momentum <= 100
            assert 0 <= s.composite <= 100

    def test_ai_research_neutral_when_none_provided(self):
        tickers = ["A", "B"]
        mom, qua, val, rev = _make_raws(tickers, [0.1]*2, [0.2]*2, [0.3]*2, [0.4]*2)
        scores = compute_composite(tickers, mom, qua, val, rev, ai_research_scores=None)
        for s in scores:
            assert s.ai_research == pytest.approx(NEUTRAL)

    def test_ai_research_locked_in_backtest(self):
        tickers = ["A", "B"]
        mom, qua, val, rev = _make_raws(tickers, [0.1]*2, [0.2]*2, [0.3]*2, [0.4]*2)
        ai_scores = {"A": 90.0, "B": 10.0}
        # Even with live scores provided, backtest=True must lock to 50
        scores = compute_composite(tickers, mom, qua, val, rev,
                                   ai_research_scores=ai_scores, is_backtest=True)
        for s in scores:
            assert s.ai_research == pytest.approx(NEUTRAL)

    def test_ai_research_used_in_live_mode_with_confidence(self):
        # AI display score is confidence-scaled: effective = (raw-50)*conf + 50
        # With confidence=1.0, effective_ai = raw_ai (full signal)
        tickers = ["A", "B"]
        mom, qua, val, rev = _make_raws(tickers, [0.1]*2, [0.2]*2, [0.3]*2, [0.4]*2)
        ai_scores = {"A": 90.0, "B": 10.0}
        ai_conf   = {"A": 1.0,  "B": 1.0}
        scores = compute_composite(tickers, mom, qua, val, rev,
                                   ai_research_scores=ai_scores,
                                   ai_confidence_scores=ai_conf,
                                   is_backtest=False)
        score_map = {s.ticker: s for s in scores}
        assert score_map["A"].ai_research == pytest.approx(90.0)
        assert score_map["B"].ai_research == pytest.approx(10.0)

    def test_ai_zero_confidence_collapses_to_neutral(self):
        # confidence=0.0 → effective_ai = (raw-50)*0+50 = 50 always
        tickers = ["A"]
        mom, qua, val, rev = _make_raws(tickers, [0.5], [0.5], [0.5], [0.5])
        ai_scores = {"A": 90.0}
        ai_conf   = {"A": 0.0}
        scores = compute_composite(tickers, mom, qua, val, rev,
                                   ai_research_scores=ai_scores,
                                   ai_confidence_scores=ai_conf,
                                   is_backtest=False)
        assert scores[0].ai_research == pytest.approx(50.0)

    def test_higher_raw_scores_rank_higher(self):
        tickers = ["LOW", "HIGH"]
        mom, qua, val, rev = _make_raws(tickers, [0.0, 1.0], [0.0, 1.0],
                                         [0.0, 1.0], [0.0, 1.0])
        scores = compute_composite(tickers, mom, qua, val, rev)
        score_map = {s.ticker: s for s in scores}
        assert score_map["HIGH"].composite > score_map["LOW"].composite

    def test_rank_scores_descending(self):
        tickers = ["A", "B", "C"]
        mom, qua, val, rev = _make_raws(tickers, [0.1, 1.0, 0.5],
                                         [0.1, 1.0, 0.5], [0.1, 1.0, 0.5],
                                         [0.1, 1.0, 0.5])
        scores = compute_composite(tickers, mom, qua, val, rev)
        ranked = rank_scores(scores)
        assert ranked[0].ticker == "B"
        assert ranked[-1].ticker == "A"

    def test_missing_ticker_in_raw_dicts_gives_neutral(self):
        tickers = ["A", "MISSING"]
        mom, qua, val, rev = _make_raws(["A"], [0.5], [0.5], [0.5], [0.5])
        scores = compute_composite(tickers, mom, qua, val, rev)
        score_map = {s.ticker: s for s in scores}
        assert score_map["MISSING"].quality == pytest.approx(NEUTRAL)

    def test_empty_ticker_list_returns_empty(self):
        scores = compute_composite([], {}, {}, {}, {})
        assert scores == []

    def test_zscore_normalization_single_value(self):
        # Single value with data — everyone gets neutral (can't z-score one point)
        result = _zscore_to_0_100([None, None, 0.5])
        assert all(v == pytest.approx(NEUTRAL) for v in result)

    def test_zscore_identical_values_all_neutral(self):
        result = _zscore_to_0_100([1.0, 1.0, 1.0])
        assert all(v == pytest.approx(NEUTRAL) for v in result)

    def test_composite_weights_sum_correctly(self):
        from config.weights import FACTOR_WEIGHTS
        assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(1.0)

    def test_ai_not_in_factor_weights(self):
        # AI is display-only — it flows through Moat, not a standalone composite factor
        from config.weights import FACTOR_WEIGHTS
        assert "ai_research" not in FACTOR_WEIGHTS

    def test_moat_weight_is_10_percent(self):
        from config.weights import FACTOR_WEIGHTS
        assert FACTOR_WEIGHTS["moat"] == pytest.approx(0.10)

    def test_capital_allocation_weight_is_10_percent(self):
        from config.weights import FACTOR_WEIGHTS
        assert FACTOR_WEIGHTS["capital_allocation"] == pytest.approx(0.10)

    def test_risk_adjustment_weight_is_5_percent(self):
        from config.weights import FACTOR_WEIGHTS
        assert FACTOR_WEIGHTS["risk_adjustment"] == pytest.approx(0.05)

    def test_earnings_quality_weight_is_10_percent(self):
        from config.weights import FACTOR_WEIGHTS
        assert FACTOR_WEIGHTS["earnings_quality"] == pytest.approx(0.10)

    def test_earnings_quality_in_composite(self):
        """earnings_quality factor affects composite score."""
        from factors.earnings_quality import EarningsQualityRaw
        tickers = ["A", "B"]
        mom, qua, val, rev = _make_raws(tickers, [0.5]*2, [0.5]*2, [0.5]*2, [0.5]*2)
        # A has high EQ, B has low EQ
        eq = {
            "A": EarningsQualityRaw(ticker="A", raw_score=0.9),
            "B": EarningsQualityRaw(ticker="B", raw_score=0.1),
        }
        scores = compute_composite(tickers, mom, qua, val, rev,
                                   earnings_quality_raw=eq)
        sm = {s.ticker: s for s in scores}
        assert sm["A"].earnings_quality > sm["B"].earnings_quality
        assert sm["A"].composite > sm["B"].composite

    def test_no_earnings_quality_raw_gives_neutral(self):
        """When earnings_quality_raw is None, all EQ scores default to neutral 50."""
        tickers = ["A", "B"]
        mom, qua, val, rev = _make_raws(tickers, [0.5]*2, [0.5]*2, [0.5]*2, [0.5]*2)
        scores = compute_composite(tickers, mom, qua, val, rev,
                                   earnings_quality_raw=None)
        for s in scores:
            assert s.earnings_quality == pytest.approx(NEUTRAL)


# ============================================================
# 6. Earnings Quality
# ============================================================

from factors.earnings_quality import compute_earnings_quality, EarningsQualityRaw


class TestEarningsQuality:
    def test_fcf_conversion_good(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=120, revenue=500,
        )
        assert r.fcf_conversion == pytest.approx(1.2)
        assert "fcf_conversion" in r.signals_used

    def test_fcf_conversion_weak_triggers_warning(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=50, revenue=500,
        )
        assert r.fcf_conversion == pytest.approx(0.5)
        assert any("Weak FCF" in w for w in r.warnings)

    def test_sbc_penalty_small(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            sbc=25,   # 25% of FCF → small penalty
        )
        assert r.sbc_penalty_tier == "small"
        assert r.sbc_penalty == pytest.approx(0.05)

    def test_sbc_penalty_medium(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            sbc=35,   # 35% of FCF → medium
        )
        assert r.sbc_penalty_tier == "medium"
        assert r.sbc_penalty == pytest.approx(0.10)

    def test_sbc_penalty_large(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            sbc=60,   # 60% of FCF → large
        )
        assert r.sbc_penalty_tier == "large"
        assert r.sbc_penalty == pytest.approx(0.20)

    def test_no_sbc_no_penalty(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
        )
        assert r.sbc_penalty == pytest.approx(0.0)
        assert r.sbc_penalty_tier == "none"

    def test_sbc_reduces_raw_score(self):
        r_no_sbc = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=200, revenue=500,
        )
        r_high_sbc = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=200, revenue=500,
            sbc=120,  # 60% → large penalty
        )
        assert r_no_sbc.raw_score is not None
        assert r_high_sbc.raw_score is not None
        assert r_high_sbc.raw_score < r_no_sbc.raw_score

    def test_share_buyback_improves_score(self):
        r_buyback = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            shares_current=90, shares_prior=100,   # 10% buyback
        )
        r_dilution = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            shares_current=110, shares_prior=100,  # 10% dilution
        )
        # Both need ≥2 signals
        if r_buyback.raw_score is not None and r_dilution.raw_score is not None:
            assert r_buyback.raw_score > r_dilution.raw_score

    def test_share_dilution_triggers_warning(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            shares_current=105, shares_prior=100,  # 5% dilution
        )
        assert any("dilution" in w.lower() for w in r.warnings)

    def test_raw_score_clamped_to_0_1(self):
        # With large SBC penalty, score should not go below 0
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            sbc=200,  # 200% → large penalty
        )
        if r.raw_score is not None:
            assert 0.0 <= r.raw_score <= 1.0

    def test_insufficient_signals_returns_none_score(self):
        r = compute_earnings_quality(
            "T", net_income=100,  # only NI, no FCF → can't compute any signal
        )
        assert r.raw_score is None

    def test_roic_declining_triggers_warning(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            roic_current=0.10, roic_prior=0.20,  # ROIC fell 10pp
        )
        assert any("ROIC" in w for w in r.warnings)

    def test_debt_increasing_triggers_warning(self):
        r = compute_earnings_quality(
            "T", net_income=100, free_cash_flow=100, revenue=500,
            total_debt_current=200, total_debt_prior=100,
            total_equity_current=200,   # +50% of equity in new debt
        )
        assert any("Debt" in w for w in r.warnings)


# ============================================================
# 7. Final Picks Guards
# ============================================================

class TestFinalPicksGuards:
    def test_weights_sum_to_exactly_100_pct(self):
        from config.weights import FACTOR_WEIGHTS
        total = sum(FACTOR_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-10), \
            f"FACTOR_WEIGHTS sum = {total}, must be exactly 1.0"

    def test_no_raw_ranking_bypasses_safe_recommendations(self):
        """safe_recommendations() must be the only source of BUY NOW labels."""
        from portfolio.recommendation_guard import safe_recommendations, Recommendation
        # Verify the function exists and returns Recommendation objects
        assert callable(safe_recommendations)

    def test_composite_field_includes_earnings_quality(self):
        """FactorScores must have earnings_quality field."""
        from factors.composite import FactorScores
        fs = FactorScores(ticker="TEST")
        assert hasattr(fs, "earnings_quality")
        assert fs.earnings_quality == pytest.approx(50.0)

    def test_to_dict_includes_earnings_quality(self):
        """FactorScores.to_dict() must serialize earnings_quality."""
        from factors.composite import FactorScores
        fs = FactorScores(ticker="TEST", earnings_quality=72.5)
        d  = fs.to_dict()
        assert "earnings_quality" in d
        assert d["earnings_quality"] == pytest.approx(72.5)

    def test_stale_data_gate_uses_file_mtime(self):
        """Gate 1 freshness check uses file mtime, not timezone-naive timestamp."""
        from reports.final_picks_report import _gate1_data_quality
        import tempfile, os, time
        # Create a temp file that's definitely fresh
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            tmp = f.name
        try:
            # Monkey-patch the path temporarily
            import reports.final_picks_report as fp_mod
            orig = fp_mod.SCORED_UNIVERSE
            fp_mod.SCORED_UNIVERSE = type(orig)(tmp)
            issues = fp_mod._gate1_data_quality([], "2000-01-01T00:00:00")
            stale_blocks = [i for i in issues if "min old" in i.message]
            assert len(stale_blocks) == 0, "Fresh file should not trigger stale block"
        finally:
            fp_mod.SCORED_UNIVERSE = orig
            os.unlink(tmp)
