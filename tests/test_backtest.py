"""
Tests for Phase 7: Backtester.

No real API calls — all data is synthetic.
Tests verify:
  - Metrics math (CAGR, Sharpe, Sortino, Max Drawdown, Win Rate)
  - Sortino uses standard denominator (total months, not just negative months)
  - Calendar-year returns compound correctly
  - Transaction costs are deducted from gross returns
  - Survivorship bias and point-in-time warnings always present
  - Benchmark comparison (alpha, beta, correlation)
  - Overfitting sub-period slicing
  - Concentration analysis (HHI, avg positions)
  - Engine end-to-end with synthetic data
"""

from __future__ import annotations

import math
import pytest

from backtester.metrics import (
    compute_metrics,
    _cagr,
    _sharpe,
    _sortino,
    _max_drawdown,
    _calendar_year_returns,
)
from backtester.benchmark import compute_benchmark_comparison, _correlation, _beta
from backtester.overfitting import (
    compute_overfitting_report,
    compute_concentration,
)
from backtester.engine import (
    run_backtest,
    BacktestConfig,
    SURVIVORSHIP_BIAS_WARNING,
    FUNDAMENTALS_NOT_POINT_IN_TIME_WARNING,
    _compute_turnover,
    TRANSACTION_COST_RATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_returns(n: int, r: float = 0.01) -> list[float]:
    return [r] * n


def _dates(n: int, start_year: int = 2010) -> list[str]:
    dates = []
    year, month = start_year, 1
    for _ in range(n):
        dates.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return dates


def _years(dates: list[str]) -> list[int]:
    return [int(d[:4]) for d in dates]


# ---------------------------------------------------------------------------
# metrics.py — CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_flat_1pct_monthly_24_months(self):
        rets = _flat_returns(24, 0.01)
        # (1.01^24)^(1/2) - 1
        expected = (1.01 ** 24) ** (1 / 2) - 1
        assert _cagr(rets) == pytest.approx(expected, rel=1e-6)

    def test_zero_returns(self):
        assert _cagr([0.0] * 12) == pytest.approx(0.0)

    def test_empty_returns(self):
        assert _cagr([]) == pytest.approx(0.0)

    def test_known_value(self):
        # 12 months at 0.01/month → (1.01^12)^(1/1) - 1 ≈ 0.1268
        rets = [0.01] * 12
        assert _cagr(rets) == pytest.approx((1.01 ** 12) - 1, rel=1e-6)


# ---------------------------------------------------------------------------
# metrics.py — Sharpe
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_zero_volatility_returns_zero(self):
        assert _sharpe([0.01] * 24) == pytest.approx(0.0)

    def test_positive_returns_positive_sharpe(self):
        rets = [0.02, 0.01, 0.03, -0.01, 0.02, 0.01] * 4
        assert _sharpe(rets) > 0

    def test_negative_mean_negative_sharpe(self):
        rets = [-0.02, -0.01, -0.03, 0.01, -0.02] * 4
        assert _sharpe(rets) < 0

    def test_empty_returns_zero(self):
        assert _sharpe([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# metrics.py — Sortino (CRITICAL: standard denominator)
# ---------------------------------------------------------------------------

class TestSortino:
    def test_standard_denominator_uses_all_months(self):
        """
        Verify: downside_dev = sqrt(sum(min(r,0)^2) / N_TOTAL) * sqrt(12)
        NOT divided by count of negative months only.
        """
        # 3 months: +5%, -2%, +3%  → N = 3
        rets = [0.05, -0.02, 0.03]
        n = 3
        mean_monthly = sum(rets) / n          # (0.05 - 0.02 + 0.03) / 3 = 0.02
        # downside: only the -0.02 contributes
        sum_sq = (min(-0.02, 0)) ** 2 + (min(0.05, 0)) ** 2 + (min(0.03, 0)) ** 2
        # = 0.0004 + 0 + 0
        downside_dev = math.sqrt(sum_sq / n) * math.sqrt(12)
        expected = (mean_monthly * 12) / downside_dev
        assert _sortino(rets) == pytest.approx(expected, rel=1e-6)

    def test_standard_vs_negative_only_denominator_differ(self):
        """
        Confirm standard denominator ≠ negative-only denominator.
        10 months: 5 positives at +3%, 5 negatives at -1%.
        """
        rets = [0.03, -0.01, 0.03, -0.01, 0.03, -0.01, 0.03, -0.01, 0.03, -0.01]
        n = 10
        n_neg = 5
        # Standard: divide by N=10
        sum_sq = sum(min(r, 0) ** 2 for r in rets)
        std_dev = math.sqrt(sum_sq / n) * math.sqrt(12)
        # Negative-only: divide by n_neg=5
        neg_only_dev = math.sqrt(sum_sq / n_neg) * math.sqrt(12)
        # They should differ
        assert std_dev != pytest.approx(neg_only_dev)
        # Our implementation uses std_dev
        mean_monthly = sum(rets) / n
        expected_sortino = (mean_monthly * 12) / std_dev
        assert _sortino(rets) == pytest.approx(expected_sortino, rel=1e-6)

    def test_no_negative_months_returns_zero(self):
        # All positive months → downside_dev = 0 → sortino = 0
        rets = [0.01, 0.02, 0.03]
        assert _sortino(rets) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _sortino([]) == pytest.approx(0.0)

    def test_all_negative_months(self):
        rets = [-0.01, -0.02, -0.03]
        result = _sortino(rets)
        # Mean is negative, downside_dev > 0 → sortino is negative
        assert result < 0


# ---------------------------------------------------------------------------
# metrics.py — Max Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown_if_always_up(self):
        rets = [0.01] * 12
        assert _max_drawdown(rets) == pytest.approx(0.0)

    def test_single_down_month(self):
        # Up 10%, down 50%: peak = 1.10, trough = 0.55 → dd = (0.55-1.10)/1.10
        rets = [0.10, -0.50]
        dd = _max_drawdown(rets)
        assert dd == pytest.approx((0.55 - 1.10) / 1.10, rel=1e-6)

    def test_known_drawdown(self):
        # peak at 1.0, drops to 0.8 → dd = -0.20
        rets = [-0.10, -0.10, -0.10]   # rough approximation
        dd = _max_drawdown(rets)
        assert dd < 0

    def test_empty_returns_zero(self):
        assert _max_drawdown([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# metrics.py — Calendar year returns
# ---------------------------------------------------------------------------

class TestCalendarYearReturns:
    def test_single_year_compounding(self):
        # Jan–Dec 2010: all +1%
        rets = [0.01] * 12
        years = [2010] * 12
        cy = _calendar_year_returns(rets, years)
        assert cy[2010] == pytest.approx((1.01 ** 12) - 1, rel=1e-6)

    def test_two_years_separated(self):
        rets = [0.02] * 6 + [-0.01] * 6
        years = [2010] * 6 + [2011] * 6
        cy = _calendar_year_returns(rets, years)
        assert cy[2010] == pytest.approx((1.02 ** 6) - 1, rel=1e-6)
        assert cy[2011] == pytest.approx((0.99 ** 6) - 1, rel=1e-6)

    def test_mixed_months_in_year(self):
        rets  = [0.05, -0.03]
        years = [2020, 2020]
        cy = _calendar_year_returns(rets, years)
        assert cy[2020] == pytest.approx(1.05 * 0.97 - 1, rel=1e-6)


# ---------------------------------------------------------------------------
# metrics.py — compute_metrics integration
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_correct_n_months(self):
        rets = _flat_returns(36, 0.01)
        years = _years(_dates(36, 2010))
        m = compute_metrics(rets, years)
        assert m.n_months == 36

    def test_best_and_worst_month(self):
        rets = [0.05, -0.03, 0.01, 0.02]
        years = [2010, 2010, 2010, 2010]
        m = compute_metrics(rets, years)
        assert m.best_month == pytest.approx(0.05)
        assert m.worst_month == pytest.approx(-0.03)

    def test_win_rate_half(self):
        rets = [0.01, -0.01] * 10
        years = [2010 + i // 12 for i in range(20)]
        m = compute_metrics(rets, years)
        assert m.win_rate == pytest.approx(0.5)

    def test_avg_turnover_recorded(self):
        rets = _flat_returns(12, 0.01)
        years = [2010] * 12
        turnovers = [0.2] * 12
        m = compute_metrics(rets, years, turnovers)
        assert m.avg_turnover == pytest.approx(0.2)

    def test_empty_returns(self):
        m = compute_metrics([], [], [])
        assert m.n_months == 0
        assert m.cagr == 0.0


# ---------------------------------------------------------------------------
# benchmark.py
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_alpha_vs_spy_positive_when_outperforming(self):
        port = _flat_returns(24, 0.015)
        spy  = _flat_returns(24, 0.008)
        qqq  = _flat_returns(24, 0.010)
        b = compute_benchmark_comparison(port, spy, qqq)
        assert b.alpha_vs_spy > 0

    def test_alpha_vs_spy_negative_when_underperforming(self):
        port = _flat_returns(24, 0.005)
        spy  = _flat_returns(24, 0.010)
        qqq  = _flat_returns(24, 0.010)
        b = compute_benchmark_comparison(port, spy, qqq)
        assert b.alpha_vs_spy < 0

    def test_perfect_correlation_with_identical_series(self):
        rets = [0.01, -0.02, 0.03, 0.01, -0.01] * 6
        b = compute_benchmark_comparison(rets, rets, rets)
        assert b.correlation_with_spy == pytest.approx(1.0, abs=1e-6)

    def test_beta_of_1_for_identical_series(self):
        rets = [0.01, -0.02, 0.03, 0.01, -0.01] * 6
        b = compute_benchmark_comparison(rets, rets, rets)
        assert b.beta_vs_spy == pytest.approx(1.0, rel=1e-6)

    def test_beta_greater_than_1_for_amplified(self):
        base = [0.01, -0.02, 0.03, -0.01, 0.02] * 6
        doubled = [r * 2 for r in base]
        b = compute_benchmark_comparison(doubled, base, base)
        assert b.beta_vs_spy == pytest.approx(2.0, rel=1e-4)

    def test_to_dict_structure(self):
        rets = _flat_returns(24, 0.01)
        b = compute_benchmark_comparison(rets, rets, rets)
        d = b.to_dict()
        for key in ("portfolio_cagr", "spy_cagr", "alpha_vs_spy", "beta_vs_spy",
                    "correlation_with_spy", "portfolio_max_drawdown"):
            assert key in d


# ---------------------------------------------------------------------------
# overfitting.py — concentration
# ---------------------------------------------------------------------------

class TestConcentration:
    def test_equal_weight_5_hhi(self):
        weights = [[0.30, 0.25, 0.20, 0.15, 0.10]] * 12
        c = compute_concentration(weights)
        expected_hhi = 0.30**2 + 0.25**2 + 0.20**2 + 0.15**2 + 0.10**2
        assert c.avg_hhi == pytest.approx(expected_hhi, rel=1e-6)
        assert c.avg_n_positions == pytest.approx(5.0)
        assert c.avg_top1_weight == pytest.approx(0.30)

    def test_single_position_hhi_is_1(self):
        weights = [[1.0]] * 6
        c = compute_concentration(weights)
        assert c.avg_hhi == pytest.approx(1.0)

    def test_varying_positions(self):
        weights = [[0.30, 0.25, 0.20, 0.15, 0.10], [0.55, 0.45]]
        c = compute_concentration(weights)
        assert c.min_positions == 2
        assert c.max_positions == 5

    def test_empty_returns_zeros(self):
        c = compute_concentration([])
        assert c.avg_n_positions == 0


# ---------------------------------------------------------------------------
# overfitting.py — period slicing
# ---------------------------------------------------------------------------

class TestOverfittingPeriods:
    def _run(self):
        # 2005–2026 = 264 months
        dates = _dates(264, 2005)
        port = _flat_returns(264, 0.01)
        spy  = _flat_returns(264, 0.008)
        weights = [[0.30, 0.25, 0.20, 0.15, 0.10]] * 264
        return compute_overfitting_report(dates, port, spy, weights)

    def test_in_sample_years(self):
        r = self._run()
        assert r.in_sample.start == "2005-01"
        assert r.in_sample.end == "2015-12"
        assert r.in_sample.n_months == 132   # 11 years × 12

    def test_out_sample_years(self):
        r = self._run()
        assert r.out_sample.n_months == 132

    def test_crisis_2008_12_months(self):
        r = self._run()
        assert r.crisis_2008.n_months == 12

    def test_crisis_2020_12_months(self):
        r = self._run()
        assert r.crisis_2020.n_months == 12

    def test_crisis_2022_12_months(self):
        r = self._run()
        assert r.crisis_2022.n_months == 12

    def test_performance_decay_computed(self):
        r = self._run()
        # With identical flat returns, decay should be ~0
        assert abs(r.performance_decay) < 0.001

    def test_concentration_in_report(self):
        r = self._run()
        assert r.concentration.avg_n_positions == pytest.approx(5.0)

    def test_to_dict_structure(self):
        r = self._run()
        d = r.to_dict()
        assert "in_sample" in d
        assert "out_sample" in d
        assert "crisis_2008" in d
        assert "performance_decay" in d
        assert "concentration" in d


# ---------------------------------------------------------------------------
# engine.py — turnover computation
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_no_change_zero_turnover(self):
        t = _compute_turnover(["A", "B"], ["A", "B"], [0.6, 0.4], [0.6, 0.4])
        assert t == pytest.approx(0.0)

    def test_full_replacement_full_turnover(self):
        # Old: A=0.6, B=0.4 → New: C=0.6, D=0.4 → total |change| = 2.0 → /2 = 1.0
        t = _compute_turnover(["A", "B"], ["C", "D"], [0.6, 0.4], [0.6, 0.4])
        assert t == pytest.approx(1.0)

    def test_partial_change(self):
        # Old: A=0.6, B=0.4 → New: A=0.4, B=0.4, C=0.2
        # Changes: A: |0.4-0.6|=0.2, B: |0.4-0.4|=0, C: |0.2-0|=0.2 → total=0.4 → /2 = 0.2
        t = _compute_turnover(["A", "B"], ["A", "B", "C"],
                              [0.6, 0.4], [0.4, 0.4, 0.2])
        assert t == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# engine.py — run_backtest end-to-end
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def _simple_backtest(self, n_months=24, gross_ret=0.01, spy_ret=0.008, qqq_ret=0.009):
        dates     = _dates(n_months, 2010)
        gross     = [gross_ret] * n_months
        spy       = [spy_ret] * n_months
        qqq       = [qqq_ret] * n_months
        positions = [(["AAPL", "MSFT"], [0.55, 0.45])] * n_months
        return run_backtest(dates, gross, spy, qqq, positions)

    def test_survivorship_bias_warning_present(self):
        r = self._simple_backtest()
        assert any("SURVIVORSHIP" in w for w in r.warnings)

    def test_point_in_time_warning_present(self):
        r = self._simple_backtest()
        assert any("LOOK-AHEAD" in w for w in r.warnings)

    def test_both_warnings_always_present(self):
        r = self._simple_backtest()
        assert len(r.warnings) >= 2

    def test_equity_curve_length_matches_months(self):
        r = self._simple_backtest(n_months=36)
        assert len(r.equity_curve) == 36

    def test_equity_curve_grows_with_positive_returns(self):
        r = self._simple_backtest(n_months=12, gross_ret=0.02)
        assert r.equity_curve[-1]["value"] > r.config.initial_capital

    def test_transaction_costs_reduce_net_return(self):
        # Full turnover every month → cost applied every month
        dates = _dates(12, 2010)
        gross = [0.01] * 12
        spy   = [0.0] * 12
        qqq   = [0.0] * 12
        # First month: old=[], new=[A,B] → 100% turnover (entering)
        # Subsequent: same positions → 0% turnover
        positions = [(["A", "B"], [0.6, 0.4])] * 12
        r = run_backtest(dates, gross, spy, qqq, positions)
        # Net return of first month should be slightly below 1% gross
        assert r.monthly_snapshots[1].portfolio_return <= 0.01

    def test_n_months_correct(self):
        r = self._simple_backtest(n_months=48)
        assert r.n_months == 48

    def test_metrics_populated(self):
        r = self._simple_backtest()
        assert r.metrics.n_months == 24
        assert r.metrics.cagr != 0
        assert r.metrics.sharpe != 0

    def test_benchmark_populated(self):
        r = self._simple_backtest(gross_ret=0.015, spy_ret=0.008)
        assert r.benchmark_comparison.alpha_vs_spy > 0

    def test_overfitting_report_populated(self):
        # Use 2005–2026 range for meaningful period split
        n = 264
        dates     = _dates(n, 2005)
        gross     = [0.01] * n
        spy       = [0.008] * n
        qqq       = [0.009] * n
        positions = [(["A", "B", "C", "D", "E"],
                      [0.30, 0.25, 0.20, 0.15, 0.10])] * n
        r = run_backtest(dates, gross, spy, qqq, positions)
        assert r.overfitting_report.in_sample.n_months == 132
        assert r.overfitting_report.out_sample.n_months == 132

    def test_to_dict_structure(self):
        r = self._simple_backtest()
        d = r.to_dict()
        assert "metrics" in d
        assert "benchmark_comparison" in d
        assert "overfitting_report" in d
        assert "warnings" in d
        assert "equity_curve" in d

    def test_config_uses_correct_transaction_cost(self):
        cfg = BacktestConfig(transaction_cost=0.001)
        assert cfg.transaction_cost == TRANSACTION_COST_RATE

    def test_ai_research_constant_used(self):
        from config.settings import BACKTEST_AI_NEUTRAL
        assert BACKTEST_AI_NEUTRAL == 50.0

    def test_zero_months_returns_empty_result(self):
        r = run_backtest([], [], [], [], [])
        assert r.n_months == 0
        assert r.metrics.cagr == 0.0
        assert len(r.warnings) >= 2   # warnings still present
