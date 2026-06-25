"""
Tests for V6 institutional validation engines:
  - Walk-Forward Engine (Phase 1)
  - Point-in-Time Snapshot System (Phase 2)
  - Model Confidence 2.0 (Phase 3)
  - Concentration Control (Phase 5)
  - Portfolio Confidence Weighting (Phase 6)
  - Advanced Stress Testing (Phase 7)
  - Factor Importance Monitor (Phase 8)
  - Walk-Forward Report (Phase 1 report)
"""

from __future__ import annotations

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ---------------------------------------------------------------------------
# Phase 3 — Model Confidence 2.0
# ---------------------------------------------------------------------------

class TestModelConfidence20:

    def test_new_weights_include_stress_reliability(self):
        from factors.model_confidence import compute_model_confidence
        r = compute_model_confidence(
            ticker="AAPL",
            data_quality_score=80.0,
            ai_confidence=0.50,
            factor_stability_score=75.0,
            num_analysts=15,
            n_stress_periods=3,
        )
        assert r.stress_reliability_input > 50.0
        assert isinstance(r.overall_confidence_score, float)

    def test_zero_stress_periods_lowers_confidence(self):
        from factors.model_confidence import compute_model_confidence
        high = compute_model_confidence("X", data_quality_score=80, n_stress_periods=4)
        low  = compute_model_confidence("X", data_quality_score=80, n_stress_periods=0)
        assert high.overall_confidence_score > low.overall_confidence_score

    def test_stress_reliability_score_mapping(self):
        from factors.model_confidence import _stress_reliability_score
        assert _stress_reliability_score(None) == pytest.approx(30.0)
        assert _stress_reliability_score(0)    == pytest.approx(30.0)
        assert _stress_reliability_score(1)    == pytest.approx(45.0)
        assert _stress_reliability_score(2)    == pytest.approx(65.0)
        assert _stress_reliability_score(3)    == pytest.approx(85.0)
        assert _stress_reliability_score(4)    == pytest.approx(100.0)

    def test_to_dict_contains_stress_reliability(self):
        from factors.model_confidence import compute_model_confidence
        r = compute_model_confidence("X", data_quality_score=70, n_stress_periods=2)
        d = r.to_dict()
        assert "stress_reliability_input" in d
        assert "overall_confidence_score" in d
        assert "label" in d

    def test_to_dict_still_has_legacy_fields(self):
        from factors.model_confidence import compute_model_confidence
        r = compute_model_confidence("X", data_quality_score=70)
        d = r.to_dict()
        assert "history_depth_input" in d
        assert "accounting_quality_input" in d

    def test_high_confidence_inputs_produce_high_label(self):
        from factors.model_confidence import compute_model_confidence
        r = compute_model_confidence(
            ticker="AAPL",
            data_quality_score=95.0,
            ai_confidence=0.90,
            factor_stability_score=95.0,
            num_analysts=25,
            n_stress_periods=4,
        )
        assert r.overall_confidence_score >= 80
        assert r.label in ("Very High", "High")

    def test_low_inputs_produce_low_label(self):
        from factors.model_confidence import compute_model_confidence
        r = compute_model_confidence(
            ticker="XYZ",
            data_quality_score=20.0,
            ai_confidence=0.05,
            factor_stability_score=10.0,
            num_analysts=0,
            n_stress_periods=0,
        )
        assert r.overall_confidence_score < 50
        assert r.label in ("Low", "Very Low")


# ---------------------------------------------------------------------------
# Phase 2 — Point-in-Time Snapshot System
# ---------------------------------------------------------------------------

class TestSnapshot:

    def _make_payload(self, n_tickers: int = 5) -> dict:
        import datetime
        return {
            "generated_at": datetime.datetime.now().isoformat(),
            "as_of_date":   str(datetime.date.today()),
            "ticker_count": n_tickers,
            "universe": [
                {
                    "ticker":    f"T{i}",
                    "composite": 50.0 + i,
                    "quality":   55.0,
                    "momentum":  50.0,
                    "shariah_status": "compliant",
                }
                for i in range(n_tickers)
            ],
        }

    def test_validate_snapshot_clean(self):
        from data_layer.snapshot import validate_snapshot
        errors = validate_snapshot(self._make_payload(10))
        assert errors == []

    def test_validate_empty_universe_is_error(self):
        from data_layer.snapshot import validate_snapshot
        payload = self._make_payload(0)
        errors = validate_snapshot(payload)
        assert any("empty" in e for e in errors)

    def test_validate_missing_generated_at(self):
        from data_layer.snapshot import validate_snapshot
        payload = self._make_payload(5)
        payload.pop("generated_at")
        errors = validate_snapshot(payload)
        assert any("generated_at" in e for e in errors)

    def test_list_snapshot_dates_empty_dir(self, tmp_path, monkeypatch):
        from data_layer import snapshot as snap_mod
        monkeypatch.setattr(snap_mod, "_HISTORY_DIR", tmp_path / "history")
        dates = snap_mod.list_snapshot_dates()
        assert dates == []

    def test_save_and_load_snapshot(self, tmp_path, monkeypatch):
        from data_layer import snapshot as snap_mod
        hist_dir = tmp_path / "history"
        monkeypatch.setattr(snap_mod, "_HISTORY_DIR", hist_dir)

        payload  = self._make_payload(5)
        result   = snap_mod.save_snapshot(payload)
        assert result is not None
        assert result.exists()

        dates = snap_mod.list_snapshot_dates()
        assert len(dates) == 1

        loaded = snap_mod.load_snapshot(dates[0])
        assert loaded is not None
        assert len(loaded["universe"]) == 5

    def test_save_snapshot_does_not_overwrite(self, tmp_path, monkeypatch):
        from data_layer import snapshot as snap_mod
        hist_dir = tmp_path / "history"
        monkeypatch.setattr(snap_mod, "_HISTORY_DIR", hist_dir)

        payload = self._make_payload(5)
        r1 = snap_mod.save_snapshot(payload)
        r2 = snap_mod.save_snapshot(payload)   # second call same day
        assert r1 is not None
        assert r2 is None  # skipped

    def test_get_snapshot_summary(self, tmp_path, monkeypatch):
        from data_layer import snapshot as snap_mod
        hist_dir = tmp_path / "history"
        monkeypatch.setattr(snap_mod, "_HISTORY_DIR", hist_dir)

        snap_mod.save_snapshot(self._make_payload(5))
        summary = snap_mod.get_snapshot_summary()
        assert summary["count"] == 1
        assert summary["first_date"] is not None


# ---------------------------------------------------------------------------
# Phase 5 — Concentration Control
# ---------------------------------------------------------------------------

class TestConcentration:

    def test_clean_portfolio_passes(self):
        from portfolio.concentration import check_concentration
        tickers  = ["A", "B", "C", "D", "E"]
        weights  = {t: 0.20 for t in tickers}
        sectors  = {t: f"Sector{i}" for i, t in enumerate(tickers)}
        inds     = {t: f"Ind{i}" for i, t in enumerate(tickers)}
        result = check_concentration(tickers, weights, sectors, inds)
        assert result.passed
        assert result.issues == []

    def test_industry_concentration_warns(self):
        from portfolio.concentration import check_concentration
        tickers  = ["A", "B", "C", "D", "E"]
        weights  = {t: 0.20 for t in tickers}
        # A, B, C all in same industry → violates max 2
        sectors  = {t: "Tech" for t in tickers}
        inds     = {"A": "SameInd", "B": "SameInd", "C": "SameInd",
                    "D": "OtherInd", "E": "OtherInd2"}
        result = check_concentration(tickers, weights, sectors, inds)
        industry_issues = [i for i in result.issues if i.rule == "industry"]
        assert len(industry_issues) > 0

    def test_sector_weight_violation_warns(self):
        from portfolio.concentration import check_concentration
        tickers  = ["A", "B", "C", "D", "E"]
        # A, B, C get heavy weight in same sector → >35%
        weights  = {"A": 0.30, "B": 0.30, "C": 0.20, "D": 0.10, "E": 0.10}
        sectors  = {"A": "Tech", "B": "Tech", "C": "Tech",
                    "D": "Health", "E": "Energy"}
        inds     = {t: f"Ind{i}" for i, t in enumerate(tickers)}
        result = check_concentration(tickers, weights, sectors, inds)
        sector_issues = [i for i in result.issues if i.rule == "sector"]
        assert len(sector_issues) > 0

    def test_optimize_respects_industry_cap(self):
        from portfolio.concentration import optimize_for_concentration
        universe = [
            {"ticker": "A", "composite": 90.0, "sector": "Tech", "industry": "Semis"},
            {"ticker": "B", "composite": 85.0, "sector": "Tech", "industry": "Semis"},
            {"ticker": "C", "composite": 80.0, "sector": "Tech", "industry": "Semis"},  # 3rd → skip
            {"ticker": "D", "composite": 75.0, "sector": "Health", "industry": "Biotech"},
            {"ticker": "E", "composite": 70.0, "sector": "Energy", "industry": "Oil"},
        ]
        selected, _ = optimize_for_concentration(universe, target_n=5)
        # "Semis" can have at most 2 (A and B); C should be skipped
        semis_count = sum(1 for t in selected if t in ("A", "B", "C"))
        assert semis_count <= 2
        assert len(selected) <= 5

    def test_optimize_returns_top_n(self):
        from portfolio.concentration import optimize_for_concentration
        universe = [
            {"ticker": f"T{i}", "composite": 90.0 - i, "sector": f"S{i}", "industry": f"I{i}"}
            for i in range(20)
        ]
        selected, _ = optimize_for_concentration(universe, target_n=5)
        assert len(selected) == 5
        assert selected[0] == "T0"   # highest composite first


# ---------------------------------------------------------------------------
# Phase 6 — Portfolio Confidence Weighting
# ---------------------------------------------------------------------------

class TestConfidenceWeighting:

    def test_weights_sum_to_one(self):
        from portfolio.confidence_weighting import compute_confidence_weights
        r = compute_confidence_weights(
            tickers           = ["A", "B", "C", "D", "E"],
            composite_scores  = [70.0, 65.0, 60.0, 55.0, 50.0],
            model_confidences = [80.0, 75.0, 70.0, 65.0, 60.0],
            risk_scores       = [70.0, 65.0, 60.0, 55.0, 50.0],
        )
        assert abs(sum(r.confidence_weights) - 1.0) < 1e-9

    def test_higher_confidence_gets_more_weight(self):
        from portfolio.confidence_weighting import compute_confidence_weights
        # Use 5 stocks so bounds don't force equal weighting for 2 stocks
        r = compute_confidence_weights(
            tickers           = ["A", "B", "C", "D", "E"],
            composite_scores  = [70.0, 70.0, 70.0, 70.0, 70.0],
            model_confidences = [90.0, 30.0, 50.0, 50.0, 50.0],
            risk_scores       = [70.0, 70.0, 70.0, 70.0, 70.0],
        )
        # A has much higher confidence than B
        assert r.confidence_weights[0] > r.confidence_weights[1]

    def test_weights_within_bounds(self):
        from portfolio.confidence_weighting import compute_confidence_weights, _MIN_W, _MAX_W
        r = compute_confidence_weights(
            tickers           = ["A", "B", "C", "D", "E"],
            composite_scores  = [99.0, 50.0, 50.0, 50.0, 10.0],
            model_confidences = [99.0, 50.0, 50.0, 50.0, 10.0],
            risk_scores       = [99.0, 50.0, 50.0, 50.0, 10.0],
        )
        for w in r.confidence_weights:
            assert _MIN_W - 1e-6 <= w <= _MAX_W + 1e-6

    def test_to_dict_structure(self):
        from portfolio.confidence_weighting import compute_confidence_weights
        r = compute_confidence_weights(
            tickers=["A", "B"],
            composite_scores=[65.0, 60.0],
            model_confidences=[80.0, 70.0],
            risk_scores=[70.0, 65.0],
        )
        d = r.to_dict()
        assert "tickers" in d
        assert "confidence_weights" in d
        assert "equal_weights" in d

    def test_empty_input_returns_empty(self):
        from portfolio.confidence_weighting import compute_confidence_weights
        r = compute_confidence_weights([], [], [], [])
        assert r.tickers == []
        assert r.confidence_weights == []


# ---------------------------------------------------------------------------
# Phase 7 — Advanced Stress Testing (regional_banks_2023 crisis window)
# ---------------------------------------------------------------------------

class TestAdvancedStressTest:

    def test_four_crisis_windows_defined(self):
        from factors.stress_test import CRISIS_WINDOWS
        assert "gfc_2008" in CRISIS_WINDOWS
        assert "covid_2020" in CRISIS_WINDOWS
        assert "rates_2022" in CRISIS_WINDOWS
        assert "regional_banks_2023" in CRISIS_WINDOWS

    def test_crisis_result_has_spy_fields(self):
        from factors.stress_test import CrisisResult
        cr = CrisisResult(
            name="gfc_2008",
            max_drawdown=-0.40,
            volatility=0.60,
            recovered=False,
            spy_drawdown=-0.50,
            relative_vs_spy=0.10,
        )
        assert cr.relative_vs_spy == pytest.approx(0.10)
        assert cr.spy_drawdown    == pytest.approx(-0.50)

    def test_stress_test_result_to_dict_includes_n_crises(self):
        from factors.stress_test import StressTestResult, CrisisResult
        cr = CrisisResult("gfc_2008", -0.30, 0.40, True, -0.45, 0.15)
        r  = StressTestResult(ticker="AAPL", resilience_score=70.0,
                              crises=[cr], label="Moderate")
        d  = r.to_dict()
        assert "n_crises_with_data" in d
        assert d["n_crises_with_data"] == 1
        assert d["crises"][0]["spy_drawdown"] is not None
        assert d["crises"][0]["relative_vs_spy"] is not None

    def test_sector_etfs_defined(self):
        from factors.stress_test import SECTOR_ETFS
        assert "Technology" in SECTOR_ETFS
        assert "Financials" in SECTOR_ETFS

    def test_resilience_scoring_is_unchanged(self):
        from factors.stress_test import _resilience_from_crises, CrisisResult
        crises = [
            CrisisResult("gfc_2008",   -0.10, 0.25, True),
            CrisisResult("covid_2020", -0.15, 0.20, True),
        ]
        score = _resilience_from_crises(crises)
        assert score >= 75.0   # mild drawdowns → Resilient


# ---------------------------------------------------------------------------
# Phase 1 — Walk-Forward Engine
# ---------------------------------------------------------------------------

class TestWalkForward:

    def test_quarterly_rebal_dates(self):
        from analysis.walk_forward import _quarterly_rebal_dates
        from datetime import date
        dates = _quarterly_rebal_dates(date(2023, 1, 1), date(2024, 1, 1))
        assert len(dates) >= 4
        for d in dates:
            assert d.month in {3, 6, 9, 12}

    def test_period_return_calculation(self):
        """Test period return from simple price series."""
        from analysis.walk_forward import _period_return
        import pandas as pd
        prices = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.to_datetime(["2023-01-01", "2023-06-01", "2023-12-31"])
        )
        ret = _period_return(prices, "2023-01-01", "2023-12-31")
        assert ret is not None
        assert abs(ret - 0.10) < 0.01

    def test_metrics_computation(self):
        from analysis.walk_forward import _compute_metrics
        port = [0.05, 0.08, -0.03, 0.10, 0.06, -0.02, 0.07, 0.09]
        spy  = [0.03, 0.05, -0.05, 0.08, 0.04, -0.04, 0.05, 0.07]
        m = _compute_metrics(port, spy)
        assert m.n_periods == 8
        assert isinstance(m.sharpe, float)
        assert isinstance(m.cagr, float)
        assert m.max_drawdown <= 0

    def test_metrics_to_dict(self):
        from analysis.walk_forward import _compute_metrics
        port = [0.05, 0.08, -0.03, 0.10]
        spy  = [0.03, 0.05, -0.05, 0.08]
        m = _compute_metrics(port, spy)
        d = m.to_dict()
        required = ["cagr", "sharpe", "sortino", "calmar", "max_drawdown",
                    "win_rate", "information_ratio", "alpha", "beta",
                    "turnover", "tracking_error", "n_periods"]
        for key in required:
            assert key in d

    def test_walk_forward_no_snapshots_returns_warning(self, tmp_path, monkeypatch):
        """With no snapshots, walk-forward should return a warning, not crash."""
        from analysis import walk_forward as wf
        monkeypatch.setattr(wf, "_HISTORY_DIR", tmp_path / "no_history")
        monkeypatch.setattr(wf, "_CACHE_FILE", tmp_path / "wf_results.json")

        result = wf.run_walk_forward()
        assert len(result.warnings) > 0
        assert result.metrics is None

    def test_closest_snapshot_before(self):
        from analysis.walk_forward import _closest_snapshot_before
        from datetime import date
        snaps = ["2023-03-31", "2023-06-30", "2023-09-30"]
        assert _closest_snapshot_before(snaps, date(2023, 7, 1)) == "2023-06-30"
        assert _closest_snapshot_before(snaps, date(2023, 3, 30)) == None
        assert _closest_snapshot_before(snaps, date(2023, 3, 31)) == "2023-03-31"


# ---------------------------------------------------------------------------
# Phase 8 — Factor Importance Monitor
# ---------------------------------------------------------------------------

class TestFactorMonitor:

    def test_monitor_with_no_snapshots(self, tmp_path, monkeypatch):
        from analysis import factor_monitor as fm
        monkeypatch.setattr(fm, "_HISTORY_DIR", tmp_path / "no_history")
        monkeypatch.setattr(fm, "_OUTPUT_FILE", tmp_path / "factor_monitor.json")
        result = fm.run_factor_monitor()
        assert len(result.warnings) > 0
        assert result.n_snapshots_used == 0

    def test_spearman_correlation(self):
        from analysis.factor_monitor import _spearman
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]  # perfect correlation
        ic = _spearman(xs, ys)
        assert ic is not None
        assert abs(ic - 1.0) < 0.001

    def test_spearman_anti_correlation(self):
        from analysis.factor_monitor import _spearman
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]  # perfect negative correlation
        ic = _spearman(xs, ys)
        assert ic is not None
        assert abs(ic + 1.0) < 0.001

    def test_spearman_too_few_points(self):
        from analysis.factor_monitor import _spearman
        ic = _spearman([1.0, 2.0], [1.0, 2.0])
        assert ic is None

    def test_monitor_result_to_dict(self, tmp_path, monkeypatch):
        from analysis import factor_monitor as fm
        monkeypatch.setattr(fm, "_HISTORY_DIR", tmp_path / "no_history")
        monkeypatch.setattr(fm, "_OUTPUT_FILE", tmp_path / "factor_monitor.json")
        result = fm.run_factor_monitor()
        d = result.to_dict()
        assert "generated_at" in d
        assert "n_snapshots" in d
        assert "warnings" in d
        assert "factors" in d
        assert "redundant_pairs" in d

    def test_factor_stats_fields(self):
        from analysis.factor_monitor import FactorStats
        fs = FactorStats(
            factor="quality", ic=0.05, ic_std=0.10,
            hit_rate=0.55, avg_contribution=22.0,
            n_observations=10, flag_decay=False, flag_instability=False,
        )
        d = fs.to_dict()
        assert d["factor"] == "quality"
        assert d["flag_decay"] is False
        assert "ic" in d


# ---------------------------------------------------------------------------
# Walk-Forward Report
# ---------------------------------------------------------------------------

class TestWalkForwardReport:

    def test_report_handles_missing_cache(self, tmp_path, monkeypatch, capsys):
        from reports import walk_forward_report as wfr
        monkeypatch.setattr(wfr, "_CACHE_FILE", tmp_path / "no_file.json")
        wfr.print_walk_forward_report()
        captured = capsys.readouterr()
        assert "No walk-forward results" in captured.out or "snapshot" in captured.out.lower()

    def test_report_renders_from_data(self, capsys):
        """Smoke-test that the report printer doesn't crash with minimal data."""
        from reports.walk_forward_report import print_walk_forward_report
        from analysis.walk_forward import WalkForwardResult
        result = WalkForwardResult(
            warnings=["Test warning — no snapshot data"],
            snapshot_dates=[],
            n_snapshots_used=0,
        )
        print_walk_forward_report(result)
        captured = capsys.readouterr()
        assert "WALK-FORWARD" in captured.out
