"""
Tests for V5 validation engines:
  - Data Quality Engine (Phase 2)
  - Model Confidence Engine (Phase 3)
  - Factor Stability Engine (Phase 4)
  - Red Flag Engine (Phase 6)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from factors.data_quality import compute_data_quality, DataQualityScore
from factors.model_confidence import compute_model_confidence, ModelConfidenceResult
from factors.factor_stability import compute_factor_stability, FactorStabilityResult
from factors.red_flags import compute_red_flags, RedFlagResult


# ---------------------------------------------------------------------------
# Data Quality Engine
# ---------------------------------------------------------------------------

class TestDataQuality:
    def _full_record(self) -> dict:
        return {
            "price": 100.0, "mkt_cap": 10e9, "avg_volume": 5e6,
            "quality": 70.0, "momentum": 65.0, "valuation": 60.0,
            "earnings_revisions": 68.0, "earnings_quality": 62.0,
            "moat": 55.0, "capital_allocation": 58.0, "risk_adjustment": 60.0,
            "quality_detail": {"roic": 0.20}, "revisions_detail": {"eps_7d_change": 0.02},
            "moat_detail": {"quant_moat_score": 55.0},
            "earnings_quality_detail": {"fcf_conversion": 0.85},
            "capital_allocation_detail": {"buyback_rate": 0.01},
            "risk_detail": {"risk_label": "Low"},
        }

    def test_full_record_gives_high_score(self):
        from datetime import datetime, timezone, timedelta
        now_iso = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        r = compute_data_quality("AAPL", self._full_record(), now_iso)
        assert isinstance(r, DataQualityScore)
        assert r.overall > 60
        assert r.coverage_score > 80

    def test_empty_record_gives_low_coverage(self):
        r = compute_data_quality("XYZ", {})
        assert r.coverage_score == pytest.approx(0.0)
        assert r.overall < 50

    def test_fresh_data_gives_full_freshness(self):
        from datetime import datetime, timezone, timedelta
        now_iso = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        r = compute_data_quality("AAPL", {}, now_iso)
        assert r.freshness_score == pytest.approx(100.0)

    def test_stale_data_reduces_freshness(self):
        from datetime import datetime, timezone, timedelta
        old_iso = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        r = compute_data_quality("AAPL", {}, old_iso)
        assert r.freshness_score < 50

    def test_label_is_excellent_for_high_score(self):
        from datetime import datetime, timezone, timedelta
        now_iso = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        r = compute_data_quality("AAPL", self._full_record(), now_iso)
        assert r.label in ("Excellent", "Good")

    def test_label_is_poor_for_empty_stale(self):
        from datetime import datetime, timezone, timedelta
        old_iso = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        r = compute_data_quality("XYZ", {}, old_iso)
        assert r.label in ("Poor", "Fair")

    def test_to_dict_has_required_keys(self):
        r = compute_data_quality("AAPL", {})
        d = r.to_dict()
        assert "overall" in d
        assert "freshness_score" in d
        assert "coverage_score" in d
        assert "source_score" in d
        assert "label" in d


# ---------------------------------------------------------------------------
# Model Confidence Engine
# ---------------------------------------------------------------------------

class TestModelConfidence:
    def test_high_inputs_give_high_confidence(self):
        r = compute_model_confidence(
            ticker="AAPL",
            data_quality_score=95.0,
            num_analysts=25,
            years_of_data=5,
            ai_confidence=0.85,
            factor_stability_score=90.0,
            distortion_flags=[],
            sbc_fcf_ratio=0.05,
        )
        assert r.overall_confidence_score >= 80
        assert r.label in ("Very High", "High")

    def test_low_inputs_give_low_confidence(self):
        r = compute_model_confidence(
            ticker="XYZ",
            data_quality_score=30.0,
            num_analysts=1,
            years_of_data=1,
            ai_confidence=0.05,
            factor_stability_score=20.0,
            distortion_flags=["merger_growth"],
            sbc_fcf_ratio=0.60,
        )
        assert r.overall_confidence_score < 50
        assert r.label in ("Low", "Very Low")

    def test_no_analysts_reduces_coverage_score(self):
        r = compute_model_confidence("X", data_quality_score=80, num_analysts=0)
        assert r.analyst_coverage_input < 50

    def test_many_analysts_gives_full_coverage(self):
        r = compute_model_confidence("X", data_quality_score=80, num_analysts=25)
        assert r.analyst_coverage_input == pytest.approx(100.0)

    def test_five_years_gives_full_history(self):
        r = compute_model_confidence("X", data_quality_score=80, years_of_data=5)
        assert r.history_depth_input == pytest.approx(100.0)

    def test_distortion_flags_reduce_accounting_score(self):
        clean = compute_model_confidence("X", data_quality_score=80, distortion_flags=[])
        dirty = compute_model_confidence("X", data_quality_score=80,
                                         distortion_flags=["merger", "sbc", "tax"])
        assert dirty.accounting_quality_input < clean.accounting_quality_input

    def test_to_dict_has_required_keys(self):
        r = compute_model_confidence("X", data_quality_score=70)
        d = r.to_dict()
        assert "overall_confidence_score" in d
        assert "label" in d

    def test_labels_are_valid(self):
        for score, expected_label in [
            (95, "Very High"), (85, "High"), (70, "Medium"), (55, "Low"), (40, "Very Low")
        ]:
            r = compute_model_confidence("X", data_quality_score=score,
                                          factor_stability_score=score,
                                          num_analysts=25 if score > 80 else 2)
            assert r.label in ("Very High", "High", "Medium", "Low", "Very Low")


# ---------------------------------------------------------------------------
# Factor Stability Engine
# ---------------------------------------------------------------------------

class TestFactorStability:
    def test_no_history_gives_neutral_score(self):
        r = compute_factor_stability("AAPL", 65.0, history={})
        assert r.stability_score == pytest.approx(60.0)
        assert r.label == "Moderate"

    def test_stable_scores_give_high_stability(self):
        today_key = "2026-06-21"
        history = {
            "2026-06-14": {"AAPL": 64.5},   # 7 days ago
            "2026-05-22": {"AAPL": 64.0},   # ~30 days ago
            "2026-03-23": {"AAPL": 65.0},   # ~90 days ago
        }
        from unittest.mock import patch
        import datetime
        with patch("factors.factor_stability.date") as mock_date:
            mock_date.today.return_value = datetime.date(2026, 6, 21)
            mock_date.fromisoformat = datetime.date.fromisoformat
            r = compute_factor_stability("AAPL", 65.0, history=history)
        assert r.stability_score >= 80
        assert r.label == "Stable"

    def test_wild_swing_gives_low_stability(self):
        import datetime
        history = {
            "2026-06-14": {"AAPL": 40.0},   # 25-point swing from today's 65
        }
        from unittest.mock import patch
        with patch("factors.factor_stability.date") as mock_date:
            mock_date.today.return_value = datetime.date(2026, 6, 21)
            mock_date.fromisoformat = datetime.date.fromisoformat
            r = compute_factor_stability("AAPL", 65.0, history=history)
        assert r.swing_7d == pytest.approx(25.0)
        assert r.stability_score < 60

    def test_to_dict_has_required_keys(self):
        r = compute_factor_stability("AAPL", 65.0, history={})
        d = r.to_dict()
        assert "stability_score" in d
        assert "label" in d


# ---------------------------------------------------------------------------
# Red Flag Engine
# ---------------------------------------------------------------------------

class TestRedFlags:
    def test_clean_stock_gives_high_score(self):
        r = compute_red_flags(
            ticker="AAPL",
            sbc_fcf_ratio=0.05,
            equity_ratio=0.80,
            revenue_growth=0.10,
            ai_red_flags=[],
            distortion_flags=[],
            capital_allocation_warnings=[],
            net_upgrades_90d=3,
            risk_level="Low",
        )
        assert r.red_flag_score >= 85
        assert r.label == "Clean"

    def test_extreme_sbc_deducts_points(self):
        clean = compute_red_flags("AAPL", sbc_fcf_ratio=0.05)
        dirty = compute_red_flags("AAPL", sbc_fcf_ratio=0.60)
        assert dirty.red_flag_score < clean.red_flag_score
        assert any("SBC" in f for f in dirty.flags)

    def test_high_leverage_deducts_points(self):
        r = compute_red_flags("XYZ", equity_ratio=0.15)
        assert r.red_flag_score < 90
        assert any("leverage" in f.lower() for f in r.flags)

    def test_many_ai_red_flags_reduce_score(self):
        r = compute_red_flags("XYZ", ai_red_flags=["f1","f2","f3","f4","f5","f6"])
        assert r.red_flag_score <= 80
        assert any("AI" in f for f in r.flags)

    def test_net_analyst_downgrades_reduce_score(self):
        r = compute_red_flags("XYZ", net_upgrades_90d=-5)
        assert r.red_flag_score <= 92

    def test_high_risk_label_deducts_points(self):
        low  = compute_red_flags("AAPL", risk_level="Low")
        high = compute_red_flags("AAPL", risk_level="High")
        assert high.red_flag_score < low.red_flag_score

    def test_red_flag_label_for_worst_case(self):
        r = compute_red_flags(
            ticker="BAD",
            sbc_fcf_ratio=0.80,
            equity_ratio=0.10,
            ai_red_flags=["f1","f2","f3","f4","f5","f6"],
            distortion_flags=["merger","sbc","tax"],
            capital_allocation_warnings=["excessive_debt","poor_returns"],
            net_upgrades_90d=-10,
            risk_level="High",
        )
        assert r.label in ("Red Flag", "Warning")
        assert r.red_flag_score < 50

    def test_to_dict_has_required_keys(self):
        r = compute_red_flags("AAPL")
        d = r.to_dict()
        assert "red_flag_score" in d
        assert "flags" in d
        assert "label" in d

    def test_score_clamped_to_0_100(self):
        # Even with no flags, score should never exceed 100
        r = compute_red_flags("PERFECT")
        assert 0 <= r.red_flag_score <= 100
