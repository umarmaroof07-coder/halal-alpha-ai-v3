"""Tests for the value trap guard."""

import pytest
from factors.value_trap import apply_value_trap_guard, apply_value_trap_guard_batch


# ---------------------------------------------------------------------------
# Unit tests — apply_value_trap_guard (works on 0-100 z-scored valuation)
# ---------------------------------------------------------------------------

class TestApplyValueTrapGuard:
    def test_fires_when_all_conditions_met(self):
        score, warning = apply_value_trap_guard("APA", 92.0, -0.08, "Energy")
        assert score == 40.0
        assert warning is not None
        assert "value trap" in warning.lower()

    def test_does_not_fire_positive_revenue(self):
        score, warning = apply_value_trap_guard("X", 92.0, 0.05, "Energy")
        assert score == 92.0
        assert warning is None

    def test_does_not_fire_non_cyclical_sector(self):
        score, warning = apply_value_trap_guard("X", 92.0, -0.10, "Technology")
        assert score == 92.0
        assert warning is None

    def test_does_not_fire_low_valuation(self):
        # 70 is below the 85 threshold
        score, warning = apply_value_trap_guard("X", 70.0, -0.10, "Energy")
        assert score == 70.0
        assert warning is None

    def test_does_not_fire_on_none_revenue(self):
        score, warning = apply_value_trap_guard("X", 92.0, None, "Energy")
        assert score == 92.0
        assert warning is None

    def test_does_not_fire_on_none_valuation(self):
        score, warning = apply_value_trap_guard("X", None, -0.10, "Energy")
        assert score is None
        assert warning is None

    def test_boundary_revenue_exactly_minus5_pct(self):
        # -5% exactly is NOT below -5% threshold (strict <)
        score, warning = apply_value_trap_guard("X", 92.0, -0.05, "Energy")
        assert warning is None

    def test_boundary_revenue_just_below_minus5_pct(self):
        score, warning = apply_value_trap_guard("X", 92.0, -0.051, "Energy")
        assert warning is not None

    def test_boundary_valuation_exactly_at_threshold(self):
        # 85.0 exactly is NOT above threshold (strict >)
        score, warning = apply_value_trap_guard("X", 85.0, -0.10, "Energy")
        assert warning is None

    def test_boundary_valuation_just_above_threshold(self):
        score, warning = apply_value_trap_guard("X", 85.1, -0.10, "Energy")
        assert warning is not None

    def test_materials_sector_fires(self):
        score, warning = apply_value_trap_guard("X", 90.0, -0.10, "Materials")
        assert warning is not None

    def test_oil_gas_sector_fires(self):
        score, warning = apply_value_trap_guard("X", 90.0, -0.10, "Oil & Gas")
        assert warning is not None

    def test_cap_value_is_correct(self):
        score, _ = apply_value_trap_guard("X", 99.0, -0.15, "Energy")
        assert score == 40.0


# ---------------------------------------------------------------------------
# Batch tests — apply_value_trap_guard_batch (operates on FactorScores list)
# ---------------------------------------------------------------------------

class TestApplyValueTrapGuardBatch:
    def _make_fs(self, ticker, valuation, composite=65.0):
        from factors.composite import FactorScores
        fs = FactorScores(ticker=ticker)
        fs.valuation  = valuation
        fs.composite  = composite
        return fs

    def test_batch_caps_matching_tickers(self):
        fs_apa  = self._make_fs("APA",  valuation=92.0, composite=70.0)
        fs_msft = self._make_fs("MSFT", valuation=60.0, composite=65.0)
        weights = {"valuation": 0.10}

        warnings = apply_value_trap_guard_batch(
            tickers        = ["APA", "MSFT"],
            factor_scores  = [fs_apa, fs_msft],
            revenue_growth = {"APA": -0.084, "MSFT": 0.15},
            sector_map     = {"APA": "Energy", "MSFT": "Technology"},
            factor_weights = weights,
        )
        assert "APA" in warnings
        assert "MSFT" not in warnings
        assert fs_apa.valuation == 40.0
        # composite: 70 + (40-92)*0.10 - 5.0 = 70 - 5.2 - 5.0 = 59.8
        assert abs(fs_apa.composite - 59.8) < 0.01
        assert fs_msft.valuation == 60.0   # unchanged
        assert fs_msft.composite == 65.0   # unchanged

    def test_batch_returns_empty_when_no_traps(self):
        fs = self._make_fs("NVDA", valuation=55.0, composite=64.0)

        warnings = apply_value_trap_guard_batch(
            tickers        = ["NVDA"],
            factor_scores  = [fs],
            revenue_growth = {"NVDA": 0.20},
            sector_map     = {"NVDA": "Technology"},
            factor_weights = {"valuation": 0.10},
        )
        assert warnings == {}
        assert fs.valuation == 55.0

    def test_batch_skips_missing_revenue(self):
        fs = self._make_fs("GHOST", valuation=95.0)
        warnings = apply_value_trap_guard_batch(
            tickers        = ["GHOST"],
            factor_scores  = [fs],
            revenue_growth = {},
            sector_map     = {"GHOST": "Energy"},
            factor_weights = {"valuation": 0.10},
        )
        assert warnings == {}
        assert fs.valuation == 95.0   # not capped — revenue unknown
