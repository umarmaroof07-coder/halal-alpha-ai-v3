"""
Tests for the Shariah screener — Phase 3.

Covers:
  - Industry normalization (all punctuation variants)
  - Banks, insurance, gambling, alcohol, tobacco
  - SPACs
  - Unknown / missing data
  - Manual CSV overrides (ratio only, not industry)
  - Compliant pass-through
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from screener.exclusion_list import normalize_industry, is_haram_industry, is_spac
from screener.shariah_filter import screen_ticker, ShariahResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_RATIOS = dict(
    total_debt=100,
    total_assets=1_000,        # debt/assets = 10% ✓
    interest_income=1,
    total_revenue=100,         # interest/revenue = 1% ✓
    accounts_receivable=50,    # AR/assets = 5% ✓
)


def make_stock(
    ticker="TEST",
    industry="Software",
    sector="Technology",
    company_name="Test Corp",
    **ratio_kwargs,
) -> dict:
    ratios = {**GOOD_RATIOS, **ratio_kwargs}
    return dict(
        ticker=ticker,
        industry=industry,
        sector=sector,
        company_name=company_name,
        **ratios,
    )


def screen(**kwargs) -> ShariahResult:
    stock = make_stock(**kwargs)
    return screen_ticker(
        ticker              = stock["ticker"],
        industry            = stock["industry"],
        sector              = stock["sector"],
        company_name        = stock["company_name"],
        total_debt          = stock.get("total_debt"),
        total_assets        = stock.get("total_assets"),
        interest_income     = stock.get("interest_income"),
        total_revenue       = stock.get("total_revenue"),
        accounts_receivable = stock.get("accounts_receivable"),
        manual_overrides    = {},
    )


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_em_dash(self):
        assert normalize_industry("Banks—Regional") == "banks regional"

    def test_hyphen_with_spaces(self):
        assert normalize_industry("Banks - Regional") == "banks regional"

    def test_hyphen_no_spaces(self):
        assert normalize_industry("Banks-Regional") == "banks regional"

    def test_all_caps_with_spaces(self):
        assert normalize_industry("BANKS - REGIONAL") == "banks regional"

    def test_parentheses_stripped(self):
        assert normalize_industry("Real Estate (REITs)") == "real estate reits"

    def test_en_dash(self):
        assert normalize_industry("Oil–Gas") == "oil gas"

    def test_multiple_spaces_collapsed(self):
        assert normalize_industry("consumer   staples") == "consumer staples"

    def test_empty_string(self):
        assert normalize_industry("") == ""


# ---------------------------------------------------------------------------
# 2. Banks
# ---------------------------------------------------------------------------

class TestBanks:
    def test_banks_regional_em_dash(self):
        assert is_haram_industry("Banks—Regional") is True

    def test_banks_regional_hyphen(self):
        assert is_haram_industry("Banks-Regional") is True

    def test_banks_regional_spaces(self):
        assert is_haram_industry("Banks - Regional") is True

    def test_banks_regional_all_caps(self):
        assert is_haram_industry("BANKS - REGIONAL") is True

    def test_diversified_banks(self):
        assert is_haram_industry("Diversified Banks") is True

    def test_bank_screen_result(self):
        r = screen(industry="Banks—Regional", sector="Financial Services")
        assert r.status == "non_compliant"
        assert r.industry_pass is False

    def test_bank_cannot_be_overridden(self):
        r = screen_ticker(
            ticker="JPM", industry="Banks—Regional", sector="Financial Services",
            company_name="JP Morgan",
            total_debt=None, total_assets=None, interest_income=None,
            total_revenue=None, accounts_receivable=None,
            manual_overrides={"JPM": "compliant"},  # should have no effect
        )
        assert r.status == "non_compliant"
        assert r.manual_override is False


# ---------------------------------------------------------------------------
# 3. Insurance
# ---------------------------------------------------------------------------

class TestInsurance:
    def test_insurance_industry(self):
        assert is_haram_industry("Insurance") is True

    def test_life_insurance(self):
        assert is_haram_industry("Life Insurance") is True

    def test_property_casualty_insurance(self):
        assert is_haram_industry("Property & Casualty Insurance") is True

    def test_reinsurance(self):
        assert is_haram_industry("Reinsurance") is True

    def test_insurance_screen_result(self):
        r = screen(industry="Property & Casualty Insurance", sector="Financial Services")
        assert r.status == "non_compliant"
        assert r.industry_pass is False


# ---------------------------------------------------------------------------
# 4. Gambling
# ---------------------------------------------------------------------------

class TestGambling:
    def test_casino(self):
        assert is_haram_industry("Casino & Gaming") is True

    def test_casinos_resorts(self):
        assert is_haram_industry("Casinos & Resorts") is True

    def test_sports_betting(self):
        assert is_haram_industry("Sports Betting") is True

    def test_lottery(self):
        assert is_haram_industry("Lottery") is True

    def test_video_gaming_not_gambling(self):
        # "video gaming" should NOT match "gaming" haram keyword
        assert is_haram_industry("Video Gaming Software") is False

    def test_gambling_screen_result(self):
        r = screen(industry="Casino & Gaming", sector="Consumer Discretionary")
        assert r.status == "non_compliant"
        assert r.industry_pass is False


# ---------------------------------------------------------------------------
# 5. Alcohol
# ---------------------------------------------------------------------------

class TestAlcohol:
    def test_brewery(self):
        assert is_haram_industry("Brewery") is True

    def test_distillers_vintners(self):
        assert is_haram_industry("Distillers & Vintners") is True

    def test_wine(self):
        assert is_haram_industry("Wine Production") is True

    def test_spirits(self):
        assert is_haram_industry("Spirits & Liquor") is True

    def test_alcohol_screen_result(self):
        r = screen(industry="Brewery", sector="Consumer Staples")
        assert r.status == "non_compliant"
        assert r.industry_pass is False


# ---------------------------------------------------------------------------
# 6. Tobacco
# ---------------------------------------------------------------------------

class TestTobacco:
    def test_tobacco(self):
        assert is_haram_industry("Tobacco") is True

    def test_cigarettes(self):
        assert is_haram_industry("Cigarette Manufacturing") is True

    def test_tobacco_screen_result(self):
        r = screen(industry="Tobacco", sector="Consumer Staples")
        assert r.status == "non_compliant"
        assert r.industry_pass is False


# ---------------------------------------------------------------------------
# 7. SPACs
# ---------------------------------------------------------------------------

class TestSPACs:
    def test_spac_industry(self):
        assert is_haram_industry("Special Purpose Acquisition") is True

    def test_spac_blank_check(self):
        assert is_haram_industry("Blank Check Company") is True

    def test_spac_name_detection(self):
        assert is_spac("Churchill Capital Acquisition Corp") is True

    def test_spac_screen_result(self):
        r = screen(
            ticker="SPAC1",
            industry="Blank Check",
            sector="Financial Services",
            company_name="Alpha Acquisition Corp",
        )
        assert r.status == "non_compliant"

    def test_spac_cannot_be_overridden(self):
        r = screen_ticker(
            ticker="SPAC1", industry="Blank Check", sector="Financial Services",
            company_name="Alpha Acquisition Corp",
            total_debt=None, total_assets=None, interest_income=None,
            total_revenue=None, accounts_receivable=None,
            manual_overrides={"SPAC1": "compliant"},
        )
        assert r.status == "non_compliant"
        assert r.manual_override is False


# ---------------------------------------------------------------------------
# 8. Unknown / missing data
# ---------------------------------------------------------------------------

class TestUnknownData:
    def test_missing_industry_is_unknown(self):
        r = screen_ticker(
            ticker="UNKWN", industry="", sector="",
            company_name="Unknown Corp",
            total_debt=None, total_assets=None,
            interest_income=None, total_revenue=None,
            accounts_receivable=None,
            manual_overrides={},
        )
        assert r.status == "unknown"

    def test_missing_ratio_data_is_unknown_not_compliant(self):
        r = screen_ticker(
            ticker="NODATA", industry="Software", sector="Technology",
            company_name="No Data Corp",
            total_debt=None, total_assets=None,     # missing
            interest_income=None, total_revenue=None,
            accounts_receivable=None,
            manual_overrides={},
        )
        assert r.status == "unknown"
        assert r.status != "compliant"

    def test_partial_ratio_data_is_unknown(self):
        r = screen_ticker(
            ticker="PART", industry="Software", sector="Technology",
            company_name="Partial Corp",
            total_debt=100, total_assets=None,      # assets missing
            interest_income=None, total_revenue=None,
            accounts_receivable=None,
            manual_overrides={},
        )
        assert r.status == "unknown"


# ---------------------------------------------------------------------------
# 9. Manual CSV overrides
# ---------------------------------------------------------------------------

class TestManualOverrides:
    def test_override_applies_to_ratio_failure(self):
        # Debt/assets = 50% > 33% threshold → would fail ratio screen
        r = screen_ticker(
            ticker="HIDB", industry="Software", sector="Technology",
            company_name="High Debt Corp",
            total_debt=500, total_assets=1_000,    # 50% > 33%
            interest_income=1, total_revenue=100,
            accounts_receivable=50,
            manual_overrides={"HIDB": "compliant"},
        )
        assert r.status == "compliant"
        assert r.manual_override is True

    def test_override_does_not_apply_to_industry_failure(self):
        r = screen_ticker(
            ticker="GS", industry="Banks—Regional", sector="Financial Services",
            company_name="Goldman Sachs",
            total_debt=100, total_assets=1_000,
            interest_income=1, total_revenue=100,
            accounts_receivable=50,
            manual_overrides={"GS": "compliant"},  # must NOT work
        )
        assert r.status == "non_compliant"
        assert r.manual_override is False

    def test_no_override_when_not_in_csv(self):
        r = screen_ticker(
            ticker="HIDB2", industry="Software", sector="Technology",
            company_name="High Debt Corp 2",
            total_debt=500, total_assets=1_000,
            interest_income=1, total_revenue=100,
            accounts_receivable=50,
            manual_overrides={},   # empty — no override
        )
        assert r.status == "non_compliant"
        assert r.manual_override is False


# ---------------------------------------------------------------------------
# 10. Compliant pass-through
# ---------------------------------------------------------------------------

class TestCompliant:
    def test_clean_tech_company(self):
        r = screen(industry="Semiconductors", sector="Technology")
        assert r.status == "compliant"
        assert r.industry_pass is True
        assert r.ratio_pass is True

    def test_healthcare_device(self):
        r = screen(industry="Medical Devices", sector="Health Care")
        assert r.status == "compliant"

    def test_ratio_debt_exactly_at_limit(self):
        # 33% exactly — should pass (threshold is >, not >=)
        r = screen(total_debt=330, total_assets=1_000)
        assert r.status == "compliant"

    def test_ratio_debt_just_over_limit(self):
        r = screen(total_debt=331, total_assets=1_000)
        assert r.status == "non_compliant"
        assert r.ratio_pass is False

    def test_interest_income_exactly_at_limit(self):
        r = screen(interest_income=5, total_revenue=100)   # 5% exactly
        assert r.status == "compliant"

    def test_interest_income_just_over_limit(self):
        r = screen(interest_income=6, total_revenue=100)   # 6% > 5%
        assert r.status == "non_compliant"
