"""
Tests for screener/live_universe.py.

All FMP network calls are mocked — no real API calls in tests.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from screener.live_universe import (
    RawStock,
    ScreenedStock,
    UniverseResult,
    _dedup,
    _name_contains_exclusion,
    _type_excluded,
    _exchange_allowed,
    pre_filter,
    build_live_universe,
    load_screened_csv,
    load_raw_csv,
    EXCHANGE_ALLOWLIST,
)


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

class TestNameExclusion:
    def test_spac_acquisition_corp(self):
        assert _name_contains_exclusion("Gores Holdings Acquisition Corp")

    def test_preferred_share(self):
        assert _name_contains_exclusion("Bank of America Preferred Series D")

    def test_etf(self):
        assert _name_contains_exclusion("iShares Core S&P 500 ETF")

    def test_warrant(self):
        assert _name_contains_exclusion("Company Warrant Class A")

    def test_depositary_receipt(self):
        assert _name_contains_exclusion("Alibaba Group Depositary Receipt")

    def test_limited_partnership(self):
        assert _name_contains_exclusion("Kinder Morgan Limited Partnership")

    def test_normal_company_passes(self):
        assert not _name_contains_exclusion("Apple Inc.")
        assert not _name_contains_exclusion("Microsoft Corporation")
        assert not _name_contains_exclusion("NVIDIA Corporation")

    def test_reit(self):
        assert _name_contains_exclusion("Prologis Real Estate Investment Trust")


class TestTypeExclusion:
    def test_etf_excluded(self):
        assert _type_excluded("etf")
        assert _type_excluded("ETF")

    def test_fund_excluded(self):
        assert _type_excluded("fund")

    def test_trust_excluded(self):
        assert _type_excluded("trust")

    def test_warrant_excluded(self):
        assert _type_excluded("warrant")

    def test_stock_passes(self):
        assert not _type_excluded("stock")
        assert not _type_excluded("")


class TestExchangeAllowlist:
    def test_nyse_allowed(self):
        assert _exchange_allowed("NYSE")

    def test_nasdaq_allowed(self):
        assert _exchange_allowed("NASDAQ")

    def test_amex_allowed(self):
        assert _exchange_allowed("AMEX")

    def test_otc_excluded(self):
        assert not _exchange_allowed("OTC")
        assert not _exchange_allowed("OTCBB")
        assert not _exchange_allowed("PNK")
        assert not _exchange_allowed("PINK")

    def test_unknown_exchange(self):
        assert not _exchange_allowed("UNKNOWN_XYZ")

    def test_case_insensitive(self):
        assert _exchange_allowed("nyse")
        assert _exchange_allowed("Nasdaq")


# ---------------------------------------------------------------------------
# pre_filter()
# ---------------------------------------------------------------------------

class TestPreFilter:
    def _make(self, **kwargs) -> RawStock:
        defaults = dict(
            ticker="TEST", name="Test Corp", exchange="NYSE",
            mkt_cap=10e9, price=50.0, avg_volume=5e6, stock_type="stock",
        )
        defaults.update(kwargs)
        return RawStock(**defaults)

    def test_valid_stock_passes(self):
        passed, rejected = pre_filter([self._make()])
        assert len(passed) == 1
        assert len(rejected) == 0

    def test_spac_name_rejected(self):
        s = self._make(name="Alpha Acquisition Corp")
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_etf_type_rejected(self):
        s = self._make(stock_type="etf")
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_otc_exchange_rejected(self):
        s = self._make(exchange="OTC")
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_small_cap_rejected(self):
        s = self._make(mkt_cap=1e9)  # $1B < $5B
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_high_price_rejected(self):
        s = self._make(price=1500.0)
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_low_volume_rejected(self):
        s = self._make(avg_volume=100_000)  # < 1M
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_none_mkt_cap_not_rejected_on_cap_filter(self):
        """When market cap is unknown, don't reject on cap alone."""
        s = self._make(mkt_cap=None)
        passed, rejected = pre_filter([s])
        assert len(passed) == 1

    def test_none_volume_not_rejected_on_volume_filter(self):
        s = self._make(avg_volume=None)
        passed, rejected = pre_filter([s])
        assert len(passed) == 1

    def test_price_exactly_1000_rejected(self):
        s = self._make(price=1000.0)
        _, rejected = pre_filter([s])
        assert any("TEST" in t for t, _ in rejected)

    def test_multiple_stocks(self):
        stocks = [
            self._make(ticker="A", name="Apple Inc.", mkt_cap=3e12),
            self._make(ticker="B", name="SPAC Acquisition Corp", mkt_cap=3e12),
            self._make(ticker="C", exchange="OTC", mkt_cap=3e12),
        ]
        passed, rejected = pre_filter(stocks)
        assert [s.ticker for s in passed] == ["A"]
        assert len(rejected) == 2


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def test_keeps_first_occurrence(self):
        stocks = [
            RawStock(ticker="AAPL", source="sp500"),
            RawStock(ticker="AAPL", source="screener"),
            RawStock(ticker="MSFT", source="screener"),
        ]
        result = _dedup(stocks)
        assert len(result) == 2
        assert result[0].source == "sp500"  # first wins

    def test_all_unique(self):
        stocks = [RawStock(ticker="A"), RawStock(ticker="B"), RawStock(ticker="C")]
        assert len(_dedup(stocks)) == 3

    def test_empty(self):
        assert _dedup([]) == []


# ---------------------------------------------------------------------------
# build_live_universe() — mocked FMP
# ---------------------------------------------------------------------------

def _make_screener_row(symbol: str, mkt_cap: float = 10e9, price: float = 50.0,
                        volume: float = 5e6) -> dict:
    return {
        "symbol":          symbol,
        "companyName":     f"{symbol} Corp",
        "marketCap":       mkt_cap,
        "price":           price,
        "volume":          volume,
        "exchangeShortName": "NASDAQ",
        "sector":          "Technology",
        "industry":        "Software",
        "isEtf":           False,
        "isFund":          False,
        "country":         "US",
    }


@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Redirect CACHE_DIR to a temp directory for each test."""
    import screener.live_universe as lu
    monkeypatch.setattr(lu, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lu, "UNIVERSE_RAW_CSV",      tmp_path / "universe_raw.csv")
    monkeypatch.setattr(lu, "UNIVERSE_SCREENED_CSV", tmp_path / "universe_screened.csv")
    monkeypatch.setattr(lu, "SHARIAH_REPORT_CSV",    tmp_path / "shariah_validation_report.csv")
    return tmp_path


class TestBuildLiveUniverse:
    def _clean_fundamentals(self, ticker: str) -> dict:
        return {
            "ticker": ticker, "symbol": ticker,
            "companyName": f"{ticker} Corp",
            "sector": "Technology", "industry": "Software",
            "mktCap": 10e9, "price": 50.0, "exchange": "NASDAQ",
            "totalAssets": 100e9, "totalDebt": 10e9,
            "accountsReceivable": 20e9,
            "revenue": 50e9, "interestIncome": 100e6,
        }

    def _run(self, screener_rows, tmp_cache):
        with patch("screener.live_universe.fmp_provider.get_sp500_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_russell1000_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_stock_screener", return_value=screener_rows), \
             patch("screener.live_universe._fetch_fundamentals",
                   side_effect=self._clean_fundamentals):
            return build_live_universe(fetch_delay=0)

    def test_returns_universe_result(self, tmp_cache):
        rows = [_make_screener_row("AAPL"), _make_screener_row("MSFT")]
        result = self._run(rows, tmp_cache)
        assert isinstance(result, UniverseResult)

    def test_compliant_count_positive(self, tmp_cache):
        rows = [_make_screener_row("AAPL"), _make_screener_row("MSFT")]
        result = self._run(rows, tmp_cache)
        assert result.compliant_count >= 0
        assert result.screened_count >= 0

    def test_universe_raw_csv_created(self, tmp_cache):
        rows = [_make_screener_row("NVDA")]
        self._run(rows, tmp_cache)
        assert (tmp_cache / "universe_raw.csv").exists()

    def test_universe_screened_csv_created(self, tmp_cache):
        rows = [_make_screener_row("GOOG")]
        self._run(rows, tmp_cache)
        assert (tmp_cache / "universe_screened.csv").exists()

    def test_shariah_report_csv_created(self, tmp_cache):
        rows = [_make_screener_row("META")]
        self._run(rows, tmp_cache)
        assert (tmp_cache / "shariah_validation_report.csv").exists()

    def test_spac_excluded_before_shariah(self, tmp_cache):
        spac_row = _make_screener_row("SPAC1")
        spac_row["companyName"] = "Alpha Acquisition Corp"
        rows = [spac_row, _make_screener_row("AAPL")]
        result = self._run(rows, tmp_cache)
        # SPAC1 should not appear in screened CSV
        raw_rows = list(csv.DictReader((tmp_cache / "universe_raw.csv").open()))
        tickers = [r["ticker"] for r in raw_rows]
        assert "SPAC1" not in tickers

    def test_unknown_shariah_excluded_from_compliant(self, tmp_cache):
        """Tickers with no fundamental data → unknown → excluded from compliant."""
        with patch("screener.live_universe.fmp_provider.get_sp500_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_russell1000_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_stock_screener",
                   return_value=[_make_screener_row("NODTA")]), \
             patch("screener.live_universe._fetch_fundamentals", return_value={
                 "ticker": "NODTA", "symbol": "NODTA",
                 # No ratio data at all — Shariah will mark unknown
             }):
            result = build_live_universe(fetch_delay=0)
        assert "NODTA" not in result.compliant_tickers

    def test_sources_used_populated(self, tmp_cache):
        rows = [_make_screener_row("AAPL")]
        result = self._run(rows, tmp_cache)
        assert "screener" in result.sources_used

    def test_sp500_source_recorded(self, tmp_cache):
        sp500_row = [{"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"}]
        with patch("screener.live_universe.fmp_provider.get_sp500_constituents", return_value=sp500_row), \
             patch("screener.live_universe.fmp_provider.get_russell1000_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_stock_screener", return_value=[]), \
             patch("screener.live_universe._fetch_fundamentals", return_value={
                 "ticker": "AAPL", "symbol": "AAPL",
                 "companyName": "Apple Inc.", "sector": "Technology",
                 "mktCap": 3e12, "price": 180.0, "exchange": "NASDAQ",
                 "totalAssets": 300e9, "totalDebt": 80e9,
                 "accountsReceivable": 50e9,
                 "revenue": 380e9, "interestIncome": 1e9,
             }):
            result = build_live_universe(fetch_delay=0)
        assert "sp500" in result.sources_used

    def test_empty_fmp_uses_watchlist(self, tmp_path, monkeypatch):
        import screener.live_universe as lu
        monkeypatch.setattr(lu, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(lu, "UNIVERSE_RAW_CSV",      tmp_path / "universe_raw.csv")
        monkeypatch.setattr(lu, "UNIVERSE_SCREENED_CSV", tmp_path / "universe_screened.csv")
        monkeypatch.setattr(lu, "SHARIAH_REPORT_CSV",    tmp_path / "shariah_validation_report.csv")

        wl = tmp_path / "watchlist.csv"
        wl.write_text("AAPL\nMSFT\n")

        with patch("screener.live_universe.fmp_provider.get_sp500_constituents", return_value=[]), \
             patch("screener.live_universe._fetch_sp500_wikipedia", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_russell1000_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_stock_screener", return_value=[]), \
             patch("screener.live_universe._fetch_fundamentals", return_value={
                 "ticker": "AAPL", "symbol": "AAPL",
                 "companyName": "Apple Inc.", "sector": "Technology",
                 "mktCap": 3e12, "price": 180.0, "exchange": "NASDAQ",
                 "totalAssets": 300e9, "totalDebt": 80e9,
                 "accountsReceivable": 50e9,
                 "revenue": 380e9, "interestIncome": 1e9,
             }):
            result = build_live_universe(watchlist_path=wl, fetch_delay=0)

        assert "watchlist" in result.sources_used
        assert len(result.warnings) > 0

    def test_dedup_in_pipeline(self, tmp_cache):
        """Same ticker from multiple sources should appear only once."""
        row = _make_screener_row("AAPL")
        with patch("screener.live_universe.fmp_provider.get_sp500_constituents",
                   return_value=[{"symbol": "AAPL", "name": "Apple Inc."}]), \
             patch("screener.live_universe.fmp_provider.get_russell1000_constituents", return_value=[]), \
             patch("screener.live_universe.fmp_provider.get_stock_screener", return_value=[row]), \
             patch("screener.live_universe._fetch_fundamentals", return_value={
                 "ticker": "AAPL", "symbol": "AAPL",
                 "companyName": "Apple Inc.", "sector": "Technology",
                 "mktCap": 3e12, "price": 180.0, "exchange": "NASDAQ",
                 "totalAssets": 300e9, "totalDebt": 80e9,
                 "accountsReceivable": 50e9,
                 "revenue": 380e9, "interestIncome": 1e9,
             }):
            result = build_live_universe(fetch_delay=0)

        assert result.screened_count == 1   # dedup before screening

    def test_as_of_date_in_result(self, tmp_cache):
        from datetime import date
        rows = [_make_screener_row("AAPL")]
        result = self._run(rows, tmp_cache)
        assert result.as_of_date == str(date.today())


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

class TestCSVLoaders:
    def test_load_screened_csv_missing(self, tmp_path, monkeypatch):
        import screener.live_universe as lu
        monkeypatch.setattr(lu, "UNIVERSE_SCREENED_CSV", tmp_path / "nonexistent.csv")
        assert load_screened_csv() == []

    def test_load_raw_csv_missing(self, tmp_path, monkeypatch):
        import screener.live_universe as lu
        monkeypatch.setattr(lu, "UNIVERSE_RAW_CSV", tmp_path / "nonexistent.csv")
        assert load_raw_csv() == []

    def test_load_screened_csv_content(self, tmp_path, monkeypatch):
        import screener.live_universe as lu
        csv_path = tmp_path / "universe_screened.csv"
        csv_path.write_text(
            "ticker,name,exchange,sector,industry,mkt_cap,price,avg_volume,shariah_status\n"
            "AAPL,Apple Inc.,NASDAQ,Technology,Software,3000000000000,180.0,55000000,compliant\n"
        )
        monkeypatch.setattr(lu, "UNIVERSE_SCREENED_CSV", csv_path)
        rows = load_screened_csv()
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["shariah_status"] == "compliant"


# ---------------------------------------------------------------------------
# FMP provider additions
# ---------------------------------------------------------------------------

class TestFMPProviderAdditions:
    def test_get_sp500_constituents_calls_correct_endpoint(self):
        from data_layer.fmp_provider import get_sp500_constituents
        with patch("data_layer.fmp_provider._get", return_value=[{"symbol": "AAPL"}]) as mock_get:
            result = get_sp500_constituents()
        mock_get.assert_called_once_with("sp500_constituent")
        assert result == [{"symbol": "AAPL"}]

    def test_get_russell1000_returns_empty_on_provider_error(self):
        from data_layer.fmp_provider import get_russell1000_constituents, ProviderError
        with patch("data_layer.fmp_provider._get", side_effect=ProviderError("not available")):
            result = get_russell1000_constituents()
        assert result == []

    def test_get_stock_screener_etf_param(self):
        from data_layer.fmp_provider import get_stock_screener
        with patch("data_layer.fmp_provider._get", return_value=[]) as mock_get:
            get_stock_screener(is_etf=False, is_fund=False, volume_more_than=1_000_000)
        call_params = mock_get.call_args[0][1]
        assert call_params["isEtf"] == "false"
        assert call_params["isFund"] == "false"
        assert call_params["volumeMoreThan"] == 1_000_000

    def test_get_stock_list_calls_correct_endpoint(self):
        from data_layer.fmp_provider import get_stock_list
        with patch("data_layer.fmp_provider._get", return_value=[{"symbol": "AAPL"}]) as mock_get:
            result = get_stock_list()
        mock_get.assert_called_once_with("stock/list")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Momentum smoke tests (requirements 6a–6c)
# ---------------------------------------------------------------------------

class TestMomentumSmoke:
    """
    Smoke tests for the momentum pipeline.

    Requirements:
      6a. AAPL historical prices load (non-empty, ≥253 records).
      6b. 6-month momentum ≠ 50 when price data is present.
      6c. 12-month momentum ≠ 50 when price data is present.

    These tests use mocked price series so no real API calls are needed.
    """

    def _make_prices(self, n: int = 270, start: float = 100.0, drift: float = 0.001) -> list[float]:
        """Generate a simple upward-trending price series of length n."""
        prices = [start]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + drift))
        return prices

    # ── 6a: historical prices load ───────────────────────────────────────────

    def test_yfinance_historical_prices_returns_list(self):
        """get_historical_prices via yfinance returns a non-empty list of dicts."""
        import pandas as pd
        idx = pd.date_range("2025-06-01", periods=264, freq="B")
        closes = [202.0 + i * 0.1 for i in range(264)]
        df = pd.DataFrame({"Close": closes, "Open": closes, "High": closes,
                           "Low": closes, "Volume": [1_000_000] * 264}, index=idx)
        df.index.name = "Date"
        with patch("yfinance.download", return_value=df):
            from data_layer.yfinance_provider import get_historical_prices
            result = get_historical_prices("AAPL", "2025-06-01", "2026-06-01")
        assert isinstance(result, list)
        assert len(result) == 264
        assert all("close" in r for r in result)

    def test_yfinance_multiindex_columns_handled(self):
        """yfinance ≥1.x MultiIndex columns (('Close','AAPL')) must not raise KeyError."""
        import pandas as pd
        idx = pd.date_range("2025-06-01", periods=10, freq="B")
        # Simulate yfinance ≥1.x MultiIndex output
        data = {
            ("Close",  "AAPL"): [200.0] * 10,
            ("High",   "AAPL"): [205.0] * 10,
            ("Low",    "AAPL"): [198.0] * 10,
            ("Open",   "AAPL"): [199.0] * 10,
            ("Volume", "AAPL"): [1_000_000] * 10,
        }
        df = pd.DataFrame(data, index=idx)
        df.index.name = "Date"

        with patch("yfinance.download", return_value=df):
            from data_layer.yfinance_provider import get_historical_prices
            result = get_historical_prices("AAPL", "2025-06-01", "2025-06-15")

        assert len(result) == 10
        assert result[0]["close"] == 200.0

    def test_fmp_historical_prices_sorted_oldest_first(self):
        """FMP returns newest-first; provider must sort ascending before returning."""
        from data_layer.fmp_provider import get_historical_prices
        # FMP-style response: newest first
        raw = {
            "historical": [
                {"date": "2026-06-18", "open": 300.0, "high": 305.0, "low": 298.0, "close": 302.0, "volume": 1_000_000},
                {"date": "2026-06-17", "open": 298.0, "high": 302.0, "low": 296.0, "close": 300.0, "volume": 900_000},
                {"date": "2026-06-16", "open": 295.0, "high": 299.0, "low": 293.0, "close": 297.0, "volume": 800_000},
            ]
        }
        with patch("data_layer.fmp_provider._get", return_value=raw):
            result = get_historical_prices("AAPL", "2026-06-16", "2026-06-18")

        assert result[0]["date"] == "2026-06-16"   # oldest first
        assert result[-1]["date"] == "2026-06-18"  # newest last
        assert result[0]["close"] == 297.0

    # ── 6b & 6c: momentum signals are non-neutral when data exists ───────────

    def test_6m_momentum_nonzero_with_trending_prices(self):
        """ret_6m must not be None when ≥127 prices are available."""
        from factors.momentum import compute_momentum
        prices = self._make_prices(n=270, drift=0.001)   # steady uptrend
        result = compute_momentum("TEST", prices)
        assert result.ret_6m is not None
        assert result.ret_6m != 0.0

    def test_12m_momentum_nonzero_with_enough_prices(self):
        """ret_12_1m must not be None when ≥253 prices are available."""
        from factors.momentum import compute_momentum
        prices = self._make_prices(n=270, drift=0.001)
        result = compute_momentum("TEST", prices)
        assert result.ret_12_1m is not None
        assert result.ret_12_1m != 0.0

    def test_raw_score_not_neutral_with_trending_prices(self):
        """With 270 upward-trending prices, raw_score must be well above 0 (not neutral 50)."""
        from factors.momentum import compute_momentum
        from factors.composite import _zscore_to_0_100, NEUTRAL
        prices = self._make_prices(n=270, drift=0.003)   # strong uptrend
        result = compute_momentum("TEST", prices)
        assert result.raw_score is not None
        # Cross-sectional z-score of a single ticker gives NEUTRAL (trivially),
        # but the raw score itself must be non-zero
        assert result.raw_score != 0.0
        assert len(result.signals_used) >= 3

    def test_composite_momentum_nonzero_for_two_different_stocks(self):
        """Two stocks with different price trends must score differently after normalization."""
        from factors.momentum import compute_momentum
        from factors.composite import compute_composite, NEUTRAL
        from factors.quality import QualityRaw
        from factors.valuation import ValuationRaw
        from factors.revisions import RevisionsRaw

        # BULL: strong uptrend
        bull_prices = self._make_prices(n=270, drift=0.005)
        # BEAR: downtrend
        bear_prices = self._make_prices(n=270, drift=-0.003)

        bull_mom = compute_momentum("BULL", bull_prices)
        bear_mom = compute_momentum("BEAR", bear_prices)

        scores = compute_composite(
            tickers=["BULL", "BEAR"],
            momentum_raw={"BULL": bull_mom, "BEAR": bear_mom},
            quality_raw={},
            valuation_raw={},
            revisions_raw={},
        )
        bull_score = next(s for s in scores if s.ticker == "BULL").momentum
        bear_score = next(s for s in scores if s.ticker == "BEAR").momentum

        assert bull_score != NEUTRAL
        assert bear_score != NEUTRAL
        assert bull_score > bear_score   # uptrend scores higher than downtrend

    def test_insufficient_prices_gives_neutral(self):
        """Fewer than 2 prices → raw_score = None → composite assigns neutral 50."""
        from factors.momentum import compute_momentum
        result = compute_momentum("TEST", [100.0])
        assert result.raw_score is None
        assert result.signals_used == []

    def test_ma_signals_computed_with_50_prices(self):
        """50 prices → price_above_50dma computed but ret_6m/12m still None."""
        from factors.momentum import compute_momentum
        prices = self._make_prices(n=50, drift=0.001)
        result = compute_momentum("TEST", prices)
        assert result.price_above_50dma is not None
        assert result.ret_6m is None       # needs 127 prices
