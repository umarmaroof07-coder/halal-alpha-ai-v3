"""
FMP (Financial Modeling Prep) — primary data provider.
Uses the /stable/ base URL (v3/v4 endpoints are deprecated on current plans).

All public methods return plain dicts/lists or raise ProviderError.
The API key is never logged or included in exception messages.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from config.settings import FMP_API_KEY, FMP_BASE_URL

log = logging.getLogger(__name__)

_TIMEOUT = 15


class ProviderError(Exception):
    """Raised when FMP returns an error or unexpected payload."""


def _get(endpoint: str, params: dict | None = None) -> Any:
    """Internal GET — injects the API key, never surfaces it in logs or exceptions."""
    url = f"{FMP_BASE_URL}/{endpoint}"
    all_params = {"apikey": FMP_API_KEY, **(params or {})}

    try:
        resp = requests.get(url, params=all_params, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise ProviderError(f"FMP network error on {endpoint}: {exc}") from exc

    if resp.status_code != 200:
        try:
            body = resp.json() if resp.content else {}
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        msg = body.get("Error Message") or body.get("message") or f"HTTP {resp.status_code}"
        raise ProviderError(f"FMP error on {endpoint}: {msg}")

    data = resp.json()
    if isinstance(data, dict):
        for err_key in ("Error Message", "message", "error"):
            if err_key in data:
                raise ProviderError(f"FMP error on {endpoint}: {data[err_key]}")

    return data


# ---------------------------------------------------------------------------
# Company profile
# ---------------------------------------------------------------------------

def get_profile(ticker: str) -> dict:
    data = _get("profile", {"symbol": ticker})
    if not data:
        raise ProviderError(f"No profile returned for {ticker}")
    r = data[0] if isinstance(data, list) else data
    return {
        "symbol":           ticker,
        "companyName":      r.get("companyName", ""),
        "sector":           r.get("sector", ""),
        "industry":         r.get("industry", ""),
        "country":          r.get("country", ""),
        "exchange":         r.get("exchange", ""),
        "exchangeShortName": r.get("exchangeShortName", ""),
        "mktCap":           r.get("mktCap"),
        "price":            r.get("price"),
        "description":      r.get("description", ""),
        "website":          r.get("website", ""),
        "ceo":              r.get("ceo", ""),
        "_source": "fmp",
        "_raw": r,
    }


# ---------------------------------------------------------------------------
# Quote / prices
# ---------------------------------------------------------------------------

def get_quote(ticker: str) -> dict:
    data = _get("quote", {"symbol": ticker})
    if not data:
        raise ProviderError(f"No quote returned for {ticker}")
    r = data[0] if isinstance(data, list) else data
    return {
        "symbol":        ticker,
        "price":         r.get("price"),
        "open":          r.get("open"),
        "high":          r.get("dayHigh"),
        "low":           r.get("dayLow"),
        "previousClose": r.get("previousClose"),
        "change":        r.get("change"),
        "changePercent": r.get("changesPercentage"),
        "volume":        r.get("volume"),
        "avgVolume":     r.get("avgVolume"),
        "marketCap":     r.get("marketCap"),
        "pe":            r.get("pe"),
        "_source": "fmp",
        "_raw": r,
    }


def get_historical_prices(ticker: str, from_date: str, to_date: str) -> list[dict]:
    data = _get("historical-price-eod/full", {"symbol": ticker, "from": from_date, "to": to_date})
    rows = data.get("historical", data) if isinstance(data, dict) else data
    result = []
    for r in (rows or []):
        result.append({
            "date":   r.get("date", ""),
            "open":   r.get("open"),
            "high":   r.get("high"),
            "low":    r.get("low"),
            "close":  r.get("close"),
            "volume": r.get("volume"),
            "_source": "fmp",
        })
    # FMP returns newest-first; sort ascending (oldest→newest) for momentum calcs
    result.sort(key=lambda r: r["date"])
    return result


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------

def get_income_statement(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("income-statement", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":            r.get("date", ""),
            "revenue":         r.get("revenue"),
            "netIncome":       r.get("netIncome"),
            "grossProfit":     r.get("grossProfit"),
            "operatingIncome": r.get("operatingIncome"),
            "ebitda":          r.get("ebitda"),
            "eps":             r.get("eps"),
            "interestIncome":  r.get("interestIncome"),
            "_source": "fmp",
            "_raw": r,
        })
    return result


def get_balance_sheet(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("balance-sheet-statement", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":               r.get("date", ""),
            "totalAssets":        r.get("totalAssets"),
            "totalDebt":          r.get("totalDebt"),
            "totalEquity":        r.get("totalStockholdersEquity"),
            "cashAndEquivalents": r.get("cashAndCashEquivalents"),
            "accountsReceivable": r.get("netReceivables"),
            "shortTermInvestments": r.get("shortTermInvestments"),
            "_source": "fmp",
            "_raw": r,
        })
    return result


def get_cash_flow(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("cash-flow-statement", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":               r.get("date", ""),
            "freeCashFlow":       r.get("freeCashFlow"),
            "operatingCashFlow":  r.get("operatingCashFlow"),
            "capitalExpenditure": r.get("capitalExpenditure"),
            "_source": "fmp",
            "_raw": r,
        })
    return result


def get_key_metrics(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("key-metrics", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":            r.get("date", ""),
            "roe":             r.get("returnOnEquity"),
            "roa":             r.get("returnOnAssets"),
            "debtToEquity":    r.get("debtToEquity"),
            "currentRatio":    r.get("currentRatio"),
            "grossMargin":     r.get("grossProfitMargin"),
            "operatingMargin": r.get("operatingCashFlowSalesRatio"),
            "netProfitMargin": r.get("netProfitMargin"),
            "peRatio":         r.get("peRatio"),
            "pbRatio":         r.get("pbRatio"),
            "evToEbitda":      r.get("evToEBITDA"),
            "evToSales":       r.get("evToSales"),
            "fcfPerShare":     r.get("freeCashFlowPerShare"),
            "marketCap":       r.get("marketCap"),
            "_source": "fmp",
            "_raw": r,
        })
    return result


def get_financial_ratios(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("ratios", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":            r.get("date", ""),
            "peRatio":         r.get("priceToEarningsRatio"),
            "pbRatio":         r.get("priceToBookRatio"),
            "debtToEquity":    r.get("debtToEquityRatio"),
            "debtToAssets":    r.get("debtToAssetsRatio"),
            "currentRatio":    r.get("currentRatio"),
            "grossMargin":     r.get("grossProfitMargin"),
            "operatingMargin": r.get("operatingProfitMargin"),
            "netProfitMargin": r.get("netProfitMargin"),
            "evToEbitda":      r.get("enterpriseValueMultiple"),
            "fcfPerShare":     r.get("freeCashFlowPerShare"),
            "_source": "fmp",
            "_raw": r,
        })
    return result


# ---------------------------------------------------------------------------
# Analyst data
# ---------------------------------------------------------------------------

def get_analyst_estimates(ticker: str, period: str = "annual", limit: int = 4) -> list[dict]:
    data = _get("analyst-estimates", {"symbol": ticker, "period": period, "limit": limit})
    result = []
    for r in (data or []):
        result.append({
            "date":                r.get("date", ""),
            "estimatedEpsAvg":     r.get("epsAvg"),
            "estimatedEpsHigh":    r.get("epsHigh"),
            "estimatedEpsLow":     r.get("epsLow"),
            "estimatedRevenueAvg": r.get("revenueAvg"),
            "estimatedRevenueHigh": r.get("revenueHigh"),
            "estimatedRevenueLow": r.get("revenueLow"),
            "numberAnalysts":      r.get("numAnalystsRevenue"),
            "_source": "fmp",
        })
    return result


def get_price_targets(ticker: str) -> list[dict]:
    data = _get("price-target-summary", {"symbol": ticker})
    if not data:
        return []
    r = data[0] if isinstance(data, list) else data
    return [{
        "targetHigh":    r.get("lastQuarterAvgPriceTarget"),
        "targetLow":     r.get("lastYearAvgPriceTarget"),
        "targetMean":    r.get("lastQuarterAvgPriceTarget"),
        "targetAllTime": r.get("allTimeAvgPriceTarget"),
        "analysts":      r.get("allTimeCount"),
        "_source": "fmp",
    }]


def get_recommendation_trends(ticker: str) -> list[dict]:
    data = _get("analyst-stock-recommendations", {"symbol": ticker})
    result = []
    for r in (data if isinstance(data, list) else [])[:6]:
        if not isinstance(r, dict):
            continue
        result.append({
            "period":     r.get("date", "") or r.get("period", ""),
            "strongBuy":  r.get("analystRatingsStrongBuy") or r.get("strongBuy"),
            "buy":        r.get("analystRatingsBuy") or r.get("buy"),
            "hold":       r.get("analystRatingsHold") or r.get("hold"),
            "sell":       r.get("analystRatingsSell") or r.get("sell"),
            "strongSell": r.get("analystRatingsStrongSell") or r.get("strongSell"),
            "_source": "fmp",
        })
    return result


def get_earnings_calendar(ticker: str) -> list[dict]:
    data = _get("earnings-calendar", {"symbol": ticker})
    result = []
    for r in (data or [])[:10]:
        result.append({
            "date":         r.get("date", ""),
            "epsActual":    r.get("eps"),
            "epsEstimate":  r.get("epsEstimated"),
            "revenueActual": r.get("revenue"),
            "revenueEst":   r.get("revenueEstimated"),
            "_source": "fmp",
        })
    return result


# ---------------------------------------------------------------------------
# SEC filings
# ---------------------------------------------------------------------------

def get_sec_filings(ticker: str, filing_type: str = "10-K", limit: int = 5) -> list[dict]:
    # SEC filings require a higher FMP plan tier — returns empty list gracefully
    log.debug("FMP sec_filings not available on current plan for %s", ticker)
    return []


# ---------------------------------------------------------------------------
# Insider trading
# ---------------------------------------------------------------------------

def get_insider_trades(ticker: str, limit: int = 20) -> list[dict]:
    # Insider trades require a higher FMP plan tier — returns empty list gracefully
    log.debug("FMP insider_trades not available on current plan for %s", ticker)
    return []


# ---------------------------------------------------------------------------
# Universe / screening
# ---------------------------------------------------------------------------

def get_stock_screener(
    market_cap_more_than: int = 300_000_000,
    price_lower_than: float | None = None,
    volume_more_than: int | None = None,
    exchange: str = "nasdaq,nyse,amex",
    country: str = "US",
    is_etf: bool = False,
    is_fund: bool = False,
    limit: int = 1000,
) -> list[dict]:
    params: dict = {
        "marketCapMoreThan": market_cap_more_than,
        "exchange": exchange,
        "country": country,
        "isEtf": str(is_etf).lower(),
        "isFund": str(is_fund).lower(),
        "limit": limit,
    }
    if price_lower_than is not None:
        params["priceLowerThan"] = price_lower_than
    if volume_more_than is not None:
        params["volumeMoreThan"] = volume_more_than
    return _get("stock-screener", params) or []


def get_sp500_constituents() -> list[dict]:
    """Fetch S&P 500 constituents from FMP. Returns list of {symbol, name, sector, ...}."""
    return _get("sp500_constituent") or []


def get_russell1000_constituents() -> list[dict]:
    """Fetch Russell 1000 constituents. Returns [] if not available on current plan."""
    try:
        return _get("russell1000_constituent") or []
    except ProviderError:
        log.debug("Russell 1000 constituents not available on current FMP plan.")
        return []


def get_stock_list() -> list[dict]:
    """
    Fetch the full FMP stock list. Returns list of
    {symbol, name, exchange, exchangeShortName, type, price}.
    """
    return _get("stock/list") or []
