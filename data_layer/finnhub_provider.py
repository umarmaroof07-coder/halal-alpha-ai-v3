"""
Finnhub — primary data provider.

All public methods return plain dicts/lists or raise ProviderError.
The API key is passed via header, never logged or included in exceptions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config.settings import FINNHUB_API_KEY, FINNHUB_BASE_URL

log = logging.getLogger(__name__)

_TIMEOUT = 15


class ProviderError(Exception):
    """Raised when Finnhub returns an error or unexpected payload."""


def _get(endpoint: str, params: dict | None = None) -> Any:
    """Internal GET — injects the API key via header, never in logs."""
    url = f"{FINNHUB_BASE_URL}/{endpoint}"
    headers = {"X-Finnhub-Token": FINNHUB_API_KEY}

    try:
        resp = requests.get(url, params=params or {}, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise ProviderError(f"Finnhub network error on {endpoint}: {exc}") from exc

    if resp.status_code == 429:
        log.warning("Finnhub rate limit (429) on %s — backing off 1s", endpoint)
        time.sleep(1)
        raise ProviderError(f"Finnhub rate limit hit on {endpoint}")

    if resp.status_code != 200:
        raise ProviderError(f"Finnhub returned HTTP {resp.status_code} for {endpoint}")

    data = resp.json()

    if isinstance(data, dict) and data.get("error"):
        raise ProviderError(f"Finnhub error on {endpoint}: {data['error']}")

    return data


# ---------------------------------------------------------------------------
# Company profile
# ---------------------------------------------------------------------------

def get_profile(ticker: str) -> dict:
    data = _get("stock/profile2", {"symbol": ticker})
    if not data or not data.get("name"):
        raise ProviderError(f"No profile returned for {ticker}")
    return {
        "symbol": ticker,
        "companyName": data.get("name", ""),
        "sector": data.get("finnhubIndustry", ""),
        "industry": data.get("finnhubIndustry", ""),
        "country": data.get("country", ""),
        "exchange": data.get("exchange", ""),
        "exchangeShortName": data.get("exchange", ""),
        "mktCap": data.get("marketCapitalization", 0) * 1_000_000,
        "price": data.get("ipo"),          # profile doesn't carry live price
        "description": data.get("weburl", ""),
        "logo": data.get("logo", ""),
        "weburl": data.get("weburl", ""),
        "ipo": data.get("ipo", ""),
        "_source": "finnhub",
        "_raw": data,
    }


# ---------------------------------------------------------------------------
# Quote / prices
# ---------------------------------------------------------------------------

def get_quote(ticker: str) -> dict:
    data = _get("quote", {"symbol": ticker})
    if data.get("c") is None:
        raise ProviderError(f"No quote returned for {ticker}")
    return {
        "symbol": ticker,
        "price": data.get("c"),           # current price
        "open": data.get("o"),
        "high": data.get("h"),
        "low": data.get("l"),
        "previousClose": data.get("pc"),
        "change": data.get("d"),
        "changePercent": data.get("dp"),
        "volume": None,                   # not in Finnhub quote; fetched via candles
        "_source": "finnhub",
        "_raw": data,
    }


def get_historical_prices(ticker: str, from_date: str, to_date: str) -> list[dict]:
    from datetime import datetime
    from_ts = int(datetime.strptime(from_date, "%Y-%m-%d").timestamp())
    to_ts   = int(datetime.strptime(to_date,   "%Y-%m-%d").timestamp())

    data = _get("stock/candle", {
        "symbol": ticker,
        "resolution": "D",
        "from": from_ts,
        "to": to_ts,
    })

    if data.get("s") != "ok" or not data.get("t"):
        return []

    records = []
    for i, ts in enumerate(data["t"]):
        records.append({
            "date": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
            "open":   data["o"][i],
            "high":   data["h"][i],
            "low":    data["l"][i],
            "close":  data["c"][i],
            "volume": data["v"][i],
            "_source": "finnhub",
        })
    return records


# ---------------------------------------------------------------------------
# Financial statements  (Finnhub basic financials + reported financials)
# ---------------------------------------------------------------------------

def _reported_financials(ticker: str, statement: str, freq: str = "annual", limit: int = 5) -> list[dict]:
    """
    statement: "ic" | "bs" | "cf"
    freq:      "annual" | "quarterly"
    """
    data = _get("stock/financials", {
        "symbol": ticker,
        "statement": statement,
        "freq": freq,
    })
    reports = (data.get("financials") or {}).get("financials") or []
    return reports[:limit]


def get_income_statement(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    freq = "annual" if period == "annual" else "quarterly"
    rows = _reported_financials(ticker, "ic", freq, limit)
    result = []
    for r in rows:
        result.append({
            "date":        r.get("period", ""),
            "revenue":     r.get("revenue"),
            "netIncome":   r.get("netIncome"),
            "eps":         r.get("eps"),
            "ebitda":      r.get("ebitda"),
            "grossProfit": r.get("grossProfit"),
            "operatingIncome": r.get("operatingIncome"),
            "interestIncome":  r.get("interestExpense"),   # proxy
            "_source": "finnhub",
            "_raw": r,
        })
    return result


def get_balance_sheet(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    freq = "annual" if period == "annual" else "quarterly"
    rows = _reported_financials(ticker, "bs", freq, limit)
    result = []
    for r in rows:
        result.append({
            "date":          r.get("period", ""),
            "totalAssets":   r.get("totalAssets"),
            "totalDebt":     r.get("totalDebt") or r.get("longTermDebt"),
            "totalEquity":   r.get("totalEquity") or r.get("shareholderEquity"),
            "cashAndEquivalents": r.get("cashAndEquivalents"),
            "accountsReceivable": r.get("receivables"),
            "shortTermInvestments": r.get("otherCurrentAssets"),
            "_source": "finnhub",
            "_raw": r,
        })
    return result


def get_cash_flow(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    freq = "annual" if period == "annual" else "quarterly"
    rows = _reported_financials(ticker, "cf", freq, limit)
    result = []
    for r in rows:
        result.append({
            "date":               r.get("period", ""),
            "freeCashFlow":       r.get("freeCashFlow"),
            "operatingCashFlow":  r.get("operatingCashFlow"),
            "capitalExpenditure": r.get("capitalExpenditure"),
            "_source": "finnhub",
            "_raw": r,
        })
    return result


def get_key_metrics(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    data = _get("stock/metric", {"symbol": ticker, "metric": "all"})
    metric = data.get("metric", {})
    if not metric:
        return []
    return [{
        "roe":            metric.get("roeRfy") or metric.get("roeTTM"),
        "debtToEquity":   metric.get("totalDebt/totalEquityAnnual"),
        "currentRatio":   metric.get("currentRatioAnnual"),
        "grossMargin":    metric.get("grossMarginTTM"),
        "operatingMargin": metric.get("operatingMarginTTM"),
        "netProfitMargin": metric.get("netProfitMarginTTM"),
        "revenueGrowth":  metric.get("revenueGrowthTTMYoy"),
        "peRatio":        metric.get("peBasicExclExtraTTM"),
        "pbRatio":        metric.get("pbAnnual"),
        "evToEbitda":     metric.get("evToEbitda"),
        "_source": "finnhub",
        "_raw": metric,
    }]


def get_financial_ratios(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return get_key_metrics(ticker, period, limit)


# ---------------------------------------------------------------------------
# Analyst data
# ---------------------------------------------------------------------------

def get_analyst_estimates(ticker: str, period: str = "annual", limit: int = 4) -> list[dict]:
    freq = "annual" if period == "annual" else "quarterly"
    data = _get("stock/eps-estimate", {"symbol": ticker, "freq": freq})
    rows = (data.get("data") or [])[:limit]
    result = []
    for r in rows:
        result.append({
            "date":               r.get("period", ""),
            "estimatedEpsAvg":    r.get("epsAvg"),
            "estimatedEpsHigh":   r.get("epsHigh"),
            "estimatedEpsLow":    r.get("epsLow"),
            "estimatedRevenueAvg": r.get("revenueAvg"),
            "numberAnalysts":     r.get("numberAnalysts"),
            "_source": "finnhub",
        })
    return result


def get_price_targets(ticker: str) -> list[dict]:
    data = _get("stock/price-target", {"symbol": ticker})
    if not data:
        return []
    return [{
        "targetHigh":    data.get("targetHigh"),
        "targetLow":     data.get("targetLow"),
        "targetMean":    data.get("targetMean"),
        "targetMedian":  data.get("targetMedian"),
        "lastUpdated":   data.get("lastUpdated"),
        "_source": "finnhub",
    }]


def get_recommendation_trends(ticker: str) -> list[dict]:
    rows = _get("stock/recommendation", {"symbol": ticker}) or []
    result = []
    for r in rows:
        result.append({
            "period":     r.get("period", ""),
            "strongBuy":  r.get("strongBuy"),
            "buy":        r.get("buy"),
            "hold":       r.get("hold"),
            "sell":       r.get("sell"),
            "strongSell": r.get("strongSell"),
            "_source": "finnhub",
        })
    return result


def get_earnings_calendar(ticker: str) -> list[dict]:
    data = _get("stock/earnings", {"symbol": ticker})
    rows = data if isinstance(data, list) else (data.get("earningsCalendar") or [])
    result = []
    for r in rows:
        result.append({
            "date":          r.get("date", ""),
            "epsActual":     r.get("actual"),
            "epsEstimate":   r.get("estimate"),
            "revenuActual":  r.get("revenueActual"),
            "revenueEst":    r.get("revenueEstimate"),
            "_source": "finnhub",
        })
    return result


# ---------------------------------------------------------------------------
# SEC filings
# ---------------------------------------------------------------------------

def get_sec_filings(ticker: str, filing_type: str = "10-K", limit: int = 5) -> list[dict]:
    data = _get("stock/filings", {"symbol": ticker, "type": filing_type, "from": "2018-01-01"})
    rows = (data if isinstance(data, list) else [])[:limit]
    result = []
    for r in rows:
        result.append({
            "filingType":  r.get("type", ""),
            "filingDate":  r.get("filedDate", ""),
            "acceptedDate": r.get("acceptedDate", ""),
            "reportUrl":   r.get("reportUrl", ""),
            "formType":    r.get("form", ""),
            "_source": "finnhub",
        })
    return result


# ---------------------------------------------------------------------------
# Insider trading
# ---------------------------------------------------------------------------

def get_insider_trades(ticker: str, limit: int = 20) -> list[dict]:
    data = _get("stock/insider-transactions", {"symbol": ticker})
    rows = (data.get("data") or [])[:limit]
    result = []
    for r in rows:
        result.append({
            "name":            r.get("name", ""),
            "transactionDate": r.get("transactionDate", ""),
            "transactionType": r.get("transactionType", ""),
            "share":           r.get("share"),
            "change":          r.get("change"),
            "filingDate":      r.get("filingDate", ""),
            "_source": "finnhub",
        })
    return result


# ---------------------------------------------------------------------------
# Universe / screening helper
# ---------------------------------------------------------------------------

def get_stock_screener(
    market_cap_more_than: int = 300_000_000,
    exchange: str = "US",
    limit: int = 500,
) -> list[dict]:
    """
    Finnhub doesn't have a direct screener endpoint on the free tier.
    Returns S&P 500 constituents as a compliant universe starting point.
    """
    data = _get("index/constituents", {"symbol": "^GSPC"})
    constituents = data.get("constituents", [])
    return [{"symbol": s, "_source": "finnhub"} for s in constituents[:limit]]
