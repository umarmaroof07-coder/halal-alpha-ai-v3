"""
Live data provider — orchestrates Finnhub (primary) with yfinance (fallback).

Adding a new provider in the future:
  1. Create data_layer/<provider>_provider.py with the same public API.
  2. Register it in _PROVIDERS below.
  3. No other file needs to change.

API keys are never logged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from data_layer import fmp_provider, finnhub_provider, yfinance_provider
from data_layer.fmp_provider import ProviderError
from data_layer.data_freshness import get_cached, set_cached

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — ordered by preference
# ---------------------------------------------------------------------------

# FMP → Finnhub → yfinance, in order of preference
_PROVIDERS = [fmp_provider, finnhub_provider, yfinance_provider]

# Cache TTLs in hours
_TTL: dict[str, int] = {
    "get_profile":              24 * 7,
    "get_quote":                1,
    "get_historical_prices":    24,
    "get_income_statement":     24 * 7,
    "get_balance_sheet":        24 * 7,
    "get_cash_flow":            24 * 7,
    "get_key_metrics":          24 * 7,
    "get_financial_ratios":     24 * 7,
    "get_analyst_estimates":    24 * 3,
    "get_price_targets":        24 * 3,
    "get_recommendation_trends": 24 * 3,
    "get_earnings_calendar":    24,
    "get_sec_filings":          24 * 7,
    "get_insider_trades":       24 * 3,
}


def _call_with_fallback(method_name: str, ticker: str, **kwargs) -> Any:
    """
    Try each provider in order.  Cache the first successful result.
    Never surfaces API keys in logs or exceptions.
    """
    cache_params = {"ticker": ticker, **kwargs}
    ttl = _TTL.get(method_name, 24)

    cached = get_cached("live", method_name, cache_params, ttl_hours=ttl)
    if cached is not None:
        log.debug("Cache hit: %s(%s)", method_name, ticker)
        return cached

    last_exc: Exception | None = None
    for provider in _PROVIDERS:
        fn: Callable | None = getattr(provider, method_name, None)
        if fn is None:
            continue
        provider_name = getattr(provider, "__name__", str(provider)).split(".")[-1]
        try:
            result = fn(ticker, **kwargs)
            set_cached("live", method_name, cache_params, result, ttl_hours=ttl)
            if provider_name != "fmp_provider":
                log.warning("Used fallback provider (%s) for %s(%s)", provider_name, method_name, ticker)
            return result
        except Exception as exc:
            log.warning("Provider %s failed for %s(%s): %s", provider_name, method_name, ticker, exc)
            last_exc = exc

    raise ProviderError(
        f"All providers failed for {method_name}({ticker})"
    ) from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile(ticker: str) -> dict:
    return _call_with_fallback("get_profile", ticker)

def get_quote(ticker: str) -> dict:
    return _call_with_fallback("get_quote", ticker)

def get_historical_prices(ticker: str, from_date: str, to_date: str) -> list[dict]:
    return _call_with_fallback("get_historical_prices", ticker, from_date=from_date, to_date=to_date)

def get_income_statement(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_income_statement", ticker, period=period, limit=limit)

def get_balance_sheet(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_balance_sheet", ticker, period=period, limit=limit)

def get_cash_flow(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_cash_flow", ticker, period=period, limit=limit)

def get_key_metrics(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_key_metrics", ticker, period=period, limit=limit)

def get_financial_ratios(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_financial_ratios", ticker, period=period, limit=limit)

def get_analyst_estimates(ticker: str, period: str = "annual", limit: int = 4) -> list[dict]:
    return _call_with_fallback("get_analyst_estimates", ticker, period=period, limit=limit)

def get_price_targets(ticker: str) -> list[dict]:
    return _call_with_fallback("get_price_targets", ticker)

def get_recommendation_trends(ticker: str) -> list[dict]:
    return _call_with_fallback("get_recommendation_trends", ticker)

def get_earnings_calendar(ticker: str) -> list[dict]:
    return _call_with_fallback("get_earnings_calendar", ticker)

def get_sec_filings(ticker: str, filing_type: str = "10-K", limit: int = 5) -> list[dict]:
    return _call_with_fallback("get_sec_filings", ticker, filing_type=filing_type, limit=limit)

def get_insider_trades(ticker: str, limit: int = 20) -> list[dict]:
    return _call_with_fallback("get_insider_trades", ticker, limit=limit)

def get_eps_trend_and_revisions(ticker: str) -> dict:
    """EPS trend + revision breadth from yfinance (always direct, no FMP fallback)."""
    from data_layer import yfinance_provider as _yfp
    return _yfp.get_eps_trend_and_revisions(ticker)

def get_upgrades_downgrades_90d(ticker: str) -> dict:
    """90-day upgrade/downgrade counts from yfinance."""
    from data_layer import yfinance_provider as _yfp
    return _yfp.get_upgrades_downgrades_90d(ticker)


def get_live_price(ticker: str) -> dict | None:
    """
    Fetch a fresh quote bypassing the standard 1-hour cache.
    Used by the dashboard for semi-live price display (refreshed every 60 s).

    Returns {"price": float, "change_pct": float, "source": str, "timestamp": str}
    or None if all providers fail.
    """
    from datetime import datetime as _dt
    from data_layer.data_freshness import get_cached, set_cached

    _TTL_LIVE = 1 / 60  # 1-minute cache (in hours)
    cache_params = {"ticker": ticker, "_live": True}
    cached = get_cached("live_price", "get_live_price", cache_params, ttl_hours=_TTL_LIVE)
    if cached is not None:
        return cached

    for provider in _PROVIDERS:
        fn = getattr(provider, "get_quote", None)
        if fn is None:
            continue
        provider_name = getattr(provider, "__name__", str(provider)).split(".")[-1]
        try:
            qt = fn(ticker)
            if not qt or not qt.get("price"):
                continue
            result = {
                "price":      float(qt.get("price", 0)),
                "change_pct": float(qt.get("changesPercentage") or qt.get("change_pct") or 0),
                "source":     provider_name.replace("_provider", "").upper(),
                "timestamp":  _dt.now().isoformat(),
            }
            set_cached("live_price", "get_live_price", cache_params, result,
                       ttl_hours=_TTL_LIVE)
            return result
        except Exception:
            continue
    return None


def get_live_prices_batch(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch live prices for multiple tickers. Returns {ticker: price_dict}.
    Each price_dict has keys: price, change_pct, source, timestamp.
    """
    results: dict[str, dict] = {}
    for t in tickers:
        try:
            p = get_live_price(t)
            if p:
                results[t] = p
        except Exception:
            pass
    return results
