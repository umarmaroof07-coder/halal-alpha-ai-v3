"""
yfinance — emergency fallback provider only.

Used when FMP fails.  Returns the same field names as FMP where possible
so callers in live_data_provider.py need no special-casing.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _ticker(symbol: str) -> Any:
    import yfinance as yf  # lazy import — not installed in all envs
    return yf.Ticker(symbol)


def get_profile(ticker: str) -> dict:
    t = _ticker(ticker)
    info = t.info or {}
    return {
        "symbol": ticker,
        "companyName": info.get("longName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "country": info.get("country", ""),
        "exchange": info.get("exchange", ""),
        "mktCap": info.get("marketCap"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "description": info.get("longBusinessSummary", ""),
        "_source": "yfinance",
    }


def get_quote(ticker: str) -> dict:
    t = _ticker(ticker)
    info = t.info or {}
    return {
        "symbol": ticker,
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "volume": info.get("regularMarketVolume"),
        "avgVolume": info.get("averageVolume"),
        "marketCap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "_source": "yfinance",
    }


def get_historical_prices(ticker: str, from_date: str, to_date: str) -> list[dict]:
    import yfinance as yf
    import pandas as pd
    df = yf.download(ticker, start=from_date, end=to_date, progress=False, auto_adjust=True)
    if df.empty:
        return []
    # yfinance ≥1.x returns a MultiIndex when group_by is default.
    # Flatten ('Close', 'AAPL') → 'Close' so column access is uniform.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    records = []
    for _, row in df.iterrows():
        try:
            records.append({
                "date":   str(row["Date"])[:10],
                "open":   float(row["Open"])   if row.get("Open")   is not None else None,
                "high":   float(row["High"])   if row.get("High")   is not None else None,
                "low":    float(row["Low"])    if row.get("Low")    is not None else None,
                "close":  float(row["Close"])  if row.get("Close")  is not None else None,
                "volume": int(row["Volume"])   if row.get("Volume") is not None else None,
                "_source": "yfinance",
            })
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("yfinance row parse error for %s: %s", ticker, exc)
            continue
    return records


def _df_to_records(df) -> list[dict]:
    """Convert a yfinance transposed DataFrame to a list of period dicts with string keys."""
    if df is None or df.empty:
        return []
    # Transpose so each column (period) becomes a row
    df = df.T.copy()
    df.index = [str(i)[:10] for i in df.index]  # Timestamp → "YYYY-MM-DD"
    df.columns = [str(c) for c in df.columns]
    records = []
    for date_str, row in df.iterrows():
        rec = {"date": date_str, "_source": "yfinance"}
        for col, val in row.items():
            try:
                rec[col] = float(val) if val is not None else None
            except (TypeError, ValueError):
                rec[col] = None
        records.append(rec)
    return records


_INCOME_MAP = {
    "Total Revenue": "revenue",
    "Net Income": "netIncome",
    "Gross Profit": "grossProfit",
    "Operating Income": "operatingIncome",
    "EBITDA": "ebitda",
    "Basic EPS": "eps",
    "Interest Expense": "interestIncome",
}

_BALANCE_MAP = {
    "Total Assets": "totalAssets",
    "Total Debt": "totalDebt",
    "Long Term Debt": "totalDebt",
    "Stockholders Equity": "totalEquity",
    "Cash And Cash Equivalents": "cashAndEquivalents",
    "Accounts Receivable": "accountsReceivable",
    "Short Term Investments": "shortTermInvestments",
}

_CASHFLOW_MAP = {
    "Free Cash Flow": "freeCashFlow",
    "Operating Cash Flow": "operatingCashFlow",
    "Capital Expenditure": "capitalExpenditure",
    "Stock Based Compensation": "stockBasedCompensation",
}


def _normalize(records: list[dict], field_map: dict) -> list[dict]:
    """Add normalized field names alongside raw yfinance field names."""
    out = []
    for rec in records:
        normalized = dict(rec)
        for yf_key, std_key in field_map.items():
            if yf_key in rec and rec[yf_key] is not None:
                normalized.setdefault(std_key, rec[yf_key])
        out.append(normalized)
    return out


def get_income_statement(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    t = _ticker(ticker)
    df = t.financials if period == "annual" else t.quarterly_financials
    return _normalize(_df_to_records(df)[:limit], _INCOME_MAP)


def get_balance_sheet(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    t = _ticker(ticker)
    df = t.balance_sheet if period == "annual" else t.quarterly_balance_sheet
    return _normalize(_df_to_records(df)[:limit], _BALANCE_MAP)


def get_cash_flow(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    t = _ticker(ticker)
    df = t.cashflow if period == "annual" else t.quarterly_cashflow
    return _normalize(_df_to_records(df)[:limit], _CASHFLOW_MAP)


def get_key_metrics(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    t = _ticker(ticker)
    info = t.info or {}
    return [{
        "roe":             info.get("returnOnEquity"),
        "debtToEquity":    info.get("debtToEquity"),
        "currentRatio":    info.get("currentRatio"),
        "grossMargin":     info.get("grossMargins"),
        "operatingMargin": info.get("operatingMargins"),
        "netProfitMargin": info.get("profitMargins"),
        "revenueGrowth":   info.get("revenueGrowth"),
        "peRatio":         info.get("trailingPE"),
        "pbRatio":         info.get("priceToBook"),
        "evToEbitda":      info.get("enterpriseToEbitda"),
        "_source": "yfinance",
    }]


def get_financial_ratios(ticker: str, period: str = "annual", limit: int = 5) -> list[dict]:
    return get_key_metrics(ticker, period, limit)


def get_analyst_estimates(ticker: str, **_kwargs) -> list[dict]:
    """Return forward EPS estimates for current year and next year."""
    try:
        t = _ticker(ticker)
        ee = t.earnings_estimate
        if ee is None or ee.empty:
            return []
        results = []
        for period_idx, row in ee.iterrows():
            results.append({
                "period":              str(period_idx),
                "estimatedEpsAvg":     _safe_float(row.get("avg")),
                "estimatedEpsLow":     _safe_float(row.get("low")),
                "estimatedEpsHigh":    _safe_float(row.get("high")),
                "yearAgoEps":          _safe_float(row.get("yearAgoEps")),
                "numberOfAnalysts":    _safe_float(row.get("numberOfAnalysts")),
                "growth":              _safe_float(row.get("growth")),
                "_source": "yfinance",
            })
        return results
    except Exception as exc:
        log.debug("yfinance analyst_estimates failed for %s: %s", ticker, exc)
        return []


def get_eps_trend_and_revisions(ticker: str) -> dict:
    """
    Return EPS trend (estimate changes over 7d/30d/90d) and
    revision breadth (up vs down counts).
    Used by the institutional Revisions Engine v2.
    """
    try:
        t = _ticker(ticker)
        trend = t.eps_trend
        revs  = t.eps_revisions
        rev_est = t.revenue_estimate
        result: dict = {"_source": "yfinance"}

        if trend is not None and not trend.empty:
            # Prefer current-year estimate row (index "0y")
            for period in ("0y", "+1y", "0q", "+1q"):
                if period in trend.index:
                    row = trend.loc[period]
                    result["eps_current"]    = _safe_float(row.get("current"))
                    result["eps_7d_ago"]     = _safe_float(row.get("7daysAgo"))
                    result["eps_30d_ago"]    = _safe_float(row.get("30daysAgo"))
                    result["eps_90d_ago"]    = _safe_float(row.get("90daysAgo"))
                    result["eps_period"]     = period
                    break

            # Next-year row for forward revision momentum
            if "+1y" in trend.index:
                row1 = trend.loc["+1y"]
                result["eps_ny_current"] = _safe_float(row1.get("current"))
                result["eps_ny_30d_ago"] = _safe_float(row1.get("30daysAgo"))

        if revs is not None and not revs.empty:
            # Sum across periods for breadth signal
            up7   = up30  = dn30 = dn7 = 0
            for period in ("0y", "+1y", "0q", "+1q"):
                if period in revs.index:
                    row = revs.loc[period]
                    up7  += int(row.get("upLast7days", 0)   or 0)
                    up30 += int(row.get("upLast30days", 0)  or 0)
                    dn30 += int(row.get("downLast30days", 0) or 0)
                    dn7  += int(row.get("downLast7Days", 0) or 0)
            result["rev_up_7d"]   = up7
            result["rev_up_30d"]  = up30
            result["rev_dn_30d"]  = dn30
            result["rev_dn_7d"]   = dn7

        if rev_est is not None and not rev_est.empty:
            for period in ("0y", "+1y"):
                if period in rev_est.index:
                    row = rev_est.loc[period]
                    result["rev_est_avg"]    = _safe_float(row.get("avg"))
                    result["rev_est_growth"] = _safe_float(row.get("growth"))
                    break

        return result
    except Exception as exc:
        log.debug("yfinance eps_trend failed for %s: %s", ticker, exc)
        return {}


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return None if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return None


def get_price_targets(ticker: str) -> list[dict]:
    try:
        t = _ticker(ticker)
        pt = t.analyst_price_targets
        if pt is None or not isinstance(pt, dict):
            return []
        return [{
            "priceTarget":   pt.get("mean") or pt.get("median"),
            "targetMean":    pt.get("mean"),
            "targetMedian":  pt.get("median"),
            "targetHigh":    pt.get("high"),
            "targetLow":     pt.get("low"),
            "currentPrice":  pt.get("current"),
            "_source": "yfinance",
        }]
    except Exception as exc:
        log.debug("yfinance price_targets failed for %s: %s", ticker, exc)
        return []


def get_recommendation_trends(ticker: str) -> list[dict]:
    """Return most-recent analyst recommendation count summary."""
    try:
        t = _ticker(ticker)
        # recommendations_summary gives aggregated period buckets
        rs = t.recommendations_summary
        if rs is not None and not rs.empty:
            latest = rs.iloc[0] if len(rs) > 0 else None
            if latest is not None:
                return [{
                    "strongBuy":  int(latest.get("strongBuy", 0) or 0),
                    "buy":        int(latest.get("buy", 0) or 0),
                    "hold":       int(latest.get("hold", 0) or 0),
                    "sell":       int(latest.get("sell", 0) or 0),
                    "strongSell": int(latest.get("strongSell", 0) or 0),
                    "_source": "yfinance",
                }]
        # Fallback: parse raw upgrades_downgrades
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return []
        return ud.reset_index().to_dict(orient="records")
    except Exception as exc:
        log.debug("yfinance recommendation_trends failed for %s: %s", ticker, exc)
        return []


def get_upgrades_downgrades_90d(ticker: str) -> dict:
    """Return upgrade/downgrade counts over the past 90 days."""
    try:
        import pandas as pd
        t = _ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return {}
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
        ud.index = pd.to_datetime(ud.index, utc=True)
        recent = ud[ud.index >= cutoff]
        if recent.empty:
            return {}
        actions = recent.get("Action", recent.get("action", None))
        if actions is None:
            return {}
        upgrades   = int((actions.str.lower() == "up").sum())
        downgrades = int((actions.str.lower() == "down").sum())
        return {"upgrades_90d": upgrades, "downgrades_90d": downgrades,
                "net_90d": upgrades - downgrades, "_source": "yfinance"}
    except Exception as exc:
        log.debug("yfinance upgrades_downgrades_90d failed for %s: %s", ticker, exc)
        return {}


def get_earnings_calendar(ticker: str) -> list[dict]:
    t = _ticker(ticker)
    cal = t.calendar
    if cal is None:
        return []
    return [{"_source": "yfinance", "_raw": cal}]


def get_sec_filings(ticker: str, **_kwargs) -> list[dict]:
    log.warning("yfinance fallback: sec_filings not available for %s", ticker)
    return []


def get_insider_trades(ticker: str, **_kwargs) -> list[dict]:
    log.warning("yfinance fallback: insider_trades not available for %s", ticker)
    return []
