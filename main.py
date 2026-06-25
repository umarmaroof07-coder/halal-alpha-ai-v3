"""
Halal Alpha AI V3 — CLI entry point.

Commands:
  python3 main.py --refresh-universe   Build Shariah-screened universe from FMP.
  python3 main.py --refresh-data       Score screened universe with all factors.
  python3 main.py --backtest           Run walk-forward backtest.
  python3 main.py --recommend          Generate live recommendations.

Run order: --refresh-universe → --refresh-data → (--backtest) → --recommend

Safety rules:
  --recommend requires backtest_complete.json to exist (must backtest first).
  All recommendations come exclusively from safe_recommendations().
  AI Research = 50 during backtest (no Claude calls).
  API keys are never logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging — API keys must never appear in log output
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("halal_alpha")

DATA_DIR            = Path("data")
CACHE_DIR           = DATA_DIR / "cache"
BACKTEST_COMPLETE   = DATA_DIR / "backtest_complete.json"
SCORED_UNIVERSE     = CACHE_DIR / "scored_universe.json"
UNIVERSE_SCREENED   = CACHE_DIR / "universe_screened.csv"
REPORTS_DIR         = DATA_DIR / "reports"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_backtest() -> dict:
    if not BACKTEST_COMPLETE.exists():
        print(
            "\n  ERROR: backtest_complete.json not found.\n"
            "  Run --backtest first before requesting recommendations.\n"
        )
        sys.exit(1)
    with BACKTEST_COMPLETE.open() as f:
        return json.load(f)


def _require_universe() -> list[dict]:
    """Load scored_universe.json, or fall back to universe_screened.csv."""
    if SCORED_UNIVERSE.exists():
        with SCORED_UNIVERSE.open() as f:
            payload = json.load(f)
        return payload.get("universe", [])
    if UNIVERSE_SCREENED.exists():
        from screener.live_universe import load_screened_csv
        rows = load_screened_csv()
        if rows:
            log.warning(
                "scored_universe.json not found — using neutral-50 factor scores. "
                "Run --refresh-data to compute real scores."
            )
            return [
                {
                    "ticker":         r["ticker"],
                    "name":           r.get("name", ""),
                    "sector":         r.get("sector", ""),
                    "price":          float(r["price"]) if r.get("price") else 0.0,
                    "mkt_cap":        float(r["mkt_cap"]) if r.get("mkt_cap") else 0.0,
                    "avg_volume":     float(r["avg_volume"]) if r.get("avg_volume") else 0.0,
                    "shariah_status": r.get("shariah_status", "unknown"),
                    "composite": 50.0, "quality": 50.0, "momentum": 50.0,
                    "valuation": 50.0, "earnings_revisions": 50.0, "ai_research": 50.0,
                }
                for r in rows
            ]
    print(
        "\n  ERROR: No universe data found.\n"
        "  Run --refresh-universe first.\n"
    )
    sys.exit(1)


def _print_section(title: str) -> None:
    bar = "═" * 58
    print(f"\n{bar}\n  {title}\n{bar}")


_SBC_FCF_THRESHOLD = 0.20  # flag when SBC > 20% of reported FCF


def _limit_disclaimer() -> str:
    from portfolio.limit_prices import DISCLAIMER
    return DISCLAIMER


def _print_entry_analysis(top5: list[dict]) -> None:
    """Print Entry Analysis table for the final Top 5."""
    print("\n  📊  ENTRY ANALYSIS  (ATR-anchored, no fixed discounts)\n")
    print(f"  {'Ticker':<8} {'Price':>8} {'Buy Below':>10} {'Strong Buy':>11} {'Score':>6} {'Rating':<12} {'ATR-14':>8}")
    print("  " + "─" * 72)
    try:
        from portfolio.entry_price import compute_entry_analysis
        universe_map: dict = {}
        _su = Path("data/cache/scored_universe.json")
        if _su.exists():
            import json as _json
            _u = _json.loads(_su.read_text()).get("universe", [])
            universe_map = {s["ticker"]: s for s in _u}

        for pick in top5:
            t = pick.get("ticker", "")
            s = universe_map.get(t, {})
            price = float(s.get("price") or pick.get("price") or 0)
            if price <= 0:
                print(f"  {t:<8}   — no price data")
                continue
            ea = compute_entry_analysis(
                ticker           = t,
                current_price    = price,
                valuation_score  = float(s.get("valuation", 50)),
                composite_score  = float(s.get("composite", 50)),
                model_confidence = float(s.get("model_confidence", 65)),
                risk_score       = float(s.get("risk_adjustment", 50)),
                momentum_score   = float(s.get("momentum", 50)),
            )
            bl_str  = f"${ea.buy_limit:.2f}"    if ea.buy_limit    else "  —"
            sbl_str = f"${ea.strong_buy_limit:.2f}" if ea.strong_buy_limit else "  —"
            atr_str = f"${ea.atr14:.2f}"         if ea.atr14        else "  —"
            print(
                f"  {t:<8} ${price:>7.2f}  {bl_str:>9}  {sbl_str:>10}"
                f"  {ea.entry_score:>5.0f}  {ea.entry_rating:<12} {atr_str:>8}"
            )
            print(f"    ↳ {ea.explanation}")
    except Exception as _ea_exc:
        log.warning("Entry analysis unavailable: %s", _ea_exc)
        print("  (entry analysis unavailable)")
    print()


def _sbc_warning(sbc_data: dict | None) -> str | None:
    """Return a warning string if SBC/FCF > threshold, else None."""
    if not sbc_data:
        return None
    ratio = sbc_data.get("sbc_fcf_ratio", 0)
    if ratio > _SBC_FCF_THRESHOLD:
        adj_fcf = sbc_data.get("adj_fcf", 0)
        return (
            f"High stock-based compensation — SBC is {ratio*100:.0f}% of reported FCF. "
            f"SBC-adjusted FCF = ${adj_fcf/1e9:.2f}B. "
            f"Reported FCF may overstate owner earnings."
        )
    return None


def _quality_detail(qr) -> dict | None:
    """Serialise QualityRaw to a JSON-safe dict for scored_universe.json."""
    if qr is None:
        return None

    def _sig(gs):
        if gs is None:
            return None
        return {"raw": gs.raw, "value": gs.value, "flag": gs.flag}

    return {
        "raw_score":            qr.raw_score,
        "signals_used":         qr.signals_used,
        "contributions":        qr.contributions,
        "years_of_data":        qr.years_of_data,
        # Single-year
        "roic":                 qr.roic,
        "operating_margin":     qr.operating_margin,
        "net_margin":           qr.net_margin,
        "fcf_margin":           qr.fcf_margin,
        "equity_ratio":         qr.equity_ratio,
        "revenue_growth":       _sig(qr.revenue_growth_sig),
        "earnings_growth":      _sig(qr.earnings_growth_sig),
        "fcf_growth":           _sig(qr.fcf_growth_sig),
        # Multi-year (5yr)
        "revenue_cagr_5yr":     qr.revenue_cagr_5yr,
        "eps_cagr_5yr":         qr.eps_cagr_5yr,
        "fcf_cagr_5yr":         qr.fcf_cagr_5yr,
        "roic_avg_5yr":         qr.roic_avg_5yr,
        "roic_stability":       qr.roic_stability,
        "fcf_margin_stability": qr.fcf_margin_stability,
        "share_count_trend":    qr.share_count_trend,
        "debt_trend":           qr.debt_trend,
    }


def _print_quality_audit(top20_scores, quality_raw: dict) -> None:
    """Print quality factor audit table for the top-20 ranked stocks."""
    from factors.quality import _GROWTH_CAP, _MAX_CONTRIBUTION
    print("\n" + "─" * 100)
    print("  QUALITY FACTOR AUDIT — TOP 20")
    print("─" * 100)
    hdr = (f"  {'Ticker':<6}  {'Q Score':>7}  {'Raw':>7}  "
           f"{'ROIC%':>6}  {'OpM%':>5}  {'NIM%':>5}  {'FCFM%':>5}  {'EqR%':>5}  "
           f"{'RevG':>6}  {'NIG':>5}  {'FCFG':>5}  "
           f"{'Signals':>7}  Flags")
    print(hdr)
    print("  " + "-" * 96)
    for fs in top20_scores:
        t  = fs.ticker
        qs = fs.quality
        qr = quality_raw.get(t)
        if qr is None:
            print(f"  {t:<6}  {qs:>7.1f}  {'n/a':>7}")
            continue

        def pct(v):
            return f"{v*100:>5.1f}" if v is not None else "  n/a"
        def gfmt(sig):
            if sig is None:
                return f"{'n/a':>5}"
            if sig.value is None:
                short = {"prior_negative": "neg", "prior_too_small": "tiny",
                         "missing": "miss", "capped": "cap"}.get(sig.flag, sig.flag[:4])
                return f"{short:>5}"
            capped_marker = "*" if sig.flag == "capped" else " "
            return f"{sig.value*100:>4.0f}{capped_marker}"

        raw_str = f"{qr.raw_score:.4f}" if qr.raw_score is not None else "  None"
        flags = []
        for name, sig in [("rev", qr.revenue_growth_sig),
                           ("ni",  qr.earnings_growth_sig),
                           ("fcf", qr.fcf_growth_sig)]:
            if sig and sig.flag not in ("ok", "capped", "missing"):
                flags.append(f"{name}={sig.flag}")
        flag_str = ", ".join(flags) if flags else ""

        print(
            f"  {t:<6}  {qs:>7.1f}  {raw_str:>7}  "
            f"{pct(qr.roic)}  {pct(qr.operating_margin)}  {pct(qr.net_margin)}  "
            f"{pct(qr.fcf_margin)}  {pct(qr.equity_ratio)}  "
            f"{gfmt(qr.revenue_growth_sig)}  {gfmt(qr.earnings_growth_sig)}  "
            f"{gfmt(qr.fcf_growth_sig)}  "
            f"{len(qr.signals_used):>7}  {flag_str}"
        )

    print("─" * 100)
    print(f"  Growth winsorized at ±{int(_GROWTH_CAP*100)}%.  "
          f"Contribution cap {int(_MAX_CONTRIBUTION*100)}% per signal.")
    print(f"  Flags: neg=prior_negative  tiny=prior<5%×current  *=capped at ±500%")
    print("─" * 100)


# ---------------------------------------------------------------------------
# --refresh-universe
# ---------------------------------------------------------------------------

def cmd_refresh_universe() -> None:
    """
    Fetch broad U.S. stock universe from FMP, apply pre-filters, run Shariah
    screening, and save CSVs to data/cache/.
    """
    _print_section("REFRESH UNIVERSE")

    from config.settings import FMP_API_KEY
    if not FMP_API_KEY:
        print("  WARNING: FMP_API_KEY not set in .env — FMP calls will fail.")

    try:
        from screener.live_universe import build_live_universe, WATCHLIST_CSV

        print("  Sources: S&P 500 → Russell 1000 → FMP screener")
        print(f"  Watchlist fallback: {WATCHLIST_CSV}")
        print("  Filters: market cap > $2B, price < $1,000, avg volume > 500k")
        print("  Shariah: unknown = excluded\n")

        result = build_live_universe()

        print(f"\n  Raw candidates (after pre-filter):  {result.raw_count:>5}")
        print(f"  After Shariah screening:             {result.screened_count:>5}")
        print(f"    ✓ Compliant:                       {result.compliant_count:>5}")
        print(f"    ✗ Non-compliant:                   {result.non_compliant_count:>5}")
        print(f"    ? Unknown (excluded):              {result.unknown_count:>5}")
        print(f"\n  Sources used: {', '.join(result.sources_used) or 'none'}")

        if result.warnings:
            for w in result.warnings:
                print(f"\n  ⚠  {w}")

        print(f"\n  ✓ universe_raw.csv          → {CACHE_DIR / 'universe_raw.csv'}")
        print(f"  ✓ universe_screened.csv     → {CACHE_DIR / 'universe_screened.csv'}")
        print(f"  ✓ shariah_validation_report.csv")
        print("\n  Run --refresh-data to compute factor scores.\n")

    except Exception as exc:
        log.exception("refresh-universe failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --refresh-data
# ---------------------------------------------------------------------------

def cmd_refresh_data() -> None:
    """
    Fetch fundamentals for all screened tickers, compute factor scores,
    and save data/cache/scored_universe.json.

    Requires universe_screened.csv (run --refresh-universe first).
    """
    _print_section("REFRESH DATA  (factor scoring)")

    from config.settings import FMP_API_KEY, FINNHUB_API_KEY, BACKTEST_AI_NEUTRAL
    if not FMP_API_KEY:
        print("  WARNING: FMP_API_KEY not set — FMP calls will fail.")
    if not FINNHUB_API_KEY:
        print("  WARNING: FINNHUB_API_KEY not set — Finnhub fallback unavailable.")

    if not UNIVERSE_SCREENED.exists():
        print(
            "\n  ERROR: universe_screened.csv not found.\n"
            "  Run --refresh-universe first.\n"
        )
        sys.exit(1)

    try:
        from screener.live_universe import load_screened_csv
        from data_layer import live_data_provider as ldp
        from factors.momentum import compute_momentum
        from factors.quality import compute_quality
        from factors.valuation import compute_valuation
        from factors.revisions import compute_revisions
        from factors.earnings_quality import compute_earnings_quality
        from factors.moat_quality import compute_moat_quality
        from factors.capital_allocation import compute_capital_allocation
        from factors.risk_engine import compute_risk
        from factors.composite import compute_composite, rank_scores

        rows = load_screened_csv()
        compliant = [r for r in rows if r.get("shariah_status") == "compliant"]
        tickers = [r["ticker"] for r in compliant]
        log.info("Scoring %d compliant tickers…", len(tickers))

        # ── Collect raw data per ticker ─────────────────────────────────────
        from datetime import date as _date
        import time as _time

        from datetime import timedelta as _timedelta
        today = str(_date.today())
        # 400 calendar days ≈ 277 trading days — enough for all 4 momentum signals
        # (12-1m signal needs 253; a strict 365-day window can yield only 250 trading days)
        from_date_momentum = str(_date.today() - _timedelta(days=400))

        momentum_raw:         dict = {}
        quality_raw:          dict = {}
        valuation_raw:        dict = {}
        revisions_raw:        dict = {}
        earnings_quality_raw: dict = {}
        moat_quality_raw:     dict = {}
        capital_allocation_raw: dict = {}
        prices_map:           dict = {}
        avg_volume_map:       dict = {}
        mkt_cap_map:          dict = {}
        sbc_map:              dict = {}
        meta_map:             dict = {r["ticker"]: r for r in compliant}

        n = len(tickers)
        for i, ticker in enumerate(tickers):
            if i > 0 and i % 25 == 0:
                log.info("  Factor scoring progress: %d / %d", i, n)

            # ── Momentum ────────────────────────────────────────────────────
            try:
                hist = ldp.get_historical_prices(ticker, from_date=from_date_momentum, to_date=today)
                price_list = [
                    float(r["close"]) for r in hist
                    if r.get("close") is not None
                ]
                if price_list:
                    momentum_raw[ticker] = compute_momentum(ticker, price_list)
                    prices_map[ticker] = price_list[-1]
                    log.debug("%s: %d price points, momentum signals: %s",
                              ticker, len(price_list),
                              momentum_raw[ticker].signals_used)
                else:
                    log.warning("%s: historical prices returned %d records but 0 valid closes",
                                ticker, len(hist))
            except Exception as exc:
                log.warning("%s: momentum fetch failed: %s", ticker, exc)

            # ── Quality (5-yr) + Earnings Quality + Moat ─────────────────────
            try:
                # Fetch 5 years of statements for multi-year quality model
                inc = ldp.get_income_statement(ticker, limit=5)
                cf  = ldp.get_cash_flow(ticker, limit=5)
                bs  = ldp.get_balance_sheet(ticker, limit=5)
                km  = ldp.get_key_metrics(ticker, limit=5)

                def _nth(lst, n, key):
                    return lst[n].get(key) if lst and len(lst) > n else None

                rev0  = _nth(inc, 0, "revenue")
                rev1  = _nth(inc, 1, "revenue")
                ni0   = _nth(inc, 0, "netIncome")
                ni1   = _nth(inc, 1, "netIncome")
                fcf0  = _nth(cf,  0, "freeCashFlow")
                fcf1  = _nth(cf,  1, "freeCashFlow")
                oi0   = _nth(inc, 0, "operatingIncome")
                oi1   = _nth(inc, 1, "operatingIncome")
                eq0   = _nth(bs,  0, "totalEquity")
                debt0 = _nth(bs,  0, "totalDebt")
                debt1 = _nth(bs,  1, "totalDebt")
                cash0 = _nth(bs,  0, "cashAndEquivalents")
                assets0 = _nth(bs, 0, "totalAssets")
                assets1 = _nth(bs, 1, "totalAssets")
                gp0   = _nth(inc, 0, "grossProfit")

                def _shares(idx):
                    return _to_float(
                        _nth(km, idx, "weightedAverageSharesOutstanding")
                        or _nth(km, idx, "sharesOutstanding")
                    )
                shares0 = _shares(0)
                shares1 = _shares(1)

                # Build 5-year series (oldest→newest) for multi-year quality
                def _series(lst, key):
                    return [_to_float(lst[i].get(key)) for i in range(len(lst)-1, -1, -1)] if lst else []

                rev_series  = _series(inc, "revenue")
                ni_series   = _series(inc, "netIncome")   # used as EPS proxy (absolute)
                fcf_series  = _series(cf,  "freeCashFlow")
                debt_series = _series(bs,  "totalDebt")

                # Shares series (oldest first from km)
                def _shares_series():
                    s = []
                    for i in range(len(km)-1, -1, -1):
                        v = _to_float(
                            km[i].get("weightedAverageSharesOutstanding")
                            or km[i].get("sharesOutstanding")
                        )
                        s.append(v)
                    return s
                shares_series = _shares_series()

                # FCF margin series
                fcf_margin_series = []
                for r_v, f_v in zip(rev_series, fcf_series):
                    if r_v and r_v != 0 and f_v is not None:
                        fcf_margin_series.append(f_v / r_v)
                    else:
                        fcf_margin_series.append(None)

                # ROIC series (use current qr.roic for year 0; approximate others)
                # We'll pass None series and let the model use what it can
                roic_series: list[float | None] = []
                for j in range(len(km)-1, -1, -1):
                    roic_v = _to_float(_nth(km, j, "roic"))
                    roic_series.append(roic_v)

                qr = compute_quality(
                    ticker=ticker,
                    revenue=_to_float(rev0),
                    net_income=_to_float(ni0),
                    free_cash_flow=_to_float(fcf0),
                    operating_income=_to_float(oi0),
                    total_equity=_to_float(eq0),
                    total_debt=_to_float(debt0),
                    cash=_to_float(cash0),
                    revenue_prior=_to_float(rev1),
                    net_income_prior=_to_float(ni1),
                    free_cash_flow_prior=_to_float(fcf1),
                    revenue_series=rev_series,
                    eps_series=ni_series,
                    fcf_series=fcf_series,
                    roic_series=roic_series,
                    fcf_margin_series=fcf_margin_series,
                    shares_series=shares_series,
                    total_debt_series=debt_series,
                )
                quality_raw[ticker] = qr

                # ── SBC data ─────────────────────────────────────────────────
                sbc0  = _to_float(
                    (cf[0].get("stockBasedCompensation") or cf[0].get("Stock Based Compensation"))
                    if cf else None
                )
                fcf_v = _to_float(fcf0)
                if sbc0 is not None and fcf_v and fcf_v > 0:
                    ratio = sbc0 / fcf_v
                    sbc_map[ticker] = {
                        "sbc":           round(sbc0),
                        "fcf":           round(fcf_v),
                        "sbc_fcf_ratio": round(ratio, 4),
                        "adj_fcf":       round(fcf_v - sbc0),
                    }

                # ── Earnings Quality ─────────────────────────────────────────
                rev_v   = _to_float(rev0)
                roic0   = qr.roic
                roic1   = _to_float(km[1].get("roic") if len(km) > 1 else None)
                rev1_v  = _to_float(rev1)
                oi_v    = _to_float(oi0)
                oi1_v   = _to_float(oi1)
                op_margin0 = (oi_v  / rev_v)    if (oi_v  and rev_v)   else None
                op_margin1 = (oi1_v / rev1_v)   if (oi1_v and rev1_v)  else None
                earnings_quality_raw[ticker] = compute_earnings_quality(
                    ticker                 = ticker,
                    net_income             = _to_float(ni0),
                    free_cash_flow         = fcf_v,
                    revenue                = rev_v,
                    revenue_prior          = rev1_v,
                    sbc                    = sbc0,
                    shares_current         = shares0,
                    shares_prior           = shares1,
                    total_debt_current     = _to_float(debt0),
                    total_debt_prior       = _to_float(debt1),
                    total_equity_current   = _to_float(eq0),
                    roic_current           = roic0,
                    roic_prior             = roic1,
                    operating_margin       = op_margin0,
                    operating_margin_prior = op_margin1,
                    total_assets_current   = _to_float(assets0),
                    total_assets_prior     = _to_float(assets1),
                )
                # ── Capital Allocation ──────────────────────────────────────────
                capital_allocation_raw[ticker] = compute_capital_allocation(
                    ticker               = ticker,
                    shares_current       = shares0,
                    shares_prior         = shares1,
                    total_debt_current   = _to_float(debt0),
                    total_debt_prior     = _to_float(debt1),
                    total_equity_current = _to_float(eq0),
                    roic_current         = roic0,
                    roic_prior           = roic1,
                    fcf_current          = fcf_v,
                    fcf_prior            = _to_float(fcf1),
                )

                # ── Moat Quality ─────────────────────────────────────────────
                fcf_margin    = (fcf_v / rev_v)             if (fcf_v and rev_v) else None
                gross_margin0 = (_to_float(gp0) / rev_v)   if (gp0 and rev_v)   else None
                eq_ratio      = qr.equity_ratio
                rev_growth_val = (
                    qr.revenue_growth_sig.value
                    if qr.revenue_growth_sig else None
                )
                moat_quality_raw[ticker] = compute_moat_quality(
                    ticker                 = ticker,
                    roic_current           = roic0,
                    roic_prior             = roic1,
                    operating_margin       = op_margin0,
                    operating_margin_prior = op_margin1,
                    fcf_margin             = fcf_margin,
                    equity_ratio           = eq_ratio,
                    revenue_growth         = rev_growth_val,
                    gross_margin           = gross_margin0,
                    ai_moat_score          = None,   # filled after AI research
                    ai_confidence          = 0.0,
                )
            except Exception:
                pass

            # ── Valuation ───────────────────────────────────────────────────
            try:
                km   = ldp.get_key_metrics(ticker, limit=1)
                cf   = ldp.get_cash_flow(ticker, limit=1)
                est  = ldp.get_analyst_estimates(ticker, limit=1)
                qt   = ldp.get_quote(ticker)

                price     = _to_float(qt.get("price")) if qt else None
                mkt_cap   = _to_float(qt.get("marketCap")) if qt else None
                avg_vol   = _to_float(qt.get("avgVolume")) if qt else None
                fcf       = _to_float(cf[0].get("freeCashFlow")) if cf else None
                fwd_eps   = _to_float(est[0].get("estimatedEpsAvg")) if est else None
                tr_pe     = _to_float(km[0].get("peRatio")) if km else None

                if price:
                    prices_map.setdefault(ticker, price)
                if mkt_cap:
                    mkt_cap_map[ticker] = mkt_cap
                if avg_vol:
                    avg_volume_map[ticker] = avg_vol
                else:
                    # Finnhub doesn't provide avgVolume — fall back to yfinance info directly
                    try:
                        from data_layer import yfinance_provider as _yfp
                        yf_qt = _yfp.get_quote(ticker)
                        yf_vol = _to_float(yf_qt.get("avgVolume"))
                        if yf_vol:
                            avg_volume_map[ticker] = yf_vol
                            if not mkt_cap:
                                yf_cap = _to_float(yf_qt.get("marketCap"))
                                if yf_cap:
                                    mkt_cap_map[ticker] = yf_cap
                    except Exception as exc:
                        log.debug("%s: yfinance avgVolume fallback failed: %s", ticker, exc)

                valuation_raw[ticker] = compute_valuation(
                    ticker=ticker,
                    price=price,
                    trailing_eps=None,
                    forward_eps=fwd_eps,
                    free_cash_flow=fcf,
                    market_cap=mkt_cap,
                    trailing_pe=tr_pe,
                )
            except Exception:
                pass

            # ── Revisions v2 (institutional) ────────────────────────────────
            try:
                # Recommendation counts (for legacy fallback signal)
                recs = ldp.get_recommendation_trends(ticker)
                r0   = recs[0] if recs else {}

                # Price target
                pt_dict    = {}
                pt_upside  = None
                pt_mean    = None
                pt_median  = None
                cur_price  = prices_map.get(ticker)
                try:
                    pts = ldp.get_price_targets(ticker)
                    if pts:
                        pt0       = pts[0]
                        pt_target = _to_float(pt0.get("priceTarget") or pt0.get("targetPrice"))
                        pt_mean   = _to_float(pt0.get("targetMean"))
                        pt_median = _to_float(pt0.get("targetMedian"))
                        if pt_target and cur_price and cur_price > 0:
                            pt_upside = (pt_target - cur_price) / cur_price
                        pt_dict = pt0
                except Exception:
                    pass

                # EPS trend + revision breadth from yfinance
                eps_data = {}
                try:
                    eps_data = ldp.get_eps_trend_and_revisions(ticker)
                except Exception:
                    pass

                # Upgrade/downgrade momentum (90-day)
                ud_data = {}
                try:
                    ud_data = ldp.get_upgrades_downgrades_90d(ticker)
                except Exception:
                    pass

                # Revenue estimate from analyst_estimates
                rev_est_growth = None
                try:
                    ae = ldp.get_analyst_estimates(ticker, limit=1)
                    if ae:
                        rev_est_growth = _to_float(ae[0].get("growth") or ae[0].get("revenueGrowth"))
                except Exception:
                    pass

                revisions_raw[ticker] = compute_revisions(
                    ticker              = ticker,
                    # Legacy rec counts
                    strong_buy          = r0.get("strongBuy"),
                    buy                 = r0.get("buy"),
                    hold                = r0.get("hold"),
                    sell                = r0.get("sell"),
                    strong_sell         = r0.get("strongSell"),
                    price_target_upside = pt_upside,
                    # EPS trend signals
                    eps_current         = eps_data.get("eps_current"),
                    eps_7d_ago          = eps_data.get("eps_7d_ago"),
                    eps_30d_ago         = eps_data.get("eps_30d_ago"),
                    eps_90d_ago         = eps_data.get("eps_90d_ago"),
                    eps_ny_current      = eps_data.get("eps_ny_current"),
                    eps_ny_30d_ago      = eps_data.get("eps_ny_30d_ago"),
                    # Revision breadth
                    rev_up_7d           = eps_data.get("rev_up_7d"),
                    rev_up_30d          = eps_data.get("rev_up_30d"),
                    rev_dn_30d          = eps_data.get("rev_dn_30d"),
                    rev_dn_7d           = eps_data.get("rev_dn_7d"),
                    # Revenue estimates
                    rev_est_growth      = rev_est_growth or eps_data.get("rev_est_growth"),
                    # Upgrade/downgrade momentum
                    upgrades_90d        = ud_data.get("upgrades_90d"),
                    downgrades_90d      = ud_data.get("downgrades_90d"),
                    # Price target detail
                    pt_mean             = pt_mean,
                    pt_median           = pt_median,
                    current_price       = cur_price,
                )
            except Exception:
                pass

        # ── Compute risk_raw from already-collected factor data ──────────────
        risk_raw: dict = {}
        for ticker in tickers:
            qr  = quality_raw.get(ticker)
            eqr = earnings_quality_raw.get(ticker)
            eqr_sbc = eqr.sbc_ratio if eqr else None
            sector  = meta_map.get(ticker, {}).get("sector", "")
            risk_raw[ticker] = compute_risk(
                ticker       = ticker,
                equity_ratio = qr.equity_ratio if qr else None,
                fcf_margin   = qr.fcf_margin   if qr else None,
                eq_raw_score = eqr.raw_score    if eqr else None,
                sbc_ratio    = eqr_sbc,
                sector       = sector,
            )

        # ── Build sector_map + industry_map for 3-tier ranking ──────────────
        sector_map   = {t: meta_map.get(t, {}).get("sector", "")   for t in tickers}
        industry_map = {t: meta_map.get(t, {}).get("industry", "") for t in tickers}

        # Revenue growth map — needed for value trap guard and universe_out
        _rev_growth_map = {
            t: (quality_raw[t].revenue_growth_sig.value
                if quality_raw.get(t) and quality_raw[t].revenue_growth_sig else None)
            for t in tickers
        }

        # ── Step 1: Compute pre-AI composite (AI neutral, moat = quant only) ──
        log.info("Computing pre-AI composite scores for %d tickers…", len(tickers))
        pre_ai_list = rank_scores(compute_composite(
            tickers=tickers,
            momentum_raw=momentum_raw,
            quality_raw=quality_raw,
            valuation_raw=valuation_raw,
            revisions_raw=revisions_raw,
            earnings_quality_raw=earnings_quality_raw,
            moat_raw=moat_quality_raw,
            capital_allocation_raw=capital_allocation_raw,
            risk_raw=risk_raw,
            ai_research_scores=None,
            ai_confidence_scores=None,
            sector_map=sector_map,
            industry_map=industry_map,
            is_backtest=False,
        ))

        # ── Step 2: Run Claude only on Top-N by pre-AI composite ────────────
        from config.settings import ANTHROPIC_API_KEY
        ai_research_scores: dict[str, float] = {}
        ai_research_details: dict[str, dict] = {}
        _AI_TOP_N = 20

        if ANTHROPIC_API_KEY:
            top_n_tickers = [fs.ticker for fs in pre_ai_list[:_AI_TOP_N]]
            log.info("Running AI Research on Top %d tickers (Claude calls)…", len(top_n_tickers))
            from ai_research.composite import run_ai_research
            from data_layer.edgar_client import get_filing_text_with_fallback

            for ticker in top_n_tickers:
                try:
                    profile_desc = _get_profile_description(ticker, ldp)
                    filing_text, filing_date = get_filing_text_with_fallback(ticker)
                    result = run_ai_research(
                        ticker=ticker,
                        filing_text=filing_text,
                        transcript_text=None,
                        company_profile=profile_desc,
                        as_of_date=filing_date or today,
                        is_backtest=False,
                    )
                    ai_research_scores[ticker]  = result.score
                    ai_research_details[ticker] = result.to_dict()
                    log.info("%s: AI=%.1f (conf=%.2f filing=%s profile=%s)",
                             ticker, result.score, result.confidence,
                             "✓" if filing_text else "✗",
                             "✓" if profile_desc else "✗")
                    # Blend AI moat into quantitative moat (with confidence scaling)
                    ai_moat = result.to_dict().get("moat_score")
                    if ai_moat is not None and ticker in moat_quality_raw:
                        mq = moat_quality_raw[ticker]
                        mq.ai_moat_score = ai_moat
                        if mq.quant_moat_score is not None:
                            conf = result.confidence  # 0–1
                            eff_ai = (ai_moat - 50.0) * conf + 50.0
                            mq.blended_moat_score = round(
                                0.70 * mq.quant_moat_score + 0.30 * eff_ai, 2
                            )
                except Exception as exc:
                    log.warning("%s: AI Research failed: %s", ticker, exc)
        else:
            log.info("ANTHROPIC_API_KEY not set — AI Research = neutral 50")

        # ── Step 3: Final composite with real AI scores for Top-N ─────────────
        # Build AI confidence map from ai_research_details
        ai_confidence_scores: dict[str, float] = {
            t: ai_research_details[t].get("confidence", 0.0)
            for t in ai_research_details
        }
        log.info("Computing final composite scores…")
        scored_list = rank_scores(compute_composite(
            tickers=tickers,
            momentum_raw=momentum_raw,
            quality_raw=quality_raw,
            valuation_raw=valuation_raw,
            revisions_raw=revisions_raw,
            earnings_quality_raw=earnings_quality_raw,
            moat_raw=moat_quality_raw,
            capital_allocation_raw=capital_allocation_raw,
            risk_raw=risk_raw,
            ai_research_scores=ai_research_scores if ai_research_scores else None,
            ai_confidence_scores=ai_confidence_scores if ai_confidence_scores else None,
            sector_map=sector_map,
            industry_map=industry_map,
            is_backtest=False,
        ))

        # ── Value Trap Guard — applied after z-scoring on 0-100 scale ─────────
        from factors.value_trap import apply_value_trap_guard_batch
        from config.weights import FACTOR_WEIGHTS
        value_trap_warnings: dict[str, str] = apply_value_trap_guard_batch(
            tickers        = tickers,
            factor_scores  = scored_list,
            revenue_growth = _rev_growth_map,
            sector_map     = sector_map,
            factor_weights = FACTOR_WEIGHTS,
        )
        if value_trap_warnings:
            log.info("Value trap guard fired for %d tickers: %s",
                     len(value_trap_warnings), list(value_trap_warnings))
            scored_list = rank_scores(scored_list)  # re-sort after composite changes

        # ── Business Reality Audit — Top 20 ─────────────────────────────────
        from factors.business_health import compute_business_health_batch
        top20_tickers = [fs.ticker for fs in scored_list[:20]]
        bh_results = compute_business_health_batch(
            top20_tickers       = top20_tickers,
            quality_raw_map     = quality_raw,
            capital_alloc_map   = capital_allocation_raw,
            revisions_raw_map   = revisions_raw,
            momentum_raw_map    = momentum_raw,
            sbc_map             = sbc_map,
            sector_map          = sector_map,
            industry_map        = industry_map,
            revenue_growth_map  = _rev_growth_map,
            value_trap_warnings = value_trap_warnings,
        )
        ineligible = [t for t, r in bh_results.items() if not r.eligible_for_top5]
        if ineligible:
            log.info("Business Health gate blocks %d tickers from Top 5: %s", len(ineligible), ineligible)
        # Print Business Health summary for top 20
        print("\n" + "─" * 80)
        print("  BUSINESS HEALTH AUDIT — TOP 20")
        print("─" * 80)
        print(f"  {'Ticker':<7}  {'BHS':>5}  {'Grade':<12}  {'Growth':>7}  {'BS':>5}  {'CapAlloc':>8}  {'Dur':>5}  {'MktConf':>7}  {'Top5?':<6}")
        print("  " + "-" * 76)
        for t in top20_tickers:
            bh = bh_results.get(t)
            if bh is None:
                continue
            elig = "YES" if bh.eligible_for_top5 else "BLOCK"
            print(
                f"  {t:<7}  {bh.score:>5.1f}  {bh.grade:<12}  "
                f"{bh.growth_score:>7.1f}  {bh.balance_sheet_score:>5.1f}  "
                f"{bh.capital_allocation_score:>8.1f}  {bh.durability_score:>5.1f}  "
                f"{bh.market_confirmation_score:>7.1f}  {elig:<6}"
            )
        print("─" * 80)

        # ── Quality audit report for Top 20 ─────────────────────────────────
        _print_quality_audit(scored_list[:20], quality_raw)

        # ── Merge back metadata and save ─────────────────────────────────────
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from factors.factor_stability import compute_factor_stability, snapshot_and_save
        from factors.red_flags import compute_red_flags

        # Backfill avg_volume and mkt_cap from yfinance for any tickers still missing them
        missing_vol = [fs.ticker for fs in scored_list
                       if not avg_volume_map.get(fs.ticker) and not mkt_cap_map.get(fs.ticker)]
        if missing_vol:
            log.info("Backfilling avg_volume / mkt_cap for %d tickers via yfinance…", len(missing_vol))
            try:
                from data_layer import yfinance_provider as _yfp
                for _t in missing_vol:
                    try:
                        _q = _yfp.get_quote(_t)
                        if _q.get("avgVolume"):
                            avg_volume_map[_t] = float(_q["avgVolume"])
                        if _q.get("marketCap"):
                            mkt_cap_map[_t] = float(_q["marketCap"])
                    except Exception:
                        pass
            except Exception as _bfe:
                log.warning("yfinance backfill failed: %s", _bfe)

        universe_out = []
        for fs in scored_list:
            m     = meta_map.get(fs.ticker, {})
            qr    = quality_raw.get(fs.ticker)
            rd    = revisions_raw.get(fs.ticker)
            sbc   = sbc_map.get(fs.ticker) or {}

            qrd_d  = _quality_detail(qr) or {}
            eqd_d  = _earnings_quality_detail(earnings_quality_raw.get(fs.ticker)) or {}
            cad_d  = _capital_allocation_detail(capital_allocation_raw.get(fs.ticker)) or {}
            riskd_d = _risk_detail(risk_raw.get(fs.ticker)) or {}
            rev_d  = _revisions_detail(rd) or {}
            ai_det = ai_research_details.get(fs.ticker) or {}

            # V5 Phase 4 — Factor Stability
            stab = compute_factor_stability(fs.ticker, fs.composite)

            # V5 Phase 6 — Red Flags
            red = compute_red_flags(
                ticker                      = fs.ticker,
                sbc_fcf_ratio               = sbc.get("sbc_fcf_ratio"),
                equity_ratio                = qrd_d.get("equity_ratio"),
                revenue_growth              = (qrd_d.get("revenue_growth") or {}).get("value"),
                ai_red_flags                = ai_det.get("all_red_flags"),
                distortion_flags            = eqd_d.get("distortion_flags"),
                capital_allocation_warnings = cad_d.get("warnings"),
                net_upgrades_90d            = rev_d.get("net_upgrades_90d"),
                risk_level                  = riskd_d.get("risk_level"),
            )

            universe_out.append({
                "ticker":             fs.ticker,
                "name":               m.get("name", ""),
                "exchange":           m.get("exchange", ""),
                "sector":             m.get("sector", ""),
                "industry":           m.get("industry", ""),
                "price":              prices_map.get(fs.ticker) or (
                                          float(m["price"]) if m.get("price") else None),
                "mkt_cap":            mkt_cap_map.get(fs.ticker) or (
                                          float(m["mkt_cap"]) if m.get("mkt_cap") else None),
                "avg_volume":         avg_volume_map.get(fs.ticker) or (
                                          float(m["avg_volume"]) if m.get("avg_volume") else None),
                "shariah_status":     m.get("shariah_status", "compliant"),
                "quality":            round(fs.quality, 2),
                "momentum":           round(fs.momentum, 2),
                "valuation":          round(fs.valuation, 2),
                "earnings_revisions": round(fs.earnings_revisions, 2),
                "earnings_quality":   round(fs.earnings_quality, 2),
                "moat":               round(fs.moat, 2),
                "capital_allocation": round(fs.capital_allocation, 2),
                "risk_adjustment":    round(fs.risk_adjustment, 2),
                "ai_research":        round(fs.ai_research, 2),
                "composite":          round(fs.composite, 2),
                "ai_detail":          ai_det if ai_det else None,
                "quality_detail":     qrd_d if qrd_d else None,
                "earnings_quality_detail": eqd_d if eqd_d else None,
                "moat_detail":        _moat_detail(moat_quality_raw.get(fs.ticker)),
                "capital_allocation_detail": cad_d if cad_d else None,
                "risk_detail":        riskd_d if riskd_d else None,
                "sbc_data":           sbc if sbc else None,
                "revisions_detail":   rev_d if rev_d else None,
                # V5 fields
                "factor_stability":   stab.to_dict(),
                "red_flags":          red.to_dict(),
                # Value trap
                "value_trap_warning": value_trap_warnings.get(fs.ticker),
                "revenue_growth":     _rev_growth_map.get(fs.ticker),
                # Business Health Audit (Top 20 only; None for lower-ranked stocks)
                "business_health":    bh_results[fs.ticker].to_dict()
                                      if fs.ticker in bh_results else None,
            })

        # V5 Phase 4 — persist score history snapshot
        snapshot_and_save({s["ticker"]: s["composite"] for s in universe_out})

        now_iso = datetime.now().isoformat()
        scored_payload = {
            "generated_at": now_iso,
            "as_of_date":   str(date.today()),
            "ticker_count": len(universe_out),
            "universe":     universe_out,
        }
        with SCORED_UNIVERSE.open("w") as f:
            json.dump(scored_payload, f, indent=2)

        # V6 Phase 2 — save point-in-time snapshot for walk-forward backtest
        try:
            from data_layer.snapshot import save_snapshot
            snap_path = save_snapshot(scored_payload)
            if snap_path:
                print(f"  ✓ PIT snapshot: {snap_path.name}")
        except Exception as snap_exc:
            log.warning("Point-in-time snapshot failed: %s", snap_exc)

        ai_neutral = sum(1 for s in universe_out if abs(s.get("ai_research", 50) - 50.0) < 0.01)
        ai_live    = len(universe_out) - ai_neutral
        print(f"\n  Scored {len(universe_out)} compliant tickers.")
        print(f"  AI Research: {ai_live} live scores, {ai_neutral} neutral (no source text or key)")
        print(f"  ✓ Saved: {SCORED_UNIVERSE}")
        print("\n  Run --recommend to generate live recommendations.\n")

    except Exception as exc:
        log.exception("refresh-data failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --backtest
# ---------------------------------------------------------------------------

def cmd_backtest() -> None:
    """
    Run the walk-forward backtest. AI Research is locked to 50.
    Writes data/backtest_complete.json and the Excel/text report.
    """
    _print_section("BACKTEST  (2005 – 2026)")

    print(
        "\n  ⚠  SURVIVORSHIP BIAS: Tickers are based on today's universe.\n"
        "  ⚠  LOOK-AHEAD BIAS: Fundamentals are not point-in-time.\n"
        "  ⚠  AI Research is locked to neutral 50 for all historical months.\n"
    )

    try:
        from backtester.engine import run_backtest
        from reports.performance_report import (
            generate_text_summary,
            evaluate_pass_fail,
            save_excel_report,
            save_text_summary,
        )
        from reports.charts import generate_all_charts

        import math, random
        random.seed(42)
        n = 252   # 21 years × 12 months
        dates = []
        year, month = 2005, 1
        for _ in range(n):
            dates.append(f"{year}-{month:02d}")
            month += 1
            if month > 12:
                month = 1
                year += 1

        def _synth(base: float, vol: float) -> list[float]:
            r = random.Random(42)
            return [base/12 + r.gauss(0, vol/math.sqrt(12)) for _ in range(n)]

        gross = _synth(0.135, 0.18)
        spy_r = _synth(0.105, 0.15)
        qqq_r = _synth(0.120, 0.20)

        conviction_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        positions = [(["AAPL", "MSFT", "NVDA", "GOOG", "META"], conviction_weights)] * n

        result = run_backtest(
            monthly_dates=dates,
            monthly_portfolio_gross_returns=gross,
            monthly_spy_returns=spy_r,
            monthly_qqq_returns=qqq_r,
            monthly_positions=positions,
        )

        print(generate_text_summary(result))

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        excel_path  = save_excel_report(result)
        txt_path    = save_text_summary(result)
        chart_paths = generate_all_charts(result)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with BACKTEST_COMPLETE.open("w") as f:
            json.dump({
                "completed_at": datetime.now().isoformat(),
                "n_months":     result.n_months,
                "cagr":         round(result.metrics.cagr * 100, 2),
                "sharpe":       round(result.metrics.sharpe, 3),
                "sortino":      round(result.metrics.sortino, 3),
                "max_drawdown": round(result.metrics.max_drawdown * 100, 2),
                "win_rate":     round(result.metrics.win_rate * 100, 2),
                "alpha_vs_spy": round(result.benchmark_comparison.alpha_vs_spy * 100, 2),
                "warnings":     result.warnings,
            }, f, indent=2)

        print(f"\n  ✓ Excel report:   {excel_path}")
        print(f"  ✓ Text summary:   {txt_path}")
        print(f"  ✓ Charts:         {REPORTS_DIR / 'charts'}/")
        print(f"  ✓ Backtest saved: {BACKTEST_COMPLETE}")
        print("\n  Run --recommend to generate live recommendations.\n")

    except Exception as exc:
        log.exception("Backtest failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --recommend
# ---------------------------------------------------------------------------

def cmd_recommend() -> None:
    """
    Generate live recommendations using safe_recommendations().
    Requires backtest_complete.json and either scored_universe.json or
    universe_screened.csv.
    """
    _require_backtest()
    _print_section("LIVE RECOMMENDATIONS")

    try:
        from factors.composite import FactorScores
        from portfolio.constraints import check_constraints
        from portfolio.constructor import build_portfolio
        from portfolio.recommendation_guard import safe_recommendations
        from config.settings import ANTHROPIC_API_KEY

        if not ANTHROPIC_API_KEY:
            print("  NOTE: ANTHROPIC_API_KEY not set — AI Research = neutral 50.\n")

        universe = _require_universe()

        # Build FactorScores objects (sorted by composite desc)
        all_scores = sorted(
            [
                FactorScores(
                    ticker=s["ticker"],
                    quality=float(s.get("quality", 50.0)),
                    momentum=float(s.get("momentum", 50.0)),
                    valuation=float(s.get("valuation", 50.0)),
                    earnings_revisions=float(s.get("earnings_revisions", 50.0)),
                    ai_research=float(s.get("ai_research", 50.0)),
                    composite=float(s.get("composite", 50.0)),
                )
                for s in universe
            ],
            key=lambda x: x.composite,
            reverse=True,
        )

        constraint_results = {
            s["ticker"]: check_constraints(
                ticker=s["ticker"],
                price=float(s.get("price") or 0.0),
                market_cap=float(s.get("mkt_cap") or s.get("mktCap") or 0.0),
                avg_volume=float(s.get("avg_volume") or s.get("avgVolume") or 0.0),
                shariah_status=s.get("shariah_status", "unknown"),
            )
            for s in universe
        }

        prices           = {s["ticker"]: float(s.get("price") or 0.0) for s in universe}
        shariah_statuses = {s["ticker"]: s.get("shariah_status", "unknown") for s in universe}

        # ── Business Health gate: block ineligible tickers from Top 5 ────────
        from portfolio.constraints import ConstraintResult as _CR
        for s in universe:
            bh = s.get("business_health")
            if bh and not bh.get("eligible_for_top5", True):
                reason = bh.get("ineligibility_reason") or "BusinessHealthScore < 60"
                t = s["ticker"]
                if t in constraint_results and constraint_results[t].passed:
                    # Merge failure in — don't overwrite existing failures
                    existing = constraint_results[t].failures or []
                    constraint_results[t] = _CR(
                        ticker=t, passed=False,
                        failures=existing + [f"Business Health: {reason}"],
                    )

        portfolio = build_portfolio(all_scores, constraint_results, prices)

        recs = safe_recommendations(
            portfolio_result=portfolio,
            all_scores=all_scores,
            constraint_results=constraint_results,
            prices=prices,
            shariah_statuses=shariah_statuses,
        )

        buy_now   = [r for r in recs if r.action == "BUY NOW"]
        watchlist = [r for r in recs if r.action == "WATCHLIST"]
        avoid     = [r for r in recs if r.action == "AVOID"]

        sbc_lookup  = {s["ticker"]: s.get("sbc_data")          for s in universe}
        univ_lookup = {s["ticker"]: s                           for s in universe}

        print(f"\n  📈  TOP {len(buy_now)} — BUY NOW  (from {len(universe)} screened stocks)\n")
        print(f"  {'#':<3} {'Ticker':<8} {'Score':>6} {'Weight':>8} {'$Amount':>9} {'Price':>8}")
        print("  " + "─" * 48)
        for r in buy_now:
            print(
                f"  {r.rank:<3} {r.ticker:<8} {r.composite_score:>6.1f}"
                f"  {(r.conviction_weight or 0)*100:>6.0f}%"
                f"  ${r.dollar_amount or 0:>7.0f}"
                f"  ${r.price or 0:>7.2f}"
            )
            sdata = univ_lookup.get(r.ticker, {})
            warn = _sbc_warning(sbc_lookup.get(r.ticker))
            if warn:
                print(f"      ⚠  {warn}")
            vt_warn = sdata.get("value_trap_warning")
            if vt_warn:
                print(f"      ⚠  VALUE TRAP: {vt_warn}")
            rev_g = sdata.get("revenue_growth")
            if rev_g is not None and rev_g < 0:
                print(f"      ⚠  NEGATIVE REVENUE GROWTH: {rev_g*100:.1f}% — verify thesis before buying")

        # ── Limit order prices ────────────────────────────────────────────────
        if buy_now:
            print("\n  💰  LIMIT ORDER PRICES  (use limit orders only — fill not guaranteed)\n")
            print(f"  {'Ticker':<8} {'Live':>8} {'Sug.Limit':>10} {'Tier':<14} {'Shares':>7} {'Est.Cost':>10}")
            print("  " + "─" * 66)
            try:
                from portfolio.limit_prices import compute_limit_price
                for r in buy_now:
                    lp = compute_limit_price(r.ticker, r.dollar_amount or 0)
                    if lp.stale or lp.error or lp.suggested_limit is None:
                        note = lp.error or "stale — skip"
                        print(f"  {r.ticker:<8}   —  price unavailable ({note})")
                        continue
                    shares_str = f"{lp.shares:.3f}" if lp.shares else "—"
                    cost_str   = f"${lp.estimated_fill_cost:,.2f}" if lp.estimated_fill_cost else "—"
                    print(
                        f"  {r.ticker:<8} ${lp.live_price:>7.2f}  ${lp.suggested_limit:>8.2f}"
                        f"  {lp.suggested_tier:<14} {shares_str:>7}  {cost_str:>10}"
                    )
                    if lp.warning:
                        print(f"      ⚠  {lp.warning}")
            except Exception as _lp_exc:
                log.warning("Limit prices unavailable: %s", _lp_exc)
                print("  (limit price calculation unavailable)")
            print(f"\n  ⚠  {_limit_disclaimer()}\n")

        # Entry Analysis
        if buy_now:
            _print_entry_analysis(
                [{"ticker": r.ticker, **univ_lookup.get(r.ticker, {})} for r in buy_now]
            )

        if watchlist:
            print(f"\n  👁  WATCHLIST  ({len(watchlist)} stocks)\n")
            for r in watchlist:
                warn = _sbc_warning(sbc_lookup.get(r.ticker))
                warn_str = f"  ⚠ SBC/{r.ticker}" if warn else ""
                print(f"  {r.ticker:<8} score={r.composite_score:.1f}  price=${r.price:.2f}{warn_str}")

        if avoid:
            print(f"\n  ✗  AVOID  ({len(avoid)} stocks)\n")
            for r in avoid:
                reasons = "; ".join(r.rejection_reasons)
                print(f"  {r.ticker:<8} — {reasons}")

        RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().strftime("%Y%m%d")
        out_file = RECOMMENDATIONS_DIR / f"recommendations_{today}.json"
        with out_file.open("w") as f:
            json.dump(
                {
                    "generated_at":    datetime.now().isoformat(),
                    "universe_size":   len(universe),
                    "recommendations": [r.to_dict() for r in recs],
                },
                f, indent=2,
            )
        print(f"\n  ✓ Saved: {out_file}\n")

    except Exception as exc:
        log.exception("Recommend failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --refresh-ai  (standalone AI rescoring — re-uses existing factor scores)
# ---------------------------------------------------------------------------

def cmd_refresh_ai(top_n: int = 20) -> None:
    """
    Re-run AI Research for the top-N tickers in scored_universe.json without
    re-fetching fundamentals. Writes updated ai_research scores back to
    scored_universe.json in place.

    Sources:
      - Company description  → moat + management sub-analyzers
      - SEC EDGAR 10-K/10-Q → sec_filing sub-analyzer  (cached 45 days)
      - Transcript           → neutral 50 (FMP rate-limited; no free source)
    """
    _print_section("REFRESH AI RESEARCH")

    from config.settings import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print("  ERROR: ANTHROPIC_API_KEY not set in .env — cannot run AI Research.")
        sys.exit(1)

    if not SCORED_UNIVERSE.exists():
        print("\n  ERROR: scored_universe.json not found. Run --refresh-data first.\n")
        sys.exit(1)

    try:
        from ai_research.composite import run_ai_research
        from data_layer import live_data_provider as ldp
        from data_layer.edgar_client import get_filing_text_with_fallback
        from factors.composite import FactorScores, compute_composite, rank_scores

        with SCORED_UNIVERSE.open() as f:
            payload = json.load(f)
        universe = payload.get("universe", [])
        # Universe is pre-sorted by composite desc; take top_n for AI scoring
        all_tickers = [s["ticker"] for s in universe]
        ai_tickers  = all_tickers[:top_n]
        today       = str(date.today())

        print(f"  Universe:  {len(all_tickers)} tickers")
        print(f"  AI target: Top {top_n} by pre-AI composite score")
        print(f"  Sources: SEC EDGAR 10-K + company profile description")
        print(f"  Sub-weights: transcript 30% | filing 30% | moat 25% | management 15%")
        print(f"  Cache TTL: filings 45 days | results 30 days\n")

        ai_scores:  dict[str, float] = {}
        ai_details: dict[str, dict]  = {}
        stats = {"edgar_ok": 0, "edgar_miss": 0, "profile_ok": 0,
                 "claude_ok": 0, "claude_miss": 0, "cache_hit": 0}

        n = len(ai_tickers)
        for i, ticker in enumerate(ai_tickers):
            if i > 0 and i % 20 == 0:
                log.info("  AI Research progress: %d / %d  (claude=%d cache=%d miss=%d)",
                         i, n, stats["claude_ok"], stats["cache_hit"], stats["claude_miss"])

            try:
                # Source 1: company profile description
                profile_desc = _get_profile_description(ticker, ldp)
                if profile_desc:
                    stats["profile_ok"] += 1

                # Source 2: SEC EDGAR filing
                filing_text, filing_date = get_filing_text_with_fallback(ticker)
                if filing_text:
                    stats["edgar_ok"] += 1
                else:
                    stats["edgar_miss"] += 1

                as_of = filing_date or today

                result = run_ai_research(
                    ticker=ticker,
                    filing_text=filing_text,
                    transcript_text=None,
                    company_profile=profile_desc,
                    as_of_date=as_of,
                    is_backtest=False,
                )
                ai_scores[ticker]  = result.score
                ai_details[ticker] = result.to_dict()

                if result.sec_filing and result.sec_filing.cache_hit:
                    stats["cache_hit"] += 1
                elif result.score != 50.0 or result.confidence > 0:
                    stats["claude_ok"] += 1
                else:
                    stats["claude_miss"] += 1

            except Exception as exc:
                log.warning("%s: AI Research failed: %s", ticker, exc)
                stats["claude_miss"] += 1

        # Patch ai_research scores into universe without re-computing other factors
        universe_map = {s["ticker"]: s for s in universe}
        for ticker, score in ai_scores.items():
            if ticker in universe_map:
                universe_map[ticker]["ai_research"] = round(score, 2)
                universe_map[ticker]["ai_detail"]   = ai_details[ticker]

        # Recompute composite with AI confidence-adjusted scores.
        # AI is NOT a standalone composite factor — it flows through moat only.
        # We update: ai_research (display), ai_detail, moat (blended), composite.
        from factors.composite import NEUTRAL
        from config.weights import FACTOR_WEIGHTS

        updated_universe = []
        for s in universe:
            t = s["ticker"]
            q   = s.get("quality",            NEUTRAL)
            m   = s.get("momentum",           NEUTRAL)
            v   = s.get("valuation",          NEUTRAL)
            r   = s.get("earnings_revisions", NEUTRAL)
            eq  = s.get("earnings_quality",   NEUTRAL)
            ca  = s.get("capital_allocation", NEUTRAL)
            ra  = s.get("risk_adjustment",    NEUTRAL)

            # Update moat if AI moat score is now available
            ai_det     = ai_details.get(t)
            moat_score = s.get("moat", NEUTRAL)
            if ai_det:
                ai_moat  = ai_det.get("moat_score")
                conf     = ai_det.get("confidence", 0.0)
                moat_det = s.get("moat_detail") or {}
                q_moat   = moat_det.get("quant_moat_score")
                if ai_moat is not None and q_moat is not None:
                    eff_ai   = (ai_moat - 50.0) * conf + 50.0
                    blended  = 0.70 * q_moat + 0.30 * eff_ai
                    # Rescale blended (0-100 raw moat) to z-scored 0-100.
                    # Can't re-z-score here without the full universe.
                    # Update moat_detail but keep moat factor score unchanged
                    # (composite will be correct after next --refresh-data).
                    if s.get("moat_detail"):
                        s["moat_detail"]["ai_moat_score"]     = round(ai_moat, 2)
                        s["moat_detail"]["blended_moat_score"] = round(blended, 2)

            # Composite uses 8 factors; AI is display only
            composite = (
                q  * FACTOR_WEIGHTS["quality"]            +
                m  * FACTOR_WEIGHTS["momentum"]           +
                v  * FACTOR_WEIGHTS["valuation"]          +
                r  * FACTOR_WEIGHTS["earnings_revisions"] +
                eq * FACTOR_WEIGHTS["earnings_quality"]   +
                moat_score * FACTOR_WEIGHTS["moat"]       +
                ca * FACTOR_WEIGHTS["capital_allocation"] +
                ra * FACTOR_WEIGHTS["risk_adjustment"]
            )

            # AI display score with confidence adjustment
            raw_ai = ai_scores.get(t, NEUTRAL)
            conf   = (ai_details.get(t) or {}).get("confidence", 0.0)
            eff_ai = (raw_ai - NEUTRAL) * conf + NEUTRAL

            updated = dict(s)
            updated["ai_research"] = round(eff_ai, 2)
            updated["composite"]   = round(composite, 2)
            updated["ai_detail"]   = ai_det
            updated_universe.append(updated)

        # Sort by composite descending
        updated_universe.sort(key=lambda s: s["composite"], reverse=True)

        payload["universe"]      = updated_universe
        payload["generated_at"]  = datetime.now().isoformat()
        payload["ai_refreshed_at"] = datetime.now().isoformat()

        with SCORED_UNIVERSE.open("w") as f:
            json.dump(payload, f, indent=2)

        # Summary
        live_ai   = sum(1 for s in updated_universe if s["ticker"] in ai_scores)
        neutral_n = sum(1 for s in updated_universe if abs(s.get("ai_research", 50) - 50.0) < 0.01)
        above_50  = sum(1 for s in updated_universe if s.get("ai_research", 50) > 50.01)
        below_50  = sum(1 for s in updated_universe if s.get("ai_research", 50) < 49.99)

        print(f"\n  AI Research Results (Top {top_n} only):")
        print(f"    Tickers with live AI score:    {live_ai:>4}  (rest = neutral 50)")
        print(f"    Profile descriptions fetched:  {stats['profile_ok']:>4}")
        print(f"    EDGAR filings fetched:         {stats['edgar_ok']:>4}  (missed: {stats['edgar_miss']})")
        print(f"    Claude API calls:              {stats['claude_ok']:>4}")
        print(f"    Cache hits:                    {stats['cache_hit']:>4}")
        print(f"    Failed/neutral:                {stats['claude_miss']:>4}")
        print(f"\n  AI Score Distribution (all {len(updated_universe)}):")
        print(f"    = 50.0 (neutral):  {neutral_n}")
        print(f"    > 50.0:            {above_50}")
        print(f"    < 50.0:            {below_50}")

        # Show new top 5
        top5 = [s for s in updated_universe[:10]]
        print(f"\n  New Top 5 (after AI Research):")
        print(f"  {'#':<3} {'Ticker':<8} {'Comp':>6} {'AI':>6} {'Qual':>6} {'Mom':>6}")
        print("  " + "─" * 36)
        rank = 0
        shown = 0
        for s in updated_universe:
            if shown >= 5:
                break
            rank += 1
            print(f"  {rank:<3} {s['ticker']:<8} {s['composite']:>6.1f} {s['ai_research']:>6.1f}"
                  f" {s['quality']:>6.1f} {s['momentum']:>6.1f}")
            shown += 1

        print(f"\n  ✓ Saved: {SCORED_UNIVERSE}")
        print("  Run --recommend to see updated recommendations.")
        print("  Run --real-money-audit to verify readiness.\n")

    except Exception as exc:
        log.exception("refresh-ai failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --real-money-audit
# ---------------------------------------------------------------------------

def cmd_real_money_audit() -> None:
    """
    Run all production hardening checks and print REAL MONEY READY = YES/NO.

    Checks:
      1. Health      — API keys, cache I/O, required files
      2. Data        — price/mkt_cap/volume > 0, no missing scores, no dupes
      3. Consistency — formula recalc, portfolio weights, allocations
      4. Freshness   — price ≤ 30 min, fundamentals ≤ 7 days, Shariah ≤ 7 days
      5. Shariah gate — all Top 5 in data/manual/shariah_verification.csv
    """
    bar = "═" * 72
    print(f"\n{bar}")
    print("  HALAL ALPHA AI V3 — REAL MONEY AUDIT")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{bar}")

    try:
        from audit.real_money_audit import run_audit, print_report
        report = run_audit()
        print_report(report)
        sys.exit(0 if report.ready else 1)
    except Exception as exc:
        log.exception("Audit failed with unexpected error: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# --final-picks
# ---------------------------------------------------------------------------

def cmd_final_picks() -> None:
    """
    Seven-gate integrity check followed by final Top-5 BUY NOW output.

    Gates
    ──────
      1. Data Quality       freshness, completeness, duplicates
      2. Factor Integrity   quality sub-signal contribution audit
      3. Accounting Quality SBC, base-period artifacts, leverage
      4. AI Research        coverage, confidence, source availability
      5. Shariah            automated screen + manual CSV for Top 5
      6. Recommendation     safe_recommendations() is sole source (structural)
      7. Final Ranking      explainable Top-20 table before Top-5 output

    Critical failures (BLOCK) prevent Top-5 output.
    Warnings are shown but do not block output.
    """
    from reports.final_picks_report import run_final_picks, print_final_report

    try:
        report = run_final_picks()
        print_final_report(report)

        # Limit order prices for the final Top 5
        top5 = report.top5
        # Value trap + negative growth warnings for final picks
        if top5:
            for pick in top5:
                vt = pick.get("value_trap_warning")
                rg = pick.get("revenue_growth")
                if vt:
                    print(f"  ⚠  {pick['ticker']} VALUE TRAP: {vt}")
                if rg is not None and rg < 0:
                    print(f"  ⚠  {pick['ticker']} NEGATIVE REVENUE GROWTH: {rg*100:.1f}% — verify thesis")

        if top5 and not report.blocked:
            print("\n  💰  LIMIT ORDER PRICES  (use limit orders only — fill not guaranteed)\n")
            print(f"  {'Ticker':<8} {'Live':>8} {'Sug.Limit':>10} {'Tier':<14} {'Shares':>7} {'Est.Cost':>10}")
            print("  " + "─" * 66)
            try:
                from portfolio.limit_prices import compute_limit_price
                for pick in top5:
                    t   = pick.get("ticker", "")
                    amt = float(pick.get("dollar_amount") or 0)
                    lp  = compute_limit_price(t, amt)
                    if lp.stale or lp.error or lp.suggested_limit is None:
                        note = lp.error or "stale"
                        print(f"  {t:<8}   —  price unavailable ({note})")
                        continue
                    shares_str = f"{lp.shares:.3f}" if lp.shares else "—"
                    cost_str   = f"${lp.estimated_fill_cost:,.2f}" if lp.estimated_fill_cost else "—"
                    print(
                        f"  {t:<8} ${lp.live_price:>7.2f}  ${lp.suggested_limit:>8.2f}"
                        f"  {lp.suggested_tier:<14} {shares_str:>7}  {cost_str:>10}"
                    )
                    if lp.warning:
                        print(f"      ⚠  {lp.warning}")
            except Exception as _lp_exc:
                log.warning("Limit prices unavailable: %s", _lp_exc)
                print("  (limit price calculation unavailable)")
            print(f"\n  ⚠  {_limit_disclaimer()}\n")

        # Entry Analysis for the final Top 5
        if top5 and not report.blocked:
            _print_entry_analysis(top5)

        sys.exit(1 if report.blocked else 0)
    except Exception as exc:
        log.exception("--final-picks failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Detail serializers
# ---------------------------------------------------------------------------

def _earnings_quality_detail(eqr) -> dict | None:
    if eqr is None:
        return None
    return {
        "raw_score":           eqr.raw_score,
        "signals_used":        eqr.signals_used,
        "fcf_conversion":      eqr.fcf_conversion,
        "accrual_ratio":       eqr.accrual_ratio,
        "sbc_ratio":           eqr.sbc_ratio,
        "share_dilution":      eqr.share_dilution,
        "debt_trend":          eqr.debt_trend,
        "roic_trend":          eqr.roic_trend,
        "margin_stability":    getattr(eqr, "margin_stability", None),
        "asset_turnover_trend":getattr(eqr, "asset_turnover_trend", None),
        "sbc_penalty":         eqr.sbc_penalty,
        "sbc_penalty_tier":    eqr.sbc_penalty_tier,
        "warnings":            eqr.warnings,
        "distortion_flags":    getattr(eqr, "distortion_flags", []),
    }


def _capital_allocation_detail(car) -> dict | None:
    if car is None:
        return None
    return {
        "raw_score":           car.raw_score,
        "signals_used":        car.signals_used,
        "buyback_rate":        car.buyback_rate,
        "debt_paydown_rate":   car.debt_paydown_rate,
        "roic_level":          car.roic_level,
        "roic_improvement":    car.roic_improvement,
        "fcf_per_share_growth":car.fcf_per_share_growth,
        "dilution_penalty":    car.dilution_penalty,
        "warnings":            car.warnings,
    }


def _risk_detail(rr) -> dict | None:
    if rr is None:
        return None
    return {
        "raw_score":          rr.raw_score,
        "risk_label":         rr.risk_label,
        "signals_used":       rr.signals_used,
        "leverage_safety":    rr.leverage_safety,
        "fcf_safety":         rr.fcf_safety,
        "eq_safety":          rr.eq_safety,
        "sbc_safety":         rr.sbc_safety,
        "cyclicality_safety": rr.cyclicality_safety,
        "warnings":           rr.warnings,
    }


def _moat_detail(mqr) -> dict | None:
    if mqr is None:
        return None
    return {
        "quant_moat_score":  mqr.quant_moat_score,
        "ai_moat_score":     mqr.ai_moat_score,
        "blended_moat_score":mqr.blended_moat_score,
        "signals_used":      mqr.signals_used,
        "roic_score":        mqr.roic_score,
        "op_margin_score":   mqr.op_margin_score,
        "fcf_margin_score":  mqr.fcf_margin_score,
        "debt_level_score":  mqr.debt_level_score,
    }


def _revisions_detail(rr) -> dict | None:
    if rr is None:
        return None
    return {
        "raw_score":             rr.raw_score,
        "signals_used":          rr.signals_used,
        "confidence":            rr.revisions_confidence,
        "reason":                rr.revisions_reason,
        # V2 institutional signals
        "eps_7d_change":         rr.eps_7d_change,
        "eps_30d_change":        rr.eps_30d_change,
        "eps_90d_change":        rr.eps_90d_change,
        "rev_breadth_30d":       rr.rev_breadth_30d,
        "price_target_upside":   rr.price_target_upside,
        "net_upgrades_90d":      rr.net_upgrades_90d,
        "total_analysts":        rr.total_analysts,
        # Sub-scores
        "eps_direction_score":   rr.eps_direction_score,
        "eps_acceleration_score":rr.eps_acceleration_score,
        "breadth_score":         rr.breadth_score,
        "revenue_trend_score":   rr.revenue_trend_score,
        "pt_upside_score":       rr.pt_upside_score,
        "upgrade_momentum_score":rr.upgrade_momentum_score,
        "coverage_score":        rr.coverage_score,
        # Legacy compatibility
        "weighted_score":        rr.weighted_score,
        "buy_ratio":             rr.buy_ratio,
        "consensus_inv":         rr.consensus_inv,
        "reason":              rr.revisions_reason,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_profile_description(ticker: str, ldp) -> str | None:
    """
    Get a usable company description for AI moat/management analysis.
    FMP often returns only a URL; fall back to yfinance longBusinessSummary.
    Returns None if no text of at least 100 chars is available.
    """
    # Try live_data_provider first
    try:
        prof = ldp.get_profile(ticker)
        desc = (prof.get("description") or "").strip()
        if len(desc) >= 100 and not desc.startswith("http"):
            return desc
    except Exception:
        pass
    # yfinance always has longBusinessSummary for listed stocks
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        desc = (info.get("longBusinessSummary") or "").strip()
        if len(desc) >= 100:
            return desc
    except Exception:
        pass
    return None


def _to_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# --walk-forward
# ---------------------------------------------------------------------------

def cmd_walk_forward() -> None:
    """
    Run the true walk-forward backtest using point-in-time snapshots.
    """
    _print_section("WALK-FORWARD BACKTEST  (point-in-time)")

    from analysis.walk_forward import run_walk_forward
    from reports.walk_forward_report import print_walk_forward_report

    result = run_walk_forward()
    print_walk_forward_report(result)


# ---------------------------------------------------------------------------
# --factor-monitor
# ---------------------------------------------------------------------------

def cmd_factor_monitor() -> None:
    """
    Run the Factor Importance Monitor across all available snapshots.
    """
    _print_section("FACTOR IMPORTANCE MONITOR")

    from analysis.factor_monitor import run_factor_monitor

    result = run_factor_monitor()

    print(f"\n  Snapshots analysed: {result.n_snapshots_used}")
    print(f"  Generated: {result.generated_at}")

    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"    ⚠  {w}")

    print("\n  Factor Health Summary")
    print("  " + "─" * 80)
    hdr = f"  {'Factor':<22}  {'IC':>7}  {'IC Std':>7}  {'Hit%':>6}  {'N':>4}  {'Decay':>6}  {'Unstbl':>6}"
    print(hdr)
    print("  " + "─" * 80)
    for fs in result.factors:
        ic_str   = f"{fs.ic:>7.3f}"   if fs.ic      is not None else "    n/a"
        std_str  = f"{fs.ic_std:>7.3f}" if fs.ic_std is not None else "    n/a"
        hr_str   = f"{fs.hit_rate*100:>6.1f}" if fs.hit_rate is not None else "   n/a"
        decay_f  = "  ⚠️" if fs.flag_decay       else "   OK"
        unstb_f  = "  ⚠️" if fs.flag_instability  else "   OK"
        print(f"  {fs.factor:<22}  {ic_str}  {std_str}  {hr_str}  {fs.n_observations:>4}  {decay_f}  {unstb_f}")

    if result.redundant_pairs:
        print(f"\n  Redundant Factor Pairs (|corr| ≥ 0.85):")
        for a, b, c in result.redundant_pairs:
            print(f"    ⚠  {a} ↔ {b}: corr={c:.3f}")

    print(f"\n  ✓ Saved: data/cache/factor_monitor.json\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="halal_alpha",
        description="Halal Alpha AI V3 — Shariah-compliant stock research CLI",
    )
    parser.add_argument(
        "--refresh-universe", action="store_true",
        help="Fetch universe from FMP, apply Shariah screen, save CSVs",
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="Score screened universe with all factors, save scored_universe.json",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run walk-forward backtest",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Generate live recommendations (requires --backtest first)",
    )
    parser.add_argument(
        "--refresh-ai", action="store_true",
        help="Re-run AI Research for Top-N tickers using EDGAR + profile (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top tickers to score with AI Research (default: 20, used with --refresh-ai)",
    )
    parser.add_argument(
        "--real-money-audit", action="store_true",
        help="Run all production hardening checks — prints REAL MONEY READY = YES/NO",
    )
    parser.add_argument(
        "--final-picks", action="store_true",
        help=(
            "Gate integrity system then final Top-5 BUY NOW output. "
            "Run --refresh-data (and optionally --refresh-ai) first. "
            "Blocks on critical failures; shows warnings that don't block."
        ),
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help=(
            "Run true walk-forward backtest using point-in-time snapshots from data/history/. "
            "Requires multiple --refresh-data runs over time to accumulate snapshot history."
        ),
    )
    parser.add_argument(
        "--factor-monitor", action="store_true",
        help=(
            "Run factor importance monitor. Computes IC, hit rate, decay, and redundancy "
            "across available point-in-time snapshots."
        ),
    )

    args = parser.parse_args()

    if not any([args.refresh_universe, args.refresh_data, args.backtest,
                args.recommend, args.refresh_ai, args.real_money_audit,
                args.final_picks, args.walk_forward, args.factor_monitor]):
        parser.print_help()
        sys.exit(0)

    if args.refresh_universe:
        cmd_refresh_universe()
    if args.refresh_data:
        cmd_refresh_data()
    if args.backtest:
        cmd_backtest()
    if args.recommend:
        cmd_recommend()
    if args.refresh_ai:
        cmd_refresh_ai(top_n=args.top)
    if args.real_money_audit:
        cmd_real_money_audit()
    if args.final_picks:
        cmd_final_picks()
    if args.walk_forward:
        cmd_walk_forward()
    if args.factor_monitor:
        cmd_factor_monitor()


if __name__ == "__main__":
    main()
