"""
Shared session state helpers.

All expensive calls are cached in st.session_state so navigating between
pages does not re-run them. Each getter checks the cache first.

CRITICAL: get_recommendations() calls safe_recommendations() from
portfolio.recommendation_guard — the exact same function used by
`python3 main.py --recommend`. There is no separate recommendation logic here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from portfolio.constructor import PortfolioResult
    from portfolio.recommendation_guard import Recommendation
    from factors.composite import FactorScores
    from portfolio.constraints import ConstraintResult

log = logging.getLogger(__name__)

BACKTEST_COMPLETE  = Path("data/backtest_complete.json")
SCORED_UNIVERSE    = Path("data/cache/scored_universe.json")
UNIVERSE_SCREENED  = Path("data/cache/universe_screened.csv")

# ---------------------------------------------------------------------------
# Universe loaders
# ---------------------------------------------------------------------------

def _load_scored_universe() -> list[dict]:
    """
    Load scored_universe.json written by `python3 main.py --refresh-data`.
    Returns [] when the file does not exist.
    """
    if SCORED_UNIVERSE.exists():
        with SCORED_UNIVERSE.open() as f:
            import json as _json
            payload = _json.load(f)
            return payload.get("universe", [])
    return []


def _load_screened_csv() -> list[dict]:
    """
    Load universe_screened.csv written by `python3 main.py --refresh-universe`.
    Returns [] when the file does not exist.
    """
    from screener.live_universe import load_screened_csv
    return load_screened_csv()


def _build_inputs() -> tuple:
    """
    Build factor scores, constraint results, prices, and Shariah statuses.

    Priority:
      1. data/cache/scored_universe.json  — real factor scores (after --refresh-data)
      2. data/cache/universe_screened.csv — real tickers, neutral-50 scores
                                            (after --refresh-universe but before scoring)
      3. If neither exists, returns empty inputs with a logged warning.
    """
    from factors.composite import FactorScores
    from portfolio.constraints import check_constraints

    scored = _load_scored_universe()
    if scored:
        source = scored
        use_real_scores = True
    else:
        csv_rows = _load_screened_csv()
        if csv_rows:
            # Build minimal records with neutral-50 scores — real scores need --refresh-data
            source = [
                {
                    "ticker":         r["ticker"],
                    "name":           r.get("name", ""),
                    "sector":         r.get("sector", ""),
                    "price":          float(r["price"]) if r.get("price") else 0.0,
                    "mkt_cap":        float(r["mkt_cap"]) if r.get("mkt_cap") else 0.0,
                    "avg_volume":     float(r["avg_volume"]) if r.get("avg_volume") else 0.0,
                    "shariah_status": r.get("shariah_status", "unknown"),
                    # Neutral scores until --refresh-data is run
                    "composite": 50.0, "quality": 50.0, "momentum": 50.0,
                    "valuation": 50.0, "earnings_revisions": 50.0,
                    "earnings_quality": 50.0, "moat": 50.0,
                    "capital_allocation": 50.0, "risk_adjustment": 50.0,
                    "ai_research": 50.0,
                }
                for r in csv_rows
            ]
            use_real_scores = False
            log.warning(
                "No scored_universe.json found — using neutral-50 factor scores. "
                "Run `python3 main.py --refresh-data` to compute real scores."
            )
        else:
            log.error(
                "No universe data found. "
                "Run `python3 main.py --refresh-universe` first."
            )
            return [], {}, {}, {}

    all_scores = sorted(
        [
            FactorScores(
                ticker=s["ticker"],
                quality=float(s.get("quality", 50.0)),
                momentum=float(s.get("momentum", 50.0)),
                valuation=float(s.get("valuation", 50.0)),
                earnings_revisions=float(s.get("earnings_revisions", 50.0)),
                earnings_quality=float(s.get("earnings_quality", 50.0)),
                moat=float(s.get("moat", 50.0)),
                capital_allocation=float(s.get("capital_allocation", 50.0)),
                risk_adjustment=float(s.get("risk_adjustment", 50.0)),
                ai_research=float(s.get("ai_research", 50.0)),
                composite=float(s.get("composite", 50.0)),
            )
            for s in source
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
        for s in source
    }

    prices          = {s["ticker"]: float(s.get("price") or 0.0) for s in source}
    shariah_statuses = {s["ticker"]: s.get("shariah_status", "unknown") for s in source}
    return all_scores, constraint_results, prices, shariah_statuses


# ---------------------------------------------------------------------------
# Cached getters
# ---------------------------------------------------------------------------

def get_universe() -> list[dict]:
    """
    Return the screened universe. Prefers scored_universe.json; falls back to
    universe_screened.csv; returns [] if neither exists.
    """
    if "universe" not in st.session_state:
        scored = _load_scored_universe()
        if scored:
            st.session_state["universe"] = scored
        else:
            rows = _load_screened_csv()
            st.session_state["universe"] = [
                {
                    "ticker":         r["ticker"],
                    "name":           r.get("name", ""),
                    "sector":         r.get("sector", ""),
                    "price":          float(r["price"]) if r.get("price") else None,
                    "mkt_cap":        float(r["mkt_cap"]) if r.get("mkt_cap") else None,
                    "avg_volume":     float(r["avg_volume"]) if r.get("avg_volume") else None,
                    "shariah_status": r.get("shariah_status", "unknown"),
                    "composite": None,
                }
                for r in rows
            ]
    return st.session_state["universe"]


def get_factor_scores() -> list:
    if "factor_scores" not in st.session_state:
        all_scores, _, _, _ = _build_inputs()
        st.session_state["factor_scores"] = all_scores
    return st.session_state["factor_scores"]


def get_constraint_results() -> dict:
    if "constraint_results" not in st.session_state:
        _, crs, _, _ = _build_inputs()
        st.session_state["constraint_results"] = crs
    return st.session_state["constraint_results"]


def get_prices() -> dict:
    if "prices" not in st.session_state:
        _, _, prices, _ = _build_inputs()
        st.session_state["prices"] = prices
    return st.session_state["prices"]


def get_shariah_statuses() -> dict:
    if "shariah_statuses" not in st.session_state:
        _, _, _, shariah = _build_inputs()
        st.session_state["shariah_statuses"] = shariah
    return st.session_state["shariah_statuses"]


def get_portfolio() -> "PortfolioResult":
    """Build portfolio using the constructor — same pipeline as CLI."""
    if "portfolio" not in st.session_state:
        from portfolio.constructor import build_portfolio
        scores = get_factor_scores()
        crs    = get_constraint_results()
        prices = get_prices()
        st.session_state["portfolio"] = build_portfolio(scores, crs, prices)
    return st.session_state["portfolio"]


def get_recommendations() -> list:
    """
    Load recommendations from the most recent saved JSON file (written by
    --recommend). Falls back to live computation only if no file exists.
    """
    if "recommendations" not in st.session_state:
        from portfolio.recommendation_guard import Recommendation
        import json, glob

        # Try to load from saved recommendations file (most recent date)
        files = sorted(glob.glob("data/recommendations/recommendations_*.json"))
        if files:
            with open(files[-1]) as f:
                data = json.load(f)
            recs = []
            for r in data.get("recommendations", []):
                recs.append(Recommendation(
                    ticker=r["ticker"],
                    action=r["action"],
                    rank=r.get("rank"),
                    composite_score=r["composite_score"],
                    conviction_weight=r.get("conviction_weight"),
                    dollar_amount=r.get("dollar_amount"),
                    price=r.get("price"),
                    shariah_status=r.get("shariah_status", "unknown"),
                    rejection_reasons=r.get("rejection_reasons", []),
                ))
            st.session_state["recommendations"] = recs
        else:
            # Fallback: live computation with BH gate
            from portfolio.recommendation_guard import safe_recommendations
            from portfolio.constraints import ConstraintResult as _CR
            import copy
            portfolio = get_portfolio()
            scores    = get_factor_scores()
            crs       = copy.deepcopy(get_constraint_results())
            prices    = get_prices()
            shariah   = get_shariah_statuses()
            universe  = get_universe()
            for s in universe:
                bh = s.get("business_health")
                if bh and not bh.get("eligible_for_top5", True):
                    reason = bh.get("ineligibility_reason") or "BusinessHealthScore < 60"
                    t = s["ticker"]
                    if t in crs and crs[t].passed:
                        crs[t] = _CR(ticker=t, passed=False,
                                     failures=[f"Business Health: {reason}"])
            st.session_state["recommendations"] = safe_recommendations(
                portfolio_result=portfolio, all_scores=scores,
                constraint_results=crs, prices=prices, shariah_statuses=shariah,
            )
    return st.session_state["recommendations"]


def get_backtest_result() -> dict | None:
    """Load backtest_complete.json if it exists, else None."""
    if "backtest_result" not in st.session_state:
        if BACKTEST_COMPLETE.exists():
            with BACKTEST_COMPLETE.open() as f:
                st.session_state["backtest_result"] = json.load(f)
        else:
            st.session_state["backtest_result"] = None
    return st.session_state["backtest_result"]


def clear_recommendations_cache() -> None:
    """Force re-computation of portfolio + recommendations on next access."""
    for key in ("universe", "portfolio", "recommendations", "factor_scores",
                "constraint_results", "prices", "shariah_statuses"):
        st.session_state.pop(key, None)


@st.cache_data(ttl=60)
def get_live_prices_cached(tickers: tuple) -> dict:
    """
    Fetch semi-live prices for the given tickers.
    Cache TTL = 60 s — Streamlit auto-invalidates on next call after expiry.
    Returns {ticker: {"price": float, "change_pct": float, "source": str, "timestamp": str}}.
    """
    try:
        from data_layer.live_data_provider import get_live_prices_batch
        return get_live_prices_batch(list(tickers))
    except Exception as exc:
        log.warning("Live price batch fetch failed: %s", exc)
        return {}


def clear_live_prices_cache() -> None:
    """Force next get_live_prices_cached call to re-fetch from provider."""
    get_live_prices_cached.clear()


# ---------------------------------------------------------------------------
# Backend command runners
# ---------------------------------------------------------------------------

import subprocess  # noqa: E402
import sys


def _run_cmd_sync(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    """
    Run main.py synchronously. Only used for FAST commands (< 2 min).
    Never call this for --refresh-data or --refresh-ai — use jobs.py instead.
    """
    result = subprocess.run(
        [sys.executable, "main.py"] + args,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# ── Slow commands → background jobs (non-blocking) ───────────────────────────

def run_refresh_data() -> dict:
    """
    Start --refresh-data as a background job.
    Returns the job status dict immediately (status = "running" or already running).
    Dashboard must poll dashboard.jobs.poll_status("refresh_data") for completion.
    """
    from dashboard.jobs import start_job
    return start_job("refresh_data", ["--refresh-data"])


def run_refresh_ai(top: int = 20) -> dict:
    """
    Start --refresh-ai --top N as a background job.
    Returns the job status dict immediately.
    """
    from dashboard.jobs import start_job
    return start_job("refresh_ai", ["--refresh-ai", "--top", str(top)])


# ── Fast commands → synchronous (stay as before) ─────────────────────────────

def run_generate_final_picks() -> tuple[bool, str]:
    """Run --final-picks synchronously (fast — reads cached scores)."""
    code, out, err = _run_cmd_sync(["--final-picks"])
    return code == 0, out + ("\n" + err if err else "")


def run_real_money_audit() -> tuple[bool, str]:
    """Run --real-money-audit synchronously (fast — reads cached data)."""
    code, out, err = _run_cmd_sync(["--real-money-audit"])
    return code == 0, out + ("\n" + err if err else "")


def get_final_picks_report() -> dict | None:
    """Load cached final_picks report or run it inline; returns dict or None."""
    cache = Path("data/cache/final_picks_report.json")
    if cache.exists():
        with cache.open() as f:
            return json.load(f)
    return None


def get_walk_forward_result() -> dict | None:
    cache = Path("data/cache/walk_forward_results.json")
    if cache.exists():
        with cache.open() as f:
            return json.load(f)
    return None


def get_factor_monitor_result() -> dict | None:
    cache = Path("data/cache/factor_monitor.json")
    if cache.exists():
        with cache.open() as f:
            return json.load(f)
    return None
