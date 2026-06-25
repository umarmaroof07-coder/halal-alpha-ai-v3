"""
V6 Phase 1 — Walk-Forward Report

Prints a formatted report of the true walk-forward backtest results.
Reads from data/cache/walk_forward_results.json (written by analysis/walk_forward.py).
"""

from __future__ import annotations

import json
from pathlib import Path

_CACHE_FILE = Path(__file__).parent.parent / "data" / "cache" / "walk_forward_results.json"
_DIV = "─" * 72


def print_walk_forward_report(result=None) -> None:
    """
    Print the walk-forward report to stdout.
    Accepts either a WalkForwardResult object or reads from cache file.
    """
    if result is None:
        if not _CACHE_FILE.exists():
            print("\n  No walk-forward results found.")
            print("  Run --refresh-data multiple times across quarters to accumulate")
            print("  point-in-time snapshot history, then re-run --walk-forward.\n")
            return
        with _CACHE_FILE.open() as f:
            data = json.load(f)
    else:
        data = {
            "generated_at":     result.generated_at,
            "n_snapshots_used": result.n_snapshots_used,
            "snapshot_dates":   result.snapshot_dates,
            "warnings":         result.warnings,
            "metrics":          result.metrics.to_dict()     if result.metrics     else None,
            "spy_metrics":      result.spy_metrics.to_dict() if result.spy_metrics else None,
            "qqq_metrics":      result.qqq_metrics.to_dict() if result.qqq_metrics else None,
            "periods": [
                {
                    "start_date": p.start_date,
                    "end_date":   p.end_date,
                    "portfolio":  p.portfolio,
                    "port_ret":   round(p.port_ret * 100, 2),
                    "spy_ret":    round(p.spy_ret  * 100, 2),
                    "qqq_ret":    round(p.qqq_ret  * 100, 2),
                    "turnover":   round(p.turnover * 100, 1),
                }
                for p in result.periods
            ],
        }

    _print_header(data)
    _print_warnings(data)
    _print_snapshot_status(data)
    _print_metrics(data)
    _print_periods(data)


def _print_header(d: dict) -> None:
    print("\n" + "═" * 72)
    print("  HALAL ALPHA AI V6 — TRUE WALK-FORWARD BACKTEST")
    print("  Point-in-time • Quarterly rebalance • No look-ahead bias")
    print(f"  Generated: {d.get('generated_at', '')}")
    print("═" * 72)
    print()
    print("  CRITICAL: This backtest uses ONLY information available on each")
    print("  rebalance date. Factor scores are taken from dated snapshots in")
    print("  data/history/. No future fundamentals, prices, or AI scores are used.")
    print()
    print("  Remaining limitations:")
    print("  1. SURVIVORSHIP BIAS: Only tickers alive TODAY are in the universe.")
    print("  2. DATA SPARSITY: Meaningful results require 8+ quarterly periods")
    print("     (2+ years of --refresh-data snapshots).")


def _print_warnings(d: dict) -> None:
    warnings = d.get("warnings", [])
    if not warnings:
        return
    print()
    print(f"  ⚠  WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠  {w}")


def _print_snapshot_status(d: dict) -> None:
    dates = d.get("snapshot_dates", [])
    n     = d.get("n_snapshots_used", 0)
    print()
    print(f"  Snapshots available:   {len(dates)}")
    print(f"  Snapshots used:        {n}")
    if dates:
        print(f"  Date range:            {dates[0]} → {dates[-1]}")
    periods = d.get("periods", [])
    print(f"  Backtest periods:      {len(periods)}")


def _print_metrics(d: dict) -> None:
    m   = d.get("metrics")
    spy = d.get("spy_metrics")
    qqq = d.get("qqq_metrics")

    if not m:
        print()
        print(f"  No metrics computed — insufficient periods.")
        print(f"  Accumulate more snapshot history and re-run --walk-forward.")
        return

    print()
    print("  " + _DIV)
    print("  PERFORMANCE SUMMARY")
    print("  " + _DIV)
    hdr = f"  {'Metric':<22}  {'Portfolio':>12}  {'SPY':>10}  {'QQQ':>10}"
    print(hdr)
    print("  " + "─" * 58)

    def _row(label: str, key: str, fmt: str = ".2f", suffix: str = ""):
        pm  = m.get(key)
        sm  = spy.get(key) if spy else None
        qm  = qqq.get(key) if qqq else None
        pv  = f"{pm:{fmt}}{suffix}" if pm is not None else "n/a"
        sv  = f"{sm:{fmt}}{suffix}" if sm is not None else "n/a"
        qv  = f"{qm:{fmt}}{suffix}" if qm is not None else "n/a"
        print(f"  {label:<22}  {pv:>12}  {sv:>10}  {qv:>10}")

    _row("CAGR",              "cagr",              ".2f", "%")
    _row("Sharpe Ratio",      "sharpe",             ".3f")
    _row("Sortino Ratio",     "sortino",            ".3f")
    _row("Calmar Ratio",      "calmar",             ".3f")
    _row("Max Drawdown",      "max_drawdown",       ".2f", "%")
    _row("Win Rate vs SPY",   "win_rate",           ".1f", "%")
    _row("Information Ratio", "information_ratio",  ".3f")
    _row("Alpha (ann.)",      "alpha",              ".2f", "%")
    _row("Beta",              "beta",               ".3f")
    _row("Tracking Error",    "tracking_error",     ".2f", "%")
    _row("Avg Turnover",      "turnover",           ".1f", "%")
    print(f"  {'Periods':<22}  {m.get('n_periods', 'n/a'):>12}")
    print("  " + _DIV)


def _print_periods(d: dict) -> None:
    periods = d.get("periods", [])
    if not periods:
        return

    print()
    print("  " + _DIV)
    print("  PERIOD-BY-PERIOD RETURNS")
    print("  " + _DIV)
    print(f"  {'Period':<24}  {'Portfolio':>10}  {'SPY':>8}  {'QQQ':>8}  {'Active':>8}  {'Turn%':>6}  Portfolio")
    print("  " + "─" * 100)
    for p in periods:
        active   = p.get("port_ret", 0) - p.get("spy_ret", 0)
        act_sym  = "+" if active > 0 else ""
        tickers  = ", ".join(p.get("portfolio", []))
        period   = f"{p.get('start_date','')} → {p.get('end_date','')}"
        print(
            f"  {period:<24}  "
            f"{p.get('port_ret',0):>+9.2f}%  "
            f"{p.get('spy_ret',0):>+7.2f}%  "
            f"{p.get('qqq_ret',0):>+7.2f}%  "
            f"{act_sym}{active:>7.2f}%  "
            f"{p.get('turnover',0):>5.1f}%  "
            f"{tickers}"
        )
    print("  " + _DIV)
    print()
