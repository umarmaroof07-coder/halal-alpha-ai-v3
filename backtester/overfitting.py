"""
Overfitting checks — sub-period analysis and concentration metrics.

Periods:
  In-sample:   2005–2015
  Out-sample:  2016–2026
  Stress 2008: Jan 2008 – Dec 2008
  Stress 2020: Jan 2020 – Dec 2020
  Stress 2022: Jan 2022 – Dec 2022

Concentration analysis (per rebalance):
  - Average number of positions
  - Average Herfindahl-Hirschman Index: sum(weight_i²)
    HHI = 1.0 means fully concentrated; 0.2 = perfectly equal 5-stock portfolio
  - Average weight of rank-1 position
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from backtester.metrics import compute_metrics, BacktestMetrics, _cagr, _max_drawdown, _sharpe


@dataclass
class PeriodMetrics:
    period: str
    start: str
    end: str
    n_months: int
    portfolio_cagr: float
    spy_cagr: float
    portfolio_sharpe: float
    portfolio_max_drawdown: float
    spy_max_drawdown: float
    win_rate: float

    def to_dict(self) -> dict:
        pct = lambda x: round(x * 100, 2)
        return {
            "period":                  self.period,
            "start":                   self.start,
            "end":                     self.end,
            "n_months":                self.n_months,
            "portfolio_cagr":          pct(self.portfolio_cagr),
            "spy_cagr":                pct(self.spy_cagr),
            "portfolio_sharpe":        round(self.portfolio_sharpe, 3),
            "portfolio_max_drawdown":  pct(self.portfolio_max_drawdown),
            "spy_max_drawdown":        pct(self.spy_max_drawdown),
            "win_rate":                pct(self.win_rate),
        }


@dataclass
class ConcentrationAnalysis:
    avg_n_positions: float
    avg_hhi: float              # Herfindahl index avg; 0.2 = equal-weight 5 stocks
    avg_top1_weight: float      # average weight of rank-1 position
    min_positions: int
    max_positions: int

    def to_dict(self) -> dict:
        return {
            "avg_n_positions":  round(self.avg_n_positions, 1),
            "avg_hhi":          round(self.avg_hhi, 4),
            "avg_top1_weight":  round(self.avg_top1_weight * 100, 2),
            "min_positions":    self.min_positions,
            "max_positions":    self.max_positions,
        }


@dataclass
class OverfittingReport:
    in_sample:    PeriodMetrics   # 2005–2015
    out_sample:   PeriodMetrics   # 2016–2026
    crisis_2008:  PeriodMetrics
    crisis_2020:  PeriodMetrics
    crisis_2022:  PeriodMetrics
    performance_decay: float      # out_sample_cagr - in_sample_cagr (negative = decay)
    concentration: ConcentrationAnalysis

    def to_dict(self) -> dict:
        return {
            "in_sample":         self.in_sample.to_dict(),
            "out_sample":        self.out_sample.to_dict(),
            "crisis_2008":       self.crisis_2008.to_dict(),
            "crisis_2020":       self.crisis_2020.to_dict(),
            "crisis_2022":       self.crisis_2022.to_dict(),
            "performance_decay": round(self.performance_decay * 100, 2),
            "concentration":     self.concentration.to_dict(),
        }


def _slice(
    monthly_dates: Sequence[str],
    portfolio_returns: Sequence[float],
    spy_returns: Sequence[float],
    start_year: int,
    end_year: int,
) -> tuple[list[float], list[float]]:
    """Return (portfolio_returns, spy_returns) for months within [start_year, end_year]."""
    port_slice, spy_slice = [], []
    for date, pr, sr in zip(monthly_dates, portfolio_returns, spy_returns):
        yr = int(date[:4])
        if start_year <= yr <= end_year:
            port_slice.append(pr)
            spy_slice.append(sr)
    return port_slice, spy_slice


def _period_metrics(
    label: str,
    start: str,
    end: str,
    port: list[float],
    spy: list[float],
) -> PeriodMetrics:
    n = len(port)
    win_rate = sum(1 for r in port if r > 0) / n if n > 0 else 0.0
    return PeriodMetrics(
        period=label,
        start=start,
        end=end,
        n_months=n,
        portfolio_cagr=_cagr(port),
        spy_cagr=_cagr(spy),
        portfolio_sharpe=_sharpe(port),
        portfolio_max_drawdown=_max_drawdown(port),
        spy_max_drawdown=_max_drawdown(spy),
        win_rate=win_rate,
    )


def compute_concentration(
    monthly_weights: Sequence[list[float]],
) -> ConcentrationAnalysis:
    """
    Compute concentration stats from a sequence of monthly weight lists.

    Each inner list contains the conviction weights held that month
    (e.g. [0.30, 0.25, 0.20, 0.15, 0.10] for a full 5-stock portfolio).
    """
    if not monthly_weights:
        return ConcentrationAnalysis(0, 0, 0, 0, 0)

    n_pos_list = [len(w) for w in monthly_weights]
    hhi_list   = [sum(wi ** 2 for wi in w) for w in monthly_weights if w]
    top1_list  = [max(w) for w in monthly_weights if w]

    return ConcentrationAnalysis(
        avg_n_positions=sum(n_pos_list) / len(n_pos_list),
        avg_hhi=sum(hhi_list) / len(hhi_list) if hhi_list else 0.0,
        avg_top1_weight=sum(top1_list) / len(top1_list) if top1_list else 0.0,
        min_positions=min(n_pos_list),
        max_positions=max(n_pos_list),
    )


def compute_overfitting_report(
    monthly_dates: Sequence[str],      # ISO month strings e.g. "2005-01"
    portfolio_returns: Sequence[float],
    spy_returns: Sequence[float],
    monthly_weights: Sequence[list[float]],
) -> OverfittingReport:
    """
    Compute full overfitting / sub-period analysis.

    Parameters
    ----------
    monthly_dates : Sequence[str]
        "YYYY-MM" strings for each month in chronological order.
    portfolio_returns, spy_returns : Sequence[float]
        Monthly net returns aligned with monthly_dates.
    monthly_weights : Sequence[list[float]]
        Conviction weights held each month (for concentration analysis).
    """
    def _s(sy, ey):
        return _slice(monthly_dates, portfolio_returns, spy_returns, sy, ey)

    p_in, s_in = _s(2005, 2015)
    p_out, s_out = _s(2016, 2026)
    p_08, s_08 = _s(2008, 2008)
    p_20, s_20 = _s(2020, 2020)
    p_22, s_22 = _s(2022, 2022)

    in_sample  = _period_metrics("2005–2015", "2005-01", "2015-12", p_in, s_in)
    out_sample = _period_metrics("2016–2026", "2016-01", "2026-12", p_out, s_out)

    return OverfittingReport(
        in_sample=in_sample,
        out_sample=out_sample,
        crisis_2008=_period_metrics("2008 Crisis", "2008-01", "2008-12", p_08, s_08),
        crisis_2020=_period_metrics("2020 COVID", "2020-01", "2020-12", p_20, s_20),
        crisis_2022=_period_metrics("2022 Bear", "2022-01", "2022-12", p_22, s_22),
        performance_decay=out_sample.portfolio_cagr - in_sample.portfolio_cagr,
        concentration=compute_concentration(monthly_weights),
    )
