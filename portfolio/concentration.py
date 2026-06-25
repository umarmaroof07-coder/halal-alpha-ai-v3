"""
V6 Phase 5 — Industry and Sector Concentration Control

Enforces portfolio-level diversification constraints:
  - Max 2 stocks per industry
  - Max 35% of portfolio weight per sector
  - Max 25% single position (already enforced by constructor, re-checked here)

Optimization: when constraints are violated, greedily replace the lowest-scoring
violating stock with the next-best compliant candidate from the ranked universe,
preserving the highest composite score within each constraint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

MAX_PER_INDUSTRY   = 2
MAX_SECTOR_WEIGHT  = 0.35   # 35%
MAX_SINGLE_WEIGHT  = 0.25   # 25% — consistent with constructor


@dataclass
class ConcentrationIssue:
    severity:    str      # "WARN" | "BLOCK"
    rule:        str
    message:     str


@dataclass
class ConcentrationResult:
    passed:     bool
    issues:     list[ConcentrationIssue] = field(default_factory=list)
    optimized:  list[str] = field(default_factory=list)  # final ticker list after optimization

    def warnings(self) -> list[ConcentrationIssue]:
        return [i for i in self.issues if i.severity == "WARN"]

    def blocks(self) -> list[ConcentrationIssue]:
        return [i for i in self.issues if i.severity == "BLOCK"]


def _sector_weights(tickers: list[str], weight_map: dict[str, float],
                    sector_map: dict[str, str]) -> dict[str, float]:
    """Compute total weight allocated to each sector."""
    sw: dict[str, float] = {}
    for t in tickers:
        s = sector_map.get(t, "Unknown")
        sw[s] = sw.get(s, 0.0) + weight_map.get(t, 0.0)
    return sw


def check_concentration(
    portfolio: list[str],
    weight_map: dict[str, float],    # ticker → weight (0-1)
    sector_map: dict[str, str],      # ticker → sector name
    industry_map: dict[str, str],    # ticker → industry name
) -> ConcentrationResult:
    """
    Check a portfolio for concentration violations.
    Does not optimize — call optimize_for_concentration() for that.
    """
    issues: list[ConcentrationIssue] = []

    # Industry concentration
    industry_counts: dict[str, list[str]] = {}
    for t in portfolio:
        ind = industry_map.get(t, "Unknown")
        industry_counts.setdefault(ind, []).append(t)

    for ind, tickers in industry_counts.items():
        if len(tickers) > MAX_PER_INDUSTRY:
            issues.append(ConcentrationIssue(
                severity = "WARN",
                rule     = "industry",
                message  = (
                    f"Industry '{ind}' has {len(tickers)} stocks "
                    f"({', '.join(tickers)}) — max is {MAX_PER_INDUSTRY}"
                ),
            ))

    # Sector weight
    sw = _sector_weights(portfolio, weight_map, sector_map)
    for sector, w in sw.items():
        if w > MAX_SECTOR_WEIGHT:
            issues.append(ConcentrationIssue(
                severity = "WARN",
                rule     = "sector",
                message  = (
                    f"Sector '{sector}' weight {w*100:.1f}% "
                    f"exceeds {MAX_SECTOR_WEIGHT*100:.0f}% limit"
                ),
            ))

    # Single position
    for t, w in weight_map.items():
        if t in portfolio and w > MAX_SINGLE_WEIGHT + 0.001:
            issues.append(ConcentrationIssue(
                severity = "WARN",
                rule     = "single_position",
                message  = (
                    f"{t} weight {w*100:.1f}% exceeds "
                    f"{MAX_SINGLE_WEIGHT*100:.0f}% single-position cap"
                ),
            ))

    return ConcentrationResult(
        passed    = len([i for i in issues if i.severity == "BLOCK"]) == 0,
        issues    = issues,
        optimized = list(portfolio),
    )


def optimize_for_concentration(
    ranked_universe: list[dict],   # sorted by composite desc; each has ticker/sector/industry
    target_n: int = 5,
    weight_map: dict[str, float] | None = None,
    sector_map: dict[str, str]   | None = None,
    industry_map: dict[str, str] | None = None,
) -> tuple[list[str], list[ConcentrationIssue]]:
    """
    Greedy selection: pick up to target_n stocks from ranked_universe while
    respecting industry (≤2) and sector (≤35%) constraints.

    Returns (selected_tickers, violated_issues).
    """
    if sector_map is None:
        sector_map = {e["ticker"]: e.get("sector", "Unknown") for e in ranked_universe}
    if industry_map is None:
        industry_map = {e["ticker"]: e.get("industry", "Unknown") for e in ranked_universe}

    selected: list[str] = []
    industry_counts: dict[str, int] = {}
    sector_weights: dict[str, float] = {}
    violations: list[ConcentrationIssue] = []

    equal_weight = 1.0 / target_n if target_n > 0 else 0.20

    for entry in ranked_universe:
        if len(selected) >= target_n:
            break
        t   = entry["ticker"]
        ind = industry_map.get(t, "Unknown")
        sec = sector_map.get(t, "Unknown")

        # Check industry constraint
        if industry_counts.get(ind, 0) >= MAX_PER_INDUSTRY:
            log.debug("Skip %s — industry '%s' already has %d stocks", t, ind, MAX_PER_INDUSTRY)
            continue

        # Check sector weight constraint
        projected_sector_w = sector_weights.get(sec, 0.0) + equal_weight
        if projected_sector_w > MAX_SECTOR_WEIGHT + 0.001:
            log.debug("Skip %s — sector '%s' would reach %.0f%%",
                      t, sec, projected_sector_w * 100)
            violations.append(ConcentrationIssue(
                severity = "WARN",
                rule     = "sector",
                message  = f"Skipped {t} — sector '{sec}' would exceed {MAX_SECTOR_WEIGHT*100:.0f}%",
            ))
            continue

        selected.append(t)
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        sector_weights[sec]  = sector_weights.get(sec, 0.0) + equal_weight

    return selected, violations
