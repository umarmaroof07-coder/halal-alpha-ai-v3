"""
V6 Phase 6 — Portfolio Confidence Weighting (optional mode)

Computes an alternative weight allocation based on:
  w_i ∝ composite_score × model_confidence × risk_score

Bounds: min 10%, max 30%.

Compared against:
  - Current (risk-adjusted inverse-vol) allocation
  - Equal weight
  - Inverse volatility only

This is an ADDITIONAL output — it does NOT replace the existing
portfolio constructor. The final picks still use the existing system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_MIN_W = 0.10   # 10% floor
_MAX_W = 0.30   # 30% ceiling


@dataclass
class ConfidenceWeightResult:
    tickers:              list[str]
    confidence_weights:   list[float]   # proposed weights
    equal_weights:        list[float]
    composite_scores:     list[float]
    model_confidences:    list[float]   # 0-100
    risk_scores:          list[float]   # 0-100

    def to_dict(self) -> dict:
        return {
            "tickers":             self.tickers,
            "confidence_weights":  [round(w, 4) for w in self.confidence_weights],
            "equal_weights":       [round(w, 4) for w in self.equal_weights],
            "composite_scores":    [round(s, 2)  for s in self.composite_scores],
            "model_confidences":   [round(c, 1)  for c in self.model_confidences],
            "risk_scores":         [round(r, 1)  for r in self.risk_scores],
        }


def compute_confidence_weights(
    tickers:           list[str],
    composite_scores:  list[float],
    model_confidences: list[float],   # 0-100
    risk_scores:       list[float],   # 0-100; higher = safer
) -> ConfidenceWeightResult:
    """
    Compute confidence-weighted portfolio allocations.

    Raw weight = composite × (confidence/100) × (risk_score/100)
    Then normalized and bounded to [10%, 30%].
    """
    n = len(tickers)
    if n == 0:
        return ConfidenceWeightResult([], [], [], [], [], [])

    # Raw scores — all inputs on 0-100 scale
    raws = [
        composite_scores[i] * (model_confidences[i] / 100.0) * (risk_scores[i] / 100.0)
        for i in range(n)
    ]

    total = sum(raws) or 1.0
    weights = [r / total for r in raws]

    # Clamp [_MIN_W, _MAX_W] with re-normalization (max 20 iterations)
    for _ in range(20):
        clamped = [max(_MIN_W, min(_MAX_W, w)) for w in weights]
        ct = sum(clamped) or 1.0
        weights = [c / ct for c in clamped]
        if all(_MIN_W - 1e-6 <= w <= _MAX_W + 1e-6 for w in weights):
            break
    # Hard clamp to ensure bounds regardless of convergence
    weights = [max(_MIN_W, min(_MAX_W, w)) for w in weights]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    equal = [1.0 / n] * n

    return ConfidenceWeightResult(
        tickers            = list(tickers),
        confidence_weights = weights,
        equal_weights      = equal,
        composite_scores   = list(composite_scores),
        model_confidences  = list(model_confidences),
        risk_scores        = list(risk_scores),
    )
