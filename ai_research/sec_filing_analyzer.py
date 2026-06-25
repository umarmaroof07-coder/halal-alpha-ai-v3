"""
SEC Filing Analyzer — scores a 10-K or 10-Q filing text using Claude.

Looks for:
  - Revenue/margin trend language
  - Debt and liquidity commentary
  - Risk factor disclosures (going concern, litigation, regulatory)
  - Related-party transactions
  - Management discussion tone

If filing_text is None or empty → neutral 50, confidence 0.0.
If ANTHROPIC_API_KEY is missing → neutral 50, confidence 0.0.
Results cached by ticker + as_of_date for AI_CACHE_TTL_FILINGS_DAYS days.
"""

from __future__ import annotations

import logging

from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, AI_CACHE_TTL_FILINGS_DAYS
from ai_research._base import (
    AIAnalysisResult,
    neutral,
    cache_get,
    cache_set,
    _call_claude,
    parse_claude_json,
)

log = logging.getLogger(__name__)

_ANALYZER = "sec_filing"
_TTL = AI_CACHE_TTL_FILINGS_DAYS

_PROMPT_TEMPLATE = """\
You are a senior fundamental analyst reviewing a SEC filing excerpt for {ticker}.
Your job is RISK IDENTIFICATION, not cheerleading. Surface concerns before celebrating strengths.
Analyze ONLY the text provided below. Do not use any prior knowledge about this company.

Assess these dimensions:
1. Revenue and margin trajectory (growing, stable, declining — note any acceleration or deceleration)
2. Debt levels, liquidity risk, and covenant language
3. Material risk factors (going concern, litigation, regulatory, customer concentration)
4. Accounting quality (revenue recognition changes, non-GAAP adjustments, accrual trends, unusual items)
5. Related-party transactions or disclosure transparency concerns

Return ONLY valid JSON in this exact format — no prose before or after:
{{
  "score": <float 0-100, where 50=neutral, >50=positive signals, <50=warning signals>,
  "confidence": <float 0-1, where 0=insufficient text, 1=very confident in assessment>,
  "rationale": "<2-3 sentence summary of the most important findings>",
  "red_flags": ["<factual concern from the text>"],
  "positives": ["<factual strength from the text>"],
  "accounting_concerns": ["<specific accounting risk — revenue recognition, non-GAAP, accruals, etc.>"],
  "thesis_risks": ["<risk that could derail the investment thesis — competitive, regulatory, demand>"],
  "thesis_breakers": ["<single fact from the text that would make you sell immediately, if any>"]
}}

Rules:
- accounting_concerns should be specific accounting or financial reporting risks only.
- thesis_risks are structural or macro risks disclosed in the filing.
- thesis_breakers are one-line facts that are disqualifying (e.g. "going concern doubt raised by auditors").
- If no concerns exist in a category, return an empty list [].
- If text is insufficient, set confidence below 0.30 and explain in rationale.
- Do NOT hallucinate. Every item must be supported by text.

--- FILING TEXT START ---
{filing_text}
--- FILING TEXT END ---
"""

_MAX_TEXT_CHARS = 12_000  # ~3000 tokens — keeps cost predictable


def analyze_sec_filing(
    ticker: str,
    filing_text: str | None,
    as_of_date: str,
) -> AIAnalysisResult:
    """
    Score a SEC filing for *ticker*.

    Parameters
    ----------
    ticker : str
    filing_text : str | None
        Raw text of the 10-K or 10-Q. Pass None if unavailable.
    as_of_date : str
        ISO date of the filing (e.g. "2024-03-31"). Used as cache key.

    Returns
    -------
    AIAnalysisResult with source="sec_filing".
    """
    if not filing_text or not filing_text.strip():
        return neutral(ticker, _ANALYZER, as_of_date)

    if not ANTHROPIC_API_KEY:
        log.debug("ANTHROPIC_API_KEY not set — returning neutral for %s sec_filing", ticker)
        return neutral(ticker, _ANALYZER, as_of_date, reason="API key not configured.")

    cached = cache_get(ticker, _ANALYZER, as_of_date)
    if cached:
        return cached

    truncated = filing_text[:_MAX_TEXT_CHARS]
    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, filing_text=truncated)

    raw = _call_claude(prompt, ANTHROPIC_MODEL, ANTHROPIC_API_KEY)
    result = parse_claude_json(raw, ticker, _ANALYZER, as_of_date)

    if result is None:
        result = neutral(ticker, _ANALYZER, as_of_date, reason="Claude response could not be parsed.")

    cache_set(result, _ANALYZER, _TTL)
    return result
