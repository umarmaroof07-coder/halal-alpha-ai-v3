"""
Moat Scorer — scores a company's competitive moat using Claude.

Assesses from a business description / profile text:
  - Pricing power (can they raise prices without losing customers?)
  - Switching costs (how hard is it for customers to leave?)
  - Network effects (does the product get better as more people use it?)
  - Cost advantages (scale, proprietary processes, geography)
  - Intangible assets (brand strength, patents, regulatory licenses)

If company_profile is None or empty → neutral 50, confidence 0.0.
If ANTHROPIC_API_KEY is missing → neutral 50, confidence 0.0.
Results cached by ticker + as_of_date for AI_CACHE_TTL_FILINGS_DAYS days
(moat changes slowly — same TTL as filings).
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

_ANALYZER = "moat"
_TTL = AI_CACHE_TTL_FILINGS_DAYS

_PROMPT_TEMPLATE = """\
You are a fundamental equity analyst assessing the competitive moat durability of {ticker}.
Your primary job is to identify moat WEAKNESSES and threats, not to confirm existing strengths.
Analyze ONLY the text provided below. Do not use any prior knowledge.

Evaluate the presence (or absence) of each moat source from what the text says:
1. Pricing power — can they raise prices without losing customers?
2. Switching costs — how hard is it for customers to leave?
3. Network effects — does value grow as more users join?
4. Cost advantages — scale, proprietary process, or geographic lock-in?
5. Intangible assets — brand, patents, regulatory approvals?

For each moat source that exists, assess its DURABILITY and THREATS:
- Is it eroding due to new entrants, technology disruption, or regulation?
- Are switching costs declining (e.g. due to standardisation or open-source alternatives)?
- Is pricing power under pressure from commoditization?

Score 0–100 where:
  0–30 = no meaningful moat (commodity, highly competitive)
  31–50 = weak or narrow moat, erosion risk
  51–70 = moderate moat, defensible near-term
  71–100 = wide, durable moat with high confidence

Return ONLY valid JSON in this exact format — no prose before or after:
{{
  "score": <float 0-100>,
  "confidence": <float 0-1, where 0=text too vague to assess, 1=clear evidence>,
  "rationale": "<2-3 sentence summary of moat sources present and their durability>",
  "red_flags": ["<factual moat weakness from the text>"],
  "positives": ["<factual moat strength from the text>"],
  "moat_concerns": ["<specific threat to moat durability: disruption, competition, regulation>"],
  "thesis_breakers": ["<single fact that would indicate the moat has already been breached, if any>"]
}}

Rules:
- moat_concerns are forward-looking erosion risks, not just current weaknesses.
- thesis_breakers are severe facts (e.g. "core customer contracts now on 1-year terms with no renewal guarantee").
- If no concerns exist in a category, return an empty list [].
- If text is insufficient, set confidence below 0.30.
- Do NOT hallucinate. Every item must come from the text.

--- COMPANY PROFILE TEXT START ---
{profile_text}
--- COMPANY PROFILE TEXT END ---
"""

_MAX_TEXT_CHARS = 8_000


def score_moat(
    ticker: str,
    company_profile: str | None,
    as_of_date: str,
) -> AIAnalysisResult:
    """
    Score the competitive moat for *ticker*.

    Parameters
    ----------
    ticker : str
    company_profile : str | None
        Business description, typically from FMP profile or yfinance longBusinessSummary.
        Pass None if unavailable.
    as_of_date : str
        ISO date (e.g. "2024-12-31"). Used as cache key.

    Returns
    -------
    AIAnalysisResult with source="moat".
    """
    if not company_profile or not company_profile.strip():
        return neutral(ticker, _ANALYZER, as_of_date)

    if not ANTHROPIC_API_KEY:
        log.debug("ANTHROPIC_API_KEY not set — returning neutral for %s moat", ticker)
        return neutral(ticker, _ANALYZER, as_of_date, reason="API key not configured.")

    cached = cache_get(ticker, _ANALYZER, as_of_date)
    if cached:
        return cached

    truncated = company_profile[:_MAX_TEXT_CHARS]
    prompt = _PROMPT_TEMPLATE.format(ticker=ticker, profile_text=truncated)

    raw = _call_claude(prompt, ANTHROPIC_MODEL, ANTHROPIC_API_KEY)
    result = parse_claude_json(raw, ticker, _ANALYZER, as_of_date)

    if result is None:
        result = neutral(ticker, _ANALYZER, as_of_date, reason="Claude response could not be parsed.")

    cache_set(result, _ANALYZER, _TTL)
    return result
