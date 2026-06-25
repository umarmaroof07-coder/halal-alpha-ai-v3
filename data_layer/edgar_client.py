"""
SEC EDGAR client — free, no API key required.

Provides:
  get_cik(ticker)              → int CIK or None
  get_filing_text(ticker, form_type)  → cleaned text string or None

Rate limiting: EDGAR allows ~10 req/s. We add a 0.15s sleep between calls.
All results are file-cached in data/cache/edgar/ so each ticker is only
fetched once until the TTL expires.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "HalalAlphaAI research@halalalpha.com"}
_CACHE_DIR = Path("data/cache/edgar")
_TICKER_MAP_PATH = _CACHE_DIR / "ticker_cik_map.json"
_TICKER_MAP_TTL_DAYS = 30   # refresh CIK map monthly
_FILING_TTL_DAYS = 45       # re-fetch filings after 45 days
_MAX_TEXT_CHARS = 80_000    # cap before sending to Claude (will be truncated per-analyzer)
_DELAY = 0.15               # polite crawl delay


# ---------------------------------------------------------------------------
# CIK map
# ---------------------------------------------------------------------------

def _load_ticker_map() -> dict[str, int]:
    """Download or load cached ticker→CIK mapping from EDGAR."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use cached file if fresh
    if _TICKER_MAP_PATH.exists():
        mtime = datetime.fromtimestamp(_TICKER_MAP_PATH.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=_TICKER_MAP_TTL_DAYS):
            with _TICKER_MAP_PATH.open() as f:
                return json.load(f)

    log.info("Downloading EDGAR ticker→CIK map …")
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS, timeout=20,
        )
        r.raise_for_status()
        raw: dict = r.json()
        # raw is {index: {cik_str, ticker, title}}
        mapping = {
            v["ticker"].upper(): int(v["cik_str"])
            for v in raw.values()
            if v.get("ticker") and v.get("cik_str")
        }
        with _TICKER_MAP_PATH.open("w") as f:
            json.dump(mapping, f)
        return mapping
    except Exception as exc:
        log.warning("Could not download EDGAR ticker map: %s", exc)
        return {}


def get_cik(ticker: str) -> int | None:
    """Return the SEC CIK for *ticker*, or None if not found."""
    mapping = _load_ticker_map()
    return mapping.get(ticker.upper().replace("-", "."))


# ---------------------------------------------------------------------------
# Filing text fetcher
# ---------------------------------------------------------------------------

def _filing_cache_path(ticker: str, form_type: str) -> Path:
    return _CACHE_DIR / f"{ticker.upper()}_{form_type.replace('/', '_')}_text.json"


def _load_filing_cache(ticker: str, form_type: str) -> dict | None:
    path = _filing_cache_path(ticker, form_type)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        fetched_at = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
        if datetime.now() - fetched_at > timedelta(days=_FILING_TTL_DAYS):
            return None
        return data
    except Exception:
        return None


def _save_filing_cache(ticker: str, form_type: str, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _filing_cache_path(ticker, form_type)
    data["fetched_at"] = datetime.now().isoformat()
    try:
        with path.open("w") as f:
            json.dump(data, f)
    except Exception as exc:
        log.warning("EDGAR cache write error for %s: %s", ticker, exc)


def _strip_html(html: str) -> str:
    """Strip HTML/XBRL tags and normalize whitespace."""
    text = re.sub(r"<[^>]{0,300}>", " ", html)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_readable_section(text: str) -> str:
    """
    Extract the most useful text section from a 10-K/10-Q.
    Priority: MD&A → Risk Factors → Business Description → full text.
    Returns up to _MAX_TEXT_CHARS characters.
    """
    upper = text.upper()

    # Look for MD&A section
    for marker in ["MANAGEMENT'S DISCUSSION AND ANALYSIS",
                   "MANAGEMENT S DISCUSSION AND ANALYSIS",
                   "ITEM 7.", "ITEM 7 "]:
        idx = upper.find(marker)
        if idx > 0:
            excerpt = text[idx:idx + _MAX_TEXT_CHARS]
            if len(excerpt) > 500:
                return excerpt

    # Fall back to Risk Factors
    for marker in ["RISK FACTORS", "ITEM 1A"]:
        idx = upper.find(marker)
        if idx > 0:
            excerpt = text[idx:idx + _MAX_TEXT_CHARS]
            if len(excerpt) > 500:
                return excerpt

    # Fall back to Business section
    for marker in ["ITEM 1.", "ITEM 1 ", "BUSINESS\n"]:
        idx = upper.find(marker)
        if idx > 0:
            excerpt = text[idx:idx + _MAX_TEXT_CHARS]
            if len(excerpt) > 500:
                return excerpt

    return text[:_MAX_TEXT_CHARS]


def get_filing_text(
    ticker: str,
    form_type: str = "10-K",
) -> tuple[str | None, str]:
    """
    Fetch the most recent SEC filing text for *ticker*.

    Parameters
    ----------
    ticker : str
    form_type : str
        "10-K" or "10-Q"

    Returns
    -------
    (text, as_of_date)  — text is None on failure, as_of_date is "YYYY-MM-DD"
    """
    cached = _load_filing_cache(ticker, form_type)
    if cached:
        return cached.get("text"), cached.get("as_of_date", "")

    cik = get_cik(ticker)
    if not cik:
        log.debug("EDGAR: no CIK for %s", ticker)
        return None, ""

    cik_padded = str(cik).zfill(10)
    time.sleep(_DELAY)

    try:
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r = requests.get(sub_url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("EDGAR submissions fetch failed for %s: %s", ticker, exc)
        return None, ""

    filings = data.get("filings", {}).get("recent", {})
    forms  = filings.get("form", [])
    dates  = filings.get("filingDate", [])
    accs   = filings.get("accessionNumber", [])
    docs   = filings.get("primaryDocument", [])

    # Find most recent matching form
    target_idx = None
    for i, form in enumerate(forms):
        if form == form_type:
            target_idx = i
            break

    if target_idx is None:
        log.debug("EDGAR: no %s found for %s", form_type, ticker)
        _save_filing_cache(ticker, form_type, {"text": None, "as_of_date": ""})
        return None, ""

    acc_clean = accs[target_idx].replace("-", "")
    primary   = docs[target_idx]
    as_of     = dates[target_idx]
    doc_url   = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{primary}"

    time.sleep(_DELAY)
    try:
        r2 = requests.get(doc_url, headers=_HEADERS, timeout=25)
        r2.raise_for_status()
        raw_text = _strip_html(r2.text)
        text = _extract_readable_section(raw_text)
        log.debug("EDGAR: fetched %s %s for %s (%d chars)", form_type, as_of, ticker, len(text))
        _save_filing_cache(ticker, form_type, {"text": text, "as_of_date": as_of})
        return text, as_of
    except Exception as exc:
        log.warning("EDGAR filing fetch failed for %s: %s", ticker, exc)
        _save_filing_cache(ticker, form_type, {"text": None, "as_of_date": ""})
        return None, ""


def get_filing_text_with_fallback(ticker: str) -> tuple[str | None, str]:
    """Try 10-K first, fall back to 10-Q. Returns (text, as_of_date)."""
    text, date = get_filing_text(ticker, "10-K")
    if text and len(text) > 500:
        return text, date
    return get_filing_text(ticker, "10-Q")
