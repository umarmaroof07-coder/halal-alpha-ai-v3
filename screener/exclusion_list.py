"""
Shariah exclusion list — industry/activity keyword blocklist.

Key design rules:
  - normalize_industry() strips all punctuation variants so that
    "Banks—Regional", "Banks - Regional", "Banks-Regional", "BANKS - REGIONAL"
    all reduce to "banks regional" before matching.
  - Keyword matches are substring matches on the normalized string.
  - This screen can NEVER be overridden by a manual CSV.
    Only ratio screens may be overridden.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Haram industry / activity keywords
# Matched case-insensitively against the normalized industry string.
# Any single match = excluded. Add new keywords here only.
# ---------------------------------------------------------------------------

HARAM_KEYWORDS: tuple[str, ...] = (
    # Banking / interest-based lending
    "bank",
    "banking",
    "savings institution",
    "credit union",
    "thrift",
    "mortgage reit",
    "payday",
    "pawnshop",
    "moneylender",
    # Insurance (conventional)
    "insurance",
    "reinsurance",
    "surety",
    "title insurance",
    # Gambling
    "casino",
    "gambling",
    "lottery",
    "racing",
    "betting",
    "sportsbook",
    "wagering",
    "gaming" ,          # catches "gaming and wagering"; safe-listed below
    # Alcohol
    "alcohol",
    "brewery",
    "brewing",
    "distillery",
    "distilling",
    "distiller",
    "vintner",
    "winery",
    "winemaking",
    "spirits",
    "beer",
    "wine",
    "liquor",
    "malt beverage",
    # Tobacco
    "tobacco",
    "cigarette",
    "cigar",
    "vaping",
    "e-cigarette",
    "smokeless tobacco",
    # Adult entertainment
    "adult entertainment",
    "pornography",
    "pornographic",
    "escort service",
    "strip club",
    # Pork
    "pork processing",
    "swine",
    "hog farm",
    # Weapons (conventional / indiscriminate)
    "landmine",
    "cluster munition",
    "biological weapon",
    "chemical weapon",
    # Cannabis
    "cannabis",
    "marijuana",
    "marihuana",
    "thc dispensary",
    # SPACs / blank-check companies
    "blank check",
    "special purpose acquisition",
    "spac",
)

# Keywords that contain a haram substring but are themselves NOT haram.
# e.g. "video gaming" (software/entertainment) should not match "gaming".
SAFE_LIST: tuple[str, ...] = (
    "video gaming",
    "cloud gaming",
    "gaming software",
    "gaming technology",
    "gaming hardware",
    "esports",
)


def normalize_industry(raw: str) -> str:
    """
    Normalize an industry or sector string for keyword matching.

    Transformations applied (in order):
      1. Lowercase
      2. Replace em-dash (—), en-dash (–), hyphen (-) with a space
      3. Remove all remaining non-alphanumeric characters except spaces
      4. Collapse multiple spaces into one
      5. Strip leading/trailing whitespace

    Examples:
      "Banks—Regional"      → "banks regional"
      "Banks - Regional"    → "banks regional"
      "Banks-Regional"      → "banks regional"
      "BANKS - REGIONAL"    → "banks regional"
      "Real Estate (REITs)" → "real estate reits"
    """
    if not raw:
        return ""
    s = raw.lower()
    s = s.replace("—", " ").replace("–", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r" +", " ", s).strip()
    return s


def is_haram_industry(industry: str, sector: str = "") -> bool:
    """
    Return True if the normalized industry or sector string matches any
    haram keyword and is not in the safe list.

    Both *industry* and *sector* are checked — a match on either excludes
    the ticker.
    """
    norm_industry = normalize_industry(industry)
    norm_sector   = normalize_industry(sector)
    combined      = f"{norm_industry} {norm_sector}".strip()

    # Check safe list first — if any safe phrase is present, skip haram check
    # for keywords that would otherwise false-positive (e.g. "gaming")
    for safe_phrase in SAFE_LIST:
        if safe_phrase in combined:
            # Remove safe phrase from combined before checking haram keywords
            combined = combined.replace(safe_phrase, "")

    for keyword in HARAM_KEYWORDS:
        norm_kw = normalize_industry(keyword)   # normalize the keyword too
        if norm_kw and norm_kw in combined:
            return True

    return False


def is_spac(company_name: str, industry: str = "") -> bool:
    """
    Secondary SPAC check on company name in addition to industry keyword.
    Many SPACs don't declare "blank check" as their industry.
    """
    norm_name     = normalize_industry(company_name)
    norm_industry = normalize_industry(industry)
    spac_signals  = ("acquisition corp", "acquisition co", "blank check", "spac")
    for sig in spac_signals:
        if sig in norm_name or sig in norm_industry:
            return True
    return False
