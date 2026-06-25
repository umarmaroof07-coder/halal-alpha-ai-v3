from __future__ import annotations

# ---------------------------------------------------------------------------
# Shariah exclusion rules
# Conservative default: unknown = exclude
# ---------------------------------------------------------------------------

# Hard-excluded GICS sectors (exact matches against FMP sectorName field)
EXCLUDED_SECTORS: frozenset[str] = frozenset({
    "Financial Services",   # conventional banks, credit cards, payday lending
    "Insurance",            # conventional insurance
})

# Keyword blocklist — matched case-insensitively against industry / company description
# Any match = excluded
EXCLUDED_INDUSTRY_KEYWORDS: tuple[str, ...] = (
    # Banking / lending
    "bank", "banking", "savings institution", "credit union", "thrift",
    "mortgage reit", "payday", "pawnshop",
    # Insurance
    "insurance", "reinsurance", "surety",
    # Gambling
    "casino", "gambling", "lottery", "racing", "betting", "sportsbook",
    "wagering",
    # Alcohol
    "alcohol", "brewery", "brewing", "distillery", "distilling", "winery",
    "winemaking", "spirits", "beer", "wine", "liquor", "malt beverage",
    # Tobacco
    "tobacco", "cigarette", "cigar", "vaping", "e-cigarette", "smokeless",
    # Adult entertainment
    "adult entertainment", "pornography", "pornographic", "escort",
    "gentlemen's club", "strip club",
    # Pork
    "pork", "swine", "hog farm", "pork processing",
    # Weapons (conventional)
    "landmine", "cluster munition", "biological weapon", "chemical weapon",
    # Cannabis
    "cannabis", "marijuana", "marihuana", "thc dispensary",
)

# ---------------------------------------------------------------------------
# Financial ratio thresholds (AAOIFI-based)
# All ratios computed from FMP financial statements
# Missing or unavailable data → EXCLUDE
# ---------------------------------------------------------------------------

MAX_DEBT_TO_ASSETS: float = 0.33           # Total debt / Total assets
MAX_INTEREST_INCOME_RATIO: float = 0.05    # Interest income / Total revenue
MAX_ACCOUNTS_RECEIVABLE_RATIO: float = 0.45  # Accounts receivable / Total assets

# ---------------------------------------------------------------------------
# Universe constraints
# ---------------------------------------------------------------------------

ALLOWED_EXCHANGES: frozenset[str] = frozenset({
    "NYSE", "NASDAQ", "AMEX", "NYSE ARCA", "NYSE MKT",
})

# Tickers explicitly blocked regardless of screening outcome
# (meme stocks, known SPACs, known non-compliant by manual review)
MANUAL_EXCLUSIONS: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# Conservative defaults — never assume compliant
# ---------------------------------------------------------------------------

UNKNOWN_SECTOR_ACTION = "exclude"       # "exclude" | "include" (always exclude)
MISSING_RATIO_ACTION = "exclude"        # "exclude" | "skip_check"
