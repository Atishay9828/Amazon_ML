"""Conservative, source-independent text normalization for retrieval."""

from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii

_TOKEN_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_NAME_VARIANTS = {
    "corp": "corporation",
    "corpn": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "co": "company",
    "intl": "international",
}
_ADDRESS_VARIANTS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "apt": "apartment",
    "ste": "suite",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}
_GENERIC_NAME_TOKENS = frozenset(
    {"the", "and", "company", "corporation", "incorporated", "limited", "private", "llc", "llp"}
)
_GENERIC_ADDRESS_TOKENS = frozenset(
    {"road", "street", "avenue", "boulevard", "lane", "drive", "highway", "apartment",
     "suite", "north", "south", "east", "west", "near", "floor", "unit", "city", "state"}
)


def _tokens(value: str) -> list[str]:
    if not value:
        return []
    # Generic Unicode transliteration is a character normalization rule, not
    # an external business-identity lookup. It helps cross-script name variants.
    value = anyascii(unicodedata.normalize("NFKC", value)).casefold().replace("&", " and ")
    return [token for token in _TOKEN_RE.sub(" ", value).split() if token]


def normalize_name(value: str) -> str:
    """Normalize without deleting legal suffixes or meaningful digits."""
    return " ".join(_NAME_VARIANTS.get(token, token) for token in _tokens(value))


def normalize_address(value: str) -> str:
    return " ".join(_ADDRESS_VARIANTS.get(token, token) for token in _tokens(value))


def informative_name_tokens(normalized_name: str) -> tuple[str, ...]:
    """Return unique tokens eligible for the rare-token retrieval channel."""
    return tuple(sorted({t for t in normalized_name.split() if len(t) >= 3 and t not in _GENERIC_NAME_TOKENS}))


def name_token_signature(normalized_name: str) -> str:
    """Order-independent exact key for at least two informative name words."""
    tokens = informative_name_tokens(normalized_name)
    return " ".join(tokens) if len(tokens) >= 2 else ""


def compact_name(normalized_name: str) -> str:
    """Join informative words to compare a spaced name with a website-style name."""
    words = normalized_name.split()
    if words and words[0] == "www":
        words = words[1:]
    if words and words[-1] in {"com", "net", "org", "in"}:
        words = words[:-1]
    return "".join(word for word in words if len(word) >= 3 and word not in _GENERIC_NAME_TOKENS)


def informative_address_tokens(normalized_address: str) -> tuple[str, ...]:
    """Keep distinctive words and multi-digit building/PIN numbers."""
    return tuple(sorted({t for t in normalized_address.split()
                         if len(t) >= 3 and t not in _GENERIC_ADDRESS_TOKENS}))
