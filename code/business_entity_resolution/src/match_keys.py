"""Exact proposal keys that survive transliteration, OCR-style digit swaps, joined names, and street numbers.

An English name written in an Indian script (for example "इंटरनेशनल मैनेजमेंट") and its
Latin form collide on a coarse consonant skeleton after AnyAscii transliteration.
These keys are generic character rules; they use no external business data.
"""

from __future__ import annotations

import re
from functools import lru_cache

from anyascii import anyascii

KEY_NAMES = ("key_phonetic", "key_ocr", "key_compact", "key_numbers")

_LEGAL = frozenset({"private", "limited", "pvt", "ltd", "company", "corporation", "incorporated", "llc", "llp",
                    "the", "and", "inc", "co", "corp", "pra", "li", "praivet", "privet", "limitd"})
_WEB = frozenset({"www", "com", "net", "org", "in"})
# Transliterated legal words that survive as distinctive skeletons ("praivet limited").
_LEGAL_CODES = frozenset({"prvt", "lntd", "lnt"})
_OCR = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "l": "i"})
_SPLIT = re.compile(r"[^0-9a-z]+")
_NUMBER = re.compile(r"\d+[a-z]*")
_ORDINAL = re.compile(r"(st|nd|rd|th)$")
_DIGRAPHS = (("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("jh", "j"))


@lru_cache(maxsize=1 << 20)
def phon(token: str) -> str:
    """Coarse consonant skeleton; vowels and aspiration vary most between scripts."""
    t = re.sub(r"(tion|sion)", "shn", token.lower())
    t = t.replace("kp", "f").replace("ph", "f").replace("ck", "k").replace("q", "k").replace("x", "ks")
    t = re.sub(r"c(?=[eiy])", "s", t).replace("c", "k")
    t = re.sub(r"g(?=[eiy])", "j", t)
    for digraph, replacement in _DIGRAPHS:
        t = t.replace(digraph, replacement)
    t = t.replace("w", "v").replace("z", "j").replace("m", "n")
    if not t:
        return t
    t = t[0] + re.sub(r"[aeiouyh]", "", t[1:])
    t = re.sub(r"(.)\1+", r"\1", t)
    return re.sub(r"^[aeiouy]", "", t) or t


@lru_cache(maxsize=1 << 20)
def _codes(word: str) -> tuple[str, str]:
    plain = phon(word)
    return plain, (phon(word.translate(_OCR)) if any(c.isalpha() for c in word) else plain)


def keys(raw_name: str, raw_address: str) -> tuple[str, str, str, str]:
    """Return (phonetic, OCR-phonetic, compact-with-digits, numbers+first-code) keys; empty means none."""
    words = _SPLIT.sub(" ", anyascii(raw_name).lower()).split()
    plain: set[str] = set()
    ocr: set[str] = set()
    for word in words:
        if len(word) < 3 or word.isdigit() or word in _LEGAL or word in _WEB:
            continue
        p, o = _codes(word)
        if p:
            plain.add(p)
        if o:
            ocr.add(o)
    plain -= _LEGAL_CODES
    ocr -= _LEGAL_CODES
    compact = "".join(word for word in words if word not in _LEGAL and word not in _WEB)
    numbers = sorted({n for n in (_ORDINAL.sub("", t) for t in _NUMBER.findall(anyascii(raw_address).lower()))
                      if len(n) >= 2})
    first = min(plain) if plain else ""
    return (" ".join(sorted(plain)),
            " ".join(sorted(ocr)),
            compact if len(compact) >= 6 else "",
            (" ".join(numbers[:2]) + "|" + first) if numbers and first else "")
