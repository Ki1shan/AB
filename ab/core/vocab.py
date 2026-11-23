"""Vocabulary normalization: the single source of truth for how the brain
spells things. Every tag builder, keyword extractor, subject/relation
normalizer and inference matcher must use these functions, so the same
word counts the same way everywhere.

Phase 1 (audit A: I2/I6/I7/G19). Before this module, three different
regexes built tags/keywords and LTM treated ``capital_of`` and
``capital_of_country`` as different relations, so a ``capital_of`` conflict
was never flagged as functional.
"""
from __future__ import annotations

import re
from functools import lru_cache

WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_MAX_TOKENS = 12

# Only exact, explicitly listed variants. Do NOT add fuzzy or stemming
# merges here: silently merging semantically different relations would
# break the honest-by-design rule (see audit consolidated #4).
RELATION_ALIASES = {
    "capital_of_country": "capital_of",
    "capital city": "capital_of",
    "capital": "capital_of",
}

FUNCTIONAL_RELATIONS = ("boils_at", "melts_at", "capital_of")


@lru_cache(maxsize=8192)
def _normalize_text_str(s: str) -> str:
    """Memoized core of :func:`normalize_text` (Phase 12, P12-O2).

    The function is pure — same string in, same string out — and it
    is called tens of thousands of times per run on a small set of
    recurring strings (state keys, subjects, relations), so caching
    the result is behaviour-identical. The LRU bound keeps memory
    flat even on long soaks with many distinct strings."""
    s = " ".join(s.lower().split())
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s.strip(" .,;:")


def normalize_text(s) -> str:
    """Lowercase, collapse whitespace, strip a leading article and trailing
    punctuation. ``"  The Cat. "`` -> ``"cat"``; ``"a"`` stays ``"a"``."""
    if type(s) is str:
        return _normalize_text_str(s)
    return _normalize_text_str(str(s))


def tokenize(text) -> set[str]:
    """Word tokens of a text: ``[a-z0-9_]{3,}``, at most 12, deterministic."""
    words = set(WORD_RE.findall(str(text).lower()))
    if len(words) > _MAX_TOKENS:
        words = set(sorted(words)[:_MAX_TOKENS])
    return words


def normalize_relation(r) -> str:
    """Canonical relation name: normalize, then apply the alias map."""
    s = normalize_text(r)
    return RELATION_ALIASES.get(s, s)
