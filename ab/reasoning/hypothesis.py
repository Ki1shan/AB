"""Hypotheses (Phase 4): testable guesses, never facts.

A Hypothesis is a claim the brain does NOT know yet: it proposes a
possible explanation or conclusion plus the test that would settle it
(a research goal, a planned observation, or a predicted effect to
check). Hypotheses never enter LTM and never drive an action — they
only suggest what to check next. They are falsifiable by design:
``evaluate_hypothesis`` (in engine.py) corroborates or falsifies them
against an observation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..core.vocab import normalize_relation, normalize_text


def make_hypothesis_id(subject, relation, value) -> str:
    """Deterministic id: ``hyp_`` + first 8 hex chars of the SHA-256 of
    the normalized (subject, relation, value) triple. The same claim
    always yields the same id, in any process (no randomised hashing,
    no time-based ids)."""
    payload = "|".join((normalize_text(subject),
                        normalize_relation(relation),
                        normalize_text(value)))
    return "hyp_" + hashlib.sha256(payload.encode()).hexdigest()[:8]


@dataclass
class Hypothesis:
    id: str
    subject: str
    relation: str
    value: str
    confidence: float = 0.0
    source: str = ""                       # "abduction" | "analogy" | ...
    evidence_for: list = field(default_factory=list)
    evidence_against: list = field(default_factory=list)
    status: str = "open"                   # open | corroborated | falsified | disputed
    test: str | None = None                # what would settle it
