"""Long-term memory: explicit, append-only, source-tracked. Nothing is overwritten or guessed."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum

from .vocab import FUNCTIONAL_RELATIONS, normalize_relation, normalize_text

TRUST = {"teacher": 1.0, "user": 1.0, "tool": 0.8, "experience": 0.7, "ocr": 0.5, "web": 0.45,
         # Phase 8 (P8-2): file trust 0.45, same as web. A single
         # document never verifies a fact; recorded as a Phase 11
         # decision to revisit — do not silently raise this.
         "file": 0.45}


def _n(x) -> str:
    """Backwards-compatible name: the single normalizer (see ab/core/vocab.py)."""
    return normalize_text(x)


class AddResult(Enum):
    STORED = "stored"
    CORROBORATED = "corroborated"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


@dataclass
class Fact:
    subject: str
    relation: str
    value: str
    sources: dict = field(default_factory=dict)      # source_id -> trust
    status: str = "unverified"                        # verified|unverified|disputed|retracted
    id: int = 0
    added_cycle: int = 0
    history: list = field(default_factory=list)

    @property
    def confidence(self) -> float:
        p = 1.0
        for t in self.sources.values():
            p *= (1.0 - t)
        return round(1.0 - p, 4)

    def key(self) -> tuple:
        return (self.subject, self.relation, self.value)


@dataclass
class Answer:
    known: bool
    answers: list          # [(value, confidence, status, sources)]
    message: str

    @property
    def best(self):
        return self.answers[0] if self.answers else None


class LongTermMemory:
    def __init__(self, functional_relations=None,
                 trust: dict | None = None):
        # Phase 1: functional list is ("boils_at", "melts_at", "capital_of").
        # The v0.1 default used "capital_of_country", so a capital_of conflict
        # was never flagged functional (audit A I6). Relations are canonicalized
        # so an explicitly passed "capital_of_country" still maps to "capital_of".
        self.functional = {normalize_relation(r)
                           for r in (functional_relations or FUNCTIONAL_RELATIONS)}
        self.trust = dict(TRUST if trust is None else trust)
        self._facts: list[Fact] = []
        # Phase 12 (P12-O5): mutation counter for Brain.world's
        # version-gated cache. Bumped on every add/resolve/retract/
        # load (runtime-only, never persisted); a fresh object
        # starts at 0.
        self.version = 0

    # ---- writing (append-only) -------------------------------------------------
    def _trust_of(self, source_id: str) -> float:
        return self.trust.get(source_id.split(":", 1)[0], 0.3)

    def _refresh_status(self, f: Fact) -> None:
        if f.status in ("retracted", "disputed"):
            return
        trusted = any(t >= 1.0 for t in f.sources.values())
        f.status = "verified" if trusted or len(f.sources) >= 2 else "unverified"

    def add(self, subject, relation, value, source_id: str = "teacher", cycle: int = 0):
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        s, r, v = _n(subject), _n(relation), _n(value)
        live = [f for f in self._facts if f.status != "retracted"]
        for f in live:
            if f.key() == (s, r, v):
                if source_id in f.sources:
                    return AddResult.DUPLICATE, f
                f.sources[source_id] = self._trust_of(source_id)
                f.history.append(f"cycle {cycle}: corroborated by {source_id}")
                self._refresh_status(f)
                return AddResult.CORROBORATED, f
        new = Fact(s, r, v, {source_id: self._trust_of(source_id)}, id=len(self._facts) + 1,
                   added_cycle=cycle, history=[f"cycle {cycle}: added from {source_id}"])
        rivals = [f for f in live if (f.subject, f.relation) == (s, r) and f.value != v]
        self._facts.append(new)
        if r in self.functional and rivals:
            for f in rivals:
                if not any(t >= 1.0 for t in f.sources.values()) or any(
                        t >= 1.0 for t in new.sources.values()):
                    f.status = "disputed"
                    f.history.append(f"cycle {cycle}: disputed by new claim '{v}' from {source_id}")
            new.status = "disputed"
            new.history.append(f"cycle {cycle}: conflicts with existing value(s)")
            return AddResult.CONFLICT, new
        self._refresh_status(new)
        return AddResult.STORED, new

    def approve_claim(self, subject, relation, obj, source: str = "user") -> str:
        """Explicit human approval of a claim as a fact (Phase 3 §3.2).

        Adds the fact with the given source (default ``"user"``, trust 1.0
        via TRUST) and returns its status (``"verified"`` when the source
        is fully trusted and there is no functional conflict; otherwise
        the status the add produced, e.g. ``"disputed"``). This is the
        ONLY explicit-approval entry point: no automatic path in the
        codebase may call it — research and OCR never do.
        """
        _, fact = self.add(subject, relation, obj, source_id=source)
        return fact.status

    def resolve(self, subject, relation, chosen_value, source_id: str = "teacher", cycle: int = 0):
        """Teacher settles a dispute. Losing facts are retracted but kept in history."""
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        s, r, v = _n(subject), _n(relation), _n(chosen_value)
        for f in self._facts:
            if (f.subject, f.relation) == (s, r) and f.status != "retracted":
                if f.value == v:
                    f.sources[source_id] = self._trust_of(source_id)
                    f.status = "verified"
                    f.history.append(f"cycle {cycle}: resolved in favour by {source_id}")
                else:
                    f.status = "retracted"
                    f.history.append(f"cycle {cycle}: retracted by {source_id}")

    def retract(self, fact_id: int, reason: str = "", cycle: int = 0) -> None:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        for f in self._facts:
            if f.id == fact_id:
                f.status = "retracted"
                f.history.append(f"cycle {cycle}: retracted ({reason})")

    # ---- reading -----------------------------------------------------------------
    def query(self, subject=None, relation=None, value=None, statuses=("verified", "unverified", "disputed")):
        out = []
        for f in self._facts:
            if f.status not in statuses:
                continue
            if subject is not None and f.subject != _n(subject):
                continue
            if relation is not None and f.relation != _n(relation):
                continue
            if value is not None and f.value != _n(value):
                continue
            out.append(f)
        return out

    def ask(self, subject, relation) -> Answer:
        facts = sorted(self.query(subject, relation), key=lambda f: -f.confidence)
        if not facts:
            return Answer(False, [], f"I don't know {_n(relation)} of {_n(subject)}: nothing stored.")
        ans = [(f.value, f.confidence, f.status, sorted(f.sources)) for f in facts]
        disputed = any(f.status == "disputed" for f in facts)
        msg = "Sources disagree; flagged as disputed." if disputed else "Answered from stored facts."
        return Answer(True, ans, msg)

    def all_facts(self) -> list[Fact]:
        return list(self._facts)

    # ---- persistence -------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(f) for f in self._facts], fh, indent=1)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            self._facts = [Fact(**d) for d in json.load(fh)]
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
