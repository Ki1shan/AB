"""Shared typed cognitive state (Phase 1, audit consolidated #4).

One store for the brain's working view of the world, so the planner, the
rule engine and the evaluator no longer reason about three divergent
copies (``Brain.world`` dict, LTM facts, WM items).

Responsibilities are deliberately kept separate:
- Long-term memory (``ab/core/memory.py``) is the source of truth for
  *beliefs*: append-only, source-tracked, verified-only inference.
- Working memory (``ab/core/working_memory.py``) holds the small set of
  thoughts currently in focus.
- This module holds the *working view*: one entry per world key, each with
  source, confidence, cycle, provenance and status, so a rule or a plan
  can see *why* the brain believes a key has a value.

Synchronization / conflict rules:
- Every write goes through ``CognitiveState.set``. Direct ``world.update``
  calls are gone from ``Brain``.
- An entry overwritten by a *different* source is not destroyed: the old
  entry is marked ``superseded`` and kept in a capped history (200 entries).
  Overwrites by the *same* source update the entry in place.
- Phase 3: ``known:<subject>|<relation>`` is a DERIVED view (see
  ``Brain.known``), never a stored entry — no write creates one, and
  ``as_dict()`` callers that need it read it through ``Brain.world``.
- Untrusted content is stored with status ``"untrusted"`` and is excluded
  from ``as_dict()``: it can be inspected, but never drives the planner or
  the rule engine.
- ``as_dict()`` returns plain values for the planner / rule engine only;
  provenance lives here, not in that view.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .vocab import normalize_text

MAX_HISTORY = 200


@dataclass
class StateEntry:
    key: str
    value: object
    source: str                     # e.g. "env:room", "ltm:fact-3", "inference:cat is_a mammal"
    confidence: float = 1.0         # 0..1, single semantics: probability the entry is correct
    cycle: int = 0
    provenance: str = ""            # full URL / file path / fact id, not just a domain
    entry_type: str = "observation"  # observation | belief | goal | plan | inferred | result
    status: str = "active"          # active | superseded | retracted
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def id(self) -> str:
        """Stable identifier: the key itself is the identity of a world entry."""
        return self.key

    def to_json(self) -> dict:
        return {"key": self.key, "value": self.value, "source": self.source,
                "confidence": self.confidence, "cycle": self.cycle,
                "provenance": self.provenance, "entry_type": self.entry_type,
                "status": self.status, "created_at": self.created_at,
                "updated_at": self.updated_at}

    @classmethod
    def from_json(cls, d: dict) -> "StateEntry":
        if not isinstance(d, dict) or "key" not in d or "value" not in d or "source" not in d:
            raise ValueError("state entry needs 'key', 'value' and 'source'")
        return cls(key=str(d["key"]), value=d["value"], source=str(d["source"]),
                   confidence=max(0.0, min(1.0, float(d.get("confidence", 1.0)))),
                   cycle=int(d.get("cycle", 0)), provenance=str(d.get("provenance", "")),
                   entry_type=str(d.get("entry_type", "observation")),
                   status=str(d.get("status", "active")),
                   created_at=float(d.get("created_at", 0.0)),
                   updated_at=float(d.get("updated_at", 0.0)))


class CognitiveState:
    """The brain's working view of the world: one typed entry per key."""

    def __init__(self):
        self._entries: dict[str, StateEntry] = {}
        self._history: list[StateEntry] = []
        # Phase 12 (P12-O5): mutation counter for Brain.world's
        # version-gated cache. Bumped on every set/delete/clear
        # (runtime-only, never persisted); a fresh object starts at 0.
        self.version = 0

    def set(self, key, value, source: str, confidence: float = 1.0, cycle: int = 0,
            provenance: str = "", entry_type: str = "observation",
            status: str = "active") -> StateEntry | None:
        """Set an entry. Returns the previous entry (or None). A previous
        entry from a different source is marked superseded and kept in
        history; a same-source overwrite updates in place. ``status`` is
        normally "active"; untrusted content (Phase 3 research path) is
        stored with status "untrusted" so it never appears in the plain
        ``as_dict()`` view the planner / rule engine reason over."""
        key = normalize_text(key) if isinstance(key, str) and "|" not in key else str(key)
        now = time.time()
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        prev = self._entries.get(key)
        if prev is not None and prev.source != source:
            prev.status = "superseded"
            self._history.append(prev)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
            created = prev.created_at
        elif prev is not None:
            created = prev.created_at
        else:
            created = now
        entry = StateEntry(key=key, value=value, source=source,
                           confidence=max(0.0, min(1.0, float(confidence))),
                           cycle=int(cycle), provenance=provenance,
                           entry_type=entry_type, status=status,
                           created_at=created, updated_at=now)
        self._entries[key] = entry
        return prev

    def delete(self, key) -> StateEntry | None:
        """Remove an entry (Phase 3: checked predictions are deleted after
        the check). Returns the removed entry, or None if absent."""
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        return self._entries.pop(str(key), None)

    def get(self, key) -> StateEntry | None:
        return self._entries.get(str(key))

    def __contains__(self, key) -> bool:
        return str(key) in self._entries

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def clear(self) -> None:
        """Remove all entries (used when a whole world dict is assigned)."""
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        self._entries.clear()

    def as_dict(self) -> dict:
        """Plain values only, for planner / rule engine / evaluator.
        Only status "active" entries appear: superseded entries are
        history, and "untrusted" entries (Phase 3 research path) must
        never drive reasoning."""
        return {k: e.value for k, e in self._entries.items() if e.status == "active"}

    def snapshot(self) -> dict[str, StateEntry]:
        return dict(self._entries)

    def history(self) -> list[StateEntry]:
        return list(self._history)

    # ---- persistence -----------------------------------------------------
    def to_json(self) -> dict:
        return {"entries": [e.to_json() for e in self._entries.values()],
                "history": [e.to_json() for e in self._history]}

    @classmethod
    def from_json(cls, data: dict) -> "CognitiveState":
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("state file has no 'entries' list")
        st = cls()
        for d in data["entries"]:
            e = StateEntry.from_json(d)
            st._entries[e.key] = e
        for d in data.get("history", []):
            st._history.append(StateEntry.from_json(d))
        return st
