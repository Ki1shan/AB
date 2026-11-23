"""Memory prioritization (Phase 6, design §2).

Utility per episode::

    utility = 0.5 * recency + 0.5 * usefulness
    recency    = 0.9 ** age_in_cycles   (age capped at 100)
    usefulness = min(1.0, uses / 3)

where ``uses`` counts how often the episode's (action, frozen
situation) pair was retrieved or contributed to a learned pattern
or a plan change. Pruning keeps the highest-utility episodes up to
the cap and drops the rest; patterns and consolidated knowledge are
never pruned -- they are the durable summary, and ``prune()`` only
ever touches episodes.

Stdlib only, deterministic.
"""
from __future__ import annotations

import copy

SCHEMA_VERSION = 1

RECENCY_BASE = 0.9
MAX_AGE = 100
USEFULNESS_SCALE = 3


def _frozen_key(action: str, situation: dict) -> tuple:
    """(action, frozen situation) key for usefulness counts."""
    return (action, tuple(sorted((k, str(v)) for k, v in (situation or {}).items())))


def _key_to_json(key: tuple) -> list:
    action, frozen = key
    return [action, [[k, v] for k, v in frozen]]


def _key_from_json(data: list) -> tuple:
    action, frozen = data
    return (action, tuple((k, v) for k, v in frozen))


class Prioritizer:
    """Utility scoring and pruning for the episode store."""

    def __init__(self, usefulness_counts: dict | None = None):
        # (action, frozen situation) -> use count.
        self._usefulness: dict[tuple, int] = dict(usefulness_counts or {})

    # -- usefulness -------------------------------------------------
    def note_use(self, action: str, situation: dict) -> None:
        """Record one use of an (action, situation) pair: it was
        retrieved, contributed to a learned pattern, or changed a
        plan."""
        key = _frozen_key(action, situation)
        self._usefulness[key] = self._usefulness.get(key, 0) + 1

    def uses(self, action: str, situation: dict) -> int:
        return self._usefulness.get(_frozen_key(action, situation), 0)

    # -- utility ----------------------------------------------------
    def utility(self, episode: dict, current_cycle: int,
                usefulness_counts: dict | None = None) -> float:
        """0.5 * recency + 0.5 * usefulness for one episode.

        ``episode`` carries "cycle", "action", "situation".
        ``usefulness_counts`` overrides the tracker's own counts
        when supplied (e.g. by an experiment); otherwise the
        tracker's counts are used.
        """
        cycle = episode.get("cycle", current_cycle)
        age = max(0, current_cycle - cycle)
        age = min(age, MAX_AGE)
        recency = RECENCY_BASE ** age
        counts = (self._usefulness if usefulness_counts is None
                  else usefulness_counts)
        key = _frozen_key(episode.get("action", ""),
                          episode.get("situation", {}))
        uses = counts.get(key, 0)
        usefulness = min(1.0, uses / USEFULNESS_SCALE)
        return 0.5 * recency + 0.5 * usefulness

    # -- pruning ----------------------------------------------------
    def prune(self, episodes: list, cap: int,
              current_cycle: int | None = None) -> list:
        """Keep the highest-utility episodes up to ``cap``.

        Returned highest-utility first; ties keep their original
        relative order (stable). Only episodes are considered --
        patterns and consolidated knowledge are never pruned.
        """
        if cap <= 0:
            return []
        episodes = list(episodes)
        if current_cycle is None:
            cycles = [ep.get("cycle", 0) for ep in episodes]
            current_cycle = max(cycles) if cycles else 0
        scored = sorted(
            enumerate(episodes),
            key=lambda pair: (-self.utility(pair[1], current_cycle),
                              pair[0]))
        return [copy.deepcopy(ep) for _, ep in scored[:cap]]

    # -- persistence ------------------------------------------------
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "usefulness": [[_key_to_json(key), count]
                               for key, count in sorted(
                                   self._usefulness.items(),
                                   key=lambda kv: (kv[0][0], kv[0][1]))]}

    @classmethod
    def from_json(cls, data: dict) -> "Prioritizer":
        if not isinstance(data, dict):
            raise ValueError("prioritizer data must be a dict")
        counts = {}
        for key_data, count in data.get("usefulness", []):
            counts[_key_from_json(key_data)] = int(count)
        return cls(usefulness_counts=counts)

