"""Episode retrieval (Phase 6, design §2).

An index over episodes, ranked by relevance to a query's features:
Jaccard overlap over the frozen situation (key, value) pairs, plus
an effect-match bonus when the query names an effect. Ties break
most-recent-first. Used when replanning after a failure (the
retrieved episodes are the evidence the consolidation pass and the
repeated-mistake detector read) and measured by retrieval precision
on held-out query sets.

Episode dicts: {"action", "situation", "outcome"/"effect",
optional "cycle"}. The episode's effect is its "effect" entry when
present, else its "outcome" entry.

Stdlib only, deterministic.
"""
from __future__ import annotations

import copy

# Bonus added to the Jaccard score when the episode's effect matches
# the effect named in the query. Keeps the score in [0, 1.5] and
# lets an exact effect match outrank a merely similar situation.
EFFECT_MATCH_BONUS = 0.5


def _frozen(features: dict) -> frozenset:
    """A situation / feature dict as a hashable set of (key, str(value))."""
    return frozenset((k, str(v)) for k, v in (features or {}).items())


def _episode_effect(episode: dict):
    if "effect" in episode:
        return episode["effect"]
    return episode.get("outcome")


class Retriever:
    """Index over episodes, ranked by feature overlap."""

    def __init__(self, episodes: list | None = None):
        self._episodes: list[dict] = []
        self._frozen: list[frozenset] = []
        if episodes:
            self.index(episodes)

    def index(self, episodes: list) -> None:
        """(Re)build the index from a list of episode dicts."""
        self._episodes = [copy.deepcopy(ep) for ep in episodes]
        self._frozen = [_frozen(ep.get("situation", {}))
                        for ep in self._episodes]

    def rebuild(self, episodes: list | None = None) -> None:
        """Rebuild the index; with no argument, re-indexes the
        episodes already held."""
        if episodes is not None:
            self.index(episodes)
        else:
            self._frozen = [_frozen(ep.get("situation", {}))
                            for ep in self._episodes]

    def clear(self) -> None:
        self._episodes = []
        self._frozen = []

    def __len__(self) -> int:
        return len(self._episodes)

    def _score(self, query_frozen: frozenset, query_effect,
               position: int) -> float:
        ep_frozen = self._frozen[position]
        union = query_frozen | ep_frozen
        jaccard = (len(query_frozen & ep_frozen) / len(union)
                   if union else 0.0)
        score = jaccard
        if query_effect is not None and _episode_effect(self._episodes[position]) == query_effect:
            score += EFFECT_MATCH_BONUS
        return score

    def retrieve(self, features: dict, k: int = 5) -> list:
        """Top-k episodes ranked by relevance to ``features``.

        Score = Jaccard overlap over frozen situation pairs, plus
        the effect-match bonus when ``features`` names an "effect"
        (or "outcome") that the episode produced. Ties break
        most-recent-first (later in the indexed list first).
        Returns copies of the episode dicts.
        """
        if k <= 0 or not self._episodes:
            return []
        features = dict(features or {})
        query_effect = features.get("effect", features.get("outcome"))
        query_frozen = _frozen({k: v for k, v in features.items()
                                if k not in ("effect", "outcome")})
        scored = []
        for position in range(len(self._episodes)):
            score = self._score(query_frozen, query_effect, position)
            scored.append((-score, -position, position))
        scored.sort()
        return [copy.deepcopy(self._episodes[pos])
                for _, _, pos in scored[:k]]

    def precision(self, query_features: dict, relevant_effect,
                  k: int = 5) -> float:
        """Fraction of the top-k retrieved episodes whose effect is
        ``relevant_effect``. 0.0 when nothing is retrieved."""
        features = dict(query_features or {})
        features["effect"] = relevant_effect
        retrieved = self.retrieve(features, k=k)
        if not retrieved:
            return 0.0
        hits = sum(1 for ep in retrieved
                   if _episode_effect(ep) == relevant_effect)
        return hits / len(retrieved)
