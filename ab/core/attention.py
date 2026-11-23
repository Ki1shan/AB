"""Attention: scores candidate thoughts and decides what gets into focus."""
from __future__ import annotations

from .thought import Thought, clamp
from .vocab import tokenize
from .working_memory import FOCUS_K


class Attention:
    def __init__(self, w_urgency=0.35, w_goal=0.35, w_novelty=0.15, w_recency=0.15,
                 interrupt_threshold=0.9):
        total = w_urgency + w_goal + w_novelty + w_recency
        self.w = (w_urgency / total, w_goal / total, w_novelty / total, w_recency / total)
        self.interrupt_threshold = interrupt_threshold
        self._seen: set[str] = set()

    @staticmethod
    def _words(thought: Thought) -> set[str]:
        words = set(thought.tags)
        if isinstance(thought.content, str):
            words |= tokenize(thought.content)
        return words

    def goal_relevance(self, thought: Thought, goal_keywords: set[str]) -> float:
        if not goal_keywords:
            return 0.0
        overlap = len(self._words(thought) & goal_keywords)
        return clamp(overlap / max(1, min(len(goal_keywords), 3)))

    def score(self, thought: Thought, goal_keywords: set[str], cycle: int) -> float:
        novelty = 0.2 if thought.signature() in self._seen else 1.0
        recency = 1.0 / (1.0 + max(0, cycle - thought.created_cycle))
        wu, wg, wn, wr = self.w
        s = (wu * thought.urgency + wg * self.goal_relevance(thought, goal_keywords)
             + wn * novelty + wr * recency)
        return clamp(s * (0.5 + 0.5 * thought.confidence))   # doubtful thoughts count for less

    def focus(self, candidates: list[Thought], goal_keywords: set[str], cycle: int,
              k: int = FOCUS_K) -> list[Thought]:
        for t in candidates:
            t.salience = self.score(t, goal_keywords, cycle)
        ranked = sorted(candidates, key=lambda t: -t.salience)
        for t in ranked:
            self._seen.add(t.signature())
        return ranked[:k]

    def interrupts(self, thought: Thought) -> bool:
        return thought.urgency >= self.interrupt_threshold

    # ---- Phase 3: the seen-set lives in shared state ----------------------
    # The seen-set is still kept here for novelty scoring, but it is no
    # longer persisted as a separate file: Brain writes each focused
    # thought's signature to state as a ``seen:<sha>`` entry (source
    # "attention") and re-seeds ``_seen`` from those entries on load.
    def restore_seen(self, signatures) -> None:
        """Seed the seen-set (used on load, from state entries)."""
        self._seen |= {str(s) for s in signatures}

    # ---- persistence (Phase 1; legacy only in Phase 3) -------------------
    # ``to_json`` / ``from_json`` are kept so Phase 1–2 saves carrying
    # ``attention.json`` still load (Brain seeds state entries from it).
    # Phase 3 saves do NOT write attention.json.
    def to_json(self) -> dict:
        return {"w": list(self.w), "interrupt_threshold": self.interrupt_threshold,
                "seen": sorted(self._seen)}

    @classmethod
    def from_json(cls, data: dict) -> "Attention":
        if not isinstance(data, dict) or "seen" not in data:
            raise ValueError("attention file has no 'seen' list")
        a = cls(*data.get("w", [0.35, 0.35, 0.15, 0.15]),
                interrupt_threshold=data.get("interrupt_threshold", 0.9))
        a._seen = set(data["seen"])
        return a
