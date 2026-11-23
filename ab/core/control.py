"""Cognitive control: inhibition, goal-switch discipline, stall and budget checks."""
from __future__ import annotations

from .thought import Thought


class CognitiveControl:
    def __init__(self, switch_margin=0.15, max_cycles=200, stall_window=4,
                 min_confidence=0.2, inhibited_tags=()):
        self.switch_margin = switch_margin
        self.max_cycles = max_cycles
        self.stall_window = stall_window
        self.min_confidence = min_confidence
        self.inhibited_tags = {t.lower() for t in inhibited_tags}
        self._progress: list[tuple[int, float]] = []

    def record_progress(self, cycle: int, progress: float) -> None:
        """Record one goal-progress sample; keep only the last 8 entries."""
        self._progress.append((cycle, progress))
        self._progress = self._progress[-8:]

    def is_stalled_by_progress(self) -> bool:
        """True when progress has not advanced over the last stall_window entries."""
        w = self.stall_window
        if len(self._progress) < w:
            return False
        return len({p for _, p in self._progress[-w:]}) == 1

    def reset_progress(self) -> None:
        """Clear the progress record (goal selected, switched, completed, or failed)."""
        self._progress = []

    def progress_history(self) -> list[tuple[int, float]]:
        """Return a copy of the progress record, for tests/inspection."""
        return list(self._progress)

    def should_switch(self, current_priority: float, challenger_priority: float) -> bool:
        """Hysteresis: only abandon the current goal for a clearly better one."""
        return challenger_priority > current_priority + self.switch_margin

    def inhibit(self, thoughts: list[Thought]) -> tuple[list[Thought], list[Thought]]:
        kept, blocked = [], []
        for t in thoughts:
            bad = t.confidence < self.min_confidence or (t.tags & self.inhibited_tags)
            (blocked if bad else kept).append(t)
        return kept, blocked

    def over_budget(self, cycle: int) -> bool:
        return cycle >= self.max_cycles

    # Phase 4: the ``budget_exceeded`` alias for ``over_budget`` was
    # removed (mandate cleanup — it was unused and duplicated the
    # canonical name). Call sites use ``over_budget``.

    def is_stalled(self, history: list[str]) -> bool:
        w = self.stall_window
        return len(history) >= w and len(set(history[-w:])) == 1

    @staticmethod
    def resolve_conflicts(proposals: list[tuple[str, float, frozenset]]) -> list[str]:
        """proposals: (name, priority, resources). Greedy by priority, no shared resources."""
        chosen, used = [], set()
        for name, _, res in sorted(proposals, key=lambda p: -p[1]):
            if used & set(res):
                continue
            chosen.append(name)
            used |= set(res)
        return chosen
