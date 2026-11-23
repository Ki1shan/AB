"""Result evaluation: did the goal succeed, and did the world behave as predicted?"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evaluation:
    success: bool
    score: float                        # fraction of target conditions met
    progress: float                     # score change versus the 'before' state
    unmet: list = field(default_factory=list)
    reasons: list = field(default_factory=list)


class Evaluator:
    @staticmethod
    def _score(target: dict, state: dict) -> tuple[float, list]:
        if not target:
            return 1.0, []
        unmet = [(k, v, state.get(k)) for k, v in target.items() if state.get(k) != v]
        return 1.0 - len(unmet) / len(target), unmet

    def evaluate_goal(self, target: dict, before: dict, after: dict) -> Evaluation:
        s_after, unmet = self._score(target, after)
        s_before, _ = self._score(target, before)
        reasons = [f"{k}: wanted {w!r}, got {g!r}" for k, w, g in unmet] or ["all conditions met"]
        return Evaluation(s_after == 1.0, s_after, s_after - s_before, unmet, reasons)

    @staticmethod
    def check_prediction(expected_effects: dict, after: dict) -> list:
        """Returns surprises: [(key, expected, actual)]. Empty list means the world behaved."""
        return [(k, v, after.get(k)) for k, v in expected_effects.items() if after.get(k) != v]
