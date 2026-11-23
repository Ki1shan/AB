"""Decision making: transparent weighted-utility choice with explicit 'don't know' outcomes."""
from __future__ import annotations

from dataclasses import dataclass, field


# Phase 3 (mandate decision, design §6.1): feasibility is rescaled so a
# literal < 0.25 gate is meaningful. feasibility = 1 / (1 + cost / 5):
# a 4-step plan is 0.56, a 10-step plan is 0.33, a 15-step plan is 0.25,
# a 20-step plan is 0.20. The old formula 1 / (1 + cost) gave the demo's
# own 4-step treasure plan 0.20, so a literal gate refused it.
FEASIBILITY_SCALE = 5.0
FEASIBILITY_MIN = 0.25


def feasibility_for_cost(cost: float) -> float:
    """Feasibility of a plan from its cost (Phase 3 rescale)."""
    return 1.0 / (1.0 + float(cost) / FEASIBILITY_SCALE)


@dataclass
class Option:
    name: str
    scores: dict                      # criterion -> 0..1
    payload: object = None


@dataclass
class Decision:
    chosen: Option | None
    ranking: list                     # [(name, utility)]
    reason: str
    tie: bool = False
    feasibility: float = 0.0
    fact_confidence: float = 1.0
    chosen_goal_id: int | None = None
    # Phase 4: confidence of the reasoning chain behind this decision
    # (weakest derivation confidence from the last cycle's inferences,
    # or 1.0 when no inference was involved), and the contradiction
    # resolution policy in force when the decision was made, if any.
    reasoning_confidence: float = 1.0
    resolution_policy: str | None = None

    def to_json(self) -> dict:
        return {
            "chosen": self.chosen.name if self.chosen is not None else None,
            "ranking": [[name, utility] for name, utility in self.ranking],
            "reason": self.reason,
            "tie": self.tie,
            "feasibility": self.feasibility,
            "fact_confidence": self.fact_confidence,
            "chosen_goal_id": self.chosen_goal_id,
            "reasoning_confidence": self.reasoning_confidence,
            "resolution_policy": self.resolution_policy,
        }

    @classmethod
    def from_json(cls, d: dict) -> Decision:
        if not isinstance(d, dict) or "ranking" not in d or "reason" not in d:
            raise ValueError("decision JSON must be a dict with 'ranking' and 'reason'")
        chosen_name = d.get("chosen")
        chosen = Option(chosen_name, {}, None) if chosen_name is not None else None
        ranking = [(name, utility) for name, utility in d["ranking"]]
        return cls(
            chosen=chosen,
            ranking=ranking,
            reason=d["reason"],
            tie=d.get("tie", False),
            feasibility=d.get("feasibility", 0.0),
            fact_confidence=d.get("fact_confidence", 1.0),
            chosen_goal_id=d.get("chosen_goal_id", None),
            reasoning_confidence=d.get("reasoning_confidence", 1.0),
            resolution_policy=d.get("resolution_policy"),
        )


class DecisionMaker:
    def __init__(self, weights: dict, min_utility: float = 0.25, tie_margin: float = 0.03):
        total = sum(weights.values()) or 1.0
        self.weights = {k: v / total for k, v in weights.items()}
        self.min_utility, self.tie_margin = min_utility, tie_margin

    def utility(self, option: Option) -> float:
        return sum(w * max(0.0, min(1.0, option.scores.get(k, 0.0))) for k, w in self.weights.items())

    def decide(self, options: list[Option]) -> Decision:
        if not options:
            return Decision(None, [], "no options available")
        ranked = sorted(((o, self.utility(o)) for o in options), key=lambda p: -p[1])
        ranking = [(o.name, round(u, 4)) for o, u in ranked]
        best, best_u = ranked[0]
        if best_u < self.min_utility:
            return Decision(None, ranking, f"best option '{best.name}' scored {best_u:.2f} "
                                           f"< minimum {self.min_utility}: need more evidence")
        tie = len(ranked) > 1 and (best_u - ranked[1][1]) < self.tie_margin
        why = ", ".join(f"{k}={best.scores.get(k, 0):.2f}x{w:.2f}" for k, w in self.weights.items())
        return Decision(best, ranking, f"chose '{best.name}' (utility {best_u:.2f}: {why})", tie)

    def decide_with_evidence(self, options: list[Option], feasibility: float = 0.0,
                             fact_confidence: float = 1.0,
                             chosen_goal_id: int | None = None,
                             reasoning_confidence: float = 1.0) -> Decision:
        decision = self.decide(options)
        decision.feasibility = feasibility
        decision.fact_confidence = fact_confidence
        decision.chosen_goal_id = chosen_goal_id
        decision.reasoning_confidence = reasoning_confidence
        return decision
