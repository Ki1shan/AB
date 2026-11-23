"""Rule engine: structured if-then knowledge the brain executes directly."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

_OPS = {
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "in": lambda a, b: a in b, "contains": lambda a, b: b in a,
}


def check(cond: dict, state: dict) -> bool:
    key, op = cond["key"], cond["op"]
    if op == "exists":
        return key in state
    if op == "not_exists":
        return key not in state
    if key not in state or op not in _OPS:
        return False
    try:
        return bool(_OPS[op](state[key], cond["value"]))
    except TypeError:
        return False


@dataclass
class Rule:
    id: str
    when: list
    then: list
    priority: float = 0.5
    source: str = "author"
    status: str = "active"            # active | pending | retired
    fired: int = 0
    success: int = 0
    fail: int = 0

    @property
    def reliability(self) -> float:
        return (self.success + 1) / (self.success + self.fail + 2)

    @property
    def effective_priority(self) -> float:
        return self.priority * 2.0 * self.reliability

    def matches(self, state: dict) -> bool:
        return all(check(c, state) for c in self.when)


class RuleEngine:
    def __init__(self, retire_after_failures: int = 5, retire_below: float = 0.25):
        self._rules: dict[str, Rule] = {}
        self.retire_after, self.retire_below = retire_after_failures, retire_below

    def add(self, rule: Rule | dict) -> Rule:
        r = rule if isinstance(rule, Rule) else Rule(**rule)
        if r.id in self._rules:
            raise ValueError(f"rule id already exists: {r.id}")
        self._rules[r.id] = r
        return r

    def upsert(self, rule: Rule | dict) -> Rule:
        r = rule if isinstance(rule, Rule) else Rule(**rule)
        self._rules[r.id] = r
        return r

    def get(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id)

    def rules(self, status: str | None = None) -> list[Rule]:
        return [r for r in self._rules.values() if status is None or r.status == status]

    def match(self, state: dict) -> list[Rule]:
        hits = [r for r in self._rules.values() if r.status == "active" and r.matches(state)]
        return sorted(hits, key=lambda r: (-r.effective_priority, -len(r.when), r.id))

    def record_outcome(self, rule_id: str, ok: bool) -> None:
        r = self._rules.get(rule_id)
        if not r:
            return
        r.fired += 1
        r.success += int(ok)
        r.fail += int(not ok)
        if r.fail >= self.retire_after and r.reliability < self.retire_below:
            r.status = "retired"

    def record_deferred(self, rule_id: str) -> None:
        """Phase 4: record a need-more-evidence DEFERMENT, not a failure.

        The brain refused to act on insufficient evidence (the decider
        chose nothing and asked for more information instead of
        guessing). That is a deliberate deferral, not a rule failure:
        it increments ``fired`` only — neither ``success`` nor ``fail``
        — so it can never push the rule toward retirement, and the
        rule's reliability is untouched."""
        r = self._rules.get(rule_id)
        if not r:
            return
        r.fired += 1

    def approve(self, rule_id: str) -> None:
        if self._rules[rule_id].status == "pending":
            self._rules[rule_id].status = "active"

    def reject(self, rule_id: str) -> None:
        self._rules[rule_id].status = "retired"

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([r.__dict__ for r in self._rules.values()], fh, indent=1)
