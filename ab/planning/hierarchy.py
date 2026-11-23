"""Phase 7 hierarchical planning (D7-4).

``HierarchicalPlanner`` decomposes a goal whose target has >= 2 keys
into ordered subgoals — one real ``Goal`` per target key, created via
the existing ``GoalManager.decompose`` (parent = the parent goal id).
Ordering comes from the brain's own action beliefs, never from
task-specific knowledge: build a dependency graph over
(produced key -> precondition keys) from the cheapest action
producing each key, topologically sort it, and on a cycle keep
declaration order and report it (the brain logs a ``hierarchy_cycle``
event). Single-key targets are NOT decomposed.

Subgoal failure handling lives in the brain (``Brain._do_fail_goal``
records ``subgoal_failed`` on the parent; ``Brain._do_make_plan``
retries once by re-decomposing the failed key, then fails the parent
with reason ``subgoal_failed`` and its partial results preserved) —
this module only builds the ordered decomposition.
"""
from __future__ import annotations

from ..planning.goals import Goal


class HierarchicalPlanner:
    """Decompose multi-key goals into dependency-ordered subgoals."""

    def __init__(self, goals):
        """``goals`` is the ``GoalManager`` the subgoals are added to."""
        self.goals = goals

    # -- dependency graph ---------------------------------------------
    @staticmethod
    def dependency_order(target, actions) -> tuple[list, bool]:
        """Order target keys by dependency: key K depends on key J
        when the cheapest action producing K has J in its preconditions
        (J must itself be a target key). An action produces K only
        when its effect sets K to the TARGET value — an action that
        sets K to False does not produce a True target (and vice
        versa). ``target`` is the goal target dict (or a list of keys,
        meaning every value is True). Topological sort, ties broken
        by declaration order (deterministic). Returns
        ``(ordered_keys, had_cycle)``; on a cycle the remaining keys
        keep declaration order and ``had_cycle`` is True."""
        if isinstance(target, dict):
            keys = list(target)
            wanted = dict(target)
        else:
            keys = list(target)
            wanted = {k: True for k in keys}
        key_set = set(keys)
        deps: dict[str, set[str]] = {k: set() for k in keys}
        for key in keys:
            producers = [a for a in (actions or [])
                         if (getattr(a, "eff", {}) or {}).get(key) == wanted[key]]
            if not producers:
                continue
            cheapest = min(producers,
                           key=lambda a: (float(getattr(a, "cost", 1.0)),
                                          str(getattr(a, "name", ""))))
            for pre_key in (getattr(cheapest, "pre", {}) or {}):
                if pre_key in key_set and pre_key != key:
                    deps[key].add(pre_key)
        ordered: list[str] = []
        remaining = list(keys)
        had_cycle = False
        while remaining:
            ready = [k for k in remaining if deps[k] <= set(ordered)]
            if not ready:
                # Cycle: keep declaration order for the rest and report it.
                had_cycle = True
                ordered.extend(remaining)
                break
            ordered.append(ready[0])
            remaining.remove(ready[0])
        return ordered, had_cycle

    # -- decomposition --------------------------------------------------
    def decompose(self, goal, actions) -> tuple[list[Goal], bool]:
        """Decompose ``goal`` into one subgoal per target key, in
        dependency order, registered via ``GoalManager.decompose``.
        Returns ``(subgoals, had_cycle)``. A single-key target is not
        decomposed (returns ``([], False)``)."""
        target = dict(getattr(goal, "target", {}) or {})
        if len(target) < 2:
            return [], False
        order, had_cycle = self.dependency_order(target, actions)
        subgoals = [Goal(f"{goal.description} :: {k}", {k: target[k]},
                         priority=getattr(goal, "priority", 0.5))
                    for k in order]
        self.goals.decompose(goal.id, subgoals)
        return subgoals, had_cycle
