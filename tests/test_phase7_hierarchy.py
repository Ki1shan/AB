"""Phase 7 hierarchical planning (D7-4) tests.

Decomposition is driven by the brain's own action beliefs (the
dependency graph in ab/planning/hierarchy.py), never by task
knowledge: a key depends on another target key when the cheapest
action producing it needs that key. The tests use two invented
two-key worlds:

* the dependency-order world: ``get_a`` (believed to reset ``b``)
  and ``get_b`` (needs ``a``, believed to reset ``a``) — no direct
  plan can satisfy both keys, so the parent decomposes and the
  subgoals must run ``a`` before ``b`` even though the target was
  declared ``b`` first.
* the failed-subgoal world: ``get_a`` is believed to reset ``x``,
  so again no direct plan exists; subgoal ``b`` gets a plan
  (``prep_b`` then ``get_b``) that the environment invalidates —
  ``prep_b`` flips ``x`` off as an unmodelled side effect — so the
  subgoal fails mid-execution and the parent must retry once, then
  fail with reason ``subgoal_failed`` and its partial results
  (``a``) preserved.
"""

import unittest

from ab.brain import Brain
from ab.planning.goals import Goal, GoalManager
from ab.planning.hierarchy import HierarchicalPlanner
from ab.planning.planner import Action


def _events(brain, kind):
    return [e for e in brain.events if e["kind"] == kind]


class _KeyEnv:
    """act(name) runs the side effects registered for that name (a
    list of (key, value) writes). Anything not registered is a
    no-op that returns True."""

    def __init__(self, state, effects=None):
        self.s = dict(state)
        self.effects = dict(effects or {})
        self.log = []

    def observe(self):
        return dict(self.s)

    def act(self, name, args=None):
        self.log.append(name)
        for key, value in self.effects.get(name, []):
            self.s[key] = value
        return True, "ok"


def _env_action(name, pre, eff):
    return Action(name, pre, eff, args={"_do": name})


# ----------------------------------------------------------------------
# dependency_order / decompose units
# ----------------------------------------------------------------------
class TestDependencyOrder(unittest.TestCase):
    def test_dependency_orders_subgoals_against_declaration(self):
        # b's cheapest producer needs a; a's needs nothing. Declared
        # [b, a], dependency order must be [a, b].
        acts = [_env_action("get_a", {}, {"a": True, "b": False}),
                _env_action("get_b", {"a": True}, {"b": True, "a": False})]
        ordered, had_cycle = HierarchicalPlanner.dependency_order(
            ["b", "a"], acts)
        self.assertEqual(ordered, ["a", "b"])
        self.assertFalse(had_cycle)

    def test_cycle_keeps_declaration_order_and_reports(self):
        acts = [_env_action("get_a", {"b": True}, {"a": True}),
                _env_action("get_b", {"a": True}, {"b": True})]
        ordered, had_cycle = HierarchicalPlanner.dependency_order(
            ["a", "b"], acts)
        self.assertEqual(ordered, ["a", "b"])
        self.assertTrue(had_cycle)

    def test_single_key_target_is_not_decomposed(self):
        manager = GoalManager()
        planner = HierarchicalPlanner(manager)
        goal = Goal("one key", {"a": True})
        manager.add(goal)
        subs, had_cycle = planner.decompose(goal, [])
        self.assertEqual(subs, [])
        self.assertFalse(had_cycle)
        self.assertEqual(manager.all(), [goal])

    def test_decompose_registers_subgoals_in_dependency_order(self):
        manager = GoalManager()
        planner = HierarchicalPlanner(manager)
        acts = [_env_action("get_a", {}, {"a": True, "b": False}),
                _env_action("get_b", {"a": True}, {"b": True, "a": False})]
        goal = Goal("get a and b", {"b": True, "a": True})
        manager.add(goal)
        subs, had_cycle = planner.decompose(goal, acts)
        self.assertFalse(had_cycle)
        self.assertEqual([next(iter(s.target)) for s in subs], ["a", "b"])
        self.assertTrue(all(s.parent == goal.id for s in subs))
        self.assertEqual(len(manager.all()), 3)


# ----------------------------------------------------------------------
# Brain integration: dependency-order success
# ----------------------------------------------------------------------
class TestHierarchicalSuccess(unittest.TestCase):
    def test_dependency_order_success_parent_completes(self):
        # No direct plan satisfies both keys (each action is believed
        # to reset the other), so the parent decomposes; subgoals run
        # a-then-b despite the target being declared b-first, and the
        # parent completes only when both are done.
        acts = [_env_action("get_a", {}, {"a": True, "b": False}),
                _env_action("get_b", {"a": True}, {"b": True, "a": False})]
        env = _KeyEnv({"a": False, "b": False},
                      {"get_a": [("a", True)], "get_b": [("b", True)]})
        b = Brain(acts, granted={"compute", "environment"})
        parent = b.add_goal("get a and b", {"b": True, "a": True})
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["a"])
        self.assertTrue(env.s["b"])
        self.assertEqual(parent.status, "done")
        # Subgoals ran in dependency order, not declaration order:
        # a before b, even though the target was declared b-first.
        self.assertEqual(env.log, ["get_a", "get_b"])
        # The first "via subgoal" plan is for :: a (the dependency
        # order is also proven at unit level; the second subgoal may
        # be planned directly once the parent is re-selected).
        via = [e["detail"] for e in _events(b, "plan")
               if e["detail"].startswith("via subgoal")]
        self.assertTrue(via)
        self.assertIn(":: a", via[0])
        # The parent was decomposed (never silently flattened).
        subs = [g for g in b.goals.all() if g.parent == parent.id]
        self.assertEqual(len(subs), 2)
        self.assertTrue(all(g.status == "done" for g in subs))


# ----------------------------------------------------------------------
# Brain integration: failed subgoal
# ----------------------------------------------------------------------
class TestFailedSubgoal(unittest.TestCase):
    def test_failed_subgoal_retries_once_then_parent_fails(self):
        # get_a is believed to reset BOTH b and x, so no direct plan
        # exists and the parent decomposes. Subgoal b's plan (prep_b,
        # get_b) looks fine at plan time, but prep_b flips x off as
        # an unmodelled side effect, so get_b blocks on its missing
        # dependency while subgoal b is active; the subgoal fails
        # ("no plan found" on replan) and is recorded on the parent.
        # The parent retries the failed key once, the retry is not
        # plannable either, and the parent fails with reason
        # subgoal_failed — with subgoal a's partial result (a=True)
        # preserved.
        acts = [_env_action("get_a", {}, {"a": True, "b": False,
                                          "x": False}),
                _env_action("prep_b", {}, {"ready": True}),
                _env_action("get_b", {"x": True, "ready": True},
                            {"b": True})]
        env = _KeyEnv({"x": True, "ready": False, "a": False, "b": False},
                      {"get_a": [("a", True)],
                       "prep_b": [("ready", True), ("x", False)],
                       "get_b": [("b", True)]})
        b = Brain(acts, granted={"compute", "environment"})
        parent = b.add_goal("get a and b", {"a": True, "b": True})
        self.assertEqual(b.run(env, 60), "impossible")
        self.assertEqual(parent.status, "failed")
        self.assertEqual(parent.fail_reason, "subgoal_failed")
        # Partial results preserved: subgoal a's work survived.
        self.assertTrue(env.s["a"])
        self.assertFalse(env.s["b"])
        # The failure was never silent: the subgoal failure was
        # recorded on the parent and the parent retried once.
        self.assertTrue(_events(b, "subgoal_failed"))
        self.assertTrue(_events(b, "subgoal_retry"))
        # The blocked step never executed (the env_change interrupt
        # invalidated the plan before get_b could run).
        self.assertEqual(env.log.count("get_b"), 0)
        self.assertTrue(any(e["kind"] == "interrupt"
                            and "get_b" in e["detail"]
                            for e in b.events))


if __name__ == "__main__":
    unittest.main()
