import unittest
from ab.planning.goals import Goal, GoalManager
from ab.planning.planner import Action, Planner
from ab.planning.evaluator import Evaluator
from ab.body.environment import room_world_actions


class TestPlanner(unittest.TestCase):
    def test_finds_cheapest_plan(self):
        p = Planner(room_world_actions())
        s = {"at": "hall", "has_key": False, "door_open": False, "has_treasure": False}
        plan = p.plan(s, {"has_treasure": True}, banned={"pick_lock"})
        self.assertEqual([a.name for a in plan.steps],
                         ["go_lab_from_hall", "take_key", "unlock", "go_vault", "take_treasure"])

    def test_failures_make_actions_costlier_and_reroute(self):
        p = Planner(room_world_actions())
        s = {"at": "hall", "has_key": False, "door_open": False, "has_treasure": False}
        self.assertIn("pick_lock", [a.name for a in p.plan(s, {"has_treasure": True}).steps])
        p.record("pick_lock", False)
        self.assertNotIn("pick_lock", [a.name for a in p.plan(s, {"has_treasure": True}).steps])

    def test_unreachable_returns_none_and_trivial_is_empty(self):
        p = Planner([Action("a", {"x": 1}, {"y": 1})])
        self.assertIsNone(p.plan({"x": 0}, {"y": 1}))
        self.assertEqual(p.plan({"y": 1}, {"y": 1}).steps, [])

    def test_node_cap_prevents_runaway(self):
        acts = [Action(f"a{i}", {"n": i}, {"n": i + 1}) for i in range(500)]
        self.assertIsNone(Planner(acts, max_nodes=10).plan({"n": 0}, {"n": 500}))


class TestGoalsAndEval(unittest.TestCase):
    def test_goal_lifecycle_and_single_active(self):
        gm = GoalManager()
        a = gm.add(Goal("first", {"x": 1})); b = gm.add(Goal("second", {"y": 1}))
        gm.activate(a.id); gm.activate(b.id)
        self.assertEqual(gm.active().id, b.id); self.assertEqual(a.status, "pending")
        gm.complete(b.id); gm.fail(a.id, "nope")
        self.assertEqual((b.status, a.status, a.fail_reason), ("done", "failed", "nope"))
        self.assertTrue(gm.satisfied(Goal("g", {"x": 1}), {"x": 1}))

    def test_decompose_links_parent(self):
        gm = GoalManager(); p = gm.add(Goal("big", {}))
        subs = gm.decompose(p.id, [Goal("s1", {}), Goal("s2", {})])
        self.assertTrue(all(s.parent == p.id for s in subs))

    def test_evaluation_progress_and_surprise(self):
        ev = Evaluator()
        r = ev.evaluate_goal({"a": 1, "b": 2}, {"a": 0, "b": 0}, {"a": 1, "b": 0})
        self.assertFalse(r.success); self.assertAlmostEqual(r.score, 0.5); self.assertAlmostEqual(r.progress, 0.5)
        self.assertEqual(ev.check_prediction({"a": 1}, {"a": 2}), [("a", 1, 2)])
        self.assertEqual(ev.check_prediction({"a": 1}, {"a": 1}), [])


if __name__ == "__main__":
    unittest.main()
