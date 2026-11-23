"""Phase 2: interruption and resumption.

Covers the urgent-percept trigger, higher-priority preemption,
environment-change invalidation, resume/replan behaviour, and the
no-nested-interruption guard.

NOTE: the urgent-percept trigger was initially written with
``t.created_cycle == self.cycle``, which could never fire — step()
increments the cycle before the check, so a percept made between steps
carries created_cycle == cycle - 1. The trigger now accepts percepts
made since the last cycle began (``created_cycle >= cycle - 1``); the
three tests that caught this (1, 2, and the run-level test in
test_phase2_stopping.py) pass.
"""
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.planning.planner import Action


def _treasure_brain():
    """RoomWorld brain with the explicit environment grant and the
    four-step treasure goal. Returns (env, brain, goal)."""
    env = RoomWorld()
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    goal = brain.add_goal("get the treasure", {"has_treasure": True})
    return env, brain, goal


def _drive_to_lab(env, brain):
    """step select, step plan, step execute go_lab. After this the plan
    is ['pick_lock', 'go_vault', 'take_treasure'] (1 of 4 steps done)."""
    brain.step(env)
    brain.step(env)
    brain.step(env)
    assert [a.name for a in brain.plan] == ["pick_lock", "go_vault", "take_treasure"]
    assert brain.world.get("at") == "lab"


class TestUrgentPerceptInterruption(unittest.TestCase):
    def test_urgent_percept_suspends_active_goal_with_position_preserved(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        b.step(env)
        self.assertIsNone(b.goals.active())
        rec = b.goals.suspended(g.id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.plan_position, 1)
        self.assertEqual([a["name"] for a in rec.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertTrue(any(e["kind"] == "interrupt"
                            and "trigger=urgent_percept" in e["detail"]
                            for e in b.events))

    def test_resumption_completes_goal_without_redoing_steps(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        b.step(env)
        self.assertTrue(b.resume_goal(g.id))
        self.assertIs(b.goals.active(), g)
        self.assertEqual([a.name for a in b.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual(b.run(env, 60), "done")
        self.assertEqual(g.status, "done")
        go_lab = [entry for entry in env.log
                  if entry.startswith("go{'room': 'lab'}")]
        self.assertEqual(len(go_lab), 1)

    def test_higher_priority_preempts_then_lower_resumes(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        low = b.add_goal("low", {"has_key": True}, priority=0.4)
        b.step(env)   # select
        b.step(env)   # plan
        self.assertIs(b.goals.active(), low)
        urgent = b.add_goal("urgent", {"door_open": True}, priority=0.9)
        b.step(env)
        self.assertIs(b.goals.active(), urgent)
        self.assertIsNotNone(b.goals.suspended(low.id))
        b.run(env, 60)
        self.assertEqual(urgent.status, "done")
        # Phase 3 auto-resume (mandate decision): when the preempting
        # goal finished, low was already resumed automatically and ran
        # to completion in the same run — no explicit resume is needed,
        # and resume_goal correctly refuses a goal with nothing to resume.
        self.assertIsNone(b.goals.suspended(low.id))
        self.assertEqual(low.status, "done")
        self.assertFalse(b.resume_goal(low.id))
        self.assertTrue(any(e["kind"] == "resume" for e in b.events))
        # The low goal's remaining take_key precondition went stale while
        # it was suspended (the urgent goal already took the key), so the
        # resumption had to replan instead of continuing the saved plan.
        self.assertTrue(any(e["kind"] == "replan_on_resume" for e in b.events))

    def test_env_change_invalidates_plan_without_destroying_goal(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        env.s["at"] = "hall"   # sabotage: pick_lock needs at=lab
        b.step(env)
        self.assertTrue(any(e["kind"] == "interrupt"
                            and "trigger=env_change" in e["detail"]
                            for e in b.events))
        self.assertIs(b.goals.active(), g)
        self.assertEqual(g.status, "active")
        idx = next(i for i, e in enumerate(b.events)
                   if e["kind"] == "interrupt" and "env_change" in e["detail"])
        self.assertTrue(any(e["kind"] == "plan" for e in b.events[idx + 1:]))
        self.assertEqual(b.run(env, 60), "done")
        self.assertEqual(g.status, "done")

    def test_resume_with_no_plan_fails_cleanly(self):
        b = Brain([Action("a1", {"k": 1}, {"done": True})], granted={"compute"})
        g = b.add_goal("do it", {"done": True})
        b.world = {"k": 1}
        b.step(None)   # select
        b.step(None)   # plan [a1]
        self.assertEqual([a.name for a in b.plan], ["a1"])
        self.assertIsNotNone(b.suspend_active_goal("test"))
        b.world = {"k": 0}   # precondition gone while suspended
        self.assertFalse(b.resume_goal(g.id))
        self.assertEqual(g.status, "failed")
        self.assertEqual(g.fail_reason, "no plan on resume")
        self.assertTrue(any(e["kind"] == "replan_on_resume" for e in b.events))

    def test_nested_interruption_during_resume_is_queued_not_executed(self):
        env, b, g = _treasure_brain()
        b.step(env)   # select: goal active
        self.assertIs(b.goals.active(), g)
        b._resuming = True
        try:
            self.assertIsNone(b._interrupt_current_goal("urgent_percept", "test"))
        finally:
            b._resuming = False
        self.assertIs(b.goals.active(), g)
        self.assertEqual(len(b._queued_interrupts), 1)
        self.assertEqual(b._queued_interrupts[0]["trigger"], "urgent_percept")


if __name__ == "__main__":
    unittest.main()
