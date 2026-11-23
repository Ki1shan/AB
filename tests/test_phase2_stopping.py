"""Phase 2: stopping conditions — explicit stop reasons returned by
Brain.run(), persistence of the stop reason, goal-progress stall
detection, and the CognitiveControl progress unit behaviour.
"""
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.core.control import CognitiveControl


def _treasure_brain(**kwargs):
    env = RoomWorld()
    brain = Brain(room_world_actions(), granted={"compute", "environment"}, **kwargs)
    goal = brain.add_goal("get the treasure", {"has_treasure": True})
    return env, brain, goal


class StuckWorld(RoomWorld):
    """RoomWorld where take_key always reports success but never
    produces the key, and observe() alternates a meaningless "noise"
    key on every call — the world churns every cycle while goal
    progress stays flat, so only progress-based stall may fire."""

    def __init__(self):
        super().__init__()
        self._n = 0

    def observe(self):
        self._n += 1
        s = dict(self.s)
        s["noise"] = self._n % 2
        return s

    def act(self, name, args):
        if name == "take_key" and self.s["at"] == "lab":
            self.log.append(f"take_key{args or ''}: took the key")
            return True, "took the key"   # lies: has_key stays False
        return super().act(name, args)


class TestStopReasons(unittest.TestCase):
    def test_budget_exhaustion_returns_reason(self):
        env, b, g = _treasure_brain(max_cycles=3)
        result = b.run(env)
        self.assertEqual(result, "budget_exhausted")
        self.assertEqual(b.last_stop_reason, "budget_exhausted")
        self.assertLessEqual(b.cycle, 3)

    def test_impossible_goal_returns_impossible(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("fly to the moon", {"on_moon": True})
        result = b.run(env, 40)
        self.assertEqual(result, "impossible")
        self.assertEqual(g.status, "failed")
        self.assertLess(b.cycle, 40)

    def test_done_run_returns_done(self):
        env, b, g = _treasure_brain()
        result = b.run(env, 60)
        self.assertEqual(result, "done")
        self.assertEqual(g.status, "done")

    def test_repeated_failure_classified_impossible_not_stalled(self):
        # (Renamed in Phase 10.5 — was
        # test_stall_detected_on_goal_progress_not_world_churn.)
        # pick_lock fails first, then the brain replans to the key route
        # and take_key "succeeds" without effect forever; the successful
        # steps never materialise, so goal progress stays flat while the
        # noise key flips every cycle. No extra sabotage (e.g. a no-op
        # unlock) was needed.
        # §9 reclassification (Phase 10.5, P10.5-7 loop-plan rule):
        # the old expectation was that the give_up_when_stalled rule
        # fires here. It no longer does — and that is the loop rule
        # working as designed, not a regression in stall detection.
        # Measured trace (seed-independent, deterministic): pick_lock
        # fails twice and take_key fails twice, each in the identical
        # (precondition-restricted) situation, so by cycle 10 every
        # available plan is a loop plan and the goal fails as
        # ``impossible`` (``repeated_failure_no_alternative``) instead
        # of ``stalled``. The loop rule's diagnosis is more specific
        # than the stall detector's generic verdict: the brain has
        # PROVEN both routes fail, not merely made no progress. The
        # stall detector itself is unchanged and still unit-tested at
        # the predicate level (is_stalled_by_progress, below); it
        # remains the backstop for a goal that makes no progress
        # WITHOUT retrying a twice-failed step. A dedicated
        # Brain-level stall-isolation scenario was prototyped and
        # REJECTED: isolating the stall rule needs 5 distinct routes
        # (4 flat samples after the last first-violation reset, with
        # the loop rule firing on the 5th route's 2nd failure), but
        # the Phase 1 rule-retirement mechanism retires execute_plan
        # after 5 failed firings, so the brain goes idle before the
        # stall window can complete — building that test would mean
        # changing the retirement mechanism, which is out of Phase
        # 10.5 scope. The reclassification, not a workaround, is
        # recorded here and in PHASE-10.5-IMPLEMENTATION-REPORT.md.
        env = StuckWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        result = b.run(env, 60)
        self.assertEqual(g.status, "failed")
        self.assertEqual(g.fail_reason, "repeated_failure_no_alternative")
        self.assertEqual(b.last_stop_reason, "impossible")
        self.assertEqual(result, "impossible")
        self.assertTrue(any(e["kind"] == "repeated_failure_no_alternative"
                            for e in b.events))
        self.assertTrue(any(e["kind"] == "evaluate"
                            and "repeated_failure_no_alternative" in e["detail"]
                            for e in b.events))

    def test_stop_reason_persisted_and_restored(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("fly to the moon", {"on_moon": True})
        b.run(env, 40)
        self.assertEqual(b.last_stop_reason, "impossible")
        folder = tempfile.mkdtemp()
        b.save(folder)
        b2 = Brain()
        b2.load(folder)
        self.assertEqual(b2.last_stop_reason, "impossible")

    def test_interrupted_run_returns_interrupted(self):
        env, b, g = _treasure_brain()
        b.step(env)   # select
        b.step(env)   # plan
        b.step(env)   # execute go_lab
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        b.step(env)   # suspends the goal
        self.assertIsNone(b.goals.active())
        result = b.run(env, 60)
        self.assertEqual(result, "interrupted")


class TestControlProgressStallUnit(unittest.TestCase):
    def test_control_progress_stall_unit(self):
        c = CognitiveControl()
        for v in [0, 0, 0, 0]:
            c.record_progress(0, v)
        self.assertTrue(c.is_stalled_by_progress())

        c2 = CognitiveControl()
        for i, v in enumerate([0, 1, 1, 1]):
            c2.record_progress(i, v)
        self.assertFalse(c2.is_stalled_by_progress())

        c3 = CognitiveControl()
        c3.record_progress(0, 0)
        c3.record_progress(1, 0)
        self.assertFalse(c3.is_stalled_by_progress())   # fewer than 4 entries

        c.reset_progress()
        self.assertEqual(c.progress_history(), [])
        self.assertFalse(c.is_stalled_by_progress())

        # The OLD world-string stall behaviour is untouched.
        self.assertTrue(CognitiveControl().is_stalled(["a", "a", "a", "a"]))
        self.assertFalse(CognitiveControl().is_stalled(["a", "b", "a", "b"]))


if __name__ == "__main__":
    unittest.main()
