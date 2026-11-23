"""Phase 12 W7: tracemalloc-diagnosis fixes (ADR P12-12).

The first 10,000-cycle re-soak (W6) still failed boundedness on RSS
alone (+1,071.8 KiB/1,000 cycles vs limit 295.4) even though both
capped stores pinned at 1,000. The residual growth was three small
unbounded per-cycle/per-leg accumulations the soak did not sample:

1. ``Brain._goal_thoughts`` — one cached Thought retained per
   completed goal (``_reset_after_goal`` never dropped it).
2. ``WorldModel._observed_in_cycle`` + ``_decayed_cycles`` — one
   entry per cycle, never pruned.
3. ``SimulationRevision.history`` — append-only, no cap.

W7 fixed all three (brain.py drops the cached Thought with the
goal; decay() prunes dead cycle bookkeeping; history is capped at
1,000 oldest-first). These tests pin the fixes:

- Each fix holds its bound under direct driving.
- The decay prune is semantics-preserving: decay still decays,
  a repeat decay of the same cycle is still a no-op, and the
  persistence round-trip keeps the pruned sets small.
- The revision cap keeps the newest 1,000, on append and on load.
- A 2,000-cycle soak run: all three stores stay within their
  bounds throughout (the 10,000-cycle soak is the proof; this
  pins the regression in the suite).
"""
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.core.thought import Thought
from ab.eval.soak import Soak
from ab.imagination.revision import MAX_REVISION_HISTORY, SimulationRevision
from ab.world.model import WorldModel


class TestGoalThoughtDropped(unittest.TestCase):
    def test_reset_after_goal_drops_cached_thought(self):
        b = Brain()
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        b._goal_thoughts[goal.id] = Thought(
            goal.description, "goal", "goal-manager")
        self.assertIn(goal.id, b._goal_thoughts)
        b._reset_after_goal(goal)
        self.assertNotIn(goal.id, b._goal_thoughts)

    def test_reset_after_goal_none_is_safe(self):
        b = Brain()
        b._reset_after_goal(None)  # no goal: nothing to drop, no crash.

    def test_completed_goal_leaves_no_cached_thought(self):
        # End to end: a real treasure hunt to "done" must not retain
        # the completed goal's Thought.
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        env = RoomWorld()
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        reason = b.run(env, 80)
        self.assertEqual(reason, "done")
        self.assertEqual(goal.status, "done")
        self.assertNotIn(goal.id, b._goal_thoughts)

    def test_two_hunts_do_not_accumulate_thoughts(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for _ in range(2):
            env = RoomWorld()
            goal = b.add_goal("get the treasure", {"has_treasure": True})
            b.run(env, b.cycle + 80)
            self.assertEqual(goal.status, "done")
        # At most the currently-active goal's Thought, never one per
        # completed goal.
        self.assertLessEqual(len(b._goal_thoughts), 1)


class TestDecayPrune(unittest.TestCase):
    def _model_with_entity(self):
        m = WorldModel()
        m.set_property("torch", "lit", True, cycle=0)
        return m

    def _lit_confidence(self, m):
        ent = next(e for e in m.entities() if e.id == "torch")
        return ent.properties["lit"].confidence

    def test_bookkeeping_stays_o1_over_many_cycles(self):
        m = self._model_with_entity()
        for cycle in range(1, 101):
            m.decay(cycle)
            self.assertLessEqual(len(m._observed_in_cycle), 1,
                                 f"cycle {cycle}: _observed_in_cycle grew")
            self.assertLessEqual(len(m._decayed_cycles), 1,
                                 f"cycle {cycle}: _decayed_cycles grew")

    def test_decay_still_decays(self):
        m = self._model_with_entity()
        before = self._lit_confidence(m)
        m.decay(1)
        after = self._lit_confidence(m)
        self.assertLess(after, before)

    def test_repeat_decay_same_cycle_is_noop(self):
        m = self._model_with_entity()
        m.decay(1)
        conf = self._lit_confidence(m)
        m.decay(1)  # guard entry retained for the just-decayed cycle.
        self.assertEqual(self._lit_confidence(m), conf)

    def test_pruned_sets_survive_save_load(self):
        m = self._model_with_entity()
        for cycle in range(1, 51):
            m.decay(cycle)
        loaded = WorldModel.from_json(m.to_json())
        self.assertLessEqual(len(loaded._observed_in_cycle), 1)
        self.assertLessEqual(len(loaded._decayed_cycles), 1)


class TestRevisionHistoryCap(unittest.TestCase):
    def test_over_cap_keeps_newest_1000(self):
        rev = SimulationRevision()
        for i in range(MAX_REVISION_HISTORY + 500):
            rev.observe("press", "door_open", True, True, cycle=i)
        self.assertEqual(len(rev.history), MAX_REVISION_HISTORY)
        self.assertEqual(rev.history[0]["cycle"], 500)
        self.assertEqual(rev.history[-1]["cycle"], 1499)

    def test_under_cap_untouched(self):
        rev = SimulationRevision()
        for i in range(10):
            rev.observe("press", "door_open", True, True, cycle=i)
        self.assertEqual(len(rev.history), 10)
        self.assertEqual(rev.history[0]["cycle"], 0)

    def test_oversized_save_prunes_on_load(self):
        rev = SimulationRevision()
        for i in range(MAX_REVISION_HISTORY + 500):
            rev.observe("press", "door_open", True, True, cycle=i)
        data = rev.to_json()
        data["history"] = [
            dict(r, cycle=i)
            for i, r in enumerate(data["history"])]
        # Simulate a pre-fix save: 1,500 records, oldest first.
        long_history = ([{"action": "press", "effect": "door_open",
                          "variant": None, "expected": True, "actual": True,
                          "observed": 1.0, "prior": 1.0, "adjusted": 1.0,
                          "moved": 0.0, "cycle": i}
                         for i in range(MAX_REVISION_HISTORY + 500)])
        data["history"] = long_history
        loaded = SimulationRevision.from_json(data)
        self.assertEqual(len(loaded.history), MAX_REVISION_HISTORY)
        self.assertEqual(loaded.history[0]["cycle"], 500)
        self.assertEqual(loaded.history[-1]["cycle"], 1499)


class TestW7StoresBoundedInSoak(unittest.TestCase):
    def test_2000_cycle_run_keeps_w7_stores_bounded(self):
        # Fast version of the soak assertion: the 10,000-cycle soak
        # is the proof; this pins the regression in the suite.
        record = Soak(target=2000).run()
        samples = record["samples"]
        self.assertGreaterEqual(record["cycles_run"], 2000)
        self.assertLessEqual(
            max(s["goal_thoughts"] for s in samples), 2,
            "cached goal Thoughts accumulated across goals")
        self.assertLessEqual(
            max(s["decay_bookkeeping"] for s in samples), 2,
            "world-model decay bookkeeping accumulated across cycles")
        self.assertLessEqual(
            max(s["revision_history"] for s in samples),
            MAX_REVISION_HISTORY,
            "revision history exceeded its cap during the run")
        for name in ("goal_thoughts", "decay_bookkeeping"):
            self.assertTrue(record["stores"][name]["bounded"],
                            f"{name} not bounded in soak verdict")
        # revision_history is NOT asserted on its slope verdict here:
        # the cap is 1,000 but the trail fills slowly (~0.6/cycle),
        # so in a 2,000-cycle run the slope window still contains the
        # rising phase and the verdict is "not bounded" even though
        # the store cannot exceed its cap (max <= 1,000, asserted
        # above). The slope verdict is meaningful only once the store
        # has reached its cap — i.e. in the 10,000-cycle soak, whose
        # last-5,000-cycle window is flat at 1,000.


if __name__ == "__main__":
    unittest.main()
