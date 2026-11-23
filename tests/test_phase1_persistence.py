"""Phase 1, Workstreams 5 and 6: deterministic identity, versioned
persistence, and the goal-suspension foundation.

Acceptance:
- Proposed rule ids are identical across separate processes (different
  PYTHONHASHSEED values).
- Full save/load round-trip preserves LTM, rules, WM, goals, active plan
  + progress, planner stats, attention seen-set, state, history and events.
- v0.1 folders (memory.json + rules.json only) still load.
- Corrupt optional files are skipped with a warning, not a crash;
  an incompatible meta version raises.
- A suspended goal round-trips through save/load with plan, position,
  history, goal-start state and reason intact. (Resumption behaviour
  itself is Phase 2 and is NOT claimed here.)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.core.learning import Episode, stable_rule_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_in_subprocess(code: str, hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"subprocess failed: {out.stderr}"
    return out.stdout.strip()


class TestDeterministicIds(unittest.TestCase):
    def test_stable_rule_id_is_deterministic(self):
        a = stable_rule_id("pick_lock", {"at": "lab"})
        b = stable_rule_id("pick_lock", {"at": "lab"})
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("avoid_pick_lock_"))
        self.assertNotEqual(a, stable_rule_id("pick_lock", {"at": "hall"}))
        self.assertNotEqual(a, stable_rule_id("unlock", {"at": "lab"}))

    def test_proposed_rule_id_identical_across_processes(self):
        code = (
            "from ab.brain import Brain\n"
            "from ab.body.environment import room_world_actions\n"
            "from ab.core.learning import Episode\n"
            "b = Brain(room_world_actions())\n"
            "for i in range(3):\n"
            "    b.learner.record(Episode(i, 'pick_lock', {'at': 'lab'}, False))\n"
            "print(b.learner.propose_rules()[0].id)\n"
        )
        id_a = _run_in_subprocess(code, "0")
        id_b = _run_in_subprocess(code, "12345")
        self.assertEqual(id_a, id_b)
        self.assertTrue(id_a.startswith("avoid_pick_lock_"))


class TestFullPersistenceRoundTrip(unittest.TestCase):
    def _mid_run_brain(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.teach_fact("cat", "is_a", "mammal")
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select goal
        b.step(env)   # make plan
        b.step(env)   # execute go_lab_from_hall
        return env, b

    def test_round_trip_preserves_the_running_mind(self):
        env, b = self._mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        b2 = Brain(granted={"compute", "environment"})
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertEqual(b2.cycle, b.cycle)
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")
        self.assertEqual(b2.goals.active().description, "get the treasure")
        self.assertEqual([a.name for a in b2.plan], [a.name for a in b.plan])
        self.assertEqual(b2.planner.stats, b.planner.stats)
        self.assertEqual([t.content for t in b2.wm.contents()],
                         [t.content for t in b.wm.contents()])
        self.assertEqual(b2.attention._seen, b.attention._seen)
        self.assertEqual(b2.world, b.world)
        # Phase 3 §2.3 (mandated): Brain.history was removed; the event
        # log is the only history, and it round-trips (asserted below).
        self.assertFalse(hasattr(b2, "history"))
        self.assertEqual(b2.events, b.events)
        self.assertIsNotNone(b2.rules.get("select_goal"))
        # And the loaded brain can keep going in the same world.
        b2.run(env, 60)
        self.assertTrue(env.s["has_treasure"])
        self.assertEqual(b2.goals.all()[0].status, "done")

    def test_v01_folder_still_loads(self):
        b = Brain()
        b.teach_fact("a", "is_a", "b")
        d = tempfile.mkdtemp()
        b.ltm.save(os.path.join(d, "memory.json"))
        b.rules.save_json(os.path.join(d, "rules.json"))
        b2 = Brain()
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertEqual(b2.ask("a", "is_a").best[0], "b")

    def test_corrupt_optional_file_is_skipped_with_warning(self):
        _, b = self._mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "wm.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        b2 = Brain()
        b2.load(d)
        self.assertTrue(any("wm.json" in w for w in b2.load_warnings))
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")  # LTM intact

    def test_incompatible_version_raises(self):
        _, b = self._mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"version": 99, "cycle": 0}, fh)
        with self.assertRaises(ValueError):
            Brain().load(d)


class TestGoalSuspensionFoundation(unittest.TestCase):
    def test_suspend_serialize_restore_round_trip(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        b.step(env)   # plan: 4 steps
        b.step(env)   # execute go_lab_from_hall (1 of 4 done)
        rec = b.suspend_active_goal("interrupted by test")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.original_status, "active")
        self.assertEqual(rec.plan_position, 1)
        self.assertEqual([a["name"] for a in rec.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual(rec.reason, "interrupted by test")
        self.assertEqual(rec.goal_start, b.goal_start)
        self.assertEqual(b.goals.active(), None)   # goal is pending again
        d = tempfile.mkdtemp()
        b.save(d)
        b2 = Brain()
        b2.load(d)
        rec2 = b2.goals.restore_suspended(rec.goal_id)
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2.plan_position, 1)
        self.assertEqual([a["name"] for a in rec2.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual(rec2.reason, "interrupted by test")
        self.assertEqual(rec2.goal_start, rec.goal_start)

    def test_suspend_with_no_active_goal_returns_none(self):
        self.assertIsNone(Brain().suspend_active_goal("nothing to suspend"))


if __name__ == "__main__":
    unittest.main()
