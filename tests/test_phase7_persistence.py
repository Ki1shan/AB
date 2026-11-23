"""Phase 7 execution persistence (D7-3 / D7-8) tests.

The execution session + ledger live in ``execution.json`` (schema
v1) under persistence v5. Tests:

* v5 round-trip: a mid-run save keeps the session (same task id,
  interrupted after load, same steps executed, same ledger).
* §3.7 combined save / restart / resume: save mid-execution with a
  live session + ledger + plan position, load into a fresh Brain,
  resume_goal, complete — session, ledger and position survive and
  the task completes.
* v4 backwards compatibility: a v4 save (no execution.json) loads
  with a warning, empty execution state, and the brain still works.
* corrupt execution.json: skipped with a warning, other state intact.
* v0.1 folder: silent, empty execution state.
* secret redaction at the save level: a secret tool argument never
  reaches execution.json (or any event) in the clear.
"""

import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.body.tools import ToolSpec
from ab.planning.planner import Action
from tests.test_phase7_execution import _bench_brain, BenchEnv

SECRET = "sk-test-SECRET-VALUE-12345"


def _mid_run_bench_brain():
    """Bench brain two steps into the assemble task (go_bench and
    take_part executed; cut/drill/measure/assemble remain)."""
    env = BenchEnv()
    b = _bench_brain()
    g = b.add_goal("assemble the part", {"assembled": True})
    b.step(env)   # select goal
    b.step(env)   # make plan (session opens)
    b.step(env)   # execute go_bench
    b.step(env)   # execute take_part
    return env, b, g


def _read_meta(folder):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as fh:
        return json.load(fh)


class TestExecutionPersistenceRoundTrip(unittest.TestCase):
    def test_v5_round_trip_preserves_session_and_ledger(self):
        env, b, g = _mid_run_bench_brain()
        session = b.execution_monitor.session
        self.assertIsNotNone(session)
        self.assertEqual(session.status, "running")
        self.assertEqual(session.steps_executed, 2)
        self.assertEqual(len(b.execution_monitor.ledger.records), 2)
        d = tempfile.mkdtemp()
        b.save(d)
        meta = _read_meta(d)
        # Phase 11 pin update (§9 precedent): persistence v7.
        self.assertEqual(meta["version"], 7)
        self.assertIn("execution.json", meta["files"])
        self.assertEqual(meta["files"]["execution.json"], 1)
        self.assertIn("execution.json", os.listdir(d))

        b2 = _bench_brain()
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        session2 = b2.execution_monitor.session
        self.assertIsNotNone(session2)
        # A session saved "running" loads as "interrupted" (the task
        # did not finish before the save); the task id is unchanged.
        self.assertEqual(session2.task_id, session.task_id)
        self.assertEqual(session2.status, "interrupted")
        self.assertEqual(session2.stop_reason, "interrupted by load")
        self.assertEqual(session2.steps_executed, 2)
        self.assertEqual(session2.goal_id, g.id)
        self.assertEqual(
            [r["action"] for r in b2.execution_monitor.ledger.records],
            [r["action"] for r in b.execution_monitor.ledger.records])
        self.assertEqual(
            [r["outcome"] for r in b2.execution_monitor.ledger.records],
            ["ok", "ok"])

    def test_save_restart_resume_completes_task(self):
        # §3.7: save mid-execution (live session + ledger + plan
        # position), load into a fresh Brain, resume_goal, complete.
        env, b, g = _mid_run_bench_brain()
        task_id = b.execution_monitor.session.task_id
        rec = b.suspend_active_goal("save test")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.plan_position, 2)
        self.assertEqual(b.execution_monitor.session.status, "interrupted")
        d = tempfile.mkdtemp()
        b.save(d)

        b2 = _bench_brain()
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        session2 = b2.execution_monitor.session
        self.assertEqual(session2.task_id, task_id)
        self.assertEqual(session2.status, "interrupted")
        self.assertEqual(session2.steps_executed, 2)
        self.assertEqual(len(b2.execution_monitor.ledger.records), 2)
        # The suspended goal and its plan position survived.
        rec2 = b2.goals.suspended(g.id)
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2.plan_position, 2)

        self.assertTrue(b2.resume_goal(g.id))
        self.assertEqual(b2.execution_monitor.session.status, "running")
        self.assertEqual(b2.execution_monitor.session.task_id, task_id)
        self.assertEqual(b2.run(env, 60), "done")
        self.assertTrue(env.s["assembled"])
        self.assertEqual(b2.execution_monitor.session.status, "done")
        self.assertEqual(len(b2.execution_monitor.ledger.records), 6)
        self.assertEqual(
            [r["action"] for r in b2.execution_monitor.ledger.records],
            ["go_bench", "take_part", "cut_part", "drill_part",
             "measure_part", "assemble"])
        self.assertTrue(all(r["outcome"] == "ok" for r in
                            b2.execution_monitor.ledger.records))


class TestExecutionPersistenceBackwardsCompat(unittest.TestCase):
    def _v4_folder(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        b.step(env)   # make plan
        b.step(env)   # execute go_lab_from_hall
        d = tempfile.mkdtemp()
        b.save(d)
        # Downgrade to a v4 save: drop execution.json and its
        # manifest entry, set the version back to 4.
        os.remove(os.path.join(d, "execution.json"))
        meta = _read_meta(d)
        meta["version"] = 4
        del meta["files"]["execution.json"]
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        return env, d

    def test_v4_save_loads_with_warning_and_empty_execution_state(self):
        env, d = self._v4_folder()
        b2 = Brain(room_world_actions(), granted={"compute", "environment"})
        b2.load(d)
        self.assertTrue(any("execution.json" in w for w in b2.load_warnings))
        self.assertIsNone(b2.execution_monitor.session)
        self.assertEqual(b2.execution_monitor.ledger.records, [])
        # The brain still works: it completes the treasure goal.
        self.assertEqual(b2.run(env, 60), "done")
        self.assertTrue(env.s["has_treasure"])

    def test_corrupt_execution_json_skipped_with_warning(self):
        env, b, g = _mid_run_bench_brain()
        b.teach_fact("cat", "is_a", "mammal")
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "execution.json"), "w",
                  encoding="utf-8") as fh:
            fh.write("{not json")
        b2 = _bench_brain()
        b2.load(d)
        self.assertTrue(any("execution.json" in w for w in b2.load_warnings))
        self.assertIsNone(b2.execution_monitor.session)
        self.assertEqual(b2.execution_monitor.ledger.records, [])
        # Other state is intact.
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")
        self.assertEqual(b2.goals.active().description, "assemble the part")
        self.assertEqual(b2.cycle, b.cycle)

    def test_v01_folder_loads_silently_with_empty_execution_state(self):
        b = Brain()
        b.teach_fact("a", "is_a", "b")
        d = tempfile.mkdtemp()
        b.ltm.save(os.path.join(d, "memory.json"))
        b.rules.save_json(os.path.join(d, "rules.json"))
        b2 = Brain()
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertEqual(b2.ask("a", "is_a").best[0], "b")
        self.assertIsNone(b2.execution_monitor.session)
        self.assertEqual(b2.execution_monitor.ledger.records, [])


class TestExecutionSecretRedaction(unittest.TestCase):
    def test_secret_tool_arg_never_reaches_execution_json(self):
        # The cut tool takes a secret api_key argument (the value is
        # invented for this test and appears nowhere else). After a
        # full run + save, the secret must appear nowhere in
        # execution.json (the ledger holds "[REDACTED]" instead) and
        # in no event.
        def _cut_with_key(**kwargs):
            return True

        acts = [
            Action("go_bench", {"at": "store"}, {"at": "bench"},
                   args={"_do": "go_bench"}),
            Action("take_part", {"at": "bench", "has_part": False},
                   {"has_part": True}, args={"_do": "take_part"}),
            Action("cut_part", {"has_part": True}, {"cut_done": True},
                   tool="cut", store_as="cut_done",
                   args={"api_key": SECRET}),
            Action("drill_part", {"has_part": True}, {"drill_done": True},
                   tool="drill", store_as="drill_done"),
            Action("measure_part", {"cut_done": True, "drill_done": True},
                   {"measured": True}, tool="measure", store_as="measured"),
            Action("assemble", {"cut_done": True, "drill_done": True,
                                "measured": True},
                   {"assembled": True}, args={"_do": "assemble"}),
        ]
        env = BenchEnv()
        b = Brain(acts, granted={"compute", "environment"})
        b.tools.register(ToolSpec(name="cut", args=[{"name": "api_key",
                                                     "type": "str"}],
                                  return_type="bool", timeout=5.0,
                                  required_grant="compute", version=1),
                         _cut_with_key)
        from tests.test_phase7_execution import _drill, _measure
        b.tools.register(ToolSpec(name="drill", args=[],
                                  return_type="bool", timeout=5.0,
                                  required_grant="compute", version=1),
                         _drill)
        b.tools.register(ToolSpec(name="measure", args=[],
                                  return_type="bool", timeout=5.0,
                                  required_grant="compute", version=1),
                         _measure)
        b.add_goal("assemble the part", {"assembled": True})
        self.assertEqual(b.run(env, 60), "done")
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "execution.json"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertNotIn(SECRET, text)
        self.assertIn("[REDACTED]", text)
        cut_records = [r for r in b.execution_monitor.ledger.records
                       if r["action"] == "cut_part"]
        self.assertEqual(len(cut_records), 1)
        self.assertEqual(cut_records[0]["args"],
                         {"api_key": "[REDACTED]"})
        for e in b.events:
            self.assertNotIn(SECRET, e.get("detail", ""))
            self.assertNotIn(SECRET, e.get("text", ""))


if __name__ == "__main__":
    unittest.main()
