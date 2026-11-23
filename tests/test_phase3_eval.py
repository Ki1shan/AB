"""Phase 3 tests: evaluation harness and knowledge-disconnected experiment.

The harness tests run the real harness / experiment as subprocesses
with a fast subset, so the suite stays quick; the full harness run is
a deliverable, not a test.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.disconnected import ChamberWorld, WORLD_A, WORLD_B  # noqa: E402


class TestChamberWorldMechanics(unittest.TestCase):
    """Pin the invented world's hidden rules (unit level, no brain)."""

    def test_decoy_never_works(self):
        env = ChamberWorld(WORLD_A)
        env.s["at"] = "green"
        ok, _ = env.act("chant", {})
        self.assertFalse(ok)

    def test_fragile_object_works_once_then_crumbles(self):
        env = ChamberWorld(WORLD_A)
        env.s["at"] = "green"
        self.assertTrue(env.act("take_zorblat", {})[0])
        ok, info = env.act("use_zorblat", {})
        self.assertTrue(ok)
        self.assertIn("crumbles", info)
        ok, info = env.act("use_zorblat", {})
        self.assertFalse(ok)
        self.assertIn("dust", info)
        # No intact fragile object remains to take again.
        env.s["has_zorblat"] = False
        self.assertFalse(env.act("take_zorblat", {})[0])

    def test_durable_object_always_works(self):
        env = ChamberWorld(WORLD_A)
        env.s["at"] = "green"
        env.act("take_quivern", {})
        self.assertTrue(env.act("use_quivern", {})[0])
        env.s["door_open"] = False
        self.assertTrue(env.act("use_quivern", {})[0])

    def test_door_resets_exactly_once(self):
        env = ChamberWorld(WORLD_A)
        env.s["at"] = "green"
        env.s["door_open"] = True
        self.assertTrue(env.observe()["door_open"])    # 1st observe: still open
        self.assertFalse(env.observe()["door_open"])   # 2nd observe: reset
        env.s["door_open"] = True
        self.assertTrue(env.observe()["door_open"])    # no second reset
        self.assertTrue(env.observe()["door_open"])

    def test_door_not_visible_from_start_room(self):
        env = ChamberWorld(WORLD_A)
        self.assertNotIn("door_open", env.observe())
        env.s["at"] = "green"
        self.assertIn("door_open", env.observe())

    def test_world_b_is_a_renamed_copy(self):
        self.assertEqual(WORLD_B["rooms"], ("amber", "violet", "crimson"))
        self.assertNotEqual(WORLD_A["fragile"], WORLD_B["fragile"])
        env = ChamberWorld(WORLD_B)
        env.s["at"] = "violet"
        self.assertFalse(env.act("hum", {})[0])


class TestDisconnectedStandalone(unittest.TestCase):
    def test_standalone_run_produces_record_with_success_or_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "disconnected.json")
            proc = subprocess.run(
                [sys.executable, "-m", "ab.eval.disconnected",
                 "--record", record_path],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
            # Exit 0 means a full record was written, even if the brain
            # failed the task; non-zero is a harness error only.
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record_path, encoding="utf-8") as fh:
                rec = json.load(fh)
        self.assertIn("world_a", rec)
        self.assertIn("world_b_fresh", rec)
        self.assertIn("world_b_continued", rec)
        self.assertIn("behaviours", rec)
        self.assertEqual(len(rec["behaviours"]), 14)
        self.assertIn("transfer", rec)
        self.assertIn("failures", rec)          # list, possibly empty
        self.assertIsInstance(rec["world_a"]["completed"], bool)
        self.assertIsInstance(rec["world_a"]["failed_attempts"], int)
        # The task outcome itself is recorded, not asserted as a pass:
        # a failed task is a recorded result, not a crash.
        self.assertIn(rec["world_a"]["stop_reason"],
                      ("done", "impossible", "stalled", "budget_exhausted",
                       "no_feasible_goal", "interrupted"))


class TestHarnessFast(unittest.TestCase):
    def test_harness_fast_produces_record_with_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "record.json")
            summary_path = os.path.join(tmp, "summary.txt")
            # Phase 6 SS14.8 quirk fix: every record the harness
            # writes goes to an explicit temp path, so a fast-mode
            # run inside the test suite never overwrites a canonical
            # docs/ record with a fast-mode run.
            proc = subprocess.run(
                [sys.executable, "-m", "ab.eval.harness", "--fast",
                 "--record", record_path, "--summary", summary_path,
                 "--record5", os.path.join(tmp, "record5.json"),
                 "--summary5", os.path.join(tmp, "summary5.txt"),
                 "--record6", os.path.join(tmp, "record6.json"),
                 "--summary6", os.path.join(tmp, "summary6.txt"),
                 "--record7", os.path.join(tmp, "record7.json"),
                 "--summary7", os.path.join(tmp, "summary7.txt"),
                 "--record8", os.path.join(tmp, "record8.json"),
                 "--summary8", os.path.join(tmp, "summary8.txt"),
                 "--record9", os.path.join(tmp, "record9.json"),
                 "--summary9", os.path.join(tmp, "summary9.txt"),
                 "--record10", os.path.join(tmp, "record10.json"),
                 "--summary10", os.path.join(tmp, "summary10.txt")],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            for field in ("timestamp", "mode", "tests", "demo", "disconnected",
                          "baselines", "resources", "repo_state"):
                self.assertIn(field, rec, f"missing record field {field}")
            self.assertEqual(rec["mode"], "fast")
            self.assertIsInstance(rec["tests"]["ran"], int)
            self.assertGreater(rec["tests"]["ran"], 0)
            self.assertEqual(rec["demo"]["exit_code"], 0)
            self.assertEqual(rec["demo"]["outcome"], "done")
            disc = rec["disconnected"]["result"]
            self.assertIsNotNone(disc)
            self.assertIsInstance(disc["world_a"]["completed"], bool)
            self.assertIn("treasure", rec["baselines"])
            for variant in ("full", "no_learning", "non_integrated"):
                self.assertIn(variant, rec["baselines"]["treasure"])
                self.assertIn("cycles", rec["baselines"]["treasure"][variant])
            self.assertIn("peak_child_rss_kb", rec["resources"])
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, encoding="utf-8") as fh:
                self.assertIn("Phase 3 evaluation record", fh.read())


if __name__ == "__main__":
    unittest.main()
