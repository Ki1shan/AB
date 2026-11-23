"""Phase 5 tests: the transfer experiment (design SS3.3 / SS6.1).

The measurement test asserts the record structure and that the
``transferred`` boolean equals the measured comparison -- it tests
the MEASUREMENT, and passes whether or not transfer happened. The
acceptance test asserts the criterion itself (continued B strictly
fewer failed attempts than fresh B): if the pattern mechanism does
not deliver, that test FAILS honestly. It is not weakened, and no
number is tuned to force it.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.disconnected import WORLD_A, _run_with_brain  # noqa: E402
from ab.eval.transfer import (  # noqa: E402
    _WORLD_NAMES,
    prepare_brain_for_new_world,
    run_transfer_experiment,
)


class TestTransferMeasurement(unittest.TestCase):
    def test_experiment_records_honest_numbers(self):
        rec = run_transfer_experiment()
        for key in ("world_a_fresh", "world_b_fresh", "world_b_continued",
                    "transferred", "criterion", "mechanism",
                    "entry_protocol", "patterns_learned_in_a",
                    "comparison_vs_fresh_a"):
            self.assertIn(key, rec, key)
        a, b, c = (rec["world_a_fresh"], rec["world_b_fresh"],
                   rec["world_b_continued"])
        for run in (a, b, c):
            for key in ("completed", "goal_status", "stop_reason", "cycles",
                        "failed_attempts", "failure_log", "learner_episodes",
                        "planner_bans", "env_log"):
                self.assertIn(key, run, key)
        # Both fresh runs must complete: a failed fresh run would make
        # the comparison meaningless.
        self.assertTrue(a["completed"], a)
        self.assertTrue(b["completed"], b)
        # The boolean is exactly the measured comparison -- this tests
        # the measurement, not the outcome.
        expected = bool(c["completed"] and b["completed"]
                        and c["failed_attempts"] < b["failed_attempts"])
        self.assertEqual(rec["transferred"], expected)
        mech = rec["mechanism"]
        for key in ("patterns_present_before_b", "pattern_applied_events_during_b",
                    "pattern_applied_count_during_b",
                    "decoy_tried_in_continued_b",
                    "fragile_tool_retried_after_reset"):
            self.assertIn(key, mech, key)

    def test_transfer_acceptance_criterion(self):
        # Acceptance criterion (design SS6.1): the continued brain
        # completes B and needs strictly fewer failed attempts there
        # than the fresh brain. On the Phase 4 baseline this FAILS
        # honestly (2 vs 2 -- learning is bound to literal names and
        # no pattern library exists yet); it passes only when the
        # Phase 5 pattern mechanism delivers. Do not weaken it.
        rec = run_transfer_experiment()
        c, b = rec["world_b_continued"], rec["world_b_fresh"]
        self.assertTrue(c["completed"], c)
        self.assertLess(c["failed_attempts"], b["failed_attempts"],
                        f"continued B {c['failed_attempts']} failed attempts "
                        f"vs fresh B {b['failed_attempts']}; transfer did "
                        f"not happen: {rec['mechanism']}")

    def test_patterns_learned_in_a_contain_no_world_names(self):
        # A pattern id/name that mentions a world-A action or entity
        # name would be memorisation, not structural transfer.
        rec = run_transfer_experiment()
        snap = rec["patterns_learned_in_a"]
        self.assertIn("ids", snap)
        for pid in snap["ids"]:
            for name in _WORLD_NAMES:
                self.assertNotIn(name, pid.lower(),
                                 f"pattern id {pid!r} mentions world name {name!r}")
        for pname in snap["names"]:
            for name in _WORLD_NAMES:
                self.assertNotIn(name, pname.lower(),
                                 f"pattern name {pname!r} mentions world name {name!r}")

    def test_continued_brain_enters_b_with_patterns_intact_and_transient_cleared(self):
        # Entry protocol from the experiment record itself, then a
        # direct check of the entry helper on a fresh world-A brain:
        # patterns survive the entry, transient world-A observations do not.
        rec = run_transfer_experiment()
        entry = rec["entry_protocol"]
        self.assertTrue(entry["patterns_intact"], entry)
        self.assertTrue(entry["transient_state_cleared"], entry)
        self.assertIn("method", entry)
        # Direct check of the entry helper on a fresh world-A brain:
        # patterns survive the entry, transient world-A observations do not.
        from ab.eval.disconnected import _default_factory
        _, brain = _run_with_brain(WORLD_A, _default_factory, 80)
        self.assertTrue(any(k in brain.state.keys()
                            for k in ("at", "door_open", "has_zorblat")),
                        "world-A brain should hold world-A observations before entry")
        entry2 = prepare_brain_for_new_world(brain)
        self.assertTrue(entry2["patterns_intact"], entry2)
        for key in ("at", "door_open", "has_zorblat", "has_quivern", "has_relic"):
            self.assertNotIn(key, brain.state.keys(),
                             f"transient world-A key {key} survived entry to a new world")
        self.assertTrue(entry2["transient_state_cleared"], entry2)

    def test_transfer_cli_writes_record_and_exits_zero_even_when_not_transferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "transfer.json")
            proc = subprocess.run(
                [sys.executable, "-m", "ab.eval.transfer",
                 "--record", record_path],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
            # Exit 0 when the record was written -- a failed transfer
            # is a recorded result, not a harness crash.
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertIn("transferred", rec)
            self.assertIsInstance(rec["transferred"], bool)
            self.assertIn("world_b_continued", rec)


if __name__ == "__main__":
    unittest.main()
