"""Phase 4 tests: the three discriminating held-out tasks and their
harness integration (behavioural tests).

Each task is asserted on behaviour -- a derivation with a proof and an
exact confidence, a contradiction event plus disputed facts plus a
research goal, a hypothesis with a test and a recorded outcome -- never
on printed output. The tasks use unfamiliar names/relations the brain
was never taught (velkra/torvin/sarnak/kelmor, lyssan, vapour); the
task defines the world, the brain must reason its way through it.

World A/B behaviour is pinned by asserting the disconnected
experiment's own record fields (not by re-implementing the world).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.disconnected import run_experiment  # noqa: E402
from ab.eval.reasoning_tasks import (  # noqa: E402
    DEDUCTION_EXPECTED_CONFIDENCE, make_variant_factories,
    run_contradiction_task, run_deduction_task, run_hypothesis_task)


def _factories():
    return make_variant_factories()


class TestDeductionTask(unittest.TestCase):
    def test_full_brain_derives_3hop_with_exact_confidence(self):
        res = run_deduction_task(_factories()["full"])
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["derived"])
        self.assertTrue(res["legs"]["proof_has_3_steps"])
        self.assertTrue(res["legs"]["proof_path_is_is_a_is_a_located_in"])
        self.assertTrue(res["legs"]["confidence_exact_0_9025"])
        deriv = res["evidence"]["derivation"]
        self.assertEqual(
            (deriv["subject"], deriv["relation"], deriv["value"]),
            ("velkra", "located_in", "kelmor"))
        self.assertAlmostEqual(deriv["confidence"],
                               DEDUCTION_EXPECTED_CONFIDENCE, places=9)
        self.assertEqual(deriv["proof_steps"], 3)
        self.assertEqual(deriv["proof_path"],
                         ["is_a", "is_a", "located_in"])

    def test_non_integrated_brain_cannot_derive(self):
        # Structural discrimination: with _reason_from_wm disabled the
        # brain does no in-cycle inference at all, so no derivation
        # can appear -- not even the single-relation ones.
        res = run_deduction_task(_factories()["non_integrated"])
        self.assertFalse(res["passed"])
        self.assertFalse(res["legs"]["derived"])
        self.assertEqual(res["evidence"]["last_inferences_count"], 0)

    def test_no_learning_brain_derives(self):
        # Deduction needs no learning: the no-learning variant keeps
        # in-cycle reasoning, so it must derive the chain too.
        res = run_deduction_task(_factories()["no_learning"])
        self.assertTrue(res["passed"], res["detail"])


class TestContradictionTask(unittest.TestCase):
    def test_full_brain_detects_contradiction(self):
        res = run_contradiction_task(_factories()["full"])
        self.assertTrue(res["passed"], res["detail"])
        self.assertEqual(res["evidence"]["disputed_values"],
                         ["100c", "90c"])
        self.assertIsNotNone(res["evidence"]["contradiction_event"])
        self.assertIsNotNone(res["evidence"]["research_goal"])
        self.assertEqual(
            [v for v, _, _ in res["evidence"]["ask_answers"]],
            ["100c", "90c"])
        self.assertIn("disagree", res["evidence"]["ask_message"])

    def test_non_integrated_brain_marks_disputed_but_detects_nothing(self):
        # The LTM conflict path still marks both claims disputed at
        # storage level, but with in-cycle reasoning disabled there is
        # no contradiction event and no research goal.
        res = run_contradiction_task(_factories()["non_integrated"])
        self.assertFalse(res["passed"])
        self.assertTrue(res["legs"]["both_claims_disputed_in_ltm"])
        self.assertFalse(res["legs"]["contradiction_event_in_cycle"])
        self.assertFalse(res["legs"]["research_goal_created"])
        self.assertTrue(res["legs"]["neither_side_silently_picked"])

    def test_no_learning_brain_detects(self):
        # Contradiction detection needs no learning either.
        res = run_contradiction_task(_factories()["no_learning"])
        self.assertTrue(res["passed"], res["detail"])


class TestHypothesisTask(unittest.TestCase):
    def test_full_brain_generates_and_tests_hypothesis(self):
        # Phase 5 pin update (coordinator, following the Phase 2 §9 /
        # Phase 3 §8 / Phase 4 §14.1 precedent -- same intent, new
        # intended semantics, not weakened): the Phase 4 criterion
        # (hypothesis + test + recorded outcome) now ALSO requires the
        # goal to complete (design §4.2). The task's goal is completion,
        # and completion was previously blocked by the stall bug the
        # phase fixes (a model revision must clear the pair-bans keyed
        # to the old model and reset the stall window, so the brain
        # recovers by warming the coil). The new leg is asserted
        # explicitly below, alongside the unchanged Phase 4 legs.
        res = run_hypothesis_task(_factories()["full"])
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["surprise_occurred"])
        self.assertGreater(res["evidence"]["prediction_violated_events"], 0)
        hyp = res["evidence"]["hypothesis"]
        self.assertIsNotNone(hyp)
        self.assertIsNotNone(hyp.get("test"))
        # The outcome is asserted on the recorded outcome EVENT, not on
        # the hypothesis record's status field: the event log is where
        # the outcome is durably recorded (hypothesis_corroborated /
        # hypothesis_falsified). The record's own status field is
        # clobbered back to "open" by the next cycle's re-abduction
        # (worker-2 bug, see docs/phase4-progress.md Gap 2 remainder),
        # so asserting on it would test the clobber, not the outcome.
        outcome = res["evidence"]["hypothesis_outcome_event"]
        self.assertIsNotNone(outcome)
        self.assertIn(outcome["kind"],
                      ("hypothesis_corroborated", "hypothesis_falsified"))
        self.assertTrue(res["legs"]["test_outcome_recorded"])
        # Phase 5 goal leg: the brain must finish heating the vapour,
        # not just diagnose why it stayed cold.
        self.assertTrue(res["legs"]["goal_completed"], res["detail"])
        self.assertEqual(res["evidence"]["goal_status"], "done")

    def test_non_integrated_brain_generates_no_hypothesis(self):
        # The surprise still happens, but with in-cycle reasoning
        # disabled no hypothesis is ever generated -- that is the
        # discrimination. (Phase 5 pin update, coordinator, following
        # the Phase 2 §9 / Phase 3 §8 / Phase 4 §14.1 precedent --
        # same intent, new intended semantics, not weakened): goal
        # completion is NO LONGER a discriminating leg. The Phase 5
        # recovery (model revision clears pair-bans keyed to the old
        # model and resets the stall window, plan invalidated on
        # violation) lives in Brain._check_predictions, i.e. at the
        # planner level, not in the reasoning engine -- so the
        # non-integrated variant now completes the goal too. The
        # variant still FAILS the task overall, via the hypothesis
        # leg, which is the discrimination this test pins.
        res = run_hypothesis_task(_factories()["non_integrated"])
        self.assertFalse(res["passed"])
        self.assertTrue(res["legs"]["surprise_occurred"])
        self.assertFalse(res["legs"]["hypothesis_generated"])
        self.assertIsNone(res["evidence"]["hypothesis"])
        self.assertTrue(res["legs"]["goal_completed"])

    def test_no_learning_variant_records_no_episodes(self):
        # Mechanism pin (no pass/fail claim on this variant for this
        # task): abduction reasons over learner episodes, and the
        # no-learning variant records none -- so its hypothesis result
        # is whatever the record says, with the reason visible.
        res = run_hypothesis_task(_factories()["no_learning"])
        self.assertIn("passed", res)
        self.assertIn("legs", res)
        self.assertEqual(res["evidence"]["learner_episodes"], 0)
        full = run_hypothesis_task(_factories()["full"])
        self.assertGreater(full["evidence"]["learner_episodes"], 0)


class TestHarnessRecord(unittest.TestCase):
    def test_fast_harness_record_has_discriminating_tasks(self):
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
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=240)
            # The --fast subset suite passes, so the harness exits 0;
            # a failed discriminating task alone must not fail it.
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertIn("discriminating_tasks", rec)
            section = rec["discriminating_tasks"]
            self.assertIn("mechanism", section)
            self.assertIn("No winner", section["mechanism"])
            tasks = section["tasks"]
            self.assertEqual(sorted(tasks),
                             ["contradiction", "deduction_3hop",
                              "hypothesis_abduction"])
            for task_id, task in tasks.items():
                self.assertIn("criterion", task, task_id)
                for variant in ("full", "no_learning", "non_integrated"):
                    res = task["variants"][variant]
                    self.assertIn("passed", res, f"{task_id}/{variant}")
                    self.assertIsInstance(res["passed"], bool)
                    self.assertIn("cycles", res, f"{task_id}/{variant}")
                    self.assertIn("legs", res, f"{task_id}/{variant}")
                    self.assertIn("evidence", res, f"{task_id}/{variant}")
                    self.assertIn("detail", res, f"{task_id}/{variant}")
            with open(summary_path, encoding="utf-8") as fh:
                summary = fh.read()
            self.assertIn("Discriminating tasks", summary)
            for task_id in tasks:
                self.assertIn(task_id, summary)

    def test_harness_fails_visibly_when_suite_fails(self):
        # The harness must not confuse a successful report-writing
        # process with a successful evaluation: when the suite fails,
        # the record is still written, the summary says SUITE FAIL,
        # and the exit code is non-zero. Checked at the summary-text
        # level (running a real failing suite here would take the
        # full-suite time and depend on its failure mode).
        from ab.eval.harness import _summary_text
        record = {
            "timestamp": "2026-11-21T00:00:00+00:00", "mode": "fast",
            "tests": {"ran": 10, "failures": 1, "errors": 0, "skipped": 0,
                      "ok": False, "scope": "test", "duration_s": 0.1,
                      "exit_code": 1},
            "demo": {"exit_code": 0, "outcome": "done", "duration_s": 0.1},
            "disconnected": {"result": None},
            "baselines": {"treasure": {}, "disconnected_world_a": "skipped"},
            "discriminating_tasks": {"tasks": {}},
            "resources": {"total_wall_s": 0.1, "suite_wall_s": 0.1,
                          "demo_wall_s": 0.1, "disconnected_wall_s": 0.0,
                          "peak_child_rss_kb": "not measured"},
            "repo_state": {"vcs": "none", "python_files": 0, "test_files": 0},
        }
        self.assertIn("SUITE FAIL", _summary_text(record))


class TestWorldABUnchanged(unittest.TestCase):
    def test_world_a_still_completes_in_14_cycles_with_2_failed_attempts(self):
        # Assert the disconnected experiment's own record fields; the
        # world itself is not re-implemented here.
        # Phase 4 pin update (coordinator, following the Phase 2 §9 /
        # Phase 3 §8 precedent — same intent, new intended semantics,
        # not weakened): Phase 3 pinned 16 cycles / 3 failed attempts.
        # The mandated Phase 4 whole-sequence ban check means the brain
        # no longer re-fails the just-banned use_zorblat via the adopted
        # alternative, saving the wasted re-failure cycle and one more
        # cycle: world A now completes in 14 cycles / 2 failed attempts.
        # Phase 5 pin update (coordinator, following the Phase 2 §9 /
        # Phase 3 §8 / Phase 4 §14.1 precedent — same intent, new
        # intended semantics, not weakened): world A still completes in
        # 14 cycles / 2 failed attempts, but transfer is now TRUE.
        # The Phase 5 pattern library (structural P-A/P-B patterns
        # learned in A, applied by the planner in B) makes the
        # continued brain need fewer failed attempts in B (12 cycles /
        # 1 failed attempt vs fresh B 14 / 2). That is Phase 5's
        # intended outcome, measured in docs/phase5-transfer-record.json.
        rec = run_experiment()
        self.assertIn("world_a", rec)
        a = rec["world_a"]
        self.assertTrue(a["completed"])
        self.assertEqual(a["stop_reason"], "done")
        self.assertEqual(a["cycles"], 14)
        self.assertEqual(a["failed_attempts"], 2)
        self.assertEqual(len(rec["behaviours"]), 14)
        self.assertTrue(rec["transfer"]["transferred"])

    def test_disconnected_cli_disciminating_flag_lists_tasks(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ab.eval.disconnected", "--discriminating"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for task_id in ("deduction_3hop", "contradiction",
                        "hypothesis_abduction"):
            self.assertIn(task_id, proc.stdout)
        self.assertIn("No winner is declared", proc.stdout)


if __name__ == "__main__":
    unittest.main()
