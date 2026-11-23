"""Phase 5 tests: the world-model acceptance tasks and their harness
integration (design SS6.2-SS6.5 + transfer wrapper).

Each task is asserted per variant through ``run_world_tasks``; the
full variant is expected to pass each task. Two tasks are currently
blocked on Worker 2's Brain integration and FAIL honestly until it
lands -- ``transfer_task`` (no pattern library yet, so continued B
ties fresh B at 2 failed attempts) and ``prediction_revision``
(the next plan still re-predicts the old value until a model
revision clears the pair-bans keyed to it and resets the stall
window, design SS4.2). Those failures are the measurement, not a
reason to weaken the tasks; see the test comments.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.world_tasks import run_world_tasks  # noqa: E402

# One shared run for the whole file: run_world_tasks() runs five
# tasks x three variants (including three transfer experiments), so
# re-running it per test would take minutes. The section is a plain
# record dict; tests only read it.
_SECTION = None


def _section():
    global _SECTION
    if _SECTION is None:
        _SECTION = run_world_tasks()
    return _SECTION


def _full(task_id):
    return _section()["tasks"][task_id]["variants"]["full"]


class TestWorldTasksFullVariant(unittest.TestCase):
    def test_object_tracking_full_passes(self):
        res = _full("object_tracking")
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["goal_completed"])
        self.assertTrue(res["legs"]["history_order_correct"])
        self.assertTrue(res["legs"]["unobserved_interval_survived"])
        self.assertTrue(res["legs"]["identity_preserved"])
        self.assertEqual(res["cycles"], 8)

    def test_prediction_revision_full_passes(self):
        # BLOCKED on Worker 2 (design SS4.2): the revision event fires
        # and the old value is retained, but the next plan is still a
        # bare heat_vapour -- it re-predicts the old value, because no
        # model revision clears the pair-bans keyed to the old model
        # or resets the stall window yet. When that fix lands the
        # next plan warms the coil first and this test passes. The
        # leg is asserted as specified; it is not weakened.
        res = _full("prediction_revision")
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["revision_event_recorded"])
        self.assertTrue(res["legs"]["old_value_retained_in_revisions"])
        self.assertTrue(res["legs"]["next_plan_does_not_repredict_old_value"])

    def test_partial_observability_full_passes(self):
        res = _full("partial_observability")
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["goal_completed"])
        self.assertTrue(res["legs"]["door_unobserved_from_start_room"])
        self.assertTrue(res["legs"]["first_plan_does_not_assume_door_closed"])
        self.assertTrue(res["legs"]["no_plan_assumes_door_closed"])
        self.assertEqual(res["cycles"], 14)

    def test_unknown_vs_false_full_passes(self):
        res = _full("unknown_vs_false")
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["unknown_returns_not_known"])
        self.assertTrue(res["legs"]["false_claim_returns_disputed_not_unknown"])
        self.assertEqual(res["evidence"]["disputed_values"], ["40c", "50c"])


class TestWorldTasksPerVariant(unittest.TestCase):
    def test_transfer_task_reports_measured_numbers_per_variant(self):
        # The transfer task is the measurement, not the outcome: every
        # variant reports the three runs' numbers and the transferred
        # boolean computed from them. (On this baseline all three
        # variants tie at 2 failed attempts, so none transfers yet.)
        section = _section()
        task = section["tasks"]["transfer_task"]
        for name in ("full", "no_learning", "non_integrated"):
            res = task["variants"][name]
            self.assertIn("passed", res, name)
            self.assertIsInstance(res["passed"], bool, name)
            ev = res["evidence"]["transfer_record"]
            self.assertTrue(ev["world_a_fresh"]["completed"], name)
            self.assertTrue(ev["world_b_fresh"]["completed"], name)
            self.assertEqual(
                res["passed"],
                bool(ev["world_b_continued"]["completed"]
                     and ev["world_b_fresh"]["completed"]
                     and ev["world_b_continued"]["failed_attempts"]
                     < ev["world_b_fresh"]["failed_attempts"]), name)

    def test_per_variant_numbers_recorded_for_prediction_revision_and_unknown_vs_false(self):
        # Per-variant numbers are recorded for both tasks, so the
        # record shows which variants separate and which do not.
        # (Phase 5 pin update, coordinator, following the Phase 2 §9 /
        # Phase 3 §8 precedent — same intent, new intended semantics,
        # not weakened): prediction_revision now PASSES for all three
        # variants. The recovery path (model revision on a violated
        # prediction, plan invalidated so the next plan is made from
        # the revised model) lives in Brain._check_predictions, i.e.
        # at the planner level, not in the reasoning engine — so no
        # variant is separated by it. unknown_vs_false still passes
        # for all three (no learning or integration needed for
        # ask/dispute). Neither task separates the variants; the
        # separation lives in transfer_task (full/non-integrated PASS,
        # no-learning FAIL) and in the Phase 4 discriminating tasks.
        section = _section()
        for task_id in ("prediction_revision", "unknown_vs_false"):
            task = section["tasks"][task_id]
            self.assertEqual(sorted(task["variants"]),
                             ["full", "no_learning", "non_integrated"], task_id)
            for name, res in task["variants"].items():
                for key in ("passed", "cycles", "legs", "evidence", "detail"):
                    self.assertIn(key, res, f"{task_id}/{name}")
                self.assertIsInstance(res["passed"], bool, f"{task_id}/{name}")
        self.assertTrue(all(
            section["tasks"][task_id]["variants"][n]["passed"]
            for task_id in ("prediction_revision", "unknown_vs_false")
            for n in ("full", "no_learning", "non_integrated")))

    def test_all_five_tasks_present_with_criteria(self):
        section = _section()
        self.assertEqual(sorted(section["tasks"]),
                         ["object_tracking", "partial_observability",
                          "prediction_revision", "transfer_task",
                          "unknown_vs_false"])
        self.assertIn("No winner", section["mechanism"])
        for task_id, task in section["tasks"].items():
            self.assertIn("criterion", task, task_id)


class TestHarnessPhase5Record(unittest.TestCase):
    def test_fast_harness_writes_phase5_record_with_transfer_and_world_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "record.json")
            summary_path = os.path.join(tmp, "summary.txt")
            record5_path = os.path.join(tmp, "record5.json")
            summary5_path = os.path.join(tmp, "summary5.txt")
            record6_path = os.path.join(tmp, "record6.json")
            summary6_path = os.path.join(tmp, "summary6.txt")
            record7_path = os.path.join(tmp, "record7.json")
            summary7_path = os.path.join(tmp, "summary7.txt")
            proc = subprocess.run(
                [sys.executable, "-m", "ab.eval.harness", "--fast",
                 "--record", record_path, "--summary", summary_path,
                 "--record5", record5_path, "--summary5", summary5_path,
                 "--record6", record6_path, "--summary6", summary6_path,
                 "--record7", record7_path, "--summary7", summary7_path,
                 "--record8", os.path.join(tmp, "record8.json"),
                 "--summary8", os.path.join(tmp, "summary8.txt"),
                 "--record9", os.path.join(tmp, "record9.json"),
                 "--summary9", os.path.join(tmp, "summary9.txt"),
                 "--record10", os.path.join(tmp, "record10.json"),
                 "--summary10", os.path.join(tmp, "summary10.txt")],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
            # The --fast subset suite passes, so the harness exits 0;
            # a failed transfer task alone must not fail it.
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record5_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertIn("transfer", rec)
            self.assertIn("world_tasks", rec)
            self.assertIn("transferred", rec["transfer"])
            self.assertIsInstance(rec["transfer"]["transferred"], bool)
            self.assertEqual(sorted(rec["world_tasks"]["tasks"]),
                             ["object_tracking", "partial_observability",
                              "prediction_revision", "transfer_task",
                              "unknown_vs_false"])
            # The Phase 3 record is still written, with the same
            # Phase 5 sections (one record, two filenames).
            with open(record_path, encoding="utf-8") as fh:
                rec3 = json.load(fh)
            self.assertIn("transfer", rec3)
            self.assertIn("world_tasks", rec3)
            with open(summary5_path, encoding="utf-8") as fh:
                summary = fh.read()
            self.assertIn("Transfer (Phase 5)", summary)
            self.assertIn("World tasks (Phase 5", summary)
            self.assertIn("phase5-eval-record.json", summary)


if __name__ == "__main__":
    unittest.main()
