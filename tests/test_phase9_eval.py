"""Phase 9 consolidated evaluation tests (Worker 3).

Covers, per the Phase 9 design SS3.1 / SS3.4 / SS3.8 / SS3.9:

- the consolidated runner's exit-code contract BOTH ways (a green
  programme -> 0; a deliberately failing suite -> non-zero), using
  tiny stub suites patched into ``_SUITE_RUNNERS`` -- never by
  breaking the real brain;
- same-seed reproducibility (run twice, diff the numbers with
  timestamps and durations excluded), both for a stubbed full run
  and for the real discriminating suite;
- the common result format shape;
- the leakage audit reporting exactly the one known hit on this
  tree (``zorblat`` in a docstring example in ab/brain.py) and
  failing that name's experiment, per the design's grep rule;
- the regression record being written with the Phase 8 row present;
- the learning-curves shape (SS3.4);
- the no_revision ablation factory (observe() never adjusts,
  predicted_success() always falls back to the causal rate);
- the Worker 4 sidecar legs skipping honestly when the sidecars
  are absent.

The full real consolidated run (~1 min) is NOT part of this file:
it is run by the coordinator / harness, not by the unit suite.
No existing test file is touched; no test is weakened.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from ab.eval import consolidated
from ab.eval.consolidated import (
    _NoOpRevision,
    _add_exp,
    _finalise_suite,
    _new_suite,
    _no_revision_factory,
    _read_sidecar,
    _run_discriminating_suite,
    _run_perception_suite,
    _run_tool_use_suite,
    run_leakage_audit,
)
from ab.eval.learning_eval import run_learning_curves
from ab.eval.regression import build_regression_record, write_regression_record

# All suite names the consolidated runner knows (kept in sync with
# consolidated._SUITE_RUNNERS by the test below).
SUITE_NAMES = list(consolidated._SUITE_RUNNERS)


def _passing_suite(name):
    suite = {"suite": name, "experiments": {}, "n_passed": 0,
             "n_failed": 0, "n_skipped": 0, "duration_s": 0.0,
             "seed": 0, "mode": "full", "environment": {}}
    _add_exp(suite, f"{name}/check", passed=True, cycles=1,
             detail="stub pass", evidence={}, meta={})
    return suite


def _failing_suite(name):
    suite = {"suite": name, "experiments": {}, "n_passed": 0,
             "n_failed": 0, "n_skipped": 0, "duration_s": 0.0,
             "seed": 0, "mode": "full", "environment": {}}
    _add_exp(suite, f"{name}/check", passed=False, cycles=1,
             detail="stub failure (deliberate)", evidence={},
             failure_cause="stub: deliberate failure for the exit-code test",
             meta={})
    return suite


def _stub_runners(failing: str | None = None):
    runners = {}
    for name in SUITE_NAMES:
        if name == failing:
            runners[name] = lambda ctx, _n=name: _failing_suite(_n)
        else:
            runners[name] = lambda ctx, _n=name: _passing_suite(_n)
    return runners


def _strip_volatile(value):
    """Drop timestamps and durations (the reproducibility comparison
    excludes them by design). Goal ids are dropped too, but ONLY
    from goal-shaped dicts (those with both ``description`` and
    ``target``): goal ids come from a process-global counter
    (``ab/planning/goals.py`` ``_gid``), so two runs in one process
    hand out different ids while separate-process runs reproduce
    them exactly. Ids are process bookkeeping, not measurements --
    everything else (including pattern ids) is compared exactly."""
    if isinstance(value, dict):
        out = {}
        is_goal = "description" in value and "target" in value
        for k, v in value.items():
            if k in ("timestamp", "duration_s", "total_duration_s"):
                continue
            if k == "id" and is_goal:
                continue
            out[k] = _strip_volatile(v)
        return out
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


class TestCommonResultFormat(unittest.TestCase):
    def test_shape_and_counts(self):
        suite = _new_suite("probe", seed=5, mode="full",
                           meta={"persistence_version": 5,
                                 "python_version": "3.12.3",
                                 "tesseract_version": "5.3.4",
                                 "line_counts": {}, "test_counts": {}})
        _add_exp(suite, "ok", passed=True, cycles=3, detail="fine",
                 evidence={"a": 1}, meta={})
        _add_exp(suite, "bad", passed=False, cycles=2, detail="broke",
                 evidence={}, meta={})
        _add_exp(suite, "skip", None, skipped=True,
                 skip_reason="not available", evidence={}, meta={})
        _finalise_suite(suite, 0.0)
        for key in ("suite", "experiments", "n_passed", "n_failed",
                    "n_skipped", "duration_s", "seed"):
            self.assertIn(key, suite)
        self.assertEqual((suite["n_passed"], suite["n_failed"],
                          suite["n_skipped"]), (1, 1, 1))
        for exp in suite["experiments"].values():
            for key in ("passed", "cycles", "detail", "evidence",
                        "failure_cause", "seed", "persistence_version",
                        "python_version", "tesseract_version"):
                self.assertIn(key, exp)
        self.assertIsNone(suite["experiments"]["ok"]["failure_cause"])
        self.assertEqual(suite["experiments"]["bad"]["failure_cause"],
                         "unknown -- needs investigation")
        self.assertTrue(suite["experiments"]["skip"]["skipped"])

    def test_suite_names_known(self):
        # The stub table above must cover every runner the production
        # code iterates; a new suite added without a stub entry fails
        # loudly here instead of silently skipping the exit-code test.
        self.assertEqual(set(SUITE_NAMES),
                         set(consolidated._SUITE_RUNNERS))
        self.assertIn("discriminating", SUITE_NAMES)
        self.assertIn("leakage_audit", SUITE_NAMES)


class TestExitCodeContract(unittest.TestCase):
    def test_green_programme_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            record_path = os.path.join(folder, "record.json")
            summary_path = os.path.join(folder, "summary.txt")
            with mock.patch.dict(consolidated._SUITE_RUNNERS,
                                 _stub_runners(), clear=True):
                record = consolidated.run_consolidated_evaluation(
                    seed=7, record_path=record_path,
                    summary_path=summary_path)
                self.assertTrue(record["overall"]["all_passed"])
                self.assertTrue(os.path.isfile(record_path))
                self.assertTrue(os.path.isfile(summary_path))
                self.assertTrue(os.path.isfile(
                    os.path.join(folder, "phase9-regression-record.json")))
                rc = consolidated.main(argv=[
                    "--seed", "7", "--record", record_path,
                    "--summary", summary_path])
                self.assertEqual(rc, 0)

    def test_failing_suite_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as folder:
            record_path = os.path.join(folder, "record.json")
            summary_path = os.path.join(folder, "summary.txt")
            with mock.patch.dict(consolidated._SUITE_RUNNERS,
                                 _stub_runners(failing="phase8"),
                                 clear=True):
                record = consolidated.run_consolidated_evaluation(
                    seed=7, record_path=record_path,
                    summary_path=summary_path)
                self.assertFalse(record["overall"]["all_passed"])
                self.assertEqual(record["overall"]["n_suites_failed"], 1)
                self.assertIn("EVALUATION FAILED",
                              record["overall"]["verdict"])
                summary = consolidated.summary_text(record)
                self.assertIn("EVALUATION FAILED", summary)
                rc = consolidated.main(argv=[
                    "--seed", "7", "--record", record_path,
                    "--summary", summary_path])
                self.assertEqual(rc, 1)

    def test_harness_path_writes_nothing(self):
        # record_path=None / summary_path=None (the harness path):
        # the consolidated run returns the record but writes no
        # canonical file anywhere.
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(consolidated._SUITE_RUNNERS,
                                 _stub_runners(), clear=True):
                with mock.patch.object(
                        consolidated, "DEFAULT_RECORD",
                        os.path.join(folder, "must-not-exist.json")):
                    record = consolidated.run_consolidated_evaluation(
                        seed=7, record_path=None, summary_path=None)
                self.assertTrue(record["overall"]["all_passed"])
            self.assertEqual(os.listdir(folder), [])


class TestReproducibility(unittest.TestCase):
    def test_same_seed_stubbed_run_reproduces(self):
        with mock.patch.dict(consolidated._SUITE_RUNNERS,
                             _stub_runners(), clear=True):
            first = consolidated.run_consolidated_evaluation(
                seed=3, record_path=None, summary_path=None)
            second = consolidated.run_consolidated_evaluation(
                seed=3, record_path=None, summary_path=None)
        self.assertEqual(_strip_volatile(first), _strip_volatile(second))

    def test_real_discriminating_suite_reproduces(self):
        meta = consolidated.build_environment_meta(seed=3)
        first = _run_discriminating_suite(
            {"seed": 3, "fast": False, "meta": meta, "suites": {}})
        meta = consolidated.build_environment_meta(seed=3)
        second = _run_discriminating_suite(
            {"seed": 3, "fast": False, "meta": meta, "suites": {}})
        self.assertEqual(_strip_volatile(first), _strip_volatile(second))


class TestLeakageAudit(unittest.TestCase):
    def test_audit_passes_after_docstring_fix(self):
        # Phase 9 coordinator fix (Phase 2 §9 precedent — same intent,
        # new intended semantics, not weakened): the one known hit is
        # gone -- ``zorblat`` in the ``_tool_entity_name`` docstring
        # example in ab/brain.py was renamed to ``use_wrench`` ->
        # ``wrench`` -- so the audit now passes with no hits. This
        # test pins the clean finding exactly so a future leak changes
        # it loudly. (Previous name:
        # test_audit_reports_the_known_docstring_hit.)
        audit = run_leakage_audit()
        self.assertEqual(audit["hits"], {})
        self.assertEqual(audit["n_leaks"], 0)
        for name in ("velkra", "torvin", "sarnak", "kelmor", "lyssan",
                     "quivern", "tessera", "zorblat"):
            self.assertIn(name, audit["names_checked"])
            self.assertEqual(audit["hits"].get(name) or [], [])
        self.assertIn("transfer", audit["mechanisms"])
        meta = consolidated.build_environment_meta(seed=0)
        suite = consolidated._run_leakage_suite(
            {"seed": 0, "fast": False, "meta": meta, "suites": {}})
        self.assertEqual(suite["n_failed"], 0)
        self.assertEqual(suite["experiments"]["name/zorblat"]["passed"],
                         True)


class TestRegressionRecord(unittest.TestCase):
    def test_phase8_row_present(self):
        # Phase 10.5 fix: the current run's numbers are labelled with
        # the phase that ran (``current_phase``), not hard-coded "9".
        # This test is a Phase 9 run, so its row is still "9" -- same
        # intent as before, new intended semantics.
        phase9 = {"tests": {"ran": 669, "ok": True},
                  "demo": None, "transfer": {"transferred": True},
                  "behaviours": None,
                  "eval_passes": {"discriminating/deduction_3hop/full": True}}
        record = build_regression_record(current=phase9, current_phase="9")
        rows = {r["phase"]: r for r in record["rows"]}
        self.assertIn("8", rows)
        self.assertTrue(rows["8"]["record_present"])
        self.assertIn("9", rows)
        self.assertEqual(rows["9"]["eval_passes"]["discriminating/deduction_3hop/full"],
                         True)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "phase9-regression-record.json")
            written = write_regression_record(current=phase9,
                                              current_phase="9", path=path)
            with open(path, encoding="utf-8") as fh:
                reloaded = json.load(fh)
            self.assertEqual(reloaded, written)

    def test_current_run_labelled_with_its_phase_not_9(self):
        # A Phase 10.5 run's numbers land in the "10.5" row; the "9"
        # row keeps its restored verified numbers (742 tests, 102/106)
        # and its per-experiment map stays None (lost with the
        # clobbered Phase 9 record -- never reconstructed).
        current = {"tests": {"ran": 929, "ok": True},
                   "demo": None, "transfer": {"transferred": True},
                   "behaviours": None,
                   "eval_passes": {"transfer/transfer_generated_envgen/workshop": True}}
        record = build_regression_record(current=current, current_phase="10.5")
        rows = {r["phase"]: r for r in record["rows"]}
        self.assertEqual(
            rows["10.5"]["eval_passes"]["transfer/transfer_generated_envgen/workshop"],
            True)
        self.assertEqual(rows["9"]["tests"]["ran"], 742)
        self.assertEqual(rows["9"]["eval_passes_summary"],
                         {"n_passed": 102, "n_failed": 4, "n_total": 106})
        self.assertIsNone(rows["9"]["eval_passes"])

    def test_bare_build_restores_history_rows(self):
        # No phase label: the record is rebuilt from history alone --
        # rows 3/5/6/7/8 from the harness records, 9/10/10.5/11 from the
        # verified tables -- and no row is relabelled. (Phase 12 fix:
        # "11" is in the history tables now; the Phase 12 gate dropped
        # it because it was not, and this test pins it. "12" is
        # carried forward from the on-disk file by the same fix --
        # it is the Phase 12 gate's own row, not a history row.)
        record = build_regression_record()
        rows = {r["phase"]: r for r in record["rows"]}
        self.assertEqual(sorted(rows), ["10", "10.5", "11", "12", "3", "5", "6", "7", "8", "9"])
        self.assertEqual(rows["9"]["tests"]["ran"], 742)
        self.assertEqual(rows["10"]["tests"]["ran"], 850)
        self.assertEqual(rows["10.5"]["tests"]["ran"], 929)
        self.assertEqual(rows["11"]["tests"]["ran"], 1057)
        self.assertEqual(rows["10"]["eval_passes_summary"],
                         {"n_passed": 155, "n_failed": 5, "n_total": 160})
        self.assertEqual(rows["10.5"]["eval_passes_summary"],
                         {"n_passed": 156, "n_failed": 4, "n_total": 160})
        self.assertEqual(rows["11"]["eval_passes_summary"],
                         {"n_passed": 156, "n_failed": 4, "n_total": 160})
        self.assertIsNone(rows["9"]["eval_passes"])

    def test_bare_write_leaves_canonical_rows_untouched(self):
        # Writing with no phase label produces exactly the history
        # record -- a bare run can never rewrite another phase's row.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "phase9-regression-record.json")
            written = write_regression_record(path=path)
            rows = {r["phase"]: r for r in written["rows"]}
            self.assertEqual(rows["9"]["tests"]["ran"], 742)
            self.assertEqual(rows["10.5"]["tests"]["ran"], 929)
            self.assertEqual(rows["11"]["tests"]["ran"], 1057)

    def test_rebuild_carries_forward_unknown_phase_row(self):
        # Phase 12 fix: a rebuild must never silently drop a row
        # whose phase is not in the history tables (that is how the
        # Phase 11 row was lost at the Phase 12 gate). A row already
        # on disk for an unknown phase survives a rebuild, in file
        # order, while the running phase's row is still labelled
        # with its own phase.
        with tempfile.TemporaryDirectory() as folder:
            docs = os.path.join(folder, "docs")
            os.makedirs(docs)
            seed_record = {"record": "phase9-regression-record",
                           "rows": [{"phase": "99", "tests": {"ran": 1},
                                     "note": "unknown phase row"}]}
            with open(os.path.join(docs, "phase9-regression-record.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(seed_record, fh)
            current = {"tests": {"ran": 1078, "ok": True},
                       "demo": None, "transfer": {"transferred": True},
                       "behaviours": None,
                       "eval_passes": {"transfer/transfer_complex": False}}
            record = build_regression_record(current=current,
                                             current_phase="12",
                                             repo_root=folder)
            rows = {r["phase"]: r for r in record["rows"]}
            self.assertIn("99", rows)
            self.assertEqual(rows["99"]["tests"]["ran"], 1)
            self.assertEqual(rows["12"]["tests"]["ran"], 1078)
            self.assertEqual(rows["11"]["tests"]["ran"], 1057)


class TestLearningCurves(unittest.TestCase):
    def test_curve_shape(self):
        curves = run_learning_curves(levels=(0, 1), held_out_ids=("D5",),
                                     seed=0)
        self.assertEqual(curves["seed"], 0)
        self.assertEqual(len(curves["points"]), 2)
        for point, level in zip(curves["points"], (0, 1)):
            self.assertEqual(point["experience_level_requested"], level)
            self.assertEqual(point["n_prior_tasks"], level)
            self.assertEqual(point["n"], 1)
            self.assertIsNotNone(point["success_rate"])
            for key in ("cycles", "failed_attempts"):
                self.assertIn("mean", point[key])
                self.assertIn("std", point[key])
            self.assertEqual(len(point["runs"]), 1)


class TestNoRevisionAblation(unittest.TestCase):
    def test_noop_revision_never_adjusts(self):
        from ab.body.environment import room_world_actions
        from ab.brain import Brain

        brain = _no_revision_factory(room_world_actions(),
                                     {"compute", "environment"})
        self.assertIsInstance(brain.simulation_revision, _NoOpRevision)
        self.assertIs(brain.simulator.revision, brain.simulation_revision)
        adj = brain.simulation_revision.observe(
            "pick_lock", "door_open", True, False, 1, causal=brain.causal)
        self.assertEqual(adj["moved"], 0.0)
        self.assertEqual(adj["adjusted"], adj["prior"])
        self.assertEqual(brain.simulation_revision.adjusted_success, {})
        self.assertEqual(brain.simulation_revision.history, [])
        # No causal record yet: predicted success falls back to None
        # (the caller then uses the planner's 1.0 model), never to a
        # stale adjusted rate.
        self.assertIsNone(brain.simulation_revision.predicted_success(
            "pick_lock", "door_open", brain.causal))

    def test_no_revision_runs_deduction_task(self):
        from ab.eval.reasoning_tasks import run_deduction_task

        result = run_deduction_task(_no_revision_factory)
        self.assertIn("passed", result)
        self.assertIn("cycles", result)


class TestSidecarSkips(unittest.TestCase):
    def test_missing_sidecar_is_skipped_not_failed(self):
        self.assertEqual(_read_sidecar(
            os.path.join(tempfile.gettempdir(),
                         "phase9-no-such-sidecar.json"))[0], None)
        meta = consolidated.build_environment_meta(seed=0)
        ctx = {"seed": 0, "fast": False, "meta": meta, "suites": {},
               "phase8_record": {"experiments": {
                   "document_perception": {"passed": True, "detail": "ok"},
                   "image_percept": {"passed": True, "detail": "ok"}}}}
        with mock.patch.object(consolidated, "NOISY_OCR_SIDECAR",
                               os.path.join(tempfile.gettempdir(),
                                            "phase9-no-such-sidecar.json")):
            suite = _run_perception_suite(ctx)
        noisy = suite["experiments"]["noisy_ocr"]
        self.assertTrue(noisy["skipped"])
        self.assertEqual(suite["n_failed"], 0)

        ctx2 = {"seed": 0, "fast": False, "meta": meta, "suites": {},
                "execution_record": {"experiments": {
                    "budget_exceeding_tool": {"passed": True},
                    "permission_denial": {"passed": True},
                    "provenance": {"passed": True}}}}
        with mock.patch.object(consolidated, "CPU_BURNER_SIDECAR",
                               os.path.join(tempfile.gettempdir(),
                                            "phase9-no-such-sidecar.json")):
            suite2 = _run_tool_use_suite(ctx2)
        burner = suite2["experiments"]["rlimit_cpu_burner"]
        self.assertTrue(burner["skipped"])
        self.assertEqual(suite2["n_failed"], 0)


if __name__ == "__main__":
    unittest.main()
