"""Phase 6 tests: the learning evaluation programme (design SS7 + SS8).

The full evaluation is run once per module (lazy singleton, as in
test_phase5_eval.py -- it runs six variants on ten held-out tasks
plus the sub-experiments, so re-running it per test would take
minutes). The tests assert the MEASUREMENT: records are written
with per-variant numbers, the acceptance experiment completes with
numbers per step, the cap experiment produces retention + transfer
per setting, and the calibration sub-experiment reports ECE
before/after. Where a criterion is not met, the record says so --
these tests do not hide failed criteria behind weaker assertions,
and they do not assert a criterion the measurement shows failing
(see the test comments).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.learning_eval import (  # noqa: E402
    VARIANTS,
    run_acceptance_experiment,
    run_cap_experiment,
    run_calibration_experiment,
    run_learning_evaluation,
)

_RECORD = None


def _record():
    global _RECORD
    if _RECORD is None:
        _RECORD = run_learning_evaluation(record_path=None)
    return _RECORD


class TestLearningEvaluationRecord(unittest.TestCase):
    def test_evaluation_runs_and_records_per_variant_numbers(self):
        rec = _record()
        self.assertEqual(rec["experiment"], "phase6-learning-evaluation")
        self.assertEqual(sorted(rec["variants"]),
                         ["fresh", "full", "no_calibration",
                          "no_consolidation", "no_learning", "no_patterns"])
        for name, variant in rec["variants"].items():
            self.assertIn("mechanism", variant, name)
            self.assertIn("tasks", variant, name)
            # Per-variant, per-task numbers on the same 10-task set.
            self.assertEqual(sorted(variant["tasks"]),
                             [f"C{i}" for i in range(1, 6)]
                             + [f"D{i}" for i in range(1, 6)], name)
            for task_id, task in variant["tasks"].items():
                for key in ("completed", "cycles", "failed_attempts",
                            "transfer_vs_fresh",
                            "recovery_after_reset"):
                    self.assertIn(key, task, f"{name}/{task_id}")
                self.assertIsInstance(task["completed"], bool, name)
                self.assertIsInstance(task["cycles"], int, name)
                self.assertIsInstance(task["failed_attempts"], int, name)
            s = variant["summary"]
            self.assertEqual(s["n_tasks"], 10, name)
            self.assertEqual(s["n_completed"], 10, name)
            # Means with sample sizes; mean +/- std and range at n>=5.
            for metric in ("cycles", "failed_attempts"):
                m = s[metric]
                self.assertEqual(m["n"], 10, f"{name}/{metric}")
                self.assertIsNotNone(m["mean"], f"{name}/{metric}")
                self.assertIsNotNone(m["std"], f"{name}/{metric}")
                self.assertLessEqual(m["min"], m["max"], f"{name}/{metric}")
            # The full variant transfers on held-out tasks; the
            # ablations that remove learning/patterns do not beat
            # fresh as often. (Recorded, not declared a winner.)
            self.assertIn("transfer_tasks", s, name)

    def test_regression_recorded_per_variant(self):
        # Regression / overfitting: base world C re-run after the new
        # tasks, vs the same variant's fresh-C baseline. No variant
        # may silently degrade: the record carries the comparison and
        # the degraded flag.
        rec = _record()
        for name, variant in rec["variants"].items():
            if name == "fresh":
                self.assertIsNone(variant["regression"], name)
                continue
            reg = variant["regression"]
            self.assertIn("fresh_baseline", reg, name)
            self.assertIn("after_learning_new_tasks", reg, name)
            self.assertIn("degraded", reg, name)
            self.assertFalse(reg["degraded"], f"{name} degraded on base C: {reg}")

    def test_retention_after_restart_recorded(self):
        ret = _record()["retention_after_restart"]
        for key in ("patterns_before_save", "patterns_after_load",
                    "pattern_ids_preserved", "episodes_before_save",
                    "episodes_after_load", "episodes_preserved",
                    "concepts_before_save", "concepts_after_load",
                    "concepts_preserved",
                    "calibration_records_before_save",
                    "calibration_records_after_load",
                    "loaded_brain_d", "fresh_d", "retained_transfer"):
            self.assertIn(key, ret, key)
        self.assertTrue(ret["pattern_ids_preserved"], ret)
        self.assertTrue(ret["episodes_preserved"], ret)
        self.assertTrue(ret["concepts_preserved"], ret)
        self.assertTrue(ret["calibration_preserved"], ret)
        self.assertTrue(ret["retained_transfer"], ret)

    def test_no_consolidation_mechanism_is_real_not_silent(self):
        # The consolidation ablation disables the real pass (the
        # brain's concepts stay empty); it must NOT be a silent
        # full-equivalent. The mechanism note says how it was done.
        rec = _record()
        mech = rec["variants"]["no_consolidation"]["mechanism"]
        self.assertIn("consolidation pass disabled", mech)
        self.assertNotIn("full-equivalent", mech)

    def test_no_calibration_variant_runs(self):
        # The pinned-1.0 variant completes the task set (its numbers
        # are recorded); calibration only feeds prediction checks, so
        # it is not expected to change task outcomes here -- the
        # record shows that rather than hiding it.
        rec = _record()
        s = rec["variants"]["no_calibration"]["summary"]
        self.assertEqual(s["n_completed"], 10)


class TestAcceptanceExperiment(unittest.TestCase):
    def test_acceptance_completes_with_numbers_per_step(self):
        acc = run_acceptance_experiment()
        self.assertEqual(acc["experiment"], "phase6-six-step-acceptance")
        self.assertEqual(sorted(acc["steps"]),
                         ["1_fresh_brain_runs_C",
                          "2_learns_reusable_pattern",
                          "3_retained_across_save_load",
                          "4_loaded_brain_runs_D",
                          "5_pattern_applied_in_D",
                          "6_improves_vs_baselines"])
        for step_id, step in acc["steps"].items():
            self.assertIn("passed", step, step_id)
            self.assertIsInstance(step["passed"], bool, step_id)
        # Step 1: fresh brain completes C.
        self.assertTrue(acc["steps"]["1_fresh_brain_runs_C"]["passed"])
        self.assertTrue(acc["steps"]["1_fresh_brain_runs_C"]["completed"])
        # Step 2: a reusable pattern is learned in C.
        s2 = acc["steps"]["2_learns_reusable_pattern"]
        self.assertTrue(s2["passed"], s2)
        self.assertGreater(s2["patterns_in_library"], 0, s2)
        self.assertGreater(s2["pattern_learned_events"], 0, s2)
        # Step 3: pattern + concept + calibration survive save/load.
        s3 = acc["steps"]["3_retained_across_save_load"]
        self.assertTrue(s3["passed"], s3)
        self.assertTrue(s3["pattern_ids_preserved"], s3)
        self.assertTrue(s3["concepts_preserved"], s3)
        # Step 4: loaded brain completes D.
        self.assertTrue(acc["steps"]["4_loaded_brain_runs_D"]["passed"])
        # Step 5: a pattern_applied event in D names a C pattern.
        s5 = acc["steps"]["5_pattern_applied_in_D"]
        self.assertTrue(s5["passed"], s5)
        self.assertGreater(s5["pattern_applied_events"], 0, s5)
        # Step 6: continued D beats fresh D and no-learning.
        s6 = acc["steps"]["6_improves_vs_baselines"]
        self.assertTrue(s6["passed"], s6)
        self.assertTrue(s6["beats_fresh_d_on_failures_and_cycles"], s6)
        self.assertTrue(s6["beats_no_learning"], s6)
        self.assertTrue(acc["passed"], acc["failures"])


class TestCapExperiment(unittest.TestCase):
    def test_cap_experiment_produces_retention_and_transfer_per_setting(self):
        cap = run_cap_experiment()
        self.assertEqual(sorted(cap["settings"]),
                         ["cap_100_prioritized", "cap_100_tail",
                          "cap_2000_prioritized", "cap_2000_tail",
                          "cap_500_prioritized", "cap_500_tail"])
        for setting_id, setting in cap["settings"].items():
            for key in ("cap", "pruning", "episodes_before_filler",
                        "episodes_after_prune",
                        "real_episodes_retained",
                        "real_episodes_retained_fraction",
                        "pattern_evidence_episodes",
                        "pattern_evidence_retained",
                        "patterns_before", "patterns_after",
                        "patterns_retained_fraction",
                        "continued_d1", "fresh_d1",
                        "transfer_vs_fresh_d1"):
                self.assertIn(key, setting, setting_id)
            self.assertEqual(setting["episodes_after_prune"],
                             setting["cap"], setting_id)
            # Patterns are never pruned: they are the durable summary.
            self.assertEqual(setting["patterns_retained_fraction"], 1.0,
                             setting_id)
            self.assertEqual(setting["patterns_after"],
                             setting["patterns_before"], setting_id)
        # Tail pruning drops the oldest (real) episodes first: at
        # every cap the real-episode retention is 0 -- the filler is
        # newer than all of it. Prioritized pruning keeps the most
        # recent + most-used. Phase 7 §4.2 pin update (Phase 2 §9
        # precedent — same intent, new intended semantics, not
        # weakened): the Phase 7 prioritized-pruning fix marks
        # pattern-evidence episodes +1 usefulness when the pattern is
        # learned, so prioritized pruning now retains 2 of the 3
        # pattern-evidence episodes at every cap (was 0/3) and 5 of
        # 25 real episodes (was 3/25); the cap VALUE still makes no
        # difference at these scales, only the pruning policy does.
        # Transfer still holds in every setting because the patterns
        # (not the episodes) carry the transfer.
        for setting_id, setting in cap["settings"].items():
            self.assertTrue(setting["continued_d1"]["completed"], setting_id)
            self.assertTrue(setting["transfer_vs_fresh_d1"], setting_id)
        for cap_value in (100, 500, 2000):
            tail = cap["settings"][f"cap_{cap_value}_tail"]
            prio = cap["settings"][f"cap_{cap_value}_prioritized"]
            self.assertEqual(tail["real_episodes_retained"], 0, cap_value)
            self.assertGreater(prio["real_episodes_retained"], 0, cap_value)
            self.assertEqual(prio["pattern_evidence_retained"], 2, cap_value)


class TestCalibrationExperiment(unittest.TestCase):
    def test_calibration_reports_ece_before_and_after(self):
        cal = run_calibration_experiment()
        overall = cal["overall"]
        for key in ("n_predictions", "met_rate",
                    "ece_default_1.0", "ece_calibrated",
                    "brier_default_1.0", "brier_calibrated",
                    "ece_improved", "brier_improved",
                    "unreliable_subset"):
            self.assertIn(key, overall, key)
        self.assertGreater(overall["n_predictions"], 0, overall)
        # ECE before (1.0 default) vs after (calibrated) on the same
        # stream, plus the met-rate trend across runs. On this stream
        # (C1/C2/D1) the aggregate ECE improves (the fresh C1 run has
        # 3 violations of 9 predictions, so there is real
        # miscalibration for the tracker to fix); the unreliable
        # subset improves decisively too. The aggregate result is
        # stream-dependent -- on the base-C double-run stream the
        # aggregate read slightly worse than the 1.0 default (most
        # actions there are deterministic-reliable, so Laplace
        # shrinkage adds error on them) -- so the test asserts the
        # measured numbers on THIS stream, and the record note keeps
        # the stream-dependence visible rather than claiming a
        # universal improvement.
        self.assertTrue(cal["passed"], cal["overall"])
        self.assertLess(overall["ece_calibrated"],
                        overall["ece_default_1.0"], overall)
        self.assertLess(overall["brier_calibrated"],
                        overall["brier_default_1.0"], overall)
        self.assertTrue(overall["unreliable_subset"]["ece_improved"],
                        overall["unreliable_subset"])
        self.assertLess(
            overall["unreliable_subset"]["ece_calibrated"],
            overall["unreliable_subset"]["ece_default_1.0"],
            overall["unreliable_subset"])


class TestLearningEvalCli(unittest.TestCase):
    def test_cli_writes_record_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "learning.json")
            proc = subprocess.run(
                [sys.executable, "-m", "ab.eval.learning_eval",
                 "--record", record_path],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(record_path, encoding="utf-8") as fh:
                rec = json.load(fh)
            self.assertIn("variants", rec)
            self.assertIn("acceptance_experiment", rec)
            self.assertIn("cap_measurement", rec)

    def test_generated_variants_are_distinct_and_seeded(self):
        # All ten variants have distinct room/object names (no two
        # variants share a name) and generation is deterministic.
        from ab.eval.learning_eval import generate_variants
        v1, v2 = generate_variants(), generate_variants()
        self.assertEqual(v1, v2)
        rooms, objects = set(), set()
        for spec in v1.values():
            for room in spec["rooms"]:
                self.assertNotIn(room, rooms, spec["name"])
                rooms.add(room)
            for key in ("fragile", "durable", "decoy", "treasure"):
                self.assertNotIn(spec[key], objects, spec["name"])
                objects.add(spec[key])
        # No variant reuses a Phase 5 A/B name.
        from ab.eval.disconnected import WORLD_A, WORLD_B
        ab_names = set()
        for spec in (WORLD_A, WORLD_B):
            ab_names.update(spec["rooms"])
            ab_names.update([spec["fragile"], spec["durable"],
                             spec["decoy"], spec["treasure"]])
        self.assertTrue(rooms.isdisjoint(ab_names))
        self.assertTrue(objects.isdisjoint(ab_names))
        self.assertEqual(len(v1), 10)


if __name__ == "__main__":
    unittest.main()
