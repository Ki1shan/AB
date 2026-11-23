"""Phase 8 tests: the perception/imagination evaluation (design SS4).

Fast and deterministic: the Phase 8 evaluation is run once per
module in FAST mode (lazy singleton, as in test_phase7_eval.py --
fast mode runs the cheap subset, ~0.2 s) and the tests assert the
MEASUREMENT. The slow workshop P-D probe lives only in
ab/eval/phase8_eval.py run by the harness or standalone -- never
here. Nothing in this file invokes the harness or the evaluation
with default paths (the fast record is built in-process with
record_path=None), so running the suite never overwrites a canonical
docs/ record. Where a criterion is not met, the record says so --
these tests do not hide failed criteria behind weaker assertions,
and they do not assert a criterion the measurement shows failing
(see the test comments, especially the P-D probe).
"""
import json
import os
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.eval.phase8_eval import (  # noqa: E402
    KEY_ROUTE,
    PICK_LOCK_ROUTE,
    exp_pd_overgeneralisation_probe,
    run_phase8_evaluation,
)

_RECORD = None


def _record():
    global _RECORD
    if _RECORD is None:
        _RECORD = run_phase8_evaluation(fast=True, record_path=None)
    return _RECORD


class TestFastPhase8Record(unittest.TestCase):
    def test_fast_record_has_required_sections(self):
        rec = _record()
        self.assertEqual(rec["mode"], "fast")
        for section in ("experiments", "failures", "worker_dependencies",
                        "n_passed", "n_failed", "n_skipped"):
            self.assertIn(section, rec, f"missing record section {section}")
        for name in ("imagine_routes", "predict_consequences",
                     "compare_routes", "select_and_execute",
                     "imagined_vs_actual", "revision_second_task",
                     "what_if_query", "counterfactual_world_model",
                     "document_perception", "image_percept",
                     "specialist_absent", "pd_overgeneralisation_probe",
                     "simulation_does_not_mutate"):
            self.assertIn(name, rec["experiments"], f"missing experiment {name}")
        self.assertEqual(rec["n_passed"] + rec["n_failed"]
                         + rec["n_skipped"], 13)

    def test_fast_mode_skips_pd_probe_without_passing_it(self):
        probe = _record()["experiments"]["pd_overgeneralisation_probe"]
        self.assertTrue(probe.get("skipped"))
        self.assertNotIn("passed", probe)

    def test_imagine_routes_two_distinct_routes(self):
        exp = _record()["experiments"]["imagine_routes"]
        self.assertTrue(exp["passed"])
        self.assertEqual(exp["api"], "Brain.imagine")
        self.assertGreaterEqual(exp["n_routes"], 2)
        self.assertGreaterEqual(exp["n_distinct_routes"], 2)
        self.assertGreaterEqual(exp["n_distinct_predicted_states"], 2)
        self.assertTrue(exp["simulated"])

    def test_predict_consequences_evidence_based_confidences(self):
        exp = _record()["experiments"]["predict_consequences"]
        self.assertTrue(exp["passed"])
        self.assertTrue(exp["seed_run_completed"])
        self.assertTrue(exp["all_confidences_numeric"])
        self.assertTrue(exp["any_confidence_below_1"])
        # The pick_lock step's confidence is the revision-adjusted
        # rate (Laplace 1/3 pulled toward the observed 0.0), not the
        # uncalibrated 1.0 default.
        self.assertLess(exp["pick_lock_predicted_confidence"], 0.5)
        self.assertGreater(exp["pick_lock_predicted_confidence"], 0.0)
        self.assertEqual(tuple(exp["causal_pick_lock_door_open"]), (0, 1))
        # Finding recorded, not hidden: imagine() itself returns one
        # scenario post-learning (pair ban); both routes are simulated
        # directly through the same Simulator for this measurement.
        self.assertEqual(exp["imagine_scenarios_post_learning"], 1)

    def test_compare_routes_key_route_first(self):
        exp = _record()["experiments"]["compare_routes"]
        self.assertTrue(exp["passed"])
        self.assertGreaterEqual(len(exp["ranking"]), 2)
        self.assertEqual(exp["top_route"], KEY_ROUTE)
        for entry in exp["ranking"]:
            self.assertTrue(entry["reason"])

    def test_select_and_execute_completes_for_real(self):
        exp = _record()["experiments"]["select_and_execute"]
        self.assertTrue(exp["passed"])
        self.assertEqual(exp["selected_route"], PICK_LOCK_ROUTE)
        self.assertTrue(exp["executed_completed"])
        # The selected route failed once at pick_lock in reality and
        # the brain replanned; the selection itself was best-simulated
        # at selection time (success tie, lower cost).
        self.assertEqual(exp["pick_lock_attempts_during_execution"], 1)
        self.assertTrue(exp["simulation_ranking_used_in_execution"])

    def test_imagined_vs_actual_prediction_error_recorded(self):
        exp = _record()["experiments"]["imagined_vs_actual"]
        self.assertTrue(exp["passed"])
        self.assertTrue(exp["imagined_after_pick_lock"]["door_open"])
        self.assertFalse(exp["actual_after_pick_lock"]["door_open"])
        self.assertGreaterEqual(exp["n_prediction_checked_events"], 1)
        err = exp["pick_lock_prediction_error"]
        self.assertEqual((err["action"], err["effect"],
                          err["expected"], err["actual"]),
                         ("pick_lock", "door_open", True, False))

    def test_revision_second_task_moves_toward_observed(self):
        exp = _record()["experiments"]["revision_second_task"]
        self.assertTrue(exp["passed"])
        before = exp["task1_before"]
        after = exp["task1_after"]
        self.assertEqual(before["predicted_success"], 1.0)
        self.assertLess(after["predicted_success"],
                        before["predicted_success"])
        self.assertLess(abs(after["predicted_success"] - 0.0),
                        abs(before["predicted_success"] - 0.0))
        self.assertGreater(after["predicted_cost"],
                           before["predicted_cost"])
        self.assertFalse(after["pick_lock_route_still_in_plan_all"])
        self.assertEqual(
            tuple(exp["task1_observed"]["pick_lock_causal_counts"]), (0, 1))
        # Second structurally similar task: completed, and the
        # pick_lock trap is never attempted again.
        self.assertTrue(exp["task2_gem"]["completed"])
        self.assertEqual(exp["task2_gem"]["pick_lock_attempts"], 0)
        self.assertTrue(exp["task2_gem"]["pick_lock_avoided_in_second_task"])

    def test_what_if_simulated_with_assumptions(self):
        exp = _record()["experiments"]["what_if_query"]
        self.assertTrue(exp["passed"])
        self.assertEqual(exp["api"], "Brain.what_if")
        pick = exp["answers"]["pick_lock"]
        key = exp["answers"]["unlock"]
        self.assertTrue(pick["simulated"])
        self.assertTrue(key["simulated"])
        self.assertTrue(pick["assumptions"])
        self.assertTrue(key["assumptions"])
        self.assertEqual(pick["steps"][:1], ["pick_lock"])
        self.assertLess(exp["pick_lock_predicted_success"],
                        exp["unlock_predicted_success"])
        # The literal "fragile tool" phrasing is honestly declined
        # (no action bears that name); the brain does not guess.
        literal = exp["answers"]["literal_fragile_tool_phrasing"]
        self.assertIn("could not identify", literal["answer"])

    def test_counterfactual_world_model(self):
        exp = _record()["experiments"]["counterfactual_world_model"]
        self.assertTrue(exp["passed"])
        self.assertTrue(exp["outcome_changed"])
        self.assertTrue(exp["changed_assumption_named"])
        self.assertTrue(exp["real_world_model_unchanged"])
        self.assertTrue(exp["world_edit_outcome_changed"])
        self.assertIn("door_open", str(exp["difference"]))

    def test_document_perception_unverified_then_verified(self):
        exp = _record()["experiments"]["document_perception"]
        self.assertTrue(exp["passed"])
        self.assertEqual(exp["api"], "Brain.read_document")
        self.assertEqual(exp["file_trust_used"], 0.45)
        ask1 = exp["ask_after_one_document"]
        ask2 = exp["ask_after_two_documents"]
        self.assertEqual((ask1[2], ask1[3]), ("unverified", ["file:paris.md"]))
        self.assertEqual(ask2[2], "verified")
        self.assertEqual(set(ask2[3]),
                         {"file:paris.md", "file:france-notes.md"})
        self.assertGreaterEqual(exp["duplicate_on_reread"], 1)
        self.assertTrue(exp["missing_file_stored_no_partial_claims"])
        self.assertIn("FileNotFoundError", exp["missing_file_error"])

    def test_image_percept_observed_entities_and_rejection(self):
        exp = _record()["experiments"]["image_percept"]
        self.assertTrue(exp["passed"])
        direct = exp["direct_record"]
        self.assertEqual(direct["status"], "admitted")
        self.assertTrue(direct["key_observed"])
        self.assertTrue(direct["door_observed"])
        self.assertTrue(direct["spatial_left_of"])
        stub = exp["stub_specialist"]
        self.assertEqual(stub["status"], "admitted")
        self.assertEqual(stub["provenance"].get("specialist"), "stub")
        bare = exp["bare_path_no_specialist"]
        self.assertEqual(bare["status"], "rejected")
        self.assertTrue(bare["rejected_without_invented_entities"])

    def test_specialist_absent_and_present(self):
        exp = _record()["experiments"]["specialist_absent"]
        self.assertTrue(exp["passed"])
        self.assertTrue(exp["no_specialist_completes"])
        fake = exp["fake_specialist"]
        self.assertEqual(fake["specialist_api"], "present")
        self.assertEqual(fake["percept_status"], "admitted")
        self.assertTrue(fake["provenance_names_specialist"])
        self.assertTrue(fake["key_entity_observed"])
        self.assertTrue(fake["no_thought_from_specialist_output"])

    def test_pd_probe_negative_result_recorded_honestly(self):
        # Phase 9 update: the P-D probe PASSES now (variant-aware
        # simulation lands in Phase 9 §4.1), so this fast-mode test no
        # longer asserts a negative result -- it asserts the MEASUREMENT
        # shape (probe skipped in fast mode, with its reason). The full
        # probe runs standalone (not in fast mode); its numbers are
        # asserted on the standalone record file when it exists -- see
        # test_pd_probe_standalone_record.
        exp = _record()["experiments"].get("pd_overgeneralisation_probe")
        self.assertTrue(exp.get("skipped"))
        # The full probe runs standalone (not in fast mode); assert
        # its shape here via a direct call would be slow, so the
        # numbers below are asserted on the standalone record file
        # when it exists -- see test_pd_probe_standalone_record.
        self.assertIn("reason", exp)

    def test_simulation_does_not_mutate(self):
        exp = _record()["experiments"]["simulation_does_not_mutate"]
        self.assertTrue(exp["passed"])
        self.assertTrue(exp["world_model_unchanged"])
        self.assertTrue(exp["ltm_unchanged"])
        self.assertTrue(exp["planner_bans_unchanged"])
        self.assertTrue(exp["last_imagination_kept"])


class TestPdProbeStandalone(unittest.TestCase):
    def test_pd_probe_standalone_record(self):
        """The full P-D probe (workshop W1->W2) runs here, once, in
        full mode -- it is the only slow experiment (~1 s). Phase 9
        pin update (Phase 2 §9 precedent — same intent, new intended
        semantics, not weakened): the probe now PASSES -- variant-aware
        causal records separate the two press variants, so the
        misfiring route's predicted success drops below the correct
        route's. Phase 10 pin update (Phase 2 §9 precedent — same
        intent, new intended semantics, not weakened): the pattern
        multiplier in W2 is now 1.0 — P10-1 variant-aware application
        (the W1 pattern's required key `has_mould` is in no W2 action
        precondition, so the pattern cannot fire there). Both routes
        still share one action name, and the planner's candidates now
        include the cheap without-mould press as a third candidate
        (it is cost-visible once the P-A penalty no longer hides it)."""
        path = os.path.join(REPO_ROOT, "docs", "phase8-eval-record.json")
        if not os.path.exists(path):
            self.skipTest("no standalone phase8 record on disk yet")
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("mode") != "full":
            self.skipTest("standalone record is not a full-mode record")
        probe = rec["experiments"]["pd_overgeneralisation_probe"]
        self.assertTrue(probe["passed"])
        self.assertEqual(probe["press_pattern_multiplier_in_w2"], 1.0)
        self.assertTrue(probe["route_names_identical"])
        self.assertTrue(probe["predicted_success_differs"])
        self.assertLess(
            probe["route_without_mould"]["predicted_success"],
            probe["route_with_mould"]["predicted_success"])
        self.assertEqual(probe["candidate_plans_press_variants"],
                         ["with_mould", "with_mould", "without_mould"])
        self.assertTrue(probe["w1_completed"])
        self.assertTrue(probe["w2_continued_completed"])


class TestHarnessPhase8Wiring(unittest.TestCase):
    def test_harness_phase8_section_fast(self):
        from ab.eval.harness import _run_phase8, _summary_text

        section = _run_phase8(fast=True)
        self.assertEqual(section["mode"], "fast")
        self.assertIn("experiments", section)
        self.assertEqual(section["n_passed"] + section["n_failed"]
                         + section["n_skipped"], 13)
        # The summary renders the Phase 8 section from the section.
        record = {
            "timestamp": "2026-11-21T00:00:00+00:00", "mode": "fast",
            "tests": {"ran": 1, "failures": 0, "errors": 0, "skipped": 0,
                      "ok": True, "scope": "test", "duration_s": 0.0,
                      "exit_code": 0},
            "demo": {"exit_code": 0, "outcome": "done", "duration_s": 0.0},
            "disconnected": {"result": None},
            "baselines": {"treasure": {}},
            "phase8_evaluation": section,
            "resources": {"total_wall_s": 0.0, "suite_wall_s": 0.0,
                          "demo_wall_s": 0.0, "disconnected_wall_s": 0.0,
                          "peak_child_rss_kb": "not measured"},
            "repo_state": {"vcs": "none", "python_files": 0, "test_files": 0},
        }
        text = _summary_text(record)
        self.assertIn("Phase 8 evaluation", text)
        self.assertIn("imagine_routes: PASS", text)
        self.assertIn("pd_overgeneralisation_probe: SKIPPED", text)

    def test_fast_eval_writes_nothing_canonical(self):
        canonical = os.path.join(REPO_ROOT, "docs",
                                 "phase8-eval-record.json")
        summary = os.path.join(REPO_ROOT, "docs",
                               "phase8-eval-summary.txt")
        before = {}
        for path in (canonical, summary):
            before[path] = os.path.getmtime(path) if os.path.exists(path) else None
        run_phase8_evaluation(fast=True, record_path=None)
        for path in (canonical, summary):
            after = os.path.getmtime(path) if os.path.exists(path) else None
            self.assertEqual(after, before[path],
                             f"fast run touched canonical {path}")

    def test_explicit_temp_record_path(self):
        with tempfile.TemporaryDirectory() as folder:
            record_path = os.path.join(folder, "p8.json")
            rec = run_phase8_evaluation(fast=True, record_path=record_path)
            self.assertTrue(os.path.exists(record_path))
            with open(record_path, encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["mode"], "fast")
            self.assertEqual(on_disk["n_passed"], rec["n_passed"])


class TestPhase8EvalModuleDirect(unittest.TestCase):
    def test_pd_probe_runs_standalone(self):
        """Direct call of the slow probe (full mode, record_path=None
        so nothing is written): W1 and W2 both complete, the press
        pattern multiplier is 6.0, and the probe now CATCHES the
        over-generalisation (Phase 9 §4.1: the probe simulates Action
        objects, so the misfiring route's predicted success drops
        below the correct route's). Phase 9 pin update (Phase 2 §9
        precedent — same intent, new intended semantics, not
        weakened): the old assertion was probe["passed"] is False,
        which measured the Phase 8 negative result; the fix is the
        change being tested."""
        probe = exp_pd_overgeneralisation_probe()
        self.assertTrue(probe["passed"])
        self.assertTrue(probe["w1_completed"])
        self.assertTrue(probe["w2_continued_completed"])
        # Phase 10 pin update (Phase 2 §9 precedent — same intent,
        # new intended semantics, not weakened): P10-1 variant-aware
        # pattern application makes the multiplier 1.0 in W2 (the W1
        # pattern's required key is in no W2 action precondition).
        self.assertEqual(probe["press_pattern_multiplier_in_w2"], 1.0)
        self.assertTrue(probe["predicted_success_differs"])
        self.assertLess(
            probe["route_without_mould"]["predicted_success"],
            probe["route_with_mould"]["predicted_success"])


if __name__ == "__main__":
    unittest.main()
