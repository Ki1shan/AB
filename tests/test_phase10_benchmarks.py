"""Phase 10 (Worker 2): tests for the harder generated environments
(``ab/eval/envgen.py`` hard mode) and the nine Tier 2/3 benchmark
tasks (``ab/eval/benchmarks.py``).

Every test asserts on the recorded criterion, not on printed
output. Where the brain genuinely fails a task (transfer_complex:
negative transfer, recorded honestly), the test asserts the
measurement -- both legs ran, the numbers are recorded, and the
episode-based count disagrees with the log-based count -- rather
than asserting a pass the brain did not earn.
"""
import unittest

from ab.brain import Brain
from ab.eval import benchmarks as bench
from ab.eval.benchmarks import (
    audit_failure_markers, count_failures, run_benchmarks)
from ab.eval.envgen import generate, generate_hard
from ab.eval.reasoning_tasks import make_variant_factories
from ab.eval.transfer import prepare_brain_for_new_world
from ab.planning.planner import Action


def _full():
    return make_variant_factories()["full"]


class TestFailureMarkersAudit(unittest.TestCase):
    def test_audit_finds_unmatched_characteristic_strings(self):
        """P10-2 audit: the surprise cold-coil line, the press
        no-mould line and the cut-nothing line match NO consolidated
        FAILURE_MARKER, while the treasure cannot-reach line does."""
        audit = audit_failure_markers()
        strings = audit["strings"]
        self.assertFalse(strings["surprise/cold-coil"]["matched"])
        self.assertFalse(strings["workshop/press-no-mould"]["matched"])
        self.assertFalse(strings["workshop/cut-nothing"]["matched"])
        self.assertTrue(strings["treasure/cannot-reach"]["matched"])
        self.assertIn("surprise/cold-coil", audit["missed"])
        self.assertIn("workshop/press-no-mould", audit["missed"])

    def test_failing_env_log_line_is_counted_as_failure(self):
        """A failing env log line that matches no marker is still
        counted: the learner-episode count is authoritative and the
        two counts disagree (the audit's empirical leg)."""
        from ab.core.learning import Episode
        brain = Brain([], granted={"compute", "environment"})
        brain.learner.record(Episode(1, "heat_alloy", {"at": "orin"}, False))
        log = ["heat_alloy: the coil hums but the substance stays cold"]
        counted = count_failures(log, brain, episodes_before=0,
                                 episodes_available=True)
        self.assertEqual(counted["log_count"], 0)
        self.assertEqual(counted["episode_count"], 1)
        self.assertEqual(counted["authoritative"], 1)
        self.assertEqual(counted["authoritative_source"], "learner_episodes")
        self.assertTrue(counted["disagree"])
        self.assertEqual(counted["unmatched_failure_lines"], log)

    def test_no_learning_falls_back_to_log_count(self):
        """For the no_learning variant the learner is a no-op, so
        episodes are unavailable and the log count is authoritative
        (recorded, not silently zero)."""
        brain = Brain([], granted={"compute", "environment"})
        log = ["go{'room': 'x'}: cannot reach x"]
        counted = count_failures(log, brain, episodes_before=0,
                                 episodes_available=False)
        self.assertEqual(counted["log_count"], 1)
        self.assertEqual(counted["authoritative"], 1)
        self.assertIn("env_log", counted["authoritative_source"])


class TestCarriedBrainDedupe(unittest.TestCase):
    def test_generated_leg_never_double_adds_shared_actions(self):
        """Coordinator integration (P10-56 follow-up): the carried
        brain in ``_run_generated_leg`` already holds world A's
        actions; the B leg must not add the actions identical in
        both worlds a second time. The hard workshop pair (seeds
        0/1) shares exactly 1 action, so the carried brain must
        hold len(A) + len(B) - 1 unique actions after the B leg."""
        factory = _full()
        _rec_a, brain = bench._run_generated_leg(factory, 0, "workshop")
        _rec_b, brain = bench._run_generated_leg(factory, 1, "workshop",
                                                 brain=brain)
        idents = [bench._action_identity(a)
                  for a in brain.planner.actions]
        self.assertEqual(len(idents), len(set(idents)),
                         "carried brain holds a duplicated action")
        _env_a, actions_a, _g_a, _s_a = generate(
            0, "workshop", difficulty="hard")
        _env_b, actions_b, _g_b, _s_b = generate(
            1, "workshop", difficulty="hard")
        shared = ({bench._action_identity(a) for a in actions_a}
                  & {bench._action_identity(a) for a in actions_b})
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(brain.planner.actions),
                         len(actions_a) + len(actions_b) - len(shared))

    def test_action_identity_is_content_canonical(self):
        """Identity ignores dict key order and distinguishes
        actions that differ only in args."""
        a1 = Action("cut", {"has_zinc": True}, {"cut_b": True},
                    args={"material": "zinc"})
        a2 = Action("cut", {"has_zinc": True}, {"cut_b": True},
                    args={"material": "zinc"})
        a3 = Action("cut", {"has_zinc": True}, {"cut_b": True},
                    args={"material": "copper"})
        self.assertEqual(bench._action_identity(a1),
                         bench._action_identity(a2))
        self.assertNotEqual(bench._action_identity(a1),
                            bench._action_identity(a3))


class TestHardEnvgen(unittest.TestCase):
    def test_hard_treasure_deterministic_and_harder(self):
        a = generate(0, "treasure", difficulty="hard")
        b = generate(0, "treasure", difficulty="hard")
        self.assertEqual(a[3], b[3])
        self.assertEqual(a[3]["rooms"], b[3]["rooms"])
        self.assertEqual(len(a[3]["rooms"]), 5)
        self.assertEqual(a[3]["min_steps"], 9)
        self.assertEqual(a[3]["difficulty"]["second_gate"], True)
        self.assertEqual(a[3]["n_decoys"], 1)
        self.assertEqual(a[3]["n_gates"], 2)
        c = generate(1, "treasure", difficulty="hard")
        self.assertNotEqual(a[3]["rooms"], c[3]["rooms"])

    def test_hard_workshop_has_decoy_variant_in_vocabulary(self):
        env, actions, goal, shape = generate(0, "workshop", difficulty="hard")
        self.assertEqual(shape["difficulty"]["decoy_variant"], True)
        self.assertTrue(shape["breakable_tool"])
        # The decoy mould key IS in the world's action vocabulary
        # (a press variant requires it) and so is the correct one.
        pre_keys = {k for a in actions for k in a.pre}
        mould_keys = {k for k in pre_keys if k.startswith("has_") and
                      ("mould" in k or "die" in k or "cast" in k or "form" in k)}
        self.assertGreaterEqual(len(mould_keys), 2)
        press_variants = [a for a in actions if a.name == "press"]
        self.assertEqual(len(press_variants), 3)

    def test_hard_surprise_two_stage(self):
        env, actions, goal, shape = generate(0, "surprise", difficulty="hard")
        self.assertEqual(shape["min_steps"], 3)
        self.assertEqual(shape["key_actions"],
                         ["warm_coil", "stoke_coil", shape["key_actions"][2]])
        self.assertEqual(shape["n_hidden_preconditions"], 2)
        self.assertEqual(generate(0, "surprise", difficulty="hard")[3],
                         shape)

    def test_generate_hard_helper_matches_generate(self):
        for family in ("treasure", "workshop", "surprise"):
            self.assertEqual(generate_hard(0, family)[3],
                             generate(0, family, difficulty="hard")[3])

    def test_baseline_generate_unchanged(self):
        """No difficulty argument keeps the exact Phase 9 baseline
        shape (seed 0 treasure), so existing callers are unaffected."""
        _env, _actions, _goal, shape = generate(0, "treasure")
        self.assertEqual(shape["rooms"], ["talvik", "drelm", "krelm"])
        self.assertEqual(shape["min_steps"], 5)
        self.assertNotIn("difficulty", shape)

    def test_bad_difficulty_and_family_raise(self):
        with self.assertRaises(ValueError):
            generate(0, "treasure", difficulty={"bogus_knob": True})
        with self.assertRaises(ValueError):
            generate(0, "treasure", difficulty="medium")
        with self.assertRaises(ValueError):
            generate(0, "nope", difficulty="hard")

    def test_hard_worlds_solvable_by_full_brain_with_real_failures(self):
        """Hard treasure/surprise complete, and the fresh brain
        actually fails on the way (decoy tried once; two-stage
        warming fails twice) -- the difficulty carries signal."""
        factory = _full()
        for family, min_failed in (("treasure", 1), ("surprise", 2)):
            env, actions, goal, _shape = generate(
                0, family, difficulty="hard")
            rec, _brain = bench._run_brain(
                factory, actions, "hard goal", dict(goal), env)
            self.assertTrue(rec["completed"], f"hard {family}: {rec}")
            self.assertGreaterEqual(rec["failed_attempts"], min_failed,
                                    f"hard {family} showed no failure signal")
            self.assertGreater(rec["failures"]["episode_count"], 0)


class TestBenchmarkTasks(unittest.TestCase):
    def test_navigate_unfamiliar_full_passes_without_undiscovered_plan(self):
        r = bench.run_navigate_unfamiliar(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertTrue(r["legs"]["reached_all_rooms"])
        self.assertTrue(r["legs"]["no_undiscovered_room_planned"])
        self.assertEqual(r["evidence"]["plan_violations"], [])
        self.assertEqual(r["evidence"]["arrived_in_order"],
                         list(bench.NAV_ROOMS))

    def test_find_and_use_resources_full_passes_wasting_one_charge(self):
        r = bench.run_find_and_use_resources(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertEqual(r["evidence"]["wax_attempts"], 1)
        self.assertEqual(r["evidence"]["charges_left"], 0)
        self.assertTrue(r["legs"]["resource_never_exhausted_before_final_step"])

    def test_changing_objective_full_passes_no_stale_step(self):
        r = bench.run_changing_objective(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertTrue(r["legs"]["new_goal_completed"])
        self.assertTrue(r["legs"]["no_stale_plan_step_after_change"])
        self.assertTrue(r["legs"]["useful_progress_kept_key_not_retaken"])
        self.assertEqual(r["evidence"]["stale_steps_after_change"], [])

    def test_unexpected_event_full_passes_with_model_revision(self):
        r = bench.run_unexpected_event(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertTrue(r["evidence"]["event_injected"])
        self.assertGreaterEqual(r["evidence"]["model_revised_events"], 1)
        self.assertGreaterEqual(r["evidence"]["world_changed_events"], 1)

    def test_uncertainty_full_chooses_safe_route(self):
        r = bench.run_planning_under_uncertainty(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertNotIn("cross_rift", r["evidence"]["first_plan"])
        self.assertEqual(r["evidence"]["first_plan_true_expected_success"], 1.0)
        self.assertEqual(r["evidence"]["alternative_true_expected_success"], 0.6)
        self.assertEqual(r["evidence"]["cross_attempts"], 0)

    def test_uncertainty_fresh_gambles_and_fails_the_choice_leg(self):
        """Variant separation: the untrained brain gambles on the
        cross (true expected success 0.6 < safe 1.0), the cross
        fails on seed 0, and the fallback safe route still
        completes the goal -- so the goal leg passes but the
        choice leg fails, and the task fails for fresh."""
        r = bench.run_planning_under_uncertainty(
            _full(), seed=0, variant="fresh")
        self.assertFalse(r["passed"])
        self.assertFalse(r["legs"]["chose_higher_expected_success_plan"])
        self.assertTrue(r["legs"]["goal_completed"])
        self.assertTrue(r["legs"]["fallback_completes_after_uncertain_failure"])
        self.assertEqual(r["evidence"]["first_plan"][0], "cross_rift")
        self.assertEqual(r["evidence"]["cross_attempts"], 1)

    def test_uncertainty_no_learning_passes_via_causal_path(self):
        """Recorded finding: the no_learning brain also chooses the
        safe route. Its learner records nothing (failure rate 0),
        so the pricing comes from the causal model / simulation
        ranking, which IS populated by training outcomes -- not
        from learner episodes."""
        factories = make_variant_factories()
        r = bench.run_planning_under_uncertainty(
            factories["no_learning"], seed=0, variant="no_learning")
        self.assertTrue(r["passed"], r["detail"])
        self.assertEqual(r["evidence"]["cross_failure_rate_at_selection"], 0.0)
        self.assertEqual(r["evidence"]["training"]["cross_failures"], 0)

    def test_transfer_complex_records_both_legs_and_audit_disagreement(self):
        """Transfer is NEGATIVE on the hard workshop pair (continued
        3 failed / 21 cycles vs fresh 2 / 17 at seed 0) -- the test
        asserts the measurement, not a pass: both legs completed,
        the numbers are recorded, and the log-based count (0) is
        shown to disagree with the episode-based count."""
        r = bench.run_transfer_complex(_full(), seed=0, variant="full")
        ev = r["evidence"]
        self.assertTrue(ev["world_a"]["completed"])
        self.assertTrue(ev["world_b_continued"]["completed"])
        self.assertTrue(ev["world_b_fresh"]["completed"])
        self.assertIn("failed_attempts", ev["world_b_continued"])
        self.assertIn("failed_attempts", ev["world_b_fresh"])
        self.assertTrue(
            ev["world_b_continued"]["failures"]["disagree"]
            or ev["world_b_fresh"]["failures"]["disagree"],
            "expected the log count to disagree with the episode count")
        self.assertEqual(ev["world_b_continued"]["failures"]["log_count"], 0)
        self.assertGreater(ev["world_b_continued"]["failures"]["episode_count"], 0)
        self.assertFalse(r["passed"])
        self.assertEqual(ev["world_b_continued"]["failed_attempts"], 3)
        self.assertEqual(ev["world_b_fresh"]["failed_attempts"], 2)

    def test_long_task_interrupt_restart_full_passes(self):
        r = bench.run_long_task_interrupt_restart(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertTrue(r["legs"]["stages_in_order"])
        self.assertTrue(r["legs"]["no_stage_repeated_or_skipped"])
        self.assertTrue(r["legs"]["load_clean"])
        self.assertEqual(r["evidence"]["stage_counts"],
                         {s: 1 for s in bench.LONG_STAGES})

    def test_unrecoverable_failure_stops_within_budget(self):
        r = bench.run_unrecoverable_failure(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertIn(r["evidence"]["stop_reason"], ("impossible", "stalled"))
        self.assertLessEqual(r["cycles"], bench.MAX_CYCLES)
        self.assertEqual(r["evidence"]["unlock_attempts_after_event"], 0)
        self.assertTrue(r["evidence"]["key_taken_before_destruction"])

    def test_competing_goals_decision_recorded_and_lower_suspended(self):
        r = bench.run_competing_goals(_full(), seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        decision = r["evidence"]["decision"]
        self.assertIsNotNone(decision)
        self.assertIn("zarnik", str(decision["chosen"]))
        self.assertIn("priority", decision["reason"])
        self.assertEqual(r["evidence"]["low_goal_status"], "pending")
        self.assertIn("deferred", r["evidence"]["suspended_reason"])
        self.assertEqual(r["evidence"]["charges_left"], 0)

    def test_run_benchmarks_return_contract(self):
        """``run_benchmarks(seed=0)`` returns {"tasks": {name:
        {passed, cycles, detail, evidence, failure_cause,
        variants: {variant: {passed, cycles, detail, evidence,
        failure_cause, legs}}}}} with all nine tasks and all four
        variants, and no task/variant errored."""
        record = run_benchmarks(seed=0)
        self.assertEqual(set(record), {"tasks"})
        self.assertEqual(set(record["tasks"]),
                         {t[0] for t in bench.TASKS})
        self.assertEqual(len(record["tasks"]), 9)
        for name, task in record["tasks"].items():
            for key in ("passed", "cycles", "detail", "evidence",
                        "failure_cause", "variants"):
                self.assertIn(key, task, f"{name} missing {key}")
            self.assertEqual(set(task["variants"]),
                             set(bench.VARIANT_NAMES), name)
            for vname, variant in task["variants"].items():
                for key in ("passed", "cycles", "detail", "evidence",
                            "failure_cause", "legs"):
                    self.assertIn(key, variant, f"{name}/{vname} missing {key}")
                self.assertNotIn("error", variant.get("evidence") or {},
                                 f"{name}/{vname} errored: "
                                 f"{(variant.get('evidence') or {}).get('error')}")
        full_passes = [n for n, t in record["tasks"].items() if t["passed"]]
        self.assertEqual(sorted(full_passes), sorted(
            [n for n in record["tasks"] if n != "transfer_complex"]))


if __name__ == "__main__":
    unittest.main()
