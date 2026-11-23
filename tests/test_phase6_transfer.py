"""Phase 6 tests: two-family transfer, procedural learning, and
success-based learning (design SS6.2, SS6.3, SS6.11; SS7.1, SS7.2).

Each test runs the named experiment from ``ab.eval.learning_eval``
and asserts the criterion itself -- these are acceptance tests, not
measurement-only tests. The numbers they assert were verified by
probing the landed Phase 6 brain on worlds C/D before they were
written; if the brain changes, a failure here is a finding about
the brain, not a reason to weaken the test.
"""
import unittest

from ab.eval.learning_eval import (
    run_procedural_experiment,
    run_success_withheld_experiment,
    run_two_family_transfer,
)


class TestTwoFamilyTransfer(unittest.TestCase):
    def test_at_least_two_pattern_families_transfer_to_held_out_d(self):
        rec = run_two_family_transfer()
        # P-A: learned in C, applied in D (pattern_applied event
        # penalising the decoy). P-B: learned in C, live in D
        # (fragile multiplier 6.0) and the fragile tool is NOT
        # retried after the reset in continued D while fresh D DOES
        # retry it.
        self.assertTrue(rec["passed"], rec)
        self.assertGreaterEqual(rec["n_families_transferred"], 2, rec)
        self.assertIn("unreliable_toolless_opener",
                      rec["families_transferred"], rec)
        self.assertIn("one_use_tool", rec["families_transferred"], rec)
        pa = rec["families"]["unreliable_toolless_opener"]
        self.assertTrue(pa["learned_in_c"], pa)
        self.assertTrue(pa["applied_in_d"], pa)
        self.assertEqual(pa["planner_multiplier_in_d"], 6.0, pa)
        pb = rec["families"]["one_use_tool"]
        self.assertTrue(pb["learned_in_c"], pb)
        self.assertEqual(pb["planner_multiplier_in_d"], 6.0, pb)
        self.assertTrue(pb["fragile_retried_after_reset_fresh_d"], pb)
        self.assertFalse(pb["fragile_retried_after_reset_continued_d"], pb)
        self.assertTrue(pb["fragile_avoided_in_continued_d"], pb)
        # Continued D beats fresh D on the transfer itself.
        self.assertLess(
            rec["world_d_continued"]["failed_attempts"],
            rec["world_d_fresh"]["failed_attempts"], rec)

    def test_pc_pd_do_not_form_in_chamber_worlds_recorded_honestly(self):
        # P-C / P-D are structurally unlearnable in chamber worlds
        # (see the experiment notes): the consolidator queue stays
        # empty and neither family is claimed transferred. This test
        # pins that honest negative result so a future change in the
        # mechanism (or the task family) shows up here instead of
        # silently passing.
        rec = run_two_family_transfer()
        self.assertFalse(rec["families"]["durable_tool_P-C"]["transferred"])
        self.assertFalse(
            rec["families"]["requires_precondition_P-D"]["transferred"])
        self.assertEqual(
            rec["families"]["durable_tool_P-C"]["consolidation_before_d"],
            {"present": True, "candidates": 0, "promoted": 0,
             "rejected": 0, "ids": []})


class TestProceduralLearning(unittest.TestCase):
    def test_run2_beats_run1_with_learning_on(self):
        rec = run_procedural_experiment()
        self.assertTrue(rec["passed"], rec)
        full = rec["renamed_full"]
        self.assertTrue(full["improved_vs_fresh"], full)
        self.assertLess(
            (full["run2_C2_continued"]["failed_attempts"],
             full["run2_C2_continued"]["cycles"]),
            (full["run2_C2_fresh"]["failed_attempts"],
             full["run2_C2_fresh"]["cycles"]), full)

    def test_improvement_disappears_with_learning_off(self):
        # Renamed form (C1 -> C2, same type): pair-bans are
        # name-bound, so only durable learning can carry across the
        # rename. With learning off, the continued run ties fresh.
        rec = run_procedural_experiment()
        nl = rec["renamed_no_learning"]
        self.assertFalse(nl["improved_vs_fresh"], nl)
        self.assertEqual(
            (nl["run2_C2_continued"]["failed_attempts"],
             nl["run2_C2_continued"]["cycles"]),
            (nl["run2_C2_fresh"]["failed_attempts"],
             nl["run2_C2_fresh"]["cycles"]), nl)

    def test_same_name_control_is_ban_driven_not_learner_driven(self):
        # Negative control: same-name run 2 improves EVEN with
        # learning off (pair-bans survive the transient clear), so
        # same-name improvement is not evidence of procedural
        # learning. The renamed form above is the discriminating test.
        rec = run_procedural_experiment()
        full = rec["same_name_control_full"]
        nl = rec["same_name_control_no_learning"]
        self.assertLess(
            (full["run2"]["failed_attempts"], full["run2"]["cycles"]),
            (full["run1"]["failed_attempts"], full["run1"]["cycles"]), full)
        self.assertLess(
            (nl["run2"]["failed_attempts"], nl["run2"]["cycles"]),
            (nl["run1"]["failed_attempts"], nl["run1"]["cycles"]), nl)


class TestSuccessBasedLearning(unittest.TestCase):
    def test_pc_c1_improvement_vanishes_when_successes_withheld(self):
        rec = run_success_withheld_experiment()
        self.assertTrue(rec["passed"], rec)
        full = rec["full"]
        sw = rec["success_withheld"]
        # Full: the C-1 durable_tool concept gains a positive example
        # (repeated durable successes) and continued D beats fresh D.
        self.assertTrue(full["durable_tool_concept_positive"], full)
        self.assertTrue(full["improved_vs_fresh_d"], full)
        self.assertLess(
            full["world_d_continued"]["failed_attempts"],
            rec["fresh_d"]["failed_attempts"], full)
        # The concept answers from its learned examples.
        self.assertEqual(
            full["concept_is_a_durable_tool"]["answer"][0], True, full)
        # Success-withheld: no pattern, no concept positive, no
        # improvement -- continued D ties fresh D.
        self.assertFalse(sw["durable_tool_concept_positive"], sw)
        self.assertEqual(sw["patterns_learned_in_c"]["count"], 0, sw)
        self.assertFalse(sw["improved_vs_fresh_d"], sw)
        self.assertEqual(
            sw["world_d_continued"]["failed_attempts"],
            rec["fresh_d"]["failed_attempts"], sw)


if __name__ == "__main__":
    unittest.main()
