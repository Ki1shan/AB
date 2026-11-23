"""Phase 9 W4: long-horizon task (design §3.5) + combined save/resume
scenario (design §4.4).

The eval itself lives in ``ab/eval/long_horizon.py``; these tests pin
its contract: the 6-stage chain completes in order over >= 25 cycles
with a mid-task interruption and a save/restart/resume leg, and a
brain saved mid-deduction-task resumes to the exact derivation.
"""
import unittest

from ab.eval.long_horizon import (GOAL_TARGET, STAGES,
                                  run_deduction_save_resume,
                                  run_long_horizon_task)


class TestLongHorizonTask(unittest.TestCase):
    def test_chain_completes_in_order_with_interrupt_and_resume(self):
        result = run_long_horizon_task()
        self.assertTrue(result["passed"], result["detail"])
        self.assertGreaterEqual(result["cycles"], 25)
        ev = result["evidence"]
        self.assertEqual(ev["goal_status"], "done")
        self.assertTrue(ev["stages_in_order"])
        self.assertTrue(ev["interrupted"])
        self.assertTrue(ev["resumed"])
        self.assertEqual(ev["load_warnings"], [])
        self.assertEqual(ev["stop_reason"], "done")
        self.assertIsNone(result["failure_cause"])

    def test_stages_are_the_six_named_stages_in_order(self):
        result = run_long_horizon_task()
        positions = result["evidence"]["stage_positions"]
        self.assertEqual(list(positions), list(STAGES))
        values = [positions[s] for s in STAGES]
        self.assertEqual(values, sorted(values))
        # The goal is met only after the last stage: assemble is the
        # final env action and the goal target is has_relic only.
        self.assertEqual(GOAL_TARGET, {"has_relic": True})

    def test_result_uses_the_consolidated_common_format(self):
        result = run_long_horizon_task()
        self.assertEqual(set(result), {"passed", "cycles", "detail",
                                       "evidence", "failure_cause"})
        self.assertIsInstance(result["passed"], bool)
        self.assertIsInstance(result["cycles"], int)


class TestDeductionSaveResume(unittest.TestCase):
    def test_save_mid_deduction_resumes_to_exact_derivation(self):
        result = run_deduction_save_resume()
        self.assertTrue(result["passed"], result["detail"])
        self.assertEqual(result["evidence"]["load_warnings"], [])
        deriv = result["evidence"]["derivation"]
        self.assertEqual((deriv["subject"], deriv["relation"], deriv["value"]),
                         ("velkra", "located_in", "kelmor"))
        self.assertAlmostEqual(deriv["confidence"], 0.9025)


if __name__ == "__main__":
    unittest.main()
