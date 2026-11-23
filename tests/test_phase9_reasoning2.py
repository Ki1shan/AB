"""Phase 9 (Worker 1): tests for the new held-out reasoning tasks
(``ab/eval/reasoning_tasks2.py``) and the novel environment generator
(``ab/eval/envgen.py``).

The full variant must pass all four reasoning tasks. Variant
failures are findings, recorded not hidden: the abstraction task
legitimately fails for ``non_integrated`` (it overrides
``_reason_from_wm``, so no category is ever proposed in-cycle), and
the analogy / causal / counterfactual tasks pass for every variant
because their mechanisms do not depend on the disabled parts —
those are recorded comparisons, not winners.
"""
import os
import re
import unittest

from ab.brain import Brain
from ab.eval.envgen import FAMILIES, generate
from ab.eval.reasoning_tasks import make_variant_factories
from ab.eval.reasoning_tasks2 import (
    TASKS, run_abstraction_task, run_analogy_task, run_causal_task,
    run_counterfactual_task, run_reasoning_tasks2)


class TestReasoningTasks2(unittest.TestCase):
    def test_full_variant_passes_all_four(self):
        factories = make_variant_factories()
        for task_id, runner, _criterion in TASKS:
            with self.subTest(task=task_id):
                result = runner(factories["full"])
                self.assertTrue(result["passed"],
                                f"{task_id} full variant failed: "
                                f"legs={result.get('legs')} "
                                f"detail={result.get('detail')}")

    def test_run_all_variants_records_without_crashing(self):
        record = run_reasoning_tasks2()
        self.assertEqual(set(record["tasks"]), {t[0] for t in TASKS})
        for task_id, section in record["tasks"].items():
            self.assertEqual(set(section["variants"]),
                             {"full", "no_learning", "non_integrated"})
            for variant, result in section["variants"].items():
                self.assertIn("passed", result)
                self.assertNotIn("error",
                                 result.get("evidence") or {},
                                 f"{task_id}/{variant} errored: "
                                 f"{(result.get('evidence') or {}).get('error')}")

    def test_abstraction_legitimately_fails_for_non_integrated(self):
        """Recorded finding, not a test failure: the non-integrated
        variant overrides _reason_from_wm with a no-op, so
        _abstract_categories never runs and no pending category is
        ever proposed. The full and no_learning variants pass."""
        factories = make_variant_factories()
        self.assertTrue(run_abstraction_task(factories["full"])["passed"])
        self.assertTrue(
            run_abstraction_task(factories["no_learning"])["passed"])
        failed = run_abstraction_task(factories["non_integrated"])
        self.assertFalse(failed["passed"])
        self.assertFalse(
            failed["legs"]["pending_category_proposed"])

    def test_causal_task_claims_only_the_reliable_variant(self):
        result = run_causal_task(make_variant_factories()["full"])
        self.assertTrue(result["passed"])
        ev = result["evidence"]
        self.assertEqual(ev["reliable_variant_counts"], (3, 3))
        self.assertEqual(ev["failing_variant_counts"], (0, 1))
        self.assertIsNotNone(ev["reliable_variant_claim"])
        self.assertIsNone(ev["failing_variant_claim"])
        self.assertIsNone(ev["pooled_claim"])

    def test_analogy_hypothesis_never_verified_alone(self):
        result = run_analogy_task(make_variant_factories()["full"])
        self.assertTrue(result["passed"])
        hyp = result["evidence"]["hypothesis"]
        self.assertEqual(hyp["source"], "analogy")
        self.assertLessEqual(hyp["confidence"], 0.4)
        self.assertEqual(hyp["status"], "open")
        self.assertFalse(result["evidence"]["ask_known"])
        self.assertEqual(result["evidence"]["ltm_facts_for_target"], [])

    def test_counterfactual_names_the_changed_fact(self):
        result = run_counterfactual_task(make_variant_factories()["full"])
        self.assertTrue(result["passed"])
        self.assertTrue(result["evidence"]["real_world_model_untouched"])
        self.assertEqual(result["evidence"]["difference"]["has_garnik"],
                         (False, True))


# ------------------------------------------------------------------ envgen
class TestEnvgen(unittest.TestCase):
    def test_deterministic_same_seed(self):
        for family in FAMILIES:
            for seed in (1, 7, 42):
                a = generate(seed, family)
                b = generate(seed, family)
                self.assertEqual(a[0].observe(), b[0].observe())
                self.assertEqual([x.name for x in a[1]],
                                 [x.name for x in b[1]])
                self.assertEqual(a[2], b[2])
                self.assertEqual(a[3], b[3])

    def test_different_seeds_differ(self):
        seen = set()
        for seed in range(1, 8):
            for family in FAMILIES:
                _env, _actions, _goal, shape = generate(seed, family)
                seen.add((family, tuple(shape["rooms"]),
                          shape["min_steps"]))
        # 7 seeds x 3 families, each family varying names/layout.
        self.assertGreater(len(seen), 15)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            generate(1, "nope")

    def test_generated_worlds_are_solvable_by_a_real_brain(self):
        for family in FAMILIES:
            env, actions, goal, shape = generate(7, family)
            brain = Brain(actions, granted={"compute", "environment"})
            brain.add_goal(f"generated {family} 7", dict(goal))
            stop = brain.run(env, brain.cycle + 80)
            self.assertEqual(brain.goals.all()[-1].status, "done",
                             f"{family} seed 7 not solved: stop={stop} "
                             f"log={env.log}")
            # The executed log respects the shape's order constraints
            # and includes its key actions. Order is existential: an
            # early FAILED attempt (e.g. a press tried before the
            # mould was fetched) does not violate a constraint that
            # the successful step comes after its prerequisite.
            names = [e.split(":")[0].split("{")[0].strip()
                     for e in env.log]
            for key_action in shape["key_actions"]:
                self.assertIn(key_action, names,
                              f"{family}: {key_action} never executed")
            for before, after in shape["order_constraints"]:
                ordered = any(
                    names[i] == before and names[j] == after
                    for i in range(len(names)) for j in range(i + 1, len(names)))
                self.assertTrue(
                    ordered,
                    f"{family}: no {after} ever follows {before}")

    def test_generated_worlds_vary_object_counts(self):
        counts = {generate(seed, "workshop")[3]["n_objects"]
                  for seed in range(1, 11)}
        self.assertGreater(len(counts), 1)


# ------------------------------------------------------------------ leakage
class TestReasoningTasks2Leakage(unittest.TestCase):
    # Invented names used by reasoning_tasks2 / envgen. None may
    # appear in ``ab/`` outside ``ab/eval/`` (word-boundary match:
    # "tessera" inside "tesseract" is not a hit).
    NAMES = ("vortan", "drelm", "forvik", "trellin", "vossar", "quivern",
             "norbex", "mellor", "garnik", "krelm", "forge", "forged",
             "catalyst")

    def test_no_task_name_leaks_into_ab_outside_eval(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ab_dir = os.path.join(repo, "ab")
        hits = []
        for root, _dirs, files in os.walk(ab_dir):
            if "eval" in root.split(os.sep):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                for name in self.NAMES:
                    if re.search(r"\b" + re.escape(name) + r"\b", text):
                        hits.append(f"{name} in {path}")
        self.assertEqual(hits, [], f"leaked task names: {hits}")

    def test_zorblat_not_used_by_reasoning_tasks2(self):
        # "zorblat" already appears as an illustrative example in
        # ab/brain.py (outside eval), so this module must not use it
        # as a task name at all.
        import ab.eval.reasoning_tasks2 as mod
        import inspect
        self.assertNotIn("zorblat", inspect.getsource(mod))


if __name__ == "__main__":
    unittest.main()
