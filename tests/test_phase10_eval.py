"""Phase 10 evaluation tests (Worker 4).

Covers, per the Phase 10 design P10-5 and the task contract:

- the no_causal ablation: ``NoOpCausal`` (a ``CausalModel`` subclass
  that never learns) injected at ``Brain`` construction, so the
  brain, the reasoning engine and the simulator all see the same
  no-op from the start -- no stale references;
- no_causal separation on the ``causal_reasoning`` held-out task
  (full passes, no_causal fails it on exactly the
  ``reliable_variant_claimed`` leg, while the forge goal itself
  still completes -- the brain is not broken, only the causal
  claim is missing);
- the ablations suite recording both ``no_revision`` and
  ``no_causal`` experiments with the same experiment names as
  Phase 9 (no existing experiment renamed);
- the benchmarks suite (suite 19) wiring against the documented
  ``run_benchmarks(seed=0)`` return contract, using a stub
  ``ab.eval.benchmarks`` module -- the real module is Worker 2's
  and may land after these tests; the wiring is verified against
  the contract, never against a guessed implementation. An absent
  module fails the suite with the reason, it never crashes the
  runner;
- the harness Phase 10 section (per-task numbers, mode full;
  fast mode skipped with the reason; absent module recorded, not
  hidden);
- demo section 10 running and exiting 0, with and without the
  benchmarks module;
- the Phase 10 follow-up fixes (P10-5x): the carried-brain
  double-add in ``_run_generic_env`` (actions are deduped by full
  identity, name + pre + eff + args) and the dual failure count
  (env-log count AND learner-episode count via Worker 2's
  ``count_failures``, episode count authoritative), with the
  re-measured transfer legs.

No existing test file is touched; no test is weakened.
"""
from __future__ import annotations

import io
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from ab.brain import Brain
from ab.eval import consolidated, harness
from ab.eval.consolidated import (
    ABLATION_REMOVED,
    _action_identity,
    _no_causal_factory,
    _no_revision_factory,
    _run_ablations_suite,
    _run_benchmarks_suite,
    _run_generic_env,
    _run_transfer_suite,
    build_environment_meta,
    run_leakage_audit,
    summary_text,
)
from ab.eval.reasoning_tasks import make_variant_factories
from ab.eval.reasoning_tasks2 import run_causal_task
from ab.reasoning.causal import CausalModel, NoOpCausal

REPO_ROOT = consolidated.REPO_ROOT


# ------------------------------------------------------------------ stub benchmarks module
def _stub_benchmarks_module():
    """A stub ``ab.eval.benchmarks`` honouring the documented
    ``run_benchmarks(seed=0)`` contract: ``{"tasks": {name:
    {"passed", "cycles", "detail", "evidence", "failure_cause",
    "variants"}}}``. Two tasks: one the full brain passes, one it
    fails (with a variant that passes it, so the suite's
    variant-separation experiment has something to separate); one
    task carries a Tier-3 difficulty vector, one carries no
    per-variant breakdown at all (recorded as its own full-variant
    result). No invented benchmark names are used here -- these are
    stub ids for the wiring test only, never the real benchmark
    task ids."""
    module = types.ModuleType("ab.eval.benchmarks")
    module.__file__ = "<stub for test_phase10_eval>"

    def run_benchmarks(seed):
        assert seed == 0, "consolidated runner passes seed=0"
        return {
            "tasks": {
                "stub_tier3_task": {
                    "passed": True,
                    "cycles": 12,
                    "detail": "stub Tier-3 task passed by the full brain",
                    "evidence": {
                        "goal": {"done": True},
                        "imagined_route": ["step_a", "step_b"],
                        "stop_reason": "done",
                        "failed_attempts": 0},
                    "failure_cause": None,
                    "tier": 3,
                    "difficulty": {
                        "n_rooms": 5, "n_decoys": 1, "resource_units": 2,
                        "n_dynamic_events": 1, "observability": "partial",
                        "n_goals": 2, "horizon_min_steps": 6},
                    "variants": {
                        "full": {"passed": True, "cycles": 12,
                                 "detail": "full passed"},
                        "fresh": {"passed": False, "cycles": 40,
                                  "detail": "fresh failed"}}},
                "stub_failing_task": {
                    "passed": False,
                    "cycles": 60,
                    "detail": "stub task the full brain fails",
                    "evidence": {"stop_reason": "impossible"},
                    "failure_cause": "stub: full brain could not finish",
                    # P10.5-6: the run_benchmarks contract records a
                    # tier on EVERY task (the stub honours that
                    # contract); this one is a Tier-2 task.
                    "tier": 2,
                    "variants": {
                        "full": {"passed": False, "cycles": 60,
                                 "detail": "full failed"},
                        "fresh": {"passed": False, "cycles": 60,
                                  "detail": "fresh failed"}}},
                "stub_no_variants_task": {
                    "passed": True,
                    "cycles": 3,
                    "detail": "stub task with no per-variant breakdown",
                    "evidence": {},
                    # P10.5-6: tier is recorded on every task, even
                    # one with no per-variant breakdown.
                    "tier": 2,
                    "failure_cause": None},
            }
        }

    module.run_benchmarks = run_benchmarks
    return module


class _StubBenchmarks:
    """Context manager installing the stub ``ab.eval.benchmarks``
    module into ``sys.modules`` (removed on exit)."""

    def __enter__(self):
        self.module = _stub_benchmarks_module()
        self.previous = sys.modules.get("ab.eval.benchmarks")
        sys.modules["ab.eval.benchmarks"] = self.module
        return self.module

    def __exit__(self, *exc):
        if self.previous is None:
            sys.modules.pop("ab.eval.benchmarks", None)
        else:
            sys.modules["ab.eval.benchmarks"] = self.previous
        return False


def _meta(seed=0):
    return build_environment_meta(seed)


class _AbsentBenchmarks:
    """Context manager making ``ab.eval.benchmarks`` unimportable
    (``None`` in ``sys.modules`` is the import system's "known
    absent" signal): the direct-import failure paths are exercised
    without touching the real module file. Simply popping the
    entry would re-import the real module from disk."""

    def __enter__(self):
        self._patch = mock.patch.dict(
            sys.modules, {"ab.eval.benchmarks": None})
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)
        return False


# ------------------------------------------------------------------ NoOpCausal
class TestNoOpCausal(unittest.TestCase):
    def test_is_causal_model_subclass_that_never_learns(self):
        noop = NoOpCausal()
        self.assertIsInstance(noop, CausalModel)
        noop.record("forge", "forged", True)
        noop.record("forge", "forged", True)
        noop.record("forge", "forged", True)
        self.assertIsNone(noop.counts("forge", "forged"))
        self.assertIsNone(noop.success_rate("forge", "forged"))
        self.assertIsNone(noop.causes("forge", "forged"))
        self.assertEqual(noop.claims(), [])

    def test_variant_reads_also_report_no_evidence(self):
        noop = NoOpCausal()
        noop.record("forge", "forged", True, variant="with_catalyst")
        self.assertIsNone(
            noop.counts("forge", "forged", variant="with_catalyst"))
        self.assertIsNone(
            noop.causes("forge", "forged", variant="with_catalyst"))
        self.assertEqual(noop.claims(), [])


# ------------------------------------------------------------------ Brain construction-time injection
class TestBrainCausalInjection(unittest.TestCase):
    def test_injected_model_seen_by_brain_engine_and_simulator(self):
        noop = NoOpCausal()
        brain = Brain(causal_model=noop)
        self.assertIs(brain.causal, noop)
        self.assertIs(brain.reasoning.causal, noop)
        self.assertIs(brain.simulator.causal, noop)
        # _get_simulator re-syncs from brain.causal (never a stale copy).
        self.assertIs(brain._get_simulator().causal, noop)

    def test_default_brain_still_builds_the_real_model(self):
        brain = Brain()
        self.assertIsInstance(brain.causal, CausalModel)
        self.assertNotIsInstance(brain.causal, NoOpCausal)
        brain.causal.record("forge", "forged", True)
        self.assertEqual(brain.causal.counts("forge", "forged"), (1, 1))

    def test_no_causal_factory_builds_a_consistent_brain(self):
        brain = _no_causal_factory([], {"compute"})
        self.assertIsInstance(brain.causal, NoOpCausal)
        self.assertIs(brain.reasoning.causal, brain.causal)
        self.assertIs(brain.simulator.causal, brain.causal)
        self.assertIs(brain._get_simulator().causal, brain.causal)


# ------------------------------------------------------------------ no_causal separation
class TestNoCausalSeparation(unittest.TestCase):
    def test_full_passes_causal_reasoning(self):
        factories = make_variant_factories()
        result = run_causal_task(factories["full"])
        self.assertTrue(result["passed"], result["detail"])
        self.assertTrue(result["legs"]["reliable_variant_claimed"])

    def test_no_causal_fails_causal_reasoning_on_the_claim_leg(self):
        result = run_causal_task(_no_causal_factory)
        self.assertFalse(result["passed"])
        self.assertEqual(result["legs"]["reliable_variant_claimed"], False)
        # Exactly one leg fails: the claim. Every other leg holds.
        failed = [k for k, v in result["legs"].items() if not v]
        self.assertEqual(failed, ["reliable_variant_claimed"])
        self.assertEqual(
            result["evidence"]["variant_claims_for_forge"], [])

    def test_no_causal_still_completes_the_forge_goal(self):
        # The ablation removes the causal model, not the brain: the
        # goal itself still completes on all three runs -- only the
        # causal claim is missing.
        result = run_causal_task(_no_causal_factory)
        self.assertTrue(result["legs"]["all_runs_completed"])
        self.assertTrue(result["legs"]["failing_variant_actually_tried"])
        self.assertTrue(result["legs"]["failing_variant_not_claimed"])
        self.assertTrue(result["legs"]["pooled_name_not_claimed"])


# ------------------------------------------------------------------ ablations suite
class TestAblationsSuite(unittest.TestCase):
    def _fake_runner(self, factory):
        # Full and no_revision brains pass; the no_causal brain
        # fails -- the runner reads the factory's causal model, so
        # the suite's wiring (not a canned outcome) decides.
        brain = factory([], {"compute"})
        passed = not isinstance(brain.causal, NoOpCausal)
        return {"passed": passed, "cycles": 1,
                "detail": "fake task: full passes, no_causal fails",
                "evidence": {}}

    def test_suite_records_both_ablations_with_phase9_names_kept(self):
        from ab.eval import reasoning_tasks as rt1
        from ab.eval import reasoning_tasks2 as rt2

        fake_tasks = [("fake_task", self._fake_runner, "fake criterion")]
        ctx = {"seed": 0, "fast": False, "meta": _meta(),
               "suites": {}}
        with (mock.patch.object(rt1, "TASKS", fake_tasks),
              mock.patch.object(rt2, "TASKS", [])):
            suite = _run_ablations_suite(ctx)
        exps = suite["experiments"]
        # Phase 9 experiment names are unchanged...
        self.assertIn("no_revision/discriminating/fake_task", exps)
        self.assertIn("no_revision/separation", exps)
        # ...and no_causal mirrors them exactly.
        self.assertIn("no_causal/discriminating/fake_task", exps)
        self.assertIn("no_causal/separation", exps)
        self.assertTrue(exps["no_revision/discriminating/fake_task"]["passed"])
        self.assertTrue(exps["no_causal/discriminating/fake_task"]["passed"])
        outcomes = ctx["ablation_outcomes"]
        self.assertIn("discriminating/fake_task", outcomes)
        outcome = outcomes["discriminating/fake_task"]
        self.assertTrue(outcome["full_passed"])
        self.assertTrue(outcome["no_revision_passed"])
        self.assertFalse(outcome["no_causal_passed"])
        self.assertFalse(outcome["no_revision_differed"])
        self.assertTrue(outcome["no_causal_differed"])

    def test_ablation_removed_records_both_mechanisms(self):
        self.assertIn("no_revision", ABLATION_REMOVED)
        self.assertIn("no_causal", ABLATION_REMOVED)
        self.assertIn("NoOpCausal", ABLATION_REMOVED["no_causal"])

    def test_summary_text_reports_both_ablations(self):
        record = {"timestamp": "2026-11-21T00:00:00+00:00",
                  "mode": "full", "seed": 0,
                  "environment": {"python_version": "3.x",
                                  "tesseract_version": "t",
                                  "persistence_version": 6,
                                  "line_counts": {"python_lines": 1,
                                                  "python_files": 1},
                                  "test_counts": {"status": "x", "ran": 1}},
                  "suites": {},
                  "overall": {"verdict": "ALL SUITES PASSED",
                              "n_suites_passed": 1, "n_suites_failed": 0,
                              "n_suites": 1, "total_duration_s": 0.0,
                              "all_passed": True},
                  "ablations": {"outcomes": {
                      "reasoning_tasks2/causal_reasoning": {
                          "full_passed": True, "no_revision_passed": True,
                          "no_revision_differed": False,
                          "no_causal_passed": False,
                          "no_causal_differed": True}}}}
        text = summary_text(record)
        self.assertIn("no_causal=FAIL (differed)", text)
        self.assertIn("no_revision=PASS (same)", text)


# ------------------------------------------------------------------ benchmarks suite
class TestBenchmarksSuite(unittest.TestCase):
    def test_suite_registered_after_ablations(self):
        # 19 suites in total; benchmarks runs after ablations and
        # before failure_analysis (which must cover every suite) --
        # the run order, not a bare count, is the contract.
        names = list(consolidated._SUITE_RUNNERS)
        self.assertEqual(len(names), 19)
        self.assertIn("benchmarks", names)
        self.assertLess(names.index("ablations"),
                        names.index("benchmarks"))
        self.assertLess(names.index("benchmarks"),
                        names.index("failure_analysis"))
        self.assertIs(consolidated._SUITE_RUNNERS["benchmarks"],
                      consolidated._run_benchmarks_suite)

    def test_wiring_against_the_documented_contract(self):
        ctx = {"seed": 0, "fast": False, "meta": _meta(), "suites": {}}
        with _StubBenchmarks():
            suite = _run_benchmarks_suite(ctx)
        self.assertEqual(suite["suite"], "benchmarks")
        self.assertEqual(suite["seed"], 0)
        self.assertEqual(suite["mode"], "full")
        exps = suite["experiments"]
        self.assertIn("stub_tier3_task/full", exps)
        self.assertIn("stub_failing_task/full", exps)
        self.assertIn("stub_no_variants_task/full", exps)
        self.assertTrue(exps["stub_tier3_task/full"]["passed"])
        self.assertFalse(exps["stub_failing_task/full"]["passed"])
        # The no-variants task is recorded as its own full result.
        self.assertTrue(exps["stub_no_variants_task/full"]["passed"])
        # Per-variant numbers live in the evidence, no winner declared.
        evidence = exps["stub_tier3_task/full"]["evidence"]
        self.assertEqual(evidence["variants"]["fresh"]["passed"], False)
        # Separation: the stub tasks separate full from fresh.
        self.assertIn("variant_separation", exps)
        self.assertTrue(exps["variant_separation"]["passed"])
        # The raw tasks are kept on the ctx for the harness/summary.
        self.assertIn("stub_tier3_task", ctx["benchmark_tasks"])

    def test_absent_module_fails_the_suite_without_crashing(self):
        ctx = {"seed": 0, "fast": False, "meta": _meta(), "suites": {}}
        with _AbsentBenchmarks():
            suite = _run_benchmarks_suite(ctx)
        self.assertEqual(suite["n_failed"], 1)
        self.assertIn("benchmarks/module_present", suite["experiments"])
        exp = suite["experiments"]["benchmarks/module_present"]
        self.assertFalse(exp["passed"])
        # Direct wiring: the failure cause says the import failed.
        self.assertIn("import failed", exp["failure_cause"])

    def test_bad_result_shape_fails_the_suite_without_crashing(self):
        module = types.ModuleType("ab.eval.benchmarks")
        module.run_benchmarks = lambda seed=0: {"not_tasks": {}}
        ctx = {"seed": 0, "fast": False, "meta": _meta(), "suites": {}}
        previous = sys.modules.get("ab.eval.benchmarks")
        sys.modules["ab.eval.benchmarks"] = module
        try:
            suite = _run_benchmarks_suite(ctx)
        finally:
            if previous is None:
                sys.modules.pop("ab.eval.benchmarks", None)
            else:
                sys.modules["ab.eval.benchmarks"] = previous
        self.assertEqual(suite["n_failed"], 1)
        self.assertTrue(any("suite_error" in name or "benchmarks" in name
                            for name in suite["experiments"]))


# ------------------------------------------------------------------ harness Phase 10
class TestHarnessPhase10(unittest.TestCase):
    def test_section_runs_and_records_mode_full(self):
        with _StubBenchmarks():
            section = harness._run_phase10(fast=False)
        self.assertEqual(section["mode"], "full")
        self.assertEqual(section["seed"], 0)
        self.assertIn("stub_tier3_task", section["tasks"])
        task = section["tasks"]["stub_tier3_task"]
        self.assertTrue(task["passed"])
        self.assertEqual(task["cycles"], 12)
        self.assertEqual(task["variants"]["fresh"]["passed"], False)
        self.assertEqual(task["tier"], 3)

    def test_fast_mode_skips_with_the_reason(self):
        section = harness._run_phase10(fast=True)
        self.assertTrue(section["skipped"])
        self.assertIn("fast mode", section["reason"])

    def test_absent_module_is_recorded_not_hidden(self):
        with _AbsentBenchmarks():
            section = harness._run_phase10(fast=False)
        self.assertIn("error", section)
        self.assertIn("benchmarks", section["error"])

    def test_summary_text_reports_benchmark_tasks(self):
        with _StubBenchmarks():
            section = harness._run_phase10(fast=False)
        record = {"timestamp": "2026-11-21T00:00:00+00:00",
                  "mode": "full",
                  "tests": {"ran": 1, "failures": 0, "errors": 0,
                            "skipped": 0, "ok": True, "scope": "stub",
                            "duration_s": 0.0, "exit_code": 0},
                  "demo": {"exit_code": 0, "outcome": "done",
                           "duration_s": 0.0},
                  "disconnected": {},
                  "baselines": {"treasure": {}},
                  "resources": {"total_wall_s": 0.0, "suite_wall_s": 0.0,
                                "demo_wall_s": 0.0, "disconnected_wall_s": 0.0,
                                "peak_child_rss_kb": "not measured"},
                  "repo_state": {"vcs": "none", "python_files": 1,
                                 "test_files": 1},
                  "phase10_evaluation": section}
        text = harness._summary_text(record)
        self.assertIn("Phase 10 benchmarks", text)
        self.assertIn("stub_tier3_task", text)
        self.assertIn("full=PASS (12 cycles)", text)
        self.assertIn("fresh=FAIL (40 cycles)", text)


# ------------------------------------------------------------------ demo section 10
class TestDemoSection10(unittest.TestCase):
    def test_section10_shows_tier3_task_end_to_end(self):
        from ab.demo import _section10

        with _StubBenchmarks():
            buf = io.StringIO()
            with redirect_stdout(buf):
                _section10()
        out = buf.getvalue()
        self.assertIn("Tier-3 task: stub_tier3_task", out)
        self.assertIn("goal:", out)
        self.assertIn("imagined route:", out)
        self.assertIn("step_a", out)
        self.assertIn("execution:", out)
        self.assertIn("outcome: PASS", out)

    def test_section10_without_module_reports_and_does_not_raise(self):
        from ab.demo import _section10

        with _AbsentBenchmarks():
            buf = io.StringIO()
            with redirect_stdout(buf):
                _section10()
        self.assertIn("not landed yet", buf.getvalue())

    def test_demo_exits_zero_with_section_10_present(self):
        proc = subprocess.run(
            [sys.executable, "-m", "ab.demo"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("10. Complex real-world environments", proc.stdout)


# ------------------------------------------------------------------ leakage-audit refactor
class TestLeakageNamesRefactor(unittest.TestCase):
    def test_reasoning2_names_unchanged_by_the_refactor(self):
        from ab.eval.consolidated import _new_task_names_from_reasoning2

        names = _new_task_names_from_reasoning2()
        for name in ("vortan", "mellor", "norbex", "garnik",
                     "trellin", "vossar", "krelm", "forvik"):
            self.assertIn(name, names)

    def test_benchmark_names_harvested_and_audit_clean(self):
        # Worker 2's module has landed: its invented entity names
        # are harvested for the audit, the generic evaluation
        # vocabulary is not, and the audit stays clean.
        from ab.eval.consolidated import _new_task_names_from_benchmarks

        names = _new_task_names_from_benchmarks()
        for name in ("quendal", "brelm", "zarnik", "phelik", "vessik"):
            self.assertIn(name, names)
        for generic in ("full", "fresh", "partial", "observability",
                        "focused"):
            self.assertNotIn(generic, names)
        audit = run_leakage_audit()
        self.assertEqual(audit["hits"], {})
        self.assertEqual(audit["n_leaks"], 0)
        for name in ("quendal", "zarnik"):
            self.assertIn(name, audit["names_checked"])


# ------------------------------------------------------------------ real benchmarks module (Worker 2)
class TestRealBenchmarks(unittest.TestCase):
    """Wiring verification against the real ``ab.eval.benchmarks``
    (landed during this work). These tests assert the wiring, not
    Worker 2's results: per-task outcomes are checked for
    consistency with the raw module output, never pinned to a
    particular pass/fail -- the measured numbers live in
    docs/phase10-progress.md, not in assertions that would break
    when Worker 2 improves a task."""

    def test_benchmarks_suite_against_the_real_module(self):
        from ab.eval.benchmarks import TASKS as real_tasks

        ctx = {"seed": 0, "fast": False, "meta": _meta(), "suites": {}}
        suite = _run_benchmarks_suite(ctx)
        self.assertEqual(suite["suite"], "benchmarks")
        self.assertEqual(suite["seed"], 0)
        self.assertEqual(suite["mode"], "full")
        exps = suite["experiments"]
        # One /full experiment per real task, named by task id.
        self.assertEqual(len(real_tasks), 9)
        for task_id, _runner, _criterion in real_tasks:
            self.assertIn(f"{task_id}/full", exps)
        # Each /full experiment agrees with the raw task result, and
        # the per-variant numbers are recorded in the evidence.
        # (Only scalars are compared: the suite stores evidence
        # JSON-normalised with a depth cap, so deep structures are
        # stringified at different depths on the two sides -- the
        # numbers that matter are the per-variant pass/cycles.)
        for task_id, task in ctx["benchmark_tasks"].items():
            exp = exps[f"{task_id}/full"]
            self.assertEqual(exp["passed"], bool(task["passed"]))
            self.assertEqual(exp["cycles"], task["cycles"])
            for vname, vresult in task["variants"].items():
                stored = exp["evidence"]["variants"][vname]
                self.assertEqual(stored["passed"],
                                 bool(vresult["passed"]),
                                 f"{task_id}/{vname}")
                self.assertEqual(stored["cycles"], vresult["cycles"],
                                 f"{task_id}/{vname}")
        # At least one task separates the variants (the suite's
        # separation experiment is the check for that).
        self.assertIn("variant_separation", exps)
        self.assertTrue(exps["variant_separation"]["passed"])

    def test_harness_phase10_against_the_real_module(self):
        section = harness._run_phase10(fast=False)
        self.assertEqual(section["mode"], "full")
        self.assertEqual(section["seed"], 0)
        self.assertEqual(len(section["tasks"]), 9)
        for name, task in section["tasks"].items():
            self.assertIn("passed", task)
            self.assertIn("variants", task)
            self.assertIn("full", task["variants"])

    def test_demo_section10_against_the_real_module(self):
        from ab.demo import _section10

        buf = io.StringIO()
        with redirect_stdout(buf):
            _section10()
        out = buf.getvalue()
        # The design's Tier-3 composite tasks are the ones shown.
        self.assertIn("Tier-3 task: long_task_interrupt_restart", out)
        self.assertIn("Tier-3 task: unrecoverable_failure", out)
        self.assertIn("Tier-3 task: competing_goals", out)
        self.assertIn("goal:", out)
        self.assertIn("execution:", out)
        self.assertIn("outcome:", out)

    def test_benchmark_numbers_reproduce_except_goal_ids(self):
        # Same seed twice: per-task passed / cycles / detail are
        # identical. The one known difference is the process-global
        # goal counter (competing_goals evidence decision
        # chosen_goal_id) -- the Phase 9 precedent: goal ids are
        # process bookkeeping, not measurements, and are excluded
        # here for exactly that reason.
        from ab.eval.benchmarks import run_benchmarks

        first = run_benchmarks(seed=0)["tasks"]
        second = run_benchmarks(seed=0)["tasks"]
        self.assertEqual(set(first), set(second))
        for name in first:
            self.assertEqual(first[name]["passed"],
                             second[name]["passed"], name)
            self.assertEqual(first[name]["cycles"],
                             second[name]["cycles"], name)
            self.assertEqual(first[name]["detail"],
                             second[name]["detail"], name)
            for vname in first[name]["variants"]:
                self.assertEqual(
                    first[name]["variants"][vname]["passed"],
                    second[name]["variants"][vname]["passed"],
                    f"{name}/{vname}")


# ------------------------------------------------------------------ P10-5x: double-add fix + dual failure count
class TestActionIdentity(unittest.TestCase):
    def test_same_name_pre_eff_args_is_the_same_action(self):
        from ab.planning.planner import Action

        a = Action(name="press", pre={"has_mould": True},
                   eff={"pressed": True}, args={"mould": "zinc"})
        b = Action(name="press", pre={"has_mould": True},
                   eff={"pressed": True}, args={"mould": "zinc"})
        self.assertEqual(_action_identity(a), _action_identity(b))

    def test_dict_key_order_does_not_matter(self):
        from ab.planning.planner import Action

        a = Action(name="cut", pre={"a": 1, "b": 2}, eff={},
                   args={"x": "zinc", "y": "iron"})
        b = Action(name="cut", pre={"b": 2, "a": 1}, eff={},
                   args={"y": "iron", "x": "zinc"})
        self.assertEqual(_action_identity(a), _action_identity(b))

    def test_different_args_is_a_different_action(self):
        from ab.planning.planner import Action

        a = Action(name="press", pre={}, eff={}, args={"mould": "zinc"})
        b = Action(name="press", pre={}, eff={}, args={"mould": "iron"})
        self.assertNotEqual(_action_identity(a), _action_identity(b))

    def test_different_pre_is_a_different_action(self):
        from ab.planning.planner import Action

        a = Action(name="press", pre={"has_mould": True}, eff={})
        b = Action(name="press", pre={"has_mould": False}, eff={})
        self.assertNotEqual(_action_identity(a), _action_identity(b))


class TestGenericEnvDedupe(unittest.TestCase):
    def _carried(self, family):
        from ab.eval.envgen import generate

        factory = make_variant_factories()["full"]
        env_a, actions_a, goal_a, _shape = generate(0, family)
        carried = factory(actions_a, {"compute", "environment"})
        return factory, carried, (env_a, actions_a, goal_a)

    def test_readding_the_same_actions_adds_nothing(self):
        # The double-add bug: the carried brain was built with
        # actions_a, then _run_generic_env re-added actions_a, so
        # the workshop planner held 15 actions twice (30 total).
        factory, carried, (env_a, actions_a, goal_a) = self._carried(
            "workshop")
        self.assertEqual(len(carried.planner.actions), 15)
        rec = _run_generic_env(env_a, actions_a, goal_a, factory,
                               brain=carried)
        self.assertEqual(rec["actions_added"], 0)
        self.assertEqual(rec["actions_skipped_duplicate"], 15)
        self.assertEqual(len(carried.planner.actions), 15)
        ids = [_action_identity(a) for a in carried.planner.actions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_b_leg_adds_only_new_actions(self):
        # Workshop seeds 0/1 share no identical action, so the B leg
        # adds all 12 of its actions and the carried planner ends at
        # 15 + 12 unique actions, each exactly once.
        from ab.eval.envgen import generate
        from ab.eval.transfer import prepare_brain_for_new_world

        factory, carried, (env_a, actions_a, goal_a) = self._carried(
            "workshop")
        _run_generic_env(env_a, actions_a, goal_a, factory,
                         brain=carried)
        prepare_brain_for_new_world(carried)
        env_b, actions_b, goal_b, _shape = generate(1, "workshop")
        rec_b = _run_generic_env(env_b, actions_b, goal_b, factory,
                                 brain=carried)
        self.assertEqual(rec_b["actions_added"], 12)
        self.assertEqual(rec_b["actions_skipped_duplicate"], 0)
        self.assertEqual(len(carried.planner.actions), 27)
        ids = [_action_identity(a) for a in carried.planner.actions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_b_leg_skips_actions_shared_with_a(self):
        # Treasure seeds 0/1 share exactly one identical action; it
        # is skipped on the B leg, the other 6 are added.
        from ab.eval.envgen import generate
        from ab.eval.transfer import prepare_brain_for_new_world

        factory, carried, (env_a, actions_a, goal_a) = self._carried(
            "treasure")
        _run_generic_env(env_a, actions_a, goal_a, factory,
                         brain=carried)
        prepare_brain_for_new_world(carried)
        env_b, actions_b, goal_b, _shape = generate(1, "treasure")
        rec_b = _run_generic_env(env_b, actions_b, goal_b, factory,
                                 brain=carried)
        self.assertEqual(rec_b["actions_added"], 6)
        self.assertEqual(rec_b["actions_skipped_duplicate"], 1)
        ids = [_action_identity(a) for a in carried.planner.actions]
        self.assertEqual(len(ids), len(set(ids)))


class TestGenericEnvFailureCounting(unittest.TestCase):
    def _leg(self, family, seed=0):
        from ab.eval.envgen import generate

        factory = make_variant_factories()["full"]
        env, actions, goal, _shape = generate(seed, family)
        return _run_generic_env(env, actions, goal, factory)

    def _continued_b(self, family):
        from ab.eval.envgen import generate
        from ab.eval.transfer import prepare_brain_for_new_world

        factory = make_variant_factories()["full"]
        env_a, actions_a, goal_a, _shape = generate(0, family)
        carried = factory(actions_a, {"compute", "environment"})
        rec_a = _run_generic_env(env_a, actions_a, goal_a, factory,
                                 brain=carried)
        prepare_brain_for_new_world(carried)
        env_b, actions_b, goal_b, _shape = generate(1, family)
        rec_b = _run_generic_env(env_b, actions_b, goal_b, factory,
                                 brain=carried)
        env_f, actions_f, goal_f, _shape = generate(1, family)
        rec_f = _run_generic_env(env_f, actions_f, goal_f, factory)
        return rec_a, rec_b, rec_f

    def test_surprise_cold_coil_counted_from_episodes_not_the_log(self):
        # The P10-2 audit's missed line: the env reports the
        # cold-coil heat as ok=True with no effect, so no
        # FAILURE_MARKER matches it and the log count is 0 -- the
        # learner episodes record the failure. Measured seed 0:
        # log 0, episodes 1.
        rec = self._leg("surprise")
        failures = rec["failures"]
        self.assertEqual(failures["log_count"], 0)
        self.assertEqual(failures["episode_count"], 1)
        self.assertEqual(failures["authoritative"], 1)
        self.assertEqual(rec["failed_attempts"], 1)
        self.assertEqual(failures["authoritative_source"],
                         "learner_episodes")
        self.assertTrue(failures["disagree"])
        self.assertTrue(any("stays cold" in line
                            for line in
                            failures["unmatched_failure_lines"]))

    def test_workshop_continued_undercount_corrected(self):
        # Continued workshop B, seed 0: the log sees 0 failures, the
        # episodes see 0 — the continued leg now has NO failures at
        # all (the P10.5-9 current-world tie-break avoids the zinc
        # failure entirely; the press failure was already avoided by
        # the W2 adopted rate). The authoritative count is the episode
        # count, and the undercount phenomenon this test pins (log <
        # episodes, unmatched lines exactly the missed failures) is
        # now demonstrated by the FRESH leg alone (log 0, episodes 1:
        # press-no-form). The continued leg pins the other side: a
        # leg with no failures counts 0 both ways.
        # §9 pin update (Phase 10.5 W1, re-measured seed 0): was log
        # 2 / episodes 7 with four "nothing to cut here" lines. The
        # variant-aware ban (cut-zinc banned, cut-tin free) cut the
        # repeated zinc failures from four to one, so the leg now
        # completes in 16 cycles instead of stalling at 24 — the
        # undercount phenomenon this test pins (log < episodes, and
        # the unmatched lines are exactly the missed failures) is
        # unchanged, only the numbers moved with the fix.
        # §9 pin update (Phase 10.5 W2, re-measured seed 0): the
        # adopted variant-aware rate (P10.5-8 key-shape fallback)
        # steers the continued brain to the mould plan, so the
        # press-no-form failure is avoided as well — episodes 2 -> 1
        # (the remaining failure is the single cut-zinc line) and
        # the leg completes in 14 cycles, tying fresh. The
        # undercount phenomenon is unchanged (log 0 < episodes 1).
        # §9 pin update (Phase 10.5 coordinator, re-measured seed 0):
        # the P10.5-9 tie-break avoids the zinc failure too, so the
        # continued leg is 0 failed / 14 cycles — it now BEATS fresh
        # (0 vs 1), which is gate criterion (a). The undercount is
        # pinned by the fresh leg; the continued leg pins 0/0.
        _rec_a, rec_b, rec_f = self._continued_b("workshop")
        self.assertEqual(rec_b["failures"]["log_count"], 0)
        self.assertEqual(rec_b["failures"]["episode_count"], 0)
        self.assertEqual(rec_b["failed_attempts"], 0)
        unmatched = rec_b["failures"]["unmatched_failure_lines"]
        self.assertFalse(any("has no form seated" in line
                             for line in unmatched))
        self.assertEqual(sum("nothing to cut here" in line
                             for line in unmatched), 0)
        # The fresh B leg hides one failure the same way (log 0,
        # episodes 1: press-no-form).
        self.assertEqual(rec_f["failures"]["log_count"], 0)
        self.assertEqual(rec_f["failures"]["episode_count"], 1)
        self.assertEqual(rec_f["failed_attempts"], 1)

    def test_treasure_has_no_hidden_failures(self):
        # Control: treasure legs count 0 both ways -- the 0/0 tie
        # is genuine, not a measurement artefact (unlike surprise).
        rec = self._leg("treasure")
        self.assertEqual(rec["failures"]["log_count"], 0)
        self.assertEqual(rec["failures"]["episode_count"], 0)
        self.assertEqual(rec["failed_attempts"], 0)
        self.assertFalse(rec["failures"]["disagree"])
        self.assertEqual(rec["failures"]["unmatched_failure_lines"], [])


class TestTransferSuiteAfterFixes(unittest.TestCase):
    def test_suite_reports_dual_counts_and_new_numbers(self):
        ctx = {"seed": 0, "fast": False, "meta": _meta(), "suites": {}}
        suite = _run_transfer_suite(ctx)
        exps = suite["experiments"]
        self.assertEqual(len(exps), 4)
        # Verdicts: handbuilt passes; treasure ties (0/0);
        # workshop now PASSES the strict criterion — continued 0
        # failed vs fresh 1 (gate criterion (a) met); surprise ties
        # (1/1). The workshop verdict changed at the gate (was
        # False through W2); the other three are unchanged.
        self.assertTrue(exps["transfer_handbuilt"]["passed"])
        self.assertFalse(
            exps["transfer_generated_envgen/treasure"]["passed"])
        self.assertTrue(
            exps["transfer_generated_envgen/workshop"]["passed"])
        self.assertFalse(
            exps["transfer_generated_envgen/surprise"]["passed"])
        # Re-measured numbers (seed 0): workshop continued 2 failed
        # (was 7) vs fresh 1 (was 0); surprise 1 vs 1 (was 0 vs 0);
        # treasure still 0 vs 0. Cycles and completion unchanged.
        # §9 pin update (Phase 10.5 W2, re-measured seed 0): the
        # adopted variant-aware rate (P10.5-8 key-shape fallback)
        # steers the continued brain to the mould plan, so the
        # press-no-form failure is avoided — workshop continued
        # 2 -> 1 failed (the remaining failure is the single
        # cut-zinc line, the coordinator's P10.5-9 tie-break) and
        # 16 -> 14 cycles, tying fresh on failures and cycles.
        # Verdicts are unchanged — workshop still fails the strict
        # criterion (continued 1 is not < fresh 1), so the suite
        # still reports it as not passed; the remaining gap is the
        # one zinc failure, i.e. P10.5-9, not a regression here.
        # (The W1 note above still describes how the 7 -> 2
        # improvement was won.)
        # §9 pin update (Phase 10.5 coordinator, re-measured seed 0):
        # the P10.5-9 current-world tie-break avoids the zinc failure
        # too — workshop continued 1 -> 0 failed, still completed.
        # Continued 0 < fresh 1, so the workshop experiment now
        # PASSES the strict criterion (gate (a) met) and the transfer
        # suite improves 1/4 -> 2/4 experiments. Cycles drop 14 -> 12
        # for the continued leg: avoiding the zinc failure also saves
        # the two wasted go-back cycles (krelm -> forvik -> krelm);
        # fresh stays 14 (it still fails the press once).
        workshop = exps["transfer_generated_envgen/workshop"]["evidence"]
        self.assertEqual(
            workshop["world_b_continued"]["failed_attempts"], 0)
        self.assertEqual(workshop["world_b_fresh"]["failed_attempts"], 1)
        self.assertEqual(workshop["world_b_continued"]["cycles"], 12)
        self.assertEqual(workshop["world_b_fresh"]["cycles"], 14)
        self.assertTrue(workshop["world_b_continued"]["completed"])
        self.assertTrue(workshop["world_b_fresh"]["completed"])
        surprise = exps["transfer_generated_envgen/surprise"]["evidence"]
        self.assertEqual(
            surprise["world_b_continued"]["failed_attempts"], 1)
        self.assertEqual(surprise["world_b_fresh"]["failed_attempts"], 1)
        treasure = exps["transfer_generated_envgen/treasure"]["evidence"]
        self.assertEqual(
            treasure["world_b_continued"]["failed_attempts"], 0)
        self.assertEqual(treasure["world_b_fresh"]["failed_attempts"], 0)
        # Both counts are recorded in the evidence.
        # (§9, Phase 10.5 W2: was episode 2 / log 0 after W1 — see
        # the comment above; the press failure is now avoided, so
        # the episode count is 1 and the log still undercounts it.)
        # (§9, Phase 10.5 coordinator: the tie-break avoids the
        # zinc failure too, so the continued episode count is now 0
        # and the log agrees (0) — no failure to undercount.)
        self.assertEqual(
            workshop["world_b_continued"]["failures"]["episode_count"], 0)
        self.assertEqual(
            workshop["world_b_continued"]["failures"]["log_count"], 0)


if __name__ == "__main__":
    unittest.main()
