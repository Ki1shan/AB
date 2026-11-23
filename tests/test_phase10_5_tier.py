"""Phase 10.5 tier field tests (Worker 3, design P10.5-6).

Covers, per the Phase 10.5 design decision P10.5-6 and the Phase 10
design P10-3:

- the tier-derivation rule: the active-dimension count from each
  task's 7-dimension difficulty vector, and the tier from that
  vector plus the task's world kind (never from the task id
  string). The rule's documented limits are pinned, not hidden:
  ``unexpected_event`` and ``unrecoverable_failure`` share the
  identical difficulty vector yet are Tier 2 and Tier 3 (the
  difference is the world kind), and ``find_and_use_resources`` /
  ``changing_objective`` each show 3 active dimensions yet are
  Tier 2 (incidental structure, not a composite of mechanisms);
- ``run_benchmarks(seed=0)`` records ``"tier"`` (1|2|3) on every
  task record and on every per-variant result, with the Tier-3 set
  exactly the design's three composite tasks, and with the nine
  tasks' pass/fail outcomes and cycle counts unchanged from the
  Phase 10 report SS11 baseline (a recording change only -- any
  number that moved is a defect in the change, so the full
  per-variant table is pinned here);
- demo SS10 selects Tier-3 tasks by the recorded ``tier`` field
  only: the task-id fallback, the criterion-text match and the
  difficulty-vector recount are gone, and a task without a tier
  fails loudly (``ValueError``) instead of being guessed;
- the tier survives the consolidated converter into the record's
  per-task evidence, through the same path ``difficulty`` takes
  (the per-variant results carried in the suite evidence).

No existing test is weakened: the only change to an existing file
is the stub ``ab.eval.benchmarks`` in ``test_phase10_eval.py``,
whose two tierless stub tasks now carry ``"tier": 2`` because the
documented ``run_benchmarks`` contract they honour now records a
tier on every task.
"""
from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import redirect_stdout

from ab.demo import _section10, _tier3_tasks
from ab.eval import benchmarks as bench
from ab.eval.benchmarks import (
    _active_dimensions,
    _kind_of,
    _task_tier,
    _tier_for,
    run_benchmarks,
)
from ab.eval.consolidated import (
    _run_benchmarks_suite,
    build_environment_meta,
)

# Expected tiers (design P10-3/P10-4): the three composite worlds
# are Tier 3; the other six benchmark tasks are Tier 2 (generated
# harder envs and single-mechanism harder worlds). None of the nine
# is a Tier-1 existing world.
EXPECTED_TIERS = {
    "navigate_unfamiliar": 2,
    "find_and_use_resources": 2,
    "changing_objective": 2,
    "unexpected_event": 2,
    "planning_under_uncertainty": 2,
    "transfer_complex": 2,
    "long_task_interrupt_restart": 3,
    "unrecoverable_failure": 3,
    "competing_goals": 3,
}

# Expected active-dimension counts per task (see
# ``_active_dimensions`` for the predicates), with the active
# dimension names. The count alone does not determine the tier --
# see TestTierDerivation.test_count_alone_does_not_determine_tier.
EXPECTED_ACTIVE = {
    "navigate_unfamiliar": (3, ["n_rooms", "observability",
                                "horizon_min_steps"]),
    "find_and_use_resources": (3, ["n_rooms", "n_decoys",
                                    "resource_units"]),
    "changing_objective": (3, ["n_rooms", "n_dynamic_events",
                               "n_goals"]),
    "unexpected_event": (2, ["n_rooms", "n_dynamic_events"]),
    "planning_under_uncertainty": (1, ["n_rooms"]),
    "transfer_complex": (3, ["n_rooms", "n_decoys",
                             "horizon_min_steps"]),
    "long_task_interrupt_restart": (3, ["n_rooms", "n_dynamic_events",
                                        "horizon_min_steps"]),
    "unrecoverable_failure": (2, ["n_rooms", "n_dynamic_events"]),
    "competing_goals": (3, ["n_rooms", "resource_units", "n_goals"]),
}

# Phase 10 report SS11 baseline, seed 0, per variant
# (passed, cycles). Pinned so a recording-only change cannot move
# a number: if any entry here changes, the tier change itself is
# defective, not the brain.
BASELINE = {
    "navigate_unfamiliar": {
        "full": (True, 26), "no_learning": (True, 26),
        "non_integrated": (True, 26), "fresh": (True, 26)},
    "find_and_use_resources": {
        "full": (True, 8), "no_learning": (True, 8),
        "non_integrated": (True, 8), "fresh": (True, 8)},
    "changing_objective": {
        "full": (True, 10), "no_learning": (True, 10),
        "non_integrated": (True, 10), "fresh": (True, 10)},
    "unexpected_event": {
        "full": (True, 10), "no_learning": (True, 10),
        "non_integrated": (True, 10), "fresh": (True, 10)},
    "planning_under_uncertainty": {
        "full": (True, 8), "no_learning": (True, 8),
        "non_integrated": (True, 8), "fresh": (False, 9)},
    "transfer_complex": {
        "full": (False, 23), "no_learning": (False, 21),
        "non_integrated": (False, 23), "fresh": (True, 17)},
    "long_task_interrupt_restart": {
        "full": (True, 25), "no_learning": (True, 25),
        "non_integrated": (True, 25), "fresh": (True, 25)},
    "unrecoverable_failure": {
        # Phase 10.5 gate update (P10.5-7): the loop-plan rule stops
        # the goal as ``impossible`` at cycle 8 instead of stalling
        # at cycle 10 — same pass verdicts, two fewer cycles. The
        # pin moves 10 -> 8 for every variant for that reason, not
        # because the task got easier. EXCEPT no_learning, which
        # stays (True, 10) stalled: the loop rule consumes failure
        # counts, and the no-op learner never records any — the
        # ablation demonstrating the mechanism's value, exactly as
        # the variant is meant to.
        "full": (True, 8), "no_learning": (True, 10),
        "non_integrated": (True, 8), "fresh": (True, 8)},
    "competing_goals": {
        "full": (True, 6), "no_learning": (True, 6),
        "non_integrated": (True, 6), "fresh": (True, 6)},
}

TASK_DIFFICULTY = {
    "navigate_unfamiliar": bench.NAV_DIFFICULTY,
    "find_and_use_resources": bench.RESOURCE_DIFFICULTY,
    "changing_objective": bench.CHANGE_DIFFICULTY,
    "unexpected_event": bench.UNEXPECTED_DIFFICULTY,
    "planning_under_uncertainty": bench.UNCERTAINTY_DIFFICULTY,
    "transfer_complex": bench.TRANSFER_DIFFICULTY,
    "long_task_interrupt_restart": bench.LONG_DIFFICULTY,
    "unrecoverable_failure": bench.UNRECOVERABLE_DIFFICULTY,
    "competing_goals": bench.COMPETING_DIFFICULTY,
}

TASK_RUNNER = {name: runner for name, runner, _c in bench.TASKS}

# One real run_benchmarks(seed=0) shared (read-only) by the tests
# that need the real record -- the run is deterministic per seed
# and takes seconds, so re-running it per test would only burn
# time. Tests that need their own run (the demo section, which
# runs the benchmarks itself) call it directly.
_REAL_RECORD = None


def _real_record():
    global _REAL_RECORD
    if _REAL_RECORD is None:
        _REAL_RECORD = run_benchmarks(seed=0)
    return _REAL_RECORD


class TestTierDerivation(unittest.TestCase):
    def test_active_dimension_counts_match_the_documented_rule(self):
        for name, difficulty in TASK_DIFFICULTY.items():
            count, active = _active_dimensions(difficulty)
            expected_count, expected_active = EXPECTED_ACTIVE[name]
            self.assertEqual(count, expected_count, name)
            self.assertEqual(active, expected_active, name)

    def test_tiers_match_the_p10_3_definitions(self):
        for name, difficulty in TASK_DIFFICULTY.items():
            kind = _kind_of(TASK_RUNNER[name])
            self.assertEqual(_tier_for(difficulty, kind),
                             EXPECTED_TIERS[name], name)

    def test_world_kind_per_task(self):
        kinds = {name: _kind_of(TASK_RUNNER[name])
                 for name in TASK_DIFFICULTY}
        self.assertEqual(kinds, {
            "navigate_unfamiliar": "single",
            "find_and_use_resources": "single",
            "changing_objective": "single",
            "unexpected_event": "single",
            "planning_under_uncertainty": "single",
            "transfer_complex": "generated",
            "long_task_interrupt_restart": "composite",
            "unrecoverable_failure": "composite",
            "competing_goals": "composite",
        })

    def test_identical_vector_different_tier_by_world_kind(self):
        # The proof that the vector alone cannot determine the
        # tier: unexpected_event and unrecoverable_failure share
        # the identical difficulty vector, yet are Tier 2 and
        # Tier 3 by design -- the difference is the world kind
        # (a recoverable gate reset vs the terminal destruction
        # of the only key).
        self.assertEqual(bench.UNEXPECTED_DIFFICULTY,
                         bench.UNRECOVERABLE_DIFFICULTY)
        self.assertEqual(
            _tier_for(bench.UNEXPECTED_DIFFICULTY, "single"), 2)
        self.assertEqual(
            _tier_for(bench.UNRECOVERABLE_DIFFICULTY, "composite"), 3)

    def test_count_alone_does_not_determine_tier(self):
        # find_and_use_resources and changing_objective each show
        # 3 active dimensions -- as many as the Tier-3 tasks --
        # yet are Tier 2: their extra dimensions are incidental
        # structure, not a composite of challenge mechanisms.
        for name in ("find_and_use_resources", "changing_objective"):
            count, _active = _active_dimensions(TASK_DIFFICULTY[name])
            self.assertEqual(count, 3, name)
            self.assertEqual(EXPECTED_TIERS[name], 2, name)
        # And a Tier-3 task shows only 2 active dimensions.
        count, _active = _active_dimensions(bench.UNRECOVERABLE_DIFFICULTY)
        self.assertEqual(count, 2)
        self.assertEqual(EXPECTED_TIERS["unrecoverable_failure"], 3)

    def test_unknown_kind_fails_loudly(self):
        with self.assertRaises(ValueError):
            _tier_for(bench.NAV_DIFFICULTY, "mystery")
        with self.assertRaises(ValueError):
            _tier_for(bench.NAV_DIFFICULTY, "")

    def test_malformed_difficulty_fails_loudly(self):
        with self.assertRaises(ValueError):
            _active_dimensions(None)
        with self.assertRaises(ValueError):
            _active_dimensions({"n_rooms": 5})
        bad = dict(bench.NAV_DIFFICULTY)
        del bad["n_goals"]
        with self.assertRaises(ValueError):
            _active_dimensions(bad)
        bad = dict(bench.NAV_DIFFICULTY)
        bad["observability"] = "foggy"
        with self.assertRaises(ValueError):
            _active_dimensions(bad)
        bad = dict(bench.NAV_DIFFICULTY)
        bad["n_rooms"] = "five"
        with self.assertRaises(ValueError):
            _active_dimensions(bad)
        bad = dict(bench.NAV_DIFFICULTY)
        bad["resource_units"] = "two"
        with self.assertRaises(ValueError):
            _active_dimensions(bad)

    def test_composite_with_fewer_than_two_active_dimensions_fails(self):
        thin = {"n_rooms": 1, "n_decoys": 0, "resource_units": None,
                "n_dynamic_events": 0, "observability": "full",
                "n_goals": 1, "horizon_min_steps": 1}
        self.assertEqual(_active_dimensions(thin)[0], 0)
        with self.assertRaises(ValueError):
            _tier_for(thin, "composite")
        # A single-kind task with no active dimensions is Tier 2,
        # not an error: the count guards composites only.
        self.assertEqual(_tier_for(thin, "single"), 2)
        self.assertEqual(_tier_for(thin, "generated"), 2)
        self.assertEqual(_tier_for(thin, "existing"), 1)

    def test_task_tier_lookup_fails_loudly_for_an_unknown_runner(self):
        def _not_a_benchmark_task(factory, seed=0, variant="full"):
            raise AssertionError("must not run")

        with self.assertRaises(ValueError):
            _task_tier(_not_a_benchmark_task, "not_a_task")

    def test_task_tier_lookup_returns_the_recorded_tier(self):
        for name, runner in TASK_RUNNER.items():
            self.assertEqual(_task_tier(runner, name),
                             EXPECTED_TIERS[name], name)


class TestRunBenchmarksTier(unittest.TestCase):
    def test_every_task_records_a_tier_in_1_2_3(self):
        record = _real_record()
        self.assertEqual(set(record["tasks"]), set(EXPECTED_TIERS))
        for name, task in record["tasks"].items():
            self.assertIn(task["tier"], (1, 2, 3), name)
            self.assertEqual(task["tier"], EXPECTED_TIERS[name], name)

    def test_every_variant_result_carries_the_same_tier(self):
        record = _real_record()
        for name, task in record["tasks"].items():
            for vname, variant in task["variants"].items():
                self.assertEqual(variant.get("tier"), task["tier"],
                                 f"{name}/{vname}")

    def test_numbers_unchanged_from_the_phase10_baseline(self):
        # Recording-only change: pass/fail and cycles must match
        # the Phase 10 report SS11 table exactly.
        record = _real_record()
        for name, variants in BASELINE.items():
            task = record["tasks"][name]
            self.assertEqual(task["passed"], variants["full"][0], name)
            self.assertEqual(task["cycles"], variants["full"][1], name)
            for vname, (passed, cycles) in variants.items():
                variant = task["variants"][vname]
                self.assertEqual(variant["passed"], passed,
                                 f"{name}/{vname}")
                self.assertEqual(variant["cycles"], cycles,
                                 f"{name}/{vname}")
        full_passes = [n for n, t in record["tasks"].items()
                       if t["passed"]]
        self.assertEqual(len(full_passes), 8)
        self.assertNotIn("transfer_complex", full_passes)


class TestDemoSection10Tier(unittest.TestCase):
    def test_tier3_selection_by_field_returns_exactly_the_three(self):
        record = _real_record()
        selected = _tier3_tasks(record["tasks"])
        self.assertEqual(set(selected), {
            "long_task_interrupt_restart",
            "unrecoverable_failure",
            "competing_goals",
        })

    def test_selection_ignores_ids_criterion_and_difficulty(self):
        # A Tier-2 task that looks Tier-3 by every old heuristic
        # (a design Tier-3 id, a "Tier-3" criterion, a 3-active
        # difficulty vector) is NOT selected: only the field counts.
        tasks = {
            "long_task_interrupt_restart": {
                "passed": True, "cycles": 1, "tier": 2,
                "evidence": {"criterion": "a Tier-3 task"},
                "difficulty": {"n_rooms": 5, "n_decoys": 1,
                               "resource_units": 2,
                               "n_dynamic_events": 1,
                               "observability": "partial",
                               "n_goals": 2, "horizon_min_steps": 6}},
            "unrelated_task": {
                "passed": True, "cycles": 1, "tier": 3,
                "evidence": {"criterion": "an ordinary task"},
                "difficulty": {"n_rooms": 1, "n_decoys": 0,
                               "resource_units": None,
                               "n_dynamic_events": 0,
                               "observability": "full",
                               "n_goals": 1, "horizon_min_steps": 1}},
        }
        selected = _tier3_tasks(tasks)
        self.assertEqual(set(selected), {"unrelated_task"})

    def test_missing_tier_fails_loudly_no_fallback(self):
        tasks = {"some_task": {"passed": True, "cycles": 1}}
        with self.assertRaises(ValueError):
            _tier3_tasks(tasks)
        # A tierless task among tiered ones fails the whole
        # selection -- it is never silently skipped.
        tasks = {
            "long_task_interrupt_restart": {
                "passed": True, "cycles": 1, "tier": 3},
            "tierless_task": {"passed": True, "cycles": 1},
        }
        with self.assertRaises(ValueError):
            _tier3_tasks(tasks)

    def test_invalid_tier_or_malformed_record_fails_loudly(self):
        with self.assertRaises(ValueError):
            _tier3_tasks({"t": {"passed": True, "tier": 4}})
        with self.assertRaises(ValueError):
            _tier3_tasks({"t": {"passed": True, "tier": "three"}})
        with self.assertRaises(ValueError):
            _tier3_tasks({"t": "not a record"})

    def test_section10_with_tierless_tasks_reports_and_shows_nothing(self):
        module = types.ModuleType("ab.eval.benchmarks")
        module.__file__ = "<stub for test_phase10_5_tier>"

        def run_benchmarks_stub(seed=0):
            assert seed == 0
            return {"tasks": {
                "long_task_interrupt_restart": {
                    "passed": True, "cycles": 25,
                    "detail": "stub Tier-3 task",
                    "evidence": {"goal": "forge the keepsake"},
                    "failure_cause": None, "tier": 3,
                    "variants": {"full": {"passed": True,
                                          "cycles": 25}}},
                "tierless_task": {
                    "passed": True, "cycles": 1,
                    "detail": "stub task with no tier recorded",
                    "evidence": {}, "failure_cause": None,
                    "variants": {"full": {"passed": True,
                                          "cycles": 1}}},
            }}

        module.run_benchmarks = run_benchmarks_stub
        previous = sys.modules.get("ab.eval.benchmarks")
        sys.modules["ab.eval.benchmarks"] = module
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                _section10()
        finally:
            if previous is None:
                sys.modules.pop("ab.eval.benchmarks", None)
            else:
                sys.modules["ab.eval.benchmarks"] = previous
        out = buf.getvalue()
        self.assertIn("FAILED", out)
        self.assertIn("no tier field", out)
        # No Tier-3 task is shown and no first-task fallback runs.
        self.assertNotIn("Tier-3 task:", out)
        self.assertNotIn("showing the first task instead", out)

    def test_section10_with_the_real_module_shows_exactly_three(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            _section10()
        out = buf.getvalue()
        self.assertEqual(out.count("Tier-3 task:"), 3)
        for name in ("long_task_interrupt_restart",
                     "unrecoverable_failure", "competing_goals"):
            self.assertIn(f"Tier-3 task: {name}", out)


class TestConsolidatedTierFlow(unittest.TestCase):
    def test_tier_survives_the_converter_into_the_record(self):
        # A stub benchmarks module honouring the run_benchmarks
        # contract with tiers on the task records AND on every
        # per-variant result (the path difficulty takes): the
        # converter copies the task-level tier, and the suite
        # evidence carries the per-variant tiers.
        module = types.ModuleType("ab.eval.benchmarks")
        module.__file__ = "<stub for test_phase10_5_tier>"

        def run_benchmarks_stub(seed=0):
            assert seed == 0
            return {"tasks": {
                "stub_tier3_task": {
                    "passed": True, "cycles": 12,
                    "detail": "stub Tier-3 task",
                    "evidence": {"goal": {"done": True}},
                    "failure_cause": None, "tier": 3,
                    "difficulty": {"n_rooms": 5, "n_decoys": 1,
                                   "resource_units": 2,
                                   "n_dynamic_events": 1,
                                   "observability": "partial",
                                   "n_goals": 2,
                                   "horizon_min_steps": 6},
                    "variants": {
                        "full": {"passed": True, "cycles": 12,
                                 "detail": "full passed", "tier": 3,
                                 "difficulty": {
                                     "n_rooms": 5, "n_decoys": 1,
                                     "resource_units": 2,
                                     "n_dynamic_events": 1,
                                     "observability": "partial",
                                     "n_goals": 2,
                                     "horizon_min_steps": 6}},
                        "fresh": {"passed": False, "cycles": 40,
                                  "detail": "fresh failed", "tier": 3}}},
                "stub_tier2_task": {
                    "passed": True, "cycles": 3,
                    "detail": "stub Tier-2 task",
                    "evidence": {}, "failure_cause": None, "tier": 2,
                    "variants": {
                        "full": {"passed": True, "cycles": 3,
                                 "detail": "full passed", "tier": 2}}},
            }}

        module.run_benchmarks = run_benchmarks_stub
        previous = sys.modules.get("ab.eval.benchmarks")
        sys.modules["ab.eval.benchmarks"] = module
        try:
            ctx = {"seed": 0, "fast": False,
                   "meta": build_environment_meta(0), "suites": {}}
            suite = _run_benchmarks_suite(ctx)
        finally:
            if previous is None:
                sys.modules.pop("ab.eval.benchmarks", None)
            else:
                sys.modules["ab.eval.benchmarks"] = previous
        # The raw tasks are kept on the ctx for the harness/summary
        # with their task-level tiers.
        self.assertEqual(ctx["benchmark_tasks"]["stub_tier3_task"]["tier"], 3)
        self.assertEqual(ctx["benchmark_tasks"]["stub_tier2_task"]["tier"], 2)
        # The suite's per-task evidence carries the per-variant
        # tiers -- the same path difficulty takes.
        exps = suite["experiments"]
        tier3_ev = exps["stub_tier3_task/full"]["evidence"]
        self.assertEqual(tier3_ev["variants"]["full"]["tier"], 3)
        self.assertEqual(tier3_ev["variants"]["fresh"]["tier"], 3)
        self.assertEqual(
            tier3_ev["variants"]["full"]["difficulty"]["n_rooms"], 5)
        tier2_ev = exps["stub_tier2_task/full"]["evidence"]
        self.assertEqual(tier2_ev["variants"]["full"]["tier"], 2)


if __name__ == "__main__":
    unittest.main()
