"""Phase 10.5 (P10.5-7) — loop-plan rule: a genuinely unrecoverable
failure stops as ``impossible``, not ``stalled``.

The measured trace (coordinator, 2026-11-22): in the unrecoverable
benchmark the key is destroyed by an env event, ``take_vaskel`` then
fails twice in the identical situation ``{at: brelm,
has_vaskel: False}``, and the P4-9 last-resort relaxation twice
adopts a go-away-and-back plan that retries the failed step in the
identical situation — then the stall detector fires (cycle 10) and
the stop reason is ``stalled``. A plan that retries a twice-failed
(action, args, situation) triple in the identical situation is a
loop, not a recovery attempt; when every available plan is a loop
plan the goal fails as ``impossible``. The simulation through the
prefix effects is the guard: a plan whose earlier steps change the
world before the retry (warming a cold coil before heating) is NOT
a loop. Stdlib only, deterministic.
"""
from __future__ import annotations

import unittest

from ab.brain import Brain
from ab.core.learning import Episode
from ab.eval import benchmarks as bench
from ab.eval.reasoning_tasks import make_variant_factories
from ab.planning.planner import Action, Plan, Planner

TAKE = Action("take_vaskel", {"at": "brelm", "has_vaskel": False},
              {"has_vaskel": True}, cost=0.5, args={"_do": "take_vaskel"})
UNLOCK = Action("unlock", {"at": "brelm", "has_vaskel": True},
                {"door_open": True}, cost=0.5, args={"_do": "unlock"})
GO_OUT = Action("go_quendal_from_brelm", {"at": "brelm"},
                {"at": "quendal"}, cost=1.0,
                args={"_do": "go", "room": "quendal"})
GO_BACK = Action("go_brelm_from_quendal", {"at": "quendal"},
                 {"at": "brelm"}, cost=1.0,
                 args={"_do": "go", "room": "brelm"})
SITUATION = {"at": "brelm", "has_vaskel": False}

HEAT = Action("heat_alloy", {"at": "orin"}, {"alloy_hot": True},
              cost=1.0, args={"_do": "heat_alloy"})
WARM = Action("warm_coil", {"at": "orin", "coil": "cold"},
              {"coil": "warm"}, cost=1.0, args={"_do": "warm_coil"})


def _brain_with(actions, world):
    b = Brain(list(actions), granted={"compute", "environment"})
    b.world = dict(world)
    return b


def _fail_twice(brain, action, situation):
    for cycle in (1, 2):
        brain.learner.record(Episode(
            cycle, action.name, dict(situation), False, [],
            args=dict(action.args)))


class TestLoopPlanDetection(unittest.TestCase):
    def test_retry_after_two_identical_failures_is_a_loop(self):
        b = _brain_with([TAKE, UNLOCK], dict(SITUATION))
        _fail_twice(b, TAKE, SITUATION)
        self.assertTrue(b._plan_is_loop_on_twice_failed_step(
            Plan([TAKE, UNLOCK], 1.0)))

    def test_one_failure_is_not_a_loop(self):
        b = _brain_with([TAKE, UNLOCK], dict(SITUATION))
        b.learner.record(Episode(1, TAKE.name, dict(SITUATION), False, [],
                                 args=dict(TAKE.args)))
        self.assertFalse(b._plan_is_loop_on_twice_failed_step(
            Plan([TAKE, UNLOCK], 1.0)))

    def test_go_away_and_back_then_retry_is_still_a_loop(self):
        # The measured unrecoverable shape: the prefix changes
        # ``at`` away and back, so the take runs in the identical
        # situation it failed in.
        b = _brain_with([TAKE, UNLOCK, GO_OUT, GO_BACK], dict(SITUATION))
        _fail_twice(b, TAKE, SITUATION)
        self.assertTrue(b._plan_is_loop_on_twice_failed_step(
            Plan([GO_OUT, GO_BACK, TAKE, UNLOCK], 4.0)))

    def test_warming_first_is_not_a_loop(self):
        # The cold-coil guard: heat failed twice with a cold coil,
        # but this plan warms the coil first, so heat runs with
        # ``coil: warm`` — a different situation, a genuine recovery.
        # NOTE: episodes are recorded with the precondition-
        # restricted situation, exactly as ``Brain._act`` records
        # them (heat's pre is only ``at`` — the coil is a hidden
        # precondition, which is the whole point: the net-state
        # half of the rule is what sees the warming).
        b = _brain_with([HEAT, WARM], {"at": "orin", "coil": "cold"})
        for cycle in (1, 2):
            b.learner.record(Episode(
                cycle, HEAT.name, {"at": "orin"}, False, [],
                args=dict(HEAT.args)))
        self.assertFalse(b._plan_is_loop_on_twice_failed_step(
            Plan([WARM, HEAT], 2.0)))

    def test_heat_without_warming_is_a_loop(self):
        b = _brain_with([HEAT, WARM], {"at": "orin", "coil": "cold"})
        for cycle in (1, 2):
            b.learner.record(Episode(
                cycle, HEAT.name, {"at": "orin"}, False, [],
                args=dict(HEAT.args)))
        self.assertTrue(b._plan_is_loop_on_twice_failed_step(
            Plan([HEAT], 1.0)))

    def test_sibling_variant_is_not_a_loop(self):
        # Args-aware: cut-zinc failed twice; a plan cutting tin in
        # the same situation is not a loop.
        zinc = Action("cut", {"at": "krelm", "has_awl": True},
                      {"cut_b": True}, args={"material": "zinc"})
        tin = Action("cut", {"at": "krelm", "has_awl": True},
                     {"cut_b": True}, args={"material": "tin"})
        b = _brain_with([zinc, tin], {"at": "krelm", "has_awl": True})
        for cycle in (1, 2):
            b.learner.record(Episode(
                cycle, "cut", {"at": "krelm", "has_awl": True}, False, [],
                args={"material": "zinc"}))
        self.assertFalse(b._plan_is_loop_on_twice_failed_step(
            Plan([tin], 1.0)))
        self.assertTrue(b._plan_is_loop_on_twice_failed_step(
            Plan([zinc], 1.0)))

    def test_empty_plan_is_not_a_loop(self):
        b = _brain_with([TAKE], dict(SITUATION))
        self.assertFalse(b._plan_is_loop_on_twice_failed_step(
            Plan([], 0.0)))


class TestUnrecoverableStopReason(unittest.TestCase):
    def test_unrecoverable_stops_impossible_not_stalled(self):
        factory = make_variant_factories()["full"]
        rec, brain = bench._run_brain(
            factory, bench._destroy_key_actions(), "take the durnik",
            {"has_durnik": True}, bench.DestroyKeyWorld(), variant="full")
        self.assertEqual(rec["stop_reason"], "impossible", rec)
        self.assertEqual(rec["goal_status"], "failed")
        self.assertLess(rec["cycles"], 10)
        kinds = [e["kind"] for e in brain.events]
        self.assertIn("repeated_failure_no_alternative", kinds)
        # Bounded recovery: the destroyed step was retried exactly
        # twice (the threshold), never a third time.
        fails = [ep for ep in brain.learner.episodes
                 if ep.action == "take_vaskel" and not ep.ok]
        self.assertEqual(len(fails), 2)
        self.assertTrue(all(
            dict(ep.situation) == SITUATION for ep in fails))

    def test_unrecoverable_benchmark_still_passes(self):
        r = bench.run_unrecoverable_failure(
            make_variant_factories()["full"], seed=0, variant="full")
        self.assertTrue(r["passed"], r["detail"])
        self.assertEqual(r["evidence"]["stop_reason"], "impossible")
        self.assertEqual(r["evidence"]["unlock_attempts_after_event"], 0)
        self.assertLessEqual(r["evidence"]["retake_attempts_after_event"], 3)
        self.assertTrue(r["evidence"]["key_taken_before_destruction"])


if __name__ == "__main__":
    unittest.main()
