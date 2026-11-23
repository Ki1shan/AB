"""Phase 10.5 (W2) — stall/progress fix (P10.5-5), key-shape fallback
(P10.5-8), and adoption of the variant-aware failure rate (P10-1b).

The P10-1b history this pins: wiring the variant rate into
``effective_cost`` WITHOUT the stall fix turned the generated-
workshop leg from 2 failed / 26 cycles / stalled into 1 failed /
80 cycles / budget_exhausted — the never-failed with-form press
variant (rate 0.0) made the press-first retry plan cheap enough for
the last-resort planner to re-adopt every cycle, and each re-press
"success" advanced the progress count, defeating the progress-based
stall detector. The fix: ``_steps_executed`` advances only when the
step CHANGED the world on one of its effect keys (or its
``store_as`` key). Stdlib only, deterministic.
"""
from __future__ import annotations

import unittest

from ab.brain import Brain
from ab.core.learning import Episode, Learner
from ab.eval.consolidated import _run_generic_env
from ab.eval.envgen import generate
from ab.eval.reasoning_tasks import make_variant_factories
from ab.eval.transfer import prepare_brain_for_new_world
from ab.planning.planner import Action, Planner
from ab.reasoning.rules import RuleEngine


# --- shared fixtures ----------------------------------------------------

def _press(pre, cost=0.3):
    return Action("press", dict(pre), {"pressed": True}, cost=cost,
                  args={"_do": "press"})


def _cut(material, pre=None):
    return Action("cut", dict(pre or {"at": "krelm", "has_awl": True}),
                  {"cut_b": True}, args={"material": material})


def _learner_with_press_episodes():
    """Planner + learner holding ONE press failure with no mould:
    ``press`` at drelm failed; ``press`` at vexar with a die
    succeeded. Both episodes are the A-leg record the B-leg brain
    carries (design P10.5-8)."""
    cheap = _press({"at": "drelm"})
    mould = _press({"at": "vexar", "has_die": True}, cost=1.0)
    planner = Planner([cheap, mould])
    learner = Learner(planner, RuleEngine())
    planner.attach_learner(learner)
    learner.record(Episode(0, "press", {"at": "drelm"}, False))
    learner.record(Episode(1, "press", {"at": "vexar", "has_die": True},
                           True))
    return planner, learner, cheap, mould


class _FakeEnv:
    """Minimal env: observe() returns the state dict; act() runs the
    named behaviour (a callable taking (env, args) -> (ok, info))."""

    def __init__(self, state, behaviours=None):
        self.s = dict(state)
        self.behaviours = dict(behaviours or {})
        self.log = []

    def observe(self):
        return dict(self.s)

    def act(self, name, args):
        self.log.append(name)
        fn = self.behaviours.get(name)
        if fn is None:
            return False, "nothing happened"
        return fn(self, args)


def _press_brain(world):
    press = _press({"at": "bench"})
    take = Action("take", {"at": "bench"}, {"has_awl": True},
                  args={"_do": "take"})
    brain = Brain([press, take], granted={"compute", "environment"})
    brain.world = dict(world)
    goal = brain.add_goal("press the parts", {"pressed": True})
    brain.goals.activate(goal.id)
    def _press_ok(env, args):
        # Idempotent by construction: setting the key to True when it
        # is already True changes nothing, and the env still reports
        # success — exactly the re-press / re-take the old code
        # counted as progress.
        env.s["pressed"] = True
        return True, "pressed"

    def _take_ok(env, args):
        env.s["has_awl"] = True
        return True, "took the awl"

    behaviours = {"press": _press_ok, "take": _take_ok}
    return brain, press, take, _FakeEnv(brain.world, behaviours)


# --- (1) stall / progress fix -------------------------------------------

class TestStallFix(unittest.TestCase):
    def test_idempotent_repress_does_not_advance_progress(self):
        brain, press, _take, env = _press_brain(
            {"at": "bench", "pressed": True})
        brain._set_plan([press])
        self.assertTrue(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 0)
        # A second identical re-press is still not progress.
        brain._set_plan([press])
        self.assertTrue(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 0)

    def test_idempotent_retake_does_not_advance_progress(self):
        brain, _press, take, env = _press_brain(
            {"at": "bench", "pressed": False, "has_awl": True})
        brain._set_plan([take])
        self.assertTrue(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 0)

    def test_real_effect_change_advances_progress(self):
        brain, press, take, env = _press_brain(
            {"at": "bench", "pressed": False, "has_awl": False})
        brain._set_plan([press])
        self.assertTrue(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 1)
        brain._set_plan([take])
        self.assertTrue(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 2)

    def test_failed_step_still_never_advances_progress(self):
        brain, press, _take, env = _press_brain(
            {"at": "bench", "pressed": False})
        env.behaviours["press"] = lambda e, a: (False, "jammed")
        brain._set_plan([press])
        self.assertFalse(brain._do_execute_step({}, env))
        self.assertEqual(brain._steps_executed, 0)

    def test_store_as_change_counts_but_same_value_does_not(self):
        from ab.body.tools import ToolSpec
        brain = Brain([], granted={"compute", "environment"})
        brain.world = {}
        goal = brain.add_goal("read the gauge", {"reading": "v1"})
        brain.goals.activate(goal.id)
        brain.tools.register(ToolSpec(
            name="gauge", args=[], return_type="str", timeout=5.0,
            required_grant="compute",
            description="Read the gauge"), lambda: "v1")
        read = Action("read", {}, {}, tool="gauge", store_as="reading")
        brain.planner.add_action(read)
        brain._set_plan([read])
        self.assertTrue(brain._do_execute_step({}, None))
        self.assertEqual(brain._steps_executed, 1)
        self.assertEqual(brain.world.get("reading"), "v1")
        # The same reading again changes nothing: not progress.
        brain._set_plan([read])
        self.assertTrue(brain._do_execute_step({}, None))
        self.assertEqual(brain._steps_executed, 1)

    def test_other_progress_paths_checked(self):
        # Audit note (recorded, not just asserted): the ONLY increment
        # of ``_steps_executed`` in ab/brain.py is the P10.5-5 site
        # in ``_do_execute_step``; the other mentions are resets
        # (goal switch / resume / failure paths), the save/load
        # restore (``rec.plan_position``), and the ``_goal_progress``
        # reader, whose only consumer is
        # ``control.record_progress`` (the stall detector's input).
        # So no other place can advance progress on a non-change.
        import ab.brain
        import inspect
        src = inspect.getsource(ab.brain)
        self.assertEqual(src.count("_steps_executed += 1"), 1)
        self.assertIn("_steps_executed", inspect.getsource(
            ab.brain.Brain._goal_progress))


# --- (1b) planner prunes no-change steps; record() fallback --------

class TestPlannerNoChangePrune(unittest.TestCase):
    def test_plan_never_starts_with_a_no_change_step(self):
        # The stall fix's premise, made true in plan_all: a step
        # whose effects already hold in the current state is never
        # expanded — not even as a first step (the visited set alone
        # did not prune first steps). The cheap redundant press
        # must not appear in any returned plan.
        press = Action("press", {"at": "bench"}, {"pressed": True},
                       cost=0.1, args={"_do": "press"})
        take = Action("take", {"at": "bench"}, {"has_awl": True},
                      cost=1.0, args={"_do": "take"})
        planner = Planner([press, take])
        plans = planner.plan_all({"at": "bench", "pressed": True,
                                  "has_awl": False},
                                 {"has_awl": True, "pressed": True},
                                 limit=2)
        self.assertEqual(len(plans), 1)
        self.assertEqual([a.name for a in plans[0].steps], ["take"])

    def test_store_as_outside_effects_is_not_pruned(self):
        # A tool action whose stored output is genuinely unmodeled
        # (``store_as`` names a key OUTSIDE its effects — a gauge
        # reading, not a declared effect) may still change state
        # through the stored value, so it stays plannable even
        # though it has no modeled effect at all.
        read = Action("read", {}, {}, tool="gauge", store_as="reading")
        finish = Action("finish", {}, {"done": True},
                        args={"_do": "finish"})
        planner = Planner([read, finish])
        plans = planner.plan_all({}, {"done": True}, limit=2)
        self.assertIn(["read", "finish"],
                      [[a.name for a in p.steps] for p in plans])

    def test_store_as_inside_effects_is_pruned_when_holding(self):
        # ... but when ``store_as`` names a key the effects already
        # declare (the workshop cutter: ``store_as=cut_b`` with
        # ``eff={cut_b: True}``, tool returns True), the modeled
        # outcome IS the effect value — with it already holding,
        # the step is prunable like any other.
        cut = Action("cut", {"at": "bench", "has_cutter": True},
                     {"cut_b": True}, tool="cutter", store_as="cut_b")
        finish = Action("finish", {"cut_b": True}, {"done": True},
                        args={"_do": "finish"})
        planner = Planner([cut, finish])
        plans = planner.plan_all({"at": "bench", "has_cutter": True,
                                  "cut_b": True},
                                 {"done": True}, limit=2)
        self.assertEqual([[a.name for a in p.steps] for p in plans],
                         [["finish"]])

    def test_direct_record_still_steers_cost_when_no_episodes(self):
        # test_brain's "learned earlier": a failure recorded
        # straight on the planner (no learner episode) must still
        # inflate the variant rate — otherwise the brain re-tries
        # the known-bad action.
        planner = Planner([])
        learner = Learner(planner, RuleEngine())
        planner.attach_learner(learner)
        planner.record("pick_lock", False)
        pick = Action("pick_lock", {"at": "hall"}, {"door_open": True})
        self.assertEqual(planner.failure_rate_for(pick), 1.0)
        self.assertEqual(planner.failure_counts_for(pick), [0, 1])

    def test_direct_record_does_not_leak_across_variants(self):
        # ... but once episodes exist for the name, the variant data
        # is authoritative: the pooled counts (which mirror the
        # episodes via record()) must not inflate a variant with no
        # episodes of its own. (P9-7 guard for the fallback above.)
        planner, _learner, _cheap, _mould = _learner_with_press_episodes()
        # The learner holds press episodes; a press variant with no
        # episode of its own in either tier — different key set AND
        # different values — stays 0.0.
        other = _press({"at": "krelm", "has_form": True}, cost=1.0)
        self.assertEqual(planner.failure_rate_for(other), 0.0)


# --- (2) key-shape fallback ----------------------------------------------

class TestKeyShapeFallback(unittest.TestCase):
    def test_press_no_mould_rate_transfers_across_rooms(self):
        planner, _learner, _cheap, _mould = _learner_with_press_episodes()
        # No episode has at=forvik, but the key shape {at} matches
        # the drelm failure: the room value is irrelevant to whether
        # a press needs a mould.
        other_room = _press({"at": "forvik"})
        self.assertEqual(planner.failure_rate_for(other_room), 1.0)
        self.assertEqual(planner.failure_counts_for(other_room), [0, 1])

    def test_mould_press_rate_stays_zero_p9_7_not_reintroduced(self):
        planner, _learner, _cheap, mould = _learner_with_press_episodes()
        # The mould press's own episode succeeded; the cheap press's
        # failure must never inflate it — different key sets
        # ({at, has_die} vs {at}) match neither tier.
        self.assertEqual(planner.failure_rate_for(mould), 0.0)
        # And a mould press in a room never seen before is still 0.0:
        # the key-shape tier needs the SAME keys, and {at, has_form}
        # is a different key set from the {at} failure.
        other_mould = _press({"at": "krelm", "has_form": True}, cost=1.0)
        self.assertEqual(planner.failure_rate_for(other_mould), 0.0)

    def test_zinc_and_tin_still_separated_by_args(self):
        planner = Planner([_cut("zinc"), _cut("tin")])
        learner = Learner(planner, RuleEngine())
        planner.attach_learner(learner)
        learner.record(Episode(0, "cut", {"at": "krelm", "has_awl": True},
                               False, args={"material": "zinc"}))
        # Tin in a DIFFERENT room: same key set as the zinc episode,
        # but the args differ, so the key-shape tier must not match.
        tin_elsewhere = _cut("tin", {"at": "drelm", "has_awl": True})
        self.assertEqual(planner.failure_rate_for(tin_elsewhere), 0.0)
        self.assertEqual(planner.failure_counts_for(tin_elsewhere), [0, 0])

    def test_exact_variant_wins_over_key_shape(self):
        planner = Planner([_press({"at": "drelm"}),
                           _press({"at": "forvik"})])
        learner = Learner(planner, RuleEngine())
        planner.attach_learner(learner)
        learner.record(Episode(0, "press", {"at": "drelm"}, False))
        learner.record(Episode(1, "press", {"at": "forvik"}, True))
        # Exact tier for drelm: 1 failure / 1 episode = 1.0 (the
        # key-shape pool would say 1/2 = 0.5).
        self.assertEqual(planner.failure_rate_for(_press({"at": "drelm"})),
                         1.0)
        self.assertEqual(planner.failure_counts_for(_press({"at": "drelm"})),
                         [0, 1])
        # No exact episode for vexar: key-shape pool is 1/2 = 0.5.
        self.assertAlmostEqual(
            planner.failure_rate_for(_press({"at": "vexar"})), 0.5)
        self.assertEqual(planner.failure_counts_for(_press({"at": "vexar"})),
                         [1, 1])

    def test_no_learner_pooled_fallback_unchanged(self):
        planner = Planner([_press({"at": "drelm"})])
        planner.record("press", True)
        planner.record("press", False)
        self.assertAlmostEqual(
            planner.failure_rate_for(_press({"at": "drelm"})), 0.5)
        self.assertEqual(
            planner.failure_counts_for(_press({"at": "drelm"})), [1, 1])

    def test_no_episode_at_all_is_still_zero(self):
        planner, _learner, _cheap, _mould = _learner_with_press_episodes()
        steel = _cut("steel")
        self.assertEqual(planner.failure_rate_for(steel), 0.0)
        self.assertEqual(planner.failure_counts_for(steel), [0, 0])


# --- (3) adoption ---------------------------------------------------------

class TestAdoption(unittest.TestCase):
    def test_effective_cost_uses_variant_rate_with_fallback(self):
        planner, _learner, cheap, mould = _learner_with_press_episodes()
        # B cheap press (forvik, no episode of its own): key-shape
        # rate 1.0, so eff = 0.3 * (1 + 2*1.0) = 0.9 (no pattern or
        # concept library attached here, so both multipliers are 1.0).
        other_room = _press({"at": "forvik"})
        self.assertAlmostEqual(planner.effective_cost(other_room), 0.9)
        # Mould press: rate 0.0, so eff is its bare cost.
        self.assertAlmostEqual(planner.effective_cost(mould), 1.0)
        # The exact-variant rate still drives the cost too: the
        # drelm cheap press failed once (rate 1.0).
        self.assertAlmostEqual(planner.effective_cost(cheap), 0.9)

    def test_last_resort_ordering_uses_variant_rate(self):
        # Two cut variants, both banned in the same situation (so no
        # unbanned plan exists and the P4-9 last resort runs): zinc
        # failed twice, tin succeeded once. The pooled rate (2/3 for
        # both) cannot order them; the variant-aware rate can — the
        # tin-first plan must win, so the brain retries the variant
        # with no failures instead of the twice-failed one.
        zinc = _cut("zinc")
        tin = _cut("tin")
        brain = Brain([zinc, tin], granted={"compute", "environment"})
        brain.world = {"at": "krelm", "has_awl": True}
        goal = brain.add_goal("cut the tin", {"cut_b": True})
        brain.goals.activate(goal.id)
        brain.learner.record(Episode(0, "cut",
                                     {"at": "krelm", "has_awl": True},
                                     False, args={"material": "zinc"}))
        brain.learner.record(Episode(1, "cut",
                                     {"at": "krelm", "has_awl": True},
                                     False, args={"material": "zinc"}))
        brain.learner.record(Episode(2, "cut",
                                     {"at": "krelm", "has_awl": True},
                                     True, args={"material": "tin"}))
        brain.planner.ban("cut", {"at": "krelm", "has_awl": True},
                          {"material": "zinc"})
        brain.planner.ban("cut", {"at": "krelm", "has_awl": True},
                          {"material": "tin"})
        self.assertTrue(brain._do_make_plan({}, None))
        self.assertEqual(brain.plan[0].args, {"material": "tin"})


# --- (4) P10-1b regression scenario ---------------------------------------

class TestP101bRegression(unittest.TestCase):
    def test_generated_workshop_does_not_budget_exhaust(self):
        # The P10-1b scenario: the variant rate adopted WITHOUT the
        # stall fix turned this leg into 1 failed / 80 cycles /
        # budget_exhausted. With the stall fix + key-shape fallback
        # the leg must complete instead of exhausting the budget.
        # Measured with the fix (seed 0, episode-authoritative):
        # continued 1 failed / 14 cycles / done, fresh 1 / 14 / done
        # — the press failure is avoided; the remaining continued
        # failure is the cut-zinc tie (P10.5-9, coordinator's).
        # Phase 10.5 gate update (P10.5-9): the current-world
        # tie-break now also avoids the zinc failure, so continued
        # is 0 failed / 12 cycles / done — a strict beat of fresh,
        # not just a non-exhaustion. (12, not 14: avoiding the zinc
        # failure also saves the two wasted go-back cycles.) The
        # test's intent (no budget exhaustion, completes) is
        # unchanged; the pin moves 1 -> 0 because the mechanism the
        # test names is now in place.
        factory = make_variant_factories()["full"]
        env_a, actions_a, goal_a, _ = generate(0, "workshop")
        env_b, actions_b, goal_b, _ = generate(1, "workshop")
        carried = factory(actions_a, {"compute", "environment"})
        _run_generic_env(env_a, actions_a, goal_a, factory, brain=carried)
        prepare_brain_for_new_world(carried)
        rec_cont = _run_generic_env(env_b, actions_b, goal_b, factory,
                                    brain=carried)
        env_b2, actions_b2, goal_b2, _ = generate(1, "workshop")
        rec_fresh = _run_generic_env(env_b2, actions_b2, goal_b2, factory)
        self.assertTrue(rec_cont["completed"], rec_cont)
        self.assertNotEqual(rec_cont["stop_reason"], "budget_exhausted")
        self.assertLess(rec_cont["cycles"], 80)
        self.assertTrue(rec_fresh["completed"], rec_fresh)
        self.assertEqual(rec_cont["failed_attempts"], 0)
        self.assertEqual(rec_fresh["failed_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
