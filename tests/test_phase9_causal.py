"""Phase 9 (Worker 1): variant-aware causal records (design SS4.1, P9-1).

Unit + integration tests for the variant fix, and the P-D probe.

Variant identity is the canonical precondition set of the action
variant that actually ran (``canonical_variant``): two variants of
one action name keep separate causal rates, separate revision
adjustments, and separate simulated confidences. The pooled
(name-level) cells still exist — a legacy ``variant=None`` record
stays visible as the prior — but a variant's own evidence always
wins over the pooled rate, and a causal CLAIM never falls back to
the pooled counts.

The P-D probe test (``TestPDProbe``) re-runs the Phase 8 experiment
12 setup (W1 -> W2 workshop) and simulates the two press routes as
ACTION OBJECTS, which is what carries variant intent: the probe as
written in ``ab/eval/phase8_eval.py`` passes only action NAMES to
the simulator, and names carry no variant intent — from a state
where the mould IS seated, both name-routes resolve to the
with-mould variant and simulate identically. That is correct
name-based semantics, not a failure of the fix; the probe needs the
SS9-precedent update (pass the plan's Action objects, as this test
does) to express "press without mould". The as-written probe is
called here only to record its numbers, not to assert on its
pass/fail — see ``test_pd_probe_as_written_records_numbers``.
"""
import unittest

from ab.brain import Brain
from ab.eval.execution_eval import (
    WORKSHOP_W1, WORKSHOP_W2, _full_factory, _run_workshop,
    workshop_actions, WorkshopWorld)
from ab.eval.phase8_eval import exp_pd_overgeneralisation_probe
from ab.eval.reasoning_tasks2 import ForgeWorld, forge_actions
from ab.imagination import SimulationRevision, Simulator
from ab.imagination.simulator import laplace_rate
from ab.planning.planner import Action, Planner, canonical_variant
from ab.reasoning.causal import CausalModel
from ab.world.model import WorldModel


# ------------------------------------------------------------------ units
class TestCausalVariantModel(unittest.TestCase):
    def test_variant_cells_are_separate(self):
        c = CausalModel()
        good = "at='kiln';has_cast=True"
        bad = "at='lathe'"
        for _ in range(3):
            c.record("press", "pressed", True, variant=good)
        c.record("press", "pressed", False, variant=bad)
        self.assertEqual(c.counts("press", "pressed", variant=good), (3, 3))
        self.assertEqual(c.counts("press", "pressed", variant=bad), (0, 1))
        self.assertEqual(c.counts("press", "pressed"), (3, 4))

    def test_claim_only_for_the_reliable_variant(self):
        c = CausalModel()
        good = "at='kiln';has_cast=True"
        bad = "at='lathe'"
        for _ in range(3):
            c.record("press", "pressed", True, variant=good)
        c.record("press", "pressed", False, variant=bad)
        claim = c.causes("press", "pressed", variant=good)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["variant"], good)
        self.assertEqual(claim["observations"], 3)
        self.assertEqual(claim["counterexamples"], 0)
        self.assertIsNone(c.causes("press", "pressed", variant=bad))
        # The pooled name is blocked by the counterexample.
        self.assertIsNone(c.causes("press", "pressed"))
        # ...while the reliable variant's claim IS listed.
        self.assertIn(good, [cl["variant"] for cl in c.claims()
                             if cl["action"] == "press"])

    def test_legacy_variant_none_record_is_a_prior_not_a_claim(self):
        # Old callers record without a variant: the pooled rate stays
        # visible to variant reads (backwards compatible), but no
        # variant may CLAIM on pooled evidence alone.
        c = CausalModel()
        for _ in range(3):
            c.record("boil", "water_boils", True)
        self.assertEqual(c.success_rate("boil", "water_boils",
                                        variant="at='lab'"), 1.0)
        self.assertEqual(c.counts("boil", "water_boils",
                                  variant="at='lab'"), (3, 3))
        self.assertIsNone(c.causes("boil", "water_boils",
                                   variant="at='lab'"))
        self.assertIsNotNone(c.causes("boil", "water_boils"))

    def test_canonical_variant_is_order_independent(self):
        self.assertEqual(
            canonical_variant({"b": 1, "a": "x"}),
            canonical_variant({"a": "x", "b": 1}))
        self.assertEqual(canonical_variant({}), "")
        self.assertNotEqual(
            canonical_variant({"at": "lathe"}),
            canonical_variant({"at": "lathe", "has_cast": True}))


class TestPlannerVariantResolution(unittest.TestCase):
    def _planner(self):
        return Planner([
            Action("cut", {"at": "lathe", "has_shears": True},
                   {"cut_a": True}, cost=1.0),
            Action("cut", {"at": "cellar", "has_shears": True},
                   {"cut_b": True}, cost=1.0),
            Action("cut", {"at": "kiln", "has_shears": True},
                   {"cut_c": True}, cost=1.0),
        ])

    def test_effects_fn_resolves_by_state_not_list_order(self):
        p = self._planner()
        # The FIRST cut variant is inapplicable in the cellar; the
        # old code returned {} for it (its preconditions failed).
        self.assertEqual(p.effects_fn("cut", {"at": "cellar",
                                              "has_shears": True}),
                         {"cut_b": True})
        self.assertEqual(p.effects_fn("cut", {"at": "kiln",
                                              "has_shears": True}),
                         {"cut_c": True})
        self.assertEqual(p.effects_fn("cut", {"at": "nowhere"}), {})
        self.assertEqual(p.effects_fn("nope", {"at": "lathe"}), {})

    def test_resolve_action_and_variant_for(self):
        p = self._planner()
        state = {"at": "cellar", "has_shears": True}
        action = p.resolve_action("cut", state)
        self.assertIsNotNone(action)
        self.assertEqual(action.eff, {"cut_b": True})
        self.assertEqual(p.variant_for("cut", state),
                         "at='cellar';has_shears=True")
        self.assertIsNone(p.resolve_action("cut", {"at": "nowhere"}))
        self.assertIsNone(p.variant_for("cut", {"at": "nowhere"}))


class TestSimulatorVariantResolution(unittest.TestCase):
    def _simulator(self, causal=None, revision=None):
        wm = WorldModel()
        planner = Planner([
            Action("go_cellar", {"at": "lathe"}, {"at": "cellar"}),
            Action("cut", {"at": "lathe", "has_shears": True},
                   {"cut_a": True}, cost=1.0),
            Action("cut", {"at": "cellar", "has_shears": True},
                   {"cut_b": True}, cost=1.0),
        ])
        return Simulator(wm, causal=causal, planner=planner,
                         revision=revision), planner

    def test_duplicate_name_simulates_each_variant_in_its_state(self):
        # The Phase 8 cut limitation: effects_fn resolved the repeated
        # 'cut' name to its FIRST variant, whose preconditions failed
        # in later rooms, so simulation recorded 'no known effect'
        # and zeroed success. Both cuts must now apply.
        sim, _ = self._simulator()
        start = {"at": "lathe", "has_shears": True,
                 "cut_a": False, "cut_b": False}
        result = sim.simulate(["cut", "go_cellar", "cut"], dict(start),
                              goal={"cut_a": True, "cut_b": True})
        self.assertTrue(result.met_goal)
        self.assertEqual(result.predicted_success, 1.0)
        self.assertEqual(
            [s.get("variant") for s in result.per_step],
            ["at='lathe';has_shears=True", "at='lathe'",
             "at='cellar';has_shears=True"])
        self.assertEqual(
            [a for a in result.assumptions if "no known effect" in a], [])

    def test_step_confidence_uses_the_variant_rate(self):
        causal = CausalModel()
        bad = "at='lathe';has_shears=True"
        good = "at='cellar';has_shears=True"
        causal.record("cut", "cut_a", False, variant=bad)
        causal.record("cut", "cut_b", True, variant=good)
        causal.record("cut", "cut_b", True, variant=good)
        causal.record("cut", "cut_b", True, variant=good)
        sim, _ = self._simulator(causal=causal)
        start = {"at": "lathe", "has_shears": True,
                 "cut_a": False, "cut_b": False}
        result = sim.simulate(["cut", "go_cellar", "cut"], dict(start),
                              goal={"cut_a": True, "cut_b": True})
        first_cut, second_cut = result.per_step[0], result.per_step[2]
        # Failing variant: Laplace (0+1)/(1+2) = 1/3.
        self.assertAlmostEqual(first_cut["confidence"], 1 / 3, places=4)
        # Reliable variant: Laplace (3+1)/(3+2) = 4/5.
        self.assertAlmostEqual(second_cut["confidence"], 4 / 5, places=4)
        self.assertAlmostEqual(result.predicted_success,
                               (1 / 3) * (4 / 5), places=4)

    def test_action_object_step_is_its_own_variant(self):
        # A step given as an Action object simulates AS that variant:
        # an inapplicable variant yields no effect instead of
        # borrowing another variant's effect.
        sim, planner = self._simulator()
        without = next(a for a in planner.actions
                       if a.name == "cut" and a.eff == {"cut_a": True})
        start = {"at": "cellar", "has_shears": True,
                 "cut_a": False, "cut_b": False}
        result = sim.simulate([without], dict(start),
                              goal={"cut_a": True})
        self.assertFalse(result.met_goal)
        self.assertEqual(result.predicted_success, 0.0)
        self.assertEqual(result.per_step[0]["variant"],
                         "at='lathe';has_shears=True")

    def test_laplace_rate_variant_falls_back_to_pooled(self):
        causal = CausalModel()
        causal.record("pick_lock", "door_open", False)
        self.assertAlmostEqual(
            laplace_rate(causal, "pick_lock", "door_open",
                         variant="at='lab'"), 1 / 3, places=4)
        self.assertIsNone(
            laplace_rate(CausalModel(), "pick_lock", "door_open",
                         variant="at='lab'"))


class TestRevisionVariant(unittest.TestCase):
    def test_variant_adjustments_are_separate(self):
        rev = SimulationRevision()
        good = "at='kiln';has_cast=True"
        bad = "at='lathe'"
        rev.observe("press", "pressed", True, False, cycle=1, variant=bad)
        rev.observe("press", "pressed", True, True, cycle=2, variant=good)
        # Failing variant: 1.0 -> 0.5. Reliable variant: 1.0 -> 1.0
        # (observed success from a 1.0 prior does not move).
        self.assertAlmostEqual(
            rev.predicted_success("press", "pressed", variant=bad),
            0.5, places=4)
        self.assertAlmostEqual(
            rev.predicted_success("press", "pressed", variant=good),
            1.0, places=4)
        # The pooled cell moved for both observations.
        self.assertLess(rev.predicted_success("press", "pressed"), 1.0)

    def test_variant_read_falls_back_to_pooled_when_unseen(self):
        rev = SimulationRevision()
        rev.observe("pick_lock", "door_open", True, False, cycle=1)
        self.assertAlmostEqual(
            rev.predicted_success("pick_lock", "door_open",
                                  variant="at='lab'"), 0.5, places=4)

    def test_to_json_round_trip_keeps_variant_cells(self):
        rev = SimulationRevision()
        rev.observe("press", "pressed", True, False, cycle=1,
                    variant="at='lathe'")
        rev.observe("press", "pressed", True, True, cycle=2,
                    variant="at='kiln';has_cast=True")
        restored = SimulationRevision.from_json(rev.to_json())
        self.assertAlmostEqual(
            restored.predicted_success("press", "pressed",
                                       variant="at='lathe'"), 0.5, places=4)
        self.assertAlmostEqual(
            restored.predicted_success("press", "pressed",
                                       variant="at='kiln';has_cast=True"),
            1.0, places=4)
        self.assertEqual(len(restored.history), 2)

    def test_from_json_loads_legacy_triple_form(self):
        rev = SimulationRevision.from_json(
            {"version": 1, "alpha": 0.5,
             "adjusted_success": [["pick_lock", "door_open", 0.5]],
             "history": []})
        self.assertAlmostEqual(
            rev.predicted_success("pick_lock", "door_open"), 0.5, places=4)


class TestBrainCheckPredictionsVariant(unittest.TestCase):
    def test_executed_variant_is_recorded(self):
        brain = Brain(forge_actions(), granted={"compute", "environment"})
        env = ForgeWorld()
        brain.add_goal("forge the part", {"forged": True})
        stop = brain.run(env, brain.cycle + 60)
        self.assertEqual(brain.goals.all()[-1].status, "done")
        good = "at='vortan';has_catalyst=True"
        bad = "at='vortan'"
        self.assertEqual(brain.causal.counts("forge", "forged",
                                             variant=bad), (0, 1))
        self.assertEqual(brain.causal.counts("forge", "forged",
                                             variant=good), (1, 1))
        # The prediction_checked events name the variant resolved.
        checked = [e for e in brain.events
                   if e["kind"] == "prediction_checked"
                   and e["detail"].startswith("forge ")]
        self.assertTrue(checked)
        self.assertTrue(all("variant=" in e["detail"] for e in checked))
        variants_seen = {("at='vortan'" in e["detail"],
                          "has_catalyst" in e["detail"]) for e in checked}
        self.assertIn((True, False), variants_seen)   # cheap forge failed
        self.assertIn((True, True), variants_seen)    # catalyst forge met


# ------------------------------------------------------------------ P-D probe
class TestPDProbe(unittest.TestCase):
    def _run_w1_w2(self):
        _rec_w1, brain_w1, _env_w1 = _run_workshop(WORKSHOP_W1, _full_factory)
        rec_w2, brain, _env_w2 = _run_workshop(WORKSHOP_W2, _full_factory,
                                               brain=brain_w1)
        return brain

    def test_pd_probe_variant_aware_routes(self):
        """The Phase 8 P-D probe, re-run after the P9-1 fix: W1 then
        W2, then simulate the with-mould route and the
        without-mould route as ACTION OBJECTS (the plan's own
        steps, which carry variant intent). Pass = the misfiring
        route's predicted success drops below the correct route's
        (0.0 vs > 0), the misfiring route does not meet the goal,
        and the duplicate-name 'cut' limitation is gone (no 'no
        known effect' step in the correct route)."""
        brain = self._run_w1_w2()
        start = dict(WorkshopWorld(WORKSHOP_W2).observe())
        target = {WORKSHOP_W2["treasure_key"]: True}
        plans = brain.planner.plan_all(dict(start), dict(target), limit=3)
        self.assertTrue(plans)
        plan_a = plans[0]
        w2_actions = workshop_actions(WORKSHOP_W2)
        press_variants = [a for a in w2_actions if a.name == "press"]
        with_mould = next(a for a in press_variants
                          if WORKSHOP_W2["mould_key"] in a.pre)
        without_mould = next(a for a in press_variants
                             if WORKSHOP_W2["mould_key"] not in a.pre)
        # The planner's own candidates already use the mould variant.
        self.assertTrue(any(a.name == "press"
                            and WORKSHOP_W2["mould_key"] in a.pre
                            for a in plan_a.steps))
        route_b_actions = [without_mould if a.name == "press" else a
                           for a in plan_a.steps]
        sim = brain._get_simulator()
        res_a = sim.simulate(list(plan_a.steps), dict(start),
                             goal=dict(target))
        res_b = sim.simulate(list(route_b_actions), dict(start),
                             goal=dict(target))
        # Variant causal records are separated after W1+W2.
        good_v = canonical_variant(with_mould.pre)
        bad_v = canonical_variant(without_mould.pre)
        good_counts = brain.causal.counts("press", "pressed", variant=good_v)
        # Phase 10.5 (P10.5-5/8, W2) §9 update: the bad variant's
        # counts must be read from its OWN cell, not via
        # ``counts()`` — that accessor falls back to the pooled
        # cell when the variant has no record, and the continued
        # brain no longer attempts the misfiring route in W2 at
        # all (the adopted variant-aware rate steers it to the
        # mould plan: W1 completes and W2 completes with zero
        # press failures), so the bad cell is ABSENT and the
        # pooled fallback (2 met / 4 total) is not the variant's
        # record. Separation is what this probe pins, and it
        # holds in the stronger form: the bad variant has zero
        # successes in its own cell (absent, or failures only),
        # while the good variant's successes equal its total.
        bad_cell = brain.causal._variant_records.get(
            ("press", "pressed", bad_v))
        self.assertTrue(bad_cell is None or bad_cell[0] == 0)
        self.assertIsNotNone(good_counts)
        self.assertEqual(good_counts[0], good_counts[1])
        self.assertGreaterEqual(good_counts[0], 1)
        # Revision separates the variants too.
        rev_good = brain.simulation_revision.predicted_success(
            "press", "pressed", causal=brain.causal, variant=good_v)
        rev_bad = brain.simulation_revision.predicted_success(
            "press", "pressed", causal=brain.causal, variant=bad_v)
        self.assertGreater(rev_good, rev_bad)
        # The simulated routes separate: the misfiring route fails.
        self.assertGreater(res_a.predicted_success, 0.0)
        self.assertTrue(res_a.met_goal)
        self.assertEqual(res_b.predicted_success, 0.0)
        self.assertFalse(res_b.met_goal)
        self.assertLess(res_b.predicted_success, res_a.predicted_success)
        # Cut-duplicate limitation: resolved. The correct route
        # simulates every cut in its own room (no 'no known effect').
        self.assertEqual(
            [a for a in res_a.assumptions if "no known effect" in a], [])
        # The press steps carry their variants in the record.
        press_a = next(s for s in res_a.per_step if s["action"] == "press")
        press_b = next(s for s in res_b.per_step if s["action"] == "press")
        self.assertEqual(press_a["variant"], good_v)
        self.assertEqual(press_b["variant"], bad_v)

    def test_name_only_routes_carry_no_variant_intent(self):
        """Pinned behaviour, not a failure: two routes that differ
        only in which Action object they use, passed to the
        simulator as NAMES from a state where the mould IS seated,
        resolve to the same (with-mould) variant and simulate
        identically. Variant intent lives in the Action objects
        (or in the state at the step), never in a bare name."""
        brain = self._run_w1_w2()
        start = dict(WorkshopWorld(WORKSHOP_W2).observe())
        target = {WORKSHOP_W2["treasure_key"]: True}
        plans = brain.planner.plan_all(dict(start), dict(target), limit=3)
        plan_a = plans[0]
        w2_actions = workshop_actions(WORKSHOP_W2)
        without_mould = next(
            a for a in w2_actions
            if a.name == "press" and WORKSHOP_W2["mould_key"] not in a.pre)
        route_b_names = [without_mould.name if a.name == "press" else a.name
                         for a in plan_a.steps]
        sim = brain._get_simulator()
        res_a = sim.simulate([a.name for a in plan_a.steps], dict(start),
                             goal=dict(target))
        res_b = sim.simulate(route_b_names, dict(start), goal=dict(target))
        self.assertEqual(res_a.predicted_success, res_b.predicted_success)
        self.assertEqual(res_a.predicted_cost, res_b.predicted_cost)

    def test_pd_probe_as_written_records_numbers(self):
        """The Phase 8 probe as written in ``phase8_eval.py``, after
        the Phase 9 §4.1 coordinator update (probe simulates Action
        objects, not bare names — Phase 2 §9 precedent: same intent,
        new intended semantics, not weakened): the probe PASSES now.
        The correct route simulates above 0.0 (the cut limitation is
        gone) AND the two routes' predicted success differs, with the
        misfiring route below the correct one."""
        result = exp_pd_overgeneralisation_probe()
        self.assertTrue(result["w1_completed"])
        self.assertTrue(result["w2_continued_completed"])
        self.assertTrue(result["passed"])
        # The cut limitation is gone: the correct route no longer
        # simulates to 0.0 (Phase 8 recorded 0.0 for both routes).
        self.assertGreater(
            result["route_with_mould"]["predicted_success"], 0.0)
        # The variant-aware fix: the two routes now simulate
        # differently, misfiring route below the correct one.
        self.assertEqual(result["predicted_success_differs"], True)
        self.assertLess(
            result["route_without_mould"]["predicted_success"],
            result["route_with_mould"]["predicted_success"])


if __name__ == "__main__":
    unittest.main()
