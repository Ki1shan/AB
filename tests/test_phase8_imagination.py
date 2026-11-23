"""Phase 8 (Worker 2, imagination): simulation over a copy of the world
model, scenario generation/comparison, what-if / counterfactual, and
the simulation-revision loop (design P8-5/P8-6).

No test weakens or deletes an existing assertion; every expectation
here is about the new imagination code. Nothing is hardcoded to an
evaluation answer: routes, confidences and rankings are all computed
from the planner, the causal model and the world model.
"""
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.imagination import (
    Scenario,
    ScenarioComparator,
    ScenarioGenerator,
    SimulationRevision,
    Simulator,
)
from ab.planning.planner import Action, Planner
from ab.reasoning.causal import CausalModel
from ab.world.model import WorldModel

TREASURE = {"at": "hall", "has_key": False, "door_open": False,
            "has_treasure": False}
LAB = {"at": "lab", "has_key": False, "door_open": False,
       "has_treasure": False}
PICK_ROUTE = ["go_lab_from_hall", "pick_lock", "go_vault", "take_treasure"]
KEY_ROUTE = ["go_lab_from_hall", "take_key", "unlock", "go_vault",
             "take_treasure"]


def _treasure_brain(state=None):
    b = Brain(room_world_actions(), granted={"compute", "environment"})
    b.world = dict(state if state is not None else TREASURE)
    return b


def _simulator(causal=None, revision=None):
    wm = WorldModel()
    for key, value in TREASURE.items():
        wm.observe(key, value, cycle=1)
    planner = Planner(room_world_actions())
    return Simulator(wm, causal=causal, planner=planner, revision=revision), wm


class TestSimulator(unittest.TestCase):
    def test_predicted_state_per_step_confidence_and_assumptions(self):
        sim, _wm = _simulator()
        result = sim.simulate(PICK_ROUTE, TREASURE, goal={"has_treasure": True})
        self.assertTrue(result.met_goal)
        self.assertTrue(result.predicted_state["has_treasure"])
        self.assertEqual(result.steps, PICK_ROUTE)
        self.assertEqual(len(result.per_step), len(PICK_ROUTE))
        # No causal record yet: every step is the planner's own model.
        for step in result.per_step:
            self.assertEqual(step["confidence"], 1.0)
        self.assertEqual(result.predicted_success, 1.0)
        self.assertEqual(result.risk, 0.0)
        self.assertTrue(result.simulated)
        self.assertTrue(result.assumptions)
        for a in result.assumptions:
            self.assertTrue(a.startswith("simulated:"),
                            f"assumption not labelled simulated: {a!r}")

    def test_confidence_from_causal_model_and_expected_cost(self):
        causal = CausalModel()
        causal.record("pick_lock", "door_open", False)
        sim, _wm = _simulator(causal=causal)
        result = sim.simulate(PICK_ROUTE, TREASURE, goal={"has_treasure": True})
        pick = next(s for s in result.per_step if s["action"] == "pick_lock")
        # Laplace (0 + 1) / (1 + 2) = 1/3 for the failed pair.
        self.assertAlmostEqual(pick["confidence"], 1 / 3, places=4)
        self.assertAlmostEqual(result.predicted_success, 1 / 3, places=4)
        self.assertAlmostEqual(result.risk, 2 / 3, places=4)
        # Expected cost divides each step by its confidence: 1 + 3 + 1 + 1.
        self.assertAlmostEqual(result.predicted_cost, 6.0, places=4)

    def test_unknown_action_records_assumption_and_zero_success(self):
        sim, _wm = _simulator()
        result = sim.simulate(["teleport"], TREASURE)
        self.assertEqual(result.predicted_success, 0.0)
        self.assertEqual(result.predicted_state, TREASURE)
        self.assertTrue(any("no known effect for 'teleport'" in a
                            for a in result.assumptions))

    def test_bounded_steps_truncate_and_say_so(self):
        sim, _wm = _simulator()
        result = sim.simulate(KEY_ROUTE, TREASURE,
                              goal={"has_treasure": True}, max_steps=2)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.steps), 2)
        self.assertTrue(any("truncated to the first 2 steps" in a
                            for a in result.assumptions))

    def test_simulation_never_mutates_real_world_model(self):
        sim, wm = _simulator()
        before = wm.to_json_str()
        result = sim.simulate(PICK_ROUTE, TREASURE, goal={"has_treasure": True})
        self.assertTrue(result.met_goal)
        self.assertEqual(wm.to_json_str(), before)
        # The copy learned the simulated door state; the real model did not.
        self.assertEqual(wm.status_of("door", "open").value, "observed")
        door = wm.entity("door")
        self.assertEqual(door.properties["open"].value, False)

    def test_counterfactual_edit_applies_to_copy_only(self):
        sim, wm = _simulator()
        before = wm.to_json_str()
        result = sim.simulate(["go_vault", "take_treasure"], LAB,
                              goal={"has_treasure": True},
                              state_edits={"door_open": True})
        self.assertTrue(result.predicted_state["has_treasure"])
        self.assertTrue(any("assumed door_open=True" in a
                            for a in result.assumptions))
        self.assertEqual(wm.to_json_str(), before)
        self.assertFalse(wm.entity("door").properties["open"].value)
        # The world-model-fact form behaves the same way.
        result2 = sim.simulate(["go_vault", "take_treasure"], LAB,
                               goal={"has_treasure": True},
                               world_edits={"entity": "door",
                                            "property": "open", "value": True})
        self.assertTrue(result2.predicted_state["has_treasure"])
        self.assertEqual(wm.to_json_str(), before)

    def test_simulation_flags_learned_door_gated_violation(self):
        # A planner that believes the vault move needs no door: its
        # effect claims the move while the door is closed, and the
        # world model — which learned the gate from a failed entry —
        # flags it. (The RoomWorld go_vault action already requires
        # door_open, so its own effect can never produce the move
        # that violates the gate.)
        actions = [Action("go_vault", {"at": "lab"}, {"at": "vault"}),
                   Action("take_treasure", {"at": "vault"},
                          {"has_treasure": True})]
        wm = WorldModel()
        for key, value in LAB.items():
            wm.observe(key, value, cycle=1)
        wm.note_failed_entry("vault", cycle=2)
        sim = Simulator(wm, planner=Planner(actions))
        result = sim.simulate(["go_vault"], LAB)
        self.assertIn("door_blocks_entry", result.violations)
        self.assertTrue(any("door_blocks_entry" in a for a in result.assumptions))
        # With the door open the same step is fine.
        ok = sim.simulate(["go_vault"], {**LAB, "door_open": True})
        self.assertNotIn("door_blocks_entry", ok.violations)

    def test_revision_moves_predicted_cost_toward_observed(self):
        revision = SimulationRevision()
        sim, _wm = _simulator(revision=revision)
        before = sim.simulate(PICK_ROUTE, TREASURE,
                              goal={"has_treasure": True})
        self.assertAlmostEqual(before.predicted_cost, 4.0, places=4)
        revision.observe("pick_lock", "door_open", True, False, cycle=1)
        after = sim.simulate(PICK_ROUTE, TREASURE,
                             goal={"has_treasure": True})
        # Observed failure: predicted success falls, expected cost rises.
        self.assertLess(after.predicted_success, before.predicted_success)
        self.assertGreater(after.predicted_cost, before.predicted_cost)


class TestScenarios(unittest.TestCase):
    def test_two_distinct_routes_with_distinct_future_states(self):
        b = _treasure_brain()
        scenarios = ScenarioGenerator(b.planner, b.simulator).generate(
            {"has_treasure": True}, dict(b.world))
        self.assertGreaterEqual(len(scenarios), 2)
        seqs = {tuple(s.steps) for s in scenarios}
        self.assertGreaterEqual(len(seqs), 2)
        states = {tuple(sorted(s.predicted_state.items())) for s in scenarios}
        self.assertEqual(len(states), len(scenarios),
                         "each scenario needs a distinct predicted future state")
        for s in scenarios:
            self.assertTrue(s.met_goal)
            self.assertTrue(s.predicted_state["has_treasure"])

    def test_generator_bounded_to_three_scenarios(self):
        b = _treasure_brain()
        gen = ScenarioGenerator(b.planner, b.simulator, max_scenarios=3)
        scenarios = gen.generate({"has_treasure": True}, dict(b.world))
        self.assertLessEqual(len(scenarios), 3)
        for s in scenarios:
            self.assertLessEqual(len(s.steps), 20)

    def test_comparator_ranks_success_then_cost_then_risk(self):
        scenarios = [
            Scenario("a", ["x"], predicted_success=0.9, predicted_cost=5.0,
                     risk=0.1),
            Scenario("b", ["y"], predicted_success=0.9, predicted_cost=3.0,
                     risk=0.1),
            Scenario("c", ["z"], predicted_success=0.5, predicted_cost=1.0,
                     risk=0.5),
        ]
        ranking = ScenarioComparator().rank(scenarios)
        self.assertEqual([e["name"] for e in ranking], ["b", "a", "c"])
        self.assertEqual([e["rank"] for e in ranking], [1, 2, 3])
        for entry in ranking:
            self.assertIn("predicted success", entry["reason"])
            self.assertIn("predicted cost", entry["reason"])

    def test_comparator_tie_breaks_on_risk(self):
        scenarios = [
            Scenario("a", ["x"], predicted_success=0.9, predicted_cost=3.0,
                     risk=0.5),
            Scenario("b", ["y"], predicted_success=0.9, predicted_cost=3.0,
                     risk=0.1),
        ]
        ranking = ScenarioComparator().rank(scenarios)
        self.assertEqual(ranking[0]["name"], "b")


class TestBrainImagine(unittest.TestCase):
    def test_imagine_two_routes_ranked_with_reasons(self):
        b = _treasure_brain()
        result = b.imagine({"has_treasure": True})
        self.assertTrue(result["simulated"])
        self.assertGreaterEqual(len(result["scenarios"]), 2)
        self.assertIsNotNone(result["best"])
        self.assertEqual(result["ranking"][0]["rank"], 1)
        self.assertTrue(result["ranking"][0]["reason"])
        # The pick_lock route is cheaper on paper with no evidence yet.
        self.assertEqual(result["best"]["steps"], PICK_ROUTE)
        self.assertTrue(any(e["kind"] == "simulation_compared"
                            for e in b.events))
        for a in result["best"]["assumptions"]:
            self.assertTrue(a.startswith("simulated:"))

    def test_imagine_does_not_mutate_world_model_state_or_ltm(self):
        b = _treasure_brain()
        model_before = b.world_model.to_json_str()
        world_before = dict(b.world)
        ltm_before = len(b.ltm.all_facts())
        b.imagine({"has_treasure": True})
        b.what_if("what if I use pick_lock?")
        b.counterfactual(edits={"door_open": True},
                         steps=["go_vault", "take_treasure"],
                         goal={"has_treasure": True})
        self.assertEqual(b.world_model.to_json_str(), model_before)
        self.assertEqual(dict(b.world), world_before)
        self.assertEqual(len(b.ltm.all_facts()), ltm_before)

    def test_imagine_no_route_is_honest(self):
        b = Brain([], granted={"compute", "environment"})
        b.world = dict(TREASURE)
        result = b.imagine({"has_treasure": True})
        self.assertEqual(result["scenarios"], [])
        self.assertIsNone(result["best"])
        self.assertTrue(result["simulated"])
        self.assertTrue(result["note"])

    def test_imagine_accepts_goal_object(self):
        b = _treasure_brain()
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        result = b.imagine(goal)
        self.assertGreaterEqual(len(result["scenarios"]), 2)


class TestWhatIf(unittest.TestCase):
    def test_what_if_use_action_finds_route(self):
        b = _treasure_brain()
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        b.goals.activate(goal.id)
        out = b.what_if("what if I use pick_lock?")
        self.assertEqual(out["action"], "pick_lock")
        self.assertTrue(out["simulated"])
        self.assertIn("pick_lock", out["steps"])
        self.assertTrue(out["predicted_state"]["has_treasure"])
        self.assertIn("(simulated)", out["answer"])
        for a in out["assumptions"]:
            self.assertTrue(a.startswith("simulated:"))

    def test_what_if_use_action_without_goal_is_honest(self):
        b = _treasure_brain()
        out = b.what_if("what if I use pick_lock?")
        self.assertEqual(out["action"], "pick_lock")
        # From the hall pick_lock has no known effect: say so, don't invent one.
        self.assertEqual(out["predicted_success"], 0.0)
        self.assertIn("no known effect", out["answer"])

    def test_what_if_door_open_reports_difference(self):
        b = _treasure_brain(LAB)
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        b.goals.activate(goal.id)
        out = b.what_if("what if the door were open?")
        self.assertEqual(out["state_edits"], {"door_open": True})
        self.assertTrue(out["counterfactual"]["predicted_state"]["has_treasure"])
        self.assertIn("door_open", out["answer"])
        self.assertTrue(all(a.startswith("simulated:")
                            for a in out["assumptions"]))

    def test_what_if_unidentified_is_honest(self):
        b = _treasure_brain()
        out = b.what_if("what if the sky falls?")
        self.assertIsNone(out["action"])
        self.assertTrue(out["simulated"])
        self.assertIn("could not identify", out["answer"])


class TestCounterfactual(unittest.TestCase):
    def test_counterfactual_differs_from_actual_history_simulation(self):
        b = _treasure_brain(LAB)
        before = b.world_model.to_json_str()
        out = b.counterfactual(edits={"door_open": True},
                               steps=["go_vault", "take_treasure"],
                               goal={"has_treasure": True})
        self.assertTrue(out["simulated"])
        self.assertTrue(out["outcome_changed"])
        self.assertEqual(out["difference"]["has_treasure"], (False, True))
        self.assertFalse(out["baseline"]["predicted_state"]["has_treasure"])
        self.assertTrue(out["counterfactual"]["predicted_state"]["has_treasure"])
        self.assertTrue(any("assumed door_open=True" in a
                            and "actual value was False" in a
                            for a in out["assumptions"]))
        self.assertEqual(b.world_model.to_json_str(), before)
        self.assertFalse(b.world["door_open"])

    def test_counterfactual_no_change_edit_changes_nothing(self):
        b = _treasure_brain(LAB)
        out = b.counterfactual(edits={"door_open": False},
                               steps=["go_vault", "take_treasure"],
                               goal={"has_treasure": True})
        self.assertFalse(out["outcome_changed"])
        self.assertEqual(out["difference"], {})

    def test_counterfactual_world_model_fact_form(self):
        b = _treasure_brain(LAB)
        out = b.counterfactual(
            edits={"entity": "door", "property": "open", "value": True},
            steps=["go_vault", "take_treasure"],
            goal={"has_treasure": True})
        self.assertTrue(out["outcome_changed"])
        self.assertTrue(any("door.open=True" in a for a in out["assumptions"]))

    def test_counterfactual_keyword_form(self):
        b = _treasure_brain(LAB)
        out = b.counterfactual(door_open=True,
                               steps=["go_vault", "take_treasure"],
                               goal={"has_treasure": True})
        self.assertTrue(out["outcome_changed"])


class TestSimulationRevision(unittest.TestCase):
    def test_observe_moves_success_toward_observed(self):
        rev = SimulationRevision()
        failed = rev.observe("pick_lock", "door_open", True, False, cycle=1)
        self.assertAlmostEqual(failed["prior"], 1.0, places=4)
        self.assertAlmostEqual(failed["adjusted"], 0.5, places=4)
        self.assertAlmostEqual(failed["moved"], -0.5, places=4)
        succeeded = rev.observe("pick_lock", "door_open", True, True, cycle=2)
        self.assertAlmostEqual(succeeded["adjusted"], 0.75, places=4)
        self.assertAlmostEqual(
            rev.predicted_success("pick_lock", "door_open"), 0.75, places=4)

    def test_observe_uses_causal_rate_as_prior(self):
        causal = CausalModel()
        causal.record("pick_lock", "door_open", False)
        rev = SimulationRevision()
        rec = rev.observe("pick_lock", "door_open", True, False, cycle=1,
                          causal=causal)
        # Prior is the Laplace rate 1/3; one observed failure halves the gap to 0.
        self.assertAlmostEqual(rec["prior"], 1 / 3, places=4)
        self.assertAlmostEqual(rec["adjusted"], 1 / 6, places=4)

    def test_observe_records_prediction_error_deduplicated(self):
        wm = WorldModel()
        rev = SimulationRevision()
        rev.observe("pick_lock", "door_open", True, False, cycle=3,
                    world_model=wm)
        rev.observe("pick_lock", "door_open", True, False, cycle=3,
                    world_model=wm)
        self.assertEqual(len(wm.prediction_errors()), 1)
        rev.observe("pick_lock", "door_open", True, False, cycle=4,
                    world_model=wm)
        self.assertEqual(len(wm.prediction_errors()), 2)

    def test_compare_matches_and_revises_mismatches(self):
        wm = WorldModel()
        rev = SimulationRevision()
        per_step = [{"action": "pick_lock", "effects": {"door_open": True}}]
        out = rev.compare({"door_open": True, "at": "lab"},
                          {"door_open": False, "at": "lab"},
                          per_step=per_step, cycle=5, world_model=wm)
        self.assertEqual(out["outcome"], "partial")
        self.assertEqual([m["key"] for m in out["matched"]], ["at"])
        self.assertEqual([m["key"] for m in out["mismatched"]], ["door_open"])
        self.assertEqual(len(wm.prediction_errors()), 1)
        self.assertLess(rev.predicted_success("pick_lock", "door_open"), 1.0)

    def test_revision_second_task_moves_toward_observed(self):
        # Fresh brain: the pick_lock route is predicted to work.
        fresh = _treasure_brain(LAB)
        fresh_scen = ScenarioGenerator(
            fresh.planner, fresh.simulator).generate({"door_open": True},
                                                    dict(fresh.world))
        pick_fresh = next(s for s in fresh_scen if "pick_lock" in s.steps)
        self.assertAlmostEqual(pick_fresh.predicted_success, 1.0, places=4)
        # Same brain shape, but after really failing pick_lock once:
        # run a door_open goal from the lab in the real RoomWorld.
        env = RoomWorld()
        env.s = dict(LAB)
        b = _treasure_brain(LAB)
        goal = b.add_goal("open the door", {"door_open": True})
        b.goals.activate(goal.id)
        b.run(env, 30)
        self.assertEqual(b.goals.all()[-1].status, "done")
        self.assertTrue(env.s["door_open"])
        # Second, structurally similar task from the same situation:
        # the pick_lock route's predicted success has moved toward 0.
        # The execution failure ALSO banned pick_lock in this
        # situation (a separate mechanism); clear that ban so this
        # test measures the simulation revision, not the ban.
        b.world = dict(LAB)
        b.planner.clear_bans()
        revised_scen = ScenarioGenerator(
            b.planner, b.simulator).generate({"door_open": True},
                                            dict(b.world))
        pick_revised = next(s for s in revised_scen if "pick_lock" in s.steps)
        self.assertLess(pick_revised.predicted_success,
                        pick_fresh.predicted_success)
        self.assertLess(pick_revised.predicted_success, 0.6)
        self.assertGreater(pick_revised.predicted_cost,
                           pick_fresh.predicted_cost)
        # The ranking flips: the key route is now ranked first.
        ranking = ScenarioComparator().rank(revised_scen)
        self.assertNotIn("pick_lock", ranking[0]["scenario"].steps)
        self.assertTrue(ranking[0]["reason"])

    def test_check_predictions_hooks_revision(self):
        b = _treasure_brain(LAB)
        action = next(a for a in room_world_actions() if a.name == "pick_lock")
        b._store_predictions(action)
        self.assertIn("predicted:pick_lock:door_open", b.unchecked_predictions())
        b._check_predictions()
        self.assertEqual(b.unchecked_predictions(), [])
        adjusted = b.simulation_revision.adjusted_success.get(
            ("pick_lock", "door_open"))
        self.assertIsNotNone(adjusted)
        self.assertLess(adjusted, 1.0)
        # The violated prediction is still recorded exactly once, by
        # the existing violation branch — the hook must not double it.
        errors = [r for r in b.world_model.prediction_errors()
                  if r["action"] == "pick_lock"]
        self.assertEqual(len(errors), 1)
        self.assertTrue(any(e["kind"] == "simulation_revised" for e in b.events))


class TestSimulationBasedPlanning(unittest.TestCase):
    def _planned_first_step(self, with_failure: bool):
        b = _treasure_brain(LAB)
        if with_failure:
            b.causal.record("pick_lock", "door_open", False)
        goal = b.add_goal("open the door", {"door_open": True})
        b.goals.activate(goal.id)
        self.assertTrue(b._do_make_plan({}, None))
        return b

    def test_planner_cost_order_is_fallback_without_evidence(self):
        b = self._planned_first_step(with_failure=False)
        # No causal evidence: simulations cannot discriminate, so the
        # cheaper pick_lock plan stands.
        self.assertEqual(b.plan[0].name, "pick_lock")
        self.assertTrue(any(e["kind"] == "simulation_compared" for e in b.events))

    def test_simulation_picks_better_simulated_plan(self):
        b = self._planned_first_step(with_failure=True)
        # One observed pick_lock failure: predicted success 1/3 vs 1.0,
        # so simulation ranking prefers the take_key route.
        self.assertEqual(b.plan[0].name, "take_key")
        comp = b.last_simulation_comparison
        self.assertIsNotNone(comp)
        self.assertTrue(comp["discriminates"])
        self.assertTrue(comp["simulated"])

    def test_simulation_ranking_is_bounded(self):
        b = _treasure_brain()
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        b.goals.activate(goal.id)
        self.assertTrue(b._do_make_plan({}, None))
        comp = b.last_simulation_comparison
        self.assertIsNotNone(comp)
        self.assertLessEqual(len(comp["scenarios"]), 3)
        for s in comp["scenarios"]:
            self.assertLessEqual(len(s["steps"]), 20)


if __name__ == "__main__":
    unittest.main()
