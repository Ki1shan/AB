"""Phase 5 brain-integration tests (Worker 2).

Brain-level coverage for the Phase 5 integration in ``ab/brain.py`` /
``ab/planning/planner.py``: hypothesis branching, SurpriseWorld
recovery, induced-rule cost preference, Laplace-smoothed prediction
confidence, world-model revision on a violated prediction, partial
observability, unknown vs false, pattern cost through the planner, and
exogenous-change detection.

All worlds here are the eval worlds themselves (imported, not
re-implemented): SurpriseWorld from ``ab/eval/reasoning_tasks.py`` and
ChamberWorld from ``ab/eval/disconnected.py``.
"""

import unittest

from ab.brain import Brain
from ab.core.learning import Episode
from ab.eval.disconnected import (
    WORLD_A,
    _run_with_brain,
    chamber_world_actions,
    make_variant_factories,
)
from ab.eval.reasoning_tasks import (
    SurpriseWorld,
    run_hypothesis_task,
    surprise_world_actions,
)
from ab.planning.planner import Action
from ab.world.model import EpistemicStatus
from ab.world.patterns import PatternLibrary


def _full_factory():
    return make_variant_factories()["full"]


class TestHypothesisBranching(unittest.TestCase):
    def test_two_competing_hypotheses_held_then_one_pruned(self):
        # Phase 4 carry-forward: branching had no brain-level test.
        # The surprise abduces TWO competing hypotheses about the same
        # action; both stay open through the next cycle; new evidence
        # then prunes one (falsified + event) while the other survives.
        env = SurpriseWorld()
        brain = _full_factory()(surprise_world_actions(),
                                {"compute", "environment"})
        brain.add_goal("heat the vapour", {"vapour_hot": True})
        for _ in range(4):
            brain.step(env)
        held = brain.hypotheses
        self.assertEqual(len(held), 2)
        self.assertTrue(all(h["status"] == "open" for h in held))
        self.assertEqual({h["subject"] for h in held}, {"heat_vapour"})
        # Held through another cycle: still open after cycle 5.
        brain.step(env)
        held = brain.hypotheses
        self.assertEqual(len(held), 2)
        self.assertTrue(all(h["status"] == "open" for h in held))
        # New evidence against the first: the repeat succeeded, so a
        # hypothesis that predicted failure is falsified.
        first, second = held[0], held[1]
        updated = brain.evaluate_hypothesis(
            first["id"],
            {"met": False,
             "detail": "repeat 'heat_vapour' succeeded; hypothesis "
                       "predicted failure"})
        self.assertEqual(updated.status, "falsified")
        self.assertTrue(any(
            e["kind"] == "hypothesis_falsified"
            and first["id"] in e["detail"] for e in brain.events))
        survivors = {h["id"]: h for h in brain.hypotheses}
        self.assertEqual(survivors[first["id"]]["status"], "falsified")
        self.assertEqual(survivors[second["id"]]["status"], "open")


class TestSurpriseWorldRecovery(unittest.TestCase):
    def test_surprise_world_completes_end_to_end(self):
        # The Phase 5 criterion (design SS4.2): the brain must finish
        # heating the vapour, not just diagnose why it stayed cold.
        res = run_hypothesis_task(_full_factory())
        self.assertTrue(res["passed"], res["detail"])
        self.assertTrue(res["legs"]["goal_completed"])
        self.assertEqual(res["evidence"]["goal_status"], "done")
        self.assertEqual(res["evidence"]["env_log"], [
            "heat_vapour: the coil hums but the vapour stays cold",
            "warm_coil: the coil is warm now",
            "heat_vapour: the vapour boils"])

    def test_world_model_revision_on_violated_prediction(self):
        # A violated prediction revises the model (old value retained
        # in revisions) and the next plan is built from the revised
        # model: it warms the coil first instead of re-heating cold.
        env = SurpriseWorld()
        brain = _full_factory()(surprise_world_actions(),
                                {"compute", "environment"})
        brain.add_goal("heat the vapour", {"vapour_hot": True})
        for _ in range(4):
            brain.step(env)
        revised = [e for e in brain.events if e["kind"] == "model_revised"]
        self.assertTrue(revised, "no model_revised event")
        self.assertTrue(any("vapour_hot" in e["detail"]
                            and "True -> False" in e["detail"]
                            for e in revised))
        revs = [r for r in brain.world_model.revisions()
                if r["key"] == "vapour_hot"]
        self.assertTrue(revs, "no vapour_hot revision retained")
        self.assertEqual(revs[0]["old_value"], True)
        self.assertEqual(revs[0]["new_value"], False)
        plans = [e["detail"] for e in brain.events if e["kind"] == "plan"]
        self.assertTrue(any(d.startswith("warm_coil") for d in plans),
                        f"no post-revision plan warms first: {plans}")

    def test_prediction_confidence_uses_causal_rate(self):
        # First prediction is uncalibrated 1.0; after the failure is
        # recorded, the next prediction for the same action/effect is
        # Laplace-smoothed (0 + 1) / (1 + 2) = 1/3.
        env = SurpriseWorld()
        brain = _full_factory()(surprise_world_actions(),
                                {"compute", "environment"})
        brain.add_goal("heat the vapour", {"vapour_hot": True})
        for _ in range(3):
            brain.step(env)
        first = brain.state.get("predicted:heat_vapour:vapour_hot")
        self.assertIsNotNone(first)
        self.assertEqual(first.value, True)
        self.assertEqual(first.confidence, 1.0)
        for _ in range(3):
            brain.step(env)
        self.assertEqual(brain.cycle, 6)
        later = brain.state.get("predicted:heat_vapour:vapour_hot")
        self.assertIsNotNone(later)
        self.assertEqual(later.value, True)
        self.assertLess(later.confidence, 1.0)
        self.assertAlmostEqual(later.confidence, 1 / 3, places=9)


class TestInducedRuleCostPreference(unittest.TestCase):
    def test_approval_drops_effective_cost_and_reorders_plan(self):
        # Two equal-cost alternatives for the same effect; the tie
        # breaks to the first-listed action until the induced success
        # rule for the other is approved, then the planner prefers it.
        actions = [
            Action("fetch_beta", {"at": "lab"}, {"has_widget": True},
                   cost=1.0),
            Action("fetch_alpha", {"at": "lab"}, {"has_widget": True},
                   cost=1.0),
        ]
        brain = Brain(actions, granted={"compute", "environment"})
        brain.world = {"at": "lab", "has_widget": False}
        for _ in range(3):
            brain.learner.record(Episode(
                brain.cycle, "fetch_alpha", {"at": "lab"}, True, []))
        brain.perceive_text("widget")
        brain.step()
        pending = [d for d in brain.pending_rules()
                   if d.get("action") == "fetch_alpha"
                   and d.get("outcome") == "success"]
        self.assertTrue(pending, "no induced success rule proposed")
        alpha = next(a for a in actions if a.name == "fetch_alpha")
        beta = next(a for a in actions if a.name == "fetch_beta")
        self.assertEqual(brain.planner.effective_cost(alpha), 1.0)
        self.assertEqual(brain.planner.effective_cost(beta), 1.0)
        before = brain.planner.plan(brain.world, {"has_widget": True})
        self.assertEqual([a.name for a in before.steps], ["fetch_beta"])
        self.assertTrue(brain.approve_rule(pending[0]["rule_id"]))
        self.assertEqual(brain.planner.effective_cost(alpha), 0.8)
        self.assertEqual(brain.planner.effective_cost(beta), 1.0)
        after = brain.planner.plan(brain.world, {"has_widget": True})
        self.assertEqual([a.name for a in after.steps], ["fetch_alpha"])


class TestPartialObservability(unittest.TestCase):
    def test_chamber_completes_without_assuming_unobserved_key_false(self):
        # From the start room the door cannot be seen at all, so
        # door_open is unknown (not False) — yet the goal completes,
        # and no plan ever required the unobserved key to be False.
        actions = chamber_world_actions(WORLD_A)
        fresh = Brain(actions, granted={"compute", "environment"})
        self.assertEqual(fresh.world_model.status_of("door", "open"),
                         EpistemicStatus.unknown)
        self.assertEqual(fresh.world_model.observed_keys(), set())
        rec, brain = _run_with_brain(WORLD_A, _full_factory(), 80)
        self.assertTrue(rec["completed"])
        self.assertEqual(rec["cycles"], 14)
        pre_by_name = {a.name: dict(a.pre) for a in actions}
        for event in brain.events:
            if event["kind"] != "plan":
                continue
            detail = event["detail"].split(" (cost")[0]
            for name in detail.split(" > "):
                pre = pre_by_name.get(name.strip())
                if pre is None:
                    continue
                self.assertIsNot(
                    pre.get("door_open"), False,
                    f"plan step {name} requires unobserved door_open False")


class TestUnknownVsFalse(unittest.TestCase):
    def test_unknown_is_not_known_and_false_is_disputed(self):
        brain = Brain([], granted={"compute"})
        # Unknown: nothing stored, nothing derivable.
        answer = brain.ask("zorblax", "capital_of")
        self.assertFalse(answer.known)
        self.assertIn("don't know", answer.message)
        # False contradicted by a verified fact: boils_at is
        # functional, so two teacher-taught values conflict and both
        # become disputed in LTM.
        brain.teach_fact("water", "boils_at", "100", "teacher")
        self.assertEqual(
            brain.ltm.query("water", "boils_at", "100")[0].status,
            "verified")
        brain.teach_fact("water", "boils_at", "90", "teacher")
        facts = brain.ltm.query(
            "water", "boils_at",
            statuses=("verified", "unverified", "disputed", "retracted"))
        self.assertEqual(sorted(f.status for f in facts),
                         ["disputed", "disputed"])
        self.assertIn("disputed",
                      brain.verify("water", "boils_at", "90"))
        # After one reasoning cycle the brain answers with BOTH values
        # as disputed — it never silently picks one side.
        brain.perceive_text("water boils_at")
        brain.step()
        disputed = brain.ask("water", "boils_at")
        self.assertTrue(disputed.known)
        self.assertEqual(sorted(v for v, _, _, _ in disputed.answers),
                         ["100", "90"])
        self.assertTrue(all(s == "disputed"
                            for _, _, s, _ in disputed.answers))


class TestPatternCostThroughPlanner(unittest.TestCase):
    def test_penalised_action_deprioritised_when_alternative_exists(self):
        # P-A learned from contrast (toolless chant fails, tooled
        # use_zorblat succeeds for door_open): the planner then
        # penalises the decoy 6x and the first plan routes through a
        # tool instead.
        actions = chamber_world_actions(WORLD_A)
        planner = Brain(actions, granted={"compute", "environment"}).planner
        lib = PatternLibrary()
        learned = lib.learn_from_episodes(
            [{"action": "chant", "situation": {"at": "green"},
              "outcome": "failure"},
             {"action": "use_zorblat",
              "situation": {"at": "green", "has_zorblat": True},
              "outcome": "success"}],
            action_effects={a.name: dict(a.eff) for a in actions})
        self.assertTrue(any(p.kind == "unreliable_toolless_opener"
                            for p in learned))
        chant = next(a for a in actions if a.name == "chant")
        world = {"at": "green", "has_zorblat": False,
                 "has_quivern": False, "door_open": False,
                 "has_relic": False}
        self.assertEqual(planner.effective_cost(chant), 0.5)
        first_before = planner.plan_all(
            world, {"door_open": True}, set(), limit=2)[0]
        self.assertEqual([a.name for a in first_before.steps], ["chant"])
        planner.attach_patterns(lib, lambda: (None, planner.actions))
        self.assertEqual(planner.effective_cost(chant), 3.0)
        first_after = planner.plan_all(
            world, {"door_open": True}, set(), limit=2)[0]
        self.assertEqual([a.name for a in first_after.steps],
                         ["take_zorblat", "use_zorblat"])


class TestExogenousChange(unittest.TestCase):
    def test_door_reset_recorded_as_exogenous(self):
        # The chamber mechanism resets once mid-task: the door swings
        # shut with no action between the observations, so the brain
        # records it as an exogenous change, in both its own event log
        # and the world model's event trail.
        rec, brain = _run_with_brain(WORLD_A, _full_factory(), 80)
        self.assertTrue(rec["completed"])
        self.assertTrue(rec["mid_task_reset_observed"])
        changed = [e for e in brain.events if e["kind"] == "world_changed"]
        self.assertTrue(any("door_open" in e["detail"] for e in changed),
                        f"no world_changed for door_open: {changed}")
        exogenous = [ev for ev in brain.world_model.events()
                     if ev.cause == "exogenous"]
        self.assertTrue(any(ev.key == "door_open" for ev in exogenous))
        reset = next(ev for ev in exogenous if ev.key == "door_open")
        self.assertEqual(reset.old_value, True)
        self.assertEqual(reset.new_value, False)


if __name__ == "__main__":
    unittest.main()
