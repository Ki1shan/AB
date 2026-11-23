"""Phase 3: core integration — derived views, plan in the goal record,
predictions, feasibility rescale, diverse alternatives, auto-resume,
queued interruptions, single decision path, learning/research wiring.
"""
import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.body.providers import MemoryKnowledgeProvider
from ab.body.research import Researcher
from ab.core.learning import Episode
from ab.planning.planner import Action, Planner
from ab.reasoning.decision import Decision, feasibility_for_cost


PAGES = {"https://a.org/p": "Paris is the capital of France.",
         "https://b.org/p": "Paris is the capital city of France."}


def _research_brain(urls):
    # Phase 11 integration (W5): the research tool is isolated, so
    # the child rebuilds its provider from the provider spec — a
    # closure transport would not survive the fork. The memory
    # provider's spec carries the pages, so the same fixture works
    # in-process (investigate) and through Brain.run (isolated).
    return Brain(researcher=Researcher(MemoryKnowledgeProvider(PAGES), None),
                 granted={"compute", "network"})


def _treasure_brain():
    env = RoomWorld()
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    goal = brain.add_goal("get the treasure", {"has_treasure": True})
    return env, brain, goal


def _drive_to_lab(env, brain):
    brain.step(env)
    brain.step(env)
    brain.step(env)
    assert [a.name for a in brain.plan] == ["pick_lock", "go_vault", "take_treasure"]
    assert brain.world.get("at") == "lab"


class TestKnownDerivedView(unittest.TestCase):
    def test_known_is_derived_view_not_a_stored_entry(self):
        b = Brain()
        self.assertIs(b.known("paris", "capital_of"), False)
        self.assertIsNone(b.state.get("known:paris|capital_of"))
        b.teach_fact("paris", "capital_of", "france")
        self.assertIs(b.known("paris", "capital_of"), True)
        # The derived view changes, but nothing is stored under known:*.
        self.assertIsNone(b.state.get("known:paris|capital_of"))
        self.assertFalse(any(k.startswith("known:") for k in b.state.keys()))
        self.assertIs(b.world.get("known:paris|capital_of"), True)

    def test_known_reads_inferred_state_entries(self):
        b = Brain()
        b._set_state("inferred:paris|capital_of", "france", source="inference:x",
                     confidence=0.9, entry_type="inferred")
        self.assertIs(b.known("paris", "capital_of"), True)
        self.assertFalse(any(k.startswith("known:") for k in b.state.keys()))

    def test_known_drives_goal_completion_end_to_end(self):
        b = _research_brain(list(PAGES))
        b.add_research_goal("paris", "capital_of", list(PAGES))
        self.assertEqual(b.run(None, 30), "done")
        self.assertIs(b.known("paris", "capital_of"), True)
        self.assertFalse(any(k.startswith("known:") for k in b.state.keys()))


class TestPlanLivesInGoalRecord(unittest.TestCase):
    def test_suspend_load_resume_same_object_path(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        rec = b.suspend_active_goal("interrupted by test")
        self.assertIsNotNone(rec)
        # The plan is in the goal record, not in a Brain attribute.
        self.assertEqual([a["name"] for a in b.goals.get(g.id).plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertFalse(hasattr(rec, "history"))
        self.assertTrue(rec.events)  # suspension keeps the event log
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "goals.json"), encoding="utf-8") as fh:
            goals = json.load(fh)
        self.assertEqual([a["name"] for a in goals["goals"][0]["plan"]],
                         ["pick_lock", "go_vault", "take_treasure"])
        with open(os.path.join(d, "plan.json"), encoding="utf-8") as fh:
            plan = json.load(fh)
        self.assertNotIn("steps", plan)
        self.assertNotIn("history", plan)
        b2 = Brain(room_world_actions(), granted={"compute", "environment"})
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertTrue(b2.resume_goal(g.id))
        self.assertEqual([a.name for a in b2.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual([a.name for a in b2.goals.get_plan(g.id)],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual(b2.run(env, 60), "done")
        self.assertEqual(b2.goals.get(g.id).status, "done")

    def test_plan_property_is_read_only(self):
        b = Brain()
        with self.assertRaises(AttributeError):
            b.plan = []
        self.assertEqual(b.plan, [])


class TestPlannerStatsDerivedView(unittest.TestCase):
    def test_stats_derived_from_learner_episodes(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for i in range(3):
            b.learner.record(Episode(i, "pick_lock", {"at": "lab"}, False))
        self.assertEqual(b.planner.stats, {"pick_lock": [0, 3]})
        self.assertEqual(b.planner.failure_rate("pick_lock"), 1.0)
        # Planner cost changes as episodes accumulate.
        action = next(a for a in b.planner.actions if a.name == "pick_lock")
        self.assertGreater(b.planner.effective_cost(action), action.cost)
        b.learner.record(Episode(3, "take_key", {"at": "lab"}, True))
        self.assertEqual(b.planner.stats["take_key"], [1, 0])
        # And the derived view survives save/load (no planner.json).
        d = tempfile.mkdtemp()
        b.save(d)
        self.assertNotIn("planner.json", os.listdir(d))
        b2 = Brain(room_world_actions(), granted={"compute", "environment"})
        b2.load(d)
        self.assertEqual(b2.planner.stats, b.planner.stats)
        self.assertEqual(len(b2.learner.episodes), 4)

    def test_standalone_planner_falls_back_to_direct_counts(self):
        p = Planner(room_world_actions())
        p.record("pick_lock", False)
        self.assertEqual(p.stats, {"pick_lock": [0, 1]})


class TestPredictionsStoredAndChecked(unittest.TestCase):
    def test_prediction_present_after_action_checked_next_cycle(self):
        env, b, g = _treasure_brain()
        b.step(env)   # select
        b.step(env)   # plan
        b.step(env)   # execute go_lab_from_hall
        preds = b.unchecked_predictions()
        self.assertEqual(preds, ["predicted:go_lab_from_hall:at"])
        entry = b.state.get(preds[0])
        self.assertEqual(entry.value, "lab")
        self.assertEqual(entry.source, "planner")
        self.assertEqual(entry.confidence, 1.0)
        b.step(env)   # next cycle: checks the prediction, then pick_lock fails
        self.assertTrue(any(e["kind"] == "prediction_checked"
                            and "go_lab_from_hall" in e["detail"]
                            and "outcome=met" in e["detail"] for e in b.events))
        self.assertNotIn("predicted:go_lab_from_hall:at", b.unchecked_predictions())

    def test_violated_prediction_is_logged(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        b.step(env)   # execute pick_lock: it fails
        self.assertIn("predicted:pick_lock:door_open", b.unchecked_predictions())
        b.step(env)   # next cycle checks it: door_open stayed False
        self.assertTrue(any(e["kind"] == "prediction_checked"
                            and "pick_lock" in e["detail"]
                            and "outcome=violated" in e["detail"] for e in b.events))
        self.assertNotIn("predicted:pick_lock:door_open", b.unchecked_predictions())

    def test_unchecked_prediction_is_detectable(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        # The go_lab prediction stored last cycle is still unchecked:
        # it is checked at the start of the NEXT cycle.
        self.assertEqual(b.unchecked_predictions(), ["predicted:go_lab_from_hall:at"])
        b.step(env)   # checks go_lab (met), executes pick_lock (fails)
        self.assertEqual(b.unchecked_predictions(), ["predicted:pick_lock:door_open"])


class TestFeasibilityRescale(unittest.TestCase):
    def test_four_step_plan_feasibility_above_gate_not_refused(self):
        self.assertAlmostEqual(feasibility_for_cost(4), 1 / (1 + 4 / 5), places=4)
        self.assertGreater(feasibility_for_cost(4), 0.25)
        self.assertAlmostEqual(feasibility_for_cost(15), 0.25, places=4)
        env, b, g = _treasure_brain()
        b.step(env)   # select
        self.assertIs(b.goals.active(), g)
        self.assertGreater(b.last_decision.feasibility, 0.25)
        self.assertAlmostEqual(b.last_decision.feasibility, 1 / (1 + 4 / 5), places=4)
        self.assertFalse(any(e["kind"] == "need_more_evidence" for e in b.events))

    def test_twenty_step_plan_refused_with_need_more_evidence(self):
        acts = [Action(f"s{i}", {"n": i}, {"n": i + 1}) for i in range(20)]
        b = Brain(acts, granted={"compute"})
        b.world = {"n": 0}
        g = b.add_goal("count to twenty", {"n": 20}, priority=0.9)
        b.step(None)   # select: 20-step plan, feasibility 0.20 < 0.25
        self.assertIsNot(b.goals.active(), g)
        self.assertIsNone(b.goals.active())
        self.assertTrue(any(e["kind"] == "need_more_evidence" for e in b.events))
        self.assertIsNotNone(b.last_decision)
        self.assertIsNone(b.last_decision.chosen)
        self.assertAlmostEqual(b.last_decision.feasibility, 0.20, places=4)


class TestAlternativesDiverseByFullSequence(unittest.TestCase):
    def test_two_plans_sharing_first_action_both_retained(self):
        p = Planner(room_world_actions())
        s = {"at": "hall", "has_key": False, "door_open": False, "has_treasure": False}
        plans = p.plan_all(s, {"has_treasure": True}, limit=2)
        self.assertEqual(len(plans), 2)
        seqs = [[a.name for a in pl.steps] for pl in plans]
        self.assertTrue(all(sq[0] == "go_lab_from_hall" for sq in seqs))
        self.assertNotEqual(seqs[0][1], seqs[1][1])  # pick_lock vs take_key
        self.assertIn("pick_lock", seqs[0])
        self.assertIn("take_key", seqs[1])

    def test_plan_keeps_signature_and_cheapest_behaviour(self):
        p = Planner(room_world_actions())
        s = {"at": "hall", "has_key": False, "door_open": False, "has_treasure": False}
        plan = p.plan(s, {"has_treasure": True})
        self.assertEqual([a.name for a in plan.steps],
                         ["go_lab_from_hall", "pick_lock", "go_vault", "take_treasure"])


class TestAutoResume(unittest.TestCase):
    def test_auto_resume_end_to_end(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        low = b.add_goal("low", {"has_key": True}, priority=0.4)
        b.step(env)   # select low
        b.step(env)   # plan low
        self.assertIs(b.goals.active(), low)
        urgent = b.add_goal("urgent", {"door_open": True}, priority=0.9)
        b.step(env)   # switch: low suspended, urgent active
        self.assertIs(b.goals.active(), urgent)
        self.assertIsNotNone(b.goals.suspended(low.id))
        self.assertEqual(b.run(env, 60), "done")
        # No explicit resume_goal() was called: the low goal was
        # auto-resumed when the urgent goal finished, and completed.
        self.assertEqual(urgent.status, "done")
        self.assertEqual(low.status, "done")
        self.assertTrue(any(e["kind"] == "resume" and "low" in e["detail"]
                            for e in b.events))
        self.assertTrue(any(e["kind"] == "replan_on_resume" for e in b.events))

    def test_explicit_resume_still_works(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        b.step(env)
        self.assertIsNone(b.goals.active())
        self.assertTrue(b.resume_goal(g.id))
        self.assertEqual(b.run(env, 60), "done")
        self.assertEqual(g.status, "done")


class TestQueuedInterruptions(unittest.TestCase):
    def _queued(self, b, trigger="urgent_percept"):
        b._resuming = True
        try:
            b._interrupt_current_goal(trigger, "test queued trigger")
        finally:
            b._resuming = False
        self.assertEqual(len(b._queued_interrupts), 1)

    def test_queued_interruption_executes_when_condition_persists(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        self._queued(b)
        b.step(env)   # next cycle: the urgent percept is still fresh
        self.assertIsNone(b.goals.active())
        self.assertIsNotNone(b.goals.suspended(g.id))
        self.assertFalse(any(e["kind"] == "dropped_interrupt" for e in b.events))

    def test_queued_interruption_dropped_when_condition_gone(self):
        env, b, g = _treasure_brain()
        _drive_to_lab(env, b)
        self._queued(b)   # no fresh urgent percept anywhere
        b.step(env)
        self.assertIs(b.goals.active(), g)
        self.assertIsNone(b.goals.suspended(g.id))
        self.assertTrue(any(e["kind"] == "dropped_interrupt"
                            and "urgent_percept" in e["detail"] for e in b.events))


class TestHistoryRemoved(unittest.TestCase):
    def test_no_brain_history_attribute(self):
        b = Brain()
        self.assertFalse(hasattr(b, "history"))
        env, b2, g = _treasure_brain()
        b2.run(env, 60)
        self.assertFalse(hasattr(b2, "history"))
        self.assertTrue(b2.events)  # the event log is the only history
        d = tempfile.mkdtemp()
        b2.save(d)
        with open(os.path.join(d, "plan.json"), encoding="utf-8") as fh:
            plan = json.load(fh)
        self.assertNotIn("history", plan)

    def test_stall_uses_goal_progress_not_history(self):
        # No code path calls the old is_stalled(history) world-string
        # check from the cycle: Brain has no history to pass it.
        import inspect
        from ab import brain as brain_mod
        src = inspect.getsource(brain_mod)
        self.assertNotIn("is_stalled(", src)


class TestSingleDecisionPath(unittest.TestCase):
    def test_last_decision_set_on_selection_and_switch(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        low = b.add_goal("low", {"has_key": True}, priority=0.4)
        b.step(env)   # select
        self.assertIsInstance(b.last_decision, Decision)
        self.assertIs(b.last_decision.chosen.payload, low)
        self.assertEqual(b.last_decision.chosen_goal_id, low.id)
        urgent = b.add_goal("urgent", {"door_open": True}, priority=0.9)
        b.step(env)   # switch
        self.assertIs(b.last_decision.chosen.payload, urgent)
        self.assertEqual(b.last_decision.chosen_goal_id, urgent.id)


class TestLearningActivationWiring(unittest.TestCase):
    def test_pending_approve_then_planner_avoids_action(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for i in range(3):
            b.learner.record(Episode(i, "pick_lock", {"at": "lab"}, False))
        pending = b.pending_rules()
        self.assertEqual(len(pending), 1)
        rid = pending[0]["rule_id"]
        self.assertTrue(rid.startswith("avoid_pick_lock_"))
        self.assertEqual(pending[0]["failures"], 3)
        self.assertTrue(b.approve_rule(rid))
        self.assertEqual(b.rules.get(rid).status, "active")
        self.assertEqual(b.pending_rules(), [])
        self.assertFalse(b.approve_rule("no_such_rule"))
        # The planner now consults the approved avoid rule: no plan
        # from the lab may use pick_lock.
        s = {"at": "lab", "has_key": False, "door_open": False, "has_treasure": False}
        plans = b.planner.plan_all(s, {"has_treasure": True}, limit=2)
        self.assertTrue(plans)
        self.assertTrue(all("pick_lock" not in [a.name for a in pl.steps]
                            for pl in plans))
        self.assertEqual([a.name for a in plans[0].steps],
                         ["take_key", "unlock", "go_vault", "take_treasure"])

    def test_reject_rule_marks_rejected(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for i in range(3):
            b.learner.record(Episode(i, "pick_lock", {"at": "lab"}, False))
        rid = b.pending_rules()[0]["rule_id"]
        self.assertTrue(b.reject_rule(rid))
        self.assertEqual(b.rules.get(rid).status, "retired")
        self.assertEqual(b.pending_rules(), [])
        self.assertFalse(b.reject_rule(rid))

    def test_approve_claim_admits_fact(self):
        b = Brain()
        self.assertEqual(b.approve_claim("paris", "capital_of", "france"), "verified")
        self.assertEqual(b.ask("paris", "capital_of").best[0], "france")
        self.assertIs(b.known("paris", "capital_of"), True)


class TestResearchUntrustedRouting(unittest.TestCase):
    def test_single_domain_claim_is_untrusted_state_not_ltm(self):
        b = _research_brain(["https://a.org/p"])
        out = b._research_tool(b.researcher.investigate(["https://a.org/p"], b.cycle))
        self.assertIsNone(b.ask("paris", "capital_of").best)
        self.assertIs(b.known("paris", "capital_of"), False)
        entry = b.state.get("untrusted:paris|capital_of")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "france")
        self.assertEqual(entry.status, "untrusted")
        self.assertEqual(entry.source, "web:a.org")
        self.assertIn("https://a.org/p", entry.provenance)
        self.assertIn("registrable_domain", entry.provenance)
        # Untrusted entries never enter the planner's world view.
        self.assertNotIn("untrusted:paris|capital_of", b.world)

    def test_two_domain_claim_admitted_no_untrusted_entry(self):
        b = _research_brain(list(PAGES))
        b._research_tool(b.researcher.investigate(list(PAGES), b.cycle))
        self.assertEqual(b.ask("paris", "capital_of").best[0], "france")
        self.assertIsNone(b.state.get("untrusted:paris|capital_of"))


class TestAuditRedactionFallback(unittest.TestCase):
    def test_event_redacts_secret_marker_values(self):
        b = Brain()
        b._event("act", "tool returned api_key=SECRET123 and token=TOK456")
        detail = b.events[-1]["detail"]
        # Format-agnostic: the secret VALUES must not survive, whatever
        # the redaction marker is (audit.py uses "[REDACTED]").
        self.assertNotIn("SECRET123", detail)
        self.assertNotIn("TOK456", detail)
        self.assertIn("api_key", detail)
        self.assertIn("token", detail)


if __name__ == "__main__":
    unittest.main()
