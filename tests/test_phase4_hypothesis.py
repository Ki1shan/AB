"""Phase 4: hypotheses, induction, abstraction, and the Phase 4
integration wiring (behavioural tests).

Hypotheses are testable guesses, never facts: they never enter LTM,
never drive an action, and a test corroborates or falsifies them
without ever verifying them. Induction proposes pending rules from
repeated episodes; abstraction proposes pending categories from
shared properties. The remaining tests pin the wiring items: goal
thoughts refreshed on resumption, the whole-sequence ban check, the
has_goal-conditioned ban, need-more-evidence as a deferment, the
budget_exceeded removal, and Goal.decompose wiring.
"""
import unittest

from ab.body.environment import Environment
from ab.brain import Brain
from ab.core.control import CognitiveControl
from ab.core.learning import Episode
from ab.planning.planner import Action, Planner


class SurpriseWorld(Environment):
    """heat_vapour is believed to make the vapour hot, but with a cold
    coil the vapour stays cold: the predicted effect fails."""

    def __init__(self):
        self.s = {"at": "lab", "coil": "cold", "vapour_hot": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        if name == "heat_vapour" and self.s["at"] == "lab":
            info = ("the coil hums but the vapour stays cold"
                    if self.s["coil"] == "cold" else "the vapour boils")
            if self.s["coil"] == "warm":
                self.s["vapour_hot"] = True
            self.log.append(f"{name}: {info}")
            return True, info
        if name == "warm_coil" and self.s["at"] == "lab":
            self.s["coil"] = "warm"
            self.log.append(f"{name}: the coil is warm now")
            return True, "the coil is warm now"
        self.log.append(f"{name}: nothing happened")
        return False, "nothing happened"


def surprise_actions() -> list[Action]:
    return [
        Action("heat_vapour", {"at": "lab"}, {"vapour_hot": True},
               cost=1.0, args={"_do": "heat_vapour"}),
        Action("warm_coil", {"at": "lab", "coil": "cold"}, {"coil": "warm"},
               cost=1.0, args={"_do": "warm_coil"}),
    ]


def _brain_with_open_hypothesis():
    """Drive a brain to the point where it holds an OPEN hypothesis
    (surprise seen, abduction run, no repeat executed yet), and return
    (brain, hypothesis_id)."""
    env = SurpriseWorld()
    b = Brain(surprise_actions(), granted={"compute", "environment"})
    b.add_goal("heat the vapour", {"vapour_hot": True})
    b.step(env)   # select
    b.step(env)   # plan [heat_vapour]
    b.step(env)   # execute heat_vapour: predicted effect fails
    b._set_plan([])  # keep the next cycle from immediately re-executing
    b.step(env)   # prediction violated -> abduction generates hypotheses
    open_hyps = [h for h in b.hypotheses if h.get("status") == "open"]
    assert open_hyps, "setup failed: no open hypothesis generated"
    return b, open_hyps[0]["id"]


class TestHypothesisGeneration(unittest.TestCase):
    def test_surprise_generates_ranked_hypothesis_with_test(self):
        env = SurpriseWorld()
        b = Brain(surprise_actions(), granted={"compute", "environment"})
        b.add_goal("heat the vapour", {"vapour_hot": True})
        b.run(env, 30)
        self.assertTrue(any(e["kind"] == "prediction_checked"
                            and "violated" in e["detail"] for e in b.events))
        hyps = b.hypotheses
        self.assertTrue(hyps, "no hypothesis was generated from the surprise")
        self.assertTrue(all(h.get("test") for h in hyps))
        # Ranked: non-increasing confidence, best first.
        confs = [h["confidence"] for h in hyps]
        self.assertEqual(confs, sorted(confs, reverse=True))
        self.assertTrue(all(h["subject"] == "heat_vapour" for h in hyps))
        # The hypothesis names the surprise's cause and how to test it.
        self.assertTrue(any("heat_vapour" in (h.get("test") or "")
                            for h in hyps))

    def test_failed_test_falsifies_with_event(self):
        b, hyp_id = _brain_with_open_hypothesis()
        updated = b.evaluate_hypothesis(
            hyp_id, {"met": False,
                     "detail": "coil warm, vapour still cold"})
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, "falsified")
        self.assertTrue(any(e["kind"] == "hypothesis_falsified"
                            for e in b.events))
        stored = [h for h in b.hypotheses if h["id"] == hyp_id][0]
        self.assertEqual(stored["status"], "falsified")

    def test_passed_test_corroborates_but_never_verifies(self):
        b, hyp_id = _brain_with_open_hypothesis()
        hyp = [h for h in b.hypotheses if h["id"] == hyp_id][0]
        before = hyp["confidence"]
        updated = b.evaluate_hypothesis(
            hyp_id, {"met": True, "detail": "repeat failed as predicted"})
        self.assertEqual(updated.status, "corroborated")
        self.assertGreater(updated.confidence, before)
        self.assertLessEqual(updated.confidence, 0.9)
        self.assertTrue(any(e["kind"] == "hypothesis_corroborated"
                            for e in b.events))
        # Corroboration is not verification: no LTM fact is created.
        self.assertEqual(b.ltm.query(hyp["subject"], hyp["relation"]), [])

    def test_hypotheses_never_enter_ltm_and_never_drive_action(self):
        b, hyp_id = _brain_with_open_hypothesis()
        hyp = [h for h in b.hypotheses if h["id"] == hyp_id][0]
        # No LTM fact from any hypothesis, and no inferred:* entry
        # carries a hypothesis claim into planning.
        self.assertEqual(b.ltm.query(hyp["subject"], hyp["relation"]), [])
        self.assertIsNone(
            b.state.get(f"inferred:{hyp['subject']}|{hyp['relation']}"))
        # The next plan uses only actions from the believed model —
        # nothing invented from a hypothesis.
        b.step()
        names = {a.name for a in b.plan}
        self.assertTrue(names <= {"heat_vapour", "warm_coil"},
                        f"plan invented a step from a hypothesis: {names}")

    def test_simulation_predicts_without_executing(self):
        b = Brain(surprise_actions(), granted={"compute", "environment"})
        b.world = {"at": "lab", "coil": "cold", "vapour_hot": False}
        steps = [a for a in surprise_actions()
                 if a.name in ("warm_coil", "heat_vapour")]
        state, met, assumptions = b.simulate_plan(
            steps, goal={"vapour_hot": True})
        self.assertTrue(met)
        self.assertTrue(state["vapour_hot"])
        self.assertTrue(assumptions)
        # Nothing was executed: the world is untouched.
        self.assertEqual(b.world.get("vapour_hot"), False)


class TestInductionAndAbstraction(unittest.TestCase):
    def test_induction_from_failures_proposes_pending_rule_and_approval_avoids(self):
        from ab.body.environment import RoomWorld, room_world_actions
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.world = {"at": "lab", "has_key": False, "door_open": False,
                   "has_treasure": False}
        for _ in range(3):
            b.learner.record(Episode(b.cycle, "pick_lock",
                                     {"at": "lab"}, False, []))
        pending = b.pending_rules()
        pick = [d for d in pending if d.get("action") == "pick_lock"]
        self.assertTrue(pick, "no pending rule proposed from 3 failures")
        self.assertEqual(pick[0]["status"], "pending")
        rid = pick[0]["rule_id"]
        self.assertTrue(b.approve_rule(rid))
        self.assertFalse(any(d.get("rule_id") == rid
                             for d in b.pending_rules()))
        self.assertEqual(b.rules.get(rid).status, "active")
        # Approval makes the next plan avoid the action.
        plan = b.planner.plan(b.world, {"has_treasure": True})
        self.assertIsNotNone(plan)
        self.assertNotIn("pick_lock", [a.name for a in plan.steps])

    def test_induction_from_successes_proposes_pending_rule_and_approval_applies(self):
        from ab.body.environment import RoomWorld, room_world_actions
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.world = {"at": "lab", "has_key": False, "door_open": False,
                   "has_treasure": False}
        for _ in range(3):
            b.learner.record(Episode(b.cycle, "take_key",
                                     {"at": "lab", "has_key": False},
                                     True, []))
        b.perceive_text("lab")
        b.step()
        pending = b.pending_rules()
        take = [d for d in pending if d.get("action") == "take_key"
                and d.get("outcome") == "success"]
        self.assertTrue(take, "no pending induced rule from 3 successes")
        rid = take[0]["rule_id"]
        self.assertTrue(rid.startswith("induce_"))
        self.assertTrue(b.approve_rule(rid))
        self.assertEqual(b.rules.get(rid).status, "active")
        self.assertFalse(any(d.get("rule_id") == rid
                             for d in b.pending_rules()))
        # The approved regularity is applied: the action it names stays
        # usable in planning (it is not banned, it is preferred
        # knowledge). With the cheaper decoy excluded, the next plan
        # routes through it.
        plan = b.planner.plan(b.world, {"has_treasure": True},
                              banned={"pick_lock"})
        self.assertIsNotNone(plan)
        self.assertIn("take_key", [a.name for a in plan.steps])

    def test_abstraction_proposes_pending_category_for_shared_properties(self):
        b = Brain()
        for subj in ("cat", "dog"):
            b.teach_fact(subj, "is_a", "mammal")
            b.teach_fact(subj, "located_in", "house")
            b.teach_fact(subj, "eats", "meat")
        b.perceive_text("cat dog")
        b.step()
        cats = b.proposed_categories
        self.assertTrue(cats, "no pending category proposed")
        cat = cats[0]
        self.assertEqual(sorted(cat["members"]), ["cat", "dog"])
        self.assertEqual(len(cat["shared"]), 3)
        self.assertEqual(cat["status"], "pending")
        # A proposal is not knowledge: nothing about it is in LTM.
        self.assertFalse(any("cat_" in f.subject or "cat_" in f.value
                             for f in b.ltm.all_facts()))


class TestWiring(unittest.TestCase):
    def test_goal_thoughts_refreshed_on_resumption(self):
        from ab.body.environment import RoomWorld, room_world_actions
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        b.step(env)   # plan
        old_cycle = b._goal_thoughts[g.id].created_cycle
        self.assertIsNotNone(b.suspend_active_goal("test"))
        b.step(env)
        b.step(env)
        self.assertGreater(b.cycle, old_cycle)
        self.assertTrue(b.resume_goal(g.id))
        thought = b._goal_thoughts[g.id]
        self.assertEqual(thought.created_cycle, b.cycle)
        self.assertEqual(thought.urgency, g.priority)
        self.assertGreater(thought.created_cycle, old_cycle)

    def test_whole_sequence_ban_check_rejects_later_banned_step(self):
        # Best and alternative differ in their FIRST action (plan_all
        # collapses prefixes that converge on one state with the same
        # first action, so a same-first-action alternative is never
        # retained in the first place).
        start_a = Action("start_a", {"at": "home"}, {"at": "a"}, cost=1.0)
        finish_a = Action("finish_a", {"at": "a"}, {"done": True}, cost=1.0)
        start_b = Action("start_b", {"at": "home"}, {"at": "b"}, cost=1.0)
        finish_b = Action("finish_b", {"at": "b"}, {"done": True}, cost=2.0)
        direct = Action("direct", {"at": "home"}, {"done": True}, cost=5.0)
        b = Brain([start_a, finish_a, start_b, finish_b, direct],
                  granted={"compute"})
        b.world = {"at": "home"}
        g = b.add_goal("finish", {"done": True})
        b.step()   # select
        b.step()   # plan: best [start_a, finish_a], alternative [start_b, finish_b]
        self.assertEqual([a.name for a in b.plan], ["start_a", "finish_a"])
        self.assertEqual([a.name for a in b.alternative_plan],
                         ["start_b", "finish_b"])
        # The helper itself: the second step is banned in the situation
        # the first step would leave the world in.
        b.planner.ban("finish_b", {"at": "b"})
        self.assertTrue(Planner.sequence_banned(
            b.alternative_plan, b.world, b.planner.banned_pairs))
        self.assertFalse(Planner.sequence_banned(
            [start_a, finish_a], b.world, b.planner.banned_pairs))
        # The brain must NOT adopt the alternative; it replans instead
        # (via the only remaining route, [direct]).
        self.assertTrue(b._try_alternative(finish_a, {"at": "a"}))
        self.assertTrue(any(e["kind"] == "alternative_rejected"
                            for e in b.events))
        self.assertEqual([a.name for a in b.plan], ["direct"])
        self.assertNotEqual([a.name for a in b.plan],
                            ["start_b", "finish_b"])

    def test_banned_matches_on_has_goal_conditioned_rule(self):
        risky = Action("risky", {}, {"done": True}, cost=1.0)
        b = Brain([risky], granted={"compute"})
        self.assertNotIn("risky", b._banned())
        b.teach_rule({"id": "ban_risky_when_goal", "priority": 0.9,
                      "source": "test",
                      "when": [{"key": "has_goal", "op": "==", "value": True}],
                      "then": [{"do": "avoid_action",
                                "args": {"name": "risky"}}]})
        b.add_goal("finish", {"done": True})
        b.step()  # select the goal: has_goal is now true
        self.assertIn("risky", b._banned())

    def test_need_more_evidence_recorded_as_deferred_not_fail(self):
        actions = []
        for i in range(20):
            actions.append(Action(f"a{i}", {f"s{i}": True},
                                  {f"s{i + 1}": True}, cost=1.0))
        b = Brain(actions, granted={"compute"})
        b.world = {"s0": True}
        b.add_goal("reach the end", {"s20": True})
        rule = b.rules.get("select_goal")
        fired_before, fail_before = rule.fired, rule.fail
        b.step()  # select_goal fires; the plan is too costly to attempt
        self.assertTrue(any(e["kind"] == "need_more_evidence"
                            for e in b.events))
        self.assertEqual(rule.fired, fired_before + 1)
        self.assertEqual(rule.fail, fail_before,
                         "a deferment must not count as a rule failure")
        self.assertEqual(rule.status, "active")

    def test_budget_exceeded_removed_over_budget_works(self):
        self.assertFalse(hasattr(CognitiveControl, "budget_exceeded"))
        self.assertFalse(hasattr(CognitiveControl(), "budget_exceeded"))
        c = CognitiveControl(max_cycles=10)
        self.assertFalse(c.over_budget(9))
        self.assertTrue(c.over_budget(10))
        self.assertTrue(c.over_budget(11))

    def test_goal_decompose_wired_for_conjunctive_goal(self):
        get_a = Action("get_a", {"x": True}, {"a": True}, cost=1.0)
        get_b = Action("get_b", {"x": False}, {"b": True}, cost=1.0)
        b = Brain([get_a, get_b], granted={"compute"})
        b.world = {"x": True}
        g = b.add_goal("get both", {"a": True, "b": True})
        self.assertTrue(hasattr(b.goals, "decompose"),
                        "Goal.decompose was removed, not wired")
        b.step()   # select
        b.step()   # make_plan: no direct plan -> decompose -> plan subgoal a
        subs = [sg for sg in b.goals.all() if sg.parent == g.id]
        self.assertEqual(len(subs), 2)
        self.assertEqual(sorted(tuple(sorted(sg.target.items())) for sg in subs),
                         [(("a", True),), (("b", True),)])
        self.assertEqual(g.status, "pending")
        active = b.goals.active()
        self.assertIsNotNone(active)
        self.assertEqual(active.parent, g.id)
        self.assertEqual([a.name for a in b.plan], ["get_a"])


if __name__ == "__main__":
    unittest.main()
