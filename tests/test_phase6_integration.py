"""Phase 6 integration tests (Worker 2): Brain wiring for learning.

Repeated-mistake detection, end-of-cycle consolidation (P-C/P-D
candidates + promotion + provenance), concepts C-1/C-2 used by
reasoning and planning, calibration wiring into predictions, the
retriever event on replanning, P-C/P-D learning + application (both
directions), and the Phase 6 §4.1 constraint-fallback narrowing.

All names here are invented for the tests (alpha/beta tools, gate_open
effect, hall/lab/vault rooms) — no world specs are imported, and no
pattern is learned from, or matched by, a name.
"""
import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.core.learning import Episode
from ab.planning.planner import Action
from ab.world.model import WorldModel
from ab.world.patterns import PatternLibrary

EFFECTS = {
    "open_gate": {"gate_open": True},
    "use_alpha_tool": {"gate_open": True},
    "use_beta_tool": {"gate_open": True},
    "go_vault": {"at": "vault"},
    "take_treasure": {"has_treasure": True},
}

ALL_ACTIONS = [
    ("open_gate", {"at": "hall"}, {"gate_open": True}),
    ("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, {"gate_open": True}),
    ("use_beta_tool", {"at": "hall", "has_beta_tool": True}, {"gate_open": True}),
]


def gate_actions(alpha_cost=2.0, beta_cost=1.0, decoy_cost=0.5):
    """Two tool openers (alpha durable, beta fragile) and a tool-less
    decoy, all claiming gate_open, plus the vault/treasure actions
    used by the §4.1 tests."""
    return [
        Action("open_gate", {"at": "hall"}, {"gate_open": True},
               cost=decoy_cost, args={"_do": "open_gate"}),
        Action("use_alpha_tool", {"at": "hall", "has_alpha_tool": True},
               {"gate_open": True}, cost=alpha_cost,
               args={"_do": "use_alpha_tool"}),
        Action("use_beta_tool", {"at": "hall", "has_beta_tool": True},
               {"gate_open": True}, cost=beta_cost,
               args={"_do": "use_beta_tool"}),
        Action("go_vault", {"at": "hall"}, {"at": "vault"},
               args={"_do": "go", "room": "vault"}),
        Action("take_treasure", {"at": "vault"}, {"has_treasure": True},
               args={"_do": "take_treasure"}),
    ]


def _ep(action, situation, ok, cycle=1):
    return {"action": action, "situation": dict(situation),
            "outcome": "success" if ok else "failed", "world": "test_world"}


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


def _open_gate_brain(alpha_cost=2.0, beta_cost=1.0, decoy_cost=0.5):
    brain = Brain(gate_actions(alpha_cost, beta_cost, decoy_cost),
                  granted={"compute", "environment"})
    brain.world = {"at": "hall", "has_alpha_tool": True,
                   "has_beta_tool": True, "gate_open": False}
    return brain


def _activate_goal(brain, target):
    goal = brain.add_goal("open the gate", dict(target))
    brain.goals.activate(goal.id)
    return goal


def _events_of(brain, kind):
    return [e for e in brain.events if e["kind"] == kind]


class TestLearnerRepeatedMistakes(unittest.TestCase):
    def test_two_failures_same_keys_different_values(self):
        brain = _open_gate_brain()
        learner = brain.learner
        learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        self.assertIsNone(learner.repeated_mistakes("open_gate", {"at": "hall"}))
        learner.record(Episode(2, "open_gate", {"at": "lab"}, False))
        rep = learner.repeated_mistakes("open_gate", {"at": "lab"})
        self.assertIsNotNone(rep)
        self.assertEqual(rep["action"], "open_gate")
        self.assertEqual(rep["features"], ["at"])
        self.assertEqual(rep["failures"], 2)
        self.assertEqual(rep["successes"], 0)

    def test_success_in_same_feature_set_clears_it(self):
        brain = _open_gate_brain()
        learner = brain.learner
        learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        learner.record(Episode(2, "open_gate", {"at": "lab"}, False))
        learner.record(Episode(3, "open_gate", {"at": "vault"}, True))
        self.assertIsNone(learner.repeated_mistakes("open_gate", {"at": "vault"}))

    def test_different_feature_set_is_a_different_mistake(self):
        brain = _open_gate_brain()
        learner = brain.learner
        learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        learner.record(Episode(2, "open_gate", {"at": "lab", "has_key": True}, False))
        # The second failure shares no feature set with the first.
        self.assertIsNone(learner.repeated_mistakes("open_gate", {"at": "lab"}))
        self.assertIsNone(
            learner.repeated_mistakes("open_gate", {"at": "lab", "has_key": True}))

    def test_max_episodes_is_a_constructor_parameter(self):
        brain = _open_gate_brain()
        from ab.core.learning import Learner
        learner = Learner(brain.planner, brain.rules, max_episodes=3)
        for i in range(5):
            learner.record(Episode(i, "open_gate", {"at": "hall"}, False))
        self.assertEqual(len(learner.episodes), 3)
        self.assertEqual(learner.max_episodes, 3)
        # The default is unchanged.
        self.assertEqual(brain.learner.max_episodes, 500)


class TestRepeatedMistakeEvent(unittest.TestCase):
    def test_second_feature_matched_failure_emits_event_and_flag(self):
        brain = _open_gate_brain()
        brain.learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        brain.learner.record(Episode(2, "open_gate", {"at": "lab"}, False))
        _activate_goal(brain, {"gate_open": True})
        open_gate = next(a for a in brain.planner.actions if a.name == "open_gate")
        brain._set_plan([open_gate])
        env = _FakeEnv(brain.world, {"open_gate": lambda e, a: (False, "no move")})
        self.assertFalse(brain._do_execute_step({}, env))
        events = _events_of(brain, "repeated_mistake")
        self.assertEqual(len(events), 1)
        self.assertIn("open_gate", events[0]["detail"])
        self.assertTrue(brain._prefer_alternative)
        self.assertEqual(brain._repeated_mistake_action, "open_gate")

    def test_next_plan_differs_from_the_failed_one(self):
        brain = _open_gate_brain(alpha_cost=1.0, beta_cost=2.0)
        _activate_goal(brain, {"gate_open": True})
        # The detector has fired (covered by the test above); the flag
        # is the input here, the changed plan is the assertion.
        brain._prefer_alternative = True
        brain._repeated_mistake_action = "open_gate"
        self.assertTrue(brain._do_make_plan({}, None))
        self.assertEqual(brain.plan[0].name, "use_alpha_tool")
        self.assertEqual(len(_events_of(brain, "alternative_preferred")), 1)
        # The flag is one-cycle: consumed by this planning pass.
        self.assertFalse(brain._prefer_alternative)

    def test_no_repeated_mistake_event_on_first_failure(self):
        brain = _open_gate_brain()
        _activate_goal(brain, {"gate_open": True})
        open_gate = next(a for a in brain.planner.actions if a.name == "open_gate")
        brain._set_plan([open_gate])
        env = _FakeEnv(brain.world, {"open_gate": lambda e, a: (False, "no move")})
        brain._do_execute_step({}, env)
        self.assertEqual(_events_of(brain, "repeated_mistake"), [])
        self.assertFalse(brain._prefer_alternative)


class TestRetrieverOnReplan(unittest.TestCase):
    def test_try_alternative_logs_retrieved_experience(self):
        brain = _open_gate_brain()
        brain.learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        brain.learner.record(Episode(2, "use_alpha_tool",
                                     {"at": "hall", "has_alpha_tool": True}, True))
        _activate_goal(brain, {"gate_open": True})
        open_gate = next(a for a in brain.planner.actions if a.name == "open_gate")
        alpha = next(a for a in brain.planner.actions if a.name == "use_alpha_tool")
        brain.alternative_plan = [alpha]
        brain._set_plan([open_gate])
        self.assertTrue(brain._try_alternative(open_gate, {"at": "hall"}))
        events = _events_of(brain, "retrieved_experience")
        self.assertEqual(len(events), 1)
        # Both indexed episodes are retrieved (k=5): the failed
        # open_gate episode and the successful use_alpha_tool one.
        self.assertIn("2 relevant episode(s)", events[0]["detail"])
        self.assertEqual(brain.plan[0].name, "use_alpha_tool")

    def test_episode_recording_feeds_retriever_and_prioritizer(self):
        brain = _open_gate_brain()
        ep = Episode(1, "open_gate", {"at": "hall"}, False)
        brain.learner.record(ep)
        self.assertEqual(len(brain.retriever), 1)
        self.assertEqual(brain.prioritizer.uses("open_gate", {"at": "hall"}), 1)
        found = brain.retriever.retrieve({"at": "hall"}, k=5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["action"], "open_gate")


class TestConsolidation(unittest.TestCase):
    def _pc_episodes(self, brain):
        # The durable tool succeeds 3 times across 2 situations.
        for cycle, sit in ((1, {"at": "hall", "has_alpha_tool": True}),
                           (2, {"at": "lab", "has_alpha_tool": True}),
                           (3, {"at": "hall", "has_alpha_tool": True})):
            brain.learner.record(Episode(cycle, "use_alpha_tool", sit, True))

    def test_pc_candidate_promoted_with_event_and_provenance(self):
        brain = _open_gate_brain()
        self._pc_episodes(brain)
        brain._consolidation_pass()
        promoted = brain.consolidator.promoted()
        self.assertEqual(len(promoted), 1)
        cand = promoted[0]
        self.assertEqual(cand["kind"], "pattern")
        self.assertEqual(cand["content"]["kind"], "durable_tool")
        self.assertEqual(cand["content"]["effect"], "gate_open")
        self.assertEqual(cand["promoted_evidence"], 3)
        self.assertEqual(cand["promoted_situations"], 2)
        events = _events_of(brain, "consolidated")
        self.assertEqual(len(events), 1)
        self.assertIn("durable_tool:gate_open", events[0]["detail"])
        self.assertIn("3 episodes", events[0]["detail"])
        self.assertIn("2 situations", events[0]["detail"])
        # The pattern is durable knowledge now, and the tool is
        # marked durable in the world model.
        self.assertEqual([p.kind for p in brain.patterns.patterns()],
                         ["durable_tool"])
        ent = brain.world_model.entity("alpha_tool")
        self.assertIsNotNone(ent)
        self.assertIs(ent.properties["breaks_after_first_use"].value, False)

    def test_below_bar_candidate_stays_candidate_until_forced(self):
        brain = _open_gate_brain()
        cid = brain.consolidator.add_candidate(
            "pattern",
            {"name": "durable_tool:gate_open", "kind": "durable_tool",
             "effect": "gate_open", "roles": [], "relations": []},
            [{"action": "use_alpha_tool", "outcome": "success"}],
            1, provenance="test: below the bar")
        brain._consolidation_pass()
        self.assertEqual(brain.consolidator.get(cid)["status"], "candidate")
        self.assertEqual(_events_of(brain, "consolidated"), [])
        self.assertTrue(brain.consolidate(cid))
        self.assertEqual(brain.consolidator.get(cid)["status"], "promoted")
        self.assertEqual(len(_events_of(brain, "consolidated")), 1)
        self.assertTrue(any(p.kind == "durable_tool"
                            for p in brain.patterns.patterns()))

    def test_pd_candidate_promoted(self):
        brain = _open_gate_brain()
        # A tooled action (so no P-A contrast fires): it fails twice
        # lacking has_key and succeeds once with it present.
        brain.learner.record(
            Episode(1, "use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, False))
        brain.learner.record(
            Episode(2, "use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, False))
        brain.learner.record(
            Episode(3, "use_alpha_tool",
                    {"at": "hall", "has_alpha_tool": True, "has_key": True}, True))
        brain._consolidation_pass()
        kinds = [p.kind for p in brain.patterns.patterns()]
        self.assertEqual(kinds, ["requires_precondition"])
        pd = [p for p in brain.patterns.patterns()
              if p.kind == "requires_precondition"][0]
        self.assertEqual(pd.name, "requires_has_key:gate_open")
        self.assertEqual(pd.effect, "gate_open")

    def test_pa_pb_still_apply_immediately_not_as_candidates(self):
        brain = _open_gate_brain()
        brain.learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        brain.learner.record(
            Episode(2, "use_alpha_tool",
                    {"at": "hall", "has_alpha_tool": True}, True))
        brain._consolidation_pass()
        self.assertEqual([p.kind for p in brain.patterns.patterns()],
                         ["unreliable_toolless_opener"])
        self.assertEqual(brain.consolidator.review(), [])
        self.assertEqual(len(_events_of(brain, "pattern_learned")), 1)

    def test_consolidate_unknown_id_returns_false(self):
        brain = _open_gate_brain()
        self.assertFalse(brain.consolidate("no_such_candidate"))


class TestConcepts(unittest.TestCase):
    def test_durable_tool_concept_learned_from_successes(self):
        brain = _open_gate_brain()
        for cycle, sit in ((1, {"at": "hall", "has_alpha_tool": True}),
                           (2, {"at": "lab", "has_alpha_tool": True})):
            brain.learner.record(Episode(cycle, "use_alpha_tool", sit, True))
        brain._consolidation_pass()
        concept = brain.concepts.get("durable_tool")
        self.assertIsNotNone(concept)
        self.assertTrue(concept.positives)
        self.assertEqual(concept.features, {"breaks_after_first_use": False})

    def test_broken_tool_is_a_negative_example(self):
        brain = _open_gate_brain()
        sit = {"at": "hall", "has_beta_tool": True}
        brain.learner.record(Episode(1, "use_beta_tool", sit, True))
        brain.learner.record(Episode(2, "use_beta_tool", sit, False))
        brain._consolidation_pass()
        concept = brain.concepts.get("durable_tool")
        self.assertIsNotNone(concept)
        self.assertTrue(concept.negatives)

    def test_requires_concept_learned_from_failures_and_success(self):
        brain = _open_gate_brain()
        brain.learner.record(Episode(1, "open_gate", {"at": "hall"}, False))
        brain.learner.record(Episode(2, "open_gate", {"at": "lab"}, False))
        brain.learner.record(
            Episode(3, "open_gate", {"at": "hall", "has_key": True}, True))
        brain._consolidation_pass()
        self.assertIn("requires_has_key", brain.concepts.names())

    def test_concept_used_by_reasoning(self):
        brain = _open_gate_brain()
        # The tool is durable (concept learned) and the world model
        # knows its defining property.
        brain.concepts.learn_durable_tool("alpha_tool", 3, False)
        brain.world_model.set_property(
            "alpha_tool", "breaks_after_first_use", False,
            status="inferred", confidence=0.7,
            provenance="test", cycle=0)
        brain.teach_fact("alpha_tool", "is_a", "device")
        brain.perceive_text("alpha_tool")
        brain.step()
        derived = {(d["subject"], d["relation"], d["value"])
                   for d in brain.last_inferences}
        self.assertIn(("alpha_tool", "is_a", "durable_tool"), derived)

    def test_concept_used_by_planning(self):
        # Equal base costs, and the decoy priced out of the way: the
        # durable-tool preference (0.8) must decide the plan.
        actions = gate_actions(alpha_cost=1.0, beta_cost=1.0, decoy_cost=5.0)
        brain = Brain(actions, granted={"compute", "environment"})
        brain.world = {"at": "hall", "has_alpha_tool": True,
                       "has_beta_tool": True}
        brain.concepts.learn_durable_tool("alpha_tool", 3, False)
        brain.world_model.set_property(
            "alpha_tool", "breaks_after_first_use", False,
            status="inferred", confidence=0.7,
            provenance="test", cycle=0)
        alpha = next(a for a in actions if a.name == "use_alpha_tool")
        beta = next(a for a in actions if a.name == "use_beta_tool")
        self.assertEqual(brain.planner.concept_multiplier(alpha), 0.8)
        self.assertEqual(brain.planner.concept_multiplier(beta), 1.0)
        _activate_goal(brain, {"gate_open": True})
        self.assertTrue(brain._do_make_plan({}, None))
        self.assertEqual(brain.plan[0].name, "use_alpha_tool")


class TestCalibration(unittest.TestCase):
    def test_confidence_moves_toward_observed_rate(self):
        brain = _open_gate_brain()
        action = next(a for a in brain.planner.actions if a.name == "open_gate")
        # No record: uncalibrated 1.0.
        brain._store_predictions(action)
        entry = brain.state.get("predicted:open_gate:gate_open")
        self.assertEqual(entry.confidence, 1.0)
        # 3 failures + 1 success: observed rate 0.25, Laplace 2/6.
        for outcome in (False, False, False, True):
            brain.calibration.record("open_gate", "gate_open", 1.0, outcome)
        brain.state.delete("predicted:open_gate:gate_open")
        brain._store_predictions(action)
        entry = brain.state.get("predicted:open_gate:gate_open")
        self.assertAlmostEqual(entry.confidence, 2 / 6)
        observed = 0.25
        self.assertLess(abs(entry.confidence - observed), abs(1.0 - observed))

    def test_execute_step_records_calibration(self):
        brain = _open_gate_brain()
        _activate_goal(brain, {"gate_open": True})
        open_gate = next(a for a in brain.planner.actions if a.name == "open_gate")
        brain._set_plan([open_gate])
        env = _FakeEnv(brain.world, {"open_gate": lambda e, a: (False, "no move")})
        brain._do_execute_step({}, env)
        self.assertEqual(brain.calibration.counts("open_gate", "gate_open"),
                         (0, 1))
        self.assertAlmostEqual(
            brain.calibration.calibrated_confidence("open_gate", "gate_open"),
            1 / 3)


class TestLearnPCPD(unittest.TestCase):
    def test_pc_learned_as_candidate_not_added(self):
        lib = PatternLibrary()
        episodes = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
                    _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, True),
                    _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True)]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        pattern = learned[0]
        self.assertEqual(pattern.kind, "durable_tool")
        self.assertEqual(pattern.name, "durable_tool:gate_open")
        self.assertTrue(pattern.candidate)
        self.assertEqual(lib.patterns(), [])
        self.assertEqual(
            {r["name"]: r["features"] for r in pattern.roles},
            {"tool": {"breaks_after_first_use": False},
             "fragile_tool": {"breaks_after_first_use": True}})

    def test_pc_needs_three_successes_two_situations_no_failure(self):
        lib = PatternLibrary()
        two = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
               _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, True)]
        self.assertEqual(lib.learn_from_episodes(two, EFFECTS), [])
        one_sit = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True)] * 3
        self.assertEqual(lib.learn_from_episodes(one_sit, EFFECTS), [])
        broken = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
                  _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, True),
                  _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
                  # The break is in a third situation, so no P-B
                  # (success-then-failure in the SAME situation) fires
                  # either: the P-C exclusion itself is what is tested.
                  _ep("use_alpha_tool", {"at": "vault", "has_alpha_tool": True}, False)]
        self.assertEqual(lib.learn_from_episodes(broken, EFFECTS), [])

    def test_pc_applies_both_directions(self):
        lib = PatternLibrary()
        episodes = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
                    _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, True),
                    _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True)]
        pattern = lib.learn_from_episodes(episodes, EFFECTS)[0]
        lib.add_pattern(pattern)
        model = WorldModel()
        # Durable tool, marked in the model: preferred (0.8).
        model.set_property("alpha_tool", "breaks_after_first_use", False,
                           status="inferred", confidence=0.7, cycle=1)
        self.assertEqual(
            lib.match("use_alpha_tool", {"at": "hall", "has_alpha_tool": True},
                      {"gate_open": True}, model), 0.8)
        # A suspect tool for the same effect: penalised (6.0).
        model.mark_single_use_suspect("beta_tool", cycle=2)
        self.assertEqual(
            lib.match("use_beta_tool", {"at": "hall", "has_beta_tool": True},
                      {"gate_open": True}, model), 6.0)
        # No marking in the model: no preference either way.
        fresh = WorldModel()
        self.assertEqual(
            lib.match("use_alpha_tool", {"at": "hall", "has_alpha_tool": True},
                      {"gate_open": True}, fresh), 1.0)

    def test_pd_learned_as_candidate_not_added(self):
        # A tooled action, so the P-A contrast (tool-less failure +
        # tooled success) cannot fire here: only the P-D candidate.
        lib = PatternLibrary()
        episodes = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, False),
                    _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, False),
                    _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True,
                                           "has_key": True}, True)]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        pattern = learned[0]
        self.assertEqual(pattern.kind, "requires_precondition")
        self.assertEqual(pattern.name, "requires_has_key:gate_open")
        self.assertTrue(pattern.candidate)
        self.assertEqual(lib.patterns(), [])

    def test_pd_needs_shared_missing_key_and_a_success(self):
        lib = PatternLibrary()
        base = {"at": "hall", "has_alpha_tool": True}
        one_failure = [_ep("use_alpha_tool", base, False),
                       _ep("use_alpha_tool", {**base, "has_key": True}, True)]
        self.assertEqual(lib.learn_from_episodes(one_failure, EFFECTS), [])
        no_success = [_ep("use_alpha_tool", base, False),
                      _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, False)]
        self.assertEqual(lib.learn_from_episodes(no_success, EFFECTS), [])
        # Failures do not share a missing key: each lacks a different key.
        unshared = [_ep("use_alpha_tool", {**base, "has_key": True}, False),
                    _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True,
                                           "has_other": True}, False),
                    _ep("use_alpha_tool", {**base, "has_key": True,
                                           "has_other": True}, True)]
        self.assertEqual(lib.learn_from_episodes(unshared, EFFECTS), [])

    def test_pd_applies_both_directions(self):
        lib = PatternLibrary()
        episodes = [_ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, False),
                    _ep("use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, False),
                    _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True,
                                           "has_key": True}, True)]
        pattern = lib.learn_from_episodes(episodes, EFFECTS)[0]
        lib.add_pattern(pattern)
        model = WorldModel()
        pre = {"at": "hall", "has_alpha_tool": True}
        # Key absent in the current state: penalised (6.0).
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model, state={"at": "hall", "has_key": False}), 6.0)
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model, state={"at": "hall"}), 6.0)
        # Key present: no penalty (1.0).
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model, state={"at": "hall", "has_key": True}), 1.0)
        # No state supplied: the precondition cannot be checked, so
        # the pattern never fires.
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model), 1.0)

    def test_pc_pd_do_not_change_pa_pb(self):
        # The Phase 5 record: P-A contrast + P-B success-then-failure
        # still learn immediately, with the same kinds and counts.
        lib = PatternLibrary()
        episodes = [_ep("open_gate", {"at": "hall"}, False),
                    _ep("use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, True),
                    _ep("use_beta_tool", {"at": "hall", "has_beta_tool": True}, True),
                    _ep("use_beta_tool", {"at": "hall", "has_beta_tool": True}, False)]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual([p.kind for p in learned],
                         ["unreliable_toolless_opener", "one_use_tool"])
        self.assertEqual(len(lib.patterns()), 2)
        self.assertTrue(all(not p.candidate for p in learned))

    def test_promoted_pc_prefers_durable_tool_in_planning(self):
        brain = _open_gate_brain()
        for cycle, sit in ((1, {"at": "hall", "has_alpha_tool": True}),
                           (2, {"at": "lab", "has_alpha_tool": True}),
                           (3, {"at": "hall", "has_alpha_tool": True})):
            brain.learner.record(Episode(cycle, "use_alpha_tool", sit, True))
        brain._consolidation_pass()
        alpha = next(a for a in brain.planner.actions if a.name == "use_alpha_tool")
        # Pattern 0.8 x concept 0.8 on the base cost 2.0.
        self.assertAlmostEqual(brain.planner.effective_cost(alpha), 2.0 * 0.8 * 0.8)

    def test_promoted_pd_penalises_only_when_key_absent(self):
        brain = _open_gate_brain()
        # A tooled action, so no P-A contrast fires alongside P-D.
        brain.learner.record(
            Episode(1, "use_alpha_tool", {"at": "hall", "has_alpha_tool": True}, False))
        brain.learner.record(
            Episode(2, "use_alpha_tool", {"at": "lab", "has_alpha_tool": True}, False))
        brain.learner.record(
            Episode(3, "use_alpha_tool",
                    {"at": "hall", "has_alpha_tool": True, "has_key": True}, True))
        brain._consolidation_pass()
        alpha = next(a for a in brain.planner.actions if a.name == "use_alpha_tool")
        brain.world = {"at": "hall", "has_alpha_tool": True,
                       "has_beta_tool": True, "gate_open": False,
                       "has_key": False}
        self.assertEqual(brain.planner.pattern_multiplier(alpha), 6.0)
        brain.world = {"at": "hall", "has_alpha_tool": True,
                       "has_beta_tool": True, "gate_open": False,
                       "has_key": True}
        self.assertEqual(brain.planner.pattern_multiplier(alpha), 1.0)


class TestFallbackNarrowing(unittest.TestCase):
    def test_learned_door_gated_edge_fails_goal_as_impossible(self):
        actions = [Action("go_vault", {"at": "hall"}, {"at": "vault"},
                          args={"_do": "go", "room": "vault"})]
        brain = Brain(actions, granted={"compute", "environment"})
        brain.world = {"at": "hall", "door_open": False}
        # The model learned the gate from an observed failed entry.
        brain.world_model.note_failed_entry("vault", cycle=0)
        env = _FakeEnv(brain.world, {"go": lambda e, a: (False, "blocked")})
        brain.add_goal("reach the vault", {"at": "vault"})
        stop = brain.run(env, 60)
        self.assertEqual(stop, "impossible")
        self.assertEqual(brain.last_stop_reason, "impossible")
        self.assertTrue(any(g.status == "failed" for g in brain.goals.all()))
        self.assertEqual(len(_events_of(brain, "model_disagreement_learned_constraint")), 1)
        self.assertEqual(_events_of(brain, "plan_model_disagreement"), [])
        self.assertFalse(brain.plan)

    def test_builtin_constraint_keeps_the_fallback(self):
        actions = [Action("go_vault", {"at": "hall"}, {"at": "vault"},
                          args={"_do": "go", "room": "vault"}),
                   Action("take_treasure", {"at": "vault"}, {"has_treasure": True},
                          args={"_do": "take_treasure"})]
        brain = Brain(actions, granted={"compute", "environment"})
        brain.world = {"at": "hall", "door_open": True}
        # The model believes the treasure is in the lab, not the
        # vault: taking it in the vault violates built-in containment.
        brain.world_model.add_relation("treasure", "located_in", "lab")
        env = _FakeEnv(brain.world, {"go": _go_move,
                                     "take_treasure": _take_treasure})
        brain.add_goal("get the treasure", {"has_treasure": True})
        # The plan is kept despite the disagreement, and the goal
        # completes on the planner's model.
        stop = brain.run(env, 60)
        self.assertEqual(len(_events_of(brain, "plan_model_disagreement")), 1)
        self.assertEqual(stop, "done")


def _go_move(env, args):
    env.s["at"] = args.get("room", "vault")
    return True, "moved"


def _take_treasure(env, args):
    env.s["has_treasure"] = True
    return True, "took it"


if __name__ == "__main__":
    unittest.main()
