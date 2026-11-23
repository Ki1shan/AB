"""Phase 4: ReasoningEngine agenda loop — budget, strategy selection,
rule recording, insufficient evidence, constraints (behavioural tests)."""
import unittest

from ab.core.memory import LongTermMemory
from ab.reasoning.engine import ReasoningEngine
from ab.reasoning.inference import Inference
from ab.reasoning.rules import Rule, RuleEngine


def _world():
    """LTM with a small derivation graph for three subjects."""
    m = LongTermMemory()
    for subj, parent in (("cat", "mammal"), ("dog", "mammal"), ("bird", "animal")):
        m.add(subj, "is_a", parent, "teacher")
    m.add("mammal", "located_in", "house", "teacher")
    m.add("animal", "located_in", "forest", "teacher")
    return m


def _facts_fn(ltm):
    def fn(subject, relation=None):
        return ltm.query(subject, relation, statuses=("verified",))
    return fn


def _engine(ltm=None, max_steps=20, max_branches=4):
    ltm = ltm or _world()
    return ReasoningEngine(Inference(ltm), RuleEngine(), max_steps=max_steps,
                           max_branches=max_branches), ltm


class TestBudget(unittest.TestCase):
    def test_small_budget_exhausts_and_returns_partial_results(self):
        engine, ltm = _engine(max_steps=2)
        full, _ = _engine()
        full_result = full.reason(["cat", "dog", "bird"], _facts_fn(ltm))
        result = engine.reason(["cat", "dog", "bird"], _facts_fn(ltm))
        self.assertTrue(result.budget_exhausted)
        self.assertEqual(result.steps_used, 2)
        self.assertIn("reasoning_budget_exhausted", [k for k, _ in result.events])
        # Partial: fewer derivations than the unbounded run, but not empty.
        self.assertTrue(0 < len(result.derivations) < len(full_result.derivations))

    def test_generous_budget_does_not_exhaust(self):
        engine, ltm = _engine(max_steps=20)
        result = engine.reason(["cat", "dog"], _facts_fn(ltm))
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(result.steps_used, 3)  # rules + 2 subjects


class TestStrategy(unittest.TestCase):
    def test_deduction_by_default(self):
        engine, ltm = _engine()
        result = engine.reason(["cat"], _facts_fn(ltm))
        self.assertEqual(result.strategy, "deduction")

    def test_abduction_on_surprise(self):
        engine, ltm = _engine()
        episodes = ([{"action": "heat_water", "situation": {"heat": "high"},
                      "outcome": "boiling"} for _ in range(3)]
                    + [{"action": "heat_water", "situation": {"heat": "low"},
                        "outcome": "warm"} for _ in range(2)])
        predictions = [{"action": "heat_water", "expected": "warm",
                        "actual": "boiling", "met": False}]
        result = engine.reason(["cat"], _facts_fn(ltm), episodes=episodes,
                               predictions=predictions)
        self.assertEqual(result.strategy, "abduction")
        self.assertTrue(result.hypotheses)
        self.assertEqual(result.hypotheses[0].value, "heat=high")
        self.assertIsNotNone(result.hypotheses[0].test)

    def test_induction_on_repeated_outcomes(self):
        engine, ltm = _engine()
        episodes = [{"action": "boil", "situation": {"liquid": "water"},
                     "outcome": "boils"} for _ in range(3)]
        result = engine.reason(["cat"], _facts_fn(ltm), episodes=episodes)
        self.assertEqual(result.strategy, "induction")
        self.assertIn("rule_proposed", [k for k, _ in result.events])

    def test_met_prediction_is_not_a_surprise(self):
        engine, ltm = _engine()
        predictions = [{"action": "heat_water", "expected": "warm",
                        "actual": "warm", "met": True}]
        result = engine.reason(["cat"], _facts_fn(ltm), predictions=predictions)
        self.assertEqual(result.strategy, "deduction")
        self.assertEqual(result.hypotheses, [])


class TestRulesRecorded(unittest.TestCase):
    def _engine_with_rules(self):
        engine, ltm = _engine()
        engine.rules.add(Rule(id="r_high",
                              when=[{"key": "subject", "op": "==", "value": "cat"}],
                              then=[{"do": "note", "args": {}}], priority=0.9))
        engine.rules.add(Rule(id="r_mid",
                              when=[{"key": "has_goal", "op": "==", "value": False}],
                              then=[{"do": "note", "args": {}}], priority=0.5))
        engine.rules.add(Rule(id="r_low",
                              when=[{"key": "surprise", "op": "==", "value": True}],
                              then=[{"do": "note", "args": {}}], priority=0.3))
        return engine, ltm

    def test_considered_rules_records_all_matched_not_just_first(self):
        engine, ltm = self._engine_with_rules()
        predictions = [{"action": "x", "expected": "a", "actual": "b", "met": False}]
        result = engine.reason(["cat"], _facts_fn(ltm), predictions=predictions)
        self.assertEqual({r.id for r in result.considered_rules},
                         {"r_high", "r_mid", "r_low"})
        self.assertEqual(len(result.considered_rules), 3)

    def test_fired_rules_ordered_by_effective_priority(self):
        engine, ltm = self._engine_with_rules()
        predictions = [{"action": "x", "expected": "a", "actual": "b", "met": False}]
        result = engine.reason(["cat"], _facts_fn(ltm), predictions=predictions)
        self.assertEqual([r.id for r in result.fired_rules],
                         ["r_high", "r_mid", "r_low"])

    def test_no_matching_rules_is_empty(self):
        engine, ltm = _engine()
        result = engine.reason(["cat"], _facts_fn(ltm))
        self.assertEqual(result.considered_rules, [])
        self.assertEqual(result.fired_rules, [])


class TestInsufficientEvidence(unittest.TestCase):
    def test_unknown_goal_sets_flag_with_needed_text_and_no_guess(self):
        engine, ltm = _engine()
        result = engine.reason(["cat"], _facts_fn(ltm),
                               goal_targets=[{"subject": "atlantis",
                                              "relation": "capital_of"}])
        self.assertTrue(result.insufficient_evidence)
        self.assertIsInstance(result.needed, str)
        self.assertIn("atlantis", result.needed)
        # No guessed value: nothing derived for the unknown target.
        self.assertFalse(any(d.subject == "atlantis" for d in result.derivations))

    def test_supported_goal_is_not_insufficient(self):
        engine, ltm = _engine()
        result = engine.reason(["cat"], _facts_fn(ltm),
                               goal_targets=[{"subject": "cat",
                                              "relation": "located_in",
                                              "value": "house"}])
        self.assertFalse(result.insufficient_evidence)
        self.assertIsNone(result.needed)
        self.assertTrue(any(d.subject == "cat" and d.relation == "located_in"
                            and d.value == "house" for d in result.derivations))


class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.engine, _ = _engine()

    def _deriv(self, confidence):
        from ab.reasoning.engine import Derivation
        return Derivation("cat", "located_in", "house", None, confidence)

    def test_rejects_violating_derivation_naming_constraint(self):
        ok, name = self.engine.check_constraints(
            self._deriv(0.5),
            [{"key": "confidence", "op": ">=", "value": 0.8}])
        self.assertFalse(ok)
        self.assertEqual(name, "confidence")

    def test_named_constraint_reports_its_name(self):
        ok, name = self.engine.check_constraints(
            self._deriv(0.5),
            [{"name": "min_conf", "key": "confidence", "op": ">=", "value": 0.8}])
        self.assertFalse(ok)
        self.assertEqual(name, "min_conf")

    def test_passing_derivation(self):
        ok, name = self.engine.check_constraints(
            self._deriv(0.9),
            [{"key": "confidence", "op": ">=", "value": 0.8}])
        self.assertTrue(ok)
        self.assertIsNone(name)

    def test_reason_drops_violating_derivations(self):
        engine, ltm = _engine()
        result = engine.reason(["cat"], _facts_fn(ltm),
                               constraints=[{"key": "confidence", "op": ">=",
                                             "value": 0.99}])
        # The 2-step cat->house derivation has confidence 0.95 < 0.99.
        self.assertFalse(any(d.value == "house" for d in result.derivations))
        self.assertIn("constraint_violation", [k for k, _ in result.events])


class TestEvaluateHypothesis(unittest.TestCase):
    def setUp(self):
        self.engine, _ = _engine()

    def _hyp(self, confidence=0.5):
        from ab.reasoning.hypothesis import Hypothesis, make_hypothesis_id
        return Hypothesis(id=make_hypothesis_id("heat_water", "caused_by", "heat=high"),
                          subject="heat_water", relation="caused_by", value="heat=high",
                          confidence=confidence, source="abduction")

    def test_corrobation_raises_confidence_capped_at_0_9(self):
        h = self._hyp(0.85)
        out = self.engine.evaluate_hypothesis(h, {"met": True, "detail": "observed boiling"})
        self.assertEqual(out.status, "corroborated")
        self.assertEqual(out.confidence, 0.9)
        self.assertIn("observed boiling", out.evidence_for)

    def test_falsification_marks_and_emits_event(self):
        h = self._hyp(0.5)
        out = self.engine.evaluate_hypothesis(h, {"met": False, "detail": "stayed warm"})
        self.assertEqual(out.status, "falsified")
        self.assertEqual(out.confidence, 0.5)
        self.assertIn("hypothesis_falsified", [k for k, _ in self.engine.events])
        self.assertIn("stayed warm", out.evidence_against)


class TestReasonEndToEnd(unittest.TestCase):
    def test_contradictions_surface_in_reason_result(self):
        ltm = _world()
        ltm.add("paris", "capital_of", "germany", "teacher")
        engine = ReasoningEngine(Inference(ltm), RuleEngine())
        # A derived claim for the same functional relation, other value.
        from ab.reasoning.engine import Derivation
        deriv = Derivation("paris", "capital_of", "france", None, 0.9)
        out = engine.detect_contradictions([deriv], _facts_fn(ltm))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].policy, "disputed+research")


if __name__ == "__main__":
    unittest.main()
