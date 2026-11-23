"""Phase 4: mixed-relation deduction, noisy-or, induction, abduction,
analogy, causal claims, contradictions, simulation (behavioural tests)."""
import unittest

from ab.core.memory import Fact, LongTermMemory
from ab.reasoning.causal import CausalModel
from ab.reasoning.engine import Derivation, ReasoningEngine
from ab.reasoning.hypothesis import Hypothesis, make_hypothesis_id
from ab.reasoning.inference import Inference, Proof
from ab.reasoning.rules import RuleEngine


def _fact(subject, relation, value, trust=1.0, status="verified"):
    """A hand-built verified Fact with an exact confidence."""
    return Fact(subject, relation, value, {"src": trust}, status=status)


def _facts_fn(facts):
    def fn(subject, relation=None):
        # subject=None returns all facts, mirroring ltm.query(None).
        return [f for f in facts
                if (subject is None or f.subject == subject)
                and (relation is None or f.relation == relation)]
    return fn


def _teach():
    """LTM: cat is_a mammal (1.0), mammal located_in house (0.8)."""
    m = LongTermMemory()
    m.add("cat", "is_a", "mammal", "teacher")
    # A verified 0.8 fact: teacher (1.0) would give 1.0, so teach the
    # second link from two weaker sources instead: tool 0.8 + web 0.45
    # gives 0.89, not 0.8 — so this helper is only for the LTM-backed
    # closure test (confidences 1.0); the exact-0.76 test below uses
    # hand-built Facts through facts_fn.
    m.add("mammal", "located_in", "house", "teacher")
    return m


class TestClosureMulti(unittest.TestCase):
    def setUp(self):
        self.ltm = _teach()
        self.inf = Inference(self.ltm)

    def test_mixed_relation_chain(self):
        hits = self.inf.closure_multi("cat")
        by_value = {v: (p, path) for v, p, path in hits}
        self.assertIn("house", by_value)
        proof, path = by_value["house"]
        self.assertEqual(path, ("is_a", "located_in"))
        # 2 steps, weakest premise 1.0 -> 1.0 * 0.95 = 0.95
        self.assertAlmostEqual(proof.confidence, 0.95, places=9)
        self.assertEqual([f.subject for f in proof.steps], ["cat", "mammal"])

    def test_single_relation_chain(self):
        hits = self.inf.closure_multi("cat", relations=["is_a"])
        self.assertEqual([v for v, _, _ in hits], ["mammal"])

    def test_unknown_subject_is_empty(self):
        self.assertEqual(self.inf.closure_multi("atlantis"), [])

    def test_proof_confidence(self):
        one = [Fact("a", "is_a", "b", {"t": 0.8}, status="verified")]
        self.assertAlmostEqual(Inference.proof_confidence(one), 0.8, places=9)
        two = one + [Fact("b", "is_a", "c", {"t": 1.0}, status="verified")]
        self.assertAlmostEqual(Inference.proof_confidence(two), 0.8 * 0.95, places=9)
        three = two + [Fact("c", "is_a", "d", {"t": 0.9}, status="verified")]
        self.assertAlmostEqual(Inference.proof_confidence(three), 0.8 * 0.95 ** 2, places=9)
        self.assertEqual(Inference.proof_confidence([]), 0.0)


class TestCombineProofs(unittest.TestCase):
    def test_noisy_or_two_independent_proofs(self):
        # Steps at full confidence (teacher facts, 1.0) with proof
        # confidences 0.6 each: noisy-or = 1 - 0.4*0.4 = 0.84, and the
        # weakest-premise cap (1.0) does not bind.
        p1 = Proof([_fact("cat", "is_a", "mammal", 1.0)], 0.6)
        p2 = Proof([_fact("cat", "is_a", "pet", 1.0)], 0.6)
        self.assertAlmostEqual(Inference.combine_proofs([p1, p2]), 0.84, places=9)

    def test_capped_at_weakest_premise(self):
        # Noisy-or of 0.9 and 0.9 is 0.99, but the weakest premise step
        # is 0.5, so the combination is capped at 0.5.
        p1 = Proof([_fact("a", "is_a", "b", 0.5)], 0.9)
        p2 = Proof([_fact("a", "is_a", "c", 0.5)], 0.9)
        self.assertAlmostEqual(Inference.combine_proofs([p1, p2]), 0.5, places=9)

    def test_empty_and_single(self):
        self.assertEqual(Inference.combine_proofs([]), 0.0)
        p = Proof([_fact("a", "is_a", "b", 1.0)], 0.7)
        self.assertAlmostEqual(Inference.combine_proofs([p]), 0.7, places=9)


class TestDeduceExactConfidence(unittest.TestCase):
    def test_two_step_mixed_chain_is_0_76(self):
        facts = [_fact("cat", "is_a", "mammal", 1.0),
                 _fact("mammal", "located_in", "house", 0.8)]
        engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())
        derivs = engine.deduce("cat", _facts_fn(facts))
        hit = [d for d in derivs
               if d.relation == "located_in" and d.value == "house"]
        self.assertEqual(len(hit), 1)
        # min premise 0.8, one extra step costs 5%: 0.8 * 0.95 = 0.76
        self.assertEqual(hit[0].confidence, 0.76)
        self.assertEqual(hit[0].status, "derived")
        self.assertEqual(hit[0].source, "deduction")

    def test_unverified_facts_never_derive(self):
        facts = [_fact("cat", "is_a", "mammal", 1.0),
                 _fact("mammal", "located_in", "house", 0.8, status="unverified")]
        engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())
        derivs = engine.deduce("cat", _facts_fn(facts))
        self.assertEqual([d.value for d in derivs], ["mammal"])


class TestInduction(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())

    def _episodes(self, n=3):
        return [{"action": "boil", "situation": {"liquid": "water", "heat": "high"},
                 "outcome": "boils"} for _ in range(n)]

    def test_proposes_pending_rule_after_three(self):
        rules = self.engine.induce(self._episodes(3))
        self.assertEqual(len(rules), 1)
        r = rules[0]
        self.assertEqual(r["action"], "boil")
        self.assertEqual(r["outcome"], "boils")
        self.assertEqual(r["situation"], {"liquid": "water", "heat": "high"})
        self.assertEqual(r["successes"], 3)
        self.assertEqual(r["status"], "pending")
        self.assertEqual(len(r["evidence"]), 3)

    def test_never_auto_approves(self):
        # Even with overwhelming repetition, induction only proposes.
        rules = self.engine.induce(self._episodes(10))
        self.assertTrue(all(r["status"] == "pending" for r in rules))

    def test_two_episodes_propose_nothing(self):
        self.assertEqual(self.engine.induce(self._episodes(2)), [])

    def test_deterministic_id(self):
        r1 = self.engine.induce(self._episodes(3))[0]
        r2 = self.engine.induce(self._episodes(3))[0]
        self.assertEqual(r1["id"], r2["id"])
        self.assertTrue(r1["id"].startswith("induce_"))


class TestAbduction(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())
        self.episodes = (
            [{"action": "heat_water", "situation": {"heat": "high"}, "outcome": "boiling"}
             for _ in range(3)]
            + [{"action": "heat_water", "situation": {"heat": "low"}, "outcome": "warm"}
               for _ in range(2)]
        )
        self.surprise = {"action": "heat_water", "expected": "warm", "actual": "boiling"}

    def test_ranks_best_cause_first_with_test(self):
        hyps = self.engine.abduce(self.surprise, self.episodes, None)
        self.assertTrue(hyps)
        # "heat=high" produced boiling 3/3 times; plain "heat_water"
        # only 3/5. The specific cause must rank first.
        self.assertEqual(hyps[0].value, "heat=high")
        self.assertEqual(hyps[0].subject, "heat_water")
        self.assertEqual(hyps[0].relation, "caused_by")
        self.assertEqual(hyps[0].status, "open")
        self.assertEqual(hyps[0].source, "abduction")
        self.assertIsInstance(hyps[0].test, str)
        self.assertIn("boiling", hyps[0].test)
        self.assertTrue(all(hyps[i].confidence >= hyps[i + 1].confidence
                            for i in range(len(hyps) - 1)))

    def test_no_actual_means_no_hypothesis(self):
        self.assertEqual(self.engine.abduce({"action": "x", "expected": "a"}, self.episodes, None), [])
        self.assertEqual(self.engine.abduce(self.surprise, [], None), [])

    def test_branch_cap_drops_lowest_with_event(self):
        engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine(), max_branches=1)
        hyps = engine.abduce(self.surprise, self.episodes, None)
        self.assertEqual(len(hyps), 1)
        kinds = [k for k, _ in engine.events]
        self.assertIn("hypothesis_dropped", kinds)


class TestAnalogy(unittest.TestCase):
    def setUp(self):
        self.facts = [_fact("cat", "is_a", "mammal", 1.0),
                      _fact("dog", "is_a", "mammal", 1.0),
                      _fact("dog", "located_in", "house", 1.0)]
        self.engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())

    def test_analogy_capped_at_0_4(self):
        h = self.engine.analogize("cat", "located_in", _facts_fn(self.facts))
        self.assertIsNotNone(h)
        self.assertEqual(h.value, "house")
        self.assertEqual(h.source, "analogy")
        self.assertEqual(h.status, "open")
        self.assertLessEqual(h.confidence, 0.4)
        self.assertIn("dog", h.test)

    def test_no_sibling_pattern_returns_none(self):
        facts = [_fact("cat", "is_a", "mammal", 1.0)]
        self.assertIsNone(self.engine.analogize("cat", "located_in", _facts_fn(facts)))

    def test_never_verified_alone(self):
        h = self.engine.analogize("cat", "located_in", _facts_fn(self.facts))
        self.assertNotEqual(h.status, "verified")


class TestCausal(unittest.TestCase):
    def test_claim_only_after_three_met_zero_counterexamples(self):
        c = CausalModel()
        self.assertIsNone(c.causes("boil", "water_boils"))
        c.record("boil", "water_boils", True)
        c.record("boil", "water_boils", True)
        self.assertIsNone(c.causes("boil", "water_boils"))
        c.record("boil", "water_boils", True)
        claim = c.causes("boil", "water_boils")
        self.assertIsNotNone(claim)
        self.assertEqual(claim["confidence"], 1.0)
        self.assertEqual(claim["observations"], 3)
        self.assertEqual(claim["counterexamples"], 0)
        self.assertEqual(c.success_rate("boil", "water_boils"), 1.0)
        self.assertEqual(len(c.claims()), 1)

    def test_counterexample_blocks_claim(self):
        c = CausalModel()
        for _ in range(3):
            c.record("boil", "water_boils", True)
        c.record("boil", "water_boils", False)
        self.assertIsNone(c.causes("boil", "water_boils"))
        self.assertAlmostEqual(c.success_rate("boil", "water_boils"), 0.75)
        self.assertEqual(c.claims(), [])

    def test_unknown_rate_is_none(self):
        self.assertIsNone(CausalModel().success_rate("nope", "nothing"))


class TestHypothesisIds(unittest.TestCase):
    def test_deterministic_and_prefixed(self):
        h1 = make_hypothesis_id("cat", "located_in", "house")
        h2 = make_hypothesis_id("cat", "located_in", "house")
        self.assertEqual(h1, h2)
        self.assertTrue(h1.startswith("hyp_"))
        self.assertNotEqual(h1, make_hypothesis_id("dog", "located_in", "house"))
        h = Hypothesis(id=h1, subject="cat", relation="located_in", value="house")
        self.assertEqual(h.status, "open")
        self.assertIsNone(h.test)


class TestContradictions(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())

    def _deriv(self, subject, relation, value):
        return Derivation(subject, relation, value, None, 0.9)

    def test_derived_vs_verified_functional_conflict(self):
        facts = [_fact("paris", "capital_of", "germany", 1.0)]
        d = self._deriv("paris", "capital_of", "france")
        out = self.engine.detect_contradictions([d], _facts_fn(facts))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].policy, "disputed+research")
        self.assertEqual(out[0].claim_a, ("paris", "capital_of", "france"))
        self.assertEqual(out[0].claim_b, ("paris", "capital_of", "germany"))
        self.assertEqual(d.status, "disputed")

    def test_two_derivations_functional_conflict(self):
        d1 = self._deriv("paris", "capital_of", "france")
        d2 = self._deriv("paris", "capital_of", "italy")
        out = self.engine.detect_contradictions([d1, d2], _facts_fn([]))
        self.assertEqual(len(out), 1)
        self.assertEqual(d1.status, "disputed")
        self.assertEqual(d2.status, "disputed")
        self.assertEqual(out[0].policy, "disputed+research")

    def test_non_functional_difference_is_not_a_contradiction(self):
        facts = [_fact("cat", "located_in", "garden", 1.0)]
        d = self._deriv("cat", "located_in", "house")
        self.assertEqual(self.engine.detect_contradictions([d], _facts_fn(facts)), [])
        self.assertEqual(d.status, "derived")

    def test_agreeing_derivation_is_not_a_contradiction(self):
        facts = [_fact("paris", "capital_of", "france", 1.0)]
        d = self._deriv("paris", "capital_of", "france")
        self.assertEqual(self.engine.detect_contradictions([d], _facts_fn(facts)), [])


class TestSimulate(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine(Inference(LongTermMemory()), RuleEngine())

    def test_predicts_plan_outcome_without_executing(self):
        effects = {"boil": {"water": "hot"}, "pour": {"cup": "full"}}
        executed = []

        def effects_fn(action, state):
            executed.append(action)   # the simulator calls it; nothing real happens
            return dict(effects.get(action, {}))

        start = {"water": "cold", "goal": {"water": "hot", "cup": "full"}}
        final, met, assumptions = self.engine.simulate(["boil", "pour"], start, effects_fn)
        self.assertEqual(final, {"water": "hot", "cup": "full"})
        self.assertTrue(met)
        self.assertEqual(len(assumptions), 2)
        # The input state is untouched: simulation never executes.
        self.assertEqual(start, {"water": "cold", "goal": {"water": "hot", "cup": "full"}})
        self.assertEqual(executed, ["boil", "pour"])

    def test_unknown_effect_means_goal_not_met(self):
        def effects_fn(action, state):
            return None
        _, met, assumptions = self.engine.simulate(["teleport"], {"x": 1}, effects_fn)
        self.assertFalse(met)
        self.assertTrue(any("teleport" in a for a in assumptions))

    def test_no_goal_means_all_known(self):
        def effects_fn(action, state):
            return {"done": True}
        _, met, _ = self.engine.simulate(["act"], {}, effects_fn)
        self.assertTrue(met)


if __name__ == "__main__":
    unittest.main()
