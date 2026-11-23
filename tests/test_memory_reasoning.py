import os, tempfile, unittest
from ab.core.memory import LongTermMemory, AddResult
from ab.reasoning.inference import Inference
from ab.reasoning.rules import Rule, RuleEngine
from ab.reasoning.decision import DecisionMaker, Option


class TestMemory(unittest.TestCase):
    def test_teacher_fact_is_verified_web_fact_is_not(self):
        m = LongTermMemory()
        _, f1 = m.add("Water", "is_a", "liquid", "teacher")
        _, f2 = m.add("Mars", "is_a", "planet", "web:a.org")
        self.assertEqual(f1.status, "verified"); self.assertEqual(f2.status, "unverified")

    def test_two_independent_sources_verify_same_source_does_not(self):
        m = LongTermMemory()
        m.add("mars", "is_a", "planet", "web:a.org")
        self.assertEqual(m.add("mars", "is_a", "planet", "web:a.org")[0], AddResult.DUPLICATE)
        r, f = m.add("mars", "is_a", "planet", "web:b.org")
        self.assertEqual(r, AddResult.CORROBORATED); self.assertEqual(f.status, "verified")
        self.assertGreater(f.confidence, 0.45)

    def test_conflict_never_overwrites(self):
        m = LongTermMemory()
        m.add("water", "boils_at", "100C", "teacher")
        r, new = m.add("water", "boils_at", "90C", "web:x.com")
        self.assertEqual(r, AddResult.CONFLICT)
        self.assertEqual(m.query("water", "boils_at", "100C")[0].status, "verified")   # still trusted
        self.assertEqual(new.status, "disputed")
        self.assertEqual(len(m.all_facts()), 2)                                          # nothing deleted

    def test_resolve_retracts_but_keeps_history(self):
        m = LongTermMemory()
        m.add("x", "boils_at", "1C", "web:a.com"); m.add("x", "boils_at", "2C", "web:b.com")
        m.resolve("x", "boils_at", "2C")
        self.assertEqual(m.ask("x", "boils_at").best[0], "2c")
        self.assertEqual(len(m.all_facts()), 2)

    def test_unknown_is_honest(self):
        a = LongTermMemory().ask("zorblax", "is_a")
        self.assertFalse(a.known); self.assertIn("don't know", a.message)

    def test_persistence_roundtrip(self):
        m = LongTermMemory(); m.add("a", "is_a", "b", "teacher")
        p = os.path.join(tempfile.mkdtemp(), "m.json"); m.save(p)
        m2 = LongTermMemory(); m2.load(p)
        self.assertEqual(m2.ask("a", "is_a").best[0], "b")


class TestInference(unittest.TestCase):
    def setUp(self):
        self.m = LongTermMemory()
        for s, v in [("cat", "mammal"), ("mammal", "animal"), ("animal", "organism")]:
            self.m.add(s, "is_a", v, "teacher")
        self.inf = Inference(self.m)

    def test_transitive_proof_chain(self):
        p = self.inf.prove("cat", "is_a", "organism")
        self.assertIsNotNone(p); self.assertEqual(len(p.steps), 3)

    def test_no_proof_for_unknown_or_unverified(self):
        self.assertIsNone(self.inf.prove("cat", "is_a", "robot"))
        self.m.add("dog", "is_a", "mammal", "web:a.org")                  # unverified
        self.assertIsNone(self.inf.prove("dog", "is_a", "animal"))


class TestRules(unittest.TestCase):
    def test_priority_specificity_and_missing_keys(self):
        e = RuleEngine()
        e.add(Rule("a", [{"key": "x", "op": ">", "value": 1}], [], priority=0.5))
        e.add(Rule("b", [{"key": "x", "op": ">", "value": 1}, {"key": "y", "op": "==", "value": 2}], [], priority=0.5))
        self.assertEqual([r.id for r in e.match({"x": 2, "y": 2})], ["b", "a"])
        self.assertEqual([r.id for r in e.match({"x": 2})], ["a"])
        self.assertEqual(e.match({}), [])

    def test_unreliable_rules_lose_priority_then_retire(self):
        e = RuleEngine()
        e.add(Rule("bad", [], [], priority=0.5))
        for _ in range(6):
            e.record_outcome("bad", False)
        self.assertEqual(e.get("bad").status, "retired")
        self.assertEqual(e.match({}), [])

    def test_pending_rules_do_not_fire_until_approved(self):
        e = RuleEngine(); e.add(Rule("p", [], [], status="pending"))
        self.assertEqual(e.match({}), [])
        e.approve("p"); self.assertEqual(len(e.match({})), 1)


class TestDecision(unittest.TestCase):
    def test_choice_tie_and_insufficient_evidence(self):
        d = DecisionMaker({"gain": 1, "risk": 1}, min_utility=0.4)
        out = d.decide([Option("a", {"gain": 0.9, "risk": 0.8}), Option("b", {"gain": 0.2, "risk": 0.1})])
        self.assertEqual(out.chosen.name, "a")
        self.assertIsNone(d.decide([Option("c", {"gain": 0.1, "risk": 0.1})]).chosen)
        tie = d.decide([Option("p", {"gain": 0.8, "risk": 0.8}), Option("q", {"gain": 0.8, "risk": 0.81})])
        self.assertTrue(tie.tie)
        self.assertIsNone(d.decide([]).chosen)


if __name__ == "__main__":
    unittest.main()
