import os, tempfile, unittest
from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.body.ocr import StubOcr, OcrUnavailable
from ab.body.research import Researcher
from ab.body.providers import MemoryKnowledgeProvider
from ab.core.learning import Episode, Learner


class TestEndToEnd(unittest.TestCase):
    def test_achieves_goal_and_learns_pick_lock_is_unreliable(self):
        env, brain = RoomWorld(), Brain(room_world_actions(), granted={"compute", "environment"})
        brain.add_goal("get the treasure", {"has_treasure": True})
        brain.run(env, 60)
        self.assertTrue(env.s["has_treasure"])
        self.assertEqual(brain.goals.all()[0].status, "done")
        self.assertTrue(brain.last_evaluation.success)
        self.assertTrue(any("surprise" in e["detail"] for e in brain.events))   # surprised by pick_lock
        self.assertEqual(brain.planner.failure_rate("pick_lock"), 1.0)    # and remembered
        self.assertEqual(brain.planner.failure_rate("unlock"), 0.0)

    def test_second_attempt_avoids_known_bad_action(self):
        brain = Brain(room_world_actions(), granted={"compute", "environment"})
        brain.planner.record("pick_lock", False)                           # learned earlier
        env = RoomWorld(); brain.add_goal("treasure", {"has_treasure": True}); brain.run(env, 60)
        self.assertTrue(env.s["has_treasure"])
        self.assertFalse(any("pick_lock" in e["detail"] and e["kind"] == "act" for e in brain.events))

    def test_working_memory_never_overflows_during_run(self):
        env, brain = RoomWorld(), Brain(room_world_actions(), granted={"compute", "environment"}, wm_capacity=3)
        brain.add_goal("treasure", {"has_treasure": True})
        while brain.cycle < 60 and (brain.goals.active() or brain.goals.pending()):
            brain.step(env); self.assertLessEqual(len(brain.wm), 3)

    def test_impossible_goal_fails_cleanly_not_forever(self):
        env, brain = RoomWorld(), Brain(room_world_actions(), granted={"compute", "environment"})
        brain.add_goal("fly to the moon", {"on_moon": True})
        brain.run(env, 40)
        g = brain.goals.all()[0]
        self.assertEqual(g.status, "failed"); self.assertLess(brain.cycle, 40)

    def test_higher_priority_goal_preempts_but_small_difference_does_not(self):
        env = RoomWorld(); b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("low", {"has_key": True}, priority=0.4); b.step(env); b.step(env)
        b.add_goal("slightly higher", {"has_treasure": True}, priority=0.5); b.step(env)
        self.assertEqual(b.goals.active().description, "low")
        b.add_goal("urgent", {"door_open": True}, priority=0.9); b.step(env)
        self.assertEqual(b.goals.active().description, "urgent")

    def test_runs_without_environment_for_tool_goals(self):
        b = Brain([], granted={"compute"})
        self.assertEqual(b.tools.call("calc", {"expression": "6*7"}, b.granted).output, 42)


class TestKnowledge(unittest.TestCase):
    def test_teach_ask_infer_and_never_hallucinate(self):
        b = Brain()
        b.teach_fact("cat", "is_a", "mammal"); b.teach_fact("mammal", "is_a", "animal")
        self.assertEqual(b.ask("cat", "is_a").best[0], "mammal")
        self.assertEqual(b.ask("cat", "is_a").best[2], "verified")
        self.assertIn("yes", b.verify("cat", "is_a", "animal"))
        self.assertFalse(b.ask("zorblax", "capital_of").known)
        self.assertTrue(b.verify("cat", "is_a", "robot").startswith("unknown"))

    def test_taught_rule_and_pattern_are_used(self):
        b = Brain(); b.teach_pattern("likes", r"^(?P<s>.+?) likes (?P<v>.+?)\.?$")
        self.assertEqual(b.extractor.extract("Tom likes tea.")[0].relation, "likes")
        b.teach_rule({"id": "r1", "when": [{"key": "x", "op": ">", "value": 1}], "then": []})
        self.assertEqual(b.rules.get("r1").source, "teacher")

    def test_memory_survives_save_and_load(self):
        b = Brain(); b.teach_fact("a", "is_a", "b")
        d = tempfile.mkdtemp(); b.save(d)
        b2 = Brain(); b2.load(d)
        self.assertEqual(b2.ask("a", "is_a").best[0], "b")
        self.assertIsNotNone(b2.rules.get("select_goal"))


class TestResearchGoal(unittest.TestCase):
    # Phase 11 §4.1 (W1): the second source words the fact differently
    # ("capital city") — the identical sentence from both domains is a
    # mirror under the mirror rule and no longer corroborates. Same
    # claim, different sentence; the goal outcome is unchanged.
    PAGES = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
             "https://b.org/p": "<p>Paris is the capital city of France.</p>"}

    def _brain(self):
        # Phase 11 integration (W5): the research tool is isolated,
        # so the child rebuilds its provider from the provider spec
        # — a closure transport would not survive the fork. The
        # memory provider's spec carries the pages.
        return Brain(researcher=Researcher(MemoryKnowledgeProvider(self.PAGES), None),
                     granted={"compute", "network"})

    def test_two_independent_sources_satisfy_research_goal(self):
        b = self._brain()
        b.add_research_goal("paris", "capital_of", ["https://a.org/p", "https://b.org/p"])
        b.run(None, 30)
        self.assertEqual(b.goals.all()[0].status, "done")
        self.assertEqual(b.ask("paris", "capital_of").best[2], "verified")

    def test_single_source_is_not_enough_goal_fails_honestly(self):
        b = self._brain()
        b.add_research_goal("paris", "capital_of", ["https://a.org/p"])
        b.run(None, 30)
        self.assertEqual(b.goals.all()[0].status, "failed")
        # Phase 3 §3.2 (mandated): a single-domain claim never enters
        # LTM at all, so there is no fact to ask about — but the
        # evidence is not lost: it is recorded in state as untrusted.
        self.assertIsNone(b.ask("paris", "capital_of").best)
        self.assertEqual(b.ltm.query("paris", "capital_of"), [])
        entry = b.state.get("untrusted:paris|capital_of")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.status, "untrusted")
        self.assertEqual(entry.value, "france")

    def test_network_permission_is_enforced(self):
        b = self._brain(); b.granted = {"compute"}
        b.add_research_goal("paris", "capital_of", ["https://a.org/p", "https://b.org/p"])
        b.run(None, 30)
        self.assertEqual(b.goals.all()[0].status, "failed")
        self.assertFalse(b.ltm.query("paris", "capital_of"))


class TestPerceptionAndLearning(unittest.TestCase):
    def test_ocr_text_becomes_unverified_knowledge_and_low_confidence_is_rejected(self):
        b = Brain(ocr=StubOcr({"good.png": "Mars is a planet."}, confidence=0.9))
        out = b.perceive_image("good.png")
        self.assertEqual(out["stored"][0][2], "unverified")
        bad = Brain(ocr=StubOcr({"blur.png": "Mars is a planet."}, confidence=0.3))
        out = bad.perceive_image("blur.png")
        self.assertEqual(len(out["rejected"]), 1); self.assertFalse(bad.ltm.query("mars"))

    def test_no_ocr_engine_raises_clearly(self):
        with self.assertRaises(OcrUnavailable):
            Brain().perceive_image("x.png")

    def test_repeated_failure_becomes_pending_rule_until_approved(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for i in range(3):
            b.learner.record(Episode(i, "pick_lock", {"at": "lab"}, False))
        prop = b.learner.propose_rules()
        self.assertEqual(len(prop), 1); self.assertEqual(prop[0].status, "pending")
        b.world = {"at": "lab"}; self.assertEqual(b._banned(), set())         # pending: no effect
        b.rules.approve(prop[0].id);   self.assertEqual(b._banned(), {"pick_lock"})


if __name__ == "__main__":
    unittest.main()
