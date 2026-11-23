"""Phase 1, Workstreams 3 and 4: working-memory integration and
inference in the cycle.

Acceptance:
- WM contents are READ during the cycle (not merely admitted), and
  changing the relevant WM contents changes a downstream cognitive
  result (goal completion), not just a return value.
- Verified-only inference derives a conclusion in-cycle, with a proof
  chain, made available to planning/decisions via a state entry — and
  the derived conclusion is NOT promoted to a verified LTM fact.
"""
import unittest

from ab.brain import Brain


def _cat_brain():
    b = Brain()
    b.teach_fact("cat", "is_a", "mammal")
    b.teach_fact("mammal", "is_a", "animal")
    return b


class TestInferenceInCycle(unittest.TestCase):
    def test_derivation_appears_in_cycle_with_proof(self):
        b = _cat_brain()
        b.perceive_text("cat")
        b.step(None)
        self.assertEqual(b.world.get("inferred:cat|is_a"), "animal")
        self.assertTrue(any(d["value"] == "animal" and "cat is_a mammal" in d["proof"]
                            for d in b.last_inferences))
        self.assertTrue(any(t.kind == "inferred" and t.content == "cat is_a animal"
                            for t in b.wm.contents()))

    def test_derived_conclusion_is_not_promoted_to_verified_fact(self):
        b = _cat_brain()
        b.perceive_text("cat")
        b.step(None)
        self.assertEqual(b.ltm.query("cat", "is_a", "animal"), [])
        self.assertIn("yes", b.verify("cat", "is_a", "animal"))  # provable, not stored

    def test_no_derivation_without_a_known_subject(self):
        b = _cat_brain()
        b.perceive_text("zorblax")
        b.step(None)
        self.assertEqual(b.last_inferences, [])
        self.assertNotIn("inferred:zorblax|is_a", b.world)

    def test_unverified_facts_never_drive_derivations(self):
        b = Brain()
        b.ltm.add("dog", "is_a", "mammal", "web:a.org")  # unverified
        b.teach_fact("mammal", "is_a", "animal")
        b.perceive_text("dog")
        b.step(None)
        self.assertEqual(b.last_inferences, [])


class TestWorkingMemoryChangesDownstreamResult(unittest.TestCase):
    def test_wm_contents_decide_goal_completion(self):
        # Same goal, same taught facts. The only difference is what the
        # brain perceived into working memory.
        # NOTE: the goal description must not itself mention "cat" -- the
        # goal thought is also admitted to WM, so a description containing
        # the subject word would let the derivation fire in both brains.
        cat = _cat_brain()
        cat.perceive_text("cat")
        cat.add_goal("know what the animal is", {"inferred:cat|is_a": "animal"})
        cat.run(None, 30)
        self.assertEqual(cat.goals.all()[0].status, "done")

        dog = _cat_brain()
        dog.perceive_text("dog")
        dog.add_goal("know what the animal is", {"inferred:cat|is_a": "animal"})
        dog.run(None, 30)
        self.assertNotEqual(dog.goals.all()[0].status, "done")

    def test_reason_event_is_recorded_each_cycle(self):
        b = _cat_brain()
        b.perceive_text("cat")
        b.step(None)
        self.assertTrue(any(e["kind"] == "reason" for e in b.events))


if __name__ == "__main__":
    unittest.main()
