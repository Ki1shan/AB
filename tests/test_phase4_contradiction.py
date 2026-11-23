"""Phase 4: contradiction policy — disputed+research (behavioural tests).

A derived claim conflicting with a verified fact, or two derivations
conflicting on a functional relation, is never silently resolved by
picking a side: both claims are marked disputed, a contradiction event
is logged, the resolution policy is recorded, and a research goal is
created for the missing evidence. Self-correction on a failed
prediction retracts (or flags) the belief the prediction depended on.
"""
import unittest

from ab.brain import Brain
from ab.core.control import CognitiveControl


def _water_brain():
    """water is_a liquid; liquid boils_at 100c (verified)."""
    b = Brain()
    b.teach_fact("water", "is_a", "liquid")
    b.teach_fact("liquid", "boils_at", "100c")
    return b


class TestDerivedVsVerifiedFact(unittest.TestCase):
    def test_derived_claim_conflicting_with_verified_fact(self):
        b = _water_brain()
        # Direct verified fact contradicting the inherited derivation
        # (water boils_at 100c via liquid).
        b.teach_fact("water", "boils_at", "90c")
        b.perceive_text("water")
        b.step()
        # The derived claim is disputed, never silently picked: no
        # inferred:* entry carries it into planning.
        self.assertIsNone(b.world.get("inferred:water|boils_at"))
        self.assertTrue(any(d["subject"] == "water" and d["relation"] == "boils_at"
                            and d["value"] == "100c" and d["status"] == "disputed"
                            for d in b.last_inferences))
        # Contradiction event + both values marked disputed in state.
        self.assertTrue(any(e["kind"] == "contradiction" for e in b.events))
        entry = b.state.get("disputed:water|boils_at")
        self.assertIsNotNone(entry)
        self.assertEqual(sorted(entry.value), ["100c", "90c"])
        self.assertEqual(entry.status, "disputed")
        # A research goal is created for the missing evidence.
        self.assertTrue(any("known:water|boils_at" in g.target
                            for g in b.goals.all()))
        # Neither side silently picked: ask returns both, disputed.
        ans = b.ask("water", "boils_at")
        self.assertTrue(ans.known)
        self.assertEqual(sorted(v for v, _, _, _ in ans.answers),
                         ["100c", "90c"])
        self.assertIn("disagree", ans.message)

    def test_two_derivations_conflicting_both_disputed(self):
        b = Brain()
        b.teach_fact("water", "is_a", "liquid")
        b.teach_fact("water", "is_a", "fluid")
        b.teach_fact("liquid", "boils_at", "100c")
        b.teach_fact("fluid", "boils_at", "80c")
        b.perceive_text("water")
        b.step()
        disputed = [d for d in b.last_inferences
                    if d["subject"] == "water" and d["relation"] == "boils_at"]
        self.assertEqual(sorted(d["value"] for d in disputed), ["100c", "80c"])
        self.assertTrue(all(d["status"] == "disputed" for d in disputed))
        self.assertIsNone(b.world.get("inferred:water|boils_at"))
        self.assertTrue(any(e["kind"] == "contradiction" for e in b.events))
        entry = b.state.get("disputed:water|boils_at")
        self.assertIsNotNone(entry)
        self.assertEqual(sorted(entry.value), ["100c", "80c"])
        self.assertTrue(any("known:water|boils_at" in g.target
                            for g in b.goals.all()))

    def test_resolve_conflicts_called_by_the_cycle(self):
        # resolve_conflicts must be reached through the contradiction
        # run itself, not by calling it directly: wrap it and watch.
        calls = []
        orig = CognitiveControl.resolve_conflicts

        def spy(proposals):
            calls.append(list(proposals))
            return orig(proposals)

        b = _water_brain()
        b.teach_fact("water", "boils_at", "90c")
        b.perceive_text("water")
        CognitiveControl.resolve_conflicts = staticmethod(spy)
        try:
            b.step()
        finally:
            CognitiveControl.resolve_conflicts = orig
        self.assertTrue(calls, "resolve_conflicts was never called by the cycle")
        flat = [p for call in calls for p in call]
        self.assertTrue(any(p[0].startswith("water|boils_at=") for p in flat))
        # Proposals carry a priority and the functional slot as the
        # shared resource (one research goal per disputed slot).
        self.assertTrue(all(isinstance(p[2], frozenset) and p[2] == frozenset({"water|boils_at"})
                            for p in flat))


class TestSelfCorrection(unittest.TestCase):
    def test_failed_prediction_retracts_belief_with_revision_event(self):
        b = Brain()
        b.teach_fact("paris", "capital_of", "lyon")
        fact = b.ltm.query("paris", "capital_of", "lyon")[0]
        self.assertEqual(fact.status, "verified")
        # A stored prediction that the next cycle will check — and fail.
        b._set_state("predicted:go:paris|capital_of", "lyon", source="planner",
                     confidence=1.0, provenance="plan:go", entry_type="inferred")
        b.step()
        self.assertEqual(b.ltm.query("paris", "capital_of", "lyon"), [])
        retracted = [f for f in b.ltm.all_facts()
                     if f.id == fact.id]
        self.assertEqual(retracted[0].status, "retracted")
        self.assertTrue(any(e["kind"] == "revision" for e in b.events))
        self.assertTrue(any(e["kind"] == "prediction_checked"
                            and "violated" in e["detail"] for e in b.events))
        self.assertEqual(b.unchecked_predictions(), [])
        # The outcome is recorded in the causal model.
        self.assertEqual(b.causal.success_rate("go", "paris|capital_of"), 0.0)

    def test_contradiction_self_corrects_without_picking_a_side(self):
        b = _water_brain()
        b.teach_fact("water", "boils_at", "90c")
        b.perceive_text("water")
        b.step()
        # The verified fact is NOT retracted (that would pick the
        # derived side) and the derived claim is NOT promoted to LTM.
        held = b.ltm.query("water", "boils_at", "90c")
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0].status, "verified")
        self.assertEqual(b.ltm.query("water", "boils_at", "100c"), [])
        self.assertTrue(any(e["kind"] == "revision" for e in b.events))


class TestDecisionDefersOnDispute(unittest.TestCase):
    def test_decision_records_policy_and_does_not_act_on_disputed_claim(self):
        b = Brain()
        b.add_goal("learn about water", {"known:water|boils_at": True})
        b.step()  # select the goal: a decision is now in progress
        self.assertIsNotNone(b.last_decision)
        b.teach_fact("water", "is_a", "liquid")
        b.teach_fact("liquid", "boils_at", "100c")
        b.teach_fact("water", "boils_at", "90c")
        b.perceive_text("water")
        b.step()  # the contradiction surfaces in-cycle
        self.assertEqual(b.last_decision.resolution_policy, "disputed+research")
        ans = b.ask("water", "boils_at")
        self.assertEqual(sorted(v for v, _, _, _ in ans.answers),
                         ["100c", "90c"])
        self.assertIn("disagree", ans.message)
        # The disputed claim never becomes actionable state, so no
        # plan step could be chosen because of it.
        self.assertIsNone(b.world.get("inferred:water|boils_at"))
        self.assertFalse(any(e["kind"] == "act" and "100c" in e["detail"]
                             for e in b.events))


if __name__ == "__main__":
    unittest.main()
