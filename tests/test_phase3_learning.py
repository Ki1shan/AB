"""Phase 3 §3.1: learning activation — pending rules become activatable
by explicit human approval only, with persistence. No auto-approval."""
import unittest

from ab.core.learning import Episode, Learner, stable_rule_id
from ab.planning.planner import Planner
from ab.reasoning.rules import RuleEngine


def make_learner():
    return Learner(Planner(), RuleEngine())


def record_failures(learner, action="pick_lock", situation=None, n=3, start=0):
    situation = situation if situation is not None else {"at": "lab"}
    for i in range(n):
        learner.record(Episode(start + i, action, dict(situation), False))


class TestPendingRuleActivation(unittest.TestCase):
    def test_three_failures_proposed_then_approve_then_approved_rules(self):
        learner = make_learner()
        record_failures(learner)
        pending = learner.pending_rules()
        self.assertEqual(len(pending), 1)
        p = pending[0]
        self.assertEqual(p["rule_id"], stable_rule_id("pick_lock", {"at": "lab"}))
        self.assertEqual(p["action"], "pick_lock")
        self.assertEqual(p["situation"], {"at": "lab"})
        self.assertEqual(p["failures"], 3)
        self.assertEqual(p["successes"], 0)
        self.assertEqual(p["status"], "pending")
        self.assertEqual(p["evidence"], 3)
        # No auto-approval: still pending in the engine, not active.
        self.assertEqual(learner.rules.get(p["rule_id"]).status, "pending")
        self.assertEqual(learner.approved_rules(), [])
        # Explicit approval activates it.
        self.assertTrue(learner.approve_rule(p["rule_id"]))
        self.assertEqual(learner.rules.get(p["rule_id"]).status, "active")
        approved = learner.approved_rules()
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["rule_id"], p["rule_id"])
        self.assertEqual(approved[0]["action"], "pick_lock")
        self.assertEqual(approved[0]["situation"], {"at": "lab"})
        self.assertEqual(learner.pending_rules(), [])
        # Approving again is not a second approval.
        self.assertFalse(learner.approve_rule(p["rule_id"]))

    def test_reject_not_reproposed_from_same_episodes(self):
        learner = make_learner()
        record_failures(learner)
        pending = learner.pending_rules()
        self.assertEqual(len(pending), 1)
        rid = pending[0]["rule_id"]
        self.assertTrue(learner.reject_rule(rid))
        self.assertEqual(learner.rules.get(rid).status, "retired")
        self.assertEqual(learner.pending_rules(), [])
        self.assertEqual(learner.approved_rules(), [])
        # Three MORE identical failures from the same episodes' situation
        # must not re-propose the rejected pair.
        record_failures(learner, n=3, start=3)
        self.assertEqual(learner.propose_rules(), [])
        self.assertEqual(learner.pending_rules(), [])
        # Rejecting again is not a second rejection.
        self.assertFalse(learner.reject_rule(rid))
        # ... and a rejected rule can never be approved afterwards.
        self.assertFalse(learner.approve_rule(rid))

    def test_approve_reject_unknown_rule_returns_false(self):
        learner = make_learner()
        record_failures(learner)
        self.assertFalse(learner.approve_rule("avoid_nope_00000000"))
        self.assertFalse(learner.reject_rule("avoid_nope_00000000"))

    def test_no_proposal_below_threshold_and_success_blocks_proposal(self):
        learner = make_learner()
        record_failures(learner, n=2)
        self.assertEqual(learner.pending_rules(), [])
        learner.record(Episode(2, "pick_lock", {"at": "lab"}, True))
        record_failures(learner, n=3, start=3)
        # One success in the same situation blocks the proposal entirely.
        self.assertEqual(learner.pending_rules(), [])
        self.assertEqual(learner.propose_rules(), [])

    def test_propose_rules_still_adds_pending_rule_to_engine(self):
        # Backwards compatibility with the Phase 1/2 call pattern.
        learner = make_learner()
        record_failures(learner)
        prop = learner.propose_rules()
        self.assertEqual(len(prop), 1)
        self.assertEqual(prop[0].status, "pending")
        # The propose() alias proposes nothing twice (no duplicate rule).
        self.assertEqual(learner.propose(), [])


class TestLearnerPersistence(unittest.TestCase):
    def test_approve_reject_persistence_round_trip(self):
        learner = make_learner()
        record_failures(learner, "pick_lock", {"at": "lab"})
        record_failures(learner, "unlock", {"at": "lab"}, start=3)
        self.assertEqual(len(learner.pending_rules()), 2)
        pick_id = stable_rule_id("pick_lock", {"at": "lab"})
        unlock_id = stable_rule_id("unlock", {"at": "lab"})
        self.assertTrue(learner.approve_rule(pick_id))
        self.assertTrue(learner.reject_rule(unlock_id))

        data = learner.to_json()
        restored = Learner.from_json(data, Planner(), RuleEngine())
        self.assertEqual(len(restored.episodes), 6)
        approved = restored.approved_rules()
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0]["rule_id"], pick_id)
        self.assertEqual(approved[0]["action"], "pick_lock")
        self.assertEqual(approved[0]["situation"], {"at": "lab"})
        # The rejected pair is not re-proposed even into a fresh engine.
        self.assertEqual(restored.propose_rules(), [])
        self.assertEqual(restored.pending_rules(), [])
        self.assertFalse(restored.approve_rule(unlock_id))
        self.assertFalse(restored.reject_rule(unlock_id))
        # And the approved pair is not re-proposed as pending either.
        self.assertFalse(restored.approve_rule(pick_id))

    def test_old_learner_json_without_decisions_still_loads(self):
        learner = make_learner()
        record_failures(learner)
        data = learner.to_json()
        del data["approved"], data["rejected"]
        restored = Learner.from_json(data, Planner(), RuleEngine())
        self.assertEqual(len(restored.episodes), 3)
        self.assertEqual(len(restored.pending_rules()), 1)

    def test_episode_cap_preserved_at_500(self):
        learner = make_learner()
        for i in range(600):
            learner.record(Episode(i, "a", {"x": 1}, False))
        self.assertEqual(len(learner.episodes), 500)
        data = learner.to_json()
        restored = Learner.from_json(data, Planner(), RuleEngine())
        self.assertEqual(len(restored.episodes), 500)


if __name__ == "__main__":
    unittest.main()
