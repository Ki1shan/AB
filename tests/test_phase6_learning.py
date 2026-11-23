"""Phase 6 learning core (Worker 1): Consolidator, ConceptStore,
CalibrationTracker, Retriever, Prioritizer.

All fixtures use invented names (alpha/beta tools, gate_open effect,
hall room) -- no test-world names appear here or in ab/learning/.
"""
import json
import unittest

from ab.learning import (CalibrationTracker, Concept, ConceptStore,
                         Consolidator, Prioritizer, Retriever)
from ab.learning.concepts import concept_id
from ab.learning.consolidation import candidate_id


def _ev(n):
    return [{"action": "open_alpha", "cycle": i, "ok": True}
            for i in range(n)]


# ------------------------------------------------------------------
# Consolidator
# ------------------------------------------------------------------
class TestConsolidatorPromotionBar(unittest.TestCase):
    def test_below_bar_stays_candidate(self):
        cons = Consolidator()
        cid = cons.add_candidate("pattern", {"kind": "durable_tool", "effect": "gate_open"},
                                 _ev(2), 2, provenance="episodes")
        self.assertFalse(cons.promote(cid))
        self.assertEqual(cons.get(cid)["status"], "candidate")
        self.assertEqual(len(cons.review()), 1)

    def test_below_bar_one_situation_stays_candidate(self):
        cons = Consolidator()
        cid = cons.add_candidate("pattern", {"kind": "durable_tool", "effect": "gate_open"},
                                 _ev(3), 1)
        self.assertFalse(cons.promote(cid))
        self.assertEqual(cons.get(cid)["status"], "candidate")

    def test_at_bar_promotes_and_records_evidence(self):
        cons = Consolidator()
        cid = cons.add_candidate("pattern", {"kind": "durable_tool", "effect": "gate_open"},
                                 _ev(3), 2, provenance="episodes")
        self.assertTrue(cons.promote(cid, cycle=7))
        got = cons.get(cid)
        self.assertEqual(got["status"], "promoted")
        self.assertEqual(got["promoted_at"], 7)
        self.assertEqual(got["promoted_evidence"], 3)
        self.assertEqual(got["promoted_situations"], 2)
        self.assertEqual(cons.review(), [])
        self.assertEqual([c["id"] for c in cons.promoted()], [cid])

    def test_force_promotes_below_bar(self):
        cons = Consolidator()
        cid = cons.add_candidate("concept", {"name": "durable_tool"},
                                 _ev(1), 1)
        self.assertFalse(cons.promote(cid))
        self.assertTrue(cons.promote(cid, force=True, cycle=3))
        self.assertEqual(cons.get(cid)["status"], "promoted")
        self.assertEqual(cons.get(cid)["promoted_at"], 3)

    def test_promote_unknown_and_rejected_return_false(self):
        cons = Consolidator()
        self.assertFalse(cons.promote("no-such-id"))
        cid = cons.add_candidate("rule", {"action": "open_alpha"}, _ev(3), 2)
        cons.reject(cid, "weak evidence")
        self.assertFalse(cons.promote(cid))
        self.assertFalse(cons.promote(cid, force=True))

    def test_promote_is_idempotent(self):
        cons = Consolidator()
        cid = cons.add_candidate("pattern", {"kind": "x"}, _ev(3), 2)
        self.assertTrue(cons.promote(cid, cycle=1))
        self.assertTrue(cons.promote(cid, cycle=9))
        self.assertEqual(cons.get(cid)["promoted_at"], 1)


class TestConsolidatorRejectAndDedupe(unittest.TestCase):
    def test_reject_retires_and_is_not_readded(self):
        cons = Consolidator()
        cid = cons.add_candidate("pattern", {"kind": "durable_tool", "effect": "gate_open"},
                                 _ev(3), 2)
        self.assertTrue(cons.reject(cid, "contradicted by later episodes"))
        got = cons.get(cid)
        self.assertEqual(got["status"], "rejected")
        self.assertEqual(got["rejection_reason"], "contradicted by later episodes")
        # Re-submitting the identical candidate does not resurrect it.
        cid2 = cons.add_candidate("pattern", {"kind": "durable_tool", "effect": "gate_open"},
                                  _ev(5), 3)
        self.assertEqual(cid2, cid)
        self.assertEqual(cons.get(cid)["status"], "rejected")
        self.assertEqual([c["id"] for c in cons.rejected()], [cid])
        self.assertEqual(cons.review(), [])

    def test_reject_unknown_returns_false(self):
        self.assertFalse(Consolidator().reject("no-such-id"))

    def test_dedupe_same_content_same_id(self):
        cons = Consolidator()
        a = cons.add_candidate("concept", {"name": "durable_tool"}, _ev(1), 1)
        b = cons.add_candidate("concept", {"name": "durable_tool"}, _ev(1), 1)
        self.assertEqual(a, b)
        self.assertEqual(len(cons), 1)
        # Same content, different kind -> different candidate.
        c = cons.add_candidate("pattern", {"name": "durable_tool"}, _ev(1), 1)
        self.assertNotEqual(c, a)
        self.assertEqual(len(cons), 2)

    def test_id_is_stable_sha256_of_kind_and_content(self):
        content = {"kind": "durable_tool", "effect": "gate_open"}
        self.assertEqual(candidate_id("pattern", content),
                         candidate_id("pattern", dict(content)))
        self.assertEqual(len(candidate_id("pattern", content)), 64)

    def test_invalid_kind_rejected(self):
        with self.assertRaises(ValueError):
            Consolidator().add_candidate("hunch", {}, [], 0)

    def test_accessors_partition_by_status(self):
        cons = Consolidator()
        keep = cons.add_candidate("pattern", {"kind": "a"}, _ev(3), 2)
        drop = cons.add_candidate("pattern", {"kind": "b"}, _ev(1), 1)
        wait = cons.add_candidate("pattern", {"kind": "c"}, _ev(1), 1)
        cons.promote(keep)
        cons.reject(drop, "no")
        self.assertEqual([c["id"] for c in cons.candidates()], [wait])
        self.assertEqual([c["id"] for c in cons.promoted()], [keep])
        self.assertEqual([c["id"] for c in cons.rejected()], [drop])
        self.assertEqual(len(cons.all()), 3)


class TestConsolidatorJson(unittest.TestCase):
    def test_round_trip_preserves_statuses_and_evidence(self):
        cons = Consolidator()
        promoted = cons.add_candidate("pattern", {"kind": "a", "effect": "gate_open"},
                                      _ev(3), 2, provenance="run-1")
        rejected = cons.add_candidate("rule", {"action": "open_alpha"}, _ev(1), 1)
        pending = cons.add_candidate("concept", {"name": "durable_tool"}, _ev(2), 2)
        cons.promote(promoted, cycle=4)
        cons.reject(rejected, "weak")
        data = cons.to_json()
        self.assertEqual(data["version"], 1)
        # JSON-serialisable exactly as produced.
        restored = Consolidator.from_json(json.loads(json.dumps(data)))
        self.assertEqual(restored.get(promoted)["status"], "promoted")
        self.assertEqual(restored.get(promoted)["promoted_at"], 4)
        self.assertEqual(restored.get(promoted)["promoted_evidence"], 3)
        self.assertEqual(restored.get(promoted)["provenance"], "run-1")
        self.assertEqual(restored.get(rejected)["status"], "rejected")
        self.assertEqual(restored.get(rejected)["rejection_reason"], "weak")
        self.assertEqual(restored.get(pending)["status"], "candidate")
        self.assertEqual(len(restored.get(pending)["evidence"]), 2)
        # A restored rejected candidate is still not re-added.
        self.assertEqual(
            restored.add_candidate("rule", {"action": "open_alpha"}, _ev(9), 9),
            rejected)
        self.assertEqual(restored.get(rejected)["status"], "rejected")


# ------------------------------------------------------------------
# Concepts
# ------------------------------------------------------------------
class TestConceptsC1(unittest.TestCase):
    def test_durable_tool_positive_after_two_successes(self):
        store = ConceptStore()
        self.assertFalse(store.learn_durable_tool("alpha_tool", 1, broke=False))
        self.assertIsNone(store.get("durable_tool"))
        self.assertTrue(store.learn_durable_tool("alpha_tool", 2, broke=False))
        concept = store.get("durable_tool")
        self.assertEqual(len(concept.positives), 1)
        self.assertEqual(len(concept.negatives), 0)
        ok, conf = store.matches("durable_tool", {"breaks_after_first_use": False})
        self.assertTrue(ok)
        self.assertGreater(conf, 0.5)
        ok, conf = store.is_a({"breaks_after_first_use": False}, "durable_tool")
        self.assertTrue(ok)

    def test_durable_tool_negative_when_broke_after_first_use(self):
        store = ConceptStore()
        self.assertTrue(store.learn_durable_tool("beta_tool", 1, broke=True))
        concept = store.get("durable_tool")
        self.assertEqual(len(concept.negatives), 1)
        self.assertEqual(len(concept.positives), 0)
        # No positive example yet: nothing can match.
        ok, _ = store.matches("durable_tool", {"breaks_after_first_use": False})
        self.assertFalse(ok)
        self.assertLess(concept.confidence, 0.5)

    def test_durable_tool_mixed_examples(self):
        store = ConceptStore()
        store.learn_durable_tool("alpha_tool", 3, broke=False)
        store.learn_durable_tool("beta_tool", 1, broke=True)
        store.learn_durable_tool("gamma_tool", 2, broke=False)
        concept = store.get("durable_tool")
        self.assertEqual(len(concept.positives), 2)
        self.assertEqual(len(concept.negatives), 1)
        # Laplace: (2+1)/(3+2) = 0.6.
        self.assertAlmostEqual(concept.confidence, 0.6)
        ok, _ = store.matches("durable_tool", {"breaks_after_first_use": False})
        self.assertTrue(ok)
        ok, _ = store.matches("durable_tool", {"breaks_after_first_use": True})
        self.assertFalse(ok)


class TestConceptsC2(unittest.TestCase):
    def test_requires_learned_from_shared_missing_key(self):
        store = ConceptStore()
        failed = [{"at": "hall"}, {"at": "hall", "has_other": True}]
        succeeded = [{"at": "hall", "has_alpha_key": True},
                     {"at": "corridor", "has_alpha_key": True}]
        names = store.learn_requires("open_alpha", failed, succeeded)
        self.assertEqual(names, ["requires_has_alpha_key"])
        ok, conf = store.matches("requires_has_alpha_key",
                                 {"has_alpha_key": True})
        self.assertTrue(ok)
        self.assertGreater(conf, 0.5)
        ok, _ = store.matches("requires_has_alpha_key", {"at": "hall"})
        self.assertFalse(ok)
        ok, _ = store.is_a({"has_alpha_key": True}, "requires_has_alpha_key")
        self.assertTrue(ok)

    def test_requires_needs_two_failures(self):
        store = ConceptStore()
        names = store.learn_requires(
            "open_alpha", [{"at": "hall"}],
            [{"at": "hall", "has_alpha_key": True}])
        self.assertEqual(names, [])
        self.assertIsNone(store.get("requires_has_alpha_key"))

    def test_requires_needs_a_success_with_key(self):
        store = ConceptStore()
        names = store.learn_requires(
            "open_alpha", [{"at": "hall"}, {"at": "hall"}],
            [{"at": "hall"}])
        self.assertEqual(names, [])

    def test_requires_no_key_shared_missing(self):
        store = ConceptStore()
        # Each failure lacks a *different* key; no single key is
        # missing in every failure, so nothing is learned.
        failed = [{"at": "hall", "has_beta_key": True},
                  {"at": "hall", "has_alpha_key": True}]
        succeeded = [{"at": "hall", "has_alpha_key": True,
                      "has_beta_key": True}]
        names = store.learn_requires("open_alpha", failed, succeeded)
        self.assertEqual(names, [])


class TestConceptsGeneral(unittest.TestCase):
    def test_matches_both_directions(self):
        store = ConceptStore()
        store.add_example("durable_tool", {"breaks_after_first_use": False},
                          positive=True, provenance="run-1")
        ok, conf = store.matches("durable_tool", {"breaks_after_first_use": False})
        self.assertTrue(ok)
        self.assertAlmostEqual(conf, store.get("durable_tool").confidence)
        ok, _ = store.matches("durable_tool", {"breaks_after_first_use": True})
        self.assertFalse(ok)
        ok, _ = store.matches("durable_tool", {})
        self.assertFalse(ok)
        ok, conf = store.matches("no_such_concept", {"x": 1})
        self.assertFalse(ok)
        self.assertEqual(conf, 0.0)

    def test_concept_id_stable_over_name_and_sorted_features(self):
        self.assertEqual(concept_id("durable_tool", {"a": 1, "b": 2}),
                         concept_id("durable_tool", {"b": 2, "a": 1}))
        self.assertNotEqual(concept_id("durable_tool", {"a": 1}),
                            concept_id("durable_tool", {"a": 2}))
        self.assertNotEqual(concept_id("durable_tool", {"a": 1}),
                            concept_id("fragile_tool", {"a": 1}))
        concept = Concept(name="durable_tool",
                          features={"breaks_after_first_use": False})
        self.assertEqual(concept.id,
                         concept_id("durable_tool",
                                    {"breaks_after_first_use": False}))

    def test_confidence_is_laplace_smoothed(self):
        store = ConceptStore()
        store.add_example("c", {"f": 1}, positive=True, provenance="p")
        self.assertAlmostEqual(store.get("c").confidence, 2 / 3)
        store.add_example("c", {"f": 1}, positive=True, provenance="p")
        store.add_example("c", {"f": 0}, positive=False, provenance="p")
        self.assertAlmostEqual(store.get("c").confidence, 3 / 5)

    def test_json_round_trip(self):
        store = ConceptStore()
        store.learn_durable_tool("alpha_tool", 2, broke=False)
        store.learn_durable_tool("beta_tool", 1, broke=True)
        store.learn_requires("open_alpha",
                             [{"at": "hall"}, {"at": "hall"}],
                             [{"at": "hall", "has_alpha_key": True}])
        data = store.to_json()
        self.assertEqual(data["version"], 1)
        restored = ConceptStore.from_json(json.loads(json.dumps(data)))
        self.assertEqual(restored.names(), store.names())
        ok, _ = restored.matches("durable_tool", {"breaks_after_first_use": False})
        self.assertTrue(ok)
        ok, _ = restored.matches("requires_has_alpha_key",
                                 {"has_alpha_key": True})
        self.assertTrue(ok)
        self.assertAlmostEqual(
            restored.get("durable_tool").confidence,
            store.get("durable_tool").confidence)


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------
class TestCalibrationBasics(unittest.TestCase):
    def test_uncalibrated_with_no_record(self):
        tracker = CalibrationTracker()
        self.assertEqual(tracker.calibrated_confidence("open_alpha", "gate_open"), 1.0)
        self.assertFalse(tracker.is_calibrated("open_alpha", "gate_open"))
        self.assertIsNone(tracker.counts("open_alpha", "gate_open"))
        self.assertIsNone(tracker.gap("open_alpha", "gate_open"))
        self.assertEqual(tracker.ece(), 0.0)
        self.assertEqual(tracker.brier(), 0.0)

    def test_laplace_calibrated_confidence(self):
        tracker = CalibrationTracker()
        for _ in range(3):
            tracker.record("open_alpha", "gate_open", 0.9, True)
        tracker.record("open_alpha", "gate_open", 0.9, False)
        self.assertEqual(tracker.counts("open_alpha", "gate_open"), (3, 4))
        # (3+1)/(4+2) = 2/3.
        self.assertAlmostEqual(
            tracker.calibrated_confidence("open_alpha", "gate_open"), 2 / 3)
        self.assertTrue(tracker.is_calibrated("open_alpha", "gate_open"))

    def test_gap_closes_as_predictions_align(self):
        miscalibrated = CalibrationTracker()
        aligned = CalibrationTracker()
        for _ in range(10):
            miscalibrated.record("open_alpha", "gate_open", 1.0, True)
            miscalibrated.record("open_alpha", "gate_open", 1.0, False)
            aligned.record("open_alpha", "gate_open", 0.5, True)
            aligned.record("open_alpha", "gate_open", 0.5, False)
        # Mean predicted 1.0 vs observed 0.5 -> gap 0.5.
        self.assertAlmostEqual(miscalibrated.gap("open_alpha", "gate_open"), 0.5)
        # Mean predicted 0.5 vs observed 0.5 -> gap 0.0.
        self.assertAlmostEqual(aligned.gap("open_alpha", "gate_open"), 0.0)
        self.assertLess(aligned.gap("open_alpha", "gate_open"),
                        miscalibrated.gap("open_alpha", "gate_open"))

    def test_ece_decreases_as_predictions_align(self):
        miscalibrated = CalibrationTracker()
        aligned = CalibrationTracker()
        for i in range(10):
            miscalibrated.record("open_alpha", "gate_open", 0.9, i % 2 == 0)
            aligned.record("open_alpha", "gate_open", 0.5, i % 2 == 0)
        # Miscalibrated: one bin, conf 0.9 vs accuracy 0.5 -> ECE 0.4.
        self.assertAlmostEqual(miscalibrated.ece(), 0.4)
        # Aligned: conf 0.5 vs accuracy 0.5 -> ECE 0.0.
        self.assertAlmostEqual(aligned.ece(), 0.0)
        self.assertLess(aligned.ece(), miscalibrated.ece())

    def test_brier(self):
        perfect = CalibrationTracker()
        perfect.record("a", "e", 1.0, True)
        perfect.record("a", "e", 0.0, False)
        self.assertAlmostEqual(perfect.brier(), 0.0)
        coinflip = CalibrationTracker()
        for outcome in (True, False, True, False):
            coinflip.record("a", "e", 0.5, outcome)
        self.assertAlmostEqual(coinflip.brier(), 0.25)
        bad = CalibrationTracker()
        bad.record("a", "e", 1.0, False)
        bad.record("a", "e", 0.0, True)
        self.assertAlmostEqual(bad.brier(), 1.0)
        self.assertLess(coinflip.brier(), bad.brier())

    def test_summary(self):
        tracker = CalibrationTracker()
        tracker.record("open_alpha", "gate_open", 0.8, True)
        tracker.record("open_alpha", "gate_open", 0.8, False)
        summary = tracker.summary()
        self.assertEqual(summary["total_records"], 2)
        pair = summary["pairs"]["open_alpha|gate_open"]
        self.assertEqual(pair["met"], 1)
        self.assertEqual(pair["total"], 2)
        self.assertAlmostEqual(pair["observed_rate"], 0.5)
        self.assertAlmostEqual(pair["mean_predicted"], 0.8)
        self.assertAlmostEqual(pair["gap"], 0.3)
        self.assertTrue(pair["is_calibrated"])
        self.assertIn("ece", summary)
        self.assertIn("brier", summary)


class TestCalibrationJson(unittest.TestCase):
    def test_round_trip_preserves_counts_brier_and_ece(self):
        tracker = CalibrationTracker()
        for i in range(6):
            tracker.record("open_alpha", "gate_open", 0.9, i % 2 == 0)
        for i in range(4):
            tracker.record("use_gamma_tool", "gate_open", 0.5, True)
        data = tracker.to_json()
        self.assertEqual(data["version"], 1)
        restored = CalibrationTracker.from_json(json.loads(json.dumps(data)))
        self.assertEqual(restored.counts("open_alpha", "gate_open"), (3, 6))
        self.assertEqual(restored.counts("use_gamma_tool", "gate_open"), (4, 4))
        self.assertAlmostEqual(
            restored.calibrated_confidence("open_alpha", "gate_open"),
            tracker.calibrated_confidence("open_alpha", "gate_open"))
        self.assertAlmostEqual(restored.brier(), tracker.brier())
        # Predictions are constant within each pair, so the
        # summary-only ECE fallback equals the exact ECE.
        self.assertAlmostEqual(restored.ece(), tracker.ece())
        self.assertTrue(restored.is_calibrated("open_alpha", "gate_open"))
        self.assertFalse(restored.is_calibrated("nope", "nope"))


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def _episode(action, situation, effect, cycle):
    return {"action": action, "situation": dict(situation),
            "effect": effect, "cycle": cycle}


class TestRetrieval(unittest.TestCase):
    def test_ranks_relevant_episodes_first(self):
        retriever = Retriever([
            _episode("open_alpha", {"at": "hall"}, "gate_open", 1),
            _episode("open_alpha", {"at": "hall", "has_alpha_key": True},
                     "gate_open", 2),
            _episode("open_beta", {"at": "garden", "has_beta_key": True},
                     "other_effect", 3),
        ])
        top = retriever.retrieve({"at": "hall", "has_alpha_key": True,
                                  "effect": "gate_open"}, k=3)
        self.assertEqual(top[0]["cycle"], 2)  # same situation + effect
        self.assertEqual(top[1]["cycle"], 1)  # same effect, partial situation
        self.assertEqual(top[2]["cycle"], 3)  # neither

    def test_effect_bonus_outranks_situation_only(self):
        retriever = Retriever([
            _episode("open_alpha", {"at": "hall", "has_alpha_key": True},
                     "other_effect", 1),
            _episode("open_alpha", {"at": "hall"}, "gate_open", 2),
        ])
        top = retriever.retrieve({"at": "hall", "has_alpha_key": True,
                                  "effect": "gate_open"}, k=2)
        # Similar situation but wrong effect (1.0) vs partial situation
        # + effect bonus (0.5 + 0.5 = 1.0): tie -> most recent first.
        self.assertEqual(top[0]["cycle"], 2)

    def test_most_recent_first_on_ties(self):
        retriever = Retriever([
            _episode("open_alpha", {"at": "hall"}, "gate_open", 1),
            _episode("open_alpha", {"at": "hall"}, "gate_open", 2),
            _episode("open_alpha", {"at": "hall"}, "gate_open", 3),
        ])
        top = retriever.retrieve({"at": "hall"}, k=3)
        self.assertEqual([ep["cycle"] for ep in top], [3, 2, 1])

    def test_k_limit_and_empty(self):
        retriever = Retriever([
            _episode("open_alpha", {"at": "hall"}, "gate_open", i)
            for i in range(5)])
        self.assertEqual(len(retriever.retrieve({"at": "hall"}, k=2)), 2)
        self.assertEqual(retriever.retrieve({"at": "hall"}, k=0), [])
        self.assertEqual(Retriever().retrieve({"at": "hall"}), [])

    def test_precision_helper(self):
        retriever = Retriever([
            _episode("open_alpha", {"at": "hall", "has_alpha_key": True},
                     "gate_open", 1),
            _episode("open_beta", {"at": "hall", "has_alpha_key": True},
                     "gate_open", 2),
            _episode("open_alpha", {"at": "garden"}, "other_effect", 3),
        ])
        precision = retriever.precision({"at": "hall", "has_alpha_key": True},
                                        "gate_open", k=2)
        self.assertAlmostEqual(precision, 1.0)
        precision = retriever.precision({"at": "hall", "has_alpha_key": True},
                                        "gate_open", k=3)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertEqual(Retriever().precision({"at": "hall"}, "gate_open"), 0.0)

    def test_clear_and_rebuild(self):
        episodes = [_episode("open_alpha", {"at": "hall"}, "gate_open", 1)]
        retriever = Retriever(episodes)
        self.assertEqual(len(retriever), 1)
        retriever.clear()
        self.assertEqual(len(retriever), 0)
        self.assertEqual(retriever.retrieve({"at": "hall"}), [])
        retriever.rebuild(episodes)
        self.assertEqual(len(retriever), 1)
        self.assertEqual(retriever.retrieve({"at": "hall"})[0]["cycle"], 1)
        retriever.index([])
        self.assertEqual(len(retriever), 0)


# ------------------------------------------------------------------
# Prioritization
# ------------------------------------------------------------------
def _ep(action, situation, cycle):
    return {"action": action, "situation": dict(situation), "cycle": cycle}


class TestPrioritization(unittest.TestCase):
    def test_utility_formula(self):
        pri = Prioritizer()
        fresh = _ep("open_alpha", {"at": "hall"}, 10)
        self.assertAlmostEqual(pri.utility(fresh, 10), 0.5)  # age 0, 0 uses
        old = _ep("open_alpha", {"at": "hall"}, 0)
        # age 10: 0.5 * 0.9**10 = 0.5 * 0.3487.
        self.assertAlmostEqual(pri.utility(old, 10), 0.5 * 0.9 ** 10)
        pri.note_use("open_alpha", {"at": "hall"})
        pri.note_use("open_alpha", {"at": "hall"})
        pri.note_use("open_alpha", {"at": "hall"})
        # 3 uses -> usefulness 1.0: 0.5 * 1.0 + 0.5 * 1.0 = 1.0.
        self.assertAlmostEqual(pri.utility(fresh, 10), 1.0)
        self.assertAlmostEqual(pri.utility(old, 10), 0.5 * 0.9 ** 10 + 0.5)

    def test_usefulness_capped_at_one(self):
        pri = Prioritizer()
        for _ in range(10):
            pri.note_use("open_alpha", {"at": "hall"})
        ep = _ep("open_alpha", {"at": "hall"}, 5)
        # usefulness min(1.0, 10/3) = 1.0; recency 0.9**0 = 1.0.
        self.assertAlmostEqual(pri.utility(ep, 5), 1.0)
        self.assertEqual(pri.uses("open_alpha", {"at": "hall"}), 10)
        self.assertEqual(pri.uses("open_alpha", {"at": "garden"}), 0)

    def test_recency_age_capped_at_100(self):
        pri = Prioritizer()
        ep100 = _ep("open_alpha", {"at": "hall"}, 0)
        ep200 = _ep("open_alpha", {"at": "hall"}, 0)
        self.assertAlmostEqual(pri.utility(ep100, 100),
                               pri.utility(ep200, 200))
        self.assertAlmostEqual(pri.utility(ep100, 100), 0.5 * 0.9 ** 100)

    def test_utility_with_explicit_counts(self):
        pri = Prioritizer()
        ep = _ep("open_alpha", {"at": "hall"}, 5)
        counts = {("open_alpha", (("at", "hall"),)): 3}
        self.assertAlmostEqual(pri.utility(ep, 5, counts), 1.0)
        self.assertAlmostEqual(pri.utility(ep, 5, {}), 0.5)

    def test_prune_keeps_high_utility_under_small_cap(self):
        pri = Prioritizer()
        episodes = [
            _ep("open_alpha", {"at": "hall"}, 0),    # old, unused -> low
            _ep("open_alpha", {"at": "hall"}, 9),    # recent -> higher
            _ep("open_beta", {"at": "garden"}, 0),   # old but used -> highest
            _ep("open_beta", {"at": "garden"}, 1),   # old, unused -> low
        ]
        for _ in range(3):
            pri.note_use("open_beta", {"at": "garden"})
        kept = pri.prune(episodes, cap=2, current_cycle=10)
        self.assertEqual(len(kept), 2)
        # Usefulness is keyed by (action, situation), so both garden
        # episodes get the full usefulness bonus; the more recent of
        # the two outranks the older one, and both outrank the
        # recent-but-unused hall episode.
        kept_cycles = sorted(ep["cycle"] for ep in kept)
        self.assertEqual(kept_cycles, [0, 1])
        self.assertTrue(all(ep["situation"] == {"at": "garden"} for ep in kept))
        # Highest utility first: the more recent garden episode.
        self.assertEqual(kept[0]["cycle"], 1)

    def test_prune_drops_low_utility(self):
        pri = Prioritizer()
        episodes = [_ep("open_alpha", {"at": "hall"}, i) for i in range(5)]
        kept = pri.prune(episodes, cap=2, current_cycle=10)
        self.assertEqual([ep["cycle"] for ep in kept], [4, 3])

    def test_prune_stable_order_on_ties(self):
        pri = Prioritizer()
        episodes = [_ep("open_alpha", {"at": "hall"}, 5) for _ in range(3)]
        kept = pri.prune(episodes, cap=2, current_cycle=5)
        self.assertEqual(len(kept), 2)
        # All identical: stable order keeps the first two.
        self.assertEqual(kept, episodes[:2])

    def test_prune_only_touches_episodes(self):
        # prune() takes episodes and returns a subset of them: no
        # pattern or consolidated record is ever introduced or
        # dropped by it -- patterns are the durable summary and
        # live outside the episode store.
        pri = Prioritizer()
        episodes = [_ep("open_alpha", {"at": "hall"}, i) for i in range(4)]
        kept = pri.prune(episodes, cap=2, current_cycle=10)
        self.assertTrue(all(ep in episodes for ep in kept))
        self.assertEqual(len(kept), 2)
        self.assertEqual(pri.prune(episodes, cap=0), [])
        self.assertEqual(len(pri.prune(episodes, cap=10)), 4)

    def test_json_round_trip(self):
        pri = Prioritizer()
        pri.note_use("open_alpha", {"at": "hall"})
        pri.note_use("open_alpha", {"at": "hall"})
        pri.note_use("open_beta", {"at": "garden"})
        data = pri.to_json()
        self.assertEqual(data["version"], 1)
        restored = Prioritizer.from_json(json.loads(json.dumps(data)))
        self.assertEqual(restored.uses("open_alpha", {"at": "hall"}), 2)
        self.assertEqual(restored.uses("open_beta", {"at": "garden"}), 1)
        ep = _ep("open_alpha", {"at": "hall"}, 5)
        self.assertAlmostEqual(restored.utility(ep, 5), pri.utility(ep, 5))


# ------------------------------------------------------------------
# Package exports
# ------------------------------------------------------------------
class TestExports(unittest.TestCase):
    def test_init_exports(self):
        import ab.learning as learning
        for name in ("Consolidator", "Concept", "ConceptStore",
                     "CalibrationTracker", "Retriever", "Prioritizer"):
            self.assertTrue(hasattr(learning, name), name)


if __name__ == "__main__":
    unittest.main()
