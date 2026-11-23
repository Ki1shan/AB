"""Phase 10 (P10-7, P10-8, P10-9): autonomous perception, revision-count
persistence, and the per-word OCR confidence experiment.

P10-7 — the cycle now ACTS on a ``need_percept`` request instead of
only recording it: when the goal names a readable source in
``goal.meta`` (``document``/``documents`` or ``image``/``images``) the
brain reads it in-cycle through the perception bus and the result
enters WM/LTM by the existing path; when the missing information is
environmental the request drives a SPECIFIC re-observation, recorded
as a ``percept_requested`` event. Boundary: no autonomous sensing loop
beyond the need-driven request. A single document still never verifies
a fact (P8-2, file trust 0.45): after reading, ``ask()`` answers from
the stored (unverified) claim while ``known()`` stays False.

P10-8 — ``Brain._pattern_contradictions`` counts persist inside
``revision.json`` under ``"pattern_contradictions"`` (no persistence
version bump, P8-7 precedent); the loader defaults to empty when the
field is absent, and the retirement threshold reads the restored
counts, so a contradiction observed before a restart still counts
toward retirement after it.

P10-9 — the per-word OCR confidence experiment (revisit of P9-3, WITH
evidence): gate thresholds 0.6 vs 0.8 on words straddling both, and
admitted-word confidence 0.65 vs 0.95 at a fixed gate. Finding (see
``TestOcrConfidenceExperiment``): the gate DOES change the outcome
(the claim is present at 0.6, absent at 0.8 — the gate working as the
mechanism), but per-word confidence among ADMITTED words changes no
stored outcome (flat 0.5 source trust either way), so no decision
reads the information propagation would carry. P9-3 STANDS.

Stdlib only, deterministic.
"""
import json
import os
import tempfile
import unittest

from ab.body.ocr import FakeWordOcr
from ab.brain import Brain
from ab.core.learning import Episode
from ab.planning.planner import Action


# ------------------------------------------------------------------ helpers
def _doc_brain(tmpdir, text, goal_target, priority=0.1, meta_key="document"):
    """Brain with one pending goal whose answer is in a document it
    has not read; the document path is named in ``goal.meta``."""
    path = os.path.join(tmpdir, "notes.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    b = Brain([], granted={"compute"})
    goal = b.add_goal("learn the missing fact", goal_target, priority=priority)
    goal.meta[meta_key] = path
    return b, path


def _percept_requested(brain):
    return [e for e in brain.events if e["kind"] == "percept_requested"]


# The P-D promotion helpers are copied from test_phase9_retirement.py
# (same world X evidence through the normal consolidation path).
def _press_actions():
    return [
        Action("press", {"at": "bench"}, {"pressed": True}, cost=1.0),
        Action("fetch_mould", {"at": "bench"}, {"has_mould": True}, cost=1.0),
    ]


def _promote_pd_pattern(brain):
    brain.learner.record(Episode(1, "press", {"at": "bench"}, False))
    brain.learner.record(
        Episode(2, "press", {"at": "bench", "has_mould": False}, False))
    brain.learner.record(
        Episode(3, "press", {"at": "bench", "has_mould": True}, True))
    brain._consolidation_pass()
    found = [p for p in brain.patterns.patterns()
             if p.kind == "requires_precondition"]
    assert len(found) == 1, (
        f"expected exactly one promoted P-D pattern, got {found!r}")
    return found[0]


def _pd_match(brain):
    return brain.patterns.match(
        "press", {"at": "bench"}, {"pressed": True}, brain.world_model,
        [(a.name, a.pre, a.eff) for a in brain.planner.actions],
        {"at": "bench"})


class TestAutonomousPerception(unittest.TestCase):
    def test_document_goal_read_in_cycle_then_answered(self):
        d = tempfile.mkdtemp()
        b, path = _doc_brain(
            d, "Quibber is the capital of Zorblat.\n",
            {"known:quibber|capital_of": True})
        self.assertFalse(b.ask("quibber", "capital_of").known)
        b.step()
        # The brain requested AND read the document in-cycle.
        reqs = _percept_requested(b)
        self.assertEqual(len(reqs), 1)
        self.assertIn("document", reqs[0]["detail"])
        self.assertIn("notes.txt", reqs[0]["detail"])
        self.assertIn("quibber", reqs[0]["detail"])
        self.assertTrue(any(e["kind"] == "document_read" for e in b.events))
        # ...and then answers from what it read, instead of failing
        # with insufficient evidence.
        answer = b.ask("quibber", "capital_of")
        self.assertTrue(answer.known)
        self.assertEqual(answer.best[0], "zorblat")
        self.assertEqual(answer.best[2], "unverified")  # P8-2: one doc never verifies
        self.assertEqual(answer.best[1], 0.45)           # flat file trust
        # known() still False: verification needs a second source.
        self.assertFalse(b.known("quibber", "capital_of"))

    def test_control_without_document_source_still_does_not_know(self):
        b = Brain([], granted={"compute"})
        b.add_goal("learn capital of quibber",
                   {"known:quibber|capital_of": True}, priority=0.1)
        b.step()
        self.assertFalse(b.ask("quibber", "capital_of").known)
        self.assertEqual(_percept_requested(b), [])
        self.assertFalse(any(e["kind"] == "document_read" for e in b.events))

    def test_document_read_once_despite_repeated_need(self):
        d = tempfile.mkdtemp()
        b, _ = _doc_brain(
            d, "Quibber is the capital of Zorblat.\n",
            {"known:quibber|capital_of": True})
        b.step()
        b.step()
        b.step()
        self.assertEqual(len(_percept_requested(b)), 1)
        self.assertEqual(
            sum(1 for e in b.events if e["kind"] == "document_read"), 1)
        self.assertTrue(b.ask("quibber", "capital_of").known)

    def test_observation_need_records_specific_percept_request(self):
        # A 20-step physical plan is refused by the feasibility gate;
        # the missing key is physical, so the request drives a
        # specific re-observation — recorded, not a document read.
        acts = [Action(f"s{i}", {"n": i}, {"n": i + 1}) for i in range(20)]
        b = Brain(acts, granted={"compute"})
        b.world = {"n": 0}
        b.add_goal("count to twenty", {"n": 20}, priority=0.9)
        b.step()
        reqs = _percept_requested(b)
        self.assertEqual(len(reqs), 1)
        self.assertIn("n", reqs[0]["detail"])
        self.assertIn("re-observing", reqs[0]["detail"])
        self.assertFalse(any(e["kind"] == "document_read" for e in b.events))
        # Still one request on the next cycle: the acted-on set dedupes.
        b.step()
        self.assertEqual(len(_percept_requested(b)), 1)

    def test_image_source_read_via_ocr_in_cycle(self):
        words = [("Quibber", 0.95), ("is", 0.9), ("a", 0.9), ("planet", 0.9)]
        b = Brain(ocr=FakeWordOcr({"sign.png": words}), granted={"compute"})
        goal = b.add_goal("learn what quibber is",
                          {"known:quibber|is_a": True}, priority=0.1)
        goal.meta["image"] = "sign.png"
        self.assertFalse(b.ask("quibber", "is_a").known)
        b.step()
        reqs = _percept_requested(b)
        self.assertEqual(len(reqs), 1)
        self.assertIn("image", reqs[0]["detail"])
        self.assertIn("sign.png", reqs[0]["detail"])
        answer = b.ask("quibber", "is_a")
        self.assertTrue(answer.known)
        self.assertEqual(answer.best[0], "planet")

    def test_missing_document_file_fails_cleanly(self):
        b = Brain([], granted={"compute"})
        goal = b.add_goal("learn capital of quibber",
                          {"known:quibber|capital_of": True}, priority=0.1)
        goal.meta["document"] = "/tmp/ab-phase10-no-such-notes.txt"
        b.step()  # must not raise: the failed read is logged, cycle continues
        self.assertEqual(len(_percept_requested(b)), 1)
        self.assertTrue(any(e["kind"] == "document_read" and "failed" in e["detail"]
                            for e in b.events))
        self.assertFalse(b.ask("quibber", "capital_of").known)

    def test_no_percept_requested_without_a_need(self):
        b = Brain([], granted={"compute"})
        b.perceive_text("hello world")
        b.step()
        self.assertEqual(_percept_requested(b), [])


class TestRevisionCountPersistence(unittest.TestCase):
    def test_counts_survive_save_load_and_retire_after_restart(self):
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_pattern(brain)
        pid = pattern.id
        self.assertEqual(_pd_match(brain), 6.0)
        # Two contradictions before the restart: not yet retired.
        for i in range(2):
            result = brain.record_pattern_contradiction(
                pid, f"press succeeded without has_mould (observation {i + 1})")
            self.assertEqual(result["contradictions"], i + 1)
            self.assertFalse(result["retired"])
        folder = tempfile.mkdtemp(prefix="ab-phase10-revision-")
        brain.save(folder)

        loaded = Brain(_press_actions(), granted={"compute", "environment"})
        loaded.load(folder)
        # The restored counts are visible before any new observation.
        self.assertIn(pid, loaded._pattern_contradictions)
        self.assertEqual(len(loaded._pattern_contradictions[pid]), 2)
        # The third contradiction AFTER the restart retires the
        # pattern: the pre-restart observations still counted.
        result = loaded.record_pattern_contradiction(
            pid, "press succeeded without has_mould (observation 3, after restart)")
        self.assertEqual(result["contradictions"], 3)
        self.assertTrue(result["retired"])
        self.assertEqual(_pd_match(loaded), 1.0)
        self.assertTrue(loaded.patterns.is_retired(pid))

    def test_revision_json_carries_counts_and_version_stays_6(self):
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_pattern(brain)
        brain.record_pattern_contradiction(pattern.id, "press succeeded")
        folder = tempfile.mkdtemp(prefix="ab-phase10-revision-")
        brain.save(folder)
        with open(os.path.join(folder, "revision.json"), encoding="utf-8") as fh:
            revision = json.load(fh)
        self.assertIn("pattern_contradictions", revision)
        self.assertIn(pattern.id, revision["pattern_contradictions"])
        recs = revision["pattern_contradictions"][pattern.id]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["detail"], "press succeeded")
        self.assertIn("cycle", recs[0])
        # No persistence version bump (P8-7 precedent).
        # Phase 11 pin update: v7 (research.json added) — the bump is
        # Phase 11's, not Phase 10's.
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["version"], 7)
        self.assertEqual(meta["files"]["revision.json"], 1)

    def test_old_save_without_field_loads_with_empty_counts(self):
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_pattern(brain)
        brain.record_pattern_contradiction(pattern.id, "press succeeded")
        folder = tempfile.mkdtemp(prefix="ab-phase10-revision-")
        brain.save(folder)
        # Simulate a pre-P10-8 save: strip the new field from disk.
        rev_path = os.path.join(folder, "revision.json")
        with open(rev_path, encoding="utf-8") as fh:
            revision = json.load(fh)
        del revision["pattern_contradictions"]
        with open(rev_path, "w", encoding="utf-8") as fh:
            json.dump(revision, fh)
        loaded = Brain(_press_actions(), granted={"compute", "environment"})
        loaded.load(folder)  # must not raise, must not warn about the field
        self.assertEqual(loaded._pattern_contradictions, {})
        result = loaded.record_pattern_contradiction(pattern.id, "again")
        self.assertEqual(result["contradictions"], 1)
        self.assertFalse(result["retired"])


class TestOcrConfidenceExperiment(unittest.TestCase):
    """P10-9: revisit P9-3 WITH EVIDENCE. Two sub-experiments.

    (A) Gate sensitivity: words straddling both thresholds — "is"
    at 0.65 and "planet" at 0.75 sit between 0.6 and 0.8.
    (B) Propagation sensitivity: the gate fixed at 0.6, admitted
    words at 0.65 vs 0.95 — does per-word confidence among ADMITTED
    words change any stored outcome?
    """
    WORDS_STRADDLING = [("Quibber", 0.95), ("is", 0.65),
                        ("a", 0.9), ("planet", 0.75)]

    def test_gate_threshold_changes_ask_outcome(self):
        # (A) At 0.6 every word is admitted: the claim is stored and
        # ask() answers. At 0.8 "is" and "planet" are uncertain and
        # never reach WM/LTM: the admitted text ("Quibber a") yields
        # no claim and ask() answers "I don't know". The gate is the
        # operative mechanism — and it is binary: it discards the
        # whole claim rather than recording a low-confidence one.
        low = Brain(ocr=FakeWordOcr({"sign.png": self.WORDS_STRADDLING}),
                    granted={"compute"})
        out_low = low.perceive_image("sign.png")
        self.assertEqual(out_low["text"], "Quibber is a planet")
        self.assertEqual(out_low["uncertain"], [])
        self.assertTrue(low.ask("quibber", "is_a").known)

        high = Brain(ocr=FakeWordOcr({"sign.png": self.WORDS_STRADDLING}),
                     granted={"compute"}, ocr_min_confidence=0.8)
        out_high = high.perceive_image("sign.png")
        self.assertEqual(out_high["text"], "Quibber a")
        self.assertEqual(sorted(w["text"] for w in out_high["uncertain"]),
                         ["is", "planet"])
        self.assertFalse(high.ask("quibber", "is_a").known)
        self.assertEqual(high.ltm.query("quibber", "is_a"), [])

    def test_admitted_word_confidence_changes_no_stored_outcome(self):
        # (B) Same gate (0.6), same sentence, wildly different
        # per-word confidences among admitted words: the stored claim
        # is IDENTICAL — flat ocr source trust 0.5, same confidence,
        # same ask() answer. No decision in the current code reads
        # the per-word confidence of an admitted word, so propagation
        # would change reported numbers only, never an outcome.
        # That is the evidence for P9-3 STANDING: the gate stays the
        # mechanism; propagation (which would need a per-claim
        # confidence field on Fact plus LTM trust-model changes)
        # is NOT implemented.
        low_words = [("Quibber", 0.65), ("is", 0.65),
                     ("a", 0.65), ("planet", 0.65)]
        high_words = [("Quibber", 0.95), ("is", 0.95),
                      ("a", 0.95), ("planet", 0.95)]
        b_low = Brain(ocr=FakeWordOcr({"sign.png": low_words}),
                      granted={"compute"})
        b_high = Brain(ocr=FakeWordOcr({"sign.png": high_words}),
                       granted={"compute"})
        out_low = b_low.perceive_image("sign.png")
        out_high = b_high.perceive_image("sign.png")
        self.assertEqual(out_low["text"], out_high["text"])
        fact_low = b_low.ltm.query("quibber", "is_a", "planet")[0]
        fact_high = b_high.ltm.query("quibber", "is_a", "planet")[0]
        self.assertEqual(fact_low.sources, {"ocr:sign.png": 0.5})
        self.assertEqual(fact_high.sources, {"ocr:sign.png": 0.5})
        self.assertEqual(fact_low.confidence, fact_high.confidence)
        self.assertEqual(fact_low.confidence, 0.5)
        self.assertEqual(fact_low.status, fact_high.status)
        self.assertEqual(b_low.ask("quibber", "is_a").answers,
                         b_high.ask("quibber", "is_a").answers)


if __name__ == "__main__":
    unittest.main()
