"""Phase 8 W1 (perception): one bus, document reading, OCR gate, image percepts.

All sources here are fakes — no real network, no real tesseract, no
real specialist model.
"""
import functools
import inspect
import os
import tempfile
import unittest

from ab.body.ocr import FakeWordOcr, OcrUnavailable
from ab.body.providers import MemoryKnowledgeProvider
from ab.body.reading import (DOCUMENT_TRUST, DocumentReader,
                             read_text_file)
from ab.brain import Brain
from ab.core.memory import TRUST
from ab.perception import (Percept, PerceptionBus, StubImageSensor,
                           make_image_record, validate_image_record)
from ab.world.model import EpistemicStatus

PARIS_DOC = "Paris is the capital of France. Mars is a planet."
PARIS_MD = "# Paris notes\n\nParis is the capital of France. Mars is a planet. The sky is very blue today.\n"


def _write_tmp(suffix, text):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


class FakeSpecialist:
    """Fake specialist model for tests: deterministic structured records."""

    name = "fake-vision"

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.calls = []

    def available(self):
        return True

    def describe_image(self, path):
        self.calls.append(path)
        record = self.table.get(path, {"objects": [
            {"name": "cup", "properties": {"color": "red"},
             "position": {"x": 10, "y": 20}, "confidence": 0.9}],
            "relations": [{"subject": "cup", "relation": "left_of",
                           "object": "plate", "confidence": 0.9}],
            "source": "fake-vision", "image_id": path})
        return dict(record)


class TestPercept(unittest.TestCase):
    def test_defaults_and_provenance(self):
        p = Percept(kind="text", content="hello", source="user",
                    confidence=0.7, provenance={"origin": "test"}, cycle=3)
        self.assertEqual(p.kind, "text")
        self.assertEqual(p.content, "hello")
        self.assertEqual(p.source, "user")
        self.assertEqual(p.confidence, 0.7)
        self.assertEqual(p.cycle, 3)
        self.assertEqual(p.status, "new")
        self.assertEqual(p.provenance["origin"], "test")
        self.assertEqual(p.provenance["source"], "user")
        self.assertEqual(p.provenance["kind"], "text")

    def test_confidence_clamped(self):
        self.assertEqual(Percept("text", "x", "u", confidence=2.0).confidence, 1.0)
        self.assertEqual(Percept("text", "x", "u", confidence=-1.0).confidence, 0.0)

    def test_to_dict_round_trips_provenance(self):
        p = Percept(kind="image", content={}, source="stub",
                    provenance={"image_id": "desk.png"})
        d = p.to_dict()
        self.assertEqual(d["provenance"]["image_id"], "desk.png")
        self.assertEqual(d["kind"], "image")


class TestBusText(unittest.TestCase):
    def test_perceive_text_returns_thought_and_queues_it(self):
        b = Brain()
        t = b.perceive_text("go to the lab", source="user", urgency=0.8)
        self.assertEqual(t.content, "go to the lab")
        self.assertEqual(t.kind, "percept")
        self.assertEqual(t.source, "user")
        self.assertEqual(t.urgency, 0.8)
        self.assertIn(t, b._incoming)

    def test_perceive_text_reaches_working_memory_on_step(self):
        b = Brain()
        b.perceive_text("go to the lab")
        b.step(None)
        self.assertTrue(any(str(t.content) == "go to the lab"
                            for t in b.wm.contents()))

    def test_unified_perceive_preserves_provenance_and_confidence(self):
        b = Brain()
        p = b.perceive("text", "hello world", source="user", confidence=0.7)
        self.assertEqual(p.kind, "text")
        self.assertEqual(p.confidence, 0.7)
        self.assertEqual(p.provenance["source"], "user")
        self.assertEqual(p.status, "admitted")
        self.assertEqual(p.content.content, "hello world")
        self.assertIn(p, b.perception_bus.percepts)

    def test_unknown_kind_raises(self):
        b = Brain()
        with self.assertRaises(ValueError):
            b.perceive("telepathy", "hello")


class TestDocumentReading(unittest.TestCase):
    def test_single_document_stores_unverified_never_verified(self):
        self.assertEqual(TRUST["file"], 0.45)
        self.assertEqual(DOCUMENT_TRUST, 0.45)
        b = Brain()
        rep = b.read_text(PARIS_DOC, source="file:paris-notes.md")
        self.assertEqual(rep["stored"], 2)
        self.assertEqual(rep["corroborated"], 0)
        self.assertEqual(rep["duplicate"], 0)
        self.assertEqual(rep["conflict"], 0)
        self.assertEqual(rep["ignored_sentences"], [])
        fact = b.ltm.query("paris", "capital_of", "france")[0]
        self.assertEqual(fact.status, "unverified")
        self.assertEqual(fact.sources, {"file:paris-notes.md": 0.45})
        self.assertEqual(b.ask("paris", "capital_of").best[2], "unverified")

    def test_second_independent_document_verifies(self):
        b = Brain()
        b.read_text(PARIS_DOC, source="file:paris-notes.md")
        rep = b.read_text("Paris is the capital of France.", source="file:atlas.md")
        self.assertEqual(rep["corroborated"], 1)
        self.assertEqual(rep["stored"], 0)
        fact = b.ltm.query("paris", "capital_of", "france")[0]
        self.assertEqual(fact.status, "verified")
        self.assertEqual(sorted(fact.sources),
                         ["file:atlas.md", "file:paris-notes.md"])
        self.assertEqual(b.ask("paris", "capital_of").best[2], "verified")

    def test_duplicate_read_not_double_counted(self):
        b = Brain()
        b.read_text(PARIS_DOC, source="file:paris-notes.md")
        rep = b.read_text(PARIS_DOC, source="file:paris-notes.md")
        self.assertEqual(rep["duplicate"], 2)
        self.assertEqual(rep["stored"], 0)
        self.assertEqual(rep["corroborated"], 0)
        fact = b.ltm.query("paris", "capital_of", "france")[0]
        self.assertEqual(len(fact.sources), 1)

    def test_conflict_becomes_disputed(self):
        b = Brain()
        b.read_text(PARIS_DOC, source="file:paris-notes.md")
        rep = b.read_text("Paris is the capital of Germany.",
                          source="file:other.md")
        self.assertEqual(rep["conflict"], 1)
        ans = b.ask("paris", "capital_of")
        self.assertEqual(ans.answers[0][2], "disputed")
        self.assertIn("disputed", ans.message.lower())

    def test_ignored_sentences_reported_and_sentence_kept_per_claim(self):
        b = Brain()
        rep = b.read_text("Paris is the capital of France. The sky is very blue today.",
                          source="file:notes.md")
        self.assertEqual(rep["stored"], 1)
        self.assertEqual(rep["ignored_sentences"],
                         ["The sky is very blue today."])
        self.assertEqual(rep["claims"][0]["sentence"],
                         "Paris is the capital of France.")
        self.assertEqual(rep["claims"][0]["subject"], "paris")

    def test_markdown_headings_and_lists_stripped(self):
        b = Brain()
        rep = b.read_document(_write_tmp(".md", PARIS_MD))
        self.assertEqual(rep["stored"], 2)
        self.assertEqual(rep["ignored_sentences"],
                         ["The sky is very blue today."])
        self.assertEqual(b.ask("mars", "is_a").best[0], "planet")

    def test_read_document_via_bus_and_brain_event(self):
        b = Brain()
        path = _write_tmp(".txt", PARIS_DOC)
        rep = b.read_document(path)
        self.assertEqual(rep["stored"], 2)
        self.assertEqual(rep["source"], f"file:{os.path.basename(path)}")
        self.assertTrue(any(e["kind"] == "document_read" for e in b.events))
        self.assertEqual(len(b.perception_bus.percepts), 1)
        self.assertEqual(b.perception_bus.percepts[0].kind, "document")

    def test_document_via_unified_perceive(self):
        b = Brain()
        p = b.perceive("document", {"text": PARIS_DOC,
                                    "source": "file:a.md"})
        self.assertEqual(p.kind, "document")
        self.assertEqual(p.content["stored"], 2)
        self.assertEqual(p.provenance["source_id"], "file:a.md")

    def test_missing_file_fails_cleanly_no_partial_claims(self):
        b = Brain()
        rep = b.read_document("/nonexistent/nope.txt")
        self.assertIn("error", rep)
        self.assertEqual(rep["stored"], 0)
        self.assertEqual(b.ltm.all_facts(), [])
        self.assertTrue(any(e["kind"] == "document_read" for e in b.events))

    def test_oversize_file_fails_cleanly_no_partial_claims(self):
        b = Brain()
        path = _write_tmp(".txt", "Paris is the capital of France. " * 5000)
        rep = b.read_document(path)
        self.assertIn("error", rep)
        self.assertEqual(rep["stored"], 0)
        self.assertEqual(b.ltm.all_facts(), [])

    def test_unsupported_extension_fails_cleanly(self):
        b = Brain()
        rep = b.read_document(_write_tmp(".pdf", PARIS_DOC))
        self.assertIn("error", rep)
        self.assertEqual(b.ltm.all_facts(), [])

    def test_read_text_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_text_file("/nonexistent/nope.txt")
        with self.assertRaises(ValueError):
            read_text_file(_write_tmp(".pdf", "x"))
        with self.assertRaises(ValueError):
            read_text_file(_write_tmp(".txt", "x" * (64 * 1024 + 1)))

    def test_document_reader_direct(self):
        from ab.core.memory import LongTermMemory
        reader = DocumentReader(LongTermMemory(), cycle=5)
        rep = reader.read_text(PARIS_DOC, source="file:direct.md")
        self.assertEqual(rep["stored"], 2)
        self.assertEqual(rep["source"], "file:direct.md")
        self.assertEqual(rep["claims"][0]["result"], "stored")

    def test_unknown_still_unknown_after_reading(self):
        b = Brain()
        b.read_text(PARIS_DOC, source="file:paris-notes.md")
        self.assertFalse(b.ask("zorblax", "capital_of").known)


class TestOcrThroughBus(unittest.TestCase):
    def _brain_with_words(self, words):
        ocr = FakeWordOcr({"sign.png": words})
        return Brain(ocr=ocr)

    def test_gate_still_enforced_through_bus(self):
        b = self._brain_with_words(
            [(w, 0.9) for w in "Paris is the capital of France".split()]
            + [("blurry", 0.3)])
        out = b.perceive_image("sign.png")
        self.assertEqual(out["text"], "Paris is the capital of France")
        self.assertEqual(len(out["uncertain"]), 1)
        self.assertEqual(out["uncertain"][0]["text"], "blurry")
        self.assertEqual(len(out["stored"]), 1)
        self.assertEqual(out["stored"][0][2], "unverified")
        # Uncertain word logged, never in WM text or LTM.
        self.assertTrue(any(e["kind"] == "ocr_uncertain" and "blurry" in e["detail"]
                            for e in b.events))
        self.assertFalse(any("blurry" in str(t.content) for t in b._incoming))
        self.assertEqual(b.ltm.query("blurry", "capital_of"), [])

    def test_threshold_boundary_0_6_admitted_0_59_uncertain(self):
        admitted = self._brain_with_words([(w, 0.6) for w in "Mars is a planet".split()])
        out = admitted.perceive_image("sign.png")
        self.assertEqual(out["text"], "Mars is a planet")
        self.assertEqual(out["uncertain"], [])
        rejected = self._brain_with_words([(w, 0.59) for w in "Mars is a planet".split()])
        out2 = rejected.perceive_image("sign.png")
        self.assertEqual(out2["text"], "")
        self.assertEqual(len(out2["uncertain"]), 4)
        self.assertEqual(out2["stored"], [])
        self.assertEqual(len(out2["rejected"]), 1)  # claim from uncertain text only

    def test_ocr_claim_never_verifies_alone(self):
        b = self._brain_with_words(
            [(w, 0.9) for w in "Paris is the capital of France".split()])
        b.perceive_image("sign.png")
        fact = b.ltm.query("paris", "capital_of", "france")[0]
        self.assertEqual(fact.status, "unverified")
        self.assertEqual(fact.sources, {"ocr:sign.png": 0.5})

    def test_no_ocr_engine_raises(self):
        with self.assertRaises(OcrUnavailable):
            Brain().perceive_image("sign.png")

    def test_ocr_percept_on_bus(self):
        b = self._brain_with_words([(w, 0.9) for w in "Mars is a planet".split()])
        p = b.perceive("ocr", "sign.png", source="ocr:sign.png")
        self.assertEqual(p.kind, "ocr")
        self.assertEqual(p.status, "admitted")
        self.assertEqual(p.provenance["engine"], "fake-words")
        self.assertEqual(p.content["text"], "Mars is a planet")


class TestImagePercept(unittest.TestCase):
    def _record(self):
        return make_image_record(
            [{"name": "cup", "properties": {"color": "red"},
              "position": {"x": 10, "y": 20}, "confidence": 0.9},
             {"name": "plate", "properties": {"color": "white"},
              "position": {"x": 40, "y": 20}, "confidence": 0.85}],
            [{"subject": "cup", "relation": "left_of", "object": "plate",
              "confidence": 0.9}],
            source="stub", image_id="desk.png")

    def test_image_objects_become_observed_entities(self):
        b = Brain()
        p = b.perceive("image", self._record(), source="stub")
        self.assertEqual(p.status, "admitted")
        self.assertEqual(p.provenance["image_id"], "desk.png")
        cup = b.world_model.entity("cup")
        self.assertIsNotNone(cup)
        self.assertEqual(cup.properties["color"].value, "red")
        self.assertEqual(cup.properties["color"].status, EpistemicStatus.observed)
        self.assertEqual(cup.properties["position"].value, {"x": 10, "y": 20})
        self.assertEqual(b.world_model.status_of("cup", "color"),
                         EpistemicStatus.observed)

    def test_image_spatial_relations(self):
        b = Brain()
        b.perceive("image", self._record(), source="stub")
        rels = b.world_model.relations(subject="cup", relation="left_of")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].object, "plate")
        self.assertEqual(rels[0].status, EpistemicStatus.observed)

    def test_image_record_is_not_flattened_to_text(self):
        b = Brain()
        b.perceive("image", self._record(), source="stub")
        self.assertEqual(b._incoming, [])  # no Thought: a record is data, not text
        self.assertTrue(any(e["kind"] == "image_percept" for e in b.events))

    def test_validate_image_record(self):
        rec = validate_image_record({"objects": [{"name": "  Red Cup  ", "confidence": 2.0}],
                                     "source": "s", "image_id": "i.png"})
        self.assertEqual(rec["objects"][0]["name"], "red cup")
        self.assertEqual(rec["objects"][0]["confidence"], 1.0)
        self.assertEqual(rec["relations"], [])
        with self.assertRaises(ValueError):
            validate_image_record("not a dict")

    def test_stub_image_sensor(self):
        sensor = StubImageSensor({"desk.png": self._record()})
        self.assertTrue(sensor.available())
        self.assertEqual(sensor.name, "stub")
        rec = sensor.describe_image("desk.png")
        self.assertEqual(len(rec["objects"]), 2)
        self.assertEqual(rec["image_id"], "desk.png")
        # Unknown path: empty record, no invented objects.
        self.assertEqual(sensor.describe_image("unknown.png")["objects"], [])
        # Deterministic: same input, same record.
        self.assertEqual(sensor.describe_image("desk.png"), rec)


class TestStructuredObservationAndEvent(unittest.TestCase):
    def test_structured_observation_to_state_and_world_model(self):
        b = Brain()
        p = b.perceive("observation", {"door_open": True}, source="sensor")
        self.assertEqual(p.status, "admitted")
        self.assertEqual(p.content, {"door_open": True})
        self.assertIs(b.world["door_open"], True)
        door = b.world_model.entity("door")
        self.assertIsNotNone(door)
        self.assertIs(door.properties["open"].value, True)
        self.assertEqual(door.properties["open"].status, EpistemicStatus.observed)
        self.assertEqual(p.provenance["source"], "sensor")

    def test_observation_payload_must_be_dict(self):
        with self.assertRaises(ValueError):
            Brain().perceive("observation", "door_open=True")

    def test_env_event_to_world_event_and_wm(self):
        b = Brain()
        p = b.perceive("event", {"kind": "door_slammed", "entity": "door",
                                 "key": "door_open", "old_value": True,
                                 "new_value": False, "cause": "exogenous"},
                       source="environment")
        self.assertEqual(p.status, "admitted")
        events = b.world_model.events()
        self.assertTrue(any(e.kind == "door_slammed" and e.entity == "door"
                            and e.new_value is False for e in events))
        self.assertTrue(any("door_slammed" in str(t.content) for t in b._incoming))
        self.assertEqual(p.content.kind, "door_slammed")


class TestSpecialist(unittest.TestCase):
    def test_absent_specialist_brain_works(self):
        b = Brain()
        self.assertIsNone(b.specialist)
        t = b.perceive_text("hello")
        self.assertEqual(t.content, "hello")
        # A bare image path with nobody to describe it is rejected,
        # never a crash and never invented objects.
        p = b.perceive("image", "photo.png", source="camera")
        self.assertEqual(p.status, "rejected")
        self.assertEqual(b.world_model.entities(), [])
        self.assertIn("no specialist", p.provenance["reason"])

    def test_fake_specialist_output_is_a_percept_with_provenance(self):
        fake = FakeSpecialist()
        b = Brain(specialist=fake)
        p = b.perceive("image", "photo.png", source="camera")
        self.assertEqual(p.status, "admitted")
        self.assertEqual(fake.calls, ["photo.png"])
        self.assertEqual(p.provenance["specialist"], "fake-vision")
        self.assertEqual(p.provenance["image_id"], "photo.png")
        cup = b.world_model.entity("cup")
        self.assertIsNotNone(cup)
        self.assertEqual(cup.properties["color"].value, "red")
        rels = b.world_model.relations(subject="cup", relation="left_of")
        self.assertEqual(len(rels), 1)
        # Specialist output is a percept, never reasoning: no Thought.
        self.assertEqual(b._incoming, [])

    def test_specialist_kind_routes_to_specialist(self):
        fake = FakeSpecialist()
        b = Brain(specialist=fake)
        p = b.perceive("specialist", "photo.png", source="camera")
        self.assertEqual(p.kind, "specialist")
        self.assertEqual(p.status, "admitted")
        self.assertIsNotNone(b.world_model.entity("cup"))

    def test_specialist_unavailable_is_rejected(self):
        class DownSpecialist(FakeSpecialist):
            def available(self):
                return False
        b = Brain(specialist=DownSpecialist())
        p = b.perceive("image", "photo.png")
        self.assertEqual(p.status, "rejected")
        self.assertEqual(b.world_model.entities(), [])


class TestBoundMethodFinding(unittest.TestCase):
    """Phase 7 (P7-11): bound-method tools run on the thread path with
    ``isolation_skipped``. Perception adds NO registered tool, so no
    perception path hits it — verified here, not assumed."""

    def test_no_perception_tool_is_registered(self):
        b = Brain()
        names = b.tools.names()
        for name in ("perceive", "perceive_text", "perceive_image",
                     "read_document", "read_text"):
            self.assertNotIn(name, names)

    def test_research_tool_is_isolated_partial(self):
        # Phase 11 §2.2 (W5): research is no longer a bound method —
        # it is a functools.partial over the module-level
        # research_fetch, so it can run in the forked child. A plain
        # function tool (calc) is still neither a bound method nor a
        # partial over brain state.
        # Brain registers research only when a researcher is given.
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>"}
        from ab.body.research import Researcher
        rb = Brain(researcher=Researcher(MemoryKnowledgeProvider(pages), None),
                   granted={"compute", "network"})
        self.assertFalse(inspect.ismethod(rb.tools._fns["research"]))
        self.assertIsInstance(rb.tools._fns["research"], functools.partial)
        self.assertFalse(inspect.ismethod(rb.tools._fns["calc"]))

    def test_filesystem_tools_are_isolated_partials(self):
        # Phase 11 §2.2 (W5): read_file / write_file are partials over
        # module-level scoped functions, not bound methods, so they
        # can run in the forked child.
        with tempfile.TemporaryDirectory() as scope:
            b = Brain(fs_scope=scope)
            for name in ("read_file", "write_file"):
                self.assertFalse(inspect.ismethod(b.tools._fns[name]))
                self.assertIsInstance(b.tools._fns[name], functools.partial)

    def test_research_call_isolated_runs_in_child(self):
        # Phase 11 §2.2 (W5): the research tool now runs in the
        # forked child when asked (isolated=True) — the bound-method
        # thread path with isolation_skipped is gone. The fixture
        # uses a memory provider, whose spec carries the pages
        # across the fork (a closure transport would not survive).
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>"}
        from ab.body.research import Researcher
        rb = Brain(researcher=Researcher(MemoryKnowledgeProvider(pages), None),
                   granted={"compute", "network"})
        res = rb.tools.call("research", {"urls": ["https://a.org/p"]},
                            granted={"compute", "network"}, isolated=True)
        self.assertTrue(res.ok)
        self.assertTrue(res.isolated)  # forked child, not the thread path
        audit = rb.tools.audit[-1]
        self.assertIsNone(audit["isolation_skipped"])

    def test_plain_function_tool_is_not_skipped(self):
        b = Brain()
        res = b.tools.call("calc", {"expression": "2+3"},
                           granted={"compute"}, isolated=True)
        self.assertTrue(res.ok)
        self.assertIsNone(b.tools.audit[-1]["isolation_skipped"])


if __name__ == "__main__":
    unittest.main()
