"""PerceptionBus (Phase 8, P8-1): the one path every source goes through.

The bus normalises each source into a :class:`Percept` and routes it:

- ``text`` -> Thought into the brain's incoming queue (WM via ``_attend``)
- ``document`` -> claims into LTM via the existing two-source/teacher
  admission path (``DocumentReader`` / ``LongTermMemory.add``)
- ``ocr`` -> the Phase 3 confidence gate (threshold 0.6): admitted
  words become a Thought, uncertain words become ``ocr_uncertain``
  events and never reach WM/LTM
- ``image`` / ``specialist`` -> structured record: objects become
  world-model observed entities (properties + spatial relations)
- ``observation`` -> structured {key: value} into state + world model
- ``event`` -> a world-model ``WorldEvent`` + a Thought into WM

The bus never reasons: specialist output arrives as a percept with
provenance, never as reasoning. The bus is brain-owned and calls back
into the brain's own admission helpers (``_incoming``,
``_set_state``, ``_event``); it holds no reasoning state of its own.
"""
from __future__ import annotations

import os

from ..body.ocr import OcrSensor, OcrUnavailable
from ..body.reading import DocumentReader
from ..core.thought import Thought
from ..core.vocab import normalize_text, tokenize
from ..world.model import EpistemicStatus
from .image import validate_image_record
from .percept import Percept

# Kinds the bus routes. Anything else is a caller error, raised
# (never silently dropped) so a misspelled kind cannot vanish.
KINDS = ("text", "document", "ocr", "image", "observation", "event",
         "specialist")


class PerceptionBus:
    """Owns the brain's perception history (``percepts``) and routes
    every incoming percept. Constructed once per Brain; the brain is
    read at call time (``brain.specialist``, ``brain.ocr`` ...), so a
    specialist attached after construction is honoured."""

    def __init__(self, brain):
        self._brain = brain
        self.percepts: list[Percept] = []

    # ------------------------------------------------------------------ entry
    def perceive(self, kind: str, payload, source: str = "user",
                 confidence: float = 1.0, urgency: float = 0.5,
                 cycle: int | None = None) -> Percept:
        """Normalise ``payload`` from ``source`` into a Percept and route it.

        Returns the Percept (status ``"admitted"`` when routed,
        ``"rejected"`` when the bus could not route it — e.g. an image
        path with no specialist attached — with the reason in
        provenance). Unknown kinds raise ``ValueError``.
        """
        kind = str(kind).lower()
        if kind not in KINDS:
            raise ValueError(f"unknown percept kind: {kind!r} (known: {sorted(KINDS)})")
        brain = self._brain
        cycle = brain.cycle if cycle is None else int(cycle)
        provenance: dict = {"source": source, "kind": kind}
        handler = getattr(self, f"_route_{kind}")
        content, status, extra_prov, conf = handler(payload, source, confidence,
                                                    urgency, cycle)
        provenance.update(extra_prov or {})
        percept = Percept(kind=kind, content=content, source=source,
                          confidence=conf, provenance=provenance,
                          cycle=cycle, status=status)
        self.percepts.append(percept)
        return percept

    # ------------------------------------------------------------------ text
    def _route_text(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        text = str(payload)
        thought = Thought(text, "percept", source, urgency, 1.0,
                          tokenize(text), cycle)
        brain._incoming.append(thought)
        return thought, "admitted", {"text_length": len(text)}, confidence

    # ------------------------------------------------------------------ document
    def _route_document(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        if isinstance(payload, dict):
            text = str(payload.get("text", ""))
            doc_source = str(payload.get("source", source))
        else:
            text = str(payload)
            doc_source = source
        reader = DocumentReader(brain.ltm, brain.extractor, cycle=cycle)
        report = reader.read_text(text, source=doc_source)
        # File trust 0.45 (P8-2): the report carries it so callers can
        # see a single document never verifies a fact.
        report["trust"] = 0.45
        return report, "admitted", {"source_id": doc_source}, confidence

    # ------------------------------------------------------------------ OCR
    def _route_ocr(self, payload, source, confidence, urgency, cycle):
        """The Phase 3 gate, unchanged, now behind the bus: only
        admitted words become a Thought or claims; uncertain words are
        logged as ``ocr_uncertain`` events with provenance and never
        reach WM or LTM. Claims enter LTM only as an ``ocr`` source
        (trust 0.5), so one OCR source alone can never verify a fact."""
        brain = self._brain
        path = str(payload)
        if brain.ocr is None or not brain.ocr.available():
            raise OcrUnavailable("no OCR engine available")
        gated = OcrSensor(brain.ocr, threshold=brain.ocr_min_confidence).gate(path)
        admitted_text = gated["admitted_text"]
        uncertain = gated["uncertain"]
        words = gated["admitted"] + uncertain
        conf = (round(sum(float(w["confidence"]) for w in words) / len(words), 3)
                if words else 0.0)
        thought = None
        if admitted_text:
            mean_admitted = round(
                sum(float(w["confidence"]) for w in gated["admitted"])
                / len(gated["admitted"]), 3)
            thought = Thought(admitted_text, "percept", f"ocr:{gated['engine']}", 0.5,
                              mean_admitted, tokenize(admitted_text), cycle)
            brain._incoming.append(thought)
        for w in uncertain:
            brain._event("ocr_uncertain",
                         f"word={w['text']!r} image={path} bbox={w.get('bbox')} "
                         f"confidence={w['confidence']}")
        stored, rejected = [], []
        for c in brain.extractor.extract(admitted_text):
            r, f = brain.ltm.add(c.subject, c.relation, c.value,
                                 f"ocr:{os.path.basename(path)}", cycle)
            stored.append((c, r.value, f.status))
        uncertain_text = " ".join(w["text"] for w in uncertain)
        for c in brain.extractor.extract(uncertain_text):
            rejected.append(c)
        brain._event("ocr", f"read {len(admitted_text)} admitted chars at confidence {conf}; "
                            f"{len(stored)} claim(s) stored, {len(rejected)} rejected as low-confidence, "
                            f"{len(uncertain)} word(s) uncertain")
        result = {"text": admitted_text, "confidence": conf, "stored": stored,
                  "rejected": rejected, "uncertain": uncertain,
                  "thought": thought}
        return result, "admitted", {"engine": gated["engine"], "image": path}, conf

    # ------------------------------------------------------------------ image
    def _route_image(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        if isinstance(payload, str):
            # A path needs a specialist to turn pixels into a record.
            return self._route_specialist(payload, source, confidence, urgency, cycle)
        record = validate_image_record(payload)
        return self._apply_image_record(record, source, confidence, cycle)

    def _route_specialist(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        specialist = getattr(brain, "specialist", None)
        path = str(payload)
        if specialist is None or not specialist.available():
            # No specialist attached: the brain still works fully with
            # the stub/built-in paths (P8-4). A bare path with nobody
            # to describe it is a rejected percept, never a crash and
            # never invented objects.
            return {}, "rejected", {"image": path,
                                    "reason": "no specialist attached"}, confidence
        record = validate_image_record(specialist.describe_image(path))
        record.setdefault("source", getattr(specialist, "name", "specialist"))
        content, status, extra, conf = self._apply_image_record(
            record, source, confidence, cycle)
        extra = dict(extra)
        extra["specialist"] = getattr(specialist, "name", "specialist")
        extra["image"] = path
        return content, status, extra, conf

    def _apply_image_record(self, record, source, confidence, cycle):
        """Objects -> world-model observed entities; relations ->
        world-model spatial relations. The record is data, never
        reasoning: nothing here becomes a Thought or an inference."""
        brain = self._brain
        model = brain.world_model
        prov = str(record.get("source", source))
        for obj in record["objects"]:
            entity = normalize_text(obj["name"])
            obj_conf = float(obj.get("confidence", 1.0))
            for key, value in dict(obj.get("properties") or {}).items():
                model.set_property(entity, str(key), value,
                                   status=EpistemicStatus.observed,
                                   confidence=obj_conf, provenance=prov,
                                   cycle=cycle, kind="object")
            if obj.get("position") is not None:
                model.set_property(entity, "position", obj["position"],
                                   status=EpistemicStatus.observed,
                                   confidence=obj_conf, provenance=prov,
                                   cycle=cycle, kind="object")
            # The entity exists even when it carries no properties:
            # record the observation itself.
            if not obj.get("properties") and obj.get("position") is None:
                model.set_property(entity, "observed", True,
                                   status=EpistemicStatus.observed,
                                   confidence=obj_conf, provenance=prov,
                                   cycle=cycle, kind="object")
        for rel in record["relations"]:
            model.add_relation(rel["subject"], rel["relation"], rel["object"],
                               status=EpistemicStatus.observed,
                               confidence=float(rel.get("confidence", 1.0)),
                               provenance=prov)
        brain._event("image_percept",
                     f"{len(record['objects'])} object(s), "
                     f"{len(record['relations'])} relation(s) from {prov} "
                     f"image_id={record.get('image_id', '')}")
        mean_conf = (round(sum(float(o.get("confidence", 1.0))
                              for o in record["objects"]) / len(record["objects"]), 3)
                       if record["objects"] else confidence)
        return record, "admitted", {"image_id": record.get("image_id", ""),
                                    "objects": len(record["objects"]),
                                    "relations": len(record["relations"])}, mean_conf

    # ------------------------------------------------------- structured observation
    def _route_observation(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        if not isinstance(payload, dict):
            raise ValueError("structured observation payload must be a dict of {key: value}")
        applied = {}
        for key, value in payload.items():
            key = str(key)
            brain._set_state(key, value, source=f"{source}:{key}",
                             confidence=confidence, provenance=f"percept:{source}",
                             entry_type="observation")
            brain.world_model.observe(key, value, cycle, source=source)
            applied[key] = value
        return applied, "admitted", {"keys": sorted(applied)}, confidence

    # ------------------------------------------------------------------ env event
    def _route_event(self, payload, source, confidence, urgency, cycle):
        brain = self._brain
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a dict")
        # Phase 9 §4.6: env events go through the world model's
        # PUBLIC record_event API (the bus is the sanctioned writer
        # of env events into the model log — P8-1: env events ->
        # WorldEvent + WM). No direct _events append here.
        event = brain.world_model.record_event(
            kind=str(payload.get("kind", "event")),
            entity=str(payload.get("entity", "")),
            key=str(payload.get("key", "")),
            old_value=payload.get("old_value"),
            new_value=payload.get("new_value"),
            cause=str(payload.get("cause", "observation")),
            cycle=cycle)
        thought = Thought(
            f"{event.kind}: {event.entity} {event.key}={event.new_value!r}"
            if event.key else f"{event.kind}: {event.entity}",
            "percept", source, urgency, confidence,
            tokenize(str(event.new_value)) if event.new_value is not None else set(),
            cycle)
        brain._incoming.append(thought)
        return event, "admitted", {"event_kind": event.kind,
                                   "entity": event.entity}, confidence
