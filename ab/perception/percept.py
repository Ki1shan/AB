"""Perception (Phase 8, P8-1): one bus every source goes through.

A :class:`Percept` is the normalised unit: kind, content, source,
confidence, provenance, cycle, status. The :class:`PerceptionBus`
normalises each source into Percepts and routes them — text to
Thought/WM, document claims to LTM via the two-source admission path,
OCR admitted words to Thought/WM (uncertain words become
``ocr_uncertain`` events, never WM/LTM), image objects to world-model
observed entities, structured observations to state + world model,
env events to WorldEvent + WM. No source bypasses the bus. Every
percept keeps its confidence, provenance, source identity and
uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Percept:
    """One normalised perception.

    ``kind`` is one of ``text`` | ``document`` | ``ocr`` | ``image`` |
    ``observation`` | ``event`` | ``specialist``. ``content`` is the
    routed payload: a Thought (text/OCR), a reading report dict
    (document), the OCR result dict (ocr), the normalised image
    record (image/specialist), the applied observation dict
    (observation), or a WorldEvent (event). ``provenance`` always
    carries the source identity and whatever the source supplied
    (file name, image id, engine, specialist name); it is never
    dropped on routing.
    """

    kind: str
    content: object
    source: str
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict)
    cycle: int = 0
    status: str = "new"  # new | admitted | rejected

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if not isinstance(self.provenance, dict):
            self.provenance = {"detail": self.provenance}
        self.provenance.setdefault("source", self.source)
        self.provenance.setdefault("kind", self.kind)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "content": self.content,
                "source": self.source, "confidence": self.confidence,
                "provenance": dict(self.provenance), "cycle": self.cycle,
                "status": self.status}
