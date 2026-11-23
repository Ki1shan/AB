"""Structured image records (Phase 8, P8-3): an image is a record, not a string.

Shape::

    {objects: [{name, properties, position, confidence}],
     relations: [{subject, relation, object, confidence}],
     source, image_id}

An image is never flattened to a bare string and fed to
``perceive_text``. Objects enter the world model as observed entities
with properties and spatial relations; the record may come from a
:class:`StubImageSensor` (deterministic, for tests/demo) or a
specialist model (see ``ab/perception/specialist.py``).
"""
from __future__ import annotations

import copy

from ..core.vocab import normalize_text


def _clamp_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def validate_image_record(record) -> dict:
    """Normalise a raw image record into the canonical shape.

    Names are normalised (lowercase, collapsed whitespace); confidences
    are clamped to 0..1; missing ``objects``/``relations`` become empty
    lists; unknown extra keys are dropped. Raises ``ValueError`` when
    ``record`` is not a dict, and ``TypeError``-style ``ValueError``
    when an object has no usable name — a record with no identifiable
    object is a caller error, not an empty perception.
    """
    if not isinstance(record, dict):
        raise ValueError(f"image record must be a dict, got {type(record).__name__}")
    objects = []
    for obj in record.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        name = normalize_text(str(obj.get("name", "")))
        if not name:
            continue
        props = obj.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        position = obj.get("position")
        if position is not None and not isinstance(position, dict):
            position = {"value": position}
        objects.append({"name": name, "properties": dict(props),
                        "position": copy.deepcopy(position),
                        "confidence": _clamp_confidence(obj.get("confidence", 1.0))})
    relations = []
    for rel in record.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        subject = normalize_text(str(rel.get("subject", "")))
        relation = normalize_text(str(rel.get("relation", "")))
        obj = rel.get("object")
        if not subject or not relation or obj is None:
            continue
        relations.append({"subject": subject, "relation": relation,
                          "object": obj,
                          "confidence": _clamp_confidence(rel.get("confidence", 1.0))})
    return {"objects": objects, "relations": relations,
            "source": str(record.get("source", "image")),
            "image_id": str(record.get("image_id", record.get("path", "")))}


def make_image_record(objects, relations=None, source: str = "image",
                      image_id: str = "") -> dict:
    """Build a structured image record (validated on construction)."""
    return validate_image_record({"objects": list(objects or []),
                                  "relations": list(relations or []),
                                  "source": source, "image_id": image_id})


class StubImageSensor:
    """Deterministic stand-in for tests/demo: maps a path to a canned
    structured image record. ``available()`` is always True; an
    unknown path yields an empty record for that path (no crash, no
    invented objects). Never used as reasoning — its output is a
    percept like any other."""

    name = "stub"

    def __init__(self, table: dict[str, dict] | None = None):
        self.table = {str(k): copy.deepcopy(v) for k, v in dict(table or {}).items()}

    def available(self) -> bool:
        return True

    def describe_image(self, path: str) -> dict:
        """Return the canned record for ``path`` (validated), or an
        empty record naming the path when nothing is canned for it."""
        record = self.table.get(str(path))
        if record is None:
            return {"objects": [], "relations": [],
                    "source": "stub", "image_id": str(path)}
        out = copy.deepcopy(record)
        out.setdefault("source", "stub")
        out.setdefault("image_id", str(path))
        return validate_image_record(out)
