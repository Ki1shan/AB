"""Learned concepts (Phase 6, design §2, rules C-1 / C-2).

A Concept is a named abstraction over (property, value) features,
backed by positive and negative examples. Concepts are structural:
they are defined by feature shapes, never by literal world names.

Learning rules:
- C-1 (durable tool): a tool whose action succeeded >= 2 times
  with no observed break becomes a positive example of concept
  ``durable_tool``; a tool that broke after first use is a negative
  example.
- C-2 (requires-object): an action that failed >= 2 times in
  situations all lacking key K and succeeded when K was present
  yields concept ``requires_<K>``.

Confidence is Laplace-smoothed: (positives + 1) / (total + 2), so a
concept never claims 1.0 from a handful of examples.

Stdlib only, deterministic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1


def concept_id(name: str, features: dict) -> str:
    """Stable SHA-256 id of name + sorted defining features."""
    payload = json.dumps([name, sorted((k, str(v)) for k, v in features.items())],
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _confidence(positives: int, negatives: int) -> float:
    return (positives + 1) / (positives + negatives + 2)


@dataclass
class Concept:
    name: str
    features: dict = field(default_factory=dict)  # defining (property, value) pairs
    positives: list = field(default_factory=list)  # example feature dicts
    negatives: list = field(default_factory=list)  # example feature dicts
    confidence: float = 0.5
    provenance: str = ""
    id: str = ""

    def __post_init__(self):
        self.confidence = _confidence(len(self.positives), len(self.negatives))
        if not self.id:
            self.id = concept_id(self.name, self.features)

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name,
                "features": dict(self.features),
                "positives": [dict(p) for p in self.positives],
                "negatives": [dict(n) for n in self.negatives],
                "confidence": self.confidence,
                "provenance": self.provenance}

    @classmethod
    def from_json(cls, d: dict) -> "Concept":
        return cls(name=d["name"], features=dict(d.get("features", {})),
                   positives=[dict(p) for p in d.get("positives", [])],
                   negatives=[dict(n) for n in d.get("negatives", [])],
                   confidence=float(d.get("confidence", 0.5)),
                   provenance=d.get("provenance", ""),
                   id=d.get("id", ""))


class ConceptStore:
    """Store of learned concepts, keyed by concept name."""

    def __init__(self, concepts: list | None = None):
        self._concepts: dict[str, Concept] = {}
        for concept in concepts or []:
            self._concepts[concept.name] = concept

    # -- learning ---------------------------------------------------
    def add_example(self, name: str, features: dict, positive: bool,
                    provenance: str = "") -> Concept:
        """Add one example to a concept, creating it if needed.

        The defining features of a new concept come from its first
        *positive* example; a concept created by a negative example
        alone has no defining features yet and cannot match until a
        positive example arrives. Confidence is Laplace-smoothed
        from the example counts.
        """
        concept = self._concepts.get(name)
        if concept is None:
            concept = Concept(name=name, provenance=provenance or "")
            self._concepts[name] = concept
        if positive:
            if not concept.features:
                concept.features = dict(features)
            concept.positives.append(dict(features))
        else:
            concept.negatives.append(dict(features))
        if not concept.provenance and provenance:
            concept.provenance = provenance
        concept.confidence = _confidence(len(concept.positives),
                                         len(concept.negatives))
        concept.id = concept_id(concept.name, concept.features)
        return concept

    def learn_durable_tool(self, tool: str, successes: int,
                           broke: bool) -> bool:
        """Rule C-1: learn whether a tool is durable.

        ``broke`` (broke after first use) -> negative example of
        ``durable_tool``. ``successes >= 2`` with no break ->
        positive example. Fewer than 2 successes with no break is
        insufficient evidence: nothing is learned. Returns True
        when an example was recorded.
        """
        if broke:
            self.add_example(
                "durable_tool", {"breaks_after_first_use": True},
                positive=False,
                provenance=f"tool {tool} broke after first use")
            return True
        if successes >= 2:
            self.add_example(
                "durable_tool", {"breaks_after_first_use": False},
                positive=True,
                provenance=(f"tool {tool} succeeded {successes} times "
                            "without breaking"))
            return True
        return False

    def learn_requires(self, action: str, failed_situations: list,
                       succeeded_situations: list) -> list:
        """Rule C-2: learn a precondition concept ``requires_<K>``.

        For each key K absent in every failed situation (>= 2
        failures required) but present in a succeeded situation, a
        concept ``requires_<K>`` is created (or reinforced): the
        succeeded situations are positive examples. The failed
        situations are supporting evidence, not counterexamples --
        a counterexample would be succeeding *without* K, or
        failing *with* K present -- so no negative examples are
        recorded here. Returns the concept names created /
        reinforced, in sorted-key order.
        """
        if len(failed_situations) < 2 or not succeeded_situations:
            return []
        failed_keys = [set(s) for s in failed_situations]
        succeeded_keys = [set(s) for s in succeeded_situations]
        # Keys absent from ALL failures: complement over succeeded keys.
        all_succeeded_keys = set.union(*succeeded_keys) if succeeded_keys else set()
        candidate_keys = sorted(
            k for k in all_succeeded_keys
            if all(k not in fk for fk in failed_keys))
        created = []
        for key in candidate_keys:
            name = f"requires_{key}"
            for situation in succeeded_situations:
                if key in situation:
                    self.add_example(
                        name, {key: True}, positive=True,
                        provenance=(f"action {action} succeeded with "
                                    f"{key} present"))
            created.append(name)
        return created

    # -- querying ---------------------------------------------------
    def get(self, name: str) -> Concept | None:
        return self._concepts.get(name)

    def names(self) -> list:
        return sorted(self._concepts)

    def _match(self, name: str, features: dict) -> tuple:
        concept = self._concepts.get(name)
        if concept is None:
            return (False, 0.0)
        if not concept.positives:
            # No positive example: nothing is known to be an instance.
            return (False, concept.confidence)
        for key, value in concept.features.items():
            if key not in features:
                return (False, concept.confidence)
            if value is True:
                if not features[key]:
                    return (False, concept.confidence)
            elif features[key] != value:
                return (False, concept.confidence)
        return (True, concept.confidence)

    def matches(self, name: str, features: dict) -> tuple:
        """(bool, confidence): do ``features`` satisfy concept ``name``?

        A defining value of True means "key present and truthy";
        any other defining value must match exactly.
        """
        return self._match(name, features)

    def is_a(self, entity_features: dict, concept_name: str) -> tuple:
        """(bool, confidence): is the entity described by
        ``entity_features`` an instance of ``concept_name``?"""
        return self._match(concept_name, entity_features)

    # -- persistence ------------------------------------------------
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "concepts": [c.to_json() for c in self._concepts.values()]}

    @classmethod
    def from_json(cls, data: dict) -> "ConceptStore":
        if not isinstance(data, dict):
            raise ValueError("concept store data must be a dict")
        return cls(concepts=[Concept.from_json(c)
                             for c in data.get("concepts", [])])

