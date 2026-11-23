"""Inference over stored facts, with proof chains. Derives only from verified facts."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core.memory import LongTermMemory, Fact
from ..core.vocab import normalize_relation, normalize_text


@dataclass
class Proof:
    steps: list            # list[Fact]
    confidence: float

    def explain(self) -> str:
        return " -> ".join(f"{f.subject} {f.relation} {f.value}" for f in self.steps)


class Inference:
    def __init__(self, ltm: LongTermMemory, transitive=("is_a", "part_of", "located_in")):
        self.ltm = ltm
        self.transitive = {normalize_relation(r) for r in transitive}

    def closure(self, subject, relation, max_depth: int = 6) -> list[tuple[str, Proof]]:
        """Everything reachable from subject via a transitive relation (verified facts only)."""
        r = normalize_relation(relation)
        if r not in self.transitive:
            return []
        found, seen = [], {normalize_text(subject)}
        queue = deque([(normalize_text(subject), [])])
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for f in self.ltm.query(node, r, statuses=("verified",)):
                if f.value in seen:
                    continue
                seen.add(f.value)
                new_path = path + [f]
                found.append((f.value, Proof(new_path, min(x.confidence for x in new_path))))
                queue.append((f.value, new_path))
        return found

    def prove(self, subject, relation, value) -> Proof | None:
        direct = self.ltm.query(subject, relation, value, statuses=("verified",))
        if direct:
            return Proof([direct[0]], direct[0].confidence)
        for v, proof in self.closure(subject, relation):
            if v == normalize_text(value):
                return proof
        return None

    @staticmethod
    def proof_confidence(steps) -> float:
        """Confidence of a proof chain: the weakest premise, discounted
        5% per extra step. A 1-step chain keeps full confidence."""
        if not steps:
            return 0.0
        weakest = min(x.confidence for x in steps)
        return weakest * (0.95 ** (len(steps) - 1))

    def closure_multi(self, subject, relations=None, max_depth: int = 6
                      ) -> list[tuple[str, Proof, tuple]]:
        """Everything reachable from subject via MIXED-relation chains
        (verified facts only). Each hit is (value, proof, relation_path),
        where relation_path is the tuple of relations used, in order
        (e.g. ("is_a", "located_in") for cat -> mammal -> house).
        ``relations`` restricts the relations followed; None follows all
        transitive relations. Proof confidence uses proof_confidence()."""
        rels = sorted({normalize_relation(r) for r in
                       (relations if relations is not None else self.transitive)})
        if not rels:
            return []
        found, seen = [], {normalize_text(subject)}
        queue = deque([(normalize_text(subject), [], ())])
        while queue:
            node, path, rel_path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for r in rels:
                for f in self.ltm.query(node, r, statuses=("verified",)):
                    if f.value in seen:
                        continue
                    seen.add(f.value)
                    new_path = path + [f]
                    new_rel_path = rel_path + (r,)
                    found.append((f.value,
                                  Proof(new_path, self.proof_confidence(new_path)),
                                  new_rel_path))
                    queue.append((f.value, new_path, new_rel_path))
        return found

    @staticmethod
    def combine_proofs(proofs) -> float:
        """Noisy-or combination of independent proofs: 1 - prod(1 - c)
        over proof confidences, capped at the weakest premise confidence
        (min over all steps of all proofs) so combined confidence never
        exceeds what the premises support."""
        proofs = list(proofs or [])
        if not proofs:
            return 0.0
        combined = 1.0
        for p in proofs:
            combined *= (1.0 - p.confidence)
        combined = 1.0 - combined
        steps = [s for p in proofs for s in (p.steps or [])]
        if steps:
            cap = min(s.confidence for s in steps)
        else:
            cap = min(p.confidence for p in proofs)
        return min(combined, cap)
