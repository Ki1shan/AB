"""Consolidation queue (Phase 6, design §2).

A candidate is generalized knowledge (a pattern, a concept, or an
induced-rule proposal) waiting for an explicit, auditable promotion
step. Nothing is promoted silently: ``promote()`` moves a candidate
to durable knowledge only when its supporting evidence reaches the
documented bar (>= 3 episodes across >= 2 distinct situations) or a
human forces it, and it records what justified the promotion.
This module emits no brain events itself -- the Brain logs
``consolidated`` / ``consolidation_rejected`` events around these
calls, so the audit trail lives in one place.

Candidate dicts (all JSON-safe)::
    {"id": stable SHA-256 of canonical (kind, content),
     "kind": "pattern" | "concept" | "rule",
     "content": the generalized payload (dict),
     "evidence": [episode summaries],
     "situations": count of distinct situations,
     "status": "candidate" | "promoted" | "rejected",
     "provenance": str naming the source episodes / mechanism,
     "promoted_at": cycle of promotion (None until promoted),
     "promoted_evidence": evidence count at promotion,
     "promoted_situations": situation count at promotion,
     "rejection_reason": str ("" until rejected)}

Stdlib only, deterministic.
"""
from __future__ import annotations

import copy
import hashlib
import json

SCHEMA_VERSION = 1

# Promotion bar (design §2): >= 3 supporting episodes across
# >= 2 distinct situations. Below the bar a candidate stays a
# candidate until more evidence arrives or a human promotes it.
PROMOTE_MIN_EVIDENCE = 3
PROMOTE_MIN_SITUATIONS = 2

VALID_KINDS = ("pattern", "concept", "rule")
VALID_STATUSES = ("candidate", "promoted", "rejected")


def candidate_id(kind: str, content: dict) -> str:
    """Stable SHA-256 id for a candidate: canonical (kind, content).

    Two candidates with the same kind and content are the same
    candidate, however many times they are submitted.
    """
    payload = json.dumps([kind, content], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _as_evidence(evidence) -> list:
    if evidence is None:
        return []
    if isinstance(evidence, list):
        return list(evidence)
    return [evidence]


def _as_situation_count(situations) -> int:
    if isinstance(situations, int):
        return situations
    if isinstance(situations, (list, tuple, set, dict)):
        return len(situations)
    return int(situations) if situations is not None else 0


class Consolidator:
    """Review queue for candidate knowledge (design §2)."""

    def __init__(self, candidates: list | None = None):
        # id -> candidate dict, insertion ordered.
        self._candidates: dict[str, dict] = {}
        for cand in candidates or []:
            self._candidates[cand["id"]] = copy.deepcopy(cand)

    # -- submission -------------------------------------------------
    def add_candidate(self, kind: str, content: dict, evidence,
                      situations, provenance: str = "") -> str:
        """Submit candidate knowledge; returns its stable id.

        Dedupe by id: submitting the same (kind, content) again
        returns the existing id without creating a second entry.
        A rejected candidate is never re-added: the call returns
        its id and the candidate stays rejected.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown candidate kind: {kind!r}")
        cid = candidate_id(kind, content)
        if cid in self._candidates:
            return cid
        self._candidates[cid] = {
            "id": cid,
            "kind": kind,
            "content": copy.deepcopy(content),
            "evidence": _as_evidence(evidence),
            "situations": _as_situation_count(situations),
            "status": "candidate",
            "provenance": provenance or "",
            "promoted_at": None,
            "promoted_evidence": None,
            "promoted_situations": None,
            "rejection_reason": "",
        }
        return cid

    # -- review -----------------------------------------------------
    def review(self) -> list:
        """Candidates awaiting a decision (status == 'candidate')."""
        return [copy.deepcopy(c) for c in self._candidates.values()
                if c["status"] == "candidate"]

    def get(self, candidate_id_: str) -> dict | None:
        """One candidate by id, or None. Returns a copy."""
        cand = self._candidates.get(candidate_id_)
        return copy.deepcopy(cand) if cand is not None else None

    # -- decisions --------------------------------------------------
    def promote(self, candidate_id_: str, force: bool = False,
                cycle: int | None = None) -> bool:
        """Promote a candidate to durable knowledge.

        Only when evidence >= 3 episodes AND >= 2 distinct
        situations, or when ``force=True`` (explicit human call).
        On promotion the status, the promotion cycle, and the
        evidence that justified it are recorded. Returns True on
        success (including re-promoting an already-promoted
        candidate); False for unknown ids, rejected candidates,
        or candidates below the bar without ``force``.
        """
        cand = self._candidates.get(candidate_id_)
        if cand is None or cand["status"] == "rejected":
            return False
        if cand["status"] == "promoted":
            return True
        meets_bar = (len(cand["evidence"]) >= PROMOTE_MIN_EVIDENCE
                     and cand["situations"] >= PROMOTE_MIN_SITUATIONS)
        if not (meets_bar or force):
            return False
        cand["status"] = "promoted"
        cand["promoted_at"] = cycle
        cand["promoted_evidence"] = len(cand["evidence"])
        cand["promoted_situations"] = cand["situations"]
        return True

    def reject(self, candidate_id_: str, reason: str = "") -> bool:
        """Retire a candidate with a reason. Returns False for
        unknown ids; True otherwise (idempotent). A rejected
        candidate is never re-added by :meth:`add_candidate`."""
        cand = self._candidates.get(candidate_id_)
        if cand is None:
            return False
        cand["status"] = "rejected"
        cand["rejection_reason"] = reason or ""
        return True

    # -- accessors --------------------------------------------------
    def candidates(self) -> list:
        """All candidates still awaiting a decision."""
        return self.review()

    def promoted(self) -> list:
        return [copy.deepcopy(c) for c in self._candidates.values()
                if c["status"] == "promoted"]

    def rejected(self) -> list:
        return [copy.deepcopy(c) for c in self._candidates.values()
                if c["status"] == "rejected"]

    def all(self) -> list:
        """Every candidate in insertion order, any status."""
        return [copy.deepcopy(c) for c in self._candidates.values()]

    def __len__(self) -> int:
        return len(self._candidates)

    # -- persistence ------------------------------------------------
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "candidates": self.all()}

    @classmethod
    def from_json(cls, data: dict) -> "Consolidator":
        if not isinstance(data, dict):
            raise ValueError("consolidation data must be a dict")
        return cls(candidates=data.get("candidates", []))

