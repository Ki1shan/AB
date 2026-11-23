"""Learning from experience (method 3): outcomes adjust action costs and rule reliability,
and repeated failures become *proposed* rules that wait for human approval.

Phase 3 (§3.1, audit gap A6): proposed rules are activatable, but ONLY by an
explicit human approval call. ``Learner.approve_rule(rule_id)`` moves a
pending rule to active (and records it in the learner's approved set, so the
planner can consult ``approved_rules()``); ``Learner.reject_rule(rule_id)``
marks it rejected so the same episodes never re-propose it. There is no
auto-approval path anywhere in this module: ``propose_rules()`` only ever
creates ``status="pending"`` rules.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..planning.planner import Planner, canonical_args
from ..reasoning.rules import Rule, RuleEngine


MAX_EPISODES = 500


def _feature_key_set(situation: dict) -> frozenset:
    """The *feature set* of a situation: its precondition keys only,
    not their values (Phase 6 §3.5). Two situations share a feature
    set when the same action was attempted with the same keys
    present, whatever the values were."""
    return frozenset(str(k) for k in (situation or {}))


def stable_rule_id(action: str, situation: dict, args_identity: str = "") -> str:
    """Deterministic id for a proposed avoid-rule (Phase 1, audit A G20).

    The v0.1 code used ``abs(hash(sig)) % 10000``; Python string hashing is
    randomised per process, so the same failure proposed a different id on
    every run and de-duplication across runs could not work. This id is a
    SHA-256 of the canonical JSON of (action, args identity, sorted
    situation), so the same (action, args, situation) always yields the
    same id in any process. Phase 10.5 (P10.5-2): the args identity is
    part of the payload, so two same-named variants that differ only
    in args (cut-zinc vs cut-tin) propose different rules. The
    default ``""`` keeps two-argument callers (empty-args actions)
    working unchanged.

    Note: the payload changed in Phase 10.5, so a rule id computed
    before 10.5 differs from the same (action, situation) after it —
    an approve/reject decision recorded in a pre-10.5 learner.json
    will not match the new id and the rule may be re-proposed as
    pending after a load. Same class of break as P9-1's variant
    keying; the alternative (a conditional payload) would have made
    the id scheme depend on emptiness rather than identity."""
    payload = json.dumps([action, args_identity,
                          sorted((k, str(v)) for k, v in situation.items())],
                         sort_keys=True)
    return f"avoid_{action}_{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


@dataclass
class Episode:
    cycle: int
    action: str
    situation: dict                  # state restricted to the action's preconditions
    ok: bool
    surprises: list = field(default_factory=list)
    # Phase 10.5 (P10.5-2): the semantic args the action ran with.
    # Default {} for backward compatibility (hand-built episodes and
    # legacy saves carry no args and attribute to empty-args actions).
    args: dict = field(default_factory=dict)


class Learner:
    def __init__(self, planner: Planner, rules: RuleEngine, proposal_threshold: int = 3,
                 max_episodes: int = MAX_EPISODES):
        self.planner, self.rules, self.threshold = planner, rules, proposal_threshold
        # Phase 6 §2: the episode cap is a constructor parameter
        # (default 500, unchanged) so the cap experiment can vary it.
        self.max_episodes = int(max_episodes)
        # Phase 6 §3: optional duck-typed hook, set by Brain
        # (``learner.on_record = callable``). Called with each
        # recorded Episode so the retriever index and the
        # prioritizer can be fed without this module importing
        # ab.learning. Default None: no hook, no behaviour change.
        self.on_record = None
        self.episodes: list[Episode] = []
        self._counts: dict[tuple, list[int]] = {}
        # Phase 3: explicit human decisions about proposed rules.
        # rule_id -> {"action": ..., "situation": {...}}; both persist in
        # learner.json so a decision survives save/load. Nothing here is
        # ever written by propose/record — only approve_rule/reject_rule
        # (or a sync from an out-of-band RuleEngine decision) touch them.
        self._approved: dict[str, dict] = {}
        self._rejected: dict[str, dict] = {}

    @staticmethod
    def _sig(situation: dict) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in situation.items()))

    @property
    def counts(self) -> dict:
        """Read-only copy of the (action, args-identity, situation) ->
        [successes, failures] counts (Phase 10.5 P10.5-2: the key
        gained the canonical semantic-args identity, so two variants
        of one action name that differ only in args count
        separately; empty-args actions keep the "" identity). The
        planner's stats view is derived from this (Phase 3 §2.4);
        the learner remains the single store."""
        return dict(self._counts)

    def episodes_for_induction(self) -> list[dict]:
        """Phase 4: the episodes as plain dicts for the reasoning
        engine's induction (``ReasoningEngine.induce``). Read-only —
        this exposes the episode history for generalisation and never
        mutates it. The episode cap (``max_episodes``, default
        ``MAX_EPISODES``) is unchanged and still enforced by
        ``record()``."""
        return [{"action": ep.action, "situation": dict(ep.situation),
                 "outcome": "success" if ep.ok else "failure"}
                for ep in self.episodes]

    def record(self, ep: Episode) -> None:
        self.episodes.append(ep)
        self.planner.record(ep.action, ep.ok)
        c = self._counts.setdefault(
            (ep.action, canonical_args(getattr(ep, "args", None) or {}),
             self._sig(ep.situation)), [0, 0])
        c[0 if ep.ok else 1] += 1
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
            self._rebuild_counts()
        # Phase 6: feed the retriever / prioritizer hooks (Brain-set,
        # duck-typed; absent by default). The hook is called
        # directly, so a broken hook fails loudly instead of
        # silently dropping learning updates.
        if callable(self.on_record):
            self.on_record(ep)

    def failures_for(self, action: str, situation: dict,
                     args: dict | None = None) -> int:
        """Phase 10.5 (P10.5-7): failures of one exact (action, args,
        situation) triple — the count behind the loop-plan rule.
        Reads the durable ``_counts`` (not the capped episode list),
        keyed by (action, canonical semantic args, frozen
        situation), so a variant's failures never count toward a
        sibling variant. ``args=None`` means empty args."""
        entry = self._counts.get(
            (action, canonical_args(args),
             self._sig(situation)))
        return int(entry[1]) if entry else 0

    def repeated_mistakes(self, action: str, situation: dict, args: dict | None = None) -> dict | None:
        """Phase 6 §3.5: feature-matched repeated failures.

        Groups episodes by (action, args identity, set of
        precondition keys) — the *feature set*, values ignored
        (Phase 10.5 P10.5-2: the args identity joined the key, so
        failures of one variant do not count as failures of a
        sibling variant). Returns ``{"action", "args", "features",
        "failures", "successes"}`` when the same action failed >= 2
        times in situations sharing the given feature set with 0
        successes in those situations, else None. The most-evidenced
        matching group wins; ties break by sorted feature list, so
        the result is deterministic. ``args=None`` means empty args
        (backward compatible for existing callers)."""
        features = _feature_key_set(situation)
        args_id = canonical_args(args)
        grouped: dict[tuple, list[int]] = {}
        for ep in self.episodes:
            if ep.action != action:
                continue
            if canonical_args(getattr(ep, "args", None) or {}) != args_id:
                continue
            key = (args_id, _feature_key_set(ep.situation))
            c = grouped.setdefault(key, [0, 0])
            c[0 if ep.ok else 1] += 1
        candidates = [(key, counts) for key, counts in grouped.items()
                      if key[1] == features and counts[1] >= 2 and counts[0] == 0]
        if not candidates:
            return None
        key, (ok, bad) = sorted(candidates, key=lambda kv: (-kv[1][1], sorted(kv[0][1])))[0]
        return {"action": action, "args": key[0], "features": sorted(key[1]),
                "failures": bad, "successes": ok}

    def _rebuild_counts(self) -> None:
        """Rebuild _counts from the surviving episodes (used after pruning)."""
        self._counts = {}
        for ep in self.episodes:
            c = self._counts.setdefault(
                (ep.action, canonical_args(getattr(ep, "args", None) or {}),
                 self._sig(ep.situation)), [0, 0])
            c[0 if ep.ok else 1] += 1

    def to_json(self) -> dict:
        """Serialize episodes (capped at max_episodes) and counts for persistence."""
        episodes = self.episodes[-self.max_episodes:]
        return {
            # Phase 6: the configured cap persists, so a brain saved
            # with a non-default cap keeps it on load.
            "max_episodes": int(self.max_episodes),
            "episodes": [
                {
                    "cycle": ep.cycle,
                    "action": ep.action,
                    "situation": dict(ep.situation),
                    "ok": ep.ok,
                    "surprises": list(ep.surprises),
                    # Phase 10.5: episodes carry the args they ran
                    # with, so variant-aware counts survive save/load.
                    "args": dict(getattr(ep, "args", None) or {}),
                }
                for ep in episodes
            ],
            "counts": [
                # Phase 10.5: the key gained the args identity:
                # [[action, args_identity, [[k, v], ...]], [ok, bad]].
                [[action, args_id, [[k, v] for k, v in sig]], [ok, bad]]
                for (action, args_id, sig), (ok, bad) in self._counts.items()
            ],
            # Phase 3: explicit approve/reject decisions persist here.
            "approved": [
                {"rule_id": rid, "action": d["action"], "situation": dict(d["situation"])}
                for rid, d in sorted(self._approved.items())
            ],
            "rejected": [
                {"rule_id": rid, "action": d["action"], "situation": dict(d["situation"])}
                for rid, d in sorted(self._rejected.items())
            ],
        }

    @classmethod
    def from_json(cls, data: dict, planner, rules) -> "Learner":
        """Restore a Learner from to_json() output.

        Missing keys default to empty; raises ValueError if "episodes" is
        present and not a list.
        """
        if not isinstance(data, dict):
            raise ValueError("Learner.from_json expects a dict")
        raw_episodes = data.get("episodes", [])
        if not isinstance(raw_episodes, list):
            raise ValueError("Learner.from_json: 'episodes' must be a list")
        learner = cls(planner, rules,
                      max_episodes=int(data.get("max_episodes", MAX_EPISODES))
                      if isinstance(data.get("max_episodes", MAX_EPISODES), int)
                      else MAX_EPISODES)
        for raw in raw_episodes:
            learner.episodes.append(Episode(
                cycle=raw.get("cycle", 0),
                action=raw.get("action", ""),
                situation=dict(raw.get("situation") or {}),
                ok=raw.get("ok", False),
                surprises=list(raw.get("surprises") or []),
                # Phase 10.5: legacy saves carry no "args"; they load
                # as empty-args episodes (see the Episode default).
                args=dict(raw.get("args") or {}),
            ))
        learner.episodes = learner.episodes[-learner.max_episodes:]
        pruned = len(raw_episodes) > learner.max_episodes
        raw_counts = data.get("counts", [])
        if isinstance(raw_counts, list) and not pruned:
            for entry in raw_counts:
                try:
                    key, counts = entry
                    if len(key) == 3:
                        # Phase 10.5 form: [action, args_identity, sig].
                        action, args_id, sig = key
                    elif len(key) == 2:
                        # Legacy form: [action, sig]; the counts were
                        # recorded before args were tracked, so they
                        # belong to the empty-args identity.
                        action, sig = key
                        args_id = ""
                    else:
                        continue
                    sig_tuple = tuple(sorted((k, str(v)) for k, v in sig))
                    learner._counts[(action, str(args_id), sig_tuple)] = [
                        int(counts[0]), int(counts[1])]
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
        if not learner._counts and learner.episodes or pruned:
            # Counts were not stored, or episodes were pruned so stored
            # counts no longer match the survivors: re-derive from episodes.
            learner._rebuild_counts()
        # Phase 3: restore explicit approve/reject decisions (missing keys
        # mean a Phase 1/2 save, whose decisions were transient by design).
        # Accept both the dict form written by to_json() and a bare list of
        # rule ids, so a hand-written or older file still loads.
        for attr, key in (("_approved", "approved"), ("_rejected", "rejected")):
            restored: dict[str, dict] = {}
            for entry in data.get(key, []) or []:
                try:
                    if isinstance(entry, str):
                        restored[entry] = {"action": "", "situation": {}}
                    else:
                        rid = entry["rule_id"]
                        restored[rid] = {"action": entry.get("action", ""),
                                         "situation": dict(entry.get("situation") or {})}
                except (TypeError, KeyError, AttributeError):
                    continue
            setattr(learner, attr, restored)
        return learner

    # ------------------------------------------------------------------ proposals
    def _qualifying(self) -> list[tuple[str, str, tuple, int, int]]:
        """(action, args-identity, situation-sig, successes, failures)
        pairs that meet the proposal threshold: >= threshold failures
        and 0 successes."""
        return [(action, args_id, sig, ok, bad)
                for (action, args_id, sig), (ok, bad) in self._counts.items()
                if bad >= self.threshold and ok == 0]

    def _detail(self, action: str, args_id: str, sig: tuple, ok: int, bad: int) -> dict:
        return {
            "rule_id": stable_rule_id(action, dict(sig), args_id),
            "action": action,
            "args": args_id,
            "situation": {k: _coerce(v) for k, v in sig},
            "failures": bad,
            "successes": ok,
            # Evidence: how many recorded episodes support this proposal
            # (same action, same args identity, same situation).
            "evidence": sum(1 for ep in self.episodes
                            if ep.action == action
                            and canonical_args(getattr(ep, "args", None) or {}) == args_id
                            and self._sig(ep.situation) == sig),
        }

    def _sync_engine_decisions(self, detail: dict) -> str | None:
        """If the RuleEngine already resolved this rule out-of-band (e.g. a
        caller approved/rejected it via ``rules.approve`` / ``rules.reject``
        directly), mirror that decision into the learner's sets so
        pending/approved views stay consistent. Returns the engine status
        ("active" / "retired") or None if the rule is not in the engine."""
        rule = self.rules.get(detail["rule_id"])
        if rule is None:
            return None
        if rule.status == "active":
            self._approved.setdefault(detail["rule_id"],
                                      {"action": detail["action"],
                                       "situation": dict(detail["situation"])})
            return "active"
        if rule.status == "retired":
            self._rejected.setdefault(detail["rule_id"],
                                      {"action": detail["action"],
                                       "situation": dict(detail["situation"])})
            return "retired"
        return rule.status

    def pending_rules(self) -> list[dict]:
        """Proposed rules still awaiting a human decision.

        Each dict has at least ``rule_id``, ``action``, ``situation``,
        ``failures``, ``successes``, ``status`` (``"pending"``) and
        ``evidence`` (episode count). Approved and rejected rules are never
        returned here. Calling this never approves anything. It does
        materialise the proposed rules in the RuleEngine (via
        ``propose_rules()``, as ``Brain.step`` already does each cycle), so
        the pending status is observable there too.
        """
        self.propose_rules()
        out = []
        for action, args_id, sig, ok, bad in self._qualifying():
            detail = self._detail(action, args_id, sig, ok, bad)
            rid = detail["rule_id"]
            if rid in self._approved or rid in self._rejected:
                continue
            if self._sync_engine_decisions(detail) in ("active", "retired"):
                continue
            out.append({**detail, "status": "pending"})
        return out

    def approved_rules(self) -> list[dict]:
        """Rules the human explicitly approved: list of dicts with
        ``rule_id``, ``action`` and ``situation``. The core worker's planner
        consults this to exclude a banned (action, situation) pair from
        plans. Read-only: calling it approves nothing."""
        approved = {rid: dict(d) for rid, d in self._approved.items()}
        for action, args_id, sig, ok, bad in self._qualifying():
            detail = self._detail(action, args_id, sig, ok, bad)
            if self._sync_engine_decisions(detail) == "active":
                approved[detail["rule_id"]] = {"action": detail["action"],
                                               "situation": dict(detail["situation"])}
        return [{"rule_id": rid, "action": d["action"], "situation": dict(d["situation"])}
                for rid, d in sorted(approved.items())]

    def approve_rule(self, rule_id: str) -> bool:
        """Explicit human approval: move a pending proposed rule to active.

        The rule is activated in the RuleEngine and recorded in the
        learner's approved set (persisted in learner.json). Returns True
        only if the rule existed and was pending; False otherwise (unknown
        id, already approved, or already rejected). Never auto-approves:
        only this call approves.
        """
        pending = {d["rule_id"]: d for d in self.pending_rules()}
        if rule_id not in pending:
            return False
        rule = self.rules.get(rule_id)
        if rule is None:
            # Proposed from counts but not yet materialised in this engine
            # (e.g. after a load into a fresh engine): materialise first.
            self.propose_rules()
            rule = self.rules.get(rule_id)
        if rule is None or rule.status != "pending":
            return False
        self.rules.approve(rule_id)
        detail = pending[rule_id]
        self._approved[rule_id] = {"action": detail["action"],
                                   "situation": dict(detail["situation"])}
        return True

    def reject_rule(self, rule_id: str) -> bool:
        """Explicit human rejection: mark a pending proposed rule rejected.

        The rejection is persisted in learner.json, so the same episodes
        never re-propose it — even after a save/load into a fresh engine.
        Returns True only if the rule existed and was pending; False
        otherwise (unknown id, already approved, or already rejected).
        """
        if rule_id in self._rejected or rule_id in self._approved:
            return False
        pending = {d["rule_id"]: d for d in self.pending_rules()}
        if rule_id not in pending:
            return False
        rule = self.rules.get(rule_id)
        if rule is not None and rule.status == "pending":
            self.rules.reject(rule_id)
        detail = pending[rule_id]
        self._rejected[rule_id] = {"action": detail["action"],
                                   "situation": dict(detail["situation"])}
        return True

    def propose_rules(self) -> list[Rule]:
        """Action that failed >= threshold times in a situation and never succeeded there.

        Proposed rules are added to the RuleEngine as ``status="pending"``
        only — they have no effect until ``approve_rule()`` is called.
        A rejected (or already approved) pair is never re-proposed, even
        from the same episodes.
        """
        new = []
        for (action, args_id, sig), (ok, bad) in self._counts.items():
            rid = stable_rule_id(action, dict(sig), args_id)
            if rid in self._approved or rid in self._rejected:
                continue
            if bad >= self.threshold and ok == 0 and self.rules.get(rid) is None:
                conds = [{"key": k, "op": "==", "value": _coerce(v)} for k, v in sig]
                new.append(self.rules.add(Rule(
                    id=rid, when=conds, then=[{"do": "avoid_action", "args": {"name": action}}],
                    priority=0.6, source="experience", status="pending")))
        return new

    def propose(self) -> list[Rule]:
        """Alias for ``propose_rules()`` (kept for callers using the short name)."""
        return self.propose_rules()


def _coerce(text: str):
    if text in ("True", "False"):
        return text == "True"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text
