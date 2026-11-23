"""Unified reasoning engine (Phase 4): deduction, induction, abduction,
analogy, counterfactual simulation, constraint checks and contradiction
detection over verified knowledge only.

Design rules (master mandate):
- Deduction is verified-only: it derives from facts whose status is
  "verified" (the injected ``facts_fn`` is the knowledge source, so
  the engine is testable standalone without a Brain or LTM; the
  LTM-backed equivalent of its traversal is
  ``Inference.closure_multi``).
- Hypotheses never enter LTM and never drive an action; they only
  suggest a test / research goal.
- Induction proposes rules as "pending" only — never auto-approved.
- Nothing is hardcoded to any evaluation task; every conclusion is
  computed from the knowledge, episodes and predictions passed in.
- Deterministic: no randomness, no time-based ids (SHA-256 ids).

The engine never imports Brain or any ab.body module.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass, field

from ..core.vocab import FUNCTIONAL_RELATIONS, normalize_relation, normalize_text
from .causal import CausalModel
from .hypothesis import Hypothesis, make_hypothesis_id
from .inference import Inference, Proof
from .rules import check as _rule_check

FUNCTIONAL = {normalize_relation(r) for r in FUNCTIONAL_RELATIONS}
DEFAULT_TRANSITIVE = ("is_a", "part_of", "located_in")
INDUCTION_MIN_EPISODES = 3


@dataclass
class Derivation:
    subject: str
    relation: str
    value: str
    proof: Proof | None
    confidence: float
    source: str = "deduction"
    status: str = "derived"                # derived | hypothesis | disputed


@dataclass
class Contradiction:
    claim_a: tuple
    claim_b: tuple
    reason: str
    policy: str = "disputed+research"      # Phase 4 policy, always this


@dataclass
class ReasoningResult:
    derivations: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    considered_rules: list = field(default_factory=list)
    fired_rules: list = field(default_factory=list)
    events: list = field(default_factory=list)   # [(kind, detail), ...]
    budget_exhausted: bool = False
    steps_used: int = 0
    insufficient_evidence: bool = False
    needed: str | None = None
    strategy: str = "deduction"


def _field(fact, name, default=None):
    """Read a field off a Fact object or a plain dict."""
    if isinstance(fact, dict):
        return fact.get(name, default)
    return getattr(fact, name, default)


def _induce_rule_id(action: str, situation: dict, outcome: str) -> str:
    payload = json.dumps([action,
                          sorted((k, str(v)) for k, v in situation.items()),
                          outcome], sort_keys=True)
    return "induce_" + hashlib.sha256(payload.encode()).hexdigest()[:8]


class ReasoningEngine:
    def __init__(self, inference, rules, max_steps=20, max_branches=4, max_depth=6):
        self.inference = inference
        self.rules = rules
        self.max_steps = max_steps
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.causal: CausalModel | None = None   # optionally attached by the caller
        self.events: list = []                  # [(kind, detail), ...]

    # ------------------------------------------------------------------ helpers
    def _event(self, kind: str, detail: str) -> None:
        self.events.append((kind, detail))

    @property
    def _transitive(self):
        rels = getattr(self.inference, "transitive", None) if self.inference is not None else None
        return sorted(rels) if rels else sorted(DEFAULT_TRANSITIVE)

    def _fetch_facts(self, facts_fn, subject, relation=None):
        """Facts for one subject from the injected source. Accepts
        facts_fn(subject), facts_fn(subject, relation), or None (falls
        back to the engine's own Inference LTM). Only verified facts
        are ever returned, whatever the source hands back."""
        out = None
        if facts_fn is not None:
            try:
                out = facts_fn(subject)
            except TypeError:
                try:
                    out = facts_fn(subject, relation)
                except TypeError:
                    out = None
        elif self.inference is not None and getattr(self.inference, "ltm", None) is not None:
            out = self.inference.ltm.query(subject, relation, statuses=("verified",))
        if not out:
            return []
        return [f for f in out
                if _field(f, "status", "verified") == "verified"]

    def _fetch_all_facts(self, facts_fn):
        """All verified facts, for sibling search in analogize(). Tries
        facts_fn(None) (works when it is LTM-query-like), then the
        engine's own Inference LTM."""
        if facts_fn is not None:
            for args in ((None,), (None, None)):
                try:
                    out = facts_fn(*args)
                except TypeError:
                    continue
                if out:
                    return [f for f in out
                            if _field(f, "status", "verified") == "verified"]
        if self.inference is not None and getattr(self.inference, "ltm", None) is not None:
            return [f for f in self.inference.ltm.all_facts() if f.status == "verified"]
        return []

    @staticmethod
    def _norm_episode(ep) -> dict:
        """Episodes arrive as dicts {action, situation, outcome} or as
        Learner Episode objects (action, situation, ok)."""
        if isinstance(ep, dict):
            outcome = ep.get("outcome")
            if outcome is None:
                outcome = "success" if ep.get("ok", False) else "failure"
            return {"action": str(ep.get("action", "")),
                    "situation": dict(ep.get("situation") or {}),
                    "outcome": str(outcome)}
        return {"action": str(getattr(ep, "action", "")),
                "situation": dict(getattr(ep, "situation", None) or {}),
                "outcome": "success" if getattr(ep, "ok", False) else "failure"}

    @staticmethod
    def _is_surprise(pred) -> bool:
        """A prediction counts as a surprise / violation when it was
        checked and failed, or expected != actual."""
        if not isinstance(pred, dict):
            return False
        if pred.get("met") is False:
            return True
        if pred.get("violated") is True or pred.get("surprise") is True:
            return True
        if str(pred.get("status", "")).lower() in ("violated", "surprise", "failed"):
            return True
        if "expected" in pred and "actual" in pred and pred["expected"] != pred["actual"]:
            return True
        if "predicted" in pred and "actual" in pred and pred["predicted"] != pred["actual"]:
            return True
        return False

    def _bfs_multi(self, subject, facts_fn):
        """Mixed-relation BFS over the injected facts_fn — the same
        traversal as Inference.closure_multi, but over the injected
        source so the engine works standalone. Returns
        [(value, steps, relation_path)]. Verified facts only."""
        rels = set(self._transitive)
        start = normalize_text(subject)
        seen = {start}
        queue = deque([(start, [], ())])
        out, cache = [], {}
        while queue:
            node, path, rel_path = queue.popleft()
            if len(path) >= self.max_depth:
                continue
            if node not in cache:
                cache[node] = self._fetch_facts(facts_fn, node)
            for f in cache[node]:
                r = normalize_relation(_field(f, "relation", ""))
                if r not in rels:
                    continue
                v = normalize_text(_field(f, "value", ""))
                if not v or v in seen:
                    continue
                seen.add(v)
                new_path, new_rel_path = path + [f], rel_path + (r,)
                out.append((v, new_path, new_rel_path))
                queue.append((v, new_path, new_rel_path))
        return out

    # ------------------------------------------------------------------ deduction
    def deduce(self, subject, facts_fn) -> list[Derivation]:
        """Multi-step mixed-relation chains from verified facts only.
        Confidence is proof_confidence() over the chain's steps."""
        out = []
        for value, steps, rel_path in self._bfs_multi(subject, facts_fn):
            conf = Inference.proof_confidence(steps)
            out.append(Derivation(subject=normalize_text(subject), relation=rel_path[-1],
                                  value=value, proof=Proof(list(steps), conf),
                                  confidence=conf, source="deduction", status="derived"))
        return out

    # ------------------------------------------------------------------ induction
    def induce(self, episodes) -> list[dict]:
        """Generalise from repeated SUCCESSES: the same outcome in >=3
        situations sharing a feature (same action + one shared
        situation key/value) proposes a general rule. Proposed rules
        are ALWAYS "pending" — this method never approves anything.
        Distinct from avoid-rules, which generalise from failures."""
        eps = [self._norm_episode(e) for e in (episodes or [])]
        groups: dict[tuple, list] = {}
        for ep in eps:
            groups.setdefault((ep["action"], ep["outcome"]), []).append(ep)
        proposed = []
        for (action, outcome), items in sorted(groups.items()):
            if len(items) < INDUCTION_MIN_EPISODES:
                continue
            counts: Counter = Counter()
            first_seen: dict = {}
            for ep in items:
                for k, v in sorted(ep["situation"].items()):
                    key = (k, str(v))
                    counts[key] += 1
                    first_seen.setdefault(key, v)
            cands = [(kv, c) for kv, c in counts.items() if c >= INDUCTION_MIN_EPISODES]
            if not cands:
                continue
            cands.sort(key=lambda kv_c: (-kv_c[1], kv_c[0]))
            joint = {k: first_seen[(k, v)] for (k, v), _ in cands}
            joint_match = [ep for ep in items
                           if all(str(ep["situation"].get(k)) == str(v)
                                  for k, v in joint.items())]
            if len(joint_match) >= INDUCTION_MIN_EPISODES:
                situation, supporting = joint, joint_match
            else:
                (k, v), _ = cands[0]
                situation = {k: first_seen[(k, v)]}
                supporting = [ep for ep in items
                              if str(ep["situation"].get(k)) == str(v)]
            successes = len(supporting)
            failures = len([ep for ep in eps
                            if ep["action"] == action and ep["outcome"] != outcome
                            and all(str(ep["situation"].get(k)) == str(v)
                                    for k, v in situation.items())])
            proposed.append({
                "id": _induce_rule_id(action, situation, outcome),
                "action": action,
                "situation": dict(situation),
                "outcome": outcome,
                "failures": failures,
                "successes": successes,
                "status": "pending",
                "evidence": [dict(ep) for ep in supporting],
            })
        return proposed

    # ------------------------------------------------------------------ abduction
    def abduce(self, surprise, episodes, causal=None) -> list[Hypothesis]:
        """Explain a surprise: rank candidate causes by prior (how often
        that cause produced this effect in the episodes) + fit (how many
        of the observed instances of this effect the cause accounts
        for). The top candidate(s) become Hypotheses with a test to run.
        Capped at max_branches; lowest-confidence hypotheses are
        dropped first, with an event each."""
        if not surprise or not isinstance(surprise, dict):
            return []
        action = normalize_text(surprise.get("action", ""))
        actual = surprise.get("actual")
        if actual is None:
            return []
        actual_s = normalize_text(str(actual))
        eps = [self._norm_episode(e) for e in (episodes or [])]
        total_actual = len([ep for ep in eps if normalize_text(ep["outcome"]) == actual_s])
        # candidate label -> [support, total]; candidates are actions and
        # situation features observed in the episodes.
        stats: dict[str, list] = {}

        def _note(label, outcome):
            cell = stats.setdefault(label, [0, 0])
            cell[1] += 1
            if normalize_text(outcome) == actual_s:
                cell[0] += 1

        for ep in eps:
            _note(f"action:{ep['action']}", ep["outcome"])
            for k, v in sorted(ep["situation"].items()):
                _note(f"{k}={v}", ep["outcome"])
        causal_model = causal if causal is not None else self.causal
        causal_backed: set = set()
        if causal_model is not None:
            for claim in causal_model.claims():
                if normalize_text(claim["effect"]) == actual_s:
                    label = f"action:{claim['action']}"
                    cell = stats.setdefault(label, [claim["observations"], claim["observations"]])
                    causal_backed.add(label)
        ranked = []
        for label, (support, total) in stats.items():
            if support == 0:
                continue
            prior = support / total if total else 0.0
            fit = support / total_actual if total_actual else 0.0
            score = prior + fit
            ranked.append((score, support, label))
        ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
        hyps = []
        for score, support, label in ranked:
            conf = round(min(score / 2.0, 0.85), 4)
            value = label[len("action:"):] if label.startswith("action:") else label
            evidence_for = [f"episode: {ep['action']} -> {ep['outcome']}"
                            for ep in eps
                            if normalize_text(ep["outcome"]) == actual_s
                            and (f"action:{ep['action']}" == label
                                 or any(f"{k}={v}" == label
                                        for k, v in ep["situation"].items()))]
            evidence_against = [f"episode: {ep['action']} -> {ep['outcome']}"
                                for ep in eps
                                if normalize_text(ep["outcome"]) != actual_s
                                and (f"action:{ep['action']}" == label
                                     or any(f"{k}={v}" == label
                                            for k, v in ep["situation"].items()))]
            if label in causal_backed:
                evidence_for.append(f"causal model claims action:{value} causes {actual_s}")
            hyps.append(Hypothesis(
                id=make_hypothesis_id(action or "unknown", "caused_by", value),
                subject=action, relation="caused_by", value=value,
                confidence=conf, source="abduction",
                evidence_for=evidence_for, evidence_against=evidence_against,
                status="open",
                test=(f"Repeat '{action}' with '{value}' present and check whether the "
                      f"outcome is '{actual_s}' (predicted effect to check: {actual_s}); "
                      f"if the outcome differs, this cause is ruled out."),
            ))
        if len(hyps) > self.max_branches:
            dropped = hyps[self.max_branches:]
            hyps = hyps[:self.max_branches]
            for h in sorted(dropped, key=lambda x: x.confidence):
                self._event("hypothesis_dropped",
                            f"dropped abduced hypothesis {h.id} ({h.value}, "
                            f"confidence {h.confidence}) to respect "
                            f"max_branches={self.max_branches}")
        return hyps

    # ------------------------------------------------------------------ analogy
    def analogize(self, subject, relation, facts_fn) -> Hypothesis | None:
        """Same relation pattern in a different domain: a sibling subject
        that shares a feature with this subject and has the requested
        relation proposes an analogous conclusion — as a Hypothesis,
        confidence capped at 0.4, never verified alone."""
        subject = normalize_text(subject)
        relation = normalize_relation(relation)
        own = self._fetch_facts(facts_fn, subject)
        if not own:
            return None

        def _rv_pairs(facts):
            return {(normalize_relation(_field(f, "relation", "")),
                     normalize_text(_field(f, "value", ""))) for f in facts}

        own_pairs = _rv_pairs(own)                       # (relation, value) pairs
        own_values = {normalize_text(_field(f, "value", ""))
                      for f in own if normalize_relation(_field(f, "relation", "")) == relation}
        all_facts = self._fetch_all_facts(facts_fn)
        by_subject: dict[str, list] = {}
        for f in all_facts:
            by_subject.setdefault(normalize_text(_field(f, "subject", "")), []).append(f)
        candidates = []
        for sibling, facts in sorted(by_subject.items()):
            if sibling == subject:
                continue
            # Features the sibling shares with the subject, other than
            # the requested relation itself.
            shared = sorted(p for p in (_rv_pairs(facts) & own_pairs) if p[0] != relation)
            if not shared:
                continue
            for f in facts:
                if normalize_relation(_field(f, "relation", "")) != relation:
                    continue
                v = normalize_text(_field(f, "value", ""))
                if v in own_values:
                    continue
                candidates.append((len(shared), sibling, v, shared))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
        shared_n, sibling, value, shared = candidates[0]
        evidence = ([f"{subject} {r} {v}" for r, v in shared]
                    + [f"{sibling} {r} {v}" for r, v in shared]
                    + [f"{sibling} {relation} {value}"])
        return Hypothesis(
            id=make_hypothesis_id(subject, relation, value),
            subject=subject, relation=relation, value=value,
            confidence=round(min(0.4, 0.2 * shared_n), 4),
            source="analogy", evidence_for=evidence, evidence_against=[],
            status="open",
            test=(f"Check whether {subject} {relation} {value} (predicted by analogy "
                  f"with {sibling}, which shares {len(shared)} feature(s)); verify by "
                  f"observation or research before trusting it."),
        )

    # ------------------------------------------------------------------ simulation
    def simulate(self, plan_steps, start_state, effects_fn) -> tuple[dict, bool, list]:
        """Counterfactual plan simulation over SYMBOLIC action effects
        only: apply effects_fn(action, state) stepwise WITHOUT executing
        anything. effects_fn returns a dict of state updates (or None /
        {} when the effect is unknown). Returns (simulated_state,
        met_goal, assumptions). Simulated outcomes are hypotheses, never
        observations. met_goal: the "goal" dict in start_state (if any)
        is satisfied by the final state; without one, True means every
        step had a known effect."""
        state = dict(start_state or {})
        goal = state.pop("goal", None)
        assumptions: list[str] = []
        all_known = True
        for step in (plan_steps or []):
            if isinstance(step, dict):
                name = step.get("action") or step.get("name") or str(step)
            else:
                name = str(step)
            result = effects_fn(name, dict(state))
            if not isinstance(result, dict) or not result:
                assumptions.append(f"no known symbolic effect for '{name}': state unchanged")
                all_known = False
                continue
            state.update(result)
            assumptions.append(f"assumed effect of '{name}': {result}")
        if isinstance(goal, dict) and goal:
            met = all(state.get(k) == v for k, v in goal.items())
        else:
            met = all_known
        return state, met, assumptions

    # ------------------------------------------------------------------ constraints
    def check_constraints(self, derivation, constraints) -> tuple[bool, str | None]:
        """Check a derivation against constraints [{key, op, value}, ...]
        in the same shape as Rule.when conditions, tested against the
        derivation's own fields (subject, relation, value, confidence,
        source, status). Returns (True, None), or (False,
        violated-constraint-name). A violating derivation is rejected,
        not silently accepted."""
        if not constraints:
            return True, None
        state = {"subject": derivation.subject, "relation": derivation.relation,
                 "value": derivation.value, "confidence": derivation.confidence,
                 "source": derivation.source, "status": derivation.status}
        for c in constraints:
            if not isinstance(c, dict):
                continue
            name = c.get("name", c.get("key"))
            try:
                ok = _rule_check(c, state)
            except (KeyError, TypeError):
                ok = False
            if not ok:
                return False, name
        return True, None

    # ------------------------------------------------------------------ contradictions
    def detect_contradictions(self, derivations, facts_fn) -> list[Contradiction]:
        """A derived claim conflicting with a verified fact, or two
        derivations conflicting on the same (subject, relation) with
        different values where the relation is functional (boils_at,
        melts_at, capital_of). Each contradiction marks the derivation(s)
        involved disputed and carries policy "disputed+research"."""
        out: list[Contradiction] = []
        seen_pairs: set = set()

        def _add(claim_a, claim_b, reason, derivs):
            pair = tuple(sorted([claim_a, claim_b]))
            if pair in seen_pairs:
                return
            seen_pairs.add(pair)
            out.append(Contradiction(claim_a=claim_a, claim_b=claim_b,
                                     reason=reason, policy="disputed+research"))
            for d in derivs:
                d.status = "disputed"

        for d in (derivations or []):
            rel = normalize_relation(d.relation)
            if rel not in FUNCTIONAL:
                continue
            for f in self._fetch_facts(facts_fn, d.subject):
                if normalize_relation(_field(f, "relation", "")) != rel:
                    continue
                fv = normalize_text(_field(f, "value", ""))
                if fv == normalize_text(d.value):
                    continue
                _add((normalize_text(d.subject), rel, normalize_text(d.value)),
                     (normalize_text(_field(f, "subject", "")), rel, fv),
                     (f"functional relation '{rel}' for '{d.subject}': derived "
                      f"'{d.value}' conflicts with verified fact '{fv}'"),
                     [d])
        by_key: dict[tuple, list] = {}
        for d in (derivations or []):
            by_key.setdefault((normalize_text(d.subject),
                               normalize_relation(d.relation)), []).append(d)
        for (subj, rel), ds in sorted(by_key.items()):
            if rel not in FUNCTIONAL:
                continue
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    if normalize_text(ds[i].value) == normalize_text(ds[j].value):
                        continue
                    _add((subj, rel, normalize_text(ds[i].value)),
                         (subj, rel, normalize_text(ds[j].value)),
                         (f"functional relation '{rel}' for '{subj}': derivations "
                          f"disagree ('{ds[i].value}' vs '{ds[j].value}')"),
                         [ds[i], ds[j]])
        return out

    # ------------------------------------------------------------------ hypotheses
    def evaluate_hypothesis(self, hypothesis, observation) -> Hypothesis:
        """Test a hypothesis against an observation {"met": bool,
        "detail": str}. met=True corroborates (confidence +0.1, capped
        at 0.9); met=False falsifies, with a hypothesis_falsified
        event. Returns the updated hypothesis."""
        obs = observation or {}
        detail = str(obs.get("detail", ""))
        if bool(obs.get("met")):
            hypothesis.status = "corroborated"
            hypothesis.confidence = round(min(0.9, hypothesis.confidence + 0.1), 4)
            if detail:
                hypothesis.evidence_for.append(detail)
            self._event("hypothesis_corroborated",
                        f"{hypothesis.id} corroborated: {detail}")
        else:
            hypothesis.status = "falsified"
            if detail:
                hypothesis.evidence_against.append(detail)
            self._event("hypothesis_falsified",
                        f"{hypothesis.id} falsified: {detail}")
        return hypothesis

    # ------------------------------------------------------------------ the agenda loop
    @staticmethod
    def _parse_goal_target(target):
        """Goal target shapes: dict {subject, relation, value?},
        (subject, relation[, value]) tuple/list, or "subject relation
        [value]" string. Returns (subject, relation, value|None)."""
        if isinstance(target, dict):
            return (normalize_text(target.get("subject", "")),
                    normalize_relation(target.get("relation", "")),
                    (normalize_text(target["value"]) if target.get("value") is not None else None))
        if isinstance(target, (list, tuple)):
            parts = list(target)
            return (normalize_text(parts[0]) if len(parts) > 0 else "",
                    normalize_relation(parts[1]) if len(parts) > 1 else "",
                    normalize_text(parts[2]) if len(parts) > 2 else None)
        parts = str(target).split()
        return (normalize_text(parts[0]) if parts else "",
                normalize_relation(parts[1]) if len(parts) > 1 else "",
                normalize_text(" ".join(parts[2:])) if len(parts) > 2 else None)

    def _goal_supported(self, target, derivations, facts_fn, constraints=None):
        """True when a verified premise supports the goal target: an
        existing derivation, a direct verified fact, or a fresh
        deduction for the target subject. Supporting derivations are
        returned (deduplicated against the ones already collected)."""
        subj, rel, value = self._parse_goal_target(target)
        if not subj or not rel:
            return False, []
        fresh = []
        for d in self.deduce(subj, facts_fn):
            if normalize_relation(d.relation) != rel:
                continue
            if value is not None and normalize_text(d.value) != value:
                continue
            if constraints:
                ok, _ = self.check_constraints(d, constraints)
                if not ok:
                    continue
            fresh.append(d)
        if fresh:
            return True, fresh
        for d in derivations:
            if (normalize_text(d.subject) == subj
                    and normalize_relation(d.relation) == rel
                    and (value is None or normalize_text(d.value) == value)):
                return True, []
        for f in self._fetch_facts(facts_fn, subj):
            if normalize_relation(_field(f, "relation", "")) != rel:
                continue
            if value is not None and normalize_text(_field(f, "value", "")) != value:
                continue
            return True, []
        return False, []

    def reason(self, subjects, facts_fn, goal_targets=None, episodes=None,
               predictions=None, constraints=None) -> ReasoningResult:
        """Agenda-based bounded reasoning over working-memory subjects,
        goal targets, and open contradictions/hypotheses.

        - Strategy: "deduction" by default; "abduction" when a
          surprise / prediction-violation is in predictions;
          "induction" when episodes has >=3 same-outcome rows.
        - Each agenda item and each rule/induction/abduction pass costs
          one step; when the budget (max_steps) is spent,
          budget_exhausted=True, a reasoning_budget_exhausted event is
          emitted, and partial results are returned.
        - ALL matched rules are recorded in considered_rules; fired_rules
          is those rules in effective-priority order (rules.match sorts
          that way).
        - A goal target with no verified premise support sets
          insufficient_evidence=True with needed text — the engine never
          guesses a value for it."""
        self.events = []
        subjects = [normalize_text(s) for s in (subjects or [])]
        goal_targets = list(goal_targets or [])
        episodes = list(episodes or [])
        predictions = list(predictions or [])
        constraints = list(constraints or [])
        norm_eps = [self._norm_episode(e) for e in episodes]
        surprises = [p for p in predictions if self._is_surprise(p)]
        outcome_counts = Counter(ep["outcome"] for ep in norm_eps)
        induction_ready = any(c >= INDUCTION_MIN_EPISODES for c in outcome_counts.values())
        if surprises:
            strategy = "abduction"
        elif induction_ready:
            strategy = "induction"
        else:
            strategy = "deduction"
        self._event("strategy_selected", strategy)

        steps_used = 0
        budget_exhausted = False
        derivations: list[Derivation] = []
        hypotheses: list[Hypothesis] = []
        seen_derivations: set = set()

        def _add_derivation(d):
            key = (normalize_text(d.subject), normalize_relation(d.relation),
                   normalize_text(d.value))
            if key in seen_derivations:
                return
            seen_derivations.add(key)
            derivations.append(d)

        def _spend():
            nonlocal steps_used, budget_exhausted
            if steps_used >= self.max_steps:
                budget_exhausted = True
                return False
            steps_used += 1
            return True

        def _exhausted(note):
            self._event("reasoning_budget_exhausted",
                        f"budget of {self.max_steps} steps spent after {steps_used} "
                        f"steps: {note}")

        # 1. Rules: record ALL matched rules against the state built
        #    from the supplied inputs; fired in effective-priority order.
        state = {"subjects": subjects,
                 "subject": subjects[0] if subjects else None,
                 "has_goal": bool(goal_targets),
                 "goal_targets": goal_targets,
                 "surprise": bool(surprises),
                 "episodes": len(episodes),
                 "predictions": len(predictions)}
        for t in goal_targets:
            if isinstance(t, dict):
                for k, v in t.items():
                    state.setdefault(k, v)
        considered = list(self.rules.match(state)) if self.rules is not None else []
        fired = list(considered)
        if not _spend():
            _exhausted("no steps left for rule matching")
            return ReasoningResult(derivations, hypotheses, [], considered, fired,
                                   list(self.events), True, steps_used, False, None, strategy)

        # 2. Agenda: subjects, then goal targets.
        agenda = [("subject", s) for s in subjects] + [("goal", t) for t in goal_targets]
        insufficient_evidence = False
        needed = None
        for idx, (kind, item) in enumerate(agenda):
            if not _spend():
                _exhausted(f"{len(agenda) - idx} agenda item(s) unprocessed")
                break
            if kind == "subject":
                for d in self.deduce(item, facts_fn):
                    if constraints:
                        ok, name = self.check_constraints(d, constraints)
                        if not ok:
                            self._event("constraint_violation",
                                        f"rejected derivation {d.subject} {d.relation} "
                                        f"{d.value}: violates constraint '{name}'")
                            continue
                    _add_derivation(d)
            else:
                supported, fresh = self._goal_supported(item, derivations, facts_fn, constraints)
                for d in fresh:
                    _add_derivation(d)
                if not supported:
                    insufficient_evidence = True
                    if needed is None:
                        needed = (f"need verified evidence for goal target {item!r}: "
                                  f"no verified premise supports it")

        # 3. Induction from repeated outcomes.
        if induction_ready and not budget_exhausted:
            if not _spend():
                _exhausted("induction pass skipped")
            else:
                for rule in self.induce(episodes):
                    self._event("rule_proposed",
                                f"induced rule {rule['id']} ({rule['action']} -> "
                                f"{rule['outcome']}) is pending human approval")

        # 4. Abduction for surprises.
        if surprises and not budget_exhausted:
            if not _spend():
                _exhausted("abduction pass skipped")
            else:
                for p in surprises:
                    surprise = {"action": p.get("action", p.get("subject", "")),
                                "expected": p.get("expected", p.get("predicted")),
                                "actual": p.get("actual", p.get("observed"))}
                    hypotheses.extend(self.abduce(surprise, episodes, self.causal))

        # 5. Contradiction check over whatever was derived (validation,
        #    not agenda work, so it is not charged to the budget).
        contradictions = self.detect_contradictions(derivations, facts_fn)

        return ReasoningResult(
            derivations=derivations,
            hypotheses=hypotheses,
            contradictions=contradictions,
            considered_rules=considered,
            fired_rules=fired,
            events=list(self.events),
            budget_exhausted=budget_exhausted,
            steps_used=steps_used,
            insufficient_evidence=insufficient_evidence,
            needed=needed,
            strategy=strategy,
        )
