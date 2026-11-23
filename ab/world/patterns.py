"""Abstract transfer patterns (Phase 5): structural, role-based patterns
learned from episodes, applied cross-world by the planner as cost priors.

Learning rules, all structural (design §3.1, Phase 6 §2.1):
- P-A (unreliable tool-less opener): for one claimed effect E, some
  opener with no object precondition FAILED while another opener with
  an object precondition SUCCEEDED for the same effect. The contrast
  is the evidence — one failure alone learns nothing.
- P-B (one-use tool): an action SUCCEEDED once and then FAILED in the
  same situation for the same effect.
- P-C (durable tool, Phase 6): an action's tool SUCCEEDED >= 3 times
  across >= 2 distinct situations for the same effect, with no
  observed failure of that action. Learned as a CANDIDATE first:
  ``learn_from_episodes`` returns it with ``candidate=True`` and
  does NOT add it to the library — the Brain's consolidation pass
  promotes it (``add_pattern``) only when the promotion bar is met.
- P-D (precondition discovery, Phase 6): an action FAILED >= 2 times
  in situations sharing a missing key K (K absent-or-falsy in every
  failure) and SUCCEEDED >= 1 time with K present, for the same
  effect. Also a candidate first, same promotion path as P-C.

Patterns store roles and relations only — never world names, never a
plan, never a world-specific answer. Binding is by structure: an
action is an "opener" for E when E is in its effects; it is
"tool-less" when none of its preconditions is a ``has_<entity>`` key
(a key-shape rule, not a name). Pattern ids are stable SHA-256 hashes
of the canonical role/relation/effect structure, so the same structure
learned in one world matches the same structure in another.

Deterministic, stdlib only, CPU-first, no LLM.
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# Cost multiplier applied by the planner when a pattern fires (design
# §3.2). A multiplier, never a ban: a penalised action can still be
# planned when nothing else works.
PATTERN_PENALTY = 6.0


def _canonical_id(kind: str, effect: str, roles, relations) -> str:
    """Stable SHA-256 of the canonical role/relation/effect structure.
    Roles are sorted by name, relations sorted, keys sorted — the same
    structure always hashes the same, whatever world it came from."""
    canonical = {
        "kind": kind,
        "effect": effect,
        "roles": sorted(
            ({"name": r["name"],
              "features": dict(sorted(r.get("features", {}).items()))}
             for r in roles),
            key=lambda r: r["name"]),
        "relations": sorted([list(rel) for rel in relations]),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _evidence_item(world, action, outcome) -> dict:
    return {"world": str(world), "action": str(action),
            "outcome": str(outcome)}


@dataclass
class AbstractPattern:
    """One learned structural pattern.

    ``roles`` is a list of ``{"name": ..., "features": {...}}`` dicts;
    ``relations`` is a list of ``(role_a, relation, role_b)`` triples;
    ``evidence`` is a list of ``{"world", "action", "outcome"}`` dicts;
    ``confidence`` is 0.5 + 0.1 per evidence item, capped at 0.9.
    ``id`` is the stable SHA-256 of the canonical structure (computed
    in ``__post_init__`` when not supplied)."""

    name: str
    kind: str  # "unreliable_toolless_opener" (P-A) | "one_use_tool" (P-B)
               # | "durable_tool" (P-C) | "requires_precondition" (P-D)
    effect: str
    roles: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: float = 0.5
    id: str = ""
    # Phase 6: True when the pattern was learned as a consolidation
    # CANDIDATE (P-C / P-D) and is not yet in the library. Not
    # serialised: candidate status lives in the Consolidator queue,
    # never in the pattern record itself.
    candidate: bool = False

    def __post_init__(self):
        # Normalise relations to tuples and evidence items to dicts.
        self.relations = [tuple(rel) for rel in self.relations]
        norm_evidence = []
        for item in self.evidence:
            if isinstance(item, dict):
                norm_evidence.append({
                    "world": str(item.get("world", "")),
                    "action": str(item.get("action", "")),
                    "outcome": str(item.get("outcome", ""))})
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                norm_evidence.append(_evidence_item(*item))
            else:
                norm_evidence.append({"world": "", "action": str(item),
                                      "outcome": ""})
        self.evidence = norm_evidence
        self.confidence = min(0.9, float(self.confidence))
        if not self.id:
            self.id = _canonical_id(self.kind, self.effect,
                                    self.roles, self.relations)

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "effect": self.effect, "roles": self.roles,
                "relations": [list(rel) for rel in self.relations],
                "evidence": [dict(e) for e in self.evidence],
                "confidence": float(self.confidence)}

    @classmethod
    def from_json(cls, d: dict) -> "AbstractPattern":
        return cls(name=str(d.get("name", "")), kind=str(d.get("kind", "")),
                   effect=str(d.get("effect", "")),
                   roles=[dict(r) for r in d.get("roles", [])],
                   relations=[tuple(rel) for rel in d.get("relations", [])],
                   evidence=[dict(e) for e in d.get("evidence", [])],
                   confidence=float(d.get("confidence", 0.5)),
                   id=str(d.get("id", "")))


def _episode_fields(episode):
    """Normalise one episode (dict or object) to
    (action, situation, ok, world). ``ok`` accepts ``ok`` / ``success``
    booleans or ``outcome`` values True/False/``"success"``/``"failed"``."""
    if isinstance(episode, dict):
        action = str(episode.get("action", ""))
        situation = dict(episode.get("situation", {}) or {})
        world = str(episode.get("world", ""))
        if "ok" in episode:
            ok = bool(episode["ok"])
        elif "success" in episode:
            ok = bool(episode["success"])
        else:
            outcome = episode.get("outcome", "")
            if isinstance(outcome, bool):
                ok = outcome
            else:
                ok = str(outcome).lower() in ("success", "ok", "succeeded",
                                              "met", "true", "passed")
        return action, situation, ok, world
    action = str(getattr(episode, "action", ""))
    situation = dict(getattr(episode, "situation", {}) or {})
    world = str(getattr(episode, "world", "") or "")
    if hasattr(episode, "ok"):
        ok = bool(episode.ok)
    elif hasattr(episode, "success"):
        ok = bool(episode.success)
    else:
        ok = bool(getattr(episode, "outcome", False))
    return action, situation, ok, world


def _is_toolless(pre: dict) -> bool:
    """Tool-less = no precondition key starting with ``has_`` (a
    key-shape rule, not a name comparison)."""
    return not any(str(k).startswith("has_") for k in (pre or {}))


def _tool_entity(pre: dict):
    """The entity named by the first ``has_<entity>`` precondition key,
    or None when the action is tool-less."""
    for k in (pre or {}):
        if str(k).startswith("has_") and len(str(k)) > 4:
            return str(k)[4:]
    return None


def _requires_keys(failures: list, successes: list) -> list:
    """Keys K missing (absent-or-falsy) in every failure situation
    and present-and-truthy in at least one success situation, in
    sorted order. The P-D / C-2 precondition shape: the failures
    share the missing key, the success has it. Structural (key
    shapes only), never a name comparison."""
    if len(failures) < 2 or not successes:
        return []
    keys: set[str] = set()
    for s in successes:
        keys.update(str(k) for k, v in (s or {}).items() if v)
    return sorted(k for k in keys
                  if all(not (s or {}).get(k) for s in failures))


def _pattern_requires_key(pattern: AbstractPattern):
    """The precondition key a P-D pattern names, from its roles
    (``{"requires": K}`` or ``{"key": K}``), or None."""
    for role in (pattern.roles or []):
        features = (role or {}).get("features", {}) or {}
        if "requires" in features:
            return str(features["requires"])
        if "key" in features:
            return str(features["key"])
    return None


def _freeze_situation(situation: dict) -> tuple:
    return tuple(sorted((str(k), str(v)) for k, v in (situation or {}).items()))


def _action_items(all_actions):
    """Normalise ``all_actions`` entries (``(name, pre, eff)`` triples,
    dicts, or objects with ``.name`` / ``.pre`` / ``.eff``) to
    ``(name, pre, eff)`` triples. One shared normaliser for the
    scoping helper below. Pre/effect dicts are yielded AS-IS, never
    copied: ``match`` runs in the planner's inner loop, so copying
    every precondition/effect dict on every helper call is
    unaffordable (measured: the pre-Phase-10 ``match`` cost 6.4
    us/call; the first Phase 10 version cost 77 us/call, almost all
    of it re-normalisation). Helpers only read the triples, never
    mutate them."""
    for item in (all_actions or []):
        if isinstance(item, tuple) and len(item) == 3:
            name, pre, eff = item
        elif isinstance(item, list) and len(item) == 3:
            name, pre, eff = item
        elif isinstance(item, dict):
            name, pre, eff = (item.get("name", ""), item.get("pre", {}),
                              item.get("eff", {}))
        else:
            name, pre, eff = (getattr(item, "name", ""),
                              getattr(item, "pre", {}),
                              getattr(item, "eff", {}))
        yield (str(name),
               pre if isinstance(pre, dict) else dict(pre or {}),
               eff if isinstance(eff, dict) else dict(eff or {}))


def _scope_actions(all_actions, state):
    """Phase 10 (P10-1): scope the action list to the CURRENT world
    in ONE pass, returning ``(items, vocabulary, produces)``:

    - ``items``: the current-world actions as ``(name, pre, eff)``
      triples. The raw list may accumulate actions across worlds (a
      continued brain keeps the old world's actions in
      ``planner.actions``); an action belongs to the current world
      when every one of its precondition keys is a key of the
      current state — the state's keys are what this world reports,
      so an action demanding a key the state never mentions (W1's
      press demanding ``has_mould`` in W2, whose state carries
      ``has_cast`` instead) is an old-world action, not a current
      one. Actions with no preconditions pass vacuously. With no
      state supplied there is nothing to scope by, so the list is
      returned unfiltered.
    - ``vocabulary``: the keys the current world mentions — every
      precondition/effect key of the current-world actions, plus
      the current state's own keys. A pattern key outside this set
      names something the world does not know about at all.
    - ``produces``: ``key -> [values]`` for effects of
      current-world actions — what the world can supply, for the
      P-A satisfiability check.

    Single-pass and copy-free by design: this runs in the planner's
    inner loop, so the normalisation + scoping + vocabulary work
    must happen once per ``match`` call at most (see ``match``,
    which computes the scope lazily and reuses it for P-A and P-D)."""
    state_keys = {str(k) for k in (state or {})}
    items: list = []
    vocabulary: set = set(state_keys)
    produces: dict = {}
    if state is None:
        for name, pre, eff in _action_items(all_actions):
            items.append((name, pre, eff))
            vocabulary.update(pre)
            vocabulary.update(eff)
            for k, v in eff.items():
                if str(k).startswith("has_"):
                    produces.setdefault(str(k), []).append(v)
        return items, vocabulary, produces
    for name, pre, eff in _action_items(all_actions):
        pre_keys = set(pre)
        if not pre_keys <= state_keys:
            continue
        items.append((name, pre, eff))
        vocabulary.update(pre_keys)
        vocabulary.update(eff)
        for k, v in eff.items():
            if str(k).startswith("has_"):
                produces.setdefault(str(k), []).append(v)
    return items, vocabulary, produces


def _preconditions_satisfiable(pre: dict, state, produces) -> bool:
    """Phase 10 (P10-1): is one variant's precondition set
    satisfiable in the current state? A precondition already true in
    ``state`` is satisfied. Non-``has_`` keys are ordinary state the
    brain can change by acting (navigation, opening doors), so a
    current mismatch does not make them unsatisfiable — the planner,
    not this check, judges reachability. A ``has_<entity>``
    requirement not met in the state is satisfiable only when the
    entity is obtainable in this world: ``produces`` (from
    ``_scope_actions``) lists the values some current-world action's
    effect produces for the key. A variant demanding a tool no
    action in this world can supply is a phantom alternative, not a
    real one."""
    state = dict(state or {})
    for k, v in (pre or {}).items():
        k = str(k)
        if k in state and state[k] == v:
            continue
        if not (k.startswith("has_") and len(k) > 4):
            continue
        if v in (produces.get(k) or []):
            continue
        return False
    return True


def _variant_preconditions(variant, pre: dict) -> dict:
    """Phase 10 (P10-1): the evaluated variant's precondition dict.
    ``variant`` is the canonical variant string from
    ``Planner.variant_of`` (``"k=v;k=v"`` with ``repr`` values) or a
    precondition dict; ``pre`` is the fallback when ``variant`` is
    None or unparseable (direct ``match`` callers that do not pass a
    variant keep working exactly as before)."""
    if variant is None:
        return dict(pre or {})
    if isinstance(variant, dict):
        return dict(variant)
    if isinstance(variant, str):
        if not variant:
            return {}
        parsed: dict = {}
        try:
            for part in variant.split(";"):
                key, _, value_text = part.partition("=")
                if not key:
                    return dict(pre or {})
                parsed[key] = ast.literal_eval(value_text)
        except (ValueError, SyntaxError):
            return dict(pre or {})
        return parsed
    return dict(pre or {})


class PatternLibrary:
    """The learned pattern store. Patterns are learned, never written
    by hand for a world; duplicates (same stable id) are never relearned."""

    def __init__(self):
        self._patterns: list[AbstractPattern] = []
        # Phase 9 §4.2: retired patterns — records of patterns a
        # human (or the >= 3 contradicting-observations trigger)
        # retired. A retired id is never re-added (see _add): the
        # library remembers the retirement so the same evidence can
        # never silently resurrect a pattern that was judged wrong.
        self._retired: list[dict] = []

    def patterns(self) -> list[AbstractPattern]:
        return list(self._patterns)

    def retired_patterns(self) -> list[dict]:
        """Retired-pattern records (copies): each has ``id``,
        ``name``, ``kind``, ``effect``, ``reason`` and ``cycle``."""
        return [dict(r) for r in self._retired]

    def is_retired(self, pattern_id: str) -> bool:
        return any(r.get("id") == str(pattern_id) for r in self._retired)

    def retire_pattern(self, pattern_id: str, reason: str,
                       cycle: int | None = None) -> dict | None:
        """Retire the pattern with ``pattern_id``: remove it from
        the library so it is never applied again, and record the
        retirement (never silent). Returns the retired record, or
        None when no such pattern is in the library (unknown id, or
        already retired)."""
        pattern_id = str(pattern_id)
        for i, p in enumerate(self._patterns):
            if p.id == pattern_id:
                removed = self._patterns.pop(i)
                record = {"id": removed.id, "name": removed.name,
                          "kind": removed.kind, "effect": removed.effect,
                          "reason": str(reason or ""),
                          "cycle": int(cycle) if cycle is not None else None}
                self._retired.append(record)
                return dict(record)
        return None

    def by_kind(self, kind: str) -> list[AbstractPattern]:
        return [p for p in self._patterns if p.kind == kind]

    def _add(self, pattern: AbstractPattern):
        """Add unless a pattern with the same id already exists, or
        the id was retired (Phase 9 §4.2: a retired pattern is never
        re-added, however the evidence re-arrives).
        Returns the pattern when newly added, else None."""
        if self.is_retired(pattern.id):
            return None
        if any(p.id == pattern.id for p in self._patterns):
            return None
        self._patterns.append(pattern)
        return pattern

    def add_pattern(self, pattern: AbstractPattern):
        """Phase 6: add a (previously candidate) pattern to the
        library — the promotion step. The pattern's ``candidate``
        flag is cleared on admission. Returns the pattern when newly
        added, else None (duplicate id: already durable)."""
        pattern = AbstractPattern(
            name=pattern.name, kind=pattern.kind, effect=pattern.effect,
            roles=[dict(r) for r in pattern.roles],
            relations=[tuple(rel) for rel in pattern.relations],
            evidence=[dict(e) for e in pattern.evidence],
            confidence=pattern.confidence, id=pattern.id)
        pattern.candidate = False
        return self._add(pattern)

    def learn_from_episodes(self, episodes, action_effects=None,
                            world_model=None) -> list[AbstractPattern]:
        """Learn patterns from episodes.

        ``episodes`` are dicts ``{action, situation, outcome}`` (or
        objects with ``.action`` / ``.situation`` / ``.ok``).
        ``action_effects`` maps action name -> its effect dict; an
        action is an "opener" for E when E is in its effects.
        ``world_model`` is accepted for interface compatibility with
        the design's call shape and is not needed by any rule.

        Returns the newly learned patterns. P-A / P-B are added to
        the library immediately, as before (duplicates are skipped,
        never relearned). P-C / P-D are returned as CANDIDATES
        (``candidate=True``) and are NOT added: the Brain's
        consolidation pass promotes them via ``add_pattern`` once
        the promotion bar is met, so a P-C/P-D pattern in the
        return value is not yet durable knowledge."""
        action_effects = action_effects or {}
        parsed = [_episode_fields(ep) for ep in (episodes or [])]
        # Group by world label when present ("" groups everything).
        worlds: dict[str, list] = {}
        for item in parsed:
            worlds.setdefault(item[3], []).append(item)
        learned: list[AbstractPattern] = []
        for world, items in sorted(worlds.items()):
            learned.extend(self._learn_pa(world, items, action_effects))
            learned.extend(self._learn_pb(world, items, action_effects))
            learned.extend(self._learn_pc(world, items, action_effects))
            learned.extend(self._learn_pd(world, items, action_effects))
        return learned

    def _learn_pc(self, world, items, action_effects) -> list[AbstractPattern]:
        """P-C (durable tool, Phase 6 §2.1): an action SUCCEEDED
        >= 3 times across >= 2 distinct situations for the same
        effect, with no observed failure of that action. Returned
        as candidates (``candidate=True``), never added here."""
        learned: list[AbstractPattern] = []
        seen_ids: set[str] = set()
        by_action: dict[str, list] = {}
        for action, situation, ok, _w in items:
            by_action.setdefault(action, []).append((situation, ok))
        for action in sorted(by_action):
            rows = by_action[action]
            if any(not ok for _s, ok in rows):
                continue  # an observed break: not durable
            successes = [s for s, ok in rows if ok]
            if len(successes) < 3:
                continue
            if len({_freeze_situation(s) for s in successes}) < 2:
                continue
            eff = action_effects.get(action) or {}
            for effect in sorted(str(k) for k in eff):
                evidence = [_evidence_item(world, action, "success")
                            for _s in successes]
                pattern = AbstractPattern(
                    name=f"durable_tool:{effect}",
                    kind="durable_tool",
                    effect=effect,
                    roles=[{"name": "tool",
                            "features": {"breaks_after_first_use": False}},
                           {"name": "fragile_tool",
                            "features": {"breaks_after_first_use": True}}],
                    relations=[("tool", "produces", effect)],
                    evidence=evidence,
                    confidence=min(0.9, 0.5 + 0.1 * len(evidence)),
                    candidate=True)
                if pattern.id in seen_ids:
                    continue
                seen_ids.add(pattern.id)
                learned.append(pattern)
        return learned

    def _learn_pd(self, world, items, action_effects) -> list[AbstractPattern]:
        """P-D (precondition discovery, Phase 6 §2.1): an action
        FAILED >= 2 times in situations sharing a missing key K (K
        absent-or-falsy in every failure) and SUCCEEDED >= 1 time
        with K present, for the same effect. One candidate per
        (effect, key), in sorted-key order. Returned as candidates
        (``candidate=True``), never added here."""
        learned: list[AbstractPattern] = []
        seen_ids: set[str] = set()
        by_action: dict[str, list] = {}
        for action, situation, ok, _w in items:
            by_action.setdefault(action, []).append((situation, ok))
        for action in sorted(by_action):
            rows = by_action[action]
            failures = [s for s, ok in rows if not ok]
            successes = [s for s, ok in rows if ok]
            if len(failures) < 2 or not successes:
                continue
            keys = _requires_keys(failures, successes)
            if not keys:
                continue
            eff = action_effects.get(action) or {}
            evidence = ([_evidence_item(world, action, "failed")
                         for _s in failures]
                        + [_evidence_item(world, action, "success")
                           for _s in successes])
            for effect in sorted(str(k) for k in eff):
                for key in keys:
                    pattern = AbstractPattern(
                        name=f"requires_{key}:{effect}",
                        kind="requires_precondition",
                        effect=effect,
                        roles=[{"name": "action",
                                "features": {"requires": key}},
                               {"name": "precondition",
                                "features": {"key": key}}],
                        relations=[("action", "requires", key)],
                        evidence=list(evidence),
                        confidence=min(0.9, 0.5 + 0.1 * len(evidence)),
                        candidate=True)
                    if pattern.id in seen_ids:
                        continue
                    seen_ids.add(pattern.id)
                    learned.append(pattern)
        return learned

    def _learn_pa(self, world, items, action_effects) -> list[AbstractPattern]:
        """P-A: for one claimed effect E, some tool-less opener FAILED
        while a tool-based opener SUCCEEDED for the same effect."""
        learned: list[AbstractPattern] = []
        effects: set[str] = set()
        for eff in action_effects.values():
            effects.update(str(k) for k in (eff or {}))
        for effect in sorted(effects):
            failed_toolless = []
            succeeded_tooled = []
            for action, situation, ok, _w in items:
                eff = action_effects.get(action)
                if not eff or effect not in eff:
                    continue
                if ok and not _is_toolless(situation):
                    succeeded_tooled.append((action, situation))
                elif not ok and _is_toolless(situation):
                    failed_toolless.append((action, situation))
            if not failed_toolless or not succeeded_tooled:
                continue  # one failure alone learns nothing: no contrast
            evidence = ([_evidence_item(world, a, "failed")
                         for a, _s in failed_toolless]
                        + [_evidence_item(world, a, "success")
                           for a, _s in succeeded_tooled])
            pattern = AbstractPattern(
                name=f"unreliable_toolless_opener:{effect}",
                kind="unreliable_toolless_opener",
                effect=effect,
                roles=[{"name": "opener",
                        "features": {"requires_object": False}},
                       {"name": "tool_opener",
                        "features": {"requires_object": True}}],
                relations=[("tool_opener", "produces", effect)],
                evidence=evidence,
                confidence=min(0.9, 0.5 + 0.1 * len(evidence)))
            added = self._add(pattern)
            if added is not None:
                learned.append(added)
        return learned

    def _learn_pb(self, world, items, action_effects) -> list[AbstractPattern]:
        """P-B: an action SUCCEEDED once and then FAILED in the same
        situation for the same effect."""
        learned: list[AbstractPattern] = []
        # (action, frozen situation) -> ordered outcomes (in episode order).
        by_pair: dict[tuple, list[bool]] = {}
        for action, situation, ok, _w in items:
            by_pair.setdefault((action, _freeze_situation(situation)),
                               []).append(ok)
        for (action, _sit), outcomes in sorted(by_pair.items()):
            if True not in outcomes:
                continue
            first_success = outcomes.index(True)
            if not any(not o for o in outcomes[first_success + 1:]):
                continue  # no failure after the success: not one-use
            eff = action_effects.get(action) or {}
            for effect in sorted(str(k) for k in eff):
                evidence = [_evidence_item(world, action, "success"),
                            _evidence_item(world, action, "failed")]
                pattern = AbstractPattern(
                    name=f"one_use_tool:{effect}",
                    kind="one_use_tool",
                    effect=effect,
                    roles=[{"name": "tool",
                            "features": {"breaks_after_first_use": True}},
                           {"name": "durable_tool",
                            "features": {"breaks_after_first_use": False}}],
                    relations=[("tool", "produces", effect),
                               ("durable_tool", "produces", effect)],
                    evidence=evidence,
                    confidence=min(0.9, 0.5 + 0.1 * len(evidence)))
                added = self._add(pattern)
                if added is not None:
                    learned.append(added)
        return learned

    def match(self, action_name: str, action_pre, action_effects,
              world_model=None, all_actions=None, state=None,
              variant=None) -> float:
        """Cost multiplier for one action in the current state.

        - 6.0 when the action is a tool-less opener for an effect
          covered by a P-A pattern AND a tool-based alternative opener
          for that effect exists among ``all_actions`` (a list of
          ``(name, pre, eff)`` triples) whose preconditions are
          satisfiable in the current state (Phase 10, P10-1: when
          ``state`` is supplied, a variant demanding a tool no action
          in this world can supply does not count; with no ``state``
          the old presence check applies, so direct callers that pass
          no state keep working). The alternative is compared by
          variant, so a tooled variant of the same action name counts
          (the learned contrast is situational); only the evaluated
          variant itself is skipped.
        - 6.0 when the action's tool entity is marked single-use
          suspect in the world model (property ``single_use_suspect``
          or ``breaks_after_first_use`` True) and a P-B pattern covers
          one of its effects.
        - 0.8 when the action's tool entity is marked durable in the
          world model (property ``breaks_after_first_use`` False) and
          a P-C pattern covers one of its effects; 6.0 when that
          tool is instead marked suspect/breaking and a P-C pattern
          covers the effect (a known-fragile retry).
        - 6.0 when a P-D pattern covers one of the action's effects
          and ALL of (Phase 10, P10-1): (a) the effect is among the
          variant's effects; (b) the required key is in the current
          world's vocabulary — it appears in some current-world
          action's preconditions or effects, or in the current state
          itself (the action list is scoped to the current world by
          the state's keys, so a continued brain's accumulated
          old-world actions do not count); a key mentioned nowhere in
          this world makes the pattern inapplicable here, never
          firing; (c) the evaluated variant
          does not already require the key truthy in its own
          preconditions (then the penalty is vacuous — the variant
          cannot run without the key anyway); (d) the key is
          absent-or-falsy in ``state``. With no ``state`` supplied a
          P-D pattern never fires — the precondition cannot be
          checked without the state.
        - 1.0 otherwise.

        ``variant`` is the evaluated action's canonical variant
        (``Planner.variant_of``); it identifies which same-named
        variant is being priced. It is optional: ``match`` works
        without it, using ``action_pre`` as the variant's
        preconditions, so existing direct callers are unaffected.

        Penalties are checked before the durable bonus, so a suspect
        tool is never also rewarded. ``action_effects`` is the
        action's own effect dict (a global action->effects map is
        also accepted: the entry for ``action_name`` is used then)."""
        action_name = str(action_name)
        pre = dict(action_pre or {})
        if isinstance(action_effects, dict) and action_name in action_effects \
                and isinstance(action_effects[action_name], dict):
            eff = dict(action_effects[action_name])
        else:
            eff = dict(action_effects or {})
        if not eff:
            return 1.0
        penalty = False
        bonus = False
        # The current world's scope (Phase 10, P10-1) — the scoped
        # action list, the vocabulary and the produces map, computed
        # in ONE pass by ``_scope_actions`` — and the evaluated
        # variant's own preconditions (``variant`` is the canonical
        # identity passed by the planner; ``pre`` is the fallback for
        # direct callers that pass no variant). Both are computed
        # LAZILY — only when a P-A or P-D pattern actually covers
        # this action's effect — because match() runs in the
        # planner's inner loop and the scope pass is its dominant
        # cost (measured: the first Phase 10 version cost 77 us/call
        # against 6.4 us/call pre-Phase-10, almost all of it
        # re-scoping the action list on every helper call; the
        # single-pass lazy version below measures ~30 us/call —
        # ~5x the old cost, but the absolute overhead is small:
        # a 26-cycle workshop run makes ~6,700 match calls, so
        # ~0.16 s per run. No further optimisation is warranted:
        # the remaining cost is the unavoidable per-call scope pass
        # over ~35 actions plus one variant-string parse.)
        _scope = None

        def scope():
            nonlocal _scope
            if _scope is None:
                _scope = _scope_actions(all_actions, state)
            return _scope

        _variant_pre = None
        _variant_pre_done = False

        def variant_pre():
            nonlocal _variant_pre, _variant_pre_done
            if not _variant_pre_done:
                _variant_pre = _variant_preconditions(variant, pre)
                _variant_pre_done = True
            return _variant_pre

        # P-A: tool-less opener with a tool-based alternative available.
        if _is_toolless(pre):
            for pattern in self.by_kind("unreliable_toolless_opener"):
                if pattern.effect not in eff:
                    continue
                items, _vocab, produces = scope()
                if self._has_tooled_alternative(pattern.effect, action_name,
                                                items, state=state,
                                                variant_pre=variant_pre(),
                                                produces=produces):
                    penalty = True
        # P-B / P-C: the action's tool entity in the world model.
        tool = _tool_entity(pre)
        if tool is not None and world_model is not None:
            suspect = False
            durable = False
            ent = world_model.entity(tool)
            if ent is not None:
                for key in ("single_use_suspect", "breaks_after_first_use"):
                    prop = ent.properties.get(key)
                    if prop is not None and prop.value is True:
                        suspect = True
                prop = ent.properties.get("breaks_after_first_use")
                if prop is not None and prop.value is False:
                    durable = True
            if suspect:
                for pattern in self.by_kind("one_use_tool"):
                    if pattern.effect in eff:
                        penalty = True
                for pattern in self.by_kind("durable_tool"):
                    if pattern.effect in eff:
                        penalty = True
            if durable:
                for pattern in self.by_kind("durable_tool"):
                    if pattern.effect in eff:
                        bonus = True
        # P-D: the required key must hold in the current state — and
        # the pattern must apply to THIS world at all (Phase 10,
        # P10-1). The key is in the world's vocabulary when some
        # current-world action mentions it (precondition or effect)
        # or the current state itself carries it; a key mentioned
        # nowhere in this world (e.g. W1's has_mould in W2, which
        # uses has_cast) can never fire here, whatever the pattern
        # learned elsewhere. The vocabulary gate needs the world's
        # action list to judge inapplicability: when no action list
        # is supplied at all, the gate is inactive (fail open to the
        # old behaviour — a key absent-or-falsy in the state fires),
        # so direct callers that pass no action list keep working.
        # A variant that already requires the key truthy is skipped:
        # it cannot run without the key anyway, so penalising it is
        # vacuous.
        if state is not None:
            state = dict(state)
            pd_patterns = [p for p in self.by_kind("requires_precondition")
                           if p.effect in eff]
            if pd_patterns:
                if all_actions:
                    _items, vocabulary, _produces = scope()
                else:
                    vocabulary = None
                vpre = variant_pre()
                for pattern in pd_patterns:
                    key = _pattern_requires_key(pattern)
                    if key is None:
                        continue
                    if vocabulary is not None and key not in vocabulary:
                        continue
                    if vpre.get(key):
                        continue
                    if not state.get(key):
                        penalty = True
        if penalty:
            return PATTERN_PENALTY
        if bonus:
            return 0.8
        return 1.0

    @staticmethod
    def _has_tooled_alternative(effect: str, action_name: str, items,
                                state=None, variant_pre=None,
                                produces=None) -> bool:
        """True when ``items`` (the current-world ``(name, pre, eff)``
        triples from ``_scope_actions``) contains a tool-based action
        whose effects include ``effect`` — and (Phase 10, P10-1, when
        ``state`` is supplied) whose preconditions are satisfiable
        in the current state: a variant demanding a tool this world
        cannot supply is not a real alternative (``produces`` is the
        scope's produces map). With no ``state`` the old presence
        check applies, so direct callers that pass no state keep
        working.

        Phase 10 (P10-1): the alternative is compared by VARIANT,
        not by name. The learned P-A contrast is situational — a
        tool-less failure versus a tooled success for the same effect
        — and the two episodes may be variants of ONE action name
        (the workshop press: cheap vs with-mould), so a tooled
        variant of the same name IS an alternative. Only the
        evaluated variant itself is skipped (same precondition dict);
        when ``variant_pre`` is None the legacy same-name skip
        applies instead."""
        if not items:
            return False
        for name, pre, eff in items:
            if variant_pre is not None:
                if (pre or {}) == (variant_pre or {}):
                    continue
            elif str(name) == str(action_name):
                continue
            if effect in (eff or {}) and not _is_toolless(pre or {}):
                if state is None:
                    return True
                if _preconditions_satisfiable(pre or {}, state,
                                              produces or {}):
                    return True
        return False

    # ---- persistence ---------------------------------------------------

    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "patterns": [p.to_json() for p in self._patterns],
                "retired": [dict(r) for r in self._retired]}

    @classmethod
    def from_json(cls, data: dict) -> "PatternLibrary":
        if not isinstance(data, dict):
            raise ValueError("pattern library file must be a dict")
        lib = cls()
        for d in data.get("patterns", []):
            lib._patterns.append(AbstractPattern.from_json(d))
        for r in data.get("retired", []) or []:
            if isinstance(r, dict) and r.get("id"):
                lib._retired.append(dict(r))
        return lib

