"""Planner: uniform-cost search over a symbolic action model. Reliability feeds back into cost."""
from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import asdict, dataclass, field


def _tool_entity(pre: dict):
    """The entity named by the first ``has_<entity>`` precondition key,
    or None when the action is tool-less (same key-shape rule as
    ``ab.world.patterns``)."""
    for k in (pre or {}):
        if str(k).startswith("has_") and len(str(k)) > 4:
            return str(k)[4:]
    return None


def canonical_variant(pre: dict) -> str:
    """Phase 9 (P9-1): the canonical identity of one action variant —
    its precondition set as sorted ``key=value`` pairs. Two variants
    of one action name differ here exactly when their preconditions
    differ, so the string identifies the variant across worlds,
    save/load and simulations (unlike an object id). An action with
    no preconditions has the empty-string variant."""
    return ";".join(f"{k}={v!r}"
                    for k, v in sorted((pre or {}).items(),
                                       key=lambda kv: str(kv[0])))


def canonical_args(args: dict) -> str:
    """Phase 10.5 (P10.5-1): the canonical identity of an action's
    SEMANTIC args — sorted ``key=value`` pairs, same format as
    ``canonical_variant``. Keys starting with ``_`` are dispatch keys
    (``_do``) and are excluded from identity; everything else
    (``material``, ``room``, ...) is included, so two same-named,
    same-precondition variants that differ only in args (one world's
    cut-zinc vs another world's cut-tin, both ``cut`` in a shared
    room with an awl) have different identities. An action with no
    semantic args (``press`` carries only ``_do``) has the
    empty-string identity, exactly like an action with no args at
    all."""
    return ";".join(f"{k}={v!r}"
                    for k, v in sorted((args or {}).items(),
                                       key=lambda kv: str(kv[0]))
                    if not str(k).startswith("_"))


def action_identity(action: "Action") -> tuple:
    """Phase 10.5 (P10.5-1): the full identity of one action object —
    ``(name, canonical_variant(pre), canonical_args(args))``. This is
    the set of facts that determines what the action DOES in the
    world: its name, the state it requires, and the parameters it is
    invoked with. Bans, failure rates and learner counts key on it;
    ``Planner.variant_of`` keeps returning the precondition variant
    alone (the pattern library's key)."""
    return (action.name,
            canonical_variant(getattr(action, "pre", None) or {}),
            canonical_args(getattr(action, "args", None) or {}))


@dataclass
class Action:
    name: str
    pre: dict                           # required state
    eff: dict                           # expected state changes
    cost: float = 1.0
    args: dict = field(default_factory=dict)
    tool: str | None = None             # if set, executed through the tool registry
    store_as: str | None = None         # state key that receives a tool's output


@dataclass
class Plan:
    steps: list
    cost: float


def _freeze(state: dict) -> tuple:
    """Hashable snapshot of a state, for the search's visited set.

    Phase 2: values are normally hashable scalars, but a tool result
    stored via Action.store_as can be a dict (or another unhashable
    value). Such values are encoded canonically (sorted JSON) instead
    of crashing the search with TypeError: unhashable type. Hashable
    values are kept as-is, so existing search behaviour is unchanged.
    """
    def _h(v):
        try:
            hash(v)
            return v
        except TypeError:
            return ("__unhashable__", json.dumps(v, sort_keys=True, default=str))
    return tuple(sorted((k, _h(v)) for k, v in state.items()))


class Planner:
    def __init__(self, actions: list[Action] | None = None, max_nodes: int = 50000):
        self.actions: list[Action] = list(actions or [])
        self.max_nodes = max_nodes
        # Fallback counts, written by record() when no learner is
        # attached (standalone Planner use, e.g. test_planning). When a
        # learner IS attached, ``stats`` is derived from its episodes
        # (see the property below) and this dict is only the fallback.
        self._stats: dict[str, list[int]] = {}
        self._learner = None       # Learner, attached by Brain (Phase 3)
        self._rules = None         # RuleEngine, attached by Brain (Phase 3)
        # Phase 10.5 (P10.5-2, P7-10): optional redactor for action
        # args, set by Brain to its spec-aware ``_redact_tool_args``
        # (same hook pattern as ``execution_monitor.redact_args``).
        # Persistence never carries secrets, so every identity the
        # planner derives from an action's args (ban keys, failure
        # rates) is computed on the REDACTED args; a secret-marked
        # tool argument is "[REDACTED]" in every key, and no
        # persisted record (learner.json, plan.json) can leak it.
        # None (standalone Planner use) means no redaction — the
        # identities are then the raw canonical args, as before.
        self.redact_args = None    # callable(tool, args) -> args
        # Failed (action, args-identity, situation) triples: a ban
        # blocks an action only in the same situation it failed in
        # AND with the same semantic args (goal-scoped, Phase 2;
        # args-aware, Phase 10.5 P10.5-3). Two same-named variants
        # that differ only in args (cut-zinc vs cut-tin) ban
        # independently: banning one must not block the other.
        self.banned_pairs: set[tuple[str, str, tuple]] = set()
        # Phase 10.5 (P10.5-9): action generations — which world an
        # action came from. The counter advances when the brain
        # starts a new environment (``mark_generation``, called from
        # ``Brain.clear_transient_for_new_world``); actions added
        # after that are tagged with the new generation, actions
        # carried over keep their old one. ``plan_all`` breaks exact
        # cost ties in favour of the current world's actions (see
        # the heap key there): on a genuine tie the brain prefers an
        # affordance it observes here over one it remembers from
        # elsewhere. The map is keyed by ``action_identity`` (name,
        # variant, args), so it survives list reassignments (the
        # benchmark runner filters ``planner.actions`` directly) and
        # save/load (persisted as a parallel list in ``to_json``).
        # Untagged actions (added before any mark, or restored from
        # a pre-10.5 save) default to generation 0 — the oldest —
        # which is the safe direction: they never win a tie over a
        # current-world action.
        self._generation: int = 0
        self._action_generation: dict[tuple, int] = {}
        for _a in self.actions:
            self._action_generation[action_identity(_a)] = 0
        # Phase 5: cost preferences from approved induced success rules
        # (action name -> cost factor, e.g. 0.8) and the attached
        # pattern library (see attach_patterns). Both are cost priors,
        # never bans: they only change effective_cost.
        self.preferred_actions: dict[str, float] = {}
        self._patterns = None      # PatternLibrary, attached by Brain
        self._patterns_context = None  # callable -> (world_model, all_actions[, state])
        # Phase 6: the learned-concept store (ConceptStore, attached
        # by Brain via attach_concepts). Read only for the
        # durable_tool preference in concept_multiplier.
        self._concepts = None      # ConceptStore, attached by Brain

    def attach_concepts(self, store) -> None:
        """Attach the Phase 6 concept store (mirrors
        attach_patterns). When none is attached the concept
        multiplier is 1.0 and planning is unchanged."""
        self._concepts = store

    def concept_multiplier(self, a: Action) -> float:
        """Phase 6: 0.8 when the action's tool matches the learned
        ``durable_tool`` concept, else 1.0. The tool is the entity
        named by the action's first ``has_<entity>`` precondition
        (falling back to the action's ``tool`` attribute); its
        features come from the world model in the attached pattern
        context (properties such as ``breaks_after_first_use``).
        The store's ``is_a`` answers from its learned examples, so
        a concept with no positive example never matches. A cost
        prior, never a ban."""
        store = self._concepts
        if store is None:
            return 1.0
        tool = _tool_entity(a.pre) or (a.tool if isinstance(a.tool, str) else None)
        if not tool:
            return 1.0
        features: dict = {}
        if callable(self._patterns_context):
            try:
                ctx = self._patterns_context()
                world_model = ctx[0] if ctx else None
            except Exception:
                world_model = None
            if world_model is not None:
                try:
                    ent = world_model.entity(tool)
                except Exception:
                    ent = None
                if ent is not None:
                    features = {k: p.value for k, p in
                                (getattr(ent, "properties", {}) or {}).items()}
        try:
            if hasattr(store, "is_a"):
                ok, _conf = store.is_a(features, "durable_tool")
                return 0.8 if ok else 1.0
            if hasattr(store, "matches"):
                ok, _conf = store.matches("durable_tool", features)
                return 0.8 if ok else 1.0
        except Exception:
            return 1.0
        return 1.0

    def attach_learner(self, learner) -> None:
        """Attach the Learner that is the single store of outcomes
        (Phase 3 §2.4): ``stats`` is then derived from its episodes."""
        self._learner = learner

    def attach_rules(self, rules) -> None:
        """Attach the RuleEngine whose ACTIVE avoid_action rules the
        planner must consult before plan_all (Phase 3 §3.1)."""
        self._rules = rules

    def attach_patterns(self, library, context_fn) -> None:
        """Attach the Phase 5 pattern library. ``context_fn`` is a
        callable returning ``(world_model, all_actions)`` at call time
        (the brain passes a lambda reading its live world model and
        action list), because ``effective_cost`` has no world model of
        its own. When no library is attached the pattern multiplier
        is 1.0 and planning is unchanged."""
        self._patterns = library
        self._patterns_context = context_fn

    def pattern_multiplier(self, a: Action) -> float:
        """Cost multiplier for one action from the attached pattern
        library (1.0 when no library is attached or no pattern fires).
        The library's ``match`` decides, given the current world
        model, action list and (Phase 6, for P-D) current state from
        the attached context callable — which may return a 2-tuple
        ``(world_model, all_actions)`` or a 3-tuple with the state
        appended. Older 2-tuple contexts keep working."""
        library = self._patterns
        if library is None:
            return 1.0
        world_model, all_actions, state = None, None, None
        if callable(self._patterns_context):
            try:
                ctx = self._patterns_context()
            except Exception:
                return 1.0
            if isinstance(ctx, (list, tuple)):
                world_model = ctx[0] if len(ctx) > 0 else None
                all_actions = ctx[1] if len(ctx) > 1 else None
                state = ctx[2] if len(ctx) > 2 else None
        try:
            return float(library.match(a.name, a.pre, a.eff,
                                       world_model=world_model,
                                       all_actions=all_actions,
                                       state=state,
                                       variant=self.variant_of(a)))
        except TypeError:
            try:
                return float(library.match(a.name, a.pre, a.eff,
                                           world_model=world_model,
                                           all_actions=all_actions))
            except Exception:
                return 1.0
        except Exception:
            return 1.0

    @property
    def stats(self) -> dict[str, list[int]]:
        """Derived view (Phase 3 §2.4, read-only): action name ->
        [successes, failures], aggregated over the attached learner's
        episodes. The learner is the single store; the planner keeps no
        second copy. The returned dict is a fresh aggregation —
        mutating it changes nothing.

        Outcomes recorded directly via ``record()`` — without going
        through Learner.record() — are merged in too: every
        Learner.record() also calls ``record()``, so the direct-write
        counts are always a superset of the episode counts per action,
        and the merge takes the per-component max, which counts each
        outcome exactly once whether it arrived via the learner or
        directly. Limitation: directly-recorded outcomes live only in
        memory (they are not learner episodes, so they do not survive
        save/load); all Brain recording goes through the learner, so
        the two views agree in normal use."""
        learner = self._learner
        if learner is not None and hasattr(learner, "episodes"):
            agg: dict[str, list[int]] = {}
            for ep in learner.episodes:
                s = agg.setdefault(ep.action, [0, 0])
                s[0 if ep.ok else 1] += 1
            for name, (s2, f2) in self._stats.items():
                s, f = agg.get(name, [0, 0])
                agg[name] = [max(s, s2), max(f, f2)]
            return agg
        return {k: list(v) for k, v in self._stats.items()}

    @stats.setter
    def stats(self, value) -> None:
        # Compat setter: Planner.from_json on a legacy planner.json, and
        # any caller that assigns counts directly, writes the fallback.
        self._stats = {k: [int(v[0]), int(v[1])] for k, v in dict(value).items()}

    def _identity_args(self, args: dict | None, tool=None) -> str:
        """Canonical semantic-args identity for ``args``, with the
        attached redactor applied first when one is set (see
        ``redact_args`` above). The tool name selects the spec whose
        ``secret`` markings apply; with no tool the redactor falls
        back to its secret-key/token heuristics. Never raises: a
        failing redactor leaves the args unredacted rather than
        breaking planning."""
        if callable(self.redact_args):
            try:
                args = self.redact_args(tool, dict(args or {}))
            except Exception:
                pass
        return canonical_args(args)

    @staticmethod
    def _freeze_situation(situation: dict) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in situation.items()))

    def ban(self, action: str, situation: dict, args: dict | None = None,
            tool=None) -> None:
        """Ban ``action`` in exactly ``situation`` (its preconditions'
        values) with exactly these semantic args (Phase 10.5
        P10.5-3). ``args=None`` means empty args (backward compatible
        for existing callers and tests). ``tool`` names the tool whose
        spec the redactor consults; pass the action's tool so a
        secret-marked argument is "[REDACTED]" in the persisted key."""
        self.banned_pairs.add((action, self._identity_args(args, tool),
                               self._freeze_situation(situation)))

    def is_banned(self, action: str, situation: dict, args: dict | None = None,
                  tool=None) -> bool:
        return (action, self._identity_args(args, tool),
                self._freeze_situation(situation)) in self.banned_pairs

    def clear_bans_for_keys(self, keys) -> int:
        """Phase 5: drop every banned pair whose recorded situation
        contains any of ``keys`` — the world model has revised those
        keys, so a ban keyed to the old model no longer applies.
        Returns the number of pairs removed."""
        keys = {str(k) for k in (keys or [])}
        if not keys:
            return 0
        before = len(self.banned_pairs)
        self.banned_pairs = {(a, args_id, frozen)
                             for a, args_id, frozen in self.banned_pairs
                             if not any(k in keys for k, _v in frozen)}
        return before - len(self.banned_pairs)

    @staticmethod
    def sequence_banned(steps, world: dict, banned_pairs, redact_args=None) -> bool:
        """Phase 4: is ANY step of a whole plan sequence banned in the
        situation it would run in? The state is simulated forward
        through the steps' effects, so a step's situation is the world
        as the preceding steps would leave it — a ban on a LATER step
        is caught, not just a ban on the first step. ``steps`` are
        Actions (or plain action dicts); ``banned_pairs`` is the
        planner's ``(action, args-identity, frozen-situation)`` set.
        Phase 10.5 (P10.5-3): each step is checked with its own
        args, so a ban on one variant (cut-zinc) does not flag a
        sequence that runs a different variant (cut-tin). Plain
        dict steps without an ``"args"`` entry count as empty args.
        ``redact_args`` is the optional callable(tool, args) -> args
        (Brain passes its spec-aware redactor): a step's args are
        redacted before the identity is computed, so the check
        matches bans recorded on redacted args and no secret-marked
        argument leaks into the comparison. With no redactor the
        raw canonical args are used, as before."""
        banned_pairs = banned_pairs or set()
        cur = dict(world or {})
        for step in (steps or []):
            if isinstance(step, dict):
                name, pre, eff = step.get("name", ""), dict(step.get("pre", {})), dict(step.get("eff", {}))
                args = step.get("args", {})
                tool = step.get("tool")
            else:
                name, pre, eff = step.name, dict(step.pre), dict(step.eff)
                args = getattr(step, "args", None) or {}
                tool = getattr(step, "tool", None)
            if callable(redact_args):
                try:
                    args = redact_args(tool, dict(args))
                except Exception:
                    pass
            situation = {k: cur.get(k) for k in pre}
            if (name, canonical_args(args),
                    Planner._freeze_situation(situation)) in banned_pairs:
                return True
            cur.update(eff)
        return False

    def resolve_action(self, name: str, state: dict) -> Action | None:
        """Phase 9 (P9-1): the variant of ``name`` that applies in
        ``state`` — the FIRST action with that name whose
        preconditions are satisfied in ``state`` (not the first in
        list order). None when no same-named variant applies."""
        for a in self.actions:
            if a.name == name and self.applicable(a, state):
                return a
        return None

    @staticmethod
    def variant_of(action: Action) -> str:
        """The canonical variant identity of one action object."""
        return canonical_variant(getattr(action, "pre", None) or {})

    def variant_for(self, name: str, state: dict) -> str | None:
        """The canonical variant of the ``name`` variant that applies
        in ``state``, or None when no variant applies."""
        action = self.resolve_action(name, state)
        return None if action is None else self.variant_of(action)

    def effects_fn(self, name: str, state: dict) -> dict:
        """Phase 4: symbolic-effect function for counterfactual
        simulation (``ReasoningEngine.simulate``). Returns the effect
        dict of the FIRST variant of the named action whose
        preconditions hold in ``state`` (Phase 9, P9-1: previously
        the first variant in list order, even when its preconditions
        failed — so a repeated name such as ``cut`` resolved to a
        variant that could not run in this state and simulation
        reported "no known effect"), else ``{}`` (unknown /
        inapplicable — the simulation records that as an assumption
        instead of inventing an effect). Nothing is executed."""
        action = self.resolve_action(name, state)
        return dict(action.eff) if action is not None else {}

    def clear_bans(self) -> None:
        self.banned_pairs.clear()

    def add_action(self, action: Action) -> None:
        self.actions.append(action)
        self._action_generation[action_identity(action)] = self._generation

    def mark_generation(self) -> None:
        """Phase 10.5 (P10.5-9): the brain has entered a new
        environment. Actions added from here on are tagged with the
        new generation; carried-over actions keep theirs. Called
        from ``Brain.clear_transient_for_new_world`` — the one place
        that knows a new world has started — never from planning
        itself, so a mid-task action addition does not re-tag."""
        self._generation += 1

    def generation_of(self, action: Action) -> int:
        """Phase 10.5 (P10.5-9): the generation an action was added
        in (0 when untagged — the oldest, so it never wins a tie
        over a current-world action)."""
        return self._action_generation.get(action_identity(action), 0)

    def record(self, name: str, ok: bool) -> None:
        s = self._stats.setdefault(name, [0, 0])
        s[0 if ok else 1] += 1

    def failure_rate(self, name: str) -> float:
        s, f = self.stats.get(name, [0, 0])
        return f / (s + f) if (s + f) else 0.0

    def failure_counts_for(self, action: Action) -> list[int]:
        """Phase 10.5 (P10.5-2 exact, P10.5-8 key-shape fallback):
        ``[successes, failures]`` for one action VARIANT, from the
        attached learner's episodes. The hierarchy is:

        1. Exact: episodes with the same name AND the same canonical
           precondition variant (values) AND the same canonical
           semantic args. If any exist, their counts.
        2. Key-shape: episodes with the same name AND the same
           precondition KEY set (different values) AND the same
           canonical semantic args. If any exist, their counts. The
           room value is irrelevant to whether a press needs a
           mould; the key shape is what matters (the Phase 6
           ``repeated_mistakes`` feature-set idea applied to rates).
        3. Otherwise ``[0, 0]`` — no inheritance from a different
           key shape or different args.

        This does NOT reintroduce the P9-7 defect: the cheap press
        keys ``{at}`` and the mould press keys ``{at, has_die}`` /
        ``{at, has_form}`` are different key sets, so no-mould
        failures never inflate the with-mould rate — the key-shape
        tier compares the episode situation's FULL key set against
        the action's precondition keys (restricting to the action's
        keys first would wrongly match a mould-press episode to a
        cheap press). And it does not conflate cut-zinc with
        cut-tin: their key sets are identical, but the fallback is
        keyed by (name, keys, **args**), and zinc/tin differ in
        args. When no learner is attached there is no episode-level
        data at all, so this returns the pooled name-level counts
        (the standalone-Planner path)."""
        learner = self._learner
        if learner is None or not hasattr(learner, "episodes"):
            return list(self.stats.get(action.name, [0, 0]))
        variant = self.variant_of(action)
        args_id = self._identity_args(getattr(action, "args", None) or {},
                                      getattr(action, "tool", None))
        pre_keys = [str(k) for k in (getattr(action, "pre", None) or {})]
        pre_key_set = set(pre_keys)
        exact = [0, 0]
        shape = [0, 0]
        seen_name = False
        for ep in learner.episodes:
            if isinstance(ep, dict):
                name = str(ep.get("action", ""))
                sit = dict(ep.get("situation", {}) or {})
                ok = bool(ep.get("ok", ep.get("success", False)))
                ep_args = ep.get("args", {}) or {}
            else:
                name = str(getattr(ep, "action", ""))
                sit = dict(getattr(ep, "situation", {}) or {})
                ok = bool(getattr(ep, "ok", False))
                ep_args = getattr(ep, "args", None) or {}
            if name != action.name:
                continue
            seen_name = True
            if canonical_args(ep_args) != args_id:
                continue
            restricted = {k: sit[k] for k in pre_keys if k in sit}
            if (len(restricted) == len(pre_keys)
                    and canonical_variant(restricted) == variant):
                exact[0 if ok else 1] += 1
            if set(sit.keys()) == pre_key_set:
                shape[0 if ok else 1] += 1
        if sum(exact):
            return exact
        if sum(shape):
            return shape
        # No episode attributes to this variant at all. When the
        # learner has never seen this action NAME either, the only
        # counts that can exist are direct ``record()`` outcomes (no
        # situation, no variant — e.g. a "learned earlier" failure
        # recorded straight on the planner) — fall back to the pooled
        # name-level counts so that signal still steers cost. When
        # episodes DO exist for the name, the variant data is
        # authoritative and ``[0, 0]`` stands: a sibling variant's
        # failures must not leak in through the pooled counts (P9-7).
        if not seen_name:
            return list(self.stats.get(action.name, [0, 0]))
        return [0, 0]

    def failure_rate_for(self, action: Action) -> float:
        """Phase 10 (P10-1), args-aware since Phase 10.5 (P10.5-2):
        the failure rate of one action VARIANT — learner episodes
        with the same name AND the same canonical variant AND the
        same canonical semantic args. A variant with no episodes of
        its own has rate 0.0: it must not inherit sibling variants'
        failures — that inheritance is the P9-7 defect (the cheap
        press's failures inflating the expensive press's cost), and
        the same defect in args form is what stalled the
        generated-workshop transfer leg (cut-zinc's failures
        inflating cut-tin's rate). When no learner is attached
        there is no episode-level variant data at all, so this falls
        back to the pooled ``failure_rate(name)`` (the standalone-
        Planner path, whose ``_stats`` counts are name-level only) —
        and likewise when the attached learner holds no episode at
        all under this action name: the only counts that can exist
        then are direct ``record()`` outcomes (no situation, no
        variant), which fall back to the pooled counts so a directly
        recorded failure still steers cost. When episodes DO exist
        for the name, the variant data is authoritative and the
        no-episode variant stays 0.0 (see ``failure_counts_for``).
        Limitation: outcomes recorded directly via ``record()``
        without going through the learner carry no situation, so
        they are not variant attributable and are excluded here;
        all Brain recording goes through the learner, so the two
        views agree in normal use. Limitation: episodes recorded
        before Phase 10.5 carry no args (``args={}``) and attribute
        only to empty-args variants — the same precedent as P9-1.

        Phase 10.5 (P10.5-8): when no episode matches the exact
        variant, the rate falls back one tier to the key-shape rate
        (same name, same precondition KEY set, same semantic args;
        see ``failure_counts_for``), and only then to 0.0. The
        key-shape tier never crosses a key-set boundary (cheap vs
        mould press) or an args boundary (zinc vs tin).

        Boundary (P10.5-4, recorded not hidden): args-aware identity
        applies wherever the Action object is in hand (planner,
        bans, failure rates, learner). State-only lookups keep the
        precondition-variant key — causal records and
        ``effects_fn`` resolve by name+state, the simulator has no
        action object to thread an identity from, and the effects
        are identical across same-precondition variants, so the
        pooled rate is the correct simulation prior there.

        ADOPTED in ``effective_cost`` (P10.5-5, Phase 10.5): the
        variant rate (with the P10.5-8 key-shape fallback) is now
        the failure-rate term in ``effective_cost`` and in the
        last-resort ordering in ``Brain._do_make_plan``. The P10-1b
        history it must not repeat: wiring the exact-only variant
        rate in turned the generated-workshop leg from 2 failed /
        26 cycles / stalled into 1 failed / 80 cycles /
        budget_exhausted — the never-failed with-form press variant
        (rate 0.0) made the press-first retry plan cheap enough for
        the last-resort planner to re-adopt every cycle, and each
        re-press "success" advanced the progress count, defeating
        the progress-based stall detector. The P10.5-5 stall fix
        (``_steps_executed`` advances only on a real world change)
        is what prevents that recurrence; the adoption A/B numbers
        are in the Phase 10.5 W2 handoff report."""
        s, f = self.failure_counts_for(action)
        if s + f == 0:
            return 0.0
        return f / (s + f)

    # ---- persistence (Phase 1; Phase 3: stats are derived, not stored) --
    # to_json no longer writes "stats": stats are derived from learner
    # episodes, so there is nothing to persist for them. from_json still
    # accepts a legacy "stats" dict (Phase 1–2 planner.json) into the
    # fallback, so an old file loads.
    def to_json(self) -> dict:
        # Phase 10.5: banned pairs serialise as
        # [name, args_identity, [[k, v], ...]]; from_json accepts the
        # legacy 2-element [name, [[k, v], ...]] form.
        # Phase 10.5 (P10.5-9): generations persist as a parallel
        # list (position i tags actions[i]) plus the counter, so a
        # loaded brain keeps preferring the current world's actions
        # on ties. A pre-10.5 save has neither key and loads with
        # every action at generation 0 (the safe default).
        return {"actions": [asdict(a) for a in self.actions],
                "banned_pairs": [[a, args_id, [[k, v] for k, v in sit]]
                                 for a, args_id, sit in sorted(self.banned_pairs)],
                "generation": int(self._generation),
                "generations": [int(self.generation_of(a))
                                for a in self.actions]}

    @classmethod
    def from_json(cls, data: dict) -> "Planner":
        if not isinstance(data, dict):
            raise ValueError("planner file must be a dict")
        p = cls(max_nodes=50000)
        if "stats" in data:
            p.stats = {k: [int(v[0]), int(v[1])] for k, v in data["stats"].items()}
        for d in data.get("actions", []):
            p.actions.append(Action(d["name"], dict(d.get("pre", {})), dict(d.get("eff", {})),
                                   float(d.get("cost", 1.0)), dict(d.get("args", {})),
                                   d.get("tool"), d.get("store_as")))
        # Phase 10.5 (P10.5-9): restore generations from the
        # parallel list (position i tags actions[i]); a missing or
        # short list (pre-10.5 save) leaves those actions at
        # generation 0, and the counter defaults to the highest
        # restored generation so a later mark still advances past
        # every loaded action.
        _gens = data.get("generations", [])
        for _a, _g in zip(p.actions, _gens if isinstance(_gens, list) else []):
            try:
                p._action_generation[action_identity(_a)] = int(_g)
            except (TypeError, ValueError):
                pass
        try:
            p._generation = int(data.get("generation", 0))
        except (TypeError, ValueError):
            p._generation = 0
        if p._action_generation:
            p._generation = max(p._generation,
                                max(p._action_generation.values()))
        for entry in data.get("banned_pairs", []):
            if len(entry) == 3:
                a, args_id, sit = entry
            elif len(entry) == 2:
                # Legacy Phase 1–10.5 save: no args identity recorded;
                # those bans were empty-args bans (or pre-args bans),
                # so they load as the empty-args identity.
                a, sit = entry
                args_id = ""
            else:
                continue
            p.banned_pairs.add((str(a), str(args_id),
                                tuple((str(k), str(v)) for k, v in sit)))
        return p

    def _avoided_by_approved_rule(self, action_name: str, state: dict) -> bool:
        """True when an ACTIVE avoid_action rule in the attached
        RuleEngine bans ``action_name`` in ``state`` (Phase 3 §3.1).
        Only active rules count: pending rules have no effect until a
        human approves them. Rules are read from the RuleEngine, never
        from Learner internals."""
        rules = self._rules
        if rules is None:
            return False
        for r in rules.rules(status="active"):
            for a in r.then:
                if a.get("do") == "avoid_action" and a.get("args", {}).get("name") == action_name:
                    if r.matches(state):
                        return True
        return False

    def effective_cost(self, a: Action) -> float:
        """Phase 5: base cost, inflated by the failure rate, then
        multiplied by the approved-induced-rule preference factor
        (``preferred_actions``, default 1.0) and by the attached
        pattern library's multiplier (1.0 when no pattern fires).
        Phase 6: also multiplied by the concept preference (0.8
        when the action's tool matches the learned ``durable_tool``
        concept, else 1.0). Multipliers are cost priors, never bans,
        and they apply here so ``plan_all`` ordering uses them.

        Phase 10.5 (P10.5-5): the failure-rate term is the
        variant-aware ``failure_rate_for(a)`` (exact variant, then
        the P10.5-8 key-shape fallback, then 0.0) — adopted after
        the P10.5-5 stall fix, which is what keeps the P10-1b
        regression (variant rate + idempotent re-press counting as
        progress = 80-cycle budget exhaustion) from recurring. See
        ``failure_rate_for`` and the Phase 10.5 W2 handoff report
        for the adoption A/B numbers."""
        cost = a.cost * (1.0 + 2.0 * self.failure_rate_for(a))
        cost *= self.preferred_actions.get(a.name, 1.0)
        cost *= self.pattern_multiplier(a)
        cost *= self.concept_multiplier(a)
        return cost

    @staticmethod
    def applicable(a: Action, state: dict) -> bool:
        return all(state.get(k) == v for k, v in a.pre.items())

    def plan_all(self, state: dict, target: dict, banned: set[str] | None = None,
                 limit: int = 2, allow_banned_pairs: bool = False) -> list[Plan]:
        """Return up to ``limit`` distinct plans, distinct by FULL action
        sequence (Phase 3 §4.2): two plans sharing a first action but
        diverging later (e.g. both starting ``go_lab_from_hall``, then
        ``pick_lock`` vs ``take_key``) are both retained. Ordered by cost
        ascending.

        Single uniform-cost search: the best cost is tracked per
        (state, first-action) pair, so the cheapest plan for each first
        action survives pruning. Frontier pops are in cost order, so the
        first plan recorded per full sequence is the cheapest for it, and
        once ``limit`` distinct sequences are recorded no cheaper
        unrecorded plan can exist. The ``max_nodes`` budget is shared by
        the whole search.

        Limitation (documented): pruning still collapses two prefixes
        that converge on the same state with the same first action to
        the cheaper one, so plans differing ONLY in such a converged
        prefix keep only the cheaper. Diversity is by full sequence at
        the result level, for plans that diverge at or after a shared
        state — the case the design's acceptance test names.
        """
        banned = banned or set()
        if all(state.get(k) == v for k, v in target.items()):
            return [Plan([], 0.0)]
        if limit <= 0:
            return []
        # Phase 12 (P12-O3): effective_cost(a) is constant within one
        # plan_all call — it reads only learner episodes, planner
        # stats, the pattern library, the concept store and the
        # brain's pattern context, none of which the search mutates
        # (it works on local state copies). The same action object
        # is expanded from many frontier nodes, so memoize per
        # action object for this call only. Keyed by id() because
        # every action here is alive in self.actions for the whole
        # call, so ids cannot be recycled mid-call; a fresh dict
        # per call means nothing stale can survive to the next plan.
        cost_cache: dict[int, float] = {}
        tie = itertools.count()
        start = _freeze(state)
        # Frontier entries carry the REAL state alongside its frozen key:
        # _freeze encodes unhashable values (e.g. a dict stored via
        # store_as) for the visited set, and reconstructing a state from
        # that encoding would replace the real value with its encoding.
        # Phase 10.5 (P10.5-9): the second key is the NEGATED
        # generation of the step that produced this entry (0 for the
        # start, which has no step) — on an exact cost tie the
        # current world's action pops first, so a carried-over
        # variant never wins a tie over the current world's own
        # variant (cut-tin over cut-zinc). Cost is still the primary
        # key: the tie-break can never override a cost difference
        # of any size. (The tie counter stays third so heapq never
        # compares states.)
        frontier = [(0.0, 0, next(tie), start, dict(state), [])]
        # (frozen state, first action name or None) -> (best cost,
        # best generation rank) seen. Cost is primary; on an exact
        # cost tie the prefix whose most recent step is from the
        # current world wins (Phase 10.5 P10.5-9) — so the push-time
        # prune must REPLACE on (equal cost, better rank), not just
        # keep the first-pushed, or the first-iterated (older)
        # variant would win every tie before the heap ever saw the
        # newer one.
        best: dict[tuple, tuple] = {(start, None): (0.0, 0)}
        found: dict[tuple, Plan] = {}
        nodes = 0
        while frontier and nodes < self.max_nodes:
            cost, gen_rank, _, node, cur, steps = heapq.heappop(frontier)
            first = steps[0].name if steps else None
            if (cost, gen_rank) > best.get((node, first), (float("inf"), 0)):
                continue
            nodes += 1
            if steps and all(cur.get(k) == v for k, v in target.items()):
                seq = tuple(a.name for a in steps)
                if seq not in found:
                    found[seq] = Plan(steps, round(cost, 4))
                    if len(found) >= limit:
                        break
                continue
            for a in self.actions:
                if a.name in banned or not self.applicable(a, cur):
                    continue
                # Phase 4: (action, situation) pair bans are the
                # experience-based avoidance (see Brain._try_alternative).
                # They are hard by default, but a caller may pass
                # allow_banned_pairs=True as a LAST RESORT when no
                # unbanned plan exists at all: a pair ban says "this
                # failed in this situation once", not "this can never
                # work" — e.g. an action that failed only because a
                # precondition the model does not represent was unmet
                # (a cold coil) can succeed after an earlier step
                # changes the world. Name bans (rule-based, possibly
                # human-approved) and approved avoid_action rules are
                # NEVER relaxed by this flag.
                if not allow_banned_pairs and self.is_banned(
                        a.name, {k: cur.get(k) for k in a.pre}, a.args, a.tool):
                    continue
                if self._avoided_by_approved_rule(a.name, cur):
                    continue
                # Phase 10.5 (P10.5-5, stall-fix companion): never
                # expand a step that changes NOTHING in the model —
                # every effect value already holds in ``cur``. The
                # design premise behind the stall fix is that in-plan
                # steps always change state in the planner's model;
                # the visited set alone does NOT guarantee that — a
                # no-change step as the FIRST step opens a fresh
                # (state, first-action) key, so a plan can start with
                # a redundant step (measured: a re-cut whose effect
                # already held was planned, executed as an idempotent
                # "success", and — once the stall fix stopped counting
                # it as progress — stalled a flaky-tool recovery
                # mid-plan). A no-change step can never help reach a
                # goal in a deterministic model: it leaves the state
                # exactly as it was, at positive cost. A tool action
                # is exempt ONLY when its ``store_as`` key is OUTSIDE
                # its effects: then the stored output is genuinely
                # unmodeled and the step may still change state
                # through it. When ``store_as`` names a key the
                # effects already declare (the workshop cutter:
                # ``store_as=cut_b`` with ``eff={cut_b: True}``,
                # tool returns True), the modeled outcome IS the
                # effect value, so the step is prunable like any
                # other when that value already holds.
                _modeled = a.store_as is None or a.store_as in a.eff
                if _modeled and all(
                        cur.get(k) == v for k, v in a.eff.items()):
                    continue
                nxt = dict(cur)
                nxt.update(a.eff)
                key = _freeze(nxt)
                _aid = id(a)
                _ec = cost_cache.get(_aid)
                if _ec is None:
                    _ec = self.effective_cost(a)
                    cost_cache[_aid] = _ec
                c = cost + _ec
                gr = -self.generation_of(a)
                bkey = (key, first if first is not None else a.name)
                if (c, gr) < best.get(bkey, (float("inf"), 0)):
                    best[bkey] = (c, gr)
                    heapq.heappush(frontier, (c, gr, next(tie), key, nxt, steps + [a]))
        return [found[seq] for seq in sorted(found, key=lambda s: found[s].cost)]

    def plan(self, state: dict, target: dict, banned: set[str] | None = None,
             allow_banned_pairs: bool = False) -> Plan | None:
        plans = self.plan_all(state, target, banned=banned, limit=1,
                              allow_banned_pairs=allow_banned_pairs)
        return plans[0] if plans else None
