"""Imagination — symbolic simulation over the world model (Phase 8, P8-5/P8-6).

A ``Simulator`` runs a sequence of actions forward over a *copy* of the
world model plus the planner's symbolic ``effects_fn`` and the causal
model's per-(action, effect) success rates. Nothing simulated ever
touches the real world model, the real state, or LTM: the copy is built
with ``WorldModel.from_json(world_model.to_json())`` on every call, and
all simulated writes go to that copy (with epistemic status
``predicted``) and to a private state dict.

A simulation result is a prediction, never an observation: every
assumption is recorded explicitly, and every result carries the
``simulated`` label at the Brain level. Deterministic, stdlib only,
CPU-first, no LLM, nothing hardcoded to any evaluation task.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..planning.planner import canonical_variant
from ..world.model import EpistemicStatus, WorldModel, _map_key_to_entity_property

#: Hard bound on simulated steps per simulation (design P8-6).
MAX_SIMULATED_STEPS = 20
#: Floor on a step confidence used as a divisor for expected cost, so an
#: inapplicable step inflates cost instead of dividing by zero.
_MIN_CONFIDENCE_DIVISOR = 0.1


def laplace_rate(causal, action: str, effect: str, variant=None):
    """Laplace-smoothed success rate ``(met + 1) / (total + 2)`` for an
    (action, effect) pair from a causal model, or None when the causal
    model has no record for the pair. Prefers the raw ``counts()`` API
    (Phase 5) and falls back to ``success_rate()``. Phase 9 (P9-1):
    a named ``variant`` reads that variant's counts alone; ``None``
    reads the pooled counts. Causal-like objects without a
    ``variant`` parameter are called the old way."""
    if causal is None:
        return None
    counts = getattr(causal, "counts", None)
    if callable(counts):
        try:
            cell = counts(action, effect, variant=variant)
        except TypeError:
            try:
                cell = counts(action, effect)
            except Exception:
                cell = None
        except Exception:
            cell = None
        if cell is not None and cell[1] > 0:
            return (cell[0] + 1) / (cell[1] + 2)
    fn = getattr(causal, "success_rate", None)
    if callable(fn):
        try:
            raw = fn(action, effect, variant=variant)
        except TypeError:
            try:
                raw = fn(action, effect)
            except Exception:
                raw = None
        except Exception:
            raw = None
        if raw is not None:
            return float(raw)
    return None


def _step_name(step) -> str:
    if isinstance(step, dict):
        return str(step.get("action") or step.get("name") or step)
    if isinstance(step, str):
        return step
    return str(getattr(step, "name", step))


def _state_key_for(entity: str, prop: str) -> str:
    """Reverse of ``_map_key_to_entity_property`` for the key shapes
    the world model uses: the state key a (entity, property) pair
    came from. Lets a world-model-fact edit also update the
    simulated state dict, so the planner effects see it."""
    if entity == "self" and prop == "location":
        return "at"
    if prop == "held":
        return f"has_{entity}"
    if entity == "door" and prop == "open":
        return "door_open"
    return f"{entity}_{prop}"


def _normalise_world_edits(edits):
    """World-model edits into a list of ``(entity, property, value)``
    triples. Accepts: ``{state_key: value}`` (mapped structurally),
    ``{"entity", "property", "value"}``, or a list of either shape."""
    out = []
    if edits is None:
        return out
    if isinstance(edits, dict) and "entity" in edits and "property" in edits:
        edits = [edits]
    if isinstance(edits, dict):
        for key, value in edits.items():
            entity, _kind, prop = _map_key_to_entity_property(str(key))
            out.append((entity, prop, value))
        return out
    for item in (edits or []):
        if isinstance(item, dict) and "entity" in item and "property" in item:
            out.append((str(item["entity"]), str(item["property"]), item.get("value")))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((str(item[0]), str(item[1]), item[2]))
    return out


@dataclass
class SimulationResult:
    """One simulated run: predicted states, per-step confidences, cost,
    success probability and the assumptions the prediction rests on."""

    steps: list = field(default_factory=list)          # action names
    states: list = field(default_factory=list)         # state after each step (start first)
    per_step: list = field(default_factory=list)       # [{action, effects, confidence, state}]
    predicted_state: dict = field(default_factory=dict)
    predicted_success: float = 1.0
    predicted_cost: float = 0.0
    risk: float = 0.0
    assumptions: list = field(default_factory=list)
    met_goal: bool = False
    violations: list = field(default_factory=list)     # world-model constraints violated
    truncated: bool = False
    simulated: bool = True

    def to_dict(self) -> dict:
        return {
            "steps": list(self.steps),
            "states": [dict(s) for s in self.states],
            "per_step": [dict(p, effects=dict(p.get("effects", {})),
                              state=dict(p.get("state", {})))
                         for p in self.per_step],
            "predicted_state": dict(self.predicted_state),
            "predicted_success": float(self.predicted_success),
            "predicted_cost": float(self.predicted_cost),
            "risk": float(self.risk),
            "assumptions": list(self.assumptions),
            "met_goal": bool(self.met_goal),
            "violations": list(self.violations),
            "truncated": bool(self.truncated),
            "simulated": True,
        }


class Simulator:
    """Simulate action sequences over a copy of the world model.

    ``world_model`` is the live model (copied per simulation);
    ``effects_fn`` is the planner's symbolic effect function
    (``effects_fn(name, state) -> dict``); ``causal`` supplies
    per-(action, effect) success rates; ``planner`` (when given)
    supplies the live ``effects_fn`` and action costs; ``revision``
    (a ``SimulationRevision``) supplies per-(action, effect)
    predicted-success adjustments learned from past outcomes, which
    take precedence over the raw causal rate.
    """

    def __init__(self, world_model, effects_fn=None, causal=None,
                 actions=None, revision=None, planner=None):
        self.world_model = world_model
        self.effects_fn = effects_fn
        self.causal = causal
        self.revision = revision
        self.planner = planner
        self._actions = list(actions or [])

    # ---- helpers ------------------------------------------------------

    def _effects(self, name: str, state: dict) -> dict:
        fn = None
        if self.planner is not None:
            fn = getattr(self.planner, "effects_fn", None)
        if fn is None:
            fn = self.effects_fn
        if fn is None:
            return {}
        try:
            return dict(fn(name, dict(state)) or {})
        except Exception:
            return {}

    def _resolve_action(self, name: str, state: dict):
        """The action object for ``name`` that applies in ``state``
        (Phase 9, P9-1): the planner's state-based resolution when a
        planner is attached, else the first applicable action in the
        simulator's own list, else the first same-named action."""
        if self.planner is not None:
            resolve = getattr(self.planner, "resolve_action", None)
            if callable(resolve):
                try:
                    found = resolve(name, dict(state))
                except Exception:
                    found = None
                if found is not None:
                    return found
        for a in (self._actions or []):
            if getattr(a, "name", None) == name:
                pre = getattr(a, "pre", None) or {}
                if all(state.get(k) == v for k, v in pre.items()):
                    return a
        for a in (self._actions or []):
            if getattr(a, "name", None) == name:
                return a
        return None

    def _variant_for(self, name: str, state: dict):
        """The canonical variant of the ``name`` variant that applies
        in ``state`` (Phase 9, P9-1), or None when none applies."""
        if self.planner is not None:
            vf = getattr(self.planner, "variant_for", None)
            if callable(vf):
                try:
                    return vf(name, dict(state))
                except Exception:
                    pass
        action = self._resolve_action(name, state)
        if action is None:
            return None
        return canonical_variant(getattr(action, "pre", None) or {})

    def _resolve_step(self, raw_step, name: str, state: dict):
        """Resolve one simulation step to ``(effects, variant,
        action_obj)``. Phase 9 (P9-1): a step given as an Action
        object IS its variant — its own preconditions/effects are
        used, and an inapplicable variant yields no effect (it must
        not silently borrow another variant's effect). A step given
        as a dict may carry an explicit ``variant`` (canonical
        string) or ``pre``; a plain name resolves by (name, state)
        through the planner. Returns the variant the step was
        simulated as, so confidence and revision can key on it."""
        if hasattr(raw_step, "pre") and hasattr(raw_step, "eff"):
            pre = getattr(raw_step, "pre", None) or {}
            variant = canonical_variant(pre)
            if all(state.get(k) == v for k, v in pre.items()):
                return dict(getattr(raw_step, "eff", None) or {}), variant, raw_step
            return {}, variant, raw_step
        if isinstance(raw_step, dict):
            variant = raw_step.get("variant")
            if variant is None and isinstance(raw_step.get("pre"), dict):
                variant = canonical_variant(raw_step["pre"])
            effects = self._effects(name, state)
            if variant is None:
                variant = self._variant_for(name, state)
            return effects, variant, self._resolve_action(name, state)
        effects = self._effects(name, state)
        variant = self._variant_for(name, state)
        return effects, variant, self._resolve_action(name, state)

    def _action_cost(self, name: str, state: dict | None = None,
                     action=None) -> float:
        """The cost of one action as the planner prices it: the
        effective cost (base cost inflated by the observed failure
        rate and by pattern/concept multipliers) when a planner is
        attached, else the action's base cost. Using the effective
        cost keeps simulation ranking consistent with the planner's
        own ordering when predicted success ties — a route the
        planner already prices as expensive because its tool failed
        must not look cheap to the simulator. Phase 9 (P9-1): the
        action is resolved by (name, state) when a state is given,
        so a cheap inapplicable variant no longer prices a step that
        another variant would actually run."""
        resolved = action
        if resolved is None:
            resolved = self._resolve_action(name, state or {})
        if resolved is not None:
            if self.planner is not None:
                try:
                    return float(self.planner.effective_cost(resolved))
                except Exception:
                    pass
            try:
                return float(getattr(resolved, "cost", 1.0))
            except (TypeError, ValueError):
                return 1.0
        return 1.0

    def _step_confidence(self, name: str, effects: dict, variant=None):
        """Weakest-link confidence over a step's effects: the
        revision's adjusted rate when one exists, else the causal
        Laplace rate, else 1.0 (the planner's own model, no evidence
        yet). Returns (confidence, sources) where sources names, per
        effect key, where the number came from. Phase 9 (P9-1): both
        rates are read for this step's variant, never for the bare
        action name."""
        if not effects:
            return 1.0, {}
        confidence = 1.0
        sources = {}
        for key in effects:
            rate = None
            source = "planner model (no evidence yet)"
            if self.revision is not None:
                try:
                    rate = self.revision.predicted_success(name, key, self.causal,
                                                           variant=variant)
                except TypeError:
                    try:
                        rate = self.revision.predicted_success(name, key, self.causal)
                    except Exception:
                        rate = None
                except Exception:
                    rate = None
                if rate is not None:
                    source = "simulation revision (adjusted toward observed rate)"
            if rate is None:
                rate = laplace_rate(self.causal, name, key, variant=variant)
                if rate is not None:
                    source = "causal model (Laplace-smoothed success rate)"
            if rate is None:
                rate = 1.0
            sources[key] = source
            confidence = min(confidence, float(rate))
        return confidence, sources

    def _copy_model(self) -> WorldModel:
        return WorldModel.from_json(self.world_model.to_json())

    # ---- the simulation -------------------------------------------------

    def simulate(self, steps, start_state: dict, goal: dict | None = None,
                 max_steps: int = MAX_SIMULATED_STEPS,
                 state_edits: dict | None = None,
                 world_edits=None) -> SimulationResult:
        """Simulate ``steps`` from ``start_state``. ``state_edits``
        (``{key: value}``) and ``world_edits`` are applied to the
        *copy* only, before the first step, and are recorded as
        assumptions naming the changed fact and its previous value.
        At most ``max_steps`` steps are simulated; the rest are
        dropped and the truncation is recorded. Never mutates the
        real world model, the caller's state, or LTM."""
        raw_steps = list(steps or [])
        names = [_step_name(s) for s in raw_steps]
        truncated = len(names) > int(max_steps)
        names = names[:int(max_steps)]
        raw_steps = raw_steps[:int(max_steps)]
        cur = dict(start_state or {})
        if state_edits:
            for k, v in dict(state_edits).items():
                cur[str(k)] = v
        model = self._copy_model()
        assumptions = ["simulated: this outcome is imagined, not observed"]
        # Counterfactual / what-if edits, applied to the copy only.
        for key, value in (dict(state_edits) if state_edits else {}).items():
            old = (start_state or {}).get(str(key))
            assumptions.append(
                f"simulated: assumed {key}={value!r} (actual value was {old!r})")
        for entity, prop, value in _normalise_world_edits(world_edits):
            ent = model.entity(entity)
            old = None
            if ent is not None and prop in (ent.properties or {}):
                old = ent.properties[prop].value
            kind = (ent.kind if ent is not None else "object")
            model.set_property(entity, prop, value,
                               status=EpistemicStatus.assumed,
                               confidence=1.0, provenance="simulation",
                               cycle=0, kind=kind)
            # The same edit in the simulated state, so planner
            # effects see the changed fact (reverse key-shape map).
            cur[_state_key_for(entity, prop)] = value
            assumptions.append(
                f"simulated: assumed world-model fact "
                f"{entity}.{prop}={value!r} (actual value was {old!r})")
        # State edits that name a world-model fact are also reflected
        # in the copy, so the copy and the simulated state agree.
        if state_edits:
            for key, value in dict(state_edits).items():
                entity, kind, prop = _map_key_to_entity_property(str(key))
                model.set_property(entity, prop, value,
                                   status=EpistemicStatus.assumed,
                                   confidence=1.0, provenance="simulation",
                                   cycle=0, kind=kind)

        states = [dict(cur)]
        per_step = []
        violations: list[str] = []
        predicted_cost = 0.0
        success = 1.0
        for raw_step, name in zip(raw_steps, names):
            effects, variant, action_obj = self._resolve_step(
                raw_step, name, cur)
            if not effects:
                per_step.append({"action": name, "effects": {},
                                 "confidence": 0.0, "variant": variant,
                                 "state": dict(cur)})
                assumptions.append(
                    f"simulated: no known effect for '{name}' in this "
                    f"state: state unchanged (action inapplicable or unknown)")
                success = 0.0
                predicted_cost += self._action_cost(name, cur, action_obj)
                states.append(dict(cur))
                continue
            confidence, sources = self._step_confidence(name, effects,
                                                        variant=variant)
            prev = dict(cur)
            cur = dict(cur)
            cur.update(effects)
            for key, value in effects.items():
                entity, kind, prop = _map_key_to_entity_property(str(key))
                model.set_property(entity, prop, value,
                                   status=EpistemicStatus.predicted,
                                   confidence=confidence,
                                   provenance="simulation", cycle=0, kind=kind)
            # Transition check on the COPY: check_plan replays this
            # step through the planner effects and checks every
            # constraint (including transition rules such as the
            # learned door-gated edge) between the pre- and
            # post-step states.
            ok, violated = model.check_plan([name], prev, self._effects)
            if not ok and violated not in violations:
                violations.append(violated)
                assumptions.append(
                    f"simulated: step '{name}' would violate "
                    f"world-model constraint '{violated}'")
            per_step.append({"action": name, "effects": dict(effects),
                             "confidence": round(float(confidence), 4),
                             "confidence_sources": sources,
                             "variant": variant,
                             "state": dict(cur)})
            assumptions.append(
                f"simulated: assumed effect of '{name}': {effects} "
                f"(confidence {confidence:.2f})")
            success *= float(confidence)
            predicted_cost += self._action_cost(name, cur, action_obj) / max(float(confidence),
                                                            _MIN_CONFIDENCE_DIVISOR)
            states.append(dict(cur))
        if truncated:
            assumptions.append(
                f"simulated: truncated to the first {int(max_steps)} steps "
                f"of {len(steps or [])} (bounded simulation)")
        if isinstance(goal, dict) and goal:
            met = all(cur.get(k) == v for k, v in goal.items())
        else:
            met = all(p.get("confidence", 0.0) > 0.0 for p in per_step) if per_step else True
        return SimulationResult(
            steps=names, states=states, per_step=per_step,
            predicted_state=dict(cur),
            predicted_success=round(float(success), 4),
            predicted_cost=round(float(predicted_cost), 4),
            risk=round(float(1.0 - success), 4),
            assumptions=assumptions, met_goal=bool(met),
            violations=violations, truncated=bool(truncated))
