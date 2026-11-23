"""World model (Phase 5): the structured view over the brain's observations.

One module the whole cycle uses. Entities, relations, a spatial room
graph, a temporal event log, constraints, epistemic status, object
permanence with decay, prediction-error tracking and append-only
revision history.

Rules that matter:
- The mapping from environment state keys to entities is structural,
  over KEY SHAPE only (``has_<name>`` -> entity ``<name>``; ``at`` ->
  the agent entity ``self``; ``door_open`` -> entity ``door``). No
  world-specific names appear anywhere in this package.
- ``unknown`` is an explicit status, never the same as ``False``: a
  key never observed has no entry at all.
- "Not observed" is never "gone": an entity is marked gone ONLY by an
  explicit ``mark_gone`` call on observed contradictory evidence.
  Otherwise it stays unobserved with decaying confidence.
- Nothing is silently overwritten: contradictions keep both values in
  a disputed list, revisions keep the old value in an append-only list.
- Deterministic, stdlib only, CPU-first, no LLM.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from enum import Enum

from ..core.audit import MAX_EVENTS

# Per-cycle confidence decay for entities not re-observed (design §2.8).
PERMANENCE_DECAY = 0.9

SCHEMA_VERSION = 1


class EpistemicStatus(str, Enum):
    """Six distinct epistemic statuses (design §2.1)."""

    observed = "observed"
    inferred = "inferred"
    assumed = "assumed"
    predicted = "predicted"
    told = "told"
    unknown = "unknown"


def _status(value) -> EpistemicStatus:
    """Coerce a string / enum to EpistemicStatus; unknown on garbage."""
    if isinstance(value, EpistemicStatus):
        return value
    try:
        return EpistemicStatus(str(value))
    except ValueError:
        return EpistemicStatus.unknown


@dataclass
class EntityProperty:
    """One typed property value with its epistemic standing."""

    value: object = None
    status: EpistemicStatus = EpistemicStatus.unknown
    confidence: float = 1.0
    provenance: str = ""

    def to_json(self) -> dict:
        return {"value": self.value, "status": _status(self.status).value,
                "confidence": float(self.confidence),
                "provenance": str(self.provenance)}

    @classmethod
    def from_json(cls, d: dict) -> "EntityProperty":
        return cls(value=d.get("value"),
                   status=_status(d.get("status", "unknown")),
                   confidence=max(0.0, min(1.0, float(d.get("confidence", 1.0)))),
                   provenance=str(d.get("provenance", "")))


@dataclass
class Entity:
    """A tracked thing in the world. Identity (``id``) is stable across
    observations: the same id is recognised after it moves, changes
    state, or goes unobserved."""

    id: str
    kind: str = "object"
    properties: dict[str, EntityProperty] = field(default_factory=dict)
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind,
                "properties": {k: p.to_json() for k, p in self.properties.items()},
                "first_seen_cycle": int(self.first_seen_cycle),
                "last_seen_cycle": int(self.last_seen_cycle)}

    @classmethod
    def from_json(cls, d: dict) -> "Entity":
        return cls(id=str(d["id"]), kind=str(d.get("kind", "object")),
                   properties={k: EntityProperty.from_json(v)
                               for k, v in dict(d.get("properties", {})).items()},
                   first_seen_cycle=int(d.get("first_seen_cycle", 0)),
                   last_seen_cycle=int(d.get("last_seen_cycle", 0)))


@dataclass
class Relation:
    """A directed, weighted relation. ``object`` is a single entity id
    (binary relation) or a list of ids (n-ary relation, e.g. a route)."""

    subject: str
    relation: str
    object: object  # str | list[str]
    strength: float = 1.0
    status: EpistemicStatus = EpistemicStatus.observed
    confidence: float = 1.0
    provenance: str = ""

    def to_json(self) -> dict:
        return {"subject": self.subject, "relation": self.relation,
                "object": self.object, "strength": float(self.strength),
                "status": _status(self.status).value,
                "confidence": float(self.confidence),
                "provenance": str(self.provenance)}

    @classmethod
    def from_json(cls, d: dict) -> "Relation":
        return cls(subject=str(d["subject"]), relation=str(d["relation"]),
                   object=d.get("object"), strength=float(d.get("strength", 1.0)),
                   status=_status(d.get("status", "observed")),
                   confidence=max(0.0, min(1.0, float(d.get("confidence", 1.0)))),
                   provenance=str(d.get("provenance", "")))


@dataclass
class WorldEvent:
    """One event the brain reasons about (distinct from the audit log).
    ``cause`` distinguishes an exogenous change (``"exogenous"``) from
    an action effect (``"action:<name>"``) and plain observation."""

    cycle: int
    kind: str
    entity: str
    key: str
    old_value: object = None
    new_value: object = None
    cause: str = "observation"

    def to_json(self) -> dict:
        return {"cycle": int(self.cycle), "kind": self.kind,
                "entity": self.entity, "key": self.key,
                "old_value": self.old_value, "new_value": self.new_value,
                "cause": self.cause}

    @classmethod
    def from_json(cls, d: dict) -> "WorldEvent":
        return cls(cycle=int(d.get("cycle", 0)), kind=str(d.get("kind", "observation")),
                   entity=str(d.get("entity", "")), key=str(d.get("key", "")),
                   old_value=d.get("old_value"), new_value=d.get("new_value"),
                   cause=str(d.get("cause", "observation")))


def _map_key_to_entity_property(key: str) -> tuple[str, str, str]:
    """Structural key-shape mapping (design §2.2). Returns
    (entity_id, kind, property_name). No world names are involved:
    the rule reads only the shape of the key.

    - ``has_<name>`` -> entity ``<name>``, kind ``object``, property ``held``
    - ``at`` -> entity ``self``, kind ``agent``, property ``location``
    - ``door_open`` -> entity ``door``, kind ``object``, property ``open``
    - ``<prefix>_<rest>`` -> entity ``<prefix>``, kind ``state``, property ``<rest>``
    - ``<key>`` (no underscore) -> entity ``<key>``, kind ``state``, property ``value``
    """
    if key.startswith("has_") and len(key) > 4:
        return key[4:], "object", "held"
    if key == "at":
        return "self", "agent", "location"
    if key == "door_open":
        return "door", "object", "open"
    if "_" in key:
        prefix, rest = key.split("_", 1)
        return prefix, "state", rest
    return key, "state", "value"


# Phase 12 (P12-O1): cached parameter counts for constraint check
# functions (see _call_check). Keyed by the callable itself; the
# number of distinct check functions is small (built-ins plus
# registered constraints), so this stays bounded.
_CHECK_ARITY: dict = {}


def _call_check(fn, state, prev_state, step_name) -> bool:
    """Call a constraint predicate with as many arguments as it accepts:
    (state), (state, prev_state) or (state, prev_state, step_name).
    Returns True when the constraint is satisfied, False on violation.

    Phase 12 (P12-O1): the parameter count is cached per callable.
    ``inspect.signature`` is pure for a given callable (nothing in
    the codebase assigns ``__signature__`` or rebinds ``__call__``),
    and it dominated this path's cost (a fresh signature build per
    check call). Unhashable callables fall back to computing it
    every time, as before."""
    n = None
    try:
        n = _CHECK_ARITY.get(fn)
    except TypeError:
        pass
    if n is None:
        try:
            n = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            n = 1
        try:
            _CHECK_ARITY[fn] = n
        except TypeError:
            pass
    if n >= 3:
        return bool(fn(state, prev_state, step_name))
    if n == 2:
        return bool(fn(state, prev_state))
    return bool(fn(state))


class WorldModel:
    """The world model: entities, relations, spatial graph, temporal
    log, constraints, permanence, prediction errors, revisions."""

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._events: list[WorldEvent] = []
        self._revisions: list[dict] = []
        self._prediction_errors: list[dict] = []
        self._contradictions: list[dict] = []
        # name -> {"check_fn": callable|None, "description": str}
        self._constraints: dict[str, dict] = {}
        self._observed_keys: set[str] = set()
        self._adjacency: dict[str, set[str]] = {}
        # Rooms the model has learned are door-gated: an attempted entry
        # failed while the entry barrier was closed (see
        # note_failed_entry, design §2.7). Only these rooms are gated by
        # the door_blocks_entry constraint — a room never observed to
        # need a door is NOT assumed gated, so an ordinary move while
        # some other door is closed is not a violation.
        self._door_gated_rooms: set[str] = set()
        # cycle -> entity ids observed in that cycle (for decay).
        self._observed_in_cycle: dict[int, set[str]] = {}
        self._decayed_cycles: set[int] = set()
        self._register_builtins()

    # ---- entities ------------------------------------------------------

    def _ensure_entity(self, entity_id: str, kind: str, cycle: int) -> Entity:
        ent = self._entities.get(entity_id)
        if ent is None:
            ent = Entity(id=entity_id, kind=kind,
                         first_seen_cycle=cycle, last_seen_cycle=cycle)
            self._entities[entity_id] = ent
        return ent

    def entity(self, name: str) -> Entity | None:
        return self._entities.get(str(name))

    def entities(self) -> list[Entity]:
        return [self._entities[k] for k in sorted(self._entities)]

    def status_of(self, entity: str, property: str) -> EpistemicStatus:
        """Epistemic status of a property; ``unknown`` when absent —
        never confused with a stored ``False`` value."""
        ent = self._entities.get(str(entity))
        if ent is None:
            return EpistemicStatus.unknown
        prop = ent.properties.get(str(property))
        if prop is None:
            return EpistemicStatus.unknown
        return _status(prop.status)

    def set_property(self, entity: str, key: str, value, *,
                     status: EpistemicStatus | str = EpistemicStatus.observed,
                     confidence: float = 1.0, provenance: str = "",
                     cycle: int = 0, kind: str = "object") -> Entity:
        """Set a property with an explicit epistemic status. This is how
        ``told`` / ``inferred`` / ``assumed`` / ``predicted`` entries are
        recorded (``observe()`` always records ``observed``). A change
        appends a ``set`` world event; an unchanged value refreshes
        ``last_seen_cycle`` only."""
        entity = str(entity)
        ent = self._ensure_entity(entity, kind, cycle)
        ent.last_seen_cycle = max(ent.last_seen_cycle, cycle)
        self._observed_in_cycle.setdefault(int(cycle), set()).add(entity)
        old = ent.properties.get(str(key))
        old_value = old.value if old is not None else None
        ent.properties[str(key)] = EntityProperty(
            value=value, status=_status(status),
            confidence=max(0.0, min(1.0, float(confidence))),
            provenance=str(provenance))
        if old is None or old_value != value:
            self._append_event(WorldEvent(
                cycle=int(cycle), kind="set", entity=entity, key=str(key),
                old_value=old_value, new_value=value,
                cause=str(provenance) or "set"))
        return ent

    def teach(self, entity: str, key: str, value, cycle: int = 0,
              source: str = "teacher") -> Entity:
        """A teacher/user-taught fact: status ``told``."""
        return self.set_property(entity, key, value, status=EpistemicStatus.told,
                                 confidence=1.0, provenance=str(source), cycle=cycle)

    # ---- observation ---------------------------------------------------

    def observe(self, key: str, value, cycle: int, source: str = "env",
                cause: str = "observation") -> Entity | None:
        """Feed one environment observation into the model.

        Maps the state key to an entity structurally (see
        ``_map_key_to_entity_property``), sets the property with status
        ``observed`` and confidence 1.0, and records a world event when
        the value changed. Derives ``located_in`` relations: when the
        agent is at room R and ``has_<name>`` is True, ``<name>`` is
        located in R; when ``has_<name>`` turns False the relation is
        dropped. Returns the entity changed, or None for an empty key.
        """
        if not key:
            return None
        key = str(key)
        cycle = int(cycle)
        entity_id, kind, prop = _map_key_to_entity_property(key)
        ent = self._ensure_entity(entity_id, kind, cycle)
        ent.last_seen_cycle = max(ent.last_seen_cycle, cycle)
        self._observed_keys.add(key)
        self._observed_in_cycle.setdefault(cycle, set()).add(entity_id)
        old = ent.properties.get(prop)
        old_value = old.value if old is not None else None
        ent.properties[prop] = EntityProperty(
            value=value, status=EpistemicStatus.observed, confidence=1.0,
            provenance=f"{source}:{key}")
        if old is None or old_value != value:
            kind_name = "exogenous" if cause == "exogenous" else "observation"
            self._append_event(WorldEvent(
                cycle=cycle, kind=kind_name, entity=entity_id, key=key,
                old_value=old_value, new_value=value, cause=str(cause)))
        # Spatial derivation: what the agent holds is where the agent is.
        if key == "at":
            self._refresh_held_locations(value, cycle)
        elif key.startswith("has_") and value is True:
            room = self._agent_location()
            if room is not None:
                self._set_located_in(entity_id, room, cycle)
        elif key.startswith("has_") and value is False:
            self._remove_relations(entity_id, "located_in")
        return ent

    def _agent_location(self):
        ent = self._entities.get("self")
        if ent is None:
            return None
        prop = ent.properties.get("location")
        return prop.value if prop is not None else None

    def _refresh_held_locations(self, room, cycle: int) -> None:
        for ent in list(self._entities.values()):
            held = ent.properties.get("held")
            if held is not None and held.value is True:
                self._set_located_in(ent.id, room, cycle)

    def _set_located_in(self, entity_id: str, room, cycle: int) -> None:
        self._remove_relations(entity_id, "located_in")
        self._relations.append(Relation(
            subject=entity_id, relation="located_in", object=room,
            strength=1.0, status=EpistemicStatus.observed, confidence=1.0,
            provenance=f"env:at={room}"))

    def _remove_relations(self, subject: str, relation: str) -> None:
        self._relations = [r for r in self._relations
                           if not (r.subject == subject and r.relation == relation)]

    def believed_location(self, entity: str):
        """(room, confidence, status) for an entity: its ``located_in``
        relation first, else its ``location`` property; ``(None, 0.0,
        unknown)`` when nothing is known."""
        entity = str(entity)
        for rel in self._relations:
            if rel.subject == entity and rel.relation == "located_in":
                return rel.object, float(rel.confidence), _status(rel.status)
        ent = self._entities.get(entity)
        if ent is not None:
            prop = ent.properties.get("location")
            if prop is not None:
                return prop.value, float(prop.confidence), _status(prop.status)
        return None, 0.0, EpistemicStatus.unknown

    # ---- relations / hierarchy / space ---------------------------------

    def add_relation(self, subject: str, relation: str, object, *,
                     strength: float = 1.0,
                     status: EpistemicStatus | str = EpistemicStatus.observed,
                     confidence: float = 1.0, provenance: str = "") -> Relation:
        """Add (or update in place, when the same subject/relation/object
        already exists) a relation. ``object`` may be a single id or a
        list of ids for an n-ary relation."""
        for rel in self._relations:
            if (rel.subject == str(subject) and rel.relation == str(relation)
                    and rel.object == object):
                rel.strength = float(strength)
                rel.status = _status(status)
                rel.confidence = max(0.0, min(1.0, float(confidence)))
                rel.provenance = str(provenance)
                return rel
        rel = Relation(subject=str(subject), relation=str(relation), object=object,
                       strength=float(strength), status=_status(status),
                       confidence=max(0.0, min(1.0, float(confidence))),
                       provenance=str(provenance))
        self._relations.append(rel)
        return rel

    def relations(self, subject=None, relation=None) -> list[Relation]:
        return [r for r in self._relations
                if (subject is None or r.subject == str(subject))
                and (relation is None or r.relation == str(relation))]

    def ancestors(self, entity: str) -> list[str]:
        """Walk ``is_a`` chains from ``entity`` (nearest first, no
        cycles, deterministic order)."""
        seen: set[str] = set()
        out: list[str] = []
        frontier = [str(entity)]
        while frontier:
            current = frontier.pop(0)
            for rel in self._relations:
                if rel.subject == current and rel.relation == "is_a":
                    parent = rel.object
                    if isinstance(parent, str) and parent not in seen:
                        seen.add(parent)
                        out.append(parent)
                        frontier.append(parent)
        return out

    def record_transition(self, from_room, to_room, cycle: int,
                          cause: str = "action") -> None:
        """Record an observed successful move between rooms: adds a
        directed edge to the room graph and a ``transition`` event.
        A no-op move (same room) adds no edge."""
        cycle = int(cycle)
        if from_room is None or to_room is None or from_room == to_room:
            return
        self._adjacency.setdefault(str(from_room), set()).add(str(to_room))
        self._append_event(WorldEvent(
            cycle=cycle, kind="transition", entity="self", key="at",
            old_value=from_room, new_value=to_room, cause=str(cause)))

    def note_failed_entry(self, room, cycle: int) -> None:
        """Learn that entry to ``room`` is door-gated: an attempted move
        into it failed while the entry barrier was closed (design §2.7:
        an edge whose traversal failed while ``door_open`` was False is
        marked ``requires: door_open``). Idempotent; records a
        ``door_gated`` world event on first learning."""
        room = str(room)
        if room in self._door_gated_rooms:
            return
        self._door_gated_rooms.add(room)
        self._append_event(WorldEvent(
            cycle=int(cycle), kind="door_gated", entity=room, key="entry",
            old_value=None, new_value="door_gated",
            cause="failed entry while closed"))

    def door_gated_rooms(self) -> set[str]:
        """Rooms learned to be door-gated (see note_failed_entry)."""
        return set(self._door_gated_rooms)

    def reachable(self, room) -> set[str]:
        """Rooms reachable from ``room`` over observed transitions
        (breadth-first, directed edges; includes ``room`` itself)."""
        start = str(room)
        seen = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            for nxt in sorted(self._adjacency.get(current, ())):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    # ---- temporal ------------------------------------------------------

    def _append_event(self, event: WorldEvent) -> WorldEvent:
        """Append one event to the temporal log, enforcing the
        MAX_EVENTS cap (Phase 12, ADR P12-11): the log is a bounded
        rolling window — the newest 1,000 events, oldest-first
        pruning, the same policy as the brain audit trail
        (ab/core/audit.py). ALL append sites go through here, so no
        path can grow the log without bound (10,000-cycle soak: it
        reached 13,410 with no cap at all). ``before()``/``after()``
        may return False for a key whose first event was pruned;
        that is the honest bounded-log semantics."""
        self._events.append(event)
        if len(self._events) > MAX_EVENTS:
            del self._events[:len(self._events) - MAX_EVENTS]
        return event

    def events_since(self, cycle: int) -> list[WorldEvent]:
        return [e for e in self._events if e.cycle >= int(cycle)]

    def events(self) -> list[WorldEvent]:
        return list(self._events)

    def record_event(self, kind: str = "", entity: str = "", key: str = "",
                     old_value=None, new_value=None, cause: str = "observation",
                     cycle: int = 0) -> WorldEvent:
        """Phase 9 §4.6: the PUBLIC way to append a WorldEvent to
        the model's log. Builds the event, appends it, and returns
        it. External writers (the perception bus) go through here;
        the model's own methods append through ``_append_event``
        too, so every path enforces the MAX_EVENTS cap (Phase 12,
        ADR P12-11)."""
        event = WorldEvent(cycle=int(cycle), kind=str(kind),
                           entity=str(entity), key=str(key),
                           old_value=old_value, new_value=new_value,
                           cause=str(cause))
        return self._append_event(event)

    def _first_event_cycle(self, key: str):
        for e in self._events:
            if e.key == str(key):
                return e.cycle
        return None

    def before(self, a_key: str, b_key: str) -> bool:
        """True when the first recorded event for ``a_key`` precedes the
        first recorded event for ``b_key`` (ordering by event cycle).
        False when either key has no recorded event."""
        a = self._first_event_cycle(a_key)
        b = self._first_event_cycle(b_key)
        if a is None or b is None:
            return False
        return a < b

    def after(self, a_key: str, b_key: str) -> bool:
        return self.before(b_key, a_key)

    def history(self, entity: str, key: str) -> list[tuple[int, object]]:
        """Ordered [(cycle, value), ...] of value changes for
        (entity, key), from the world-event log."""
        return [(e.cycle, e.new_value) for e in self._events
                if e.entity == str(entity) and e.key == str(key)]

    # ---- constraints ---------------------------------------------------

    def _register_builtins(self) -> None:
        self.add_constraint(
            "door_blocks_entry", self._door_blocks_entry,
            "Changing location into a room the model has LEARNED is "
            "door-gated (see note_failed_entry) while the entry barrier "
            "is closed in the pre-step state violates the model. A move "
            "into a room never observed to need a door is not a "
            "violation, and a move when the barrier state is unknown "
            "(not False) does not violate — unknown is never assumed "
            "closed.")
        self.add_constraint(
            "containment", self._containment,
            "A take step for an object whose location the model knows "
            "requires the agent to already be in that object's room in "
            "the pre-step state. Unknown location never violates.")

    def _door_blocks_entry(self, state, prev_state=None, step_name=None) -> bool:
        """Built-in transition rule (design §2.7, coordinator fix):
        a step that changes ``at`` into a room the model has learned
        is door-gated, while ``door_open`` is False in the pre-step
        state, violates. An ungated destination never violates, and
        ``door_open`` unknown (absent/None) is not a violation — the
        model never assumes an unobserved key is False. (The Phase 5
        worker build used a blanket rule — ANY room change while any
        door was closed — which false-positived on the standard demo
        plan (hall -> lab needs no door) and forced every plan through
        the plan_model_disagreement fallback; the gated-edge rule is
        what the design specified.)"""
        if prev_state is None:
            return True
        if state.get("at") == prev_state.get("at"):
            return True
        if prev_state.get("door_open") is not False:
            return True
        return str(state.get("at")) not in self._door_gated_rooms

    def _containment(self, state, prev_state=None, step_name=None) -> bool:
        """Built-in transition rule: a step named ``take_<name>`` requires
        ``at`` to equal the taken object's known ``located_in`` room in
        the pre-step state. Unknown location never violates."""
        if prev_state is None or not step_name:
            return True
        name = str(step_name)
        if not name.startswith("take_"):
            return True
        taken = name[len("take_"):]
        room, _conf, _status = self.believed_location(taken)
        if room is None:
            return True
        return prev_state.get("at") == room

    def add_constraint(self, name: str, check_fn, description: str = "") -> None:
        """Register a named predicate over a state dict. ``check_fn``
        takes ``(state)`` or ``(state, prev_state)`` or
        ``(state, prev_state, step_name)`` and returns True when the
        state (or transition) satisfies the constraint."""
        self._constraints[str(name)] = {
            "check_fn": check_fn, "description": str(description)}

    def constraints(self) -> list[str]:
        return sorted(self._constraints)

    def check_state(self, state: dict) -> list[str]:
        """Names of the constraints violated by a single state (state
        predicates only — transition constraints see no previous state
        here and cannot fire)."""
        state = dict(state or {})
        violated = []
        for name in sorted(self._constraints):
            fn = self._constraints[name]["check_fn"]
            if fn is None:
                continue
            if not _call_check(fn, state, None, None):
                violated.append(name)
        return violated

    def check_plan(self, steps, start_state: dict, effects_fn) -> tuple[bool, str | None]:
        """Simulate ``steps`` forward from ``start_state`` through
        ``effects_fn(name, state)`` (the planner's effect function:
        returns the step's effect dict when applicable, else ``{}``)
        and check every constraint at every intermediate state, plus
        the transition rules between states. Returns
        ``(True, None)`` when the whole plan is valid, else
        ``(False, first_violated_constraint_name)``. A plan is valid
        only if its constraints hold throughout, not just at the first
        step.

        ``steps`` may be Action objects (``.name``), dicts with a
        ``"name"`` key, or plain name strings.
        """
        cur = dict(start_state or {})
        for name in self.check_state(cur):
            return False, name
        for step in (steps or []):
            if isinstance(step, dict):
                step_name = str(step.get("name", ""))
            elif isinstance(step, str):
                step_name = step
            else:
                step_name = str(getattr(step, "name", ""))
            prev = dict(cur)
            effects = {}
            if effects_fn is not None:
                effects = dict(effects_fn(step_name, prev) or {})
            nxt = dict(prev)
            nxt.update(effects)
            for cname in sorted(self._constraints):
                fn = self._constraints[cname]["check_fn"]
                if fn is None:
                    continue
                if not _call_check(fn, nxt, prev, step_name):
                    return False, cname
            cur = nxt
        return True, None

    # ---- permanence ----------------------------------------------------

    def decay(self, cycle: int) -> None:
        """Decay confidence (``*= PERMANENCE_DECAY``) for every property
        — and every relation whose subject is that entity — of every
        entity NOT observed in ``cycle``. Once per cycle only. The
        ``exists`` flag set by ``mark_gone`` is never decayed: gone is
        evidence, not a fading belief.

        Phase 12 (P12-W7): the cycle-keyed decay bookkeeping is
        pruned here, at the end of each decay. ``_observed_in_cycle``
        and ``_decayed_cycles`` are read ONLY for the cycle being
        decayed — decay runs once per cycle with non-decreasing
        cycles (``Brain._sense``), and no other reader touches them
        except persistence (``to_json``/``from_json``). Older cycles
        are dead, so both stay O(1) instead of one entry per cycle
        (soak: 2,922 + 2,934 entries at 3,000 cycles, ~430 KiB/1,000
        cycles). Contract: decay() is defined for non-decreasing
        cycles — re-decaying an older cycle after a newer one has run
        may decay it again (its guard entry was pruned); no
        production path or test does that."""
        cycle = int(cycle)
        if cycle in self._decayed_cycles:
            return
        self._decayed_cycles.add(cycle)
        observed = self._observed_in_cycle.get(cycle, set())
        for ent in self._entities.values():
            if ent.id in observed:
                continue
            for key, prop in ent.properties.items():
                if key == "exists":
                    continue
                prop.confidence = max(0.0, prop.confidence * PERMANENCE_DECAY)
            for rel in self._relations:
                if rel.subject == ent.id:
                    rel.confidence = max(0.0, rel.confidence * PERMANENCE_DECAY)
        # Phase 12 (P12-W7): prune dead cycle bookkeeping (see
        # docstring). The cycle just decayed is itself retained, so
        # a repeat decay(cycle) still hits the guard above; only
        # strictly older cycles — never read again — are dropped.
        for c in [c for c in self._observed_in_cycle if c < cycle]:
            del self._observed_in_cycle[c]
        for c in [c for c in self._decayed_cycles if c < cycle]:
            self._decayed_cycles.discard(c)

    def mark_gone(self, entity: str, reason: str, cycle: int) -> Entity:
        """Mark an entity gone — ONLY on observed contradictory evidence
        (call this explicitly; unobserved is never treated as gone).
        Sets ``exists=False`` with status observed, confidence 1.0."""
        entity = str(entity)
        ent = self._ensure_entity(entity, "object", int(cycle))
        ent.last_seen_cycle = max(ent.last_seen_cycle, int(cycle))
        ent.properties["exists"] = EntityProperty(
            value=False, status=EpistemicStatus.observed, confidence=1.0,
            provenance=f"gone:{reason}")
        self._append_event(WorldEvent(
            cycle=int(cycle), kind="gone", entity=entity, key="exists",
            old_value=True, new_value=False, cause=str(reason)))
        return ent

    def is_gone(self, entity: str) -> bool:
        ent = self._entities.get(str(entity))
        if ent is None:
            return False
        prop = ent.properties.get("exists")
        return prop is not None and prop.value is False

    def mark_single_use_suspect(self, entity_name: str, cycle: int) -> Entity:
        """Flag a tool entity as suspected single-use: property
        ``single_use_suspect=True``, status inferred, confidence 0.7.
        (Called by the brain when a tool has produced its effect once
        and is demanded again; the pattern library reads this flag.)"""
        return self.set_property(
            entity_name, "single_use_suspect", True,
            status=EpistemicStatus.inferred, confidence=0.7,
            provenance="single-use-suspect", cycle=int(cycle))

    # ---- contradictions --------------------------------------------------

    def note_contradiction(self, entity: str, property: str, value_a, value_b,
                           cycle: int) -> dict:
        """Record two values that cannot both be true for the same
        (entity, property). Both are kept in the disputed list; the
        original entries are NEVER overwritten or silently resolved."""
        record = {"entity": str(entity), "property": str(property),
                  "value_a": value_a, "value_b": value_b,
                  "cycle": int(cycle), "status": "disputed"}
        self._contradictions.append(record)
        self._append_event(WorldEvent(
            cycle=int(cycle), kind="contradiction", entity=str(entity),
            key=str(property), old_value=value_a, new_value=value_b,
            cause="contradiction"))
        return record

    def contradictions(self) -> list[dict]:
        return [dict(c) for c in self._contradictions]

    # ---- prediction ------------------------------------------------------

    def predict(self, action_name: str, state: dict, effects_fn, causal=None):
        """(expected_state, confidence): the planner's effect model
        applied to ``state``, with confidence from the causal record
        when one exists for an (action, effect) pair — Laplace-smoothed
        ``(met + 1) / (total + 2)`` over the weakest (lowest-rate)
        claimed effect — else 1.0 (the planner's own model, no evidence
        yet). ``causal`` is a CausalModel (or anything exposing
        ``success_rate(action, effect)``)."""
        state = dict(state or {})
        expected = dict(state)
        effects = {}
        if effects_fn is not None:
            effects = dict(effects_fn(str(action_name), state) or {})
        expected.update(effects)
        if causal is None or not effects:
            return expected, 1.0
        confidence = None
        for effect_key in effects:
            rate = None
            records = getattr(causal, "_records", None)
            if isinstance(records, dict):
                try:
                    from ..core.vocab import normalize_text
                    cell = records.get((normalize_text(action_name),
                                        normalize_text(effect_key)))
                except Exception:
                    cell = None
                if cell is not None and cell[1] > 0:
                    rate = (cell[0] + 1) / (cell[1] + 2)
            if rate is None:
                fn = getattr(causal, "success_rate", None)
                if callable(fn):
                    raw = fn(action_name, effect_key)
                    if raw is not None:
                        rate = float(raw)
            if rate is not None:
                confidence = rate if confidence is None else min(confidence, rate)
        return expected, (1.0 if confidence is None else confidence)

    def record_prediction_error(self, action: str, effect: str, expected, actual,
                                cycle: int) -> dict:
        """Store one prediction error. ``error`` is 0/1 for discrete
        values, ``abs(expected - actual)`` when both are numbers."""
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
                and not isinstance(expected, bool) and not isinstance(actual, bool):
            error = abs(expected - actual)
        else:
            error = 0 if expected == actual else 1
        record = {"action": str(action), "effect": str(effect),
                  "expected": expected, "actual": actual,
                  "cycle": int(cycle), "error": error}
        self._prediction_errors.append(record)
        return record

    def prediction_errors(self) -> list[dict]:
        return [dict(r) for r in self._prediction_errors]

    def error_rate(self, action: str, effect: str):
        """Mean stored error for (action, effect), or None when no
        prediction error has been recorded for the pair."""
        errs = [r["error"] for r in self._prediction_errors
                if r["action"] == str(action) and r["effect"] == str(effect)]
        if not errs:
            return None
        return sum(errs) / len(errs)

    # ---- revision ----------------------------------------------------------

    def revise(self, key: str, old, new, reason: str, cycle: int) -> str:
        """Revise the model entry for ``key``: the old value is RETAINED
        in the append-only ``revisions`` list and the current entry is
        replaced with status ``observed``. Returns the revised key."""
        key = str(key)
        cycle = int(cycle)
        entity_id, kind, prop = _map_key_to_entity_property(key)
        prior = None
        ent = self._entities.get(entity_id)
        if ent is not None and prop in ent.properties:
            prior = _status(ent.properties[prop].status).value
        self._revisions.append({"key": key, "old_value": old, "new_value": new,
                                "reason": str(reason), "cycle": cycle,
                                "prior_status": prior})
        ent = self._ensure_entity(entity_id, kind, cycle)
        ent.last_seen_cycle = max(ent.last_seen_cycle, cycle)
        ent.properties[prop] = EntityProperty(
            value=new, status=EpistemicStatus.observed, confidence=1.0,
            provenance=f"revision:{reason}")
        self._observed_keys.add(key)
        self._append_event(WorldEvent(
            cycle=cycle, kind="revision", entity=entity_id, key=key,
            old_value=old, new_value=new, cause=str(reason)))
        return key

    def revisions(self) -> list[dict]:
        return [dict(r) for r in self._revisions]

    # ---- observability -----------------------------------------------------

    def observed_keys(self) -> set[str]:
        return set(self._observed_keys)

    def unobserved_keys(self, known_keys) -> list[str]:
        """Keys in ``known_keys`` never observed: unknown, never assumed
        absent (and never assumed False)."""
        return [k for k in known_keys if k not in self._observed_keys]

    # ---- persistence ---------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "entities": [e.to_json() for e in self.entities()],
            "relations": [r.to_json() for r in self._relations],
            "events": [e.to_json() for e in self._events],
            "revisions": [dict(r) for r in self._revisions],
            "prediction_errors": [dict(r) for r in self._prediction_errors],
            "contradictions": [dict(c) for c in self._contradictions],
            "constraints": [{"name": n, "description": c["description"]}
                            for n, c in sorted(self._constraints.items())],
            "observed_keys": sorted(self._observed_keys),
            "adjacency": {room: sorted(nbrs)
                          for room, nbrs in sorted(self._adjacency.items())},
            "door_gated_rooms": sorted(self._door_gated_rooms),
            "observed_in_cycle": {str(c): sorted(ids)
                                  for c, ids in sorted(self._observed_in_cycle.items())},
            "decayed_cycles": sorted(self._decayed_cycles),
        }

    @classmethod
    def from_json(cls, data: dict) -> "WorldModel":
        if not isinstance(data, dict):
            raise ValueError("world model file must be a dict")
        model = cls()
        for d in data.get("entities", []):
            ent = Entity.from_json(d)
            model._entities[ent.id] = ent
        for d in data.get("relations", []):
            model._relations.append(Relation.from_json(d))
        for d in data.get("events", []):
            model._append_event(WorldEvent.from_json(d))
        model._revisions = [dict(r) for r in data.get("revisions", [])]
        model._prediction_errors = [dict(r) for r in data.get("prediction_errors", [])]
        model._contradictions = [dict(c) for c in data.get("contradictions", [])]
        # Constraints persist by NAME only: check functions cannot be
        # serialised, so the caller re-registers custom ones on load.
        # Built-ins are re-registered here with their real functions.
        model._constraints = {}
        model._register_builtins()
        for c in data.get("constraints", []):
            name = str(c.get("name", ""))
            if name and name not in model._constraints:
                model._constraints[name] = {
                    "check_fn": None,
                    "description": str(c.get("description", ""))}
        model._observed_keys = set(data.get("observed_keys", []))
        model._adjacency = {str(room): set(nbrs)
                            for room, nbrs in data.get("adjacency", {}).items()}
        model._door_gated_rooms = set(data.get("door_gated_rooms", []))
        model._observed_in_cycle = {
            int(c): set(ids) for c, ids in
            data.get("observed_in_cycle", {}).items()}
        model._decayed_cycles = set(data.get("decayed_cycles", []))
        return model

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)

