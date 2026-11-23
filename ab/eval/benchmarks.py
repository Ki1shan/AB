"""Phase 10 benchmark worlds (design docs/phase10-design.md P10-3/P10-4).

Nine named Tier 2/3 tasks, each with a pass/fail criterion, each run
for full / no_learning / non_integrated / fresh (see
``run_benchmarks``). Worlds live ONLY in this file; the brain never
sees a benchmark's names or solution, and no benchmark-specific
logic lives anywhere the brain could exploit -- the runner only
observes brain-observable state (goals, events, learner episodes,
the env log) and drives the harness-level protocols the design
names (partial-map discovery via public ``Planner.add_action``,
mid-task goal swaps, save/load).

New invented names (eval-only; add them to the leakage audit):
``quendal``, ``brelm``, ``sorvik``, ``tarnel``, ``vessik``,
``mordak``, ``ullin``, ``zavrel``, ``nolvak``, ``tremik``,
``brellik``, ``sorvane``, ``kendrell``, ``nolvar``, ``vessal``,
``mordune``, ``ullik``, ``zavrik``, ``nolvik``, ``vaskel``,
``drelk``, ``durnik``, ``sarnik``, ``dralik``, ``senvor``,
``quilm``, ``keepsake``, ``zarnik``, ``phelik``, ``frelk``,
``cross_rift``, ``wax_seal``, ``press_seal``, ``stoke_coil``
(the last is shared with the hard surprise envgen world), plus
the hard-envgen decoy names ``thrum``, ``clack``, ``toll``.

Measurement (P10-2 audit): ``FAILURE_MARKERS`` in
``ab/eval/consolidated.py`` does NOT match several real env failure
strings (see :func:`audit_failure_markers`). This module therefore
counts failed attempts BOTH from the env log (consolidated markers)
and from the brain's learner episodes (``ok=False``), records both,
and treats the episode count as authoritative when they disagree
(episodes are the brain's own record of what it tried and whether
the effect materialised -- including failures the env reports as
``ok=True`` with a non-effect, e.g. the surprise world's cold-coil
heat). For the ``no_learning`` variant the learner is a no-op, so
episodes are unavailable and the log count is the authority there;
that is recorded per run, not hidden.

No LLM, stdlib only, deterministic per seed (``random.Random(seed)``
in eval code only; the brain never sees a seed).
"""
from __future__ import annotations

import json
import random
import tempfile

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action
from .envgen import generate as _generate_hard_env
from .reasoning_tasks import make_variant_factories
from .transfer import prepare_brain_for_new_world

try:
    from .consolidated import FAILURE_MARKERS as _CONSOLIDATED_MARKERS
except Exception:  # consolidated must stay importable; literal fallback
    _CONSOLIDATED_MARKERS = ("does not move", "is dust", "no intact",
                             "nothing happened", "cannot reach")
CONSOLIDATED_FAILURE_MARKERS = tuple(_CONSOLIDATED_MARKERS)

# Characteristic failure strings produced by the baseline envgen
# worlds (and the chamber/workshop worlds). The audit checks each
# against the consolidated markers by substring match.
AUDITED_FAILURE_STRINGS = {
    "treasure/cannot-reach": "cannot reach {room}",
    "workshop/press-no-mould": "the press has no mould seated",
    "workshop/cut-nothing": "nothing to cut here",
    "surprise/cold-coil": "the coil hums but the substance stays cold",
    "chamber/decoy": "the air hums, but the door does not move",
    "chamber/fragile-dust": "the tessera is dust; it cannot turn the mechanism",
    "chamber/fragile-gone": "no intact tessera remains here",
    "workshop/assemble-parts": "parts missing for assembly",
}

# Failure phrases this module's own worlds can produce (used only
# to *describe* log lines in evidence, never to steer the brain).
_KNOWN_FAILURE_PHRASES = (
    "does not move", "is dust", "no intact", "nothing happened",
    "cannot reach", "stays cold", "nothing to cut here",
    "parts missing", "cannot cut", "too cold to stoke",
    "does not hold", "no charges left", "melts away",
    "has no mould seated", "has no die seated", "has no form seated",
    "shatters into dust",
)

# New invented names, for the leakage audit (see module docstring).
NEW_NAMES = (
    "quendal", "brelm", "sorvik", "tarnel", "vessik", "mordak",
    "ullin", "zavrel", "nolvak", "tremik", "brellik", "sorvane",
    "kendrell", "nolvar", "vessal", "mordune", "ullik", "zavrik",
    "nolvik", "vaskel", "drelk", "durnik", "sarnik", "dralik",
    "senvor", "quilm", "keepsake", "zarnik", "phelik", "frelk",
    "cross_rift", "wax_seal", "press_seal", "stoke_coil",
    "thrum", "clack", "toll",
)

MAX_CYCLES = 80
VARIANT_NAMES = ("full", "no_learning", "non_integrated", "fresh")


# ------------------------------------------------------------------ measurement audit (P10-2)
def audit_failure_markers() -> dict:
    """Check every characteristic env failure string against the
    consolidated ``FAILURE_MARKERS`` by substring match, and report
    which real failures the consolidated metric would miss. Returns
    ``{"markers": [...], "strings": {name: {"string", "matched",
    "marker"}}, "missed": [...], "verdict": str}``. This is a static
    string audit (the strings are the envs' own log lines, quoted in
    ``AUDITED_FAILURE_STRINGS``); the empirical leg -- a real failing
    run counted both ways -- lives in :func:`count_failures` and in
    the tests."""
    strings = {}
    missed = []
    for name, text in AUDITED_FAILURE_STRINGS.items():
        marker = next((m for m in CONSOLIDATED_FAILURE_MARKERS if m in text),
                      None)
        strings[name] = {"string": text, "matched": marker is not None,
                         "marker": marker}
        if marker is None:
            missed.append(name)
    verdict = (
        f"{len(missed)}/{len(AUDITED_FAILURE_STRINGS)} characteristic "
        f"failure strings match no consolidated FAILURE_MARKER: {missed}. "
        f"Failed attempts counted from the env log alone undercount "
        f"real failures; the learner-episode count is authoritative "
        f"in this module (see count_failures).")
    return {"markers": list(CONSOLIDATED_FAILURE_MARKERS),
            "strings": strings, "missed": missed, "verdict": verdict}


def count_failures(env_log, brain, episodes_before: int = 0,
                   episodes_available: bool = True) -> dict:
    """Count failed attempts two ways and reconcile them.

    ``log_count``: env-log lines matching a consolidated
    ``FAILURE_MARKER``. ``episode_count``: new learner episodes with
    ``ok=False`` since ``episodes_before`` (the brain's own record;
    counts failures the env reports as ``ok=True`` but whose effect
    never materialised, e.g. a cold-coil heat). ``authoritative`` is
    the episode count when episodes are available (the real learner
    records them), else the log count (the ``no_learning`` variant's
    learner is a no-op, so no episodes exist to count). ``disagree``
    is True when both counts are available and differ.
    ``unmatched_failure_lines`` lists log lines that carry a known
    failure phrase but match no consolidated marker -- the audit's
    empirical leg."""
    log = [str(e) for e in (env_log or [])]
    log_count = sum(1 for e in log
                    if any(m in e for m in CONSOLIDATED_FAILURE_MARKERS))
    episodes = list(getattr(getattr(brain, "learner", None), "episodes", [])
                    or [])
    new_episodes = episodes[episodes_before:] if episodes_before <= len(episodes) else []
    episode_count = sum(1 for ep in new_episodes if not getattr(ep, "ok", True))
    unmatched = [e for e in log
                 if not any(m in e for m in CONSOLIDATED_FAILURE_MARKERS)
                 and any(p in e for p in _KNOWN_FAILURE_PHRASES)]
    if episodes_available:
        authoritative, source = episode_count, "learner_episodes"
    else:
        authoritative, source = log_count, "env_log (episodes unavailable: no_learning)"
    return {"log_count": log_count, "episode_count": episode_count,
            "authoritative": authoritative,
            "authoritative_source": source,
            "disagree": bool(episodes_available and episode_count != log_count),
            "unmatched_failure_lines": unmatched,
            "episodes_available": bool(episodes_available)}


def _episodes_before(brain) -> int:
    return len(list(getattr(getattr(brain, "learner", None), "episodes", []) or []))


def _events(brain, *kinds: str, start: int = 0) -> list[dict]:
    return [e for e in list(brain.events or [])[start:]
            if e.get("kind") in kinds]


def _result(passed: bool, cycles: int, detail: str, evidence: dict,
            *, tier: int, failure_cause: str | None = None,
            legs: dict | None = None,
            difficulty: dict | None = None) -> dict:
    out = {"passed": bool(passed), "cycles": int(cycles), "detail": detail,
           "evidence": evidence, "tier": int(tier),
           "failure_cause": (None if passed else failure_cause),
           "legs": dict(legs or {})}
    if difficulty is not None:
        out["difficulty"] = dict(difficulty)
    return out


# ------------------------------------------------------------------ tiers (P10.5-6)
def _active_dimensions(difficulty: dict) -> tuple[int, list[str]]:
    """Count the difficulty dimensions that are "active" for a task,
    from its 7-dimension difficulty vector (design P10-3). A
    dimension is active when:

    - ``n_rooms`` > 1 (a multi-room world),
    - ``n_decoys`` >= 1 (a decoy use or decoy object exists),
    - ``resource_units`` is present (not None -- a bounded
      consumable exists),
    - ``n_dynamic_events`` >= 1 (the env changes mid-task),
    - ``observability`` == "partial" (the brain does not see the
      whole world),
    - ``n_goals`` > 1 (more than one goal is live),
    - ``horizon_min_steps`` > 5 (a minimum solution longer than
      the 5-step baseline: 5 is the modal minimum horizon across
      the nine tasks -- five of nine sit at exactly 5 -- so a
      horizon above it marks a genuinely longer-horizon task).

    Returns ``(count, [names of the active dimensions])``. Raises
    ``ValueError`` on a missing or malformed vector (absent key,
    wrong type, unknown observability value) -- a tier must never
    be derived from a vector that cannot be read.
    """
    if not isinstance(difficulty, dict):
        raise ValueError(
            "cannot derive a tier: difficulty vector is not a dict: "
            f"{difficulty!r}")
    for key in ("n_rooms", "n_decoys", "resource_units",
                "n_dynamic_events", "observability", "n_goals",
                "horizon_min_steps"):
        if key not in difficulty:
            raise ValueError(
                "cannot derive a tier: difficulty vector is missing "
                f"dimension {key!r}: {difficulty!r}")
    for key in ("n_rooms", "n_decoys", "n_dynamic_events", "n_goals",
                "horizon_min_steps"):
        if not isinstance(difficulty[key], int) or isinstance(
                difficulty[key], bool):
            raise ValueError(
                "cannot derive a tier: difficulty dimension "
                f"{key!r} is not an int: {difficulty[key]!r}")
    if (difficulty["resource_units"] is not None
            and not isinstance(difficulty["resource_units"], int)):
        raise ValueError(
            "cannot derive a tier: difficulty dimension "
            "'resource_units' is neither an int nor None: "
            f"{difficulty['resource_units']!r}")
    if difficulty["observability"] not in ("full", "partial"):
        raise ValueError(
            "cannot derive a tier: difficulty dimension "
            "'observability' is neither 'full' nor 'partial': "
            f"{difficulty['observability']!r}")
    active: list[str] = []
    if difficulty["n_rooms"] > 1:
        active.append("n_rooms")
    if difficulty["n_decoys"] >= 1:
        active.append("n_decoys")
    if difficulty["resource_units"] is not None:
        active.append("resource_units")
    if difficulty["n_dynamic_events"] >= 1:
        active.append("n_dynamic_events")
    if difficulty["observability"] == "partial":
        active.append("observability")
    if difficulty["n_goals"] > 1:
        active.append("n_goals")
    if difficulty["horizon_min_steps"] > 5:
        active.append("horizon_min_steps")
    return len(active), active


def _tier_for(difficulty: dict, kind: str) -> int:
    """Tier (1|2|3, design P10-3) for a benchmark task, from its
    difficulty vector and its world kind -- never from the task id
    string. ``kind`` is one of:

    - ``"existing"`` -- the Tier-1 baseline worlds (chamber /
      treasure / workshop / surprise). None of the nine benchmark
      tasks is one; the value exists so the rule is complete.
    - ``"generated"`` -- a Tier-2 generated harder env (the envgen
      worlds of ``transfer_complex``).
    - ``"single"`` -- a Tier-2 single-mechanism harder world (one
      challenge mechanism isolated).
    - ``"composite"`` -- a Tier-3 composite world (several
      challenge mechanisms combined at once).

    The rule: kind ``"existing"`` -> 1; kind ``"composite"`` -> 3;
    kinds ``"generated"`` / ``"single"`` -> 2. The active-dimension
    count (see :func:`_active_dimensions`) is the cross-check, not
    the determinant, and a composite kind with fewer than 2 active
    dimensions raises ``ValueError`` (kind and vector disagree --
    a composite world must combine multiple dimensions).

    Why the kind is needed, stated plainly: the vector alone
    cannot determine the tier. Proof in the nine tasks'
    vectors (seed 0): ``unexpected_event`` and
    ``unrecoverable_failure`` share the IDENTICAL vector (3 rooms,
    0 decoys, no resource, 1 dynamic event, full observability, 1
    goal, horizon 5) yet are Tier 2 and Tier 3 by design -- the
    difference is the world kind (a recoverable gate reset vs the
    terminal destruction of the only key). And
    ``find_and_use_resources`` / ``changing_objective`` each show 3
    active dimensions yet are Tier 2: their extra dimensions are
    incidental structure (a second room for the press; a second
    goal that replaces the first), not a composite of challenge
    mechanisms. So the count is recorded and guarded, and the world
    kind -- declared where the world is built, never matched on
    the task id -- decides the tier. An unknown kind raises
    ``ValueError``: a task with no determinable tier fails loudly,
    never silently defaults.
    """
    if kind not in ("existing", "generated", "single", "composite"):
        raise ValueError(
            "cannot derive a tier: unknown world kind "
            f"{kind!r} (expected 'existing', 'generated', 'single' "
            "or 'composite')")
    count, _active = _active_dimensions(difficulty)
    if kind == "existing":
        return 1
    if kind == "composite":
        if count < 2:
            raise ValueError(
                "cannot derive a tier: composite world kind with "
                f"only {count} active difficulty dimension(s) "
                f"({_active}) -- a composite world must combine "
                "multiple dimensions; kind and vector disagree")
        return 3
    return 2


def _run_brain(factory, actions, description: str, target: dict, env,
               max_cycles: int = MAX_CYCLES, priority: float = 0.5,
               brain=None, variant: str = "full"):
    """Run one (actions, goal, env) triple through a brain. A carried
    brain gets the new actions added and its transient state cleared
    (the transfer protocol); a fresh brain is built by the factory.
    Returns ``(record, brain)``; ``record`` carries the dual failure
    count (see :func:`count_failures`). For the ``no_learning``
    variant the learner is a no-op, so episodes are unavailable and
    the log count is authoritative there."""
    episodes_available = variant != "no_learning"
    if brain is None:
        brain = factory(list(actions), {"compute", "environment"})
        limit = max_cycles
    else:
        # Coordinator integration (P10-56 follow-up): the carried
        # brain already holds the earlier world's actions; adding a
        # new world's actions blindly duplicates any action identical
        # in both worlds (1 shared action on the hard workshop pair
        # for _run_generated_leg). Dedupe by full action identity.
        existing = {_action_identity(a) for a in brain.planner.actions}
        for a in actions:
            ident = _action_identity(a)
            if ident not in existing:
                brain.planner.add_action(a)
                existing.add(ident)
        prepare_brain_for_new_world(brain)
        limit = brain.cycle + max_cycles
    ep0 = _episodes_before(brain)
    cycle_before = brain.cycle
    goal = brain.add_goal(description, dict(target), priority=priority)
    stop = brain.run(env, limit)
    goal_obj = brain.goals.all()[-1]
    failures = count_failures(getattr(env, "log", []), brain, ep0,
                              episodes_available=episodes_available)
    return {"completed": goal_obj.status == "done",
            "goal_status": goal_obj.status, "stop_reason": stop,
            "cycles": brain.cycle - cycle_before,
            "failed_attempts": failures["authoritative"],
            "failures": failures,
            "env_log": list(getattr(env, "log", []) or [])}, brain


def _step_until(brain, env, predicate, cap: int):
    """Step the brain until ``predicate()`` is True or ``cap`` steps
    have run. Returns the number of steps run."""
    used = 0
    while used < cap and not predicate():
        brain.step(env)
        used += 1
    return used


def _go_actions(rooms, gate_last: bool = False, gate_room: str | None = None,
                cost: float = 0.4) -> list[Action]:
    """Bidirectional ``go`` actions along a room chain. The last hop
    is door-gated when ``gate_last`` (precondition ``door_open``);
    ``gate_room`` gates entry to that room instead."""
    actions = []
    for a, b in zip(rooms, rooms[1:]):
        if gate_last and b == rooms[-1]:
            pre = {"at": a, "door_open": True}
        elif gate_room is not None and b == gate_room:
            pre = {"at": a, "door_open": True}
        else:
            pre = {"at": a}
        actions.append(Action(f"go_{b}_from_{a}", pre, {"at": b}, cost=cost,
                              args={"_do": "go", "room": b}))
        actions.append(Action(f"go_{a}_from_{b}", {"at": b}, {"at": a},
                              cost=cost, args={"_do": "go", "room": a}))
    return actions


# ------------------------------------------------------------------ (1) navigate_unfamiliar
NAV_ROOMS = ("quendal", "brelm", "sorvik", "tarnel", "vessik")
NAV_DIFFICULTY = {"n_rooms": 5, "n_decoys": 0, "resource_units": None,
                  "n_dynamic_events": 0, "observability": "partial",
                  "n_goals": 1, "horizon_min_steps": 9}


class PartialNavWorld(Environment):
    """5-room chain, partial map: ``observe()`` returns only the
    current room, its adjacent room names, and the agent's own
    holdings / door / treasure flags -- never the name of a room
    beyond an adjacent one, and never another room's contents. The
    key waits in room 2, the door gates room 5, the treasure waits
    there."""

    def __init__(self):
        self.rooms = list(NAV_ROOMS)
        self.adj = {r: [] for r in self.rooms}
        for a, b in zip(self.rooms, self.rooms[1:]):
            self.adj[a].append(b)
            self.adj[b].append(a)
        self.s = {"at": self.rooms[0], "has_vaskel": False,
                  "door_open": False, "has_durnik": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        at = self.s["at"]
        return {"at": at, "adjacent": list(self.adj[at]),
                "has_vaskel": self.s["has_vaskel"],
                "door_open": self.s["door_open"],
                "has_durnik": self.s["has_durnik"]}

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.adj[s["at"]] and (
                    dest != self.rooms[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_vaskel" and s["at"] == self.rooms[1] \
                and not s["has_vaskel"]:
            s["has_vaskel"], ok, info = True, True, "took the vaskel"
        elif name == "unlock" and s["at"] == self.rooms[1] and s["has_vaskel"]:
            s["door_open"], ok, info = True, True, "unlocked the door"
        elif name == "take_durnik" and s["at"] == self.rooms[-1]:
            s["has_durnik"], ok, info = True, True, "took the durnik"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _nav_actions_for(room: str) -> list[Action]:
    """Actions whose room becomes known when ``room`` is discovered:
    the key/unlock pair in room 2, the treasure take in room 5."""
    if room == NAV_ROOMS[1]:
        return [
            Action("take_vaskel", {"at": room, "has_vaskel": False},
                   {"has_vaskel": True}, cost=0.5,
                   args={"_do": "take_vaskel"}),
            Action("unlock", {"at": room, "has_vaskel": True},
                   {"door_open": True}, cost=0.5, args={"_do": "unlock"}),
        ]
    if room == NAV_ROOMS[-1]:
        return [Action("take_durnik", {"at": room}, {"has_durnik": True},
                       cost=0.5, args={"_do": "take_durnik"})]
    return []


def run_navigate_unfamiliar(factory, seed: int = 0,
                            variant: str = "full") -> dict:
    """Task (1): explore a 5-room chain under partial observation.
    The runner starts the brain with ``go`` actions only to the one
    adjacent room it can see; as the brain arrives in each room the
    runner adds ``go`` actions for the newly observed adjacent rooms
    (public ``Planner.add_action``) plus that room's own actions.
    Exploration is therefore harness-orchestrated via successive
    subgoals -- the planner cannot plan to a room whose ``go``
    action does not exist yet, so the brain itself never plans with
    an undiscovered room (that is the checked property, recorded
    per leg in the evidence). Pass: all 5 rooms discovered in
    order, the treasure goal done, and no plan step ever named an
    undiscovered room."""
    env = PartialNavWorld()
    rooms = list(NAV_ROOMS)
    brain = factory([], {"compute", "environment"})
    discovered = [rooms[0]]
    # Names observed (discovery) vs rooms actually arrived in: a
    # room's name is observed while standing in its neighbour, so
    # the discovery order legitimately runs one step ahead of the
    # arrival order. The pass leg checks ARRIVAL order.
    arrived = [rooms[0]]
    plan_violations: list[str] = []
    cycles = 0
    ep0 = _episodes_before(brain)
    events_before = len(brain.events)

    def _check_plan():
        # Every step the brain currently plans must name a
        # discovered room (a go destination) or a discovered room's
        # action; the planner literally cannot name anything else,
        # because actions for undiscovered rooms do not exist yet.
        known_dests = set(discovered)
        for step in list(brain.plan or []):
            dest = (step.args or {}).get("room")
            if dest is not None and dest not in known_dests:
                plan_violations.append(
                    f"plan step {step.name} targets undiscovered room {dest}")

    def _go_pair(a: str, b: str):
        # Bidirectional go actions between discovered rooms a/b.
        # Entry into the last room is door-gated in the believed
        # model (as the env enforces it): without the gate in the
        # model the brain tries the gated hop unkeyed, the world
        # model learns the door blocks entry, and every later plan
        # is rejected as impossible.
        pre_ab = {"at": a, "door_open": True} if b == rooms[-1] else {"at": a}
        pre_ba = {"at": b, "door_open": True} if a == rooms[-1] else {"at": b}
        brain.planner.add_action(Action(
            f"go_{b}_from_{a}", pre_ab, {"at": b}, cost=0.4,
            args={"_do": "go", "room": b}))
        brain.planner.add_action(Action(
            f"go_{a}_from_{b}", pre_ba, {"at": a}, cost=0.4,
            args={"_do": "go", "room": a}))

    def _discover(room: str):
        if room in discovered:
            return
        # Add go actions only between this room and rooms ALREADY
        # discovered: an edge to a room whose name has never been
        # observed must not become plannable yet.
        for n in env.adj[room]:
            if n not in discovered:
                continue
            _go_pair(room, n)
        discovered.append(room)
        for a in _nav_actions_for(room):
            brain.planner.add_action(a)

    # Starting position: only the adjacent room is known.
    for n in env.adj[rooms[0]]:
        _go_pair(rooms[0], n)
    for target in rooms[1:]:
        goal = brain.add_goal(f"reach {target}", {"at": target})
        before = brain.cycle
        stop = brain.run(env, brain.cycle + 25)
        cycles += brain.cycle - before
        _check_plan()
        if goal.status != "done":
            failures = count_failures(env.log, brain, ep0,
                                      episodes_available=(variant != "no_learning"))
            return _result(False, cycles,
                           f"navigate: stuck reaching {target} "
                           f"(goal {goal.status}, stop {stop})",
                           {"discovered": list(discovered),
                            "target": target, "stop_reason": stop,
                            "failures": failures,
                            "plan_violations": plan_violations,
                            "env_log": list(env.log)},
                           failure_cause=f"exploration stalled at {target}",
                           legs={"reached_all_rooms": False,
                                 "no_undiscovered_room_planned":
                                     not plan_violations},
                           difficulty=NAV_DIFFICULTY, tier=NAV_TIER)
        # Arrived: the newly observed adjacent rooms become known.
        arrived.append(target)
        for n in env.observe().get("adjacent", []):
            _discover(n)
        _discover(target)
        _check_plan()
    goal = brain.add_goal("take the durnik", {"has_durnik": True})
    before = brain.cycle
    stop = brain.run(env, brain.cycle + 25)
    cycles += brain.cycle - before
    _check_plan()
    failures = count_failures(env.log, brain, ep0,
                              episodes_available=(variant != "no_learning"))
    legs = {"reached_all_rooms": arrived == rooms,
            "treasure_taken": goal.status == "done",
            "no_undiscovered_room_planned": not plan_violations}
    passed = all(legs.values())
    return _result(passed, cycles,
                   (f"navigate: discovered {discovered}, treasure "
                    f"{goal.status}, plan violations={plan_violations}"
                    if passed else
                    f"navigate FAILED: treasure {goal.status}, stop {stop}, "
                    f"discovered={discovered}, violations={plan_violations}"),
                   {"discovered": list(discovered),
                    "arrived_in_order": list(arrived),
                    "goal_status": goal.status, "stop_reason": stop,
                    "failures": failures,
                    "plan_violations": plan_violations,
                    "events_model_revised": len(_events(brain, "model_revised", start=events_before)),
                    "env_log": list(env.log)},
                   failure_cause=(None if passed else
                                  f"treasure goal {goal.status} after full discovery"),
                   legs=legs, difficulty=NAV_DIFFICULTY, tier=NAV_TIER)


# ------------------------------------------------------------------ (2) find_and_use_resources
RESOURCE_DIFFICULTY = {"n_rooms": 2, "n_decoys": 1, "resource_units": 2,
                       "n_dynamic_events": 0, "observability": "full",
                       "n_goals": 1, "horizon_min_steps": 4}


class ResourceWorld(Environment):
    """Two rooms, one consumable with 2 charges. The seal the
    assembly needs can be made two ways: the cheap ``wax_seal``
    (wastes a charge and never holds) and the ``press_seal`` (works).
    A brain that wastes both charges on the decoy can never finish."""

    def __init__(self):
        self.s = {"at": "mordak", "charges": 2, "sealed": False,
                  "has_frelk": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in ("mordak", "ullin") and dest != s["at"]:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "wax_seal" and s["at"] == "ullin":
            if s["charges"] >= 1:
                s["charges"] -= 1
                info = "the wax melts away; the seal does not hold"
            else:
                info = "no charges left"
        elif name == "press_seal" and s["at"] == "ullin":
            if s["charges"] >= 1:
                s["charges"] -= 1
                s["sealed"], ok, info = True, True, "the press seals the parts"
            else:
                info = "no charges left"
        elif name == "assemble" and s["at"] == "ullin" and s["sealed"]:
            s["has_frelk"], ok, info = True, True, "assembled the frelk"
        elif name == "assemble":
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _resource_actions() -> list[Action]:
    return [
        Action("go_ullin_from_mordak", {"at": "mordak"}, {"at": "ullin"},
               cost=0.4, args={"_do": "go", "room": "ullin"}),
        Action("go_mordak_from_ullin", {"at": "ullin"}, {"at": "mordak"},
               cost=0.4, args={"_do": "go", "room": "mordak"}),
        Action("wax_seal", {"at": "ullin"}, {"sealed": True}, cost=0.3,
               args={"_do": "wax_seal"}),
        Action("press_seal", {"at": "ullin"}, {"sealed": True}, cost=1.0,
               args={"_do": "press_seal"}),
        Action("assemble", {"at": "ullin", "sealed": True},
               {"has_frelk": True}, cost=1.0, args={"_do": "assemble"}),
    ]


def run_find_and_use_resources(factory, seed: int = 0,
                               variant: str = "full") -> dict:
    """Task (2): 2 charges, one decoy use that wastes a charge and
    never holds. Pass: the goal is done and the resource was never
    exhausted before the final step -- i.e. no ``press_seal`` ever
    failed with ``no charges left`` and at most one charge was
    wasted on the decoy."""
    env = ResourceWorld()
    rec, brain = _run_brain(factory, _resource_actions(),
                            "assemble the frelk", {"has_frelk": True}, env,
                            variant=variant)
    wax_attempts = sum(1 for e in env.log if e.startswith("wax_seal"))
    press_no_charge = sum(1 for e in env.log
                          if e.startswith("press_seal") and "no charges left" in e)
    assemble_idx = next((i for i, e in enumerate(env.log)
                         if e.startswith("assemble") and "assembled" in e), None)
    press_idx = next((i for i, e in enumerate(env.log)
                      if e.startswith("press_seal") and "seals the parts" in e), None)
    legs = {
        "goal_completed": bool(rec["completed"]),
        "resource_never_exhausted_before_final_step": press_no_charge == 0,
        "decoy_wasted_at_most_once": wax_attempts <= 1,
        "correct_use_before_assembly": bool(
            press_idx is not None and assemble_idx is not None
            and press_idx < assemble_idx),
    }
    passed = all(legs.values())
    return _result(passed, rec["cycles"],
                   (f"resources: goal {rec['goal_status']}, wax attempts "
                    f"{wax_attempts}, charges left {env.s['charges']}"
                    if passed else
                    f"resources FAILED: goal {rec['goal_status']}, stop "
                    f"{rec['stop_reason']}, wax attempts {wax_attempts}, "
                    f"press with no charges {press_no_charge}"),
                   {"goal_status": rec["goal_status"],
                    "stop_reason": rec["stop_reason"],
                    "wax_attempts": wax_attempts,
                    "press_no_charge_failures": press_no_charge,
                    "charges_left": env.s["charges"],
                    "failures": rec["failures"], "env_log": rec["env_log"]},
                   failure_cause=(None if passed else
                                  "resource exhausted before the final step "
                                  "or goal not completed"),
                   legs=legs, difficulty=RESOURCE_DIFFICULTY, tier=RESOURCE_TIER)


# ------------------------------------------------------------------ (3) changing_objective
CHANGE_DIFFICULTY = {"n_rooms": 3, "n_decoys": 0, "resource_units": None,
                     "n_dynamic_events": 1, "observability": "full",
                     "n_goals": 2, "horizon_min_steps": 5}


class ChangeWorld(Environment):
    """3-room chain, one key, one door, two treasures in the last
    room (``durnik`` = first goal, ``sarnik`` = second goal). Both
    goals need the same key and door, so progress made toward the
    first goal (key obtained, door open) stays useful after the
    objective changes."""

    ROOMS = ("zavrel", "nolvak", "tremik")

    def __init__(self):
        self.s = {"at": self.ROOMS[0], "has_drelk": False,
                  "door_open": False, "has_durnik": False,
                  "has_sarnik": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            adj = {self.ROOMS[0]: [self.ROOMS[1]],
                   self.ROOMS[1]: [self.ROOMS[0], self.ROOMS[2]],
                   self.ROOMS[2]: [self.ROOMS[1]]}
            if dest in adj[s["at"]] and (
                    dest != self.ROOMS[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_drelk" and s["at"] == self.ROOMS[1] \
                and not s["has_drelk"]:
            s["has_drelk"], ok, info = True, True, "took the drelk"
        elif name == "unlock" and s["at"] == self.ROOMS[1] and s["has_drelk"]:
            s["door_open"], ok, info = True, True, "unlocked the door"
        elif name == "take_durnik" and s["at"] == self.ROOMS[-1]:
            s["has_durnik"], ok, info = True, True, "took the durnik"
        elif name == "take_sarnik" and s["at"] == self.ROOMS[-1]:
            s["has_sarnik"], ok, info = True, True, "took the sarnik"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _change_actions() -> list[Action]:
    r = ChangeWorld.ROOMS
    return _go_actions(list(r), gate_last=True) + [
        Action("take_drelk", {"at": r[1], "has_drelk": False},
               {"has_drelk": True}, cost=0.5, args={"_do": "take_drelk"}),
        Action("unlock", {"at": r[1], "has_drelk": True},
               {"door_open": True}, cost=0.5, args={"_do": "unlock"}),
        Action("take_durnik", {"at": r[-1]}, {"has_durnik": True},
               cost=0.5, args={"_do": "take_durnik"}),
        Action("take_sarnik", {"at": r[-1]}, {"has_sarnik": True},
               cost=0.5, args={"_do": "take_sarnik"}),
    ]


def run_changing_objective(factory, seed: int = 0,
                           variant: str = "full") -> dict:
    """Task (3): the goal changes mid-task. The brain runs toward
    ``durnik`` until the door is open; the harness then suspends
    that goal (recorded as an event with the reason) and adds the
    ``sarnik`` goal. Pass: the new goal completes, no stale plan
    step from the old goal (``take_durnik``) is executed after the
    change, and the already-obtained key is not re-taken (useful
    progress kept). The change is a harness event, recorded with
    its cycle; the brain is not told in advance."""
    env = ChangeWorld()
    brain = factory(_change_actions(), {"compute", "environment"})
    ep0 = _episodes_before(brain)
    goal_a = brain.add_goal("take the durnik", {"has_durnik": True})
    _step_until(brain, env,
                lambda: env.s["door_open"] or goal_a.status in ("done", "failed"),
                30)
    if goal_a.status != "active" or not env.s["door_open"]:
        return _result(False, brain.cycle,
                       f"changing_objective: setup failed -- goal A "
                       f"{goal_a.status}, door_open={env.s['door_open']} "
                       f"after {brain.cycle} cycles",
                       {"goal_a_status": goal_a.status,
                        "door_open": env.s["door_open"],
                        "env_log": list(env.log)},
                       failure_cause="could not reach the change point "
                                     "(door never opened)",
                       legs={"change_point_reached": False},
                       difficulty=CHANGE_DIFFICULTY, tier=CHANGE_TIER)
    change_cycle = brain.cycle
    change_log_len = len(env.log)
    suspended = brain.suspend_active_goal(
        f"objective changed at cycle {change_cycle}: new goal is the "
        f"sarnik; the durnik goal is suspended, not dropped")
    goal_b = brain.add_goal("take the sarnik", {"has_sarnik": True},
                            priority=0.9)
    _step_until(brain, env, lambda: goal_b.status == "done", 60)
    cycles = brain.cycle
    after = env.log[change_log_len:]
    stale = [e for e in after if e.startswith("take_durnik")]
    key_retaken = sum(1 for e in after if e.startswith("take_drelk"))
    failures = count_failures(env.log, brain, ep0,
                              episodes_available=(variant != "no_learning"))
    legs = {
        "change_point_reached": True,
        "old_goal_suspended_with_reason": bool(
            suspended is not None and "objective changed" in (suspended.reason or "")),
        "new_goal_completed": goal_b.status == "done",
        "no_stale_plan_step_after_change": not stale,
        "useful_progress_kept_key_not_retaken": key_retaken == 0,
    }
    passed = all(legs.values())
    return _result(passed, cycles,
                   (f"changing_objective: goal B {goal_b.status} after "
                    f"change at cycle {change_cycle}; stale steps "
                    f"{len(stale)}, key re-taken {key_retaken}x"
                    if passed else
                    f"changing_objective FAILED: goal B {goal_b.status}, "
                    f"stale steps {stale}, key re-taken {key_retaken}x"),
                   {"change_cycle": change_cycle,
                    "goal_a_status": goal_a.status,
                    "goal_b_status": goal_b.status,
                    "suspended_record_reason": suspended.reason if suspended else None,
                    "stale_steps_after_change": stale,
                    "key_retaken_after_change": key_retaken,
                    "failures": failures, "env_log": list(env.log)},
                   failure_cause=(None if passed else
                                  "new goal not completed, or a stale step "
                                  "ran after the change, or the key was re-taken"),
                   legs=legs, difficulty=CHANGE_DIFFICULTY, tier=CHANGE_TIER)


# ------------------------------------------------------------------ (4) unexpected_event
UNEXPECTED_DIFFICULTY = {"n_rooms": 3, "n_decoys": 0, "resource_units": None,
                         "n_dynamic_events": 1, "observability": "full",
                         "n_goals": 1, "horizon_min_steps": 5}


class RelockWorld(Environment):
    """3-room chain. The gate to the last room is opened with the
    durable ``reamer``; on the second observation while the gate is
    open it swings shut again (one mid-task reset, recorded in the
    env log as ``reset:``). The brain is not told in advance. The
    treasure ``nolvik`` waits in the last room."""

    ROOMS = ("brellik", "sorvane", "kendrell")

    def __init__(self):
        self.s = {"at": self.ROOMS[0], "has_reamer": False,
                  "door_open": False, "has_nolvik": False}
        self._relock_done = False
        self._open_observes = 0
        self.log: list[str] = []

    def observe(self) -> dict:
        if self.s["door_open"] and not self._relock_done:
            self._open_observes += 1
            if self._open_observes >= 2:
                self.s["door_open"] = False
                self._relock_done = True
                self.log.append("reset: the mechanism resets, the gate swings shut")
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            adj = {self.ROOMS[0]: [self.ROOMS[1]],
                   self.ROOMS[1]: [self.ROOMS[0], self.ROOMS[2]],
                   self.ROOMS[2]: [self.ROOMS[1]]}
            if dest in adj[s["at"]] and (
                    dest != self.ROOMS[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_reamer" and s["at"] == self.ROOMS[1] \
                and not s["has_reamer"]:
            s["has_reamer"], ok, info = True, True, "took the reamer"
        elif name == "pry_gate" and s["at"] == self.ROOMS[1] and s["has_reamer"]:
            s["door_open"], ok, info = True, True, "pried the gate open"
        elif name == "take_nolvik" and s["at"] == self.ROOMS[-1]:
            s["has_nolvik"], ok, info = True, True, "took the nolvik"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _relock_actions() -> list[Action]:
    r = RelockWorld.ROOMS
    return _go_actions(list(r), gate_last=True) + [
        Action("take_reamer", {"at": r[1], "has_reamer": False},
               {"has_reamer": True}, cost=0.5, args={"_do": "take_reamer"}),
        Action("pry_gate", {"at": r[1], "has_reamer": True},
               {"door_open": True}, cost=1.0, args={"_do": "pry_gate"}),
        Action("take_nolvik", {"at": r[-1]}, {"has_nolvik": True},
               cost=0.5, args={"_do": "take_nolvik"}),
    ]


def run_unexpected_event(factory, seed: int = 0,
                         variant: str = "full") -> dict:
    """Task (4): the env injects a mid-task reset (the gate swings
    shut again after first being opened). Pass: the goal completes
    after the event, the event is in the env log, and a
    ``model_revised`` event exists (the brain revised its model in
    response to the violated prediction); a ``world_changed`` event
    is recorded as the detection evidence. If the goal completes
    but no model revision is recorded, that is a failure of the
    criterion, recorded with the cause."""
    env = RelockWorld()
    events_before = 0
    rec, brain = _run_brain(factory, _relock_actions(),
                            "take the nolvik", {"has_nolvik": True}, env,
                            variant=variant)
    events = list(brain.events or [])[events_before:]
    model_revised = [e for e in events if e.get("kind") == "model_revised"]
    world_changed = [e for e in events if e.get("kind") == "world_changed"]
    event_injected = any(e.startswith("reset:") for e in env.log)
    legs = {
        "event_injected": event_injected,
        "goal_completed_after_event": bool(rec["completed"]),
        "model_revised_event_exists": len(model_revised) > 0,
        "world_changed_event_exists": len(world_changed) > 0,
    }
    passed = all(legs.values())
    return _result(passed, rec["cycles"],
                   (f"unexpected_event: goal {rec['goal_status']}, "
                    f"model_revised={len(model_revised)}, "
                    f"world_changed={len(world_changed)}"
                    if passed else
                    f"unexpected_event FAILED: goal {rec['goal_status']}, "
                    f"stop {rec['stop_reason']}, event={event_injected}, "
                    f"model_revised={len(model_revised)}, "
                    f"world_changed={len(world_changed)}"),
                   {"goal_status": rec["goal_status"],
                    "stop_reason": rec["stop_reason"],
                    "event_injected": event_injected,
                    "model_revised_events": len(model_revised),
                    "world_changed_events": len(world_changed),
                    "failures": rec["failures"], "env_log": rec["env_log"]},
                   failure_cause=(None if passed else
                                  ("goal not completed after the event"
                                   if not rec["completed"] else
                                   "no model_revised event recorded after "
                                   "the injected event")),
                   legs=legs, difficulty=UNEXPECTED_DIFFICULTY, tier=UNEXPECTED_TIER)


# ------------------------------------------------------------------ (5) planning_under_uncertainty
UNCERTAINTY_DIFFICULTY = {"n_rooms": 3, "n_decoys": 0, "resource_units": None,
                          "n_dynamic_events": 0, "observability": "full",
                          "n_goals": 1, "horizon_min_steps": 5}
UNCERTAIN_P = 0.6


class CrossTrainingWorld(Environment):
    """Training world for the uncertain step: one room pair, the
    only route is ``cross_rift``. The outcome for a whole run is
    drawn once (``random.Random(seed)`` in eval code only): the
    brain never sees the seed or the draw."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.succeeds = rng.random() < UNCERTAIN_P
        self.s = {"at": "quendal", "at_brelm": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        if name == "cross_rift" and s["at"] == "quendal":
            if self.succeeds:
                s["at"], s["at_brelm"] = "brelm", True
                info = "crossed the rift"
                self.log.append(f"{name}: {info}")
                return True, info
            info = "the rift crossing fails; you stay put"
            self.log.append(f"{name}: {info}")
            return False, info
        self.log.append(f"{name}: nothing happened")
        return False, "nothing happened"


class UncertainWorld(Environment):
    """Benchmark world: two routes to the treasure in ``sorvik``.
    The risky route is ``cross_rift`` straight from ``quendal``
    (base cost 2.0, succeeds with seeded probability 0.6 per
    attempt, a fresh ``random.Random(seed)`` draw per attempt).
    The safe route goes via ``brelm`` (key + unlock, total cost
    2.3). A brain with prior cross experience (failure rate from
    its own learner episodes) prices the risky route above the
    safe one and must choose the safe route; a brain without that
    experience prices it below and gambles."""

    ROOMS = ("quendal", "brelm", "sorvik")

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.s = {"at": "quendal", "has_vaskel": False,
                  "door_open": False, "has_durnik": False}
        self.log: list[str] = []
        self.cross_attempts = 0

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "cross_rift" and s["at"] == "quendal":
            self.cross_attempts += 1
            if self.rng.random() < UNCERTAIN_P:
                s["at"], ok, info = "sorvik", True, "crossed the rift"
            else:
                info = "the rift crossing fails; you stay put"
        elif name == "go":
            dest = args.get("room")
            adj = {"quendal": ["brelm"], "brelm": ["quendal", "sorvik"],
                   "sorvik": ["brelm"]}
            if dest in adj[s["at"]] and (
                    dest != "sorvik" or s["door_open"] or s["at"] == "quendal"):
                # The safe route enters sorvik through the door; the
                # cross route is the only door-less entry, handled
                # above. A go into sorvik needs the door open.
                if dest == "sorvik" and not s["door_open"]:
                    info = "cannot reach sorvik"
                else:
                    s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_vaskel" and s["at"] == "brelm" and not s["has_vaskel"]:
            s["has_vaskel"], ok, info = True, True, "took the vaskel"
        elif name == "unlock" and s["at"] == "brelm" and s["has_vaskel"]:
            s["door_open"], ok, info = True, True, "unlocked the door"
        elif name == "take_durnik" and s["at"] == "sorvik":
            s["has_durnik"], ok, info = True, True, "took the durnik"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _uncertain_actions() -> list[Action]:
    return [
        Action("cross_rift", {"at": "quendal"}, {"at": "sorvik"},
               cost=1.0, args={"_do": "cross_rift"}),
        Action("go_brelm_from_quendal", {"at": "quendal"}, {"at": "brelm"},
               cost=0.4, args={"_do": "go", "room": "brelm"}),
        Action("go_quendal_from_brelm", {"at": "brelm"}, {"at": "quendal"},
               cost=0.4, args={"_do": "go", "room": "quendal"}),
        Action("go_sorvik_from_brelm", {"at": "brelm", "door_open": True},
               {"at": "sorvik"}, cost=0.4,
               args={"_do": "go", "room": "sorvik"}),
        Action("go_brelm_from_sorvik", {"at": "sorvik"}, {"at": "brelm"},
               cost=0.4, args={"_do": "go", "room": "brelm"}),
        Action("take_vaskel", {"at": "brelm", "has_vaskel": False},
               {"has_vaskel": True}, cost=0.5, args={"_do": "take_vaskel"}),
        Action("unlock", {"at": "brelm", "has_vaskel": True},
               {"door_open": True}, cost=0.5, args={"_do": "unlock"}),
        Action("take_durnik", {"at": "sorvik"}, {"has_durnik": True},
               cost=0.5, args={"_do": "take_durnik"}),
    ]


def _expected_success(brain, plan_steps) -> float:
    """The brain's own expected success for a plan at selection
    time: the product over steps of (1 - planner failure rate),
    read from the planner's episode-derived stats. A step never
    tried has failure rate 0 (no evidence against it)."""
    p = 1.0
    for step in plan_steps:
        p *= (1.0 - brain.planner.failure_rate(step.name))
    return round(p, 4)


def _true_expected_success(plan_steps) -> float:
    """Eval-side ground-truth expected success for a plan in the
    uncertainty world: the uncertain step succeeds with the
    seeded probability ``UNCERTAIN_P`` (the brain never sees the
    seed or this number); every other step in this world is
    deterministic. This is the quantity the design's 'higher
    expected-success plan' leg is scored against -- the brain's
    own rate-based figure (``_expected_success``) is 1.0 for
    every plan while it has no evidence, so it cannot separate a
    gamble from a safe plan."""
    p = 1.0
    for step in plan_steps:
        if step.name == "cross_rift":
            p *= UNCERTAIN_P
    return round(p, 4)


def run_planning_under_uncertainty(factory, seed: int = 0,
                                   variant: str = "full") -> dict:
    """Task (5): one plan step (``cross_rift``) succeeds only with
    seeded probability 0.6. The ``full``/``non_integrated`` brains
    first run five training legs in a training world where the
    cross is the only route, so their planner prices the risk from
    their own episode history; the ``fresh``/``no_learning`` brains
    get no (usable) training. Pass: the goal completes, the first
    plan chosen is the higher expected-success one at selection
    time (measured from the brain's own failure rates), and -- if
    the uncertain step was attempted and failed -- the fallback
    (safe route) still completes the goal. A brain that gambles on
    the cross and recovers still fails the first leg; that is the
    variant separation this task is designed to show."""
    actions = _uncertain_actions()
    trained = variant not in ("fresh",)
    training_cross = Action("cross_rift", {"at": "quendal"},
                            {"at": "brelm"}, cost=2.0,
                            args={"_do": "cross_rift"})
    brain = factory([training_cross], {"compute", "environment"})
    training: dict = {"runs": 0, "cross_successes": 0, "cross_failures": 0,
                      "trained": trained}
    if trained:
        rng = random.Random(seed)
        for _ in range(5):
            env_t = CrossTrainingWorld(rng)
            if training["runs"] > 0:
                prepare_brain_for_new_world(brain)
            ep0 = _episodes_before(brain)
            brain.add_goal("reach brelm", {"at": "brelm"})
            brain.run(env_t, brain.cycle + 12)
            training["runs"] += 1
            for ep in list(brain.learner.episodes or [])[ep0:]:
                if ep.action == "cross_rift":
                    if ep.ok:
                        training["cross_successes"] += 1
                    else:
                        training["cross_failures"] += 1
            # The training goal may have failed; the benchmark goal
            # is added fresh below, so a failed training goal must
            # not stay active.
            active = brain.goals.active()
            if active is not None and active.description == "reach brelm":
                brain.goals.fail(active.id, "training leg over")
    # The training cross (effect: at=brelm) leaves the planner
    # before the benchmark: the benchmark world's cross has a
    # different effect (at=sorvik) and the env cannot tell variants
    # apart, so keeping the training variant would let the brain
    # execute a believed-to-brelm cross that lands in sorvik.
    brain.planner.actions = [a for a in brain.planner.actions
                             if a.name != "cross_rift"]
    for a in actions:
        brain.planner.add_action(a)
    prepare_brain_for_new_world(brain)
    env = UncertainWorld(seed)
    ep0 = _episodes_before(brain)
    cycle_before = brain.cycle
    goal = brain.add_goal("take the durnik", {"has_durnik": True})
    # Capture the first plan the brain adopts, before it executes.
    first_plan: list[str] = []
    first_plan_expected: float | None = None
    first_plan_true_expected: float | None = None
    alt_expected: float | None = None
    alt_true_expected: float | None = None
    cross_rate_at_selection: float | None = None
    for _ in range(12):
        brain.step(env)
        if brain.plan:
            cross_rate_at_selection = brain.planner.failure_rate("cross_rift")
            first_plan = [s.name for s in brain.plan]
            first_plan_expected = _expected_success(brain, list(brain.plan))
            first_plan_true_expected = _true_expected_success(list(brain.plan))
            plans = brain.planner.plan_all(
                brain.world, goal.target, brain._banned(), limit=2)
            alt = next((p for p in plans
                        if [s.name for s in p.steps] != first_plan), None)
            alt_expected = (_expected_success(brain, list(alt.steps))
                            if alt is not None else None)
            alt_true_expected = (_true_expected_success(list(alt.steps))
                                 if alt is not None else None)
            break
        if goal.status != "active":
            break
    stop = brain.run(env, brain.cycle + MAX_CYCLES)
    cycles = brain.cycle - cycle_before
    cross_attempts = env.cross_attempts
    cross_failed = sum(1 for e in env.log
                       if e.startswith("cross_rift") and "fails" in e)
    safe_used = any(e.startswith("take_vaskel") for e in env.log)
    # Scored on the true expected success: the safe plan is 1.0,
    # the cross plan is 0.6. The brain's own rate-based figure is
    # recorded in the evidence for comparison.
    chose_higher = bool(
        first_plan and first_plan_true_expected is not None
        and (alt_true_expected is None
             or first_plan_true_expected >= alt_true_expected))
    fallback_ok = (cross_failed == 0 or (safe_used and goal.status == "done"))
    legs = {
        "goal_completed": goal.status == "done",
        "chose_higher_expected_success_plan": chose_higher,
        "fallback_completes_after_uncertain_failure": fallback_ok,
    }
    passed = all(legs.values())
    failures = count_failures(env.log, brain, ep0,
                              episodes_available=(variant != "no_learning"))
    return _result(passed, cycles,
                   (f"uncertainty: goal {goal.status}, first plan "
                    f"{first_plan} (true expected "
                    f"{first_plan_true_expected} vs alternative "
                    f"{alt_true_expected}), cross attempts "
                    f"{cross_attempts} ({cross_failed} failed)"
                    if passed else
                    f"uncertainty FAILED: goal {goal.status}, stop {stop}, "
                    f"first plan {first_plan} (true expected "
                    f"{first_plan_true_expected} vs alternative "
                    f"{alt_true_expected}), cross attempts "
                    f"{cross_attempts}"),
                   {"goal_status": goal.status, "stop_reason": stop,
                    "first_plan": first_plan,
                    "first_plan_expected_success": first_plan_expected,
                    "first_plan_true_expected_success": first_plan_true_expected,
                    "alternative_expected_success": alt_expected,
                    "alternative_true_expected_success": alt_true_expected,
                    "uncertain_step_true_p": UNCERTAIN_P,
                    "cross_failure_rate_at_selection": cross_rate_at_selection,
                    "cross_attempts": cross_attempts,
                    "cross_failures": cross_failed,
                    "safe_route_used": safe_used,
                    "training": training,
                    "failures": failures, "env_log": list(env.log)},
                   failure_cause=(None if passed else
                                  ("goal not completed"
                                   if goal.status != "done" else
                                   "first plan was not the higher "
                                   "expected-success plan at selection time")),
                   legs=legs, difficulty=UNCERTAINTY_DIFFICULTY, tier=UNCERTAINTY_TIER)


# ------------------------------------------------------------------ (6) transfer_complex
TRANSFER_DIFFICULTY = {"n_rooms": 4, "n_decoys": 1, "resource_units": None,
                       "n_dynamic_events": 0, "observability": "full",
                       "n_goals": 1, "horizon_min_steps": 10}


def _action_identity(action) -> tuple:
    """Full action identity (name + pre + eff + args), canonicalised
    by content — mirrors ``ab.eval.consolidated._action_identity``.
    Two actions are the same action for planner purposes iff their
    identities match (cost / tool / store_as do not distinguish
    them). Used to dedupe the carried brain's action list (see
    ``_run_generated_leg``): the carried brain already holds world
    A's actions, so adding world B's actions blindly duplicates the
    actions identical in both worlds."""
    def _canon(value) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    return (action.name, _canon(action.pre), _canon(action.eff),
            _canon(action.args))


def _run_generated_leg(factory, seed: int, family: str, brain=None,
                       variant: str = "full"):
    """One leg of generated transfer: build the hard env for
    (seed, family), run it, return (record, brain)."""
    env, actions, goal, shape = _generate_hard_env(
        seed, family, difficulty="hard")
    if brain is None:
        brain = factory(actions, {"compute", "environment"})
        ep0 = _episodes_before(brain)
        cycle_before = brain.cycle
        brain.add_goal("generated transfer goal", dict(goal))
        stop = brain.run(env, MAX_CYCLES)
    else:
        # Phase 10.5 (P10.5-9): the generation must advance BEFORE
        # the new world's actions are added, because ``add_action``
        # tags an action with the CURRENT generation. The
        # consolidated suite (``_run_generic_env``) already does
        # prepare-first; this runner did add-first, so B's actions
        # were tagged with A's generation and the current-world
        # tie-break never fired here. Reordering to prepare-first
        # changes nothing about the worlds, the brain, or the
        # measurement — only the generation tags.
        prepare_brain_for_new_world(brain)
        # Coordinator integration (P10-56 follow-up): same dedupe as
        # _run_brain — the carried brain already holds world A's
        # actions; adding B's actions blindly duplicates the 1 action
        # identical in both worlds on the hard workshop pair.
        existing = {_action_identity(a) for a in brain.planner.actions}
        for a in actions:
            ident = _action_identity(a)
            if ident not in existing:
                brain.planner.add_action(a)
                existing.add(ident)
        ep0 = _episodes_before(brain)
        cycle_before = brain.cycle
        brain.add_goal("generated transfer goal", dict(goal))
        stop = brain.run(env, brain.cycle + MAX_CYCLES)
    goal_obj = brain.goals.all()[-1]
    failures = count_failures(env.log, brain, ep0,
                              episodes_available=(variant != "no_learning"))
    return {"completed": goal_obj.status == "done",
            "goal_status": goal_obj.status, "stop_reason": stop,
            "cycles": brain.cycle - cycle_before,
            "failed_attempts": failures["authoritative"],
            "failures": failures, "shape": shape,
            "env_log": list(env.log)}, brain


def run_transfer_complex(factory, seed: int = 0,
                         variant: str = "full") -> dict:
    """Task (6): learn in a hard Tier-2 workshop world A (seed),
    apply in a genuinely different hard Tier-2 workshop world B
    (seed+1: different names/layout). Continued vs fresh measured
    on episode-authoritative failed attempts and cycles. For the
    ``fresh`` variant only the fresh B leg runs (the continued leg
    lives in the ``full`` entry); pass there is simply the fresh
    leg completing. Pass for the comparison variants: continued
    completes AND fresh completes AND continued needs strictly
    fewer failed attempts than fresh (or fewer cycles on a tie).
    A negative result is recorded as such -- the generator is
    never tuned to flatter the brain."""
    family = "workshop"
    if variant == "fresh":
        rec_fresh, _b = _run_generated_leg(factory, seed + 1, family,
                                           variant=variant)
        legs = {"fresh_leg_completed": bool(rec_fresh["completed"])}
        return _result(bool(rec_fresh["completed"]), rec_fresh["cycles"],
                       f"transfer_complex/fresh: fresh B "
                       f"{rec_fresh['goal_status']} in "
                       f"{rec_fresh['cycles']} cycles, "
                       f"{rec_fresh['failed_attempts']} failed attempts "
                       f"(continued leg is in the full entry)",
                       {"world_b_fresh": rec_fresh,
                        "note": "fresh variant: fresh B leg only"},
                       failure_cause=(None if rec_fresh["completed"] else
                                      "fresh B leg did not complete"),
                       legs=legs, difficulty=TRANSFER_DIFFICULTY, tier=TRANSFER_TIER)
    rec_a, carried = _run_generated_leg(factory, seed, family,
                                        variant=variant)
    rec_cont, _b = _run_generated_leg(factory, seed + 1, family,
                                      brain=carried, variant=variant)
    rec_fresh, _b2 = _run_generated_leg(factory, seed + 1, family,
                                        variant=variant)
    fa, fb = rec_cont["failed_attempts"], rec_fresh["failed_attempts"]
    ca, cb = rec_cont["cycles"], rec_fresh["cycles"]
    transferred = bool(rec_cont["completed"] and rec_fresh["completed"]
                       and (fa < fb or (fa == fb and ca < cb)))
    legs = {
        "world_a_completed": bool(rec_a["completed"]),
        "continued_b_completed": bool(rec_cont["completed"]),
        "fresh_b_completed": bool(rec_fresh["completed"]),
        "continued_fewer_failed_attempts": fa < fb,
        "continued_fewer_cycles_on_tie": bool(fa == fb and ca < cb),
    }
    passed = transferred
    return _result(passed, rec_cont["cycles"],
                   (f"transfer_complex: continued B {fa} failed / "
                    f"{ca} cycles vs fresh B {fb} failed / {cb} cycles; "
                    f"A completed={rec_a['completed']}"
                    if passed else
                    f"transfer_complex FAILED: continued B {fa} failed / "
                    f"{ca} cycles vs fresh B {fb} failed / {cb} cycles; "
                    f"A completed={rec_a['completed']}"),
                   {"world_a": rec_a, "world_b_continued": rec_cont,
                    "world_b_fresh": rec_fresh,
                    "seeds": [seed, seed + 1], "family": family,
                    "criterion": ("continued completes and needs strictly "
                                  "fewer failed attempts than fresh (or "
                                  "fewer cycles on a tie)")},
                   failure_cause=(None if passed else
                                  "continued brain did not beat the fresh "
                                  "brain on the hard generated workshop B"),
                   legs=legs, difficulty=TRANSFER_DIFFICULTY, tier=TRANSFER_TIER)


# ------------------------------------------------------------------ (7) long_task_interrupt_restart
LONG_DIFFICULTY = {"n_rooms": 10, "n_decoys": 0, "resource_units": None,
                   "n_dynamic_events": 1, "observability": "full",
                   "n_goals": 1, "horizon_min_steps": 22}
LONG_ADJACENCY = {
    "nolvar": ["brellik"],
    "brellik": ["nolvar", "sorvane"],
    "sorvane": ["brellik", "tarnik", "vessal"],
    "tarnik": ["sorvane"],
    "vessal": ["sorvane", "mordune", "ullik"],
    "mordune": ["vessal"],
    "ullik": ["vessal", "zavrik", "kendrell"],
    "zavrik": ["ullik"],
    "kendrell": ["ullik", "zavrel"],
    "zavrel": ["kendrell"],
}
LONG_ITEMS = (("dralik", "tarnik", "kindled"),
              ("senvor", "mordune", "quenched"),
              ("quilm", "zavrik", "focused"))
LONG_STAGES = ("take_dralik", "take_senvor", "take_quilm",
               "use_dralik", "use_senvor", "use_quilm")


class LongTier3World(Environment):
    """Tier-3 long task: ten rooms, three cached items, one forge.
    Each item's cache must be opened before the item can be taken;
    items work at the forge only in sequence (dralik, then senvor,
    then quilm); ``assemble`` works only at the forge with all
    three flags set. Same structure as the Phase 9 long-horizon
    world, different names/rooms."""

    def __init__(self):
        self.s = {"at": "nolvar", "has_keepsake": False}
        for item, _room, flag in LONG_ITEMS:
            self.s[f"has_{item}"] = False
            self.s[f"cache_{item}"] = False
            self.s[flag] = False
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in LONG_ADJACENCY[s["at"]]:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name.startswith("open_cache_"):
            item = name[len("open_cache_"):]
            room = next(r for i, r, _f in LONG_ITEMS if i == item)
            if s["at"] == room and not s[f"cache_{item}"]:
                s[f"cache_{item}"], ok, info = True, True, f"opened the cache of {item}"
        elif name.startswith("take_"):
            item = name[len("take_"):]
            room = next(r for i, r, _f in LONG_ITEMS if i == item)
            if s["at"] == room and s[f"cache_{item}"] and not s[f"has_{item}"]:
                s[f"has_{item}"], ok, info = True, True, f"took the {item}"
        elif name.startswith("use_"):
            item = name[len("use_"):]
            flag = next(f for i, _r, f in LONG_ITEMS if i == item)
            idx = [i for i, _r, _f in LONG_ITEMS].index(item)
            prev_ok = True if idx == 0 else s[LONG_ITEMS[idx - 1][2]]
            if s["at"] == "zavrel" and s[f"has_{item}"] and prev_ok and not s[flag]:
                s[flag], ok, info = True, True, f"used the {item}"
            else:
                info = f"cannot use {item} here"
        elif name == "assemble" and s["at"] == "zavrel" and all(s[f] for _i, _r, f in LONG_ITEMS):
            s["has_keepsake"], ok, info = True, True, "assembled the keepsake"
        elif name == "assemble":
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _long_actions() -> list[Action]:
    actions: list[Action] = []
    for room, dests in LONG_ADJACENCY.items():
        for dest in dests:
            actions.append(Action(f"go_{dest}_from_{room}", {"at": room},
                                  {"at": dest}, cost=0.4,
                                  args={"_do": "go", "room": dest}))
    for item, room, _flag in LONG_ITEMS:
        actions.append(Action(
            f"open_cache_{item}", {"at": room, f"cache_{item}": False},
            {f"cache_{item}": True}, cost=1.0,
            args={"_do": f"open_cache_{item}"}))
        actions.append(Action(
            f"take_{item}",
            {"at": room, f"cache_{item}": True, f"has_{item}": False},
            {f"has_{item}": True}, cost=1.0,
            args={"_do": f"take_{item}"}))
    for idx, (item, _room, flag) in enumerate(LONG_ITEMS):
        pre = {"at": "zavrel", f"has_{item}": True, flag: False}
        if idx > 0:
            pre[LONG_ITEMS[idx - 1][2]] = True
        actions.append(Action(f"use_{item}", pre, {flag: True}, cost=1.0,
                              args={"_do": f"use_{item}"}))
    actions.append(Action(
        "assemble", {"at": "zavrel", **{f: True for _i, _r, f in LONG_ITEMS}},
        {"has_keepsake": True}, cost=1.0, args={"_do": "assemble"}))
    return actions


def _stage_positions(env_log) -> dict:
    return {stage: [i for i, e in enumerate(env_log)
                    if e.startswith(stage)]
            for stage in LONG_STAGES}


def run_long_task_interrupt_restart(factory, seed: int = 0,
                                    variant: str = "full") -> dict:
    """Task (7): a >=6-stage Tier-3 task, interrupted mid-way (after
    the second item is taken), saved, loaded into a fresh Brain,
    and resumed to completion. Pass: the goal completes, every
    stage ran exactly once and in order across the two legs (read
    off the shared env log, never off the brain's plan), and the
    load was clean (no load warnings)."""
    env = LongTier3World()
    actions = _long_actions()
    brain = factory(actions, {"compute", "environment"})
    goal = brain.add_goal("forge the keepsake", {"has_keepsake": True})
    ep0 = _episodes_before(brain)
    # Leg 1: run until the second item is taken, then save.
    _step_until(brain, env,
                lambda: env.s["has_senvor"] or goal.status in ("done", "failed"), 60)
    if not env.s["has_senvor"]:
        return _result(False, brain.cycle,
                       f"long_task: interrupt point not reached "
                       f"(has_senvor={env.s['has_senvor']}, goal "
                       f"{goal.status})",
                       {"env_log": list(env.log)},
                       failure_cause="could not reach the mid-task "
                                     "interrupt point",
                       legs={"interrupt_point_reached": False},
                       difficulty=LONG_DIFFICULTY, tier=LONG_TIER)
    cycles_before_save = brain.cycle
    folder = tempfile.mkdtemp(prefix="ab-benchmark-long-")
    brain.save(folder)
    # Leg 2: fresh brain, load, complete.
    resumed = Brain(actions, granted={"compute", "environment"})
    resumed.load(folder)
    load_warnings = list(getattr(resumed, "load_warnings", []))
    goal2 = resumed.goals.all()[-1]
    stop = resumed.run(env, resumed.cycle + 120)
    cycles = resumed.cycle
    positions = _stage_positions(env.log)
    counts = {s: len(v) for s, v in positions.items()}
    firsts = {s: (v[0] if v else None) for s, v in positions.items()}
    in_order = (all(v is not None for v in firsts.values())
                and [firsts[s] for s in LONG_STAGES] == sorted(firsts.values()))
    legs = {
        "interrupt_point_reached": True,
        "goal_completed": goal2.status == "done",
        "stages_in_order": bool(in_order),
        "no_stage_repeated_or_skipped": all(c == 1 for c in counts.values()),
        "load_clean": load_warnings == [],
    }
    passed = all(legs.values())
    failures = count_failures(env.log, resumed, ep0,
                              episodes_available=(variant != "no_learning"))
    return _result(passed, cycles,
                   (f"long_task: goal {goal2.status}, stages in order, "
                    f"each once, save at cycle {cycles_before_save}, "
                    f"resumed to cycle {cycles}"
                    if passed else
                    f"long_task FAILED: goal {goal2.status}, stop {stop}, "
                    f"stage counts {counts}, in order={in_order}, "
                    f"load warnings {load_warnings}"),
                   {"cycles_before_save": cycles_before_save,
                    "goal_status": goal2.status, "stop_reason": stop,
                    "stage_positions": {s: firsts[s] for s in LONG_STAGES},
                    "stage_counts": counts,
                    "stages_in_order": bool(in_order),
                    "load_warnings": load_warnings,
                    "failures": failures, "env_log": list(env.log)},
                   failure_cause=(None if passed else
                                  "a stage was repeated/skipped/out of "
                                  "order, or the goal did not complete, or "
                                  "the load was not clean"),
                   legs=legs, difficulty=LONG_DIFFICULTY, tier=LONG_TIER)


# ------------------------------------------------------------------ (8) unrecoverable_failure
UNRECOVERABLE_DIFFICULTY = {"n_rooms": 3, "n_decoys": 0,
                            "resource_units": None, "n_dynamic_events": 1,
                            "observability": "full", "n_goals": 1,
                            "horizon_min_steps": 5}


class DestroyKeyWorld(Environment):
    """3-room chain, one key (``vaskel``), one door, one treasure.
    After the key is taken, the SECOND observation destroys it:
    the key shatters into dust (an env event, recorded in the log).
    There is no other key and no other way through the door, so the
    task becomes genuinely unrecoverable."""

    ROOMS = ("quendal", "brelm", "sorvik")

    def __init__(self):
        self.s = {"at": self.ROOMS[0], "has_vaskel": False,
                  "door_open": False, "has_durnik": False}
        self._destroyed = False
        self._observes_since_take = 0
        self.log: list[str] = []

    def observe(self) -> dict:
        if self.s["has_vaskel"] and not self._destroyed:
            self._observes_since_take += 1
            if self._observes_since_take >= 2:
                self._destroyed = True
                self.s["has_vaskel"] = False
                self.log.append("event: the vaskel key shatters into dust")
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            adj = {self.ROOMS[0]: [self.ROOMS[1]],
                   self.ROOMS[1]: [self.ROOMS[0], self.ROOMS[2]],
                   self.ROOMS[2]: [self.ROOMS[1]]}
            if dest in adj[s["at"]] and (
                    dest != self.ROOMS[-1] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_vaskel" and s["at"] == self.ROOMS[1] \
                and not s["has_vaskel"] and not self._destroyed:
            s["has_vaskel"], ok, info = True, True, "took the vaskel"
        elif name == "unlock" and s["at"] == self.ROOMS[1] and s["has_vaskel"]:
            s["door_open"], ok, info = True, True, "unlocked the door"
        elif name == "take_durnik" and s["at"] == self.ROOMS[-1]:
            s["has_durnik"], ok, info = True, True, "took the durnik"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _destroy_key_actions() -> list[Action]:
    r = DestroyKeyWorld.ROOMS
    return _go_actions(list(r), gate_last=True) + [
        Action("take_vaskel", {"at": r[1], "has_vaskel": False},
               {"has_vaskel": True}, cost=0.5, args={"_do": "take_vaskel"}),
        Action("unlock", {"at": r[1], "has_vaskel": True},
               {"door_open": True}, cost=0.5, args={"_do": "unlock"}),
        Action("take_durnik", {"at": r[-1]}, {"has_durnik": True},
               cost=0.5, args={"_do": "take_durnik"}),
    ]


def run_unrecoverable_failure(factory, seed: int = 0,
                              variant: str = "full") -> dict:
    """Task (8): the only key is destroyed by an env event mid-task.
    Pass: the brain stops with reason ``impossible`` (or an
    equivalent recorded stop reason -- ``stalled`` is accepted and
    the exact reason is recorded), within the 80-cycle budget,
    partial results are preserved (the key WAS taken and the middle
    room reached before the destruction), and no attempt is made
    at the destroyed step (``unlock``/``take_vaskel``) after the
    event."""
    env = DestroyKeyWorld()
    rec, brain = _run_brain(factory, _destroy_key_actions(),
                            "take the durnik", {"has_durnik": True}, env,
                            variant=variant)
    destroy_idx = next((i for i, e in enumerate(env.log)
                        if e.startswith("event: the vaskel key shatters")), None)
    after = env.log[destroy_idx + 1:] if destroy_idx is not None else []
    unlock_after = sum(1 for e in after if e.startswith("unlock"))
    retake_after = sum(1 for e in after if e.startswith("take_vaskel"))
    took_before = any(e.startswith("take_vaskel") and "took the vaskel" in e
                      for e in env.log[:destroy_idx] if destroy_idx is not None)
    stop_ok = rec["stop_reason"] in ("impossible", "stalled")
    legs = {
        "key_destroyed_by_event": destroy_idx is not None,
        "stopped_with_impossible_or_equivalent": stop_ok,
        "stopped_within_budget": rec["cycles"] <= MAX_CYCLES,
        "partial_results_preserved": bool(took_before),
        "no_attempt_at_destroyed_step_after_event": unlock_after == 0,
        "key_recovery_attempts_bounded": retake_after <= 3,
    }
    passed = all(legs.values())
    return _result(passed, rec["cycles"],
                   (f"unrecoverable: stop {rec['stop_reason']} after "
                    f"{rec['cycles']} cycles, key destroyed at log "
                    f"index {destroy_idx}, attempts after event "
                    f"{unlock_after + retake_after}"
                    if passed else
                    f"unrecoverable FAILED: stop {rec['stop_reason']}, "
                    f"cycles {rec['cycles']}, destroy index "
                    f"{destroy_idx}, unlock-after {unlock_after}, "
                    f"retake-after {retake_after} (bounded recovery), "
                    f"partial {took_before}"),
                   {"goal_status": rec["goal_status"],
                    "stop_reason": rec["stop_reason"],
                    "destroy_log_index": destroy_idx,
                    "unlock_attempts_after_event": unlock_after,
                    "retake_attempts_after_event": retake_after,
                    "key_taken_before_destruction": bool(took_before),
                    "failures": rec["failures"], "env_log": rec["env_log"]},
                   failure_cause=(None if passed else
                                  f"stop reason {rec['stop_reason']} is not "
                                  f"impossible/stalled, or the brain kept "
                                  f"attempting the destroyed step"),
                   legs=legs, difficulty=UNRECOVERABLE_DIFFICULTY, tier=UNRECOVERABLE_TIER)


# ------------------------------------------------------------------ (9) competing_goals
COMPETING_DIFFICULTY = {"n_rooms": 2, "n_decoys": 0, "resource_units": 1,
                        "n_dynamic_events": 0, "observability": "full",
                        "n_goals": 2, "horizon_min_steps": 4}


class CompetingWorld(Environment):
    """Two rooms, ONE press charge. Two products each need a press
    use: ``zarnik`` (higher priority) and ``phelik`` (lower). The
    charge cannot serve both, so one goal must win and the other
    must be deferred, not silently dropped."""

    def __init__(self):
        self.s = {"at": "mordak", "charges": 1,
                  "pressed_zarnik": False, "pressed_phelik": False,
                  "has_zarnik": False, "has_phelik": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in ("mordak", "ullin") and dest != s["at"]:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name in ("press_zarnik", "press_phelik") and s["at"] == "ullin":
            flag = f"pressed_{name.split('_', 1)[1]}"
            if s["charges"] >= 1 and not s[flag]:
                s["charges"] -= 1
                s[flag], ok, info = True, True, f"pressed the {flag[8:]}"
            elif s["charges"] < 1:
                info = "no charges left"
            else:
                info = "already pressed"
        elif name == "assemble_zarnik" and s["at"] == "ullin" and s["pressed_zarnik"]:
            s["has_zarnik"], ok, info = True, True, "assembled the zarnik"
        elif name == "assemble_phelik" and s["at"] == "ullin" and s["pressed_phelik"]:
            s["has_phelik"], ok, info = True, True, "assembled the phelik"
        elif name.startswith("assemble"):
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def _competing_actions() -> list[Action]:
    return [
        Action("go_ullin_from_mordak", {"at": "mordak"}, {"at": "ullin"},
               cost=0.4, args={"_do": "go", "room": "ullin"}),
        Action("go_mordak_from_ullin", {"at": "ullin"}, {"at": "mordak"},
               cost=0.4, args={"_do": "go", "room": "mordak"}),
        Action("press_zarnik", {"at": "ullin", "pressed_zarnik": False},
               {"pressed_zarnik": True}, cost=1.0,
               args={"_do": "press_zarnik"}),
        Action("press_phelik", {"at": "ullin", "pressed_phelik": False},
               {"pressed_phelik": True}, cost=1.0,
               args={"_do": "press_phelik"}),
        Action("assemble_zarnik", {"at": "ullin", "pressed_zarnik": True},
               {"has_zarnik": True}, cost=1.0,
               args={"_do": "assemble_zarnik"}),
        Action("assemble_phelik", {"at": "ullin", "pressed_phelik": True},
               {"has_phelik": True}, cost=1.0,
               args={"_do": "assemble_phelik"}),
    ]


def run_competing_goals(factory, seed: int = 0,
                        variant: str = "full") -> dict:
    """Task (9): two goals, one press charge. The brain is given
    both goals up front (zarnik priority 0.9, phelik 0.4) and must
    pick the higher-priority one; the runner snapshots the
    brain's own ``last_decision`` at selection time (its reason
    carries the priority trade-off). Once the higher goal is done
    the runner defers the lower goal through the public suspension
    record (``GoalManager.suspend``) with the trade-off as the
    reason -- the brain's own goal machinery has no autonomous
    'defer for lack of resource' path, so the deferral is a
    harness act and is recorded as one, not attributed to the
    brain. Pass: higher goal done, decision record chose it with
    its priority in the reason, lower goal suspended (pending,
    with the trade-off reason), resource exhausted, lower goal
    neither done nor failed."""
    env = CompetingWorld()
    brain = factory(_competing_actions(), {"compute", "environment"})
    ep0 = _episodes_before(brain)
    high = brain.add_goal("assemble the zarnik", {"has_zarnik": True},
                          priority=0.9)
    low = brain.add_goal("assemble the phelik", {"has_phelik": True},
                         priority=0.4)
    decision_snapshot: dict | None = None
    for _ in range(60):
        brain.step(env)
        if decision_snapshot is None and brain.last_decision is not None:
            d = brain.last_decision
            decision_snapshot = {
                "chosen": d.chosen.name if d.chosen is not None else None,
                "ranking": list(d.ranking or []),
                "reason": d.reason,
                "chosen_goal_id": d.chosen_goal_id,
                "tie": d.tie}
        if high.status == "done":
            break
    cycles = brain.cycle
    tradeoff = ("resource: 1 press charge, 2 goals each needing 1; "
                "zarnik priority 0.9 beats phelik priority 0.4; the "
                "charge is spent on zarnik, phelik is deferred, not dropped")
    suspended = None
    if high.status == "done" and low.status == "pending":
        suspended = brain.goals.suspend(
            low, [], 0, list(brain.events), dict(brain.world),
            f"deferred: {tradeoff}", brain.cycle)
    failures = count_failures(env.log, brain, ep0,
                              episodes_available=(variant != "no_learning"))
    decided_high = bool(decision_snapshot
                        and decision_snapshot.get("chosen") is not None
                        and "zarnik" in str(decision_snapshot["chosen"]))
    reason_has_priority = bool(decision_snapshot
                               and "priority" in str(decision_snapshot.get("reason", "")))
    legs = {
        "higher_priority_goal_completed": high.status == "done",
        "decision_record_chose_higher_goal": decided_high,
        "decision_reason_states_priority_tradeoff": reason_has_priority,
        "lower_goal_suspended_not_dropped": bool(
            suspended is not None and low.status == "pending"
            and brain.goals.suspended(low.id) is not None),
        "resource_exhausted": env.s["charges"] == 0,
        "lower_goal_neither_done_nor_failed": low.status == "pending",
    }
    passed = all(legs.values())
    return _result(passed, cycles,
                   (f"competing_goals: zarnik {high.status}, phelik "
                    f"suspended, decision chose "
                    f"{decision_snapshot and decision_snapshot.get('chosen')}"
                    if passed else
                    f"competing_goals FAILED: zarnik {high.status}, "
                    f"phelik {low.status}, decision "
                    f"{decision_snapshot}, suspended={suspended is not None}"),
                   {"high_goal_status": high.status,
                    "low_goal_status": low.status,
                    "decision": decision_snapshot,
                    "tradeoff": tradeoff,
                    "suspended_reason": suspended.reason if suspended else None,
                    "charges_left": env.s["charges"],
                    "failures": failures, "env_log": list(env.log)},
                   failure_cause=(None if passed else
                                  "higher goal not completed, or the "
                                  "decision did not prioritise it, or the "
                                  "lower goal was not suspended"),
                   legs=legs, difficulty=COMPETING_DIFFICULTY, tier=COMPETING_TIER)


# ------------------------------------------------------------------ tiers (P10.5-6)
# World kind per benchmark task, keyed by the runner function that
# builds the task's world -- the kind is a property of the world,
# never of the task id string. The kind strings live only inside
# ``_kind_of`` (a function body), never in an UPPER_CASE
# assignment: the leakage audit harvests single lowercase words
# from UPPER_CASE assignments in this module, and a kind word
# ("composite", "generated", ...) appearing there would be
# harvested as an invented entity name and then flagged wherever
# it legitimately appears outside ab/eval/ (e.g. ab/demo.py).
_COMPOSITE_RUNNERS = frozenset({run_long_task_interrupt_restart,
                                run_unrecoverable_failure,
                                run_competing_goals})
_GENERATED_RUNNERS = frozenset({run_transfer_complex})


def _kind_of(runner) -> str:
    """World kind of a benchmark task, from the runner that builds
    its world (never from the task id string). The composite
    runners build worlds that combine several challenge mechanisms
    at once (a >=6-stage task with a save/load interrupt; a
    terminal event destroying the only key; two goals contending
    for one resource); the generated runner builds envgen worlds;
    every other runner isolates one mechanism."""
    if runner in _COMPOSITE_RUNNERS:
        return "composite"
    if runner in _GENERATED_RUNNERS:
        return "generated"
    return "single"


# Per-task tiers, derived once from each task's own difficulty
# vector and world kind (see ``_tier_for``). The runners pass
# these to ``_result``; ``run_benchmarks`` reads the same values
# through ``TASK_TIERS`` below, so the task record and the
# per-variant records can never disagree about a task's tier.
# Active-dimension counts (see ``_active_dimensions``):
# navigate 3, resources 3, changing 3, unexpected 2, uncertainty 1,
# transfer 3, long 3, unrecoverable 2, competing 3 -- the count
# alone does not determine the tier (see ``_tier_for``).
NAV_TIER = _tier_for(NAV_DIFFICULTY, _kind_of(run_navigate_unfamiliar))
RESOURCE_TIER = _tier_for(RESOURCE_DIFFICULTY,
                          _kind_of(run_find_and_use_resources))
CHANGE_TIER = _tier_for(CHANGE_DIFFICULTY,
                        _kind_of(run_changing_objective))
UNEXPECTED_TIER = _tier_for(UNEXPECTED_DIFFICULTY,
                            _kind_of(run_unexpected_event))
UNCERTAINTY_TIER = _tier_for(UNCERTAINTY_DIFFICULTY,
                             _kind_of(run_planning_under_uncertainty))
TRANSFER_TIER = _tier_for(TRANSFER_DIFFICULTY,
                          _kind_of(run_transfer_complex))
LONG_TIER = _tier_for(LONG_DIFFICULTY,
                      _kind_of(run_long_task_interrupt_restart))
UNRECOVERABLE_TIER = _tier_for(UNRECOVERABLE_DIFFICULTY,
                               _kind_of(run_unrecoverable_failure))
COMPETING_TIER = _tier_for(COMPETING_DIFFICULTY,
                           _kind_of(run_competing_goals))

# Runner -> tier, for ``run_benchmarks`` (the task record and the
# error-path variant records). A runner with no entry here has no
# determinable tier: ``_task_tier`` raises rather than defaulting.
TASK_TIERS = {
    run_navigate_unfamiliar: NAV_TIER,
    run_find_and_use_resources: RESOURCE_TIER,
    run_changing_objective: CHANGE_TIER,
    run_unexpected_event: UNEXPECTED_TIER,
    run_planning_under_uncertainty: UNCERTAINTY_TIER,
    run_transfer_complex: TRANSFER_TIER,
    run_long_task_interrupt_restart: LONG_TIER,
    run_unrecoverable_failure: UNRECOVERABLE_TIER,
    run_competing_goals: COMPETING_TIER,
}


def _task_tier(runner, task_id: str) -> int:
    """Tier for one registry task, looked up by its runner. Raises
    ``ValueError`` when the runner has no recorded tier or the
    recorded tier is not 1|2|3 -- a task with no determinable tier
    fails loudly, never silently defaults."""
    try:
        tier = TASK_TIERS[runner]
    except KeyError:
        raise ValueError(
            f"benchmark task {task_id!r} has no recorded tier: its "
            "runner is not in TASK_TIERS -- add the task's world "
            "kind (see _tier_for) instead of defaulting") from None
    if tier not in (1, 2, 3):
        raise ValueError(
            f"benchmark task {task_id!r} has an invalid recorded "
            f"tier {tier!r} (expected 1, 2 or 3)")
    return tier


# ------------------------------------------------------------------ registry + entry point
TASKS = (
    ("navigate_unfamiliar", run_navigate_unfamiliar,
     "5-room chain, partial map: pass = all rooms discovered in "
     "order, treasure taken, no plan step ever named an "
     "undiscovered room."),
    ("find_and_use_resources", run_find_and_use_resources,
     "2-charge consumable with a decoy use: pass = goal done, "
     "resource never exhausted before the final step, decoy wasted "
     "at most once."),
    ("changing_objective", run_changing_objective,
     "Goal changes mid-task: pass = new goal done, no stale plan "
     "step after the change, already-obtained key not re-taken."),
    ("unexpected_event", run_unexpected_event,
     "Env injects a mid-task gate reset: pass = goal done after "
     "the event, a model_revised event exists."),
    ("planning_under_uncertainty", run_planning_under_uncertainty,
     "One plan step succeeds with seeded probability 0.6: pass = "
     "goal done, first plan chosen is the higher expected-success "
     "one at selection time (scored on the true seeded "
     "probability: safe 1.0 vs cross 0.6), fallback completes the "
     "goal when the uncertain step fails."),
    ("transfer_complex", run_transfer_complex,
     "Learn in hard Tier-2 workshop A, apply in different hard "
     "Tier-2 workshop B: pass = continued completes with strictly "
     "fewer failed attempts than fresh (or fewer cycles on a tie)."),
    ("long_task_interrupt_restart", run_long_task_interrupt_restart,
     ">=6-stage Tier-3 task, interrupted mid-way, saved, loaded "
     "into a fresh Brain, resumed: pass = goal done, every stage "
     "exactly once and in order, clean load."),
    ("unrecoverable_failure", run_unrecoverable_failure,
     "The only key is destroyed mid-task: pass = stop reason "
     "impossible (or equivalent), within budget, partial results "
     "preserved, no attempt at the destroyed step after the event."),
    ("competing_goals", run_competing_goals,
     "Two goals, one press charge: pass = higher-priority goal "
     "done, decision record chose it with the priority trade-off, "
     "lower goal suspended (not silently dropped)."),
)


def run_benchmarks(seed: int = 0) -> dict:
    """Run all nine benchmark tasks for each variant and return::

        {"tasks": {name: {"passed": bool, "cycles": int,
                          "detail": str, "evidence": {...},
                          "failure_cause": str | None,
                          "variants": {variant: {"passed": bool,
                                                 "cycles": int,
                                                 "detail": str,
                                                 "evidence": {...},
                                                 "failure_cause": str | None,
                                                 "legs": {...}},
                                      ...}}}}

    ``passed`` / ``cycles`` / ``detail`` / ``evidence`` /
    ``failure_cause`` at task level are the FULL variant's result;
    per-variant numbers live under ``variants``. Every task record
    carries ``"tier"`` (1|2|3, design P10-3, derived per task by
    ``_tier_for`` from the task's difficulty vector and world kind
    -- see ``TASK_TIERS``), and every per-variant result carries
    the same ``"tier"`` (recorded by ``_result``, so the tier
    survives into the consolidated record's per-task evidence
    through the variants, exactly the path ``difficulty`` takes).
    A task with no determinable tier raises ``ValueError`` instead
    of running with a defaulted tier. ``fresh`` is the
    full factory with no task-specific prior training (for most
    tasks it duplicates ``full`` -- recorded as such in the
    evidence; for planning_under_uncertainty it skips the training
    leg, and for transfer_complex it runs only the fresh B leg).
    A brain failure in one task/variant is recorded, never raised.
    The seed is recorded in every task's evidence."""
    factories = make_variant_factories()
    factories["fresh"] = factories["full"]
    tasks: dict = {}
    for task_id, runner, criterion in TASKS:
        tier = _task_tier(runner, task_id)
        variants: dict = {}
        for vname in VARIANT_NAMES:
            factory = factories[vname]
            try:
                variants[vname] = runner(factory, seed=seed, variant=vname)
            except Exception as exc:  # a brain failure is data, not a crash
                variants[vname] = {
                    "passed": False, "cycles": None, "legs": {},
                    "tier": tier,
                    "evidence": {"error": f"{type(exc).__name__}: {exc}",
                                 "seed": seed},
                    "detail": f"task errored: {type(exc).__name__}: {exc}",
                    "failure_cause": f"harness error: {type(exc).__name__}: {exc}",
                }
        # The tier is a property of the task, not of the run: every
        # variant record must carry the same tier the registry
        # records. A mismatch is a harness bug -- fail loudly
        # instead of recording two tiers for one task.
        for vname, vresult in variants.items():
            v_tier = vresult.get("tier")
            if v_tier is None:
                raise ValueError(
                    f"benchmark task {task_id!r} variant {vname!r} "
                    "carries no tier -- the runner's _result call "
                    "must record the task's tier")
            if v_tier != tier:
                raise ValueError(
                    f"benchmark task {task_id!r} variant {vname!r} "
                    f"carries tier {v_tier!r}, but the registry "
                    f"records tier {tier!r} for the task")
        full = variants.get("full") or {}
        full_evidence = dict(full.get("evidence") or {})
        full_evidence.setdefault("seed", seed)
        full_evidence["criterion"] = criterion
        tasks[task_id] = {
            "passed": bool(full.get("passed")),
            "cycles": full.get("cycles"),
            "detail": full.get("detail", ""),
            "evidence": full_evidence,
            "failure_cause": full.get("failure_cause"),
            "tier": tier,
            "variants": variants,
        }
    return {"tasks": tasks}


def main(argv=None) -> int:
    """``python -m ab.eval.benchmarks [--seed N]``: run the nine
    tasks and print a one-line summary per task/variant. Exit 0
    when the run completed (a failed criterion is a recorded
    result, not a harness crash)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Phase 10 benchmarks")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    record = run_benchmarks(seed=args.seed)
    n_pass = 0
    for name, task in record["tasks"].items():
        per = ", ".join(
            f"{v}={'PASS' if r.get('passed') else 'FAIL'}"
            f"({r.get('cycles')}c)"
            for v, r in task["variants"].items())
        print(f"{name}: full={'PASS' if task['passed'] else 'FAIL'} | {per}")
        n_pass += 1 if task["passed"] else 0
    print(f"benchmarks: {n_pass}/{len(record['tasks'])} tasks pass on full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
