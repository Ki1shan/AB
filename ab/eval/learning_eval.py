"""Phase 6 learning evaluation (design SS7 + SS8).

Controlled experiments on held-out tasks NOT used to construct the
learning mechanism. Held-out worlds C and D (and generated variants
C1-C5 / D1-D5 with different names, fixed seeds) are defined ONLY in
this file -- no world names appear in ``ab/`` or ``ab/learning/``.
Both worlds have the same structural shapes as the Phase 5 chamber
worlds (a tool-less decoy opener, a fragile tool, a durable tool, a
door that resets once) with different names, objects and room layouts;
D has four rooms where C has three.

Variants, all run on the same task set (C1-C5 then D1-D5):

- ``full``: the brain as built, one brain carried across the task set.
- ``no_learning``: the learner replaced by a no-op object of the same
  public API (``record`` does nothing, ``propose_rules`` returns
  ``[]``); planner costs never adapt from experience. The planner is
  re-attached to the no-op learner so its derived stats agree.
- ``no_patterns``: the pattern library detached from the planner via
  ``planner.attach_patterns(None, None)`` (supported by the planner:
  the pattern multiplier is then 1.0). Patterns are still *learned*
  into ``brain.patterns`` by the brain's own hook, but never applied
  to planning -- that is the ablation, recorded as such.
- ``no_consolidation``: the end-of-cycle consolidation pass
  disabled (the brain instance's ``_consolidation_pass`` replaced
  with a no-op in this eval module): P-C/P-D candidates are never
  promoted and concepts C-1/C-2 are never learned. P-A/P-B still
  apply immediately. Manual ``Brain.consolidate()`` is never called.
- ``no_calibration``: prediction confidence pinned at 1.0 via a
  CalibrationTracker bypass -- the brain instance's tracker is
  replaced (eval module only, never ``ab/``) with a stub whose
  ``calibrated_confidence`` always returns 1.0.
- ``fresh``: a new brain per task (no carry-over); the control for
  transfer.

Per variant, per task: success, failed attempts, cycles, transfer
(continued vs fresh on the same task), recovery after the mid-task
reset, retention after restart, regression / overfitting (base world C
re-run after learning the new tasks, vs the same variant's fresh-C
baseline). Means are reported with sample sizes; where n >= 5, mean
+/- sample std and the range. No confidence interval is claimed at
this scale. Negative results are included, never hidden.

Sub-experiments (design SS7.1-SS7.5) and the six-step acceptance
experiment (design SS8) live in this module too. Each returns a plain
record dict; ``run_learning_evaluation()`` runs them all, writes
``docs/phase6-learning-record.json`` and returns the record dict the
harness summarises.

Baseline-status notes (2026-11-21, Worker 3, updated after Workers
1-2 landed): the Phase 6 learning subsystem EXISTS in this tree
(``ab/learning/``: Consolidator, ConceptStore, CalibrationTracker,
Retriever, Prioritizer; ``Brain.consolidate`` / ``_consolidation_pass``;
P-C/P-D candidates in ``ab/world/patterns.py``; ``Learner(max_episodes)``;
persistence v4 + ``learning.json``). The ablations below use those
real APIs. Measured gaps, reported not hidden:

- P-C / P-D are structurally unlearnable in chamber worlds: P-C
  needs >= 3 successes of one action across >= 2 DISTINCT
  situations, but a chamber tool is only ever used in one situation
  (the middle room with the tool held); P-D needs one action failing
  >= 2 times and succeeding >= 1 time, but no chamber action fails
  twice and succeeds. The consolidator queue stays empty and no
  ``consolidated`` event fires on this task family. Two-family
  transfer is therefore measured on P-A / P-B; P-C / P-D transfer
  needs a task family with varying situations per tool use.
- The durable_tool CONCEPT (C-1) does learn (negative after one C
  run -- the fragile tool broke; positive after repeated durable
  successes), and survives save/load; the concept's planner
  preference only fires when the world model marks a tool
  ``breaks_after_first_use: False``, which happens on P-C promotion
  -- so in chamber worlds the 0.8 preference never fires.
- Calibration: the tracker works and ECE/Brier are measured from it.
  On the C1/C2/D1 stream the aggregate ECE improves (0.269 -> 0.141)
  and Brier improves (0.269 -> 0.133); the unreliable subset improves
  decisively (ECE 1.0 -> 0.31). The aggregate result is
  stream-dependent: on the base-C double-run stream probed during
  development it read slightly worse than the 1.0 default (most
  actions there are deterministic-reliable, so Laplace shrinkage
  toward 0.5 adds error on them). Both numbers are reported per
  stream; see run_calibration_experiment.
- The brain-fed retriever index carries episode OUTCOMES
  (success/failure), not claimed effects, so ``Retriever.precision``
  with an effect never hits on the brain's own index; the retrieval
  experiment feeds the real Retriever effect-shaped episode dicts
  (its documented shape) instead.

Runnable standalone::

    python3 -m ab.eval.learning_eval [--record PATH]

Exit 0 when a full record was written, even if a criterion fails (a
failed criterion is a recorded result, not a harness crash). No LLM,
stdlib only, deterministic (fixed seeds for variant generation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import tempfile
import time
from datetime import datetime, timezone

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action
from .disconnected import (
    ChamberWorld,
    _default_factory,
    _run_with_brain,
    chamber_world_actions,
)
from .transfer import _pattern_snapshot, prepare_brain_for_new_world

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase6-learning-record.json")

MAX_CYCLES = 80
FAILURE_MARKERS = ("does not move", "is dust", "no intact",
                   "nothing happened", "cannot reach")

# ------------------------------------------------------------------ worlds C / D
WORLD_C = {
    "name": "chamber-world-C",
    "rooms": ("copper", "silver", "gold"),
    "start": "copper",
    "middle": "silver",
    "far": "gold",
    "fragile": "virel",
    "durable": "sable",
    "decoy": "murmur",
    "treasure": "crown",
}

WORLD_D = {
    "name": "chamber-world-D",
    "rooms": ("iron", "tin", "lead", "zinc"),
    "start": "iron",
    "middle": "tin",
    "far": "zinc",
    "fragile": "pelm",
    "durable": "orix",
    "decoy": "drone",
    "treasure": "sigil",
}


class ChamberWorld4(Environment):
    """Four-room chamber world: start - mid1 - mid2 - far. The door
    sits between mid2 and the far room; the decoy, fragile and durable
    tools act in mid2; the door resets exactly once, as in
    ``ChamberWorld``. Same structural shape, different room count."""

    def __init__(self, spec: dict):
        self.spec = spec
        s, m1, m2, f = spec["rooms"]
        self.adj = {s: [m1], m1: [s, m2], m2: [m1, f], f: [m2]}
        self.fragile_key = f"has_{spec['fragile']}"
        self.durable_key = f"has_{spec['durable']}"
        self.treasure_key = f"has_{spec['treasure']}"
        self.s = {"at": s, self.fragile_key: False, self.durable_key: False,
                  "door_open": False, self.treasure_key: False}
        self._fragile_intact = True
        self._relock_done = False
        self._open_observes = 0
        self.log: list[str] = []

    def observe(self) -> dict:
        if self.s["door_open"] and not self._relock_done:
            self._open_observes += 1
            if self._open_observes >= 2:
                self.s["door_open"] = False
                self._relock_done = True
                self.log.append("reset: the mechanism resets, the door swings shut")
        obs = dict(self.s)
        if self.s["at"] == self.spec["start"]:
            obs.pop("door_open", None)
        return obs

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        spec, s = self.spec, self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.adj[s["at"]] and (dest != spec["far"] or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == f"take_{spec['fragile']}" and s["at"] == spec["rooms"][2] \
                and not s[self.fragile_key] and self._fragile_intact:
            s[self.fragile_key], ok, info = True, True, f"took the {spec['fragile']}"
        elif name == f"take_{spec['fragile']}":
            info = f"no intact {spec['fragile']} remains here"
        elif name == f"take_{spec['durable']}" and s["at"] == spec["rooms"][2] \
                and not s[self.durable_key]:
            s[self.durable_key], ok, info = True, True, f"took the {spec['durable']}"
        elif name == spec["decoy"]:
            info = "the air hums, but the door does not move"
        elif name == f"use_{spec['fragile']}" and s["at"] == spec["rooms"][2] \
                and s[self.fragile_key] and self._fragile_intact:
            s["door_open"], self._fragile_intact = True, False
            ok, info = True, (f"the {spec['fragile']} turns the mechanism and "
                               f"crumbles to dust as the door swings open")
        elif name == f"use_{spec['fragile']}":
            info = f"the {spec['fragile']} is dust; it cannot turn the mechanism"
        elif name == f"use_{spec['durable']}" and s["at"] == spec["rooms"][2] \
                and s[self.durable_key]:
            s["door_open"], ok, info = True, True, "the mechanism turns and the door opens"
        elif name == f"take_{spec['treasure']}" and s["at"] == spec["far"]:
            s[self.treasure_key], ok, info = True, True, f"took the {spec['treasure']}"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def world4_actions(spec: dict) -> list[Action]:
    """The brain's (initially naive) believed model of a 4-room world:
    same beliefs as ``chamber_world_actions`` -- the decoy opens the
    door (cheapest), the fragile object opens it, the durable object
    opens it -- with no knowledge of the decoy, the single use, or
    the reset."""
    s, m1, m2, f = spec["rooms"]
    frag, dur, dec, tre = (spec["fragile"], spec["durable"],
                           spec["decoy"], spec["treasure"])
    fk, dk, tk = f"has_{frag}", f"has_{dur}", f"has_{tre}"
    return [
        Action(f"go_{m1}_from_{s}", {"at": s}, {"at": m1}, args={"_do": "go", "room": m1}),
        Action(f"go_{s}_from_{m1}", {"at": m1}, {"at": s}, args={"_do": "go", "room": s}),
        Action(f"go_{m2}_from_{m1}", {"at": m1}, {"at": m2}, args={"_do": "go", "room": m2}),
        Action(f"go_{m1}_from_{m2}", {"at": m2}, {"at": m1}, args={"_do": "go", "room": m1}),
        Action(f"go_{f}", {"at": m2, "door_open": True}, {"at": f},
               args={"_do": "go", "room": f}),
        Action(f"go_{m2}_from_{f}", {"at": f}, {"at": m2}, args={"_do": "go", "room": m2}),
        Action(f"take_{frag}", {"at": m2, fk: False}, {fk: True},
               args={"_do": f"take_{frag}"}),
        Action(f"take_{dur}", {"at": m2, dk: False}, {dk: True},
               args={"_do": f"take_{dur}"}),
        Action(dec, {"at": m2}, {"door_open": True}, cost=0.5, args={"_do": dec}),
        Action(f"use_{frag}", {"at": m2, fk: True}, {"door_open": True},
               cost=1.0, args={"_do": f"use_{frag}"}),
        Action(f"use_{dur}", {"at": m2, dk: True}, {"door_open": True},
               cost=2.0, args={"_do": f"use_{dur}"}),
        Action(f"take_{tre}", {"at": f}, {tk: True}, args={"_do": f"take_{tre}"}),
    ]


def _run4(spec: dict, brain_factory, max_cycles: int = MAX_CYCLES,
          brain: Brain | None = None):
    """Run one 4-room world; mirrors ``_run_with_brain`` exactly
    (same record keys) so 3-room and 4-room runs are comparable."""
    env = ChamberWorld4(spec)
    actions = world4_actions(spec)
    if brain is None:
        brain = brain_factory(actions, {"compute", "environment"})
        state_cleared = False
    else:
        for a in actions:
            brain.planner.add_action(a)
        brain.state.clear()
        state_cleared = True
    tk = f"has_{spec['treasure']}"
    unknown_before = not brain.ask(spec["fragile"], "is_a").known
    brain.add_goal(f"get the {spec['treasure']}", {tk: True})
    cycle_before = brain.cycle
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[-1]
    failures = [e for e in env.log
                if any(m in e for m in FAILURE_MARKERS)]
    plan_events = [e for e in brain.events if e["kind"] == "plan"]
    max_plan_steps = 0
    for e in plan_events:
        if " > " in e["detail"]:
            max_plan_steps = max(max_plan_steps, e["detail"].count(" > ") + 1)
    surprises = sum(1 for e in brain.events
                    if e["kind"] == "surprise" or "surprise" in e["detail"])
    rec = {
        "world": spec["name"],
        "rooms": list(spec["rooms"]),
        "objects": [spec["fragile"], spec["durable"], spec["treasure"]],
        "unknown_before_run": unknown_before,
        "completed": goal.status == "done",
        "goal_status": goal.status,
        "stop_reason": stop,
        "cycles": brain.cycle - cycle_before,
        "failed_attempts": len(failures),
        "failure_log": failures,
        "mid_task_reset_observed": any("reset:" in e for e in env.log),
        "max_plan_steps": max_plan_steps,
        "plan_events": len(plan_events),
        "surprise_events": surprises,
        "learner_episodes": len(brain.learner.episodes),
        "learner_rules_proposed": len(
            [r for r in brain.rules.rules() if r.source == "experience"]),
        "planner_bans": len(brain.planner.banned_pairs),
        "final_state_keys": sorted(brain.world.keys()),
        "state_cleared_for_new_world": state_cleared,
        "env_log": list(env.log),
    }
    return rec, brain


def run_world(spec: dict, brain_factory, max_cycles: int = MAX_CYCLES,
              brain: Brain | None = None):
    """Run one held-out world spec (3-room via ``ChamberWorld``,
    4-room via ``ChamberWorld4``); returns ``(record, brain)``.

    ``Brain.run`` treats ``max_cycles`` as an ABSOLUTE cycle limit
    against the brain's cumulative ``cycle`` counter, so a continued
    brain (whose counter already holds earlier tasks) is given
    ``brain.cycle + max_cycles`` -- 80 fresh cycles for this task --
    while a fresh brain gets 80. The control budget
    (``Brain(max_cycles=200)``) still caps the carried brain's total
    across the task set (~135 cycles for 10 tasks)."""
    if brain is not None:
        max_cycles = brain.cycle + max_cycles
    if len(spec["rooms"]) == 3:
        return _run_with_brain(spec, brain_factory, max_cycles, brain=brain)
    if len(spec["rooms"]) == 4:
        return _run4(spec, brain_factory, max_cycles, brain=brain)
    raise ValueError(f"unsupported room count {len(spec['rooms'])}")


def world_actions(spec: dict) -> list[Action]:
    if len(spec["rooms"]) == 3:
        return chamber_world_actions(spec)
    return world4_actions(spec)


# ------------------------------------------------------------------ generated variants
_ROOM_POOL = [
    "copper", "silver", "gold", "bronze", "pewter", "brass", "iron",
    "tin", "lead", "zinc", "nickel", "cobalt", "marble", "slate",
    "onyx", "jade", "pearl", "coral", "ivory", "ebony", "garnet",
    "topaz", "quartz", "flint", "basalt", "granite", "cedar", "birch",
    "willow", "alder", "maple", "ash", "elm", "oak", "pine", "rowan",
    "heather", "moss", "fern", "sage", "heath", "dune", "cliff", "grove",
]
_FRAGILE_POOL = ["virel", "nexil", "pelm", "quor", "zenth", "myrrh",
                 "thal", "vorin", "kess", "drayl", "oskin", "frel"]
_DURABLE_POOL = ["sable", "torva", "orix", "fenn", "garth", "luma",
                 "norin", "vex", "halcyon", "brin", "corin", "dusk"]
_DECOY_POOL = ["murmur", "whisper", "drone", "susurrus", "threnody",
               "keen", "wail", "sigh", "hush", "echo", "moan", "lilt"]
_TREASURE_POOL = ["crown", "chalice", "sigil", "sceptre", "diadem",
                  "amulet", "prism", "lens", "keystone", "monolith",
                  "talisman", "obelisk"]


def generate_variants() -> dict:
    """Deterministic held-out variants C1-C5 (3-room) and D1-D5
    (4-room). Fixed seeds; names sampled without replacement across
    the whole set (base worlds C/D names are excluded up front), so
    no two variants share a room or object name. Same structural
    shapes as C/D; only names (and D's extra room) differ."""
    used_rooms = set(WORLD_C["rooms"]) | set(WORLD_D["rooms"])
    used = {k: set() for k in ("fragile", "durable", "decoy", "treasure")}
    for spec in (WORLD_C, WORLD_D):
        for key in used:
            used[key].add(spec[key])
    out: dict = {}
    for i in range(1, 6):
        rng = random.Random(100 + i)
        rooms = _sample(rng, _ROOM_POOL, 3, used_rooms)
        spec = {"name": f"chamber-world-C{i}", "rooms": rooms,
                "start": rooms[0], "middle": rooms[1], "far": rooms[2],
                "fragile": _sample1(rng, _FRAGILE_POOL, used["fragile"]),
                "durable": _sample1(rng, _DURABLE_POOL, used["durable"]),
                "decoy": _sample1(rng, _DECOY_POOL, used["decoy"]),
                "treasure": _sample1(rng, _TREASURE_POOL, used["treasure"])}
        out[f"C{i}"] = spec
    for i in range(1, 6):
        rng = random.Random(200 + i)
        rooms = _sample(rng, _ROOM_POOL, 4, used_rooms)
        spec = {"name": f"chamber-world-D{i}", "rooms": rooms,
                "start": rooms[0], "middle": rooms[2], "far": rooms[3],
                "fragile": _sample1(rng, _FRAGILE_POOL, used["fragile"]),
                "durable": _sample1(rng, _DURABLE_POOL, used["durable"]),
                "decoy": _sample1(rng, _DECOY_POOL, used["decoy"]),
                "treasure": _sample1(rng, _TREASURE_POOL, used["treasure"])}
        out[f"D{i}"] = spec
    return out


def _sample(rng, pool, n, used):
    picked = []
    candidates = [x for x in pool if x not in used]
    for _ in range(n):
        choice = rng.choice(candidates)
        candidates.remove(choice)
        used.add(choice)
        picked.append(choice)
    return tuple(picked)


def _sample1(rng, pool, used):
    return _sample(rng, pool, 1, used)[0]


VARIANTS = generate_variants()
TASK_SET = [(f"C{i}", VARIANTS[f"C{i}"]) for i in range(1, 6)] + \
           [(f"D{i}", VARIANTS[f"D{i}"]) for i in range(1, 6)]


# ------------------------------------------------------------------ variant factories
class _NoOpLearner:
    """Learner-shaped no-op (same shape as the harness's): the same
    public API Brain uses, but nothing is ever learned."""

    def __init__(self, *args, **kwargs):
        self.episodes = []

    def record(self, ep):
        pass

    def propose_rules(self):
        return []

    def episodes_for_induction(self):
        return []

    def failures_for(self, action, situation, args=None):
        # Phase 10.5 (P10.5-7): the no-op learner records no
        # failures, so no plan is ever a loop plan for it.
        return 0

    def to_json(self):
        return {"episodes": [], "counts": []}


class _SuccessWithheldLearner:
    """Proxy over the real learner that drops SUCCESS episodes (they
    are never recorded) and forwards everything else. Used to test
    whether success-driven learning matters: with successes withheld,
    no success-contrast pattern (P-A needs a success, P-B needs a
    success-then-failure) can be learned and planner failure rates
    never improve from a success."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def episodes(self):
        return self._real.episodes

    def record(self, ep):
        if not ep.ok:
            self._real.record(ep)


def _disable_consolidation(brain) -> str:
    """Disable the consolidation pass via the design API. The pass
    (``Brain._consolidation_pass``) is what promotes P-C/P-D
    candidates and learns concepts C-1/C-2 at end of cycle, so the
    ablation replaces the brain INSTANCE's pass with a no-op (eval
    module only, never ``ab/``). P-A/P-B still apply immediately
    (they never go through the queue); manual promotion via
    ``Brain.consolidate(candidate_id)`` still exists but is never
    called by this variant. Returns a mechanism note."""
    if hasattr(brain, "_consolidation_pass"):
        brain._consolidation_pass = lambda: None
        return ("End-of-cycle consolidation pass disabled for this "
                "brain instance (brain._consolidation_pass replaced "
                "with a no-op in this eval module): P-C/P-D candidates "
                "are never promoted and concepts C-1/C-2 are never "
                "learned. P-A/P-B still apply immediately; manual "
                "Brain.consolidate() is never called.")
    if hasattr(brain, "consolidate") and callable(getattr(brain, "consolidate")):
        return ("Brain.consolidate() present but no consolidation "
                "pass found to disable; variant ran as full-equivalent.")
    return ("NO CONSOLIDATION API PRESENT (Worker 2 not landed). This "
            "variant ran as a full-equivalent; its numbers are "
            "identical to 'full' by construction, NOT an independent "
            "measurement. Re-run once Brain.consolidate / the "
            "consolidation pass lands.")


def _full_factory(actions, granted):
    return Brain(actions, granted=granted)


def _no_learning_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    brain.learner = _NoOpLearner()
    brain.planner.attach_learner(brain.learner)
    return brain


def _no_patterns_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    # Supported by Planner.attach_patterns: with no library attached
    # the pattern multiplier is 1.0 and planning is unchanged.
    # Patterns are still learned into brain.patterns (the brain's own
    # hook), but never applied -- that detachment IS the ablation.
    brain.planner.attach_patterns(None, None)
    return brain


def _no_consolidation_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    brain._no_consolidation_mechanism = _disable_consolidation(brain)
    return brain


class _PinnedCalibration:
    """CalibrationTracker-shaped stub that pins confidence at 1.0:
    ``is_calibrated`` True and ``calibrated_confidence`` 1.0 for
    every pair, records nothing. Installed on the brain instance in
    this eval module only (never ``ab/``); the brain's
    ``_store_predictions`` then takes the tracker branch and always
    stores 1.0. ``to_json`` keeps ``Brain.save`` working on such a
    brain (a load rebuilds a real tracker from the class)."""

    def is_calibrated(self, *args):
        return True

    def calibrated_confidence(self, *args):
        return 1.0

    def record(self, *args, **kwargs):
        pass

    def counts(self, *args):
        return None

    def to_json(self):
        return {"version": 1, "pairs": []}


def _no_calibration_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    # CalibrationTracker bypass (design §7): pin confidence at the
    # 1.0 default by replacing the brain instance's tracker with a
    # stub that always answers 1.0. Eval module only, never ab/.
    brain.calibration = _PinnedCalibration()
    return brain


def _fresh_factory(actions, granted):
    return Brain(actions, granted=granted)


def _success_withheld_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    brain.learner = _SuccessWithheldLearner(brain.learner)
    brain.planner.attach_learner(brain.learner)
    return brain


VARIANT_MECHANISMS = {
    "full": "Brain as built; one brain carried across the task set.",
    "no_learning": ("Learner replaced by a no-op of the same public "
                    "API (record does nothing, propose_rules returns "
                    "[]); planner re-attached to the no-op so derived "
                    "stats agree. Planner pair-bans still apply "
                    "(they are set at execution time, not by the "
                    "learner) -- see the procedural negative control."),
    "no_patterns": ("Pattern library detached from the planner via "
                    "planner.attach_patterns(None, None): pattern "
                    "multiplier is 1.0. Patterns are still learned "
                    "into brain.patterns but never applied."),
    "no_consolidation": ("End-of-cycle consolidation pass disabled "
                         "for this brain instance (the pass replaced "
                         "with a no-op in this eval module): P-C/P-D "
                         "candidates are never promoted and concepts "
                         "C-1/C-2 are never learned. P-A/P-B still "
                         "apply immediately; manual Brain.consolidate() "
                         "is never called."),
    "no_calibration": ("Prediction confidence pinned at 1.0 via a "
                       "CalibrationTracker bypass: the brain instance's "
                       "tracker is replaced in this eval module with a "
                       "stub whose calibrated_confidence always returns "
                       "1.0. No ab/ code touched."),
    "fresh": ("A new brain per task; no carry-over. The transfer "
              "control: continued == fresh by construction."),
}


def make_variant_factories() -> dict:
    """The six evaluation variants, keyed by name."""
    return {"full": _full_factory,
            "no_learning": _no_learning_factory,
            "no_patterns": _no_patterns_factory,
            "no_consolidation": _no_consolidation_factory,
            "no_calibration": _no_calibration_factory,
            "fresh": _fresh_factory}


# ------------------------------------------------------------------ stats helpers
def _mean_std(values: list) -> dict:
    """Mean with sample size; mean +/- sample std and range where
    n >= 5 (an honest uncertainty measure at this scale -- no
    confidence interval is claimed). Std is None when n < 2."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = statistics.fmean(values)
    return {"n": n, "mean": round(mean, 3),
            "std": round(statistics.stdev(values), 3) if n >= 2 else None,
            "min": min(values), "max": max(values)}


def _summarise_runs(runs: list[dict]) -> dict:
    """Summary stats over completed runs (cycles, failed attempts)."""
    done = [r for r in runs if r["completed"]]
    return {"n_tasks": len(runs), "n_completed": len(done),
            "success_rate": round(len(done) / len(runs), 3) if runs else None,
            "cycles": _mean_std([r["cycles"] for r in done]),
            "failed_attempts": _mean_std([r["failed_attempts"] for r in done])}


def _pattern_kinds(brain) -> dict:
    """Pattern kinds present in the brain, with counts and ids."""
    snap = _pattern_snapshot(brain)
    kinds: dict[str, list[str]] = {}
    for p in brain.patterns.patterns():
        kinds.setdefault(p.kind, []).append(p.id)
    return {"count": snap.get("count", 0), "ids": snap.get("ids", []),
            "names": snap.get("names", []), "by_kind": kinds}


def _applied_pattern_kinds(brain, start_event_idx: int) -> list[dict]:
    """``pattern_applied`` events from ``start_event_idx`` onwards,
    with each event's penalised step and the kinds of the patterns
    named in it (matched by id prefix)."""
    by_prefix = {}
    for p in brain.patterns.patterns():
        by_prefix[p.id[:12]] = p.kind
    out = []
    for e in brain.events[start_event_idx:]:
        if e.get("kind") != "pattern_applied":
            continue
        detail = e.get("detail", "")
        kinds = sorted({by_prefix[tok] for tok in detail.replace(",", " ").split()
                        if tok in by_prefix})
        out.append({"cycle": e.get("cycle"), "detail": detail, "kinds": kinds})
    return out


def _fragile_retried_after_reset(env_log: list[str], spec: dict) -> bool:
    reset_idx = next((i for i, e in enumerate(env_log)
                      if e.startswith("reset:")), None)
    if reset_idx is None:
        return False
    return any(e.startswith(f"use_{spec['fragile']}")
               for e in env_log[reset_idx + 1:])


def _prediction_outcomes(brain, start_event_idx: int = 0) -> tuple[int, int]:
    """(met, violated) counts from ``prediction_checked`` events."""
    met = violated = 0
    for e in brain.events[start_event_idx:]:
        if e.get("kind") != "prediction_checked":
            continue
        detail = e.get("detail", "")
        if "outcome=met" in detail:
            met += 1
        elif "outcome=violated" in detail:
            violated += 1
    return met, violated


def _consolidation_state(brain) -> dict:
    """The brain's consolidation queue as observable state: counts by
    status and the candidate ids. Empty when the brain has no
    consolidator (or no candidate ever formed)."""
    consolidator = getattr(brain, "consolidator", None)
    if consolidator is None:
        return {"present": False, "candidates": 0, "promoted": 0,
                "rejected": 0, "ids": []}
    try:
        all_cands = consolidator.all()
    except Exception:
        return {"present": True, "candidates": None, "promoted": None,
                "rejected": None, "ids": [],
                "note": "consolidator unreadable"}
    return {"present": True,
            "candidates": sum(1 for c in all_cands if c["status"] == "candidate"),
            "promoted": sum(1 for c in all_cands if c["status"] == "promoted"),
            "rejected": sum(1 for c in all_cands if c["status"] == "rejected"),
            "ids": [c["id"] for c in all_cands]}


def _concept_state(brain) -> dict:
    """The brain's concept store as observable state (names, per
    concept positives / negatives / confidence). Empty when the brain
    has no concept store."""
    store = getattr(brain, "concepts", None)
    if store is None:
        return {"present": False, "names": [], "concepts": []}
    try:
        concepts = store.to_json().get("concepts", [])
    except Exception:
        return {"present": True, "names": [], "concepts": [],
                "note": "concept store unreadable"}
    return {"present": True,
            "names": [c.get("name") for c in concepts],
            "concepts": [{"name": c.get("name"),
                          "positives": c.get("positives", []),
                          "negatives": c.get("negatives", []),
                          "confidence": c.get("confidence")}
                         for c in concepts]}


# ------------------------------------------------------------------ calibration (SS7.3)
def compute_calibration(brain) -> dict:
    """ECE + Brier on the brain's accumulated prediction stream, two
    ways, on the SAME stream:

    - ``default``: every prediction at the 1.0 default (no record
      yet) -- the uncalibrated confidence. ECE_default = 1 - met
      rate; Brier_default = 1 - met rate.
    - ``calibrated``: the CalibrationTracker's own ``ece()`` /
      ``brier()`` (exact from its recorded (predicted, outcome)
      pairs; after a load the tracker falls back to a per-pair
      aggregate, which reads higher -- see the note).

    Preferred source is the real tracker (``brain.calibration``);
    when it has no ``ece``/``brier`` (e.g. the pinned stub of the
    no_calibration variant), the computation falls back to the
    causal stream: per-(action, effect) buckets from
    ``prediction_checked`` events + ``causal.counts``, with the
    Laplace rate (met+1)/(total+2) as the calibrated prediction
    (Brier then a bucket approximation -- within a bucket every
    outcome shares the bucket's p).

    Also reported: the unreliable subset (pairs with observed rate
    < 1.0), where calibration is expected to help most -- the
    aggregate is dominated by deterministic-reliable actions whose
    first predictions at 1.0 were already right, so Laplace
    shrinkage toward 0.5 adds error on them."""
    tracker = getattr(brain, "calibration", None)
    met_total = violated_total = 0
    buckets = []
    for e in brain.events:
        if e.get("kind") != "prediction_checked":
            continue
        detail = e.get("detail", "")
        try:
            action = detail.split(" predicted ")[0].strip()
            effect = detail.split(" predicted ")[1].split("=")[0].strip()
        except IndexError:
            continue
        counts = brain.causal.counts(action, effect)
        if counts is None:
            continue
        met, total = counts
        if (action, effect) not in [(b["action"], b["effect"]) for b in buckets]:
            buckets.append({"action": action, "effect": effect,
                            "met": met, "total": total})
        if "outcome=met" in detail:
            met_total += 1
        elif "outcome=violated" in detail:
            violated_total += 1
    n = met_total + violated_total
    if n == 0:
        return {"n_predictions": 0, "note": "no prediction_checked events"}
    met_rate = met_total / n
    ece_default = brier_default = 1.0 - met_rate
    use_tracker = (tracker is not None
                   and callable(getattr(tracker, "ece", None))
                   and callable(getattr(tracker, "brier", None)))
    if use_tracker:
        try:
            ece_cal = float(tracker.ece())
            brier_cal = float(tracker.brier())
            source = "CalibrationTracker.ece()/brier() (exact raw pairs)"
        except Exception:
            use_tracker = False
    if not use_tracker:
        total_counted = sum(b["total"] for b in buckets)
        if total_counted == 0:
            return {"n_predictions": n, "met_rate": round(met_rate, 4),
                    "note": "no causal buckets readable"}
        ece_num = brier_num = 0.0
        for b in buckets:
            p = (b["met"] + 1) / (b["total"] + 2)
            observed = b["met"] / b["total"] if b["total"] else 0.0
            ece_num += b["total"] * abs(p - observed)
            brier_num += (b["met"] * (1 - p) ** 2
                          + (b["total"] - b["met"]) * p ** 2)
            b["predicted"] = round(p, 4)
            b["observed"] = round(observed, 4)
        ece_cal = ece_num / total_counted
        brier_cal = brier_num / total_counted
        source = ("causal-stream fallback (per-(action, effect) Laplace "
                  "buckets; Brier is a bucket approximation)")
    unreliable = [b for b in buckets
                  if b["total"] and b["met"] / b["total"] < 1.0]
    unrel_n = sum(b["total"] for b in unreliable)
    if unrel_n:
        unrel_default = 1.0 - sum(b["met"] for b in unreliable) / unrel_n
        unrel_cal = sum(
            b["total"] * abs((b["met"] + 1) / (b["total"] + 2)
                             - b["met"] / b["total"])
            for b in unreliable) / unrel_n
    else:
        unrel_default = unrel_cal = None
    return {"n_predictions": n, "met": met_total, "violated": violated_total,
            "met_rate": round(met_rate, 4),
            "ece_default_1.0": round(ece_default, 4),
            "ece_calibrated": round(ece_cal, 4),
            "brier_default_1.0": round(brier_default, 4),
            "brier_calibrated": round(brier_cal, 4),
            "ece_improved": ece_cal < ece_default,
            "brier_improved": brier_cal < brier_default,
            "calibrated_source": source,
            "unreliable_subset": {
                "pairs": len(unreliable), "n_predictions": unrel_n,
                "ece_default_1.0": (round(unrel_default, 4)
                                    if unrel_default is not None else None),
                "ece_calibrated": (round(unrel_cal, 4)
                                   if unrel_cal is not None else None),
                "ece_improved": (bool(unrel_cal is not None
                                      and unrel_default is not None
                                      and unrel_cal < unrel_default))},
            "buckets": buckets}


def run_calibration_experiment() -> dict:
    """SS7.3: one fresh brain runs C1, C2, D1 in sequence (continued,
    transient cleared between). Per run: met-rate trend and the
    cumulative tracker ECE/Brier after that run (default-1.0 vs
    calibrated on the same cumulative stream). The §6.7 criterion
    (ECE after learning strictly lower than ECE with the 1.0
    default) is evaluated on the final cumulative stream and on the
    unreliable subset; both numbers are reported either way."""
    factory = _full_factory
    brain = None
    per_run = []
    for task_id, spec in (("C1", VARIANTS["C1"]), ("C2", VARIANTS["C2"]),
                          ("D1", VARIANTS["D1"])):
        if brain is not None:
            prepare_brain_for_new_world(brain)
        ev0 = len(brain.events) if brain is not None else 0
        rec, brain = run_world(spec, factory, brain=brain)
        met, violated = _prediction_outcomes(brain, ev0)
        overall = compute_calibration(brain)
        per_run.append({"task": task_id, "completed": rec["completed"],
                        "cycles": rec["cycles"],
                        "failed_attempts": rec["failed_attempts"],
                        "predictions": met + violated, "met": met,
                        "violated": violated,
                        "met_rate": round(met / (met + violated), 4)
                        if met + violated else None,
                        "cumulative_ece_default_1.0": overall.get("ece_default_1.0"),
                        "cumulative_ece_calibrated": overall.get("ece_calibrated"),
                        "cumulative_brier_default_1.0": overall.get("brier_default_1.0"),
                        "cumulative_brier_calibrated": overall.get("brier_calibrated")})
    overall = compute_calibration(brain)
    passed = bool(overall.get("ece_improved", False))
    return {"criterion": ("ECE after learning strictly lower than ECE "
                          "with the 1.0 default, on the same prediction "
                          "stream; met-rate trend across runs."),
            "per_run": per_run,
            "overall": overall,
            "passed": passed,
            "note": ("ECE/Brier come from the real CalibrationTracker. "
                     "The aggregate result is stream-dependent: on this "
                     "C1/C2/D1 stream it improves; on the base-C "
                     "double-run stream probed during development it "
                     "read slightly worse than the 1.0 default (most "
                     "actions there are deterministic-reliable, so "
                     "Laplace shrinkage toward 0.5 adds error on "
                     "them). The unreliable subset is reported "
                     "alongside either way, and a 'passed: false' here "
                     "would be a finding about the calibration approach "
                     "on that stream, not a harness failure.")}


# ------------------------------------------------------------------ retrieval (SS7.4)
def _frozen_features(situation: dict) -> frozenset:
    return frozenset(f"{k}={v}" for k, v in sorted(situation.items()))


def _retrieve(episodes, query_features: frozenset, query_effect: str,
              action_effects: dict, k: int = 5) -> list:
    """Eval-side fallback ranking (the design's rule: Jaccard over
    frozen situation keys + effect match, most-recent-first on ties).
    Used only when the real ``Retriever`` cannot be imported; the
    experiment prefers the real component (see
    ``run_retrieval_experiment``)."""
    scored = []
    for idx, ep in enumerate(episodes):
        effs = action_effects.get(ep.action, {})
        effect_match = 1 if query_effect in effs else 0
        feats = _frozen_features(ep.situation)
        union = query_features | feats
        jaccard = (len(query_features & feats) / len(union)) if union else 0.0
        scored.append(((effect_match, jaccard, idx), ep))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [ep for _score, ep in scored[:k]]


def run_retrieval_experiment() -> dict:
    """SS7.4: retrieval precision on a held-out query set, early vs
    late, using the real ``Retriever``. Query set: the episodes of a
    FRESH brain's run of held-out C4 (never trained on). Early index:
    a brain trained on C1 only. Late index: a brain trained on
    C1-C3. Precision@5 = fraction of the top-5 retrieved episodes
    whose action claims the query's effect.

    The brain's own retriever index carries episode OUTCOMES
    (success/failure), not claimed effects, so ``precision`` with an
    effect never hits on it; the experiment therefore feeds the real
    Retriever episode dicts in its documented shape (action,
    situation, effect, cycle), built from the trained brain's real
    recorded episodes. That is the component under test, with
    effect-shaped inputs."""
    try:
        from ..learning import Retriever
    except ImportError:
        Retriever = None
    # Query source: fresh brain runs held-out C4.
    qrec, qbrain = run_world(VARIANTS["C4"], _full_factory)
    q_actions = {a.name: dict(a.eff) for a in world_actions(VARIANTS["C4"])}
    queries = []
    for ep in qbrain.learner.episodes:
        effs = q_actions.get(ep.action, {})
        if not effs:
            continue
        effect = sorted(effs)[0]
        queries.append({"action": ep.action, "effect": effect,
                        "features": dict(ep.situation)})
    action_effects: dict[str, dict] = {}
    for spec in (VARIANTS["C1"], VARIANTS["C2"], VARIANTS["C3"]):
        for a in world_actions(spec):
            action_effects.setdefault(a.name, dict(a.eff))

    def precision_for(train_specs):
        brain = None
        for spec in train_specs:
            if brain is not None:
                prepare_brain_for_new_world(brain)
            _rec, brain = run_world(spec, _full_factory, brain=brain)
        episodes = list(brain.learner.episodes)
        if Retriever is not None:
            index = [{"action": ep.action, "situation": dict(ep.situation),
                      "effect": sorted(action_effects.get(ep.action, {}))[0]
                      if action_effects.get(ep.action) else None,
                      "cycle": ep.cycle} for ep in episodes]
            retriever = Retriever(index)
            precisions = [retriever.precision(q["features"], q["effect"], k=5)
                          for q in queries]
        else:
            precisions = []
            for q in queries:
                top = _retrieve(episodes, _frozen_features(q["features"]),
                                q["effect"], action_effects, k=5)
                if not top:
                    continue
                hits = sum(1 for ep in top
                           if q["effect"] in action_effects.get(ep.action, {}))
                precisions.append(hits / len(top))
        return {"episodes_indexed": len(episodes),
                "n_queries": len(precisions),
                "mean_precision_at_5": (round(statistics.fmean(precisions), 4)
                                        if precisions else None),
                "precisions": [round(p, 3) for p in precisions],
                "retriever": ("ab.learning.Retriever (real component, "
                              "effect-shaped episode dicts)"
                              if Retriever is not None else
                              "eval-side fallback ranking (Retriever "
                              "import failed)")}

    early = precision_for([VARIANTS["C1"]])
    late = precision_for([VARIANTS["C1"], VARIANTS["C2"], VARIANTS["C3"]])
    improved = (late["mean_precision_at_5"] is not None
                and early["mean_precision_at_5"] is not None
                and late["mean_precision_at_5"] > early["mean_precision_at_5"])
    return {"criterion": ("Retrieval precision on a held-out query set "
                          "improves as episodes accumulate (early vs "
                          "late); reported with numbers if it does not."),
            "query_world": "C4 (fresh-brain run)",
            "query_episodes": len(queries),
            "query_run": {"completed": qrec["completed"],
                          "cycles": qrec["cycles"],
                          "failed_attempts": qrec["failed_attempts"]},
            "early_trained_on_C1": early,
            "late_trained_on_C1_C2_C3": late,
            "improved": improved,
            "note": ("The brain's own retriever index carries outcomes, "
                     "not effects (see module notes); this experiment "
                     "feeds the real Retriever effect-shaped dicts built "
                     "from the trained brain's real episodes.")}


# ------------------------------------------------------------------ cap (SS7.5)
def _prioritized_prune(brain, cap: int) -> dict:
    """Prune the brain's learner episodes with the real Prioritizer:
    episode dicts (with a hidden ``_idx`` back-pointer to the
    original Episode objects) are ranked by the design utility
    (0.5 * recency + 0.5 * usefulness, from the brain's own
    usefulness counts, fed by the episode hook during training);
    the top ``cap`` survive, restored to chronological order.
    Patterns and consolidated knowledge are never pruned (they live
    outside the episode list). Returns counts for the record."""
    episodes = list(brain.learner.episodes)
    dicts = [{"action": ep.action, "situation": dict(ep.situation),
              "cycle": ep.cycle,
              "outcome": "success" if ep.ok else "failure",
              "_idx": i} for i, ep in enumerate(episodes)]
    kept_dicts = brain.prioritizer.prune(dicts, cap,
                                         current_cycle=brain.cycle)
    keep_idx = sorted(d["_idx"] for d in kept_dicts)
    brain.learner.episodes = [episodes[i] for i in keep_idx]
    brain.learner._rebuild_counts()
    return {"episodes_before": len(episodes),
            "episodes_after": len(brain.learner.episodes)}


def run_cap_experiment() -> dict:
    """SS7.5: MAX_EPISODES in {100, 500, 2000} x {tail, prioritized}
    pruning. Protocol per setting: fresh brain trains on C1-C3, then
    synthetic filler episodes (no pattern-relevant structure) push
    the total past the cap, then prune, then measure retention
    (real training episodes surviving, and pattern-evidence episodes
    surviving; patterns themselves are never pruned) and transfer
    (continued D1 failed attempts / cycles vs fresh D1).

    Tail pruning is the real ``Learner`` cap (the brain's
    ``learner.max_episodes`` set for the setting). Prioritized
    pruning is the real ``Prioritizer.prune`` on the brain's own
    usefulness counts. The default cap stays 500 unless this
    measurement says otherwise."""
    from ..core.learning import Episode

    fresh_d1, _ = run_world(VARIANTS["D1"], _full_factory)
    fresh_d1 = {k: fresh_d1[k] for k in ("completed", "cycles", "failed_attempts")}
    results = {}
    for cap in (100, 500, 2000):
        for pruning in ("tail", "prioritized"):
            brain = None
            for spec in (VARIANTS["C1"], VARIANTS["C2"], VARIANTS["C3"]):
                if brain is not None:
                    prepare_brain_for_new_world(brain)
                _rec, brain = run_world(spec, _full_factory, brain=brain)
            patterns_before = _pattern_kinds(brain)
            pattern_actions = set()
            for p in brain.patterns.patterns():
                for ev in p.evidence:
                    pattern_actions.add(ev["action"])
            real_eps = list(brain.learner.episodes)
            n_real = len(real_eps)
            evidence_eps = [ep for ep in real_eps
                            if ep.action in pattern_actions]
            n_evidence = len(evidence_eps)
            real_ids = {id(ep) for ep in real_eps}
            evidence_ids = {id(ep) for ep in evidence_eps}
            # The cap is a per-learner setting (Learner(max_episodes));
            # tail mode enforces it live, prioritized mode lifts it
            # and prunes once at the end with the real Prioritizer.
            # Filler episodes are appended DIRECTLY to the episode
            # list (not via Learner.record): record() fires the
            # retriever re-index on every episode, which is O(n^2)
            # over the filler and would dominate the experiment, and
            # filler must not earn usefulness counts or retriever
            # entries -- it is synthetic padding, not experience.
            # The tail slice below replicates record()'s cap
            # enforcement exactly (keep the last ``cap``), and counts
            # are rebuilt once at the end, as record() does.
            brain.learner.max_episodes = cap if pruning == "tail" else 10 ** 9
            target_total = cap + 50
            while len(brain.learner.episodes) < target_total:
                i = len(brain.learner.episodes)
                ep = Episode(cycle=brain.cycle,
                             action=f"filler_{i % 7}",
                             situation={"at": "nowhere", "slot": i % 5},
                             ok=(i % 2 == 0), surprises=[])
                brain.learner.episodes.append(ep)
                brain.planner.record(ep.action, ep.ok)
            if pruning == "tail":
                brain.learner.episodes = brain.learner.episodes[-cap:]
            brain.learner._rebuild_counts()
            prune_info = {"episodes_before": target_total,
                          "episodes_after": len(brain.learner.episodes)}
            if pruning == "prioritized":
                prune_info = _prioritized_prune(brain, cap)
            survivors = list(brain.learner.episodes)
            kept_real = sum(1 for ep in survivors if id(ep) in real_ids)
            kept_evidence = sum(1 for ep in survivors
                                if id(ep) in evidence_ids)
            patterns_after = _pattern_kinds(brain)
            prepare_brain_for_new_world(brain)
            drec, _ = run_world(VARIANTS["D1"], _full_factory, brain=brain)
            results[f"cap_{cap}_{pruning}"] = {
                "cap": cap, "pruning": pruning,
                "episodes_before_filler": n_real,
                "episodes_after_prune": prune_info["episodes_after"],
                "real_episodes_retained": kept_real,
                "real_episodes_retained_fraction": (
                    round(kept_real / n_real, 4) if n_real else None),
                "pattern_evidence_episodes": n_evidence,
                "pattern_evidence_retained": kept_evidence,
                "pattern_evidence_retained_fraction": (
                    round(kept_evidence / n_evidence, 4) if n_evidence else None),
                "patterns_before": patterns_before["count"],
                "patterns_after": patterns_after["count"],
                "patterns_retained_fraction": (
                    round(patterns_after["count"] / patterns_before["count"], 4)
                    if patterns_before["count"] else None),
                "continued_d1": {k: drec[k] for k in
                                 ("completed", "cycles", "failed_attempts")},
                "fresh_d1": fresh_d1,
                "transfer_vs_fresh_d1": bool(
                    drec["completed"] and fresh_d1["completed"]
                    and drec["failed_attempts"] < fresh_d1["failed_attempts"]),
            }
    return {"criterion": ("Same task set with MAX_EPISODES in {100, 500, "
                          "2000} x tail vs prioritized pruning; retention "
                          "+ transfer per setting."),
            "fresh_d1_baseline": fresh_d1,
            "settings": results,
            "note": ("Tail pruning is the real Learner cap "
                     "(learner.max_episodes set per setting). "
                     "Prioritized pruning is the real "
                     "Prioritizer.prune on the brain's own usefulness "
                     "counts. Filler episodes are synthetic and carry "
                     "no pattern-relevant structure. The default cap "
                     "stays 500 unless this measurement says otherwise.")}


# ------------------------------------------------------------------ procedural (SS7.1)
def run_procedural_experiment() -> dict:
    """SS7.1: repeated task type, run 1 vs run 2. The discriminating
    form renames the world (C1 -> C2, same type): pair-bans are
    name-bound, so only durable learning (patterns / costs) can carry
    across the rename. Control: same-name C run 1 vs run 2 (with
    transient cleared) -- that improvement is ban-driven and persists
    with learning off, so it is reported as a negative control, not
    as procedural learning."""
    out = {}

    def renamed(factory, label):
        r1, brain = run_world(VARIANTS["C1"], factory)
        prepare_brain_for_new_world(brain)
        r2, _ = run_world(VARIANTS["C2"], factory, brain=brain)
        rf, _ = run_world(VARIANTS["C2"], factory)
        return {"run1_C1": {k: r1[k] for k in ("completed", "cycles", "failed_attempts")},
                "run2_C2_continued": {k: r2[k] for k in ("completed", "cycles", "failed_attempts")},
                "run2_C2_fresh": {k: rf[k] for k in ("completed", "cycles", "failed_attempts")},
                "improved_vs_fresh": bool(
                    r2["completed"] and rf["completed"]
                    and (r2["failed_attempts"], r2["cycles"])
                    < (rf["failed_attempts"], rf["cycles"]))}

    out["renamed_full"] = renamed(_full_factory, "full")
    out["renamed_no_learning"] = renamed(_no_learning_factory, "no_learning")

    def same_name(factory):
        r1, brain = run_world(WORLD_C, factory)
        prepare_brain_for_new_world(brain)
        r2, _ = run_world(WORLD_C, factory, brain=brain)
        return {"run1": {k: r1[k] for k in ("completed", "cycles", "failed_attempts")},
                "run2": {k: r2[k] for k in ("completed", "cycles", "failed_attempts")}}

    out["same_name_control_full"] = same_name(_full_factory)
    out["same_name_control_no_learning"] = same_name(_no_learning_factory)
    out["criterion"] = ("Second run of the same task type strictly "
                        "fewer cycles and/or failed attempts than the "
                        "first (same brain, learning on), and the "
                        "improvement disappears with learning off. "
                        "Renamed form isolates durable learning; "
                        "same-name form is a ban-driven negative control.")
    out["passed"] = bool(
        out["renamed_full"]["improved_vs_fresh"]
        and not out["renamed_no_learning"]["improved_vs_fresh"])
    out["note"] = ("Same-name run 2 improves even with learning off: "
                   "pair-bans are name-bound and survive the "
                   "transient clear, so the same-name improvement is "
                   "ban-driven, not learner-driven. The renamed form "
                   "is the discriminating test.")
    return out


# ------------------------------------------------------------------ two-family transfer (SS7.2)
def run_two_family_transfer() -> dict:
    """SS7.2: which pattern families transferred, per world (C -> D).
    P-A (unreliable tool-less opener): a ``pattern_applied`` event in
    D penalising the decoy. P-B (one-use tool): the P-B pattern
    learned in C is present before D, the fragile tool's planner
    multiplier is 6.0 in D, and the fragile tool is NOT retried after
    the reset in continued D while fresh D DOES retry it. P-C
    (durable tool) and P-D (requires precondition) are measured too,
    with their consolidation state: in chamber worlds neither forms
    (P-C needs >= 3 successes across >= 2 distinct situations, but a
    chamber tool is only ever used in one situation; P-D needs one
    action failing >= 2 times and succeeding, but no chamber action
    does that), so the consolidator queue stays empty and no
    ``consolidated`` event fires -- recorded honestly, not hidden.
    The durable_tool CONCEPT (C-1) is reported alongside: it learns
    from repeated successes and is the success-driven knowledge that
    does form here."""
    # Train on C.
    rec_c, brain = run_world(WORLD_C, _full_factory)
    patterns_before = _pattern_kinds(brain)
    kinds_before = patterns_before["by_kind"]
    consolidation_before = _consolidation_state(brain)
    concepts_before = _concept_state(brain)
    # Fresh D (control).
    rec_d_fresh, _ = run_world(WORLD_D, _full_factory)
    # Continued D.
    entry = prepare_brain_for_new_world(brain)
    ev0 = len(brain.events)
    rec_d_cont, brain_d = run_world(WORLD_D, _full_factory, brain=brain)
    applied = _applied_pattern_kinds(brain_d, ev0)
    applied_kinds = sorted({k for e in applied for k in e["kinds"]})
    # P-B liveness in D: multiplier on the fragile opener after the run.
    d_actions = {a.name: a for a in world_actions(WORLD_D)}
    pb_mult = brain_d.planner.pattern_multiplier(
        d_actions[f"use_{WORLD_D['fragile']}"])
    pa_mult = brain_d.planner.pattern_multiplier(d_actions[WORLD_D["decoy"]])
    concept_mult_durable = brain_d.planner.concept_multiplier(
        d_actions[f"use_{WORLD_D['durable']}"])
    fresh_retried = _fragile_retried_after_reset(rec_d_fresh["env_log"], WORLD_D)
    cont_retried = _fragile_retried_after_reset(rec_d_cont["env_log"], WORLD_D)
    families = {
        "unreliable_toolless_opener": {
            "learned_in_c": "unreliable_toolless_opener" in kinds_before,
            "applied_in_d": "unreliable_toolless_opener" in applied_kinds,
            "planner_multiplier_in_d": pa_mult,
            "applied_events": applied,
            "transferred": bool(
                "unreliable_toolless_opener" in kinds_before
                and "unreliable_toolless_opener" in applied_kinds),
        },
        "one_use_tool": {
            "learned_in_c": "one_use_tool" in kinds_before,
            "pattern_present_before_d": "one_use_tool" in kinds_before,
            "planner_multiplier_in_d": pb_mult,
            "fragile_retried_after_reset_fresh_d": fresh_retried,
            "fragile_retried_after_reset_continued_d": cont_retried,
            "fragile_avoided_in_continued_d": bool(fresh_retried and not cont_retried),
            "transferred": bool(
                "one_use_tool" in kinds_before
                and pb_mult == 6.0
                and fresh_retried and not cont_retried),
        },
        "durable_tool_P-C": {
            "learned_in_c": "durable_tool" in kinds_before,
            "pattern_present_before_d": "durable_tool" in kinds_before,
            "consolidation_before_d": consolidation_before,
            "concept_before_d": concepts_before,
            "concept_multiplier_for_durable_in_d": concept_mult_durable,
            "transferred": False,
            "note": ("No P-C pattern formed in C: P-C needs >= 3 "
                     "successes of one action across >= 2 distinct "
                     "situations, but a chamber tool is only ever used "
                     "in one situation. The consolidator queue is empty "
                     "and no 'consolidated' event fired. P-C transfer "
                     "is unmeasurable on this task family; it needs a "
                     "world where one tool succeeds in varying "
                     "situations."),
        },
        "requires_precondition_P-D": {
            "learned_in_c": "requires_precondition" in kinds_before,
            "pattern_present_before_d": "requires_precondition" in kinds_before,
            "consolidation_before_d": consolidation_before,
            "transferred": False,
            "note": ("No P-D pattern formed in C: P-D needs one action "
                     "failing >= 2 times and succeeding >= 1 time, but "
                     "no chamber action fails twice and succeeds. Same "
                     "structural limitation as P-C."),
        },
        "concept_durable_tool_C-1": {
            "concept_present_after_c": "durable_tool" in concepts_before["names"],
            "concept_state_after_c": concepts_before,
            "concept_multiplier_for_durable_in_d": concept_mult_durable,
            "note": ("The C-1 concept (not a pattern family): learned "
                     "from repeated tool outcomes; the planner's 0.8 "
                     "preference only fires when the world model marks "
                     "a tool breaks_after_first_use False (P-C "
                     "promotion), so in chamber worlds the multiplier "
                     "stays 1.0 and the concept is stored knowledge, "
                     "not yet applied preference."),
        },
    }
    transferred = [k for k, v in families.items() if v.get("transferred")]
    return {"criterion": ("At least two distinct pattern families "
                          "transfer to held-out world D."),
            "world_c": {k: rec_c[k] for k in ("completed", "cycles", "failed_attempts")},
            "world_d_fresh": {k: rec_d_fresh[k] for k in ("completed", "cycles", "failed_attempts")},
            "world_d_continued": {k: rec_d_cont[k] for k in ("completed", "cycles", "failed_attempts")},
            "patterns_learned_in_c": patterns_before,
            "entry_protocol": {"method": entry["method"],
                               "patterns_intact": entry["patterns_intact"]},
            "families": families,
            "families_transferred": transferred,
            "n_families_transferred": len(transferred),
            "passed": len(transferred) >= 2}


# ------------------------------------------------------------------ success-based learning
def run_success_withheld_experiment() -> dict:
    """Design SS6.11: a success-driven improvement (P-C / C-1 shape)
    on a held-out task, vanishing when success episodes are withheld.
    Training is C TWICE (same brain): one C run gives the durable
    tool a single success, but C-1 needs >= 2 successes with no break
    for a positive example, so the positive only forms on the second
    run. Full: the durable_tool concept gains a positive example and
    continued D beats fresh D. Success-withheld: no success episode
    ever reaches the learner, so no pattern is learned, no concept
    forms, and continued D ties fresh D. The withholding ablation is
    exact: success episodes never reach the learner."""
    # Full: train C twice -> continue D.
    rec_c1, brain = run_world(WORLD_C, _full_factory)
    prepare_brain_for_new_world(brain)
    rec_c2, brain = run_world(WORLD_C, _full_factory, brain=brain)
    pats_full = _pattern_kinds(brain)
    concepts_full = _concept_state(brain)
    is_a_durable = brain.concepts.is_a(
        {"breaks_after_first_use": False}, "durable_tool") \
        if hasattr(brain, "concepts") and brain.concepts is not None else (None, None)
    prepare_brain_for_new_world(brain)
    rec_d_full, _ = run_world(WORLD_D, _full_factory, brain=brain)
    rec_d_fresh, _ = run_world(WORLD_D, _full_factory)
    # Success-withheld: train C twice -> continue D.
    rec_c1_sw, brain_sw = run_world(WORLD_C, _success_withheld_factory)
    prepare_brain_for_new_world(brain_sw)
    rec_c2_sw, brain_sw = run_world(WORLD_C, _success_withheld_factory, brain=brain_sw)
    pats_sw = _pattern_kinds(brain_sw)
    concepts_sw = _concept_state(brain_sw)
    prepare_brain_for_new_world(brain_sw)
    rec_d_sw, _ = run_world(WORLD_D, _success_withheld_factory, brain=brain_sw)
    full_improved = bool(rec_d_full["completed"] and rec_d_fresh["completed"]
                         and rec_d_full["failed_attempts"] < rec_d_fresh["failed_attempts"])
    sw_improved = bool(rec_d_sw["completed"] and rec_d_fresh["completed"]
                       and rec_d_sw["failed_attempts"] < rec_d_fresh["failed_attempts"])
    full_concept_positive = any(
        c["name"] == "durable_tool" and c["positives"]
        for c in concepts_full["concepts"])
    sw_concept_positive = any(
        c["name"] == "durable_tool" and c["positives"]
        for c in concepts_sw["concepts"])
    return {"criterion": ("Success-driven improvement on held-out D "
                          "vanishes when success episodes are withheld."),
            "training": ("C run twice per brain: one C run gives the "
                         "durable tool a single success; C-1 needs >= 2 "
                         "successes with no break for a positive example."),
            "full": {"world_c_runs": [
                         {k: rec_c1[k] for k in ("completed", "cycles", "failed_attempts")},
                         {k: rec_c2[k] for k in ("completed", "cycles", "failed_attempts")}],
                     "patterns_learned_in_c": pats_full,
                     "concepts_after_c2": concepts_full,
                     "durable_tool_concept_positive": full_concept_positive,
                     "concept_is_a_durable_tool": {"features": {"breaks_after_first_use": False},
                                                   "answer": is_a_durable},
                     "world_d_continued": {k: rec_d_full[k] for k in ("completed", "cycles", "failed_attempts")},
                     "improved_vs_fresh_d": full_improved},
            "success_withheld": {
                "world_c_runs": [
                    {k: rec_c1_sw[k] for k in ("completed", "cycles", "failed_attempts")},
                    {k: rec_c2_sw[k] for k in ("completed", "cycles", "failed_attempts")}],
                "patterns_learned_in_c": pats_sw,
                "concepts_after_c2": concepts_sw,
                "durable_tool_concept_positive": sw_concept_positive,
                "episodes_recorded_in_c": len(brain_sw.learner.episodes),
                "world_d_continued": {k: rec_d_sw[k] for k in ("completed", "cycles", "failed_attempts")},
                "improved_vs_fresh_d": sw_improved},
            "fresh_d": {k: rec_d_fresh[k] for k in ("completed", "cycles", "failed_attempts")},
            "passed": bool(full_improved and not sw_improved
                           and full_concept_positive
                           and not sw_concept_positive),
            "note": ("P-C itself never forms in chamber worlds (needs "
                     ">= 3 successes across >= 2 distinct situations); "
                     "the success-driven knowledge measured here is the "
                     "C-1 durable_tool concept positive plus the P-A/P-B "
                     "contrast and planner failure rates. With successes "
                     "withheld, none of it forms.")}


# ------------------------------------------------------------------ acceptance (SS8)
def run_acceptance_experiment() -> dict:
    """Design SS8: the six-step acceptance experiment. (1) fresh brain
    runs C and completes it; (2) during C it learns a reusable pattern
    (``pattern_learned`` event + pattern in the library; the
    consolidation state is recorded -- no P-C/P-D candidate forms in
    chamber worlds, so no ``consolidated`` event fires here, see the
    step note); (3) save + load into a fresh brain -- the pattern,
    the concept and the calibration summaries survive (the grant set
    is NOT persisted by design -- the loader chooses it explicitly,
    so the experiment restores the environment grant on the loaded
    brain); (4) the loaded brain runs D (structurally different:
    different names, four rooms, decoy/tool roles reassigned); (5) a
    ``pattern_applied`` event fires in D naming the learned pattern;
    (6) continued-D beats fresh-D on failed attempts and cycles, and
    beats the no-learning variant on the same task."""
    steps: dict = {}
    failures = []
    # Step 1: fresh brain runs C.
    rec_c, brain_c = run_world(WORLD_C, _full_factory)
    steps["1_fresh_brain_runs_C"] = {
        "completed": rec_c["completed"], "cycles": rec_c["cycles"],
        "failed_attempts": rec_c["failed_attempts"],
        "stop_reason": rec_c["stop_reason"],
        "passed": bool(rec_c["completed"])}
    if not rec_c["completed"]:
        failures.append("step 1: fresh brain did not complete C")
    # Step 2: learns a reusable pattern.
    learned_events = [e for e in brain_c.events if e["kind"] == "pattern_learned"]
    consolidated_events = [e for e in brain_c.events if e["kind"] == "consolidated"]
    pats = _pattern_kinds(brain_c)
    consolidation = _consolidation_state(brain_c)
    concepts_c = _concept_state(brain_c)
    steps["2_learns_reusable_pattern"] = {
        "pattern_learned_events": len(learned_events),
        "patterns_in_library": pats["count"],
        "pattern_kinds": sorted(pats["by_kind"]),
        "pattern_ids": pats["ids"],
        "consolidated_events": len(consolidated_events),
        "consolidation_state": consolidation,
        "concepts_after_c": concepts_c,
        "consolidation_note": ("No 'consolidated' event fired and the "
                               "queue is empty: no P-C/P-D candidate "
                               "forms in chamber worlds (P-C needs >= 3 "
                               "successes across >= 2 distinct situations; "
                               "P-D needs one action failing >= 2 times "
                               "and succeeding). The reusable pattern "
                               "learned here is P-A/P-B; the consolidation "
                               "leg is not exercised by this task family."),
        "passed": bool(learned_events and pats["count"] > 0)}
    if not steps["2_learns_reusable_pattern"]["passed"]:
        failures.append("step 2: no pattern learned in C")
    learned_ids = set(pats["ids"])
    # Step 3: save + load; pattern and concept survive.
    cal_before = brain_c.calibration.total_records() \
        if hasattr(brain_c.calibration, "total_records") else None
    with tempfile.TemporaryDirectory() as folder:
        brain_c.save(folder)
        loaded = Brain()
        loaded.load(folder)
    # Grants are deliberately NOT persisted (Brain.save docstring):
    # the loader chooses the authorization set explicitly.
    loaded.granted = {"compute", "environment"}
    pats_after = _pattern_kinds(loaded)
    concepts_after = _concept_state(loaded)
    cal_after = loaded.calibration.total_records() \
        if hasattr(loaded.calibration, "total_records") else None
    concepts_preserved = (concepts_c == concepts_after)
    steps["3_retained_across_save_load"] = {
        "patterns_before_save": pats["count"],
        "patterns_after_load": pats_after["count"],
        "pattern_ids_preserved": learned_ids == set(pats_after["ids"]),
        "episodes_before_save": len(brain_c.learner.episodes),
        "episodes_after_load": len(loaded.learner.episodes),
        "episodes_preserved": (len(loaded.learner.episodes)
                               == len(brain_c.learner.episodes)),
        "concepts_before_save": concepts_c,
        "concepts_after_load": concepts_after,
        "concepts_preserved": concepts_preserved,
        "calibration_records_before_save": cal_before,
        "calibration_records_after_load": cal_after,
        "calibration_preserved": cal_before == cal_after,
        "calibration_note": ("Calibration summaries persist (met/total "
                             "per pair); after a load the tracker's ECE "
                             "uses the per-pair aggregate fallback, which "
                             "reads higher than the exact raw-pair ECE -- "
                             "same records, different estimator."),
        "load_warnings": list(loaded.load_warnings),
        "passed": bool(learned_ids == set(pats_after["ids"])
                       and len(loaded.learner.episodes)
                       == len(brain_c.learner.episodes)
                       and concepts_preserved
                       and cal_before == cal_after)}
    if not steps["3_retained_across_save_load"]["passed"]:
        failures.append("step 3: pattern/episodes/concepts/calibration did not survive save/load")
    # Step 4: loaded brain runs D.
    entry = prepare_brain_for_new_world(loaded)
    ev0 = len(loaded.events)
    rec_d_cont, brain_d = run_world(WORLD_D, _full_factory, brain=loaded)
    steps["4_loaded_brain_runs_D"] = {
        "completed": rec_d_cont["completed"], "cycles": rec_d_cont["cycles"],
        "failed_attempts": rec_d_cont["failed_attempts"],
        "stop_reason": rec_d_cont["stop_reason"],
        "entry_protocol": entry["method"],
        "patterns_intact_on_entry": entry["patterns_intact"],
        "passed": bool(rec_d_cont["completed"])}
    if not rec_d_cont["completed"]:
        failures.append("step 4: loaded brain did not complete D")
    # Step 5: pattern_applied in D naming the learned pattern.
    applied = _applied_pattern_kinds(brain_d, ev0)
    applied_ids = set()
    for p in brain_d.patterns.patterns():
        for e in applied:
            if p.id[:12] in e["detail"]:
                applied_ids.add(p.id)
    names_learned = bool(applied_ids & learned_ids)
    steps["5_pattern_applied_in_D"] = {
        "pattern_applied_events": len(applied),
        "applied_events": applied,
        "applied_pattern_ids": sorted(applied_ids),
        "names_a_pattern_learned_in_C": names_learned,
        "passed": bool(applied and names_learned)}
    if not steps["5_pattern_applied_in_D"]["passed"]:
        failures.append("step 5: no pattern_applied event in D naming a C pattern")
    # Step 6: continued-D beats fresh-D and no-learning.
    rec_d_fresh, _ = run_world(WORLD_D, _full_factory)
    rec_c_nl, brain_nl = run_world(WORLD_C, _no_learning_factory)
    prepare_brain_for_new_world(brain_nl)
    rec_d_nl, _ = run_world(WORLD_D, _no_learning_factory, brain=brain_nl)
    beats_fresh = bool(rec_d_cont["completed"] and rec_d_fresh["completed"]
                       and rec_d_cont["failed_attempts"] < rec_d_fresh["failed_attempts"]
                       and rec_d_cont["cycles"] <= rec_d_fresh["cycles"])
    beats_no_learning = bool(
        rec_d_cont["completed"] and rec_d_nl["completed"]
        and (rec_d_cont["failed_attempts"], rec_d_cont["cycles"])
        < (rec_d_nl["failed_attempts"], rec_d_nl["cycles"]))
    steps["6_improves_vs_baselines"] = {
        "continued_d": {k: rec_d_cont[k] for k in ("completed", "cycles", "failed_attempts")},
        "fresh_d": {k: rec_d_fresh[k] for k in ("completed", "cycles", "failed_attempts")},
        "no_learning_continued_d": {k: rec_d_nl[k] for k in ("completed", "cycles", "failed_attempts")},
        "beats_fresh_d_on_failures_and_cycles": beats_fresh,
        "beats_no_learning": beats_no_learning,
        "passed": bool(beats_fresh and beats_no_learning)}
    if not steps["6_improves_vs_baselines"]["passed"]:
        failures.append("step 6: continued D did not beat fresh-D and no-learning")
    passed = all(s["passed"] for s in steps.values())
    return {"experiment": "phase6-six-step-acceptance",
            "steps": steps, "passed": passed, "failures": failures}


# ------------------------------------------------------------------ retention / regression
def run_retention_experiment() -> dict:
    """Retention after restart (full variant): train on C, save, load
    into a fresh brain (grant restored explicitly -- grants are not
    persisted by design), check patterns + episodes + concepts +
    calibration survive, run D with the loaded brain vs fresh D."""
    rec_c, brain = run_world(WORLD_C, _full_factory)
    pats_before = _pattern_kinds(brain)
    eps_before = len(brain.learner.episodes)
    concepts_before = _concept_state(brain)
    cal_before = brain.calibration.total_records() \
        if hasattr(brain.calibration, "total_records") else None
    with tempfile.TemporaryDirectory() as folder:
        brain.save(folder)
        loaded = Brain()
        loaded.load(folder)
    loaded.granted = {"compute", "environment"}
    pats_after = _pattern_kinds(loaded)
    concepts_after = _concept_state(loaded)
    cal_after = loaded.calibration.total_records() \
        if hasattr(loaded.calibration, "total_records") else None
    # Snapshot the post-load counts HERE, before the D run below
    # adds its own episodes (measuring after D would count 9 C + 9 D
    # episodes and falsely report non-preservation).
    eps_after = len(loaded.learner.episodes)
    prepare_brain_for_new_world(loaded)
    rec_d, _ = run_world(WORLD_D, _full_factory, brain=loaded)
    rec_d_fresh, _ = run_world(WORLD_D, _full_factory)
    return {"criterion": ("Patterns + episodes + concepts + calibration "
                          "survive save/load; the loaded brain still "
                          "transfers to D."),
            "world_c": {k: rec_c[k] for k in ("completed", "cycles", "failed_attempts")},
            "patterns_before_save": pats_before["count"],
            "patterns_after_load": pats_after["count"],
            "pattern_ids_preserved": (set(pats_before["ids"]) == set(pats_after["ids"])),
            "episodes_before_save": eps_before,
            "episodes_after_load": eps_after,
            "episodes_preserved": eps_after == eps_before,
            "concepts_before_save": concepts_before,
            "concepts_after_load": concepts_after,
            "concepts_preserved": concepts_before == concepts_after,
            "calibration_records_before_save": cal_before,
            "calibration_records_after_load": cal_after,
            "calibration_preserved": cal_before == cal_after,
            "loaded_brain_d": {k: rec_d[k] for k in ("completed", "cycles", "failed_attempts")},
            "fresh_d": {k: rec_d_fresh[k] for k in ("completed", "cycles", "failed_attempts")},
            "retained_transfer": bool(
                rec_d["completed"] and rec_d_fresh["completed"]
                and rec_d["failed_attempts"] < rec_d_fresh["failed_attempts"]),
            "load_warnings": list(loaded.load_warnings),
            "note": ("Grants are deliberately NOT persisted by "
                     "Brain.save; the loaded brain's grant set was "
                     "restored explicitly by the experiment.")}


# ------------------------------------------------------------------ main evaluation
def run_learning_evaluation(record_path: str | None = DEFAULT_RECORD) -> dict:
    """Run the full Phase 6 learning evaluation: all six variants on
    the C1-C5 / D1-D5 task set, retention, regression, and the SS7.1-
    SS7.5 sub-experiments plus the SS8 acceptance experiment. Writes
    the record to ``record_path`` (default
    docs/phase6-learning-record.json) unless ``record_path`` is None,
    and returns the record dict."""
    started = time.monotonic()
    failures: list[str] = []
    factories = make_variant_factories()
    variants: dict = {}
    for name, factory in factories.items():
        baseline_c, _ = run_world(WORLD_C, factory)
        carried = None
        per_task: dict = {}
        for task_id, spec in TASK_SET:
            if name == "fresh":
                rec, _brain = run_world(spec, factory)
                fresh_rec = rec
                transfer = None
            else:
                if carried is None:
                    rec, carried = run_world(spec, factory)
                else:
                    prepare_brain_for_new_world(carried)
                    rec, carried = run_world(spec, factory, brain=carried)
                fresh_rec, _ = run_world(spec, factory)
                transfer = bool(rec["completed"] and fresh_rec["completed"]
                                and rec["failed_attempts"] < fresh_rec["failed_attempts"])
            per_task[task_id] = {
                "completed": rec["completed"],
                "stop_reason": rec["stop_reason"],
                "cycles": rec["cycles"],
                "failed_attempts": rec["failed_attempts"],
                "recovery_after_reset": bool(
                    rec["mid_task_reset_observed"] and rec["completed"]),
                "mid_task_reset_observed": rec["mid_task_reset_observed"],
                "transfer_vs_fresh": transfer,
                "fresh_comparison": {k: fresh_rec[k] for k in
                                     ("completed", "cycles", "failed_attempts")},
                "patterns_after_task": len(carried.patterns.patterns())
                if carried is not None else None,
                "learner_episodes_after_task": (
                    len(carried.learner.episodes) if carried is not None else None),
            }
        summary = _summarise_runs([
            {"completed": t["completed"], "cycles": t["cycles"],
             "failed_attempts": t["failed_attempts"]}
            for t in per_task.values()])
        transfer_tasks = [t for t in per_task.values()
                          if t["transfer_vs_fresh"] is not None]
        summary["transfer_tasks"] = (
            f"{sum(1 for t in transfer_tasks if t['transfer_vs_fresh'])}"
            f"/{len(transfer_tasks)}")
        # Regression / overfitting: base world C re-run with the
        # trained carried brain vs the same variant's fresh-C baseline.
        regression = None
        if name != "fresh" and carried is not None:
            prepare_brain_for_new_world(carried)
            rec_re, _ = run_world(WORLD_C, factory, brain=carried)
            regression = {
                "original_task": "chamber-world-C (base)",
                "fresh_baseline": {k: baseline_c[k] for k in
                                   ("completed", "cycles", "failed_attempts")},
                "after_learning_new_tasks": {k: rec_re[k] for k in
                                             ("completed", "cycles", "failed_attempts")},
                "degraded": bool(
                    rec_re["completed"] and baseline_c["completed"]
                    and (rec_re["failed_attempts"], rec_re["cycles"])
                    > (baseline_c["failed_attempts"], baseline_c["cycles"])),
                "note": ("degraded = re-run needed more failures/cycles "
                         "than the fresh baseline after learning the new "
                         "tasks (overfitting/regression)."),
            }
            if regression["degraded"]:
                failures.append(
                    f"regression: variant {name} degraded on base C "
                    f"after the task set "
                    f"({rec_re['failed_attempts']}/{rec_re['cycles']} vs "
                    f"fresh {baseline_c['failed_attempts']}/{baseline_c['cycles']})")
        mechanism = VARIANT_MECHANISMS[name]
        if name == "no_consolidation":
            # Record the per-run mechanism note (identical by
            # construction) from a probe brain of this variant.
            probe = factory([], {"compute"})
            mechanism += " " + getattr(
                probe, "_no_consolidation_mechanism",
                "mechanism note unavailable")
        variants[name] = {"mechanism": mechanism,
                          "fresh_baseline_C": {k: baseline_c[k] for k in
                                               ("completed", "cycles", "failed_attempts")},
                          "tasks": per_task,
                          "summary": summary,
                          "regression": regression}
    retention = run_retention_experiment()
    if not retention["retained_transfer"]:
        failures.append("retention: loaded brain did not transfer to D "
                        "better than fresh D")
    procedural = run_procedural_experiment()
    if not procedural["passed"]:
        failures.append("procedural: renamed run 2 did not improve with "
                        "learning on / disappear with learning off")
    two_family = run_two_family_transfer()
    if not two_family["passed"]:
        failures.append("two-family transfer: fewer than two pattern "
                        "families transferred to D")
    calibration = run_calibration_experiment()
    if not calibration["passed"]:
        failures.append("calibration: ECE after learning not lower than "
                        "ECE with the 1.0 default")
    retrieval = run_retrieval_experiment()
    cap = run_cap_experiment()
    acceptance = run_acceptance_experiment()
    failures.extend(f"acceptance: {f}" for f in acceptance["failures"])
    success_withheld = run_success_withheld_experiment()
    if not success_withheld["passed"]:
        failures.append("success-based learning: improvement did not "
                        "vanish when success episodes were withheld")
    record = {
        "experiment": "phase6-learning-evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_set": [t for t, _s in TASK_SET],
        "task_set_note": ("Held-out variants C1-C5 (3-room) / D1-D5 "
                          "(4-room), fixed seeds, defined only in "
                          "ab/eval/learning_eval.py. Same structural "
                          "shapes as C/D; names sampled without "
                          "replacement across the set."),
        "variants": variants,
        "retention_after_restart": retention,
        "regression_note": ("Per-variant regression lives under "
                            "variants[<name>]['regression']."),
        "procedural_learning": procedural,
        "two_family_transfer": two_family,
        "calibration": calibration,
        "retrieval_precision": retrieval,
        "cap_measurement": cap,
        "acceptance_experiment": acceptance,
        "success_based_learning": success_withheld,
        "failures": failures,
        "duration_s": round(time.monotonic() - started, 3),
        "notes": [
            ("Means carry sample sizes; where n >= 5, mean +/- sample "
             "std and the range are reported. No confidence interval "
             "is claimed at this scale."),
            ("P-C / P-D patterns never form in chamber worlds (see "
             "two_family_transfer): P-C needs >= 3 successes across "
             ">= 2 distinct situations and P-D needs one action "
             "failing >= 2 times and succeeding, neither of which a "
             "chamber world produces. The consolidator queue stays "
             "empty there. Two-family transfer is measured on P-A / "
             "P-B; P-C / P-D transfer needs a task family with "
             "varying situations per tool use."),
            ("Calibration is measured from the real CalibrationTracker. "
             "On the C1/C2/D1 stream the aggregate ECE/Brier improve "
             "(ECE 0.269 -> 0.141); the unreliable subset is reported "
             "alongside -- see calibration.overall.unreliable_subset. "
             "The aggregate result is stream-dependent (see the "
             "calibration experiment note)."),
            ("The first task per variant (C1) cannot transfer: the "
             "carried brain is fresh on it, so it ties its fresh "
             "comparison by construction; 9/10 is the ceiling for "
             "transfer_tasks."),
            ("The brain's own retriever index carries outcomes, not "
             "effects; retrieval precision feeds the real Retriever "
             "effect-shaped episode dicts built from the trained "
             "brain's real episodes."),
            ("Grants are not persisted by Brain.save by design; "
             "retention/acceptance restore the environment grant on "
             "the loaded brain explicitly."),
            ("no_consolidation disables the end-of-cycle pass (P-C/P-D "
             "promotion + concept learning); no_calibration pins the "
             "tracker at 1.0 via the stub in this eval module."),
        ],
    }
    if record_path is not None:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        with open(record_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    return record


# ------------------------------------------------------------------ learning curves (Phase 9, design SS3.4)
# ADDITIVE ONLY (Phase 9): no existing function above is changed.
# The held-out task sequence is run with increasing experience
# (0, 1, 2, 4, 8 prior tasks) and cycles / failed attempts / success
# are recorded per experience level as a curve (a list of points with
# mean +/- std where n permits), not just endpoints.
LEARNING_CURVE_LEVELS = (0, 1, 2, 4, 8)
# The held-out tasks are the LAST three of TASK_SET (D3, D4, D5);
# experience tasks are drawn from the remaining seven, so no held-out
# task is ever also an experience task. That caps the realisable
# experience at 7 distinct tasks: level 8 therefore runs 7 prior
# tasks, recorded honestly as ``n_prior_tasks`` (requested 8).
LEARNING_CURVE_HELD_OUT_IDS = ("D3", "D4", "D5")


def run_learning_curves(levels=LEARNING_CURVE_LEVELS,
                        held_out_ids=LEARNING_CURVE_HELD_OUT_IDS,
                        seed: int = 0) -> dict:
    """Phase 9 SS3.4: learning curves over the held-out task sequence.

    For each experience level, one carried brain per held-out task is
    trained on that many prior tasks (in TASK_SET order, transient
    cleared between worlds exactly as ``run_learning_evaluation``
    does), then run once on its held-out task. Per level the curve
    point reports n, success rate, and cycles / failed attempts as
    mean +/- sample std over the held-out runs (n = number of
    held-out tasks, so std is reported where n >= 2; no confidence
    interval is claimed at this scale). ``seed`` is recorded; the
    task order itself is the fixed TASK_SET order, so runs with the
    same seed reproduce the same numbers. No existing learning-eval
    behaviour or record format is changed by this function.
    """
    started = time.monotonic()
    factory = _full_factory
    by_id = dict(TASK_SET)
    held_out = [(tid, by_id[tid]) for tid in held_out_ids if tid in by_id]
    pool = [(tid, spec) for tid, spec in TASK_SET
            if tid not in {t for t, _s in held_out}]
    points: list[dict] = []
    for level in levels:
        n_prior = min(int(level), len(pool))
        experience = pool[:n_prior]
        runs: list[dict] = []
        for held_id, held_spec in held_out:
            brain = None
            for _exp_id, exp_spec in experience:
                if brain is None:
                    _rec, brain = run_world(exp_spec, factory)
                else:
                    prepare_brain_for_new_world(brain)
                    _rec, brain = run_world(exp_spec, factory, brain=brain)
            if brain is None:
                rec, _brain = run_world(held_spec, factory)
            else:
                prepare_brain_for_new_world(brain)
                rec, _brain = run_world(held_spec, factory, brain=brain)
            runs.append({"task": held_id,
                         "completed": bool(rec["completed"]),
                         "cycles": rec["cycles"],
                         "failed_attempts": rec["failed_attempts"]})
        done = [r for r in runs if r["completed"]]
        points.append({
            "experience_level_requested": int(level),
            "n_prior_tasks": n_prior,
            "prior_task_ids": [t for t, _s in experience],
            "held_out_task_ids": [t for t, _s in held_out],
            "n": len(runs),
            "success_rate": (round(len(done) / len(runs), 3)
                             if runs else None),
            "n_completed": len(done),
            "cycles": _mean_std([r["cycles"] for r in done]),
            "failed_attempts": _mean_std([r["failed_attempts"] for r in done]),
            "runs": runs,
        })
    return {
        "experiment": "phase9-learning-curves",
        "seed": int(seed),
        "levels_requested": list(levels),
        "held_out_task_ids": [t for t, _s in held_out],
        "note": ("One carried brain per (level, held-out task): the "
                 "brain trains on the level's prior tasks, then runs "
                 "its held-out task once. Level 8 requests 8 prior "
                 "tasks but only 7 distinct non-held-out tasks exist, "
                 "so n_prior_tasks is 7 there -- recorded, not "
                 "rounded up. Means carry n; std is the sample std "
                 "where n >= 2, None otherwise; no confidence "
                 "interval is claimed at this scale."),
        "points": points,
        "duration_s": round(time.monotonic() - started, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 learning evaluation")
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help="path of the JSON record to write")
    args = parser.parse_args(argv)
    try:
        record = run_learning_evaluation(record_path=args.record)
    except Exception as exc:                       # harness error, not a brain failure
        print(f"learning evaluation harness error: {type(exc).__name__}: {exc}")
        return 1
    for name, v in record["variants"].items():
        s = v["summary"]
        print(f"{name}: success {s['success_rate']} "
              f"({s['n_completed']}/{s['n_tasks']}), "
              f"cycles mean {s['cycles']['mean']} "
              f"(std {s['cycles']['std']}, range "
              f"{s['cycles']['min']}-{s['cycles']['max']}), "
              f"failed attempts mean {s['failed_attempts']['mean']} "
              f"(std {s['failed_attempts']['std']}), "
              f"transfer {s['transfer_tasks']}")
    acc = record["acceptance_experiment"]
    print(f"acceptance experiment: {'PASS' if acc['passed'] else 'FAIL'}; "
          f"failures recorded: {record['failures']}")
    print(f"record written to {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
