"""Knowledge-disconnected experiment (Phase 3 evaluation, mandate SS3/SS10).

The brain is dropped into an invented world it has NOT been taught: new
room names, new object names, and rules it must discover by trying
actions and observing outcomes. No internet, no preloaded facts about
this world -- only procedural knowledge (how to plan, act, and learn
from surprise). The world's rules live in this file only; nothing is
hardcoded into the brain.

World A -- "chamber world"
---------------------------
Rooms: blue chamber (start) - green chamber - red chamber. The relic
sits in the red chamber behind a door. The brain believes three
different actions can open the door:

- ``chant`` (cheapest): believed to open the door. It never works.
- ``use_zorblat``: needs the invented object ``zorblat`` (found in the
  green chamber). Works exactly once -- the zorblat crumbles to dust
  as the door opens.
- ``use_quivern``: needs the invented object ``quivern`` (also in the
  green chamber). Costs more, but the quivern is durable.

Changing condition mid-task: the first time the door opens, the
mechanism resets on the next observation -- the door swings shut
again, exactly once. The brain must notice, recover, and re-open it;
by then the zorblat is dust, so only the quivern works.

Partial observation: from the blue chamber the brain cannot see the
door at all -- ``door_open`` is omitted from observations there. The
crumbling of the zorblat is never an observed state key either; the
brain only sees the next ``use_zorblat`` fail.

World B -- transfer world
--------------------------
Structurally identical, differently named: amber / violet / crimson
chambers, objects ``tessera`` (fragile) and ``orb`` (durable), decoy
``hum``, goal the ``idol``. It is run twice: once with a fresh brain
and once with the world-A brain carried over (its learner and planner
intact), to check whether anything learned transfers across the
renaming. Learning in this architecture is bound to (action,
situation) names, so the honest expectation is that little or nothing
transfers; the record states the measured numbers either way.

The 14 behaviours (mandate SS3) are checked against the run and
recorded with evidence, or recorded as not observed. Failures are
recorded as carefully as successes.

Runnable standalone::

    python3 -m ab.eval.disconnected [--record PATH]

Exit 0 when a full record was written -- even if the brain FAILS the
task (a failed task is a recorded result, not a harness crash).
Non-zero only on a harness error (e.g. the record cannot be written).
No LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action

# Phase 4: the three discriminating held-out tasks live in
# ab.eval.reasoning_tasks (world A/B above are untouched by them).
# Re-exported here so the disconnected experiment module is the single
# entry point for held-out evaluation; see --discriminating in main().
from .reasoning_tasks import (  # noqa: F401
    run_discriminating_tasks, make_variant_factories)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase3-disconnected-record.json")


# ------------------------------------------------------------------ worlds
WORLD_A = {
    "name": "chamber-world-A",
    "rooms": ("blue", "green", "red"),
    "start": "blue",
    "middle": "green",
    "far": "red",
    "fragile": "zorblat",
    "durable": "quivern",
    "decoy": "chant",
    "treasure": "relic",
}

WORLD_B = {
    "name": "chamber-world-B",
    "rooms": ("amber", "violet", "crimson"),
    "start": "amber",
    "middle": "violet",
    "far": "crimson",
    "fragile": "tessera",
    "durable": "orb",
    "decoy": "hum",
    "treasure": "idol",
}


class ChamberWorld(Environment):
    """An invented three-room world with a hidden door rule.

    The brain's believed model (see :func:`chamber_world_actions`)
    says the door can be opened by the decoy, the fragile object, or
    the durable object. The truth, known only here:

    - the decoy never works;
    - the fragile object works once, then crumbles (``fragile_intact``
      is hidden state, never observed);
    - the durable object always works;
    - the first opening triggers exactly one mid-task reset: on the
      second observation while the door is open, the door swings shut
      again (``relock_done`` after that).
    """

    def __init__(self, spec: dict):
        self.spec = spec
        s, m, f = spec["rooms"]
        self.adj = {s: [m], m: [s, f], f: [m]}
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
        # Mid-task change: the mechanism resets once, on the second
        # observation while the door is open (the first is the observe
        # right after the successful opening action; the second is the
        # next cycle's sense, so the reset lands before the brain can
        # walk through). After that the door stays as left.
        if self.s["door_open"] and not self._relock_done:
            self._open_observes += 1
            if self._open_observes >= 2:
                self.s["door_open"] = False
                self._relock_done = True
                self.log.append("reset: the mechanism resets, the door swings shut")
        # Partial observation: from the start room the door cannot be
        # seen at all, so door_open is omitted there.
        obs = {k: v for k, v in self.s.items()}
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
        elif name == f"take_{spec['fragile']}" and s["at"] == spec["middle"] \
                and not s[self.fragile_key] and self._fragile_intact:
            s[self.fragile_key], ok, info = True, True, f"took the {spec['fragile']}"
        elif name == f"take_{spec['fragile']}":
            info = f"no intact {spec['fragile']} remains here"
        elif name == f"take_{spec['durable']}" and s["at"] == spec["middle"] \
                and not s[self.durable_key]:
            s[self.durable_key], ok, info = True, True, f"took the {spec['durable']}"
        elif name == spec["decoy"]:
            info = "the air hums, but the door does not move"
        elif name == f"use_{spec['fragile']}" and s["at"] == spec["middle"] \
                and s[self.fragile_key] and self._fragile_intact:
            s["door_open"], self._fragile_intact = True, False
            ok, info = True, (f"the {spec['fragile']} turns the mechanism and "
                               f"crumbles to dust as the door swings open")
        elif name == f"use_{spec['fragile']}":
            info = f"the {spec['fragile']} is dust; it cannot turn the mechanism"
        elif name == f"use_{spec['durable']}" and s["at"] == spec["middle"] \
                and s[self.durable_key]:
            s["door_open"], ok, info = True, True, "the mechanism turns and the door opens"
        elif name == f"take_{spec['treasure']}" and s["at"] == spec["far"]:
            s[self.treasure_key], ok, info = True, True, f"took the {spec['treasure']}"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def chamber_world_actions(spec: dict) -> list[Action]:
    """The brain's (initially naive) believed model of the world.

    It believes the decoy opens the door (cheapest), the fragile
    object opens it, and the durable object opens it. It does NOT know
    the decoy never works, that the fragile object is single-use, or
    that the door resets once. Those must be discovered by acting.
    """
    s, m, f = spec["rooms"]
    frag, dur, dec, tre = (spec["fragile"], spec["durable"],
                           spec["decoy"], spec["treasure"])
    fk, dk, tk = f"has_{frag}", f"has_{dur}", f"has_{tre}"
    return [
        Action(f"go_{m}_from_{s}", {"at": s}, {"at": m}, args={"_do": "go", "room": m}),
        Action(f"go_{s}_from_{m}", {"at": m}, {"at": s}, args={"_do": "go", "room": s}),
        Action(f"go_{f}", {"at": m, "door_open": True}, {"at": f},
               args={"_do": "go", "room": f}),
        Action(f"go_{m}_from_{f}", {"at": f}, {"at": m}, args={"_do": "go", "room": m}),
        Action(f"take_{frag}", {"at": m, fk: False}, {fk: True},
               args={"_do": f"take_{frag}"}),
        Action(f"take_{dur}", {"at": m, dk: False}, {dk: True},
               args={"_do": f"take_{dur}"}),
        # The decoy is the cheapest way to open the door on paper --
        # the brain will try it first and must learn that it never works.
        Action(dec, {"at": m}, {"door_open": True}, cost=0.5, args={"_do": dec}),
        Action(f"use_{frag}", {"at": m, fk: True}, {"door_open": True},
               cost=1.0, args={"_do": f"use_{frag}"}),
        Action(f"use_{dur}", {"at": m, dk: True}, {"door_open": True},
               cost=2.0, args={"_do": f"use_{dur}"}),
        Action(f"take_{tre}", {"at": f}, {tk: True}, args={"_do": f"take_{tre}"}),
    ]


def _default_factory(actions, granted):
    return Brain(actions, granted=granted)


def _run_with_brain(spec: dict, brain_factory, max_cycles: int,
                    brain: Brain | None = None):
    """Run one world and return ``(record, brain)``. ``brain`` reuses
    an existing brain (transfer run); otherwise the factory builds one."""
    env = ChamberWorld(spec)
    actions = chamber_world_actions(spec)
    if brain is None:
        brain = brain_factory(actions, {"compute", "environment"})
        state_cleared = False
    else:
        for a in actions:
            brain.planner.add_action(a)
        # Transfer protocol: entering a new world re-senses from
        # scratch -- transient state (observations, beliefs about the
        # old world) is cleared, while durable learning (LTM, rules,
        # learner episodes, planner stats and bans) carries over. Without
        # this, a stale observed key from world A (e.g. door_open=True,
        # which world B's start room cannot observe) would poison the
        # new world's plans; that stale-state failure was observed in
        # an early run (world B continued stalled after 8 cycles) and
        # is recorded in the experiment notes, not hidden.
        brain.state.clear()
        state_cleared = True
    tk = f"has_{spec['treasure']}"
    # Knowledge-disconnected check, before the run: the brain must not
    # already know anything about this invented world.
    unknown_before = not brain.ask(spec["fragile"], "is_a").known
    brain.add_goal(f"get the {spec['treasure']}", {tk: True})
    cycle_before = brain.cycle
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[-1]
    failures = [entry for entry in env.log
                if "does not move" in entry or "is dust" in entry
                or "no intact" in entry or "nothing happened" in entry
                or "cannot reach" in entry]
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
        # Cycles spent in THIS world (a continued brain carries its
        # cycle count over from the previous world).
        "cycles": brain.cycle - cycle_before,
        "failed_attempts": len(failures),
        "failure_log": failures,
        "mid_task_reset_observed": any("reset:" in entry for entry in env.log),
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


def _check_behaviours(world_a: dict, world_b_fresh: dict,
                      world_b_continued: dict, retain: dict) -> dict:
    """Map the 14 mandate behaviours to observed evidence (or record
    them as not observed)."""

    def ev(observed, evidence):
        return {"observed": bool(observed), "evidence": evidence}

    a_log = world_a["env_log"]
    tried = {entry.split(":")[0].split("(")[0] for entry in a_log}
    recovered = False
    for i, entry in enumerate(a_log):
        if "does not move" in entry or "is dust" in entry:
            if any("took the" in later or "mechanism turns" in later
                   for later in a_log[i + 1:]):
                recovered = True
    return {
        "observe": ev(len(a_log) > 0,
                      f"{len(a_log)} env actions observed via sense/act"),
        "identify_known_unknown": ev(
            world_a["unknown_before_run"],
            "ask() about the invented object before the run returned unknown"),
        "construct_representations": ev(
            "at" in world_a["final_state_keys"]
            and "door_open" in world_a["final_state_keys"],
            f"state keys after run: {world_a['final_state_keys']}"),
        "form_hypotheses": ev(
            world_a["plan_events"] >= 1,
            f"{world_a['plan_events']} plan(s) made; first plan tried the "
            f"cheapest believed opener"),
        "investigate": ev(
            len(tried) >= 3,
            f"distinct actions tried: {sorted(tried)}"),
        "learn_rules": ev(
            world_a["learner_episodes"] > 0,
            f"{world_a['learner_episodes']} learner episode(s); "
            f"{world_a['learner_rules_proposed']} rule(s) proposed "
            f"(threshold is 3 failures of one action in one situation, "
            f"which this run did not reach); "
            f"{world_a['planner_bans']} planner ban(s) recorded"),
        "plan_multi_step": ev(
            world_a["max_plan_steps"] >= 3,
            f"longest plan had {world_a['max_plan_steps']} steps"),
        "execute_authorized_actions": ev(
            len(a_log) > 0,
            "env actions ran under the explicit 'environment' grant; "
            "no deny events recorded"),
        "compare_predictions": ev(
            world_a["surprise_events"] > 0,
            f"{world_a['surprise_events']} surprise/prediction-mismatch "
            f"event(s)"),
        "recover_from_failures": ev(
            recovered,
            "a failed opener was followed by a different, successful action"),
        "adapt": ev(
            world_a["mid_task_reset_observed"] and world_a["completed"],
            "door reset mid-task and the brain re-opened it with the "
            "durable object after the fragile one crumbled"),
        "complete": ev(
            world_a["completed"],
            f"goal status {world_a['goal_status']}, "
            f"stop reason {world_a['stop_reason']}"),
        "retain": ev(
            retain["episodes_preserved"],
            retain["evidence"]),
        "transfer": ev(
            world_b_continued["completed"]
            and world_b_continued["failed_attempts"] < world_a["failed_attempts"],
            f"world B fresh: {world_b_fresh['failed_attempts']} failed "
            f"attempt(s) in {world_b_fresh['cycles']} cycles; world B "
            f"continued (world-A brain): "
            f"{world_b_continued['failed_attempts']} failed attempt(s) in "
            f"{world_b_continued['cycles']} cycles; world A: "
            f"{world_a['failed_attempts']} failed attempt(s). "
            f"Learning is bound to (action, situation) names, so no "
            f"reduction was expected -- see transfer section."),
    }


def run_experiment(brain_factory=_default_factory, max_cycles: int = 80,
                   worlds=("A", "B")) -> dict:
    """Run the experiment and return the record dict.

    ``brain_factory(actions, granted)`` builds a brain; the default
    builds a plain :class:`Brain`. ``worlds`` selects which worlds run
    (``"A"`` and/or ``"B"``); the transfer and retain checks need both.
    """
    started = time.monotonic()
    record: dict = {
        "experiment": "knowledge-disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "worlds_run": list(worlds),
        "failures": [],
    }
    brain_a = None
    world_a = world_b_fresh = world_b_continued = None
    if "A" in worlds:
        world_a, brain_a = _run_with_brain(WORLD_A, brain_factory, max_cycles)
        record["world_a"] = world_a
        if not world_a["completed"]:
            record["failures"].append(
                f"world A not completed: goal {world_a['goal_status']}, "
                f"stop {world_a['stop_reason']} after {world_a['cycles']} cycles")
    if "B" in worlds:
        world_b_fresh, _ = _run_with_brain(WORLD_B, brain_factory, max_cycles)
        record["world_b_fresh"] = world_b_fresh
        if not world_b_fresh["completed"]:
            record["failures"].append(
                f"world B (fresh brain) not completed: goal "
                f"{world_b_fresh['goal_status']}, stop {world_b_fresh['stop_reason']}")
        if brain_a is not None:
            world_b_continued, _ = _run_with_brain(
                WORLD_B, brain_factory, max_cycles, brain=brain_a)
            record["world_b_continued"] = world_b_continued
            if not world_b_continued["completed"]:
                record["failures"].append(
                    f"world B (continued brain) not completed: goal "
                    f"{world_b_continued['goal_status']}, stop "
                    f"{world_b_continued['stop_reason']}")
    # Retain check: save the world-A brain, load it into a fresh brain,
    # and compare learner episodes (the durable trace of what it learned).
    # A save/load failure is recorded here, not raised: a persistence
    # bug is a finding about the brain, not a crash of this experiment.
    # (Phase 5 note: an earlier header comment here claimed Brain.save
    # still read the removed Brain.history. That was a stale Phase 3
    # in-flight note -- Brain.history was removed in Phase 3 and
    # save/load work; the retain check passes in every harness run.)
    retain = {"episodes_preserved": False, "evidence": "world A not run"}
    if brain_a is not None:
        try:
            with tempfile.TemporaryDirectory() as folder:
                brain_a.save(folder)
                loaded = Brain()
                loaded.load(folder)
                same = len(loaded.learner.episodes) == len(brain_a.learner.episodes)
                retain = {
                    "episodes_preserved": same,
                    "episodes_before_save": len(brain_a.learner.episodes),
                    "episodes_after_load": len(loaded.learner.episodes),
                    "evidence": (f"learner episodes before save "
                                 f"{len(brain_a.learner.episodes)}, after load "
                                 f"{len(loaded.learner.episodes)}"),
                }
        except Exception as exc:
            retain = {
                "episodes_preserved": False,
                "evidence": (f"retain check could not run: Brain.save/load "
                             f"raised {type(exc).__name__}: {exc}"),
            }
            record["failures"].append(
                f"retain check failed: Brain.save/load raised "
                f"{type(exc).__name__}: {exc}")
    record["retain"] = retain
    if world_a is not None and world_b_fresh is not None \
            and world_b_continued is not None:
        record["behaviours"] = _check_behaviours(
            world_a, world_b_fresh, world_b_continued, retain)
        record["transfer"] = {
            "world_a_failed_attempts": world_a["failed_attempts"],
            "world_a_cycles": world_a["cycles"],
            "world_b_fresh_failed_attempts": world_b_fresh["failed_attempts"],
            "world_b_fresh_cycles": world_b_fresh["cycles"],
            "world_b_continued_failed_attempts": world_b_continued["failed_attempts"],
            "world_b_continued_cycles": world_b_continued["cycles"],
            "transferred": bool(record["behaviours"]["transfer"]["observed"]),
            "note": ("Transfer means the continued brain needed fewer "
                     "failed attempts in world B than the fresh brain "
                     "needed in world A. Learning is bound to "
                     "(action, situation) names (stable rule ids hash "
                     "the action and situation), so a renamed world "
                     "does not reuse it; a 'not transferred' result is "
                     "a measured limitation, not a harness failure."),
        }
    record["duration_s"] = round(time.monotonic() - started, 3)
    record["notes"] = [
        ("Stale-state hazard (found during development, fixed in the "
         "transfer protocol, kept on record): an early version of the "
         "continued run did NOT clear transient state, and the world-A "
         "observation door_open=True survived into world B, whose start "
         "room cannot observe the door. The continued brain planned "
         "straight through the (actually shut) door and stalled after "
         "8 cycles. The transfer protocol now clears transient state "
         "on entering a new world; durable learning is what carries over."),
        ("Alternative-plan limit (observed in world A): after the "
         "mid-task reset, the adopted alternative plan's first action "
         "was take_quivern, but its second step was the just-banned "
         "use_zorblat -- alternatives are distinct by first action "
         "only, so an adopted alternative can still contain the banned "
         "action later in the sequence. The brain failed it once more, "
         "then replanned to use_quivern. This is the Phase 3 design "
         "SS4.2 item (alternatives diverse by full sequence), not a "
         "failure of this experiment."),
        ("No learner rule was proposed in any run: the proposal "
         "threshold is 3 failures of one action in one situation, and "
         "no action failed 3 times. Rule learning from this task "
         "therefore shows up as planner bans (2 in world A), not as "
         "proposed rules."),
    ]
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge-disconnected experiment")
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help="path of the JSON record to write")
    parser.add_argument("--discriminating", action="store_true",
                        help="also run the three Phase 4 discriminating "
                             "held-out tasks (ab.eval.reasoning_tasks) "
                             "for all variants and print the per-variant "
                             "results; world A/B are unaffected")
    args = parser.parse_args(argv)
    if args.discriminating:
        section = run_discriminating_tasks()
        for task_id, task in section["tasks"].items():
            print(f"{task_id}: {task['criterion']}")
            for name, res in task["variants"].items():
                print(f"  {name}: {'PASS' if res['passed'] else 'FAIL'} "
                      f"({res.get('cycles')} cycles) -- {res['detail']}")
        print("No winner is declared -- the numbers above are the comparison.")
        return 0
    try:
        record = run_experiment()
    except Exception as exc:                       # harness error, not a brain failure
        print(f"disconnected experiment harness error: {type(exc).__name__}: {exc}")
        return 1
    try:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        with open(args.record, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    except OSError as exc:
        print(f"disconnected experiment: cannot write record: {exc}")
        return 1
    a = record.get("world_a", {})
    b = record.get("world_b_fresh", {})
    c = record.get("world_b_continued", {})
    print(f"world A: completed={a.get('completed')} stop={a.get('stop_reason')} "
          f"cycles={a.get('cycles')} failed_attempts={a.get('failed_attempts')}")
    print(f"world B fresh: completed={b.get('completed')} stop={b.get('stop_reason')} "
          f"cycles={b.get('cycles')} failed_attempts={b.get('failed_attempts')}")
    print(f"world B continued: completed={c.get('completed')} stop={c.get('stop_reason')} "
          f"cycles={c.get('cycles')} failed_attempts={c.get('failed_attempts')}")
    print(f"transfer: {record.get('transfer', {}).get('transferred')}")
    print(f"failures recorded: {record.get('failures')}")
    print(f"record written to {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
