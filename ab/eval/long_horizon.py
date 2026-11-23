"""Phase 9 long-horizon task (design §3.5) and the combined
save/resume scenario (design §4.4).

One task requiring sustained performance over many cycles: a 6-stage
chain — collect 3 items in 3 rooms, use them in sequence at the
forge, goal only met after all stages — with a mid-task interruption
(an urgent percept suspends the goal) and a restart/resume leg (save
mid-way, reload into a fresh brain, complete). Pass = goal completes,
all stages in order, interruption leg and resume leg complete.
Target >= 25 cycles.

The world is eval-only: invented names (``ember_shard``,
``tide_pearl``, ``storm_lens``, ``relic``, the spine/vault rooms)
defined only in this file, never taught to the brain. No LLM, stdlib
only. Results use the consolidated common format:
``{passed, cycles, detail, evidence, failure_cause}``.

Why the plan is 22 steps yet selectable: the Phase 3 feasibility
gate refuses plans with cost > 15 (feasibility < 0.25). Travel
steps cost 0.4 (cheap movement, as in the workshop worlds' 0.3–0.5
costs) and work steps cost 1.0, so the full 22-step plan costs 14.8
(feasibility 0.25) and is selected. The cost assignment is part of
the world model, stated here, not a gate workaround hidden from the
record: ``evidence`` carries the plan cost and feasibility.
"""
from __future__ import annotations

import os
import tempfile

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action
from .reasoning_tasks import (DEDUCTION_FACTS, DEDUCTION_RELATION,
                              DEDUCTION_SUBJECT, DEDUCTION_VALUE,
                              _find_derivation)

# ------------------------------------------------------------------ world
# Spine rooms with one vault off each; the forge sits behind an
# antechamber. Every vault's item sits in a cache that must be opened
# before the item can be taken (one genuine step per item).
ADJACENCY = {
    "lobby": ["gatehouse"],
    "gatehouse": ["lobby", "spine_a"],
    "spine_a": ["gatehouse", "vault_a", "spine_b"],
    "vault_a": ["spine_a"],
    "spine_b": ["spine_a", "vault_b", "spine_c"],
    "vault_b": ["spine_b"],
    "spine_c": ["spine_b", "vault_c", "antechamber"],
    "vault_c": ["spine_c"],
    "antechamber": ["spine_c", "forge"],
    "forge": ["antechamber"],
}
# (item, vault, flag set when the item is used at the forge, in order)
ITEMS = (("ember_shard", "vault_a", "kindled"),
         ("tide_pearl", "vault_b", "quenched"),
         ("storm_lens", "vault_c", "focused"))
GOAL_DESCRIPTION = "forge the relic"
GOAL_TARGET = {"has_relic": True}
# The six stages, in required order, as env-action name prefixes.
STAGES = ("take_ember_shard", "take_tide_pearl", "take_storm_lens",
          "use_ember_shard", "use_tide_pearl", "use_storm_lens")
GO_TRAVEL_COST = 0.4
# Cycles before the interruption is injected: goal selected, plan
# made, first actions executed (position 4 at cycle 7 in the
# reference run) — genuinely mid-task, not at the start.
INTERRUPT_AFTER_STEPS = 6
# Steps run on the resumed brain before the save/reload leg.
RESUME_STEPS_BEFORE_SAVE = 2


class LongHorizonWorld(Environment):
    """Ten rooms, three cached items, one forge. The truth, known
    only here: a cache must be opened before its item can be taken;
    items work at the forge only in sequence (ember, then tide, then
    storm — each use requires the previous use's flag); ``assemble``
    works only at the forge with all three flags set."""

    def __init__(self):
        self.s = {"at": "lobby", "has_relic": False}
        for item, _room, flag in ITEMS:
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
            if dest in ADJACENCY[s["at"]]:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name.startswith("open_cache_"):
            item = name[len("open_cache_"):]
            room = next(r for i, r, _f in ITEMS if i == item)
            if s["at"] == room and not s[f"cache_{item}"]:
                s[f"cache_{item}"], ok, info = True, True, f"opened the cache of {item}"
        elif name.startswith("take_"):
            item = name[len("take_"):]
            room = next(r for i, r, _f in ITEMS if i == item)
            if s["at"] == room and s[f"cache_{item}"] and not s[f"has_{item}"]:
                s[f"has_{item}"], ok, info = True, True, f"took the {item}"
        elif name.startswith("use_"):
            item = name[len("use_"):]
            flag = next(f for i, _r, f in ITEMS if i == item)
            idx = [i for i, _r, _f in ITEMS].index(item)
            prev_ok = True if idx == 0 else s[ITEMS[idx - 1][2]]
            if s["at"] == "forge" and s[f"has_{item}"] and prev_ok and not s[flag]:
                s[flag], ok, info = True, True, f"used the {item}"
            else:
                info = f"cannot use {item} here"
        elif name == "assemble" and s["at"] == "forge" and all(s[f] for _i, _r, f in ITEMS):
            s["has_relic"], ok, info = True, True, "assembled the relic"
        elif name == "assemble":
            info = "parts missing for assembly"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def long_horizon_actions() -> list[Action]:
    """The brain's believed model: travel, cache-opening, taking,
    sequential use at the forge, assembly. It does not know the
    caches must be opened first in the world — the planner discovers
    the precondition from the action model itself (the model states
    it); the world enforces the same rule, so no surprise is
    involved. Costs: travel 0.4, work 1.0 (see module docstring)."""
    actions: list[Action] = []
    for room, dests in ADJACENCY.items():
        for dest in dests:
            actions.append(Action(f"go_{dest}_from_{room}", {"at": room},
                                  {"at": dest}, cost=GO_TRAVEL_COST,
                                  args={"_do": "go", "room": dest}))
    for item, room, _flag in ITEMS:
        actions.append(Action(
            f"open_cache_{item}", {"at": room, f"cache_{item}": False},
            {f"cache_{item}": True}, args={"_do": f"open_cache_{item}"}))
        actions.append(Action(
            f"take_{item}",
            {"at": room, f"cache_{item}": True, f"has_{item}": False},
            {f"has_{item}": True}, args={"_do": f"take_{item}"}))
    for idx, (item, _room, flag) in enumerate(ITEMS):
        pre = {"at": "forge", f"has_{item}": True, flag: False}
        if idx > 0:
            pre[ITEMS[idx - 1][2]] = True
        actions.append(Action(f"use_{item}", pre, {flag: True},
                              args={"_do": f"use_{item}"}))
    actions.append(Action(
        "assemble", {"at": "forge", **{f: True for _i, _r, f in ITEMS}},
        {"has_relic": True}, args={"_do": "assemble"}))
    return actions


# ------------------------------------------------------------------ helpers
def _stage_positions(env_log: list[str]) -> dict:
    """First env-log index of each stage action, or None when the
    stage never ran. Order is read off the world's own log, never
    off the brain's plan (the plan says what was intended; the log
    says what happened)."""
    positions: dict = {}
    for stage in STAGES:
        positions[stage] = next(
            (i for i, entry in enumerate(env_log)
             if entry.startswith(stage)), None)
    return positions


def _stages_in_order(positions: dict) -> bool:
    vals = [positions[s] for s in STAGES]
    return all(v is not None for v in vals) and vals == sorted(vals)


# ------------------------------------------------------------------ the task
def run_long_horizon_task(max_cycles: int = 120) -> dict:
    """Run the 6-stage chain with a mid-task interruption and a
    save/restart/resume leg. Returns the consolidated common format.
    A brain failure is a recorded result, never a raise."""
    env = LongHorizonWorld()
    brain = Brain(long_horizon_actions(), granted={"compute", "environment"})
    goal = brain.add_goal(GOAL_DESCRIPTION, dict(GOAL_TARGET))
    evidence: dict = {"stages": STAGES, "interrupt_after_steps": INTERRUPT_AFTER_STEPS,
                      "resume_steps_before_save": RESUME_STEPS_BEFORE_SAVE}
    failure_cause: str | None = None
    try:
        # Leg 1: partial run, then interrupt with an urgent percept.
        for _ in range(INTERRUPT_AFTER_STEPS):
            brain.step(env)
        evidence["cycles_before_interrupt"] = brain.cycle
        evidence["goal_status_before_interrupt"] = goal.status
        if goal.status != "active":
            failure_cause = (f"goal not active after {INTERRUPT_AFTER_STEPS} steps "
                             f"(status {goal.status}); cannot interrupt mid-task")
            return _result(False, brain.cycle, evidence, failure_cause)
        brain.perceive_text("urgent: halt work at once", urgency=0.95)
        brain.step(env)
        interrupted = (goal.status == "pending"
                       and brain.goals.suspended(goal.id) is not None)
        interrupt_events = [e for e in brain.events
                            if e["kind"] == "interrupt"
                            and "urgent_percept" in e["detail"]]
        evidence["interrupted"] = bool(interrupted)
        evidence["interrupt_events"] = len(interrupt_events)
        evidence["cycles_at_interrupt"] = brain.cycle
        if not interrupted:
            failure_cause = "urgent percept did not suspend the active goal"
            return _result(False, brain.cycle, evidence, failure_cause)
        # Leg 2: resume, run a little further, then save.
        if not brain.resume_goal(goal.id):
            failure_cause = "resume_goal refused the suspended goal"
            return _result(False, brain.cycle, evidence, failure_cause)
        evidence["resumed"] = True
        for _ in range(RESUME_STEPS_BEFORE_SAVE):
            brain.step(env)
        evidence["cycles_before_save"] = brain.cycle
        folder = tempfile.mkdtemp(prefix="ab-long-horizon-")
        brain.save(folder)
        evidence["save_folder"] = folder
        evidence["load_warnings_before"] = list(getattr(brain, "load_warnings", []))
        # Leg 3: fresh brain, load, complete.
        resumed = Brain(long_horizon_actions(), granted={"compute", "environment"})
        resumed.load(folder)
        evidence["load_warnings"] = list(resumed.load_warnings)
        evidence["cycles_after_load"] = resumed.cycle
        evidence["goal_status_after_load"] = resumed.goals.all()[-1].status
        stop = resumed.run(env, max_cycles)
        goal2 = resumed.goals.all()[-1]
        evidence["stop_reason"] = stop
        evidence["goal_status"] = goal2.status
        evidence["cycles"] = resumed.cycle
        evidence["env_actions"] = len(env.log)
        positions = _stage_positions(env.log)
        evidence["stage_positions"] = positions
        evidence["stages_in_order"] = _stages_in_order(positions)
        # The env log carries env action names ("go", "take_...", ...),
        # not planner action names: travel steps cost 0.4, work steps 1.0.
        n_go = sum(1 for e in env.log if e.startswith("go"))
        evidence["executed_plan_cost"] = round(
            n_go * GO_TRAVEL_COST + (len(env.log) - n_go) * 1.0, 2)
        evidence["executed_travel_steps"] = n_go
        legs = {
            "goal_completed": goal2.status == "done",
            "stages_in_order": bool(evidence["stages_in_order"]),
            "interruption_suspended_goal": bool(interrupted),
            "resume_leg_completed": goal2.status == "done",
            "cycles_at_least_25": resumed.cycle >= 25,
            "load_clean": resumed.load_warnings == [],
        }
        passed = all(legs.values())
        evidence["legs"] = legs
        if not passed:
            failed = [k for k, v in legs.items() if not v]
            failure_cause = f"legs failed: {failed}"
        return _result(passed, resumed.cycle, evidence, failure_cause)
    except Exception as exc:  # a brain failure is data, not a crash
        return _result(False, brain.cycle, evidence,
                       f"{type(exc).__name__}: {exc}")


def _result(passed: bool, cycles: int, evidence: dict, failure_cause: str | None) -> dict:
    return {
        "passed": passed,
        "cycles": cycles,
        "detail": (
            f"long-horizon chain: goal {evidence.get('goal_status', '?')}, "
            f"stages in order={evidence.get('stages_in_order')}, "
            f"interrupted={evidence.get('interrupted')}, "
            f"cycles={cycles}" if passed else
            f"long-horizon chain FAILED: {failure_cause}"),
        "evidence": evidence,
        "failure_cause": failure_cause,
    }


# ------------------------------------------------------------------ §4.4 unit leg
def run_deduction_save_resume(folder: str | None = None) -> dict:
    """Design §4.4: save a brain mid-deduction-task (the Phase 4
    3-hop deduction setup from ``reasoning_tasks``) and resume it to
    completion. The split is at the natural seam: brain A teaches the
    chain and saves before perceiving/stepping; brain B loads,
    perceives the subject, steps once, and the derivation must be
    there with its 3-step proof. Returns the common format."""
    evidence: dict = {"facts": [f"{s} {r} {v}" for s, r, v in DEDUCTION_FACTS]}
    failure_cause: str | None = None
    try:
        brain = Brain([], granted={"compute"})
        brain.teach_fact(*DEDUCTION_FACTS[0], "teacher")
        brain.teach_fact(*DEDUCTION_FACTS[1], "teacher")
        brain.teach_fact(*DEDUCTION_FACTS[2], "teacher")
        folder = folder or tempfile.mkdtemp(prefix="ab-deduction-resume-")
        brain.save(folder)
        evidence["save_folder"] = folder
        evidence["cycles_before_save"] = brain.cycle
        resumed = Brain([], granted={"compute"})
        resumed.load(folder)
        evidence["load_warnings"] = list(resumed.load_warnings)
        evidence["cycles_after_load"] = resumed.cycle
        resumed.perceive_text(DEDUCTION_SUBJECT)
        resumed.step()
        deriv = _find_derivation(resumed, DEDUCTION_SUBJECT,
                                 DEDUCTION_RELATION, DEDUCTION_VALUE)
        evidence["derived"] = deriv is not None
        evidence["derivation"] = (None if deriv is None else {
            "subject": deriv["subject"], "relation": deriv["relation"],
            "value": deriv["value"], "confidence": deriv["confidence"]})
        evidence["cycles"] = resumed.cycle
        legs = {
            "derived_after_resume": deriv is not None,
            "load_clean": resumed.load_warnings == [],
        }
        passed = all(legs.values())
        evidence["legs"] = legs
        if not passed:
            failure_cause = ("derivation missing after resume"
                             if deriv is None else "load warnings present")
        return {
            "passed": passed,
            "cycles": resumed.cycle,
            "detail": (f"deduction resumed to completion: {DEDUCTION_SUBJECT} "
                       f"{DEDUCTION_RELATION} {DEDUCTION_VALUE} derived at "
                       f"confidence {deriv['confidence']}" if passed else
                       f"deduction save/resume FAILED: {failure_cause}"),
            "evidence": evidence,
            "failure_cause": failure_cause,
        }
    except Exception as exc:
        return {"passed": False, "cycles": 0, "detail": f"task errored: {exc}",
                "evidence": evidence, "failure_cause": f"{type(exc).__name__}: {exc}"}
