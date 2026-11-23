"""Phase 8 evaluation: perception and imagination acceptance experiments
(design docs/phase8-design.md SS4).

Thirteen named experiments, each returning
``{"name", "passed": bool, "evidence": {...}}``. A failed criterion is
a recorded result, never a harness crash: every experiment is wrapped
so one exception cannot take down the rest.

API note (2026-11-21, measured in this tree, not assumed): the Phase 8
Worker 1/2 packages (``ab/perception/``, ``ab/imagination/``,
``ab/body/reading.py``) landed WHILE this module was written -- the
first full run measured the pre-W1/W2 tree (9/13, with image_percept
and specialist_absent failing honestly on absent APIs), and the rerun
below measures the landed tree. Every experiment therefore probes for
the designed public API first (``Brain.imagine`` / ``what_if`` /
``counterfactual`` / ``read_document`` / ``perceive`` /
``Brain(specialist=...)``) and uses it; the fallback to the older
public paths (``ReasoningEngine.simulate``, ``ClaimExtractor`` + LTM)
remains only as defence, and the ``api`` field in each experiment's
evidence says which path actually ran. No evaluation answer is
hardcoded: every number comes from running the brain.

The two task worlds used here live ONLY in this module (like the
workshop worlds in ``ab.eval.execution_eval``): the treasure task
reuses the demo's ``RoomWorld`` (``pick_lock`` never works in reality),
and the second structurally-similar task is a small gem-fetch world
with the same action names and the same ``pick_lock`` trap.

Runnable standalone::

    python3 -m ab.eval.phase8_eval [--record PATH] [--summary PATH] [--fast]

Exit 0 when a record was written, even if a criterion fails. Fast
mode runs the cheap subset (everything except the workshop-based
P-D probe) and writes nothing canonical when called with
``record_path=None`` (the harness path). No LLM, stdlib only,
deterministic (fixed worlds; no randomised worlds).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone

from ..body.environment import Environment, RoomWorld, room_world_actions
from ..body.research import ClaimExtractor
from ..brain import Brain
from ..perception.image import StubImageSensor
from ..planning.planner import Action
from ..world.model import EpistemicStatus, WorldModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase8-eval-record.json")
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "docs", "phase8-eval-summary.txt")

MAX_CYCLES = 60

# The treasure task's start state / goal (the demo's RoomWorld).
TREASURE_START = {"at": "hall", "has_key": False, "door_open": False,
                  "has_treasure": False}
TREASURE_TARGET = {"has_treasure": True}

# The pick_lock route and the key route, as action-name sequences.
PICK_LOCK_ROUTE = ["go_lab_from_hall", "pick_lock", "go_vault", "take_treasure"]
KEY_ROUTE = ["go_lab_from_hall", "take_key", "unlock", "go_vault", "take_treasure"]

# Lab state: both door-openers are applicable here (used by what_if).
LAB_STATE = {"at": "lab", "has_key": True, "door_open": False,
             "has_treasure": False}


# ------------------------------------------------------------------ second task world
class GemWorld(Environment):
    """Structurally similar to RoomWorld: same rooms, same key/door
    shape, same ``pick_lock`` trap (never works), a different treasure
    (a gem instead of the treasure) so the continued brain faces a
    second task rather than a re-run of a satisfied goal. The brain
    reuses its existing go / take_key / unlock / pick_lock actions;
    only ``take_gem`` is added (see :func:`gem_actions`)."""

    ADJ = {"hall": ["lab"], "lab": ["hall", "vault"], "vault": ["lab"]}

    def __init__(self):
        self.s = {"at": "hall", "has_key": False, "door_open": False,
                  "has_gem": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.ADJ[s["at"]] and (dest != "vault" or s["door_open"]):
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_key" and s["at"] == "lab" and not s["has_key"]:
            s["has_key"], ok, info = True, True, "took the key"
        elif name == "unlock" and s["at"] == "lab" and s["has_key"]:
            s["door_open"], ok, info = True, True, "unlocked the vault door"
        elif name == "pick_lock":
            info = "the lock resists; picking fails"
        elif name == "take_gem" and s["at"] == "vault":
            s["has_gem"], ok, info = True, True, "took the gem"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def gem_actions() -> list[Action]:
    """The second task's actions: the treasure task's actions plus
    ``take_gem``. Only the new action is added to a continued brain
    (the shared actions already exist there)."""
    return room_world_actions() + [
        Action("take_gem", {"at": "vault"}, {"has_gem": True},
               args={"_do": "take_gem"})]


# ------------------------------------------------------------------ shared helpers
def _set_world(brain: Brain, state: dict) -> None:
    """Put the brain's believed state at ``state`` (the test-used
    whole-world setter). ``Brain.imagine`` / ``what_if`` /
    ``counterfactual`` all simulate from the brain's current world,
    so imagination experiments set the start state this way and say
    so in their evidence."""
    brain.world = dict(state)


def _count(report, key: str) -> int:
    """A reading-report count, whether the report carries ints (the
    real ``Brain.read_document``) or lists (the fallback path)."""
    value = (report or {}).get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return 0


def _action_by_name(brain: Brain, name: str) -> Action | None:
    for a in brain.planner.actions:
        if a.name == name:
            return a
    return None


def _simulate_route(brain: Brain, steps: list, start: dict,
                    target: dict) -> dict:
    """Simulate one route over the brain's real Simulator
    (``ab/imagination``: a copy of the world model + planner
    ``effects_fn`` + causal success rates). ``steps`` may be action
    names or Action objects: an Action object IS its variant (Phase 9
    §4.1 — variant-aware causal records), so a route built from
    Action objects simulates each step as its own variant, while a
    name-only route resolves by (name, state). Falls back to
    ``ReasoningEngine.simulate`` when no Simulator exists. Returns
    the scenario record; nothing is executed and nothing is written
    to the real world model, state or LTM."""
    names = [getattr(s, "name", s) for s in steps]
    simulator = None
    get_sim = getattr(brain, "_get_simulator", None)
    if callable(get_sim):
        try:
            simulator = get_sim()
        except Exception:  # noqa: BLE001 - fallback below
            simulator = None
    if simulator is not None:
        result = simulator.simulate(list(steps), dict(start),
                                    goal=dict(target))
        scenario = result.to_dict()
        steps = []
        for step in scenario["per_step"]:
            action = _action_by_name(brain, step["action"])
            steps.append({"action": step["action"],
                          "cost": (round(brain.planner.effective_cost(action), 4)
                                   if action is not None else None),
                          "confidence": step["confidence"]})
        return {
            "route": list(names),
            "steps": steps,
            "predicted_cost": scenario["predicted_cost"],
            "predicted_success": scenario["predicted_success"],
            "risk": scenario["risk"],
            "predicted_state": {k: scenario["predicted_state"].get(k) for k in
                                ("at", "has_key", "door_open", "has_treasure",
                                 "has_gem")
                                if k in scenario["predicted_state"]},
            "met_goal": scenario["met_goal"],
            "assumptions": scenario["assumptions"],
            "unknown_effect_steps": [a for a in scenario["assumptions"]
                                     if "no known effect" in a],
            "simulator": "ab.imagination.Simulator",
        }
    # Fallback: the Phase 4 symbolic simulation path.
    steps = []
    confidences = []
    cur = dict(start)
    for name in names:
        action = _action_by_name(brain, name)
        _expected, conf = brain.world_model.predict(
            name, dict(cur), brain.planner.effects_fn, causal=brain.causal)
        steps.append({"action": name,
                      "cost": (round(brain.planner.effective_cost(action), 4)
                               if action is not None else None),
                      "confidence": round(float(conf), 4)})
        confidences.append(float(conf))
        cur.update(brain.planner.effects_fn(name, dict(cur)) or {})
    sim_state, met, assumptions = brain.reasoning.simulate(
        list(names), dict(start, goal=dict(target)), brain.planner.effects_fn)
    predicted_success = round(min(confidences), 4) if confidences else None
    return {
        "route": list(names),
        "steps": steps,
        "predicted_cost": round(sum(s["cost"] for s in steps
                                    if s["cost"] is not None), 4),
        "predicted_success": predicted_success,
        "risk": (round(1.0 - predicted_success, 4)
                 if predicted_success is not None else None),
        "predicted_state": {k: sim_state.get(k) for k in
                            ("at", "has_key", "door_open", "has_treasure",
                             "has_gem") if k in sim_state},
        "met_goal": bool(met),
        "assumptions": list(assumptions),
        "unknown_effect_steps": [a for a in assumptions
                                 if a.startswith("no known symbolic effect")],
        "simulator": "ReasoningEngine.simulate (fallback)",
    }


def _adapt_imagine_scenario(scenario: dict) -> dict:
    """An ``imagine()`` scenario dict into this module's scenario
    shape (``route`` for the action names, per-step costs added)."""
    steps = []
    for step in scenario.get("per_step", []):
        steps.append({"action": step["action"],
                      "confidence": step["confidence"],
                      "effects": step.get("effects", {}),
                      "state": step.get("state", {})})
    return {
        "name": scenario.get("name"),
        "route": list(scenario.get("steps", [])),
        "steps": steps,
        "predicted_cost": scenario.get("predicted_cost"),
        "predicted_success": scenario.get("predicted_success"),
        "risk": scenario.get("risk"),
        "predicted_state": scenario.get("predicted_state", {}),
        "met_goal": scenario.get("met_goal"),
        "assumptions": scenario.get("assumptions", []),
        "violations": scenario.get("violations", []),
        "simulated": scenario.get("simulated", True),
    }


def _imagine_routes(brain: Brain, start: dict, target: dict) -> tuple[dict, str]:
    """Imagine routes to ``target`` from ``start`` via the designed
    ``Brain.imagine`` (which simulates from the brain's current
    world, so the start state is set first and recorded). Falls back
    to ``plan_all`` + :func:`_simulate_route` when ``imagine`` is
    absent. Returns (imagine_result_dict, api_tag)."""
    imagine = getattr(brain, "imagine", None)
    if callable(imagine):
        _set_world(brain, start)
        result = imagine(dict(target))
        result = dict(result)
        result["scenarios"] = [_adapt_imagine_scenario(s)
                               for s in result.get("scenarios", [])]
        ranking = []
        for entry in result.get("ranking", []):
            scenario = entry.get("scenario") or {}
            ranking.append({
                "rank": entry.get("rank"),
                "route": list(scenario.get("steps", [])),
                "predicted_success": entry.get("predicted_success"),
                "predicted_cost": entry.get("predicted_cost"),
                "risk": entry.get("risk"),
                "reason": entry.get("reason", ""),
            })
        result["ranking"] = ranking
        best = result.get("best") or {}
        result["best_route"] = list(best.get("steps", [])) if best else None
        result["start_state_set_for_imagination"] = dict(start)
        return result, "Brain.imagine"
    _set_world(brain, start)
    plans = brain.planner.plan_all(dict(start), dict(target), limit=3)
    scenarios = [_simulate_route(brain, [a.name for a in p.steps],
                                 dict(start), dict(target)) for p in plans]
    ranking = _rank_scenarios(scenarios)
    return {"goal": dict(target), "scenarios": scenarios, "ranking": ranking,
            "best": scenarios[0] if scenarios else None,
            "best_route": (scenarios[0]["route"] if scenarios else None),
            "simulated": True,
            "assumptions": [],
            "start_state_set_for_imagination": dict(start)}, (
        "planner.plan_all + _simulate_route (Brain.imagine absent)")


def _rank_scenarios(scenarios: list[dict]) -> list[dict]:
    """Rank by predicted success (desc), then predicted cost (asc),
    then risk (asc), with the reason recorded, never a bare choice."""
    ranked = sorted(
        scenarios,
        key=lambda s: (-(s["predicted_success"]
                         if s["predicted_success"] is not None else 0.0),
                       s["predicted_cost"],
                       s["risk"] if s["risk"] is not None else 1.0))
    out = []
    for i, s in enumerate(ranked):
        out.append({
            "rank": i + 1,
            "route": s["route"],
            "predicted_success": s["predicted_success"],
            "predicted_cost": s["predicted_cost"],
            "risk": s["risk"],
            "reason": (f"predicted success {s['predicted_success']} "
                       f"(highest first), then predicted cost "
                       f"{s['predicted_cost']} (lowest first), then risk "
                       f"{s['risk']}; route {s['route']}"),
        })
    return out


def _run_treasure(brain: Brain, max_cycles: int = MAX_CYCLES) -> dict:
    """Execute the treasure task for real (not simulated) and report
    the outcome."""
    env = RoomWorld()
    brain.add_goal("get the treasure", dict(TREASURE_TARGET))
    stop = brain.run(env, brain.cycle + max_cycles)
    goal = brain.goals.all()[-1]
    return {
        "completed": goal.status == "done",
        "goal_status": goal.status,
        "stop_reason": stop,
        "cycles": brain.cycle,
        "pick_lock_attempts": sum(1 for e in env.log if e.startswith("pick_lock")),
        "env_log": list(env.log),
    }


def _route_prediction(brain: Brain, names: list[str], start: dict,
                      target: dict) -> dict:
    """Predicted cost/success of one named route (used for the
    revision measurement, where the route may be pair-banned out of
    ``plan_all`` and must still be measurable)."""
    return _simulate_route(brain, list(names), dict(start), dict(target))


# ------------------------------------------------------------------ experiment 1
def exp_imagine_routes() -> dict:
    """SS4.1: from one goal, >= 2 distinct routes imagined, each with
    a distinct predicted future state."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    result, api = _imagine_routes(brain, TREASURE_START, TREASURE_TARGET)
    scenarios = result["scenarios"]
    routes = [tuple(s["route"]) for s in scenarios]
    states = [json.dumps(s["predicted_state"], sort_keys=True) for s in scenarios]
    passed = bool(len(set(routes)) >= 2 and len(set(states)) >= 2
                  and all(s["met_goal"] for s in scenarios)
                  and result.get("simulated") is True)
    return {
        "criterion": (">= 2 distinct routes imagined from one goal, "
                      "each with a distinct predicted future state, "
                      "each meeting the goal, all labelled simulated"),
        "passed": passed,
        "api": api,
        "n_routes": len(scenarios),
        "n_distinct_routes": len(set(routes)),
        "n_distinct_predicted_states": len(set(states)),
        "scenarios": scenarios,
        "simulated": result.get("simulated"),
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 2
def exp_predict_consequences() -> dict:
    """SS4.2: each imagined route predicts consequences with a
    per-step confidence from the causal model; predictions are
    recorded, not asserted after the fact. The brain first executes
    the treasure task once, so causal records exist (a fresh brain
    would report 1.0 everywhere, which is the uncalibrated default,
    not evidence).

    Finding (recorded, not hidden): after the run, ``Brain.imagine``
    itself returns ONE scenario -- the learned (pick_lock, situation)
    pair ban prunes the pick_lock candidate from ``plan_all``, so the
    generator never sees it. The dropped route is still imaginable:
    imagination is not execution, and a banned pair is an execution
    avoidance, so the experiment simulates both routes directly
    through the same Simulator ``imagine`` uses (and records
    ``imagine``'s own post-learning count alongside). Whether
    ``imagine`` should surface banned-but-imaginable routes is a
    design question for W2/coordinator, not asserted here."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    run = _run_treasure(brain)
    imagined, imagine_api = _imagine_routes(brain, TREASURE_START,
                                            TREASURE_TARGET)
    scenarios = [_simulate_route(brain, route, TREASURE_START,
                                 TREASURE_TARGET)
                 for route in (PICK_LOCK_ROUTE, KEY_ROUTE)]
    all_steps = [st for s in scenarios for st in s["steps"]]
    all_numeric = all(isinstance(st["confidence"], (int, float))
                      for st in all_steps)
    any_below_one = any(st["confidence"] < 1.0 for st in all_steps)
    pick_lock_conf = None
    for s in scenarios:
        for st in s["steps"]:
            if st["action"] == "pick_lock":
                pick_lock_conf = st["confidence"]
    assumptions = [a for s in scenarios for a in s["assumptions"]]
    passed = bool(run["completed"] and len(scenarios) >= 2 and all_steps
                  and all_numeric and any_below_one and assumptions)
    return {
        "criterion": ("every imagined step carries a numeric per-step "
                      "confidence from the causal model, at least one "
                      "step's confidence is below 1.0 (evidence-based, "
                      "not the uncalibrated default), and assumptions "
                      "are recorded per scenario"),
        "passed": passed,
        "api": "Brain.imagine + Simulator direct simulation of both routes",
        "imagine_api": imagine_api,
        "imagine_scenarios_post_learning": len(imagined["scenarios"]),
        "imagine_post_learning_routes": [s["route"] for s in imagined["scenarios"]],
        "seed_run_completed": run["completed"],
        "n_scenarios": len(scenarios),
        "n_steps_total": len(all_steps),
        "all_confidences_numeric": all_numeric,
        "any_confidence_below_1": any_below_one,
        "pick_lock_predicted_confidence": pick_lock_conf,
        "causal_pick_lock_door_open": brain.causal.counts("pick_lock", "door_open"),
        "causal_unlock_door_open": brain.causal.counts("unlock", "door_open"),
        "n_assumptions_recorded": len(assumptions),
        "scenarios": scenarios,
        "dependency": None,
        "note": ("Post-learning, imagine() returns 1 scenario (the "
                 "key route): the learned pick_lock pair ban prunes "
                 "the other candidate. Both routes are simulated "
                 "directly through the same Simulator here, so the "
                 "per-step confidences of the dropped route are still "
                 "measured (pick_lock 0.1667, revision-adjusted "
                 "toward the observed 0.0)."),
    }


# ------------------------------------------------------------------ experiment 3
def exp_compare_routes() -> dict:
    """SS4.3: scenarios compared on predicted success, cost, risk;
    the comparison is recorded with reasons. Same measurement shape
    as experiment 2: after the seed run ``Brain.imagine`` returns
    one scenario (pair ban -- see experiment 2's note), so the
    comparison is measured over both routes simulated directly
    through the same Simulator, with ``imagine``'s own ranking
    recorded alongside."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    run = _run_treasure(brain)
    imagined, imagine_api = _imagine_routes(brain, TREASURE_START,
                                            TREASURE_TARGET)
    scenarios = [_simulate_route(brain, route, TREASURE_START,
                                 TREASURE_TARGET)
                 for route in (PICK_LOCK_ROUTE, KEY_ROUTE)]
    ranking = _rank_scenarios(scenarios)
    top = ranking[0] if ranking else None
    passed = bool(run["completed"] and len(ranking) >= 2 and top is not None
                  and all(r["reason"] for r in ranking)
                  and top["route"] == KEY_ROUTE)
    return {
        "criterion": (">= 2 scenarios ranked by predicted success, "
                      "then cost, then risk, with a recorded reason "
                      "per scenario; the key route ranks first after "
                      "the pick_lock failure is learned"),
        "passed": passed,
        "api": "Simulator direct simulation of both routes + _rank_scenarios",
        "imagine_api": imagine_api,
        "imagine_ranking_post_learning": imagined["ranking"],
        "seed_run_completed": run["completed"],
        "n_scenarios": len(ranking),
        "ranking": ranking,
        "top_route": top["route"] if top else None,
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 4
def exp_select_and_execute() -> dict:
    """SS4.4: the best-simulated route is selected with a recorded
    reason and then executed for real (not simulated). On a fresh
    brain both routes predict success 1.0, so cost breaks the tie
    and the pick_lock route is selected; it fails at pick_lock in
    reality and the brain replans to the key route -- recorded, not
    hidden. The brain's own _do_make_plan now ranks plans by
    simulation (P8-6), so execution itself exercises the simulated
    ranking."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    result, api = _imagine_routes(brain, TREASURE_START, TREASURE_TARGET)
    ranking = result["ranking"]
    selected = ranking[0] if ranking else None
    run = _run_treasure(brain)
    errors = brain.world_model.prediction_errors()
    sim_comparison = brain.last_simulation_comparison
    passed = bool(selected is not None and run["completed"])
    return {
        "criterion": ("best-simulated route selected with a recorded "
                      "reason, then executed for real; the goal "
                      "completes"),
        "passed": passed,
        "api": api,
        "selected_route": selected["route"] if selected else None,
        "selection_reason": selected["reason"] if selected else None,
        "selected_predicted_success": selected["predicted_success"] if selected else None,
        "selected_predicted_cost": selected["predicted_cost"] if selected else None,
        "executed_completed": run["completed"],
        "executed_cycles": run["cycles"],
        "pick_lock_attempts_during_execution": run["pick_lock_attempts"],
        "simulation_ranking_used_in_execution": (
            sim_comparison is not None),
        "note": ("The selected pick_lock route failed at pick_lock in "
                 "reality (prediction error recorded below); the "
                 "brain replanned and completed via the key route. "
                 "Selection was best-simulated at selection time "
                 "(success tie 1.0 vs 1.0, cost 4.0 vs 5.0); the "
                 "failure is the imagination gap experiment 5 "
                 "measures, not a selection error."),
        "prediction_errors_after_execution": errors,
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 5
def exp_imagined_vs_actual() -> dict:
    """SS4.5: after execution, the imagined outcome is compared
    against reality; the prediction error is recorded. Compares the
    imagined intermediate state after pick_lock (door_open True)
    with the observed state (door_open False) via the brain's own
    prediction-checking machinery."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    result, api = _imagine_routes(brain, TREASURE_START, TREASURE_TARGET)
    best = None
    for s in result["scenarios"]:
        if s["route"] == result.get("best_route"):
            best = s
            break
    imagined_after_pick_lock = None
    if best is not None:
        for step in best["steps"]:
            if step["action"] == "pick_lock":
                imagined_after_pick_lock = {
                    "door_open": step.get("state", {}).get("door_open"),
                    "confidence": step.get("confidence"),
                    "note": ("simulated state after pick_lock in the "
                             "best imagined route")}
                break
    run = _run_treasure(brain)
    checked = [e for e in brain.events if e.get("kind") == "prediction_checked"]
    errors = brain.world_model.prediction_errors()
    pick_lock_error = [r for r in errors
                       if r["action"] == "pick_lock"
                       and r["effect"] == "door_open"]
    passed = bool(run["completed"] and checked and pick_lock_error
                  and pick_lock_error[0]["expected"] is True
                  and pick_lock_error[0]["actual"] is False
                  and imagined_after_pick_lock is not None
                  and imagined_after_pick_lock["door_open"] is True)
    return {
        "criterion": ("imagined outcome compared against reality after "
                      "execution; at least one prediction_checked "
                      "event and a recorded prediction error for "
                      "pick_lock/door_open (expected True, actual "
                      "False); the imagined post-pick_lock state said "
                      "door_open True"),
        "passed": passed,
        "api": api,
        "best_route": result.get("best_route"),
        "imagined_after_pick_lock": imagined_after_pick_lock,
        "actual_after_pick_lock": ({"door_open": False,
                                    "source": "prediction error record"}
                                   if pick_lock_error else None),
        "n_prediction_checked_events": len(checked),
        "n_prediction_errors": len(errors),
        "pick_lock_prediction_error": pick_lock_error[0] if pick_lock_error else None,
        "executed_completed": run["completed"],
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 6
def exp_revision_second_task() -> dict:
    """SS4.6: a second, structurally similar task shows the revision.
    Predicted cost/success of the pick_lock route are measured BEFORE
    task 1 (treasure) and AFTER it, on the same brain and the same
    start state (via the Simulator); then a second real task (the gem
    world: same rooms, same key/door shape, same pick_lock trap,
    different treasure) is executed. Pass: the predicted success
    moves toward the observed pick_lock success rate (0.0), or the
    route ranking changes (the pick_lock route drops out of plan_all
    / is no longer first). If nothing moves, this is a recorded
    negative result, not a pass -- the numbers below are the verdict
    either way."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    before = _route_prediction(brain, PICK_LOCK_ROUTE, TREASURE_START,
                               TREASURE_TARGET)
    plans_before = brain.planner.plan_all(dict(TREASURE_START),
                                          dict(TREASURE_TARGET), limit=3)
    first_before = ([a.name for a in plans_before[0].steps]
                    if plans_before else None)
    run1 = _run_treasure(brain)
    after = _route_prediction(brain, PICK_LOCK_ROUTE, TREASURE_START,
                              TREASURE_TARGET)
    plans_after = brain.planner.plan_all(dict(TREASURE_START),
                                         dict(TREASURE_TARGET), limit=3)
    first_after = ([a.name for a in plans_after[0].steps]
                   if plans_after else None)
    pick_lock_in_plans_after = any(
        "pick_lock" in [a.name for a in p.steps] for p in plans_after)
    observed_rate = brain.causal.counts("pick_lock", "door_open")
    observed_success = (observed_rate[0] / observed_rate[1]
                        if observed_rate else None)
    # Second structurally similar task, for real, on the same brain.
    gem_start = {"at": "hall", "has_key": False, "door_open": False,
                 "has_gem": False}
    gem_target = {"has_gem": True}
    take_gem = next(a for a in gem_actions() if a.name == "take_gem")
    brain.planner.add_action(take_gem)
    gem_route_before2 = _route_prediction(
        brain, ["go_lab_from_hall", "pick_lock", "go_vault", "take_gem"],
        gem_start, gem_target)
    env2 = GemWorld()
    brain.add_goal("get the gem", dict(gem_target))
    stop2 = brain.run(env2, brain.cycle + MAX_CYCLES)
    goal2 = brain.goals.all()[-1]
    gem_pick_lock_attempts = sum(1 for e in env2.log if e.startswith("pick_lock"))
    moved_toward = bool(
        before["predicted_success"] is not None
        and after["predicted_success"] is not None
        and observed_success is not None
        and abs(after["predicted_success"] - observed_success)
        < abs(before["predicted_success"] - observed_success))
    cost_moved = bool(after["predicted_cost"] > before["predicted_cost"])
    ranking_changed = bool(first_before != first_after
                           or not pick_lock_in_plans_after)
    passed = bool(moved_toward or ranking_changed)
    return {
        "criterion": ("second structurally similar task shows the "
                      "revision: predicted success moves toward the "
                      "observed value, or the route ranking changes"),
        "passed": passed,
        "subcriteria": {
            "predicted_success_moved_toward_observed": moved_toward,
            "predicted_cost_rose": cost_moved,
            "ranking_changed": ranking_changed,
        },
        "task1_before": {"predicted_cost": before["predicted_cost"],
                         "predicted_success": before["predicted_success"],
                         "first_plan_first_route": first_before},
        "task1_after": {"predicted_cost": after["predicted_cost"],
                        "predicted_success": after["predicted_success"],
                        "first_plan_first_route": first_after,
                        "pick_lock_route_still_in_plan_all": pick_lock_in_plans_after},
        "task1_observed": {"completed": run1["completed"],
                           "pick_lock_attempts": run1["pick_lock_attempts"],
                           "pick_lock_causal_counts": observed_rate,
                           "pick_lock_observed_success_rate": observed_success,
                           "planner_failure_rate_pick_lock":
                               round(brain.planner.failure_rate("pick_lock"), 4)},
        "task2_gem": {"predicted_cost_pick_lock_route":
                          gem_route_before2["predicted_cost"],
                      "predicted_success_pick_lock_route":
                          gem_route_before2["predicted_success"],
                      "completed": goal2.status == "done",
                      "goal_status": goal2.status,
                      "stop_reason": stop2,
                      "pick_lock_attempts": gem_pick_lock_attempts,
                      "pick_lock_avoided_in_second_task":
                          gem_pick_lock_attempts == 0},
        "dependency": None,
        "note": ("Revision rides on existing persisted structures "
                 "(causal records + planner failure rates + pair bans "
                 "+ the Simulator's revision-adjusted confidences, "
                 "P8-7): no persistence version bump, no new stored "
                 "scenario records."),
    }


# ------------------------------------------------------------------ experiment 7
def exp_what_if_query() -> dict:
    """SS4.7: 'what if I use the fragile tool?' answered from
    simulation with explicit assumptions, labelled simulated. The
    fragile tool in this world is pick_lock (it never works); the
    parser identifies actions by name, so the experiment asks the
    parseable forms ('what if I use pick_lock?' / 'what if I use
    unlock?') from the lab state where both are applicable, and
    records the literal 'fragile tool' phrasing separately: the
    brain honestly declines it (no action or tool bears that name)
    rather than guessing."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    _run_treasure(brain)  # seed causal records, so the answer is evidence-based
    _set_world(brain, LAB_STATE)
    goal = brain.add_goal("get the treasure", dict(TREASURE_TARGET))
    # Activate the goal (as the demo does): what_if then simulates the
    # action PLUS the rest of the route to the goal, not just the one
    # step. Without an active goal it answers the single step only.
    activate = getattr(brain.goals, "activate", None)
    if callable(activate):
        activate(goal.id)
    what_if = getattr(brain, "what_if", None)
    if callable(what_if):
        api = "Brain.what_if"
        fragile = what_if("what if I use pick_lock?")
        key = what_if("what if I use unlock?")
        literal = what_if("what if I use the fragile tool?")
    else:
        api = ("_simulate_route (Brain.what_if absent)")
        fragile = _simulate_route(brain, PICK_LOCK_ROUTE, LAB_STATE,
                                  TREASURE_TARGET)
        fragile = {"question": "what if I use pick_lock?",
                   "simulated": True, "predicted_success": fragile["predicted_success"],
                   "predicted_cost": fragile["predicted_cost"],
                   "assumptions": fragile["assumptions"],
                   "predicted_state": fragile["predicted_state"]}
        key = _simulate_route(brain, KEY_ROUTE, LAB_STATE, TREASURE_TARGET)
        key = {"question": "what if I use unlock?",
               "simulated": True, "predicted_success": key["predicted_success"],
               "predicted_cost": key["predicted_cost"],
               "assumptions": key["assumptions"],
               "predicted_state": key["predicted_state"]}
        literal = {"question": "what if I use the fragile tool?",
                   "answer": "Brain.what_if absent; fallback has no parser"}
    fragile_ok = bool(fragile.get("simulated") is True
                      and fragile.get("assumptions")
                      and fragile.get("predicted_success") is not None)
    key_ok = bool(key.get("simulated") is True and key.get("assumptions")
                  and key.get("predicted_success") is not None)
    lower = bool(fragile_ok and key_ok
                 and fragile["predicted_success"] < key["predicted_success"])
    passed = bool(fragile_ok and key_ok and lower)
    return {
        "criterion": ("both what-if questions answered from simulation "
                      "with explicit non-empty assumptions, labelled "
                      "simulated; the fragile-tool (pick_lock) route "
                      "predicts lower success than the key (unlock) "
                      "route"),
        "passed": passed,
        "api": api,
        "answers": {"pick_lock": fragile, "unlock": key,
                    "literal_fragile_tool_phrasing": literal},
        "pick_lock_predicted_success": fragile.get("predicted_success"),
        "unlock_predicted_success": key.get("predicted_success"),
        "dependency": None,
        "note": ("The literal 'fragile tool' phrasing is not parsed "
                 "(no action or tool bears that name); the brain "
                 "answers that it could not identify an action rather "
                 "than guessing. W2 parser coverage is the gap, not "
                 "the simulation: the parseable forms answer fully."),
    }


# ------------------------------------------------------------------ experiment 8
def exp_counterfactual_world_model() -> dict:
    """SS4.8: 'if the door had been open...' over the world model
    (not just planner effects): the counterfactual edits a fact in a
    COPY of the world model, re-simulates, and the outcome differs
    from the actual-history simulation, with the changed assumption
    named. The real world model must be unchanged afterwards. Both
    edit forms are exercised: a state-key edit and a world-model
    entity/property edit."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    _set_world(brain, TREASURE_START)
    brain.add_goal("get the treasure", dict(TREASURE_TARGET))
    real_before = brain.world_model.to_json()
    direct = ["go_lab_from_hall", "go_vault", "take_treasure"]
    counterfactual = getattr(brain, "counterfactual", None)
    if callable(counterfactual):
        api = "Brain.counterfactual"
        result = counterfactual(edits={"door_open": True},
                                goal=dict(TREASURE_TARGET), steps=direct)
        result_wm = counterfactual(
            edits={"entity": "door", "property": "open", "value": True},
            goal=dict(TREASURE_TARGET), steps=direct)
    else:
        api = "world-model-copy simulation (Brain.counterfactual absent)"
        baseline = _simulate_route(brain, direct, TREASURE_START, TREASURE_TARGET)
        cf_start = dict(TREASURE_START, door_open=True)
        changed = _simulate_route(brain, direct, cf_start, TREASURE_TARGET)
        result = {"baseline": baseline, "counterfactual": changed,
                  "difference": {k: (baseline["predicted_state"].get(k),
                                     changed["predicted_state"].get(k))
                                 for k in set(baseline["predicted_state"])
                                 | set(changed["predicted_state"])
                                 if baseline["predicted_state"].get(k)
                                 != changed["predicted_state"].get(k)},
                  "outcome_changed": baseline["met_goal"] != changed["met_goal"],
                  "assumptions": ["simulated: assumed door_open=True "
                                  "(actual value was False)"],
                  "simulated": True}
        result_wm = result
    real_after = brain.world_model.to_json()
    real_unchanged = real_after == real_before
    outcome_changed = bool(result.get("outcome_changed"))
    difference = result.get("difference") or {}
    assumptions = result.get("assumptions") or []
    names_change = any("door_open" in a for a in assumptions)
    wm_changed = bool(result_wm.get("outcome_changed"))
    passed = bool(outcome_changed and difference and names_change
                  and result.get("simulated") is True and real_unchanged
                  and wm_changed)
    return {
        "criterion": ("counterfactual outcome differs from the "
                      "actual-history simulation, the changed "
                      "assumption is named in the assumptions, the "
                      "result is labelled simulated, and the real "
                      "world model is unchanged; the world-model "
                      "entity/property edit form changes the outcome "
                      "too"),
        "passed": passed,
        "api": api,
        "changed_assumption_named": names_change,
        "baseline_met_goal": (result.get("baseline") or {}).get("met_goal"),
        "counterfactual_met_goal": (result.get("counterfactual") or {}).get("met_goal"),
        "difference": difference,
        "outcome_changed": outcome_changed,
        "world_edit_outcome_changed": wm_changed,
        "assumptions": assumptions,
        "real_world_model_unchanged": real_unchanged,
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 9
def _read_document_fallback(brain: Brain, path: str) -> dict:
    """Document ingestion through the Phase 3 admission path
    (``ClaimExtractor`` + LTM ``add`` with provenance ``file:<name>``)
    -- used only if ``Brain.read_document`` is absent. Size-capped;
    a missing/oversize file fails cleanly with no partial claims."""
    name = os.path.basename(path)
    source = f"file:{name}"
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return {"path": path, "source": source, "stored": 0,
                "corroborated": 0, "duplicate": 0, "conflict": 0,
                "ignored_sentences": [],
                "error": f"{type(exc).__name__}: {exc}"}
    if size > 64 * 1024:
        return {"path": path, "source": source, "stored": 0,
                "corroborated": 0, "duplicate": 0, "conflict": 0,
                "ignored_sentences": [],
                "error": "oversize: file larger than 64 KiB; no claims read"}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    report = {"path": path, "source": source, "stored": 0,
              "corroborated": 0, "duplicate": 0, "conflict": 0,
              "ignored_sentences": [], "claims": []}
    for c in brain.extractor.extract(text):
        result, fact = brain.ltm.add(c.subject, c.relation, c.value,
                                     source, brain.cycle)
        report[result.value] += 1
        report["claims"].append({"subject": fact.subject,
                                 "relation": fact.relation,
                                 "value": fact.value,
                                 "sentence": c.sentence,
                                 "status": fact.status})
    for sent in [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]:
        if not any(c.sentence.startswith(sent) for c in brain.extractor.extract(text)):
            report["ignored_sentences"].append(sent)
    return report


def exp_document_perception() -> dict:
    """SS4.9: a local .txt/.md document ingested through the bus with
    provenance ``file:<name>``; a single-source claim is unverified;
    a second independent document (or teacher) verifies it. A missing
    file fails cleanly with no partial claims."""
    brain = Brain()
    read_document = getattr(brain, "read_document", None)
    api = ("Brain.read_document" if callable(read_document) else
           "ClaimExtractor + LTM admission path (Brain.read_document absent)")
    with tempfile.TemporaryDirectory() as folder:
        doc1 = os.path.join(folder, "paris.md")
        doc2 = os.path.join(folder, "france-notes.md")
        missing = os.path.join(folder, "nope.md")
        with open(doc1, "w", encoding="utf-8") as fh:
            fh.write("Paris is the capital of France. Mars is a planet. "
                     "The quick brown fox jumps.\n")
        with open(doc2, "w", encoding="utf-8") as fh:
            fh.write("Paris is the capital of France.\n")
        read = read_document if callable(read_document) else (
            lambda p: _read_document_fallback(brain, p))
        rep1 = read(doc1)
        ask1 = brain.ask("paris", "capital_of").best
        rep2 = read(doc2)
        ask2 = brain.ask("paris", "capital_of").best
        rep1_dup = read(doc1)
        rep_missing = read(missing)
        claims_before = len(brain.ltm.all_facts())
        rep_missing2 = read(missing)
        claims_after = len(brain.ltm.all_facts())
    from ..core.memory import TRUST
    trust_used = TRUST.get("file", 0.3)
    single_unverified = bool(ask1 and ask1[2] == "unverified"
                             and ask1[3] == ["file:paris.md"])
    two_verified = bool(ask2 and ask2[2] == "verified"
                        and set(ask2[3]) == {"file:paris.md",
                                             "file:france-notes.md"})
    dup_counted = bool(_count(rep1_dup, "duplicate") >= 1)
    ignored_counted = bool(_count(rep1, "ignored_sentences") >= 1)
    missing_clean = bool(rep_missing.get("error")
                         and _count(rep_missing, "stored") == 0
                         and claims_before == claims_after)
    passed = bool(single_unverified and two_verified and dup_counted
                  and ignored_counted and missing_clean)
    return {
        "criterion": ("single-source claim unverified with provenance "
                      "file:<name>; same claim from a second "
                      "independent document verified with both "
                      "sources; re-reading the first document counts "
                      "as duplicate; an unmatched sentence is counted "
                      "ignored; a missing file fails cleanly with no "
                      "partial claims"),
        "passed": passed,
        "api": api,
        "file_trust_used": trust_used,
        "file_trust_specified_p8_2": 0.45,
        "ask_after_one_document": list(ask1) if ask1 else None,
        "ask_after_two_documents": list(ask2) if ask2 else None,
        "report_doc1": {k: rep1.get(k) for k in
                        ("stored", "corroborated", "duplicate", "conflict",
                         "ignored_sentences", "trust")},
        "report_doc2": {k: rep2.get(k) for k in
                        ("stored", "corroborated", "duplicate", "conflict",
                         "ignored_sentences", "trust")},
        "duplicate_on_reread": _count(rep1_dup, "duplicate"),
        "missing_file_error": rep_missing.get("error"),
        "missing_file_stored_no_partial_claims": missing_clean,
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 10
IMAGE_RECORD = {
    "objects": [
        {"name": "key", "properties": {"held": False},
         "position": {"room": "lab", "x": 0.3, "y": 0.5},
         "confidence": 0.95},
        {"name": "door", "properties": {"open": False},
         "position": {"room": "lab", "x": 0.8, "y": 0.5},
         "confidence": 0.9},
    ],
    "relations": [{"subject": "key", "relation": "left_of",
                   "object": "door"}],
    "source": "image:lab-scene",
    "image_id": "lab-scene-1",
}


def _image_entities_observed(brain: Brain) -> dict:
    key_ent = brain.world_model.entity("key")
    door_ent = brain.world_model.entity("door")
    key_observed = bool(
        key_ent is not None
        and key_ent.properties.get("held") is not None
        and key_ent.properties["held"].status.value == "observed")
    door_observed = bool(
        door_ent is not None
        and door_ent.properties.get("open") is not None
        and door_ent.properties["open"].status.value == "observed")
    return {
        "key_observed": key_observed,
        "door_observed": door_observed,
        "spatial_left_of": bool(brain.world_model.relations("key", "left_of")),
        "key_property": (repr(key_ent.properties.get("held"))
                         if key_ent is not None else None),
        "door_property": (repr(door_ent.properties.get("open"))
                          if door_ent is not None else None),
    }


def exp_image_percept() -> dict:
    """SS4.10: an image percept with objects/properties enters the
    world model as observed entities with spatial relations -- via
    the bus (``Brain.perceive("image", record)``), via a stub sensor
    acting as the specialist, and (rejection case) a bare image path
    with no specialist attached is a rejected percept, never invented
    objects."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    percept = brain.perceive("image", dict(IMAGE_RECORD),
                             source="image:lab-scene")
    direct = _image_entities_observed(brain)
    direct_ok = bool(percept.status == "admitted" and direct["key_observed"]
                     and direct["door_observed"] and direct["spatial_left_of"])
    # Stub sensor as the specialist: a path becomes the same record.
    stub_brain = Brain(room_world_actions(), granted={"compute", "environment"},
                       specialist=StubImageSensor({"lab.png": IMAGE_RECORD}))
    stub_percept = stub_brain.perceive("image", "lab.png", source="image:lab.png")
    stub = _image_entities_observed(stub_brain)
    stub_ok = bool(stub_percept.status == "admitted"
                   and stub_percept.provenance.get("specialist") == "stub"
                   and stub["key_observed"] and stub["spatial_left_of"])
    # No specialist: a bare path is rejected, never invented objects.
    bare_brain = Brain(room_world_actions(), granted={"compute", "environment"})
    bare = bare_brain.perceive("image", "lab.png", source="image:lab.png")
    bare_entities_before = len(bare_brain.world_model.entities())
    bare_ok = bool(bare.status == "rejected"
                   and bare.provenance.get("reason") == "no specialist attached"
                   and len(bare_brain.world_model.entities()) == bare_entities_before)
    passed = bool(direct_ok and stub_ok and bare_ok)
    return {
        "criterion": ("structured image record via the bus enters the "
                      "world model as observed entities with a spatial "
                      "relation; the same record via a stub specialist "
                      "arrives with specialist provenance; a bare image "
                      "path with no specialist is rejected, never "
                      "invented objects"),
        "passed": passed,
        "api": "Brain.perceive('image', ...) via PerceptionBus",
        "direct_record": {"status": percept.status,
                          "provenance": percept.provenance,
                          **direct},
        "stub_specialist": {"status": stub_percept.status,
                            "provenance": stub_percept.provenance,
                            **stub},
        "bare_path_no_specialist": {"status": bare.status,
                                    "provenance": bare.provenance,
                                    "rejected_without_invented_entities": bare_ok},
        "dependency": None,
    }


# ------------------------------------------------------------------ experiment 11
def exp_specialist_absent() -> dict:
    """SS4.11: the brain works fully with no specialist attached; and
    with a fake specialist attached, its output arrives as a percept
    with provenance (not as reasoning)."""
    # (a) No specialist: the treasure task completes on a plain brain.
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    run = _run_treasure(brain)
    part_a = bool(run["completed"])
    # (b) A fake specialist attached at construction.
    part_b = {}
    try:
        brain2 = Brain(room_world_actions(), granted={"compute", "environment"},
                       specialist=_FakeSpecialist())
    except TypeError as exc:
        brain2 = None
        part_b = {"specialist_api": "absent",
                  "detail": f"Brain(specialist=...) raised TypeError: {exc}"}
    if brain2 is not None:
        spec = getattr(brain2, "specialist", None)
        if spec is None:
            part_b = {"specialist_api": "accepted but not retained",
                      "detail": "Brain(specialist=...) did not keep the specialist"}
        else:
            percept = brain2.perceive("specialist", "lab-scene.png",
                                      source="specialist:fake-specialist")
            observed = _image_entities_observed(brain2)
            part_b = {
                "specialist_api": "present",
                "specialist_available": bool(spec.available()),
                "percept_status": percept.status,
                "percept_provenance": percept.provenance,
                "provenance_names_specialist":
                    percept.provenance.get("specialist") == "fake-specialist",
                "key_entity_observed": observed["key_observed"],
                "no_thought_from_specialist_output":
                    not any(getattr(t, "source", "") == "specialist:fake-specialist"
                            for t in brain2._incoming),
            }
    b_ok = bool(part_b.get("specialist_api") == "present"
                and part_b.get("percept_status") == "admitted"
                and part_b.get("provenance_names_specialist")
                and part_b.get("key_entity_observed"))
    passed = bool(part_a and b_ok)
    return {
        "criterion": ("brain completes the treasure task with no "
                      "specialist attached, and a fake specialist's "
                      "output arrives as an admitted percept with "
                      "provenance naming the specialist, entering the "
                      "world model as observed (never as reasoning)"),
        "passed": passed,
        "api": "Brain(specialist=...) + Brain.perceive('specialist', ...)",
        "no_specialist_completes": part_a,
        "no_specialist_cycles": run["cycles"],
        "fake_specialist": part_b,
        "dependency": None,
    }


class _FakeSpecialist:
    """A fake specialist perception model for the test: describes an
    image as a structured record, the same shape as P8-3."""

    name = "fake-specialist"

    def available(self) -> bool:
        return True

    def describe_image(self, path: str) -> dict:
        return {
            "objects": [{"name": "key",
                         "properties": {"held": False},
                         "position": {"room": "lab", "x": 0.3, "y": 0.5},
                         "confidence": 0.95}],
            "relations": [],
            "source": f"specialist:{self.name}",
            "image_id": str(path),
        }


# ------------------------------------------------------------------ experiment 12
def exp_pd_overgeneralisation_probe() -> dict:
    """SS4.12 / P8-8: simulation-based planning on the W1->W2 workshop
    shape; record whether simulation catches the over-generalisation
    before execution. The Phase 7 negative result: a P-D pattern
    learned in W1 (press fails without the mould) penalises EVERY
    press in W2 x6.0, including the correct press-with-mould variant.
    The probe builds both press variants explicitly (same action
    name, different preconditions) and asks the simulator whether
    their predicted success differs or the simulated ranking changes.
    Phase 9 §4.1: the probe now simulates the routes as Action
    objects (each step carries its variant), because a bare name
    carries no variant intent. A 'not caught' after that change
    would be a real failure of the variant-aware fix, recorded
    honestly -- it is no longer the expected negative result."""
    from .execution_eval import (WORKSHOP_W1, WORKSHOP_W2, _full_factory,
                                 _run_workshop, workshop_actions)

    rec_w1, brain_w1, _env_w1 = _run_workshop(WORKSHOP_W1, _full_factory)
    rec_w2, brain, _env_w2 = _run_workshop(WORKSHOP_W2, _full_factory,
                                           brain=brain_w1)
    # The start state for planning is a FRESH W2 world: the env that
    # just ran already holds the treasure, so planning from its state
    # would return the empty plan (target already satisfied).
    from .execution_eval import WorkshopWorld as _WorkshopWorld
    start = dict(_WorkshopWorld(WORKSHOP_W2).observe())
    target = {WORKSHOP_W2["treasure_key"]: True}
    plans = brain.planner.plan_all(dict(start), dict(target), limit=3)
    w2_actions = workshop_actions(WORKSHOP_W2)
    press_variants = [a for a in w2_actions if a.name == "press"]
    with_mould = next((a for a in press_variants
                       if WORKSHOP_W2["mould_key"] in a.pre), None)
    without_mould = next((a for a in press_variants
                          if WORKSHOP_W2["mould_key"] not in a.pre), None)
    pattern_mult = (brain.planner.pattern_multiplier(with_mould)
                    if with_mould is not None else None)
    # Both candidate plans use the with-mould press; build the
    # misfiring route explicitly by swapping in the without-mould
    # variant (same action name -- that sameness is the point).
    plan_a = plans[0] if plans else None
    route_a_names = [a.name for a in plan_a.steps] if plan_a else []
    route_b_actions = None
    if plan_a is not None and without_mould is not None:
        route_b_actions = [without_mould if a.name == "press" else a
                           for a in plan_a.steps]
    route_b_names = ([a.name for a in route_b_actions]
                     if route_b_actions is not None else [])
    sim_a = (_simulate_route(brain, list(plan_a.steps), dict(start),
                            dict(target))
             if route_a_names else None)
    sim_b = (_simulate_route(brain, list(route_b_actions), dict(start),
                            dict(target))
             if route_b_actions is not None else None)
    cost_a = (round(sum(brain.planner.effective_cost(a) for a in plan_a.steps), 4)
              if plan_a is not None else None)
    cost_b = (round(sum(brain.planner.effective_cost(a) for a in route_b_actions), 4)
              if route_b_actions is not None else None)
    naive_first = route_a_names
    sim_ranking = _rank_scenarios([s for s in (sim_a, sim_b) if s])
    sim_first = sim_ranking[0]["route"] if sim_ranking else None
    success_differs = bool(
        sim_a is not None and sim_b is not None
        and sim_a["predicted_success"] != sim_b["predicted_success"])
    ranking_changes = bool(sim_first is not None and naive_first
                           and sim_first != naive_first)
    caught = bool(success_differs or ranking_changes)
    candidates_press = []
    for p in plans:
        for a in p.steps:
            if a.name == "press":
                candidates_press.append(
                    "with_mould" if WORKSHOP_W2["mould_key"] in a.pre
                    else "without_mould")
    return {
        "criterion": ("simulation catches the W1->W2 over-"
                      "generalisation before execution: the misfiring "
                      "press-without-mould route's predicted success "
                      "drops below the correct press-with-mould "
                      "route's, or the simulated ranking changes in "
                      "favour of the mould route"),
        "passed": caught,
        "w1_completed": rec_w1["completed"],
        "w1_cycles": rec_w1["cycles"],
        "w2_continued_completed": rec_w2["completed"],
        "w2_continued_cycles": rec_w2["cycles"],
        "press_pattern_multiplier_in_w2": pattern_mult,
        "n_candidate_plans_w2": len(plans),
        "candidate_plans_press_variants": candidates_press,
        "naive_cost_ranking_first": naive_first,
        "simulated_ranking_first": sim_first,
        "ranking_changed": ranking_changes,
        "route_with_mould": {"names": route_a_names,
                             "variant_aware_cost": cost_a,
                             "predicted_success": sim_a["predicted_success"] if sim_a else None,
                             "predicted_cost_simulator": sim_a["predicted_cost"] if sim_a else None},
        "route_without_mould": {"names": route_b_names,
                                "variant_aware_cost": cost_b,
                                "predicted_success": sim_b["predicted_success"] if sim_b else None,
                                "predicted_cost_simulator": sim_b["predicted_cost"] if sim_b else None},
        "route_names_identical": route_a_names == route_b_names,
        "predicted_success_differs": success_differs,
        "causal_press_counts_after_w1_w2": brain.causal.counts(
            "press", "pressed"),
        "note": ("Phase 9 §4.1 update (same §9 precedent as the "
                 "persistence pins: the probe's simulation now passes "
                 "Action objects, not bare names, so each step carries "
                 "its variant intent): causal success is recorded per "
                 "(action, effect, variant) and effects_fn/simulate "
                 "resolve an action by (name, state), so the "
                 "press-with-mould and press-without-mould routes "
                 "simulate differently -- the misfiring route's "
                 "predicted success drops below the correct route's. "
                 "Name-only simulation still resolves by state and "
                 "cannot separate two same-named routes from a state "
                 "where only one variant applies; that is a property "
                 "of names, not a simulation failure, and the probe "
                 "records route_names_identical for it. The pattern "
                 "multiplier (x6.0) still applies to the NAME, so it "
                 "penalises the correct variant exactly as much as "
                 "the misfiring one -- that over-generalisation is "
                 "now caught by simulation, not by the multiplier, "
                 "and variant-aware pattern application is a Phase 10 "
                 "item. The duplicate-name 'cut' limitation is "
                 "resolved by the same fix: effects_fn resolves each "
                 "'cut' to the variant whose preconditions hold in "
                 "the current simulated state. Phase 8's note below "
                 "is superseded: simulation DOES catch the misfire "
                 "now (see predicted_success_differs)."),
    }


# ------------------------------------------------------------------ experiment 13
def exp_simulation_does_not_mutate() -> dict:
    """SS4.13: simulating leaves the real world model, state and LTM
    byte-identical (unit-level). ``imagine`` / ``what_if`` /
    ``counterfactual`` legitimately ADD events (simulation_compared /
    counterfactual are recorded by design) and keep the best
    scenario in ``last_imagination`` -- those are recorded here as
    designed side effects, not model mutation; the criterion is the
    model, state, LTM and planner bans."""
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    _run_treasure(brain)
    wm_before = json.dumps(brain.world_model.to_json(), sort_keys=True)
    world_before = json.dumps(dict(brain.world), sort_keys=True, default=str)
    ltm_before = json.dumps(
        [(f.subject, f.relation, f.value, f.status, sorted(f.sources))
         for f in brain.ltm.all_facts()], sort_keys=True)
    bans_before = len(brain.planner.banned_pairs)
    events_before = len(brain.events)
    _set_world(brain, TREASURE_START)
    brain.imagine(dict(TREASURE_TARGET))
    brain.what_if("what if I use pick_lock?")
    brain.counterfactual(edits={"door_open": True},
                         goal=dict(TREASURE_TARGET), steps=PICK_LOCK_ROUTE)
    brain.simulate_plan(PICK_LOCK_ROUTE, dict(TREASURE_TARGET))
    brain.world_model.check_plan(KEY_ROUTE, dict(TREASURE_START),
                                 brain.planner.effects_fn)
    # The imagination calls above set the world to the start state by
    # design (see _set_world); restore the comparison on the model,
    # state and LTM, not on the (deliberately reset) world dict.
    wm_after = json.dumps(brain.world_model.to_json(), sort_keys=True)
    ltm_after = json.dumps(
        [(f.subject, f.relation, f.value, f.status, sorted(f.sources))
         for f in brain.ltm.all_facts()], sort_keys=True)
    world_after_start = dict(brain.world)
    passed = bool(wm_before == wm_after and ltm_before == ltm_after
                  and len(brain.planner.banned_pairs) == bans_before)
    return {
        "criterion": ("world model and LTM byte-identical after "
                      "simulation; no new planner bans; the brain's "
                      "world after the calls is the imagination start "
                      "state (set deliberately, not mutated by "
                      "simulation)"),
        "passed": passed,
        "world_model_unchanged": wm_before == wm_after,
        "ltm_unchanged": ltm_before == ltm_after,
        "planner_bans_unchanged": len(brain.planner.banned_pairs) == bans_before,
        "world_after_calls_is_imagination_start_state":
            world_after_start == TREASURE_START,
        "events_added_by_design": len(brain.events) - events_before,
        "event_kinds_added": sorted({e.get("kind") for e in
                                     brain.events[events_before:]}),
        "last_imagination_kept": brain.last_imagination is not None,
        "dependency": None,
        "note": ("State is deliberately reset to the imagination "
                 "start state by the experiment itself (imagine / "
                 "what_if / counterfactual simulate from the brain's "
                 "current world), so the state comparison is that "
                 "the post-call world equals the start state, not "
                 "the pre-call end-of-task state."),
    }


# ------------------------------------------------------------------ runner
EXPERIMENTS = (
    ("imagine_routes", exp_imagine_routes),
    ("predict_consequences", exp_predict_consequences),
    ("compare_routes", exp_compare_routes),
    ("select_and_execute", exp_select_and_execute),
    ("imagined_vs_actual", exp_imagined_vs_actual),
    ("revision_second_task", exp_revision_second_task),
    ("what_if_query", exp_what_if_query),
    ("counterfactual_world_model", exp_counterfactual_world_model),
    ("document_perception", exp_document_perception),
    ("image_percept", exp_image_percept),
    ("specialist_absent", exp_specialist_absent),
    ("pd_overgeneralisation_probe", exp_pd_overgeneralisation_probe),
    ("simulation_does_not_mutate", exp_simulation_does_not_mutate),
)

# Fast mode runs the cheap subset: the workshop-based P-D probe is
# skipped (its W1+W2 runs are the slow part), not passed.
FAST_SKIP = ("pd_overgeneralisation_probe",)


def _strip_private(record: dict) -> dict:
    return {k: v for k, v in record.items() if not k.startswith("_")}


def run_phase8_evaluation(fast: bool = False,
                          record_path: str | None = DEFAULT_RECORD) -> dict:
    """Run the Phase 8 acceptance experiments. A failed criterion is
    a recorded result, never a crash: each experiment is wrapped so
    one exception cannot take down the rest. ``record_path=None``
    writes nothing (the harness path -- fast runs must never
    overwrite the canonical record)."""
    started = time.monotonic()
    record: dict = {
        "evaluation": "ab.eval.phase8_eval (Phase 8)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "fast" if fast else "full",
        "experiments": {},
        "failures": [],
        "worker_dependencies": [],
    }
    for name, fn in EXPERIMENTS:
        if fast and name in FAST_SKIP:
            record["experiments"][name] = {
                "skipped": True,
                "reason": "fast mode (harness --fast / unit tests)",
            }
            continue
        try:
            result = fn()
            result.setdefault("name", name)
            record["experiments"][name] = _strip_private(result)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            record["experiments"][name] = {
                "name": name,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "note": ("experiment failed; recorded here, not hidden "
                         "-- see the error above"),
            }
            record["failures"].append(f"{name}: {type(exc).__name__}: {exc}")
    deps = set()
    for section in record["experiments"].values():
        if isinstance(section, dict) and section.get("dependency"):
            deps.add(section["dependency"])
    record["worker_dependencies"] = sorted(deps)
    record["duration_s"] = round(time.monotonic() - started, 3)
    record["n_passed"] = sum(1 for s in record["experiments"].values()
                             if isinstance(s, dict) and s.get("passed") is True)
    record["n_failed"] = sum(1 for s in record["experiments"].values()
                             if isinstance(s, dict) and s.get("passed") is False)
    record["n_skipped"] = sum(1 for s in record["experiments"].values()
                              if isinstance(s, dict) and s.get("skipped"))
    record["notes"] = [
        ("Each experiment returns {name, passed, evidence}; a failed "
         "criterion is a recorded result, not a harness crash."),
        ("Experiments run through the designed W1/W2 public APIs "
         "(Brain.imagine / what_if / counterfactual / read_document "
         "/ perceive / Brain(specialist=...)); the 'api' field in "
         "each experiment says which path ran."),
        ("No evaluation answer is hardcoded: every number comes from "
         "running the brain on fixed worlds."),
    ]
    if record_path:
        try:
            os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
            with open(record_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        except OSError as exc:
            record["failures"].append(
                f"could not write record to {record_path}: {exc}")
    return record


def summary_text(record: dict) -> str:
    lines = [
        f"Phase 8 evaluation record -- {record['timestamp']} (mode: {record['mode']})",
        "",
        f"Experiments: {record.get('n_passed')} passed, "
        f"{record.get('n_failed')} failed, "
        f"{record.get('n_skipped')} skipped "
        f"({record.get('duration_s')}s)",
    ]
    for name, section in record.get("experiments", {}).items():
        if not isinstance(section, dict):
            continue
        if section.get("skipped"):
            lines.append(f"  {name}: SKIPPED ({section.get('reason')})")
        elif "passed" in section:
            lines.append(f"  {name}: {'PASS' if section['passed'] else 'FAIL'}")
            if section.get("dependency"):
                lines.append(f"    dependency: {section['dependency']}")
            if section.get("error"):
                lines.append(f"    error: {section['error']}")
    if record.get("failures"):
        lines.append("Failures recorded: " + "; ".join(record["failures"]))
    if record.get("worker_dependencies"):
        lines.append("Worker dependencies:")
        for dep in record["worker_dependencies"]:
            lines.append(f"  - {dep}")
    lines.append("")
    lines.append("Full record: docs/phase8-eval-record.json")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 evaluation")
    parser.add_argument("--record", default=None,
                        help="path of the JSON record to write (default: "
                             "docs/phase8-eval-record.json in full mode; "
                             "nothing is written in fast mode unless "
                             "--record is given explicitly, so a fast run "
                             "never overwrites the canonical full record)")
    parser.add_argument("--summary", default=None,
                        help="path of the summary text to write (default: "
                             "docs/phase8-eval-summary.txt in full mode; "
                             "nothing is written in fast mode unless "
                             "--summary is given explicitly)")
    parser.add_argument("--fast", action="store_true",
                        help="cheap subset only (skips the workshop P-D probe)")
    args = parser.parse_args(argv)
    record_path = (args.record if args.record is not None
                   else (None if args.fast else DEFAULT_RECORD))
    summary_path = (args.summary if args.summary is not None
                    else (None if args.fast else DEFAULT_SUMMARY))
    try:
        record = run_phase8_evaluation(fast=args.fast,
                                       record_path=record_path)
    except Exception as exc:  # noqa: BLE001 - harness error, not a brain failure
        print(f"phase 8 evaluation harness error: {type(exc).__name__}: {exc}")
        return 1
    if summary_path:
        try:
            with open(summary_path, "w", encoding="utf-8") as fh:
                fh.write(summary_text(record))
        except OSError as exc:
            print(f"phase 8 evaluation harness error: cannot write summary: {exc}")
            return 1
    print(summary_text(record))
    if record_path:
        print(f"record written to {record_path}")
    else:
        print("fast mode: no canonical record written "
              "(pass --record to write one explicitly)")
    if summary_path:
        print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
