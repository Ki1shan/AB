"""Phase 5 acceptance tasks (design SS6.2-SS6.5 + transfer wrapper).

Each task runs in-process for the three brain variants (full /
no_learning / non_integrated -- see ``make_variant_factories`` in
``ab.eval.reasoning_tasks``) and returns ``{passed, cycles, legs,
evidence, detail}``. Numbers are recorded; no winner is declared.

Tasks:

- ``transfer_task`` -- thin wrapper over the Phase 5 transfer
  experiment (``ab.eval.transfer``). When the harness passes its
  transfer record, the ``full`` variant reuses it instead of
  re-running; the other variants run the experiment with their own
  factory.
- ``object_tracking`` -- a small invented two-room world defined HERE
  (new names: study/garden/cellar, the lantern). The lantern moves
  through >= 3 transitions (taken in the cellar, carried through the
  garden where it is unobserved, re-observed in the study); identity
  must be preserved (one take, same key/entity throughout) and the
  history order correct.
- ``prediction_revision`` -- in ``SurpriseWorld`` (from
  ``ab.eval.reasoning_tasks``) the brain predicts ``heat_vapour``
  makes the vapour hot, the environment contradicts it, and the
  model must be revised: a revision event exists, the old value is
  retained in the revision trail, and the next plan does not
  re-predict the old value (it warms the coil first, or otherwise
  does not start with a bare ``heat_vapour``).
- ``partial_observability`` -- chamber world A, where ``door_open``
  is unobservable from the start room: the goal must complete while
  no plan depends on assuming ``door_open == False``.
- ``unknown_vs_false`` -- ``ask()`` on something never taught returns
  not-known; a claim contradicted by a verified fact returns
  disputed, not unknown.

Where Worker 2's world model exists (``brain.world_model`` with
``entity``/``history``/``revisions``), the task asserts on it; where
it does not exist yet, the same legs are asserted on the
brain-observable equivalents that already exist (state entries, the
``revision`` / ``prediction_checked`` event trail) and the record
says which path was used (``world_model_path: True/False``). The
fallback measures the same behaviour through the older surface; it
does not weaken the leg.

Runnable standalone::

    python3 -m ab.eval.world_tasks [--record PATH]

Exit 0 when a record was written, even if a task fails (a failed
task is a recorded result, not a harness crash). No LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action
from .disconnected import WORLD_A, ChamberWorld, _run_with_brain, chamber_world_actions
from .reasoning_tasks import (
    SurpriseWorld,
    make_variant_factories,
    surprise_world_actions,
)
from .transfer import run_transfer_experiment

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase5-world-tasks-record.json")


# ------------------------------------------------------------------ helpers
def _plan_steps(detail: str) -> list[str]:
    """Action names from a ``plan`` event detail
    (``"a > b > c (cost N)"`` or a ``via subgoal`` variant)."""
    text = detail or ""
    if text.startswith("via subgoal"):
        text = text.split(": ", 1)[1] if ": " in text else ""
    text = text.split(" (cost")[0].strip()
    if not text or text == "(already true)":
        return []
    return [s.strip() for s in text.split(" > ") if s.strip()]


def _first_plan_after(brain, cycle: int) -> list[str] | None:
    """Steps of the first ``plan`` event at or after ``cycle``, or None
    when no plan event follows (the brain never replanned)."""
    for e in brain.events:
        if e.get("kind") == "plan" and e.get("cycle", 0) >= cycle:
            steps = _plan_steps(e.get("detail", ""))
            if steps:
                return steps
    return None


def _result(passed: bool, cycles, legs: dict, evidence: dict, detail: str) -> dict:
    return {"passed": bool(passed), "cycles": cycles, "legs": legs,
            "evidence": evidence, "detail": detail}


def _safe(fn, *args, **kwargs) -> dict:
    """Run one task/variant: a brain failure is recorded, never raised."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded as task evidence
        return _result(False, None, {},
                       {"error": f"{type(exc).__name__}: {exc}"},
                       f"task errored: {type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ transfer wrapper
def run_transfer_task(brain_factory, transfer_record=None) -> dict:
    """Thin wrapper reporting the transfer experiment numbers. With a
    precomputed record (the harness passes its own), the numbers are
    reported, not re-run; otherwise the experiment runs here."""
    if transfer_record is not None:
        rec = transfer_record
        reran = False
    else:
        rec = run_transfer_experiment(brain_factory=brain_factory)
        reran = True
    a, b, c = rec["world_a_fresh"], rec["world_b_fresh"], rec["world_b_continued"]
    cycles = a["cycles"] + b["cycles"] + c["cycles"]
    legs = {
        "fresh_a_completed": bool(a["completed"]),
        "fresh_b_completed": bool(b["completed"]),
        "continued_b_completed": bool(c["completed"]),
        "continued_fewer_failures_than_fresh_b": bool(
            c["failed_attempts"] < b["failed_attempts"]),
    }
    return _result(
        bool(rec["transferred"]), cycles, legs,
        {"transfer_record": {
            "world_a_fresh": {k: a[k] for k in
                              ("completed", "cycles", "failed_attempts")},
            "world_b_fresh": {k: b[k] for k in
                              ("completed", "cycles", "failed_attempts")},
            "world_b_continued": {k: c[k] for k in
                                  ("completed", "cycles", "failed_attempts")},
            "transferred": rec["transferred"],
            "reran_experiment_here": reran}},
        f"transfer {'PASS' if rec['transferred'] else 'FAIL'}: fresh A "
        f"{a['failed_attempts']} failures/{a['cycles']} cycles, fresh B "
        f"{b['failed_attempts']}/{b['cycles']}, continued B "
        f"{c['failed_attempts']}/{c['cycles']}")


# ------------------------------------------------------------------ object tracking
class LanternWorld(Environment):
    """Two-room invented world (study - garden - cellar) with one
    object, the lantern, in the cellar. The brain must fetch it and
    bring it back to the study. While the brain is in the garden the
    lantern is NOT observed (``has_lantern`` is omitted there), so the
    return leg is an unobserved interval: the brain must retain the
    belief it still holds the lantern. New names, defined only here."""

    ADJ = {"study": ["garden"], "garden": ["study", "cellar"],
           "cellar": ["garden"]}

    def __init__(self):
        self.s = {"at": "study", "has_lantern": False}
        self.log: list[str] = []
        self.observations: list[dict] = []

    def observe(self) -> dict:
        obs = dict(self.s)
        if self.s["at"] == "garden":
            obs.pop("has_lantern", None)  # unobserved in transit
        self.observations.append(dict(obs))
        return obs

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go":
            dest = args.get("room")
            if dest in self.ADJ[s["at"]]:
                s["at"], ok, info = dest, True, f"moved to {dest}"
            else:
                info = f"cannot reach {dest}"
        elif name == "take_lantern" and s["at"] == "cellar" \
                and not s["has_lantern"]:
            s["has_lantern"], ok, info = True, True, "took the lantern"
        self.log.append(f"{name}{args or ''}: {info}")
        return ok, info


def lantern_world_actions() -> list[Action]:
    return [
        Action("go_garden_from_study", {"at": "study"}, {"at": "garden"},
               args={"_do": "go", "room": "garden"}),
        Action("go_study_from_garden", {"at": "garden"}, {"at": "study"},
               args={"_do": "go", "room": "study"}),
        Action("go_cellar_from_garden", {"at": "garden"}, {"at": "cellar"},
               args={"_do": "go", "room": "cellar"}),
        Action("go_garden_from_cellar", {"at": "cellar"}, {"at": "garden"},
               args={"_do": "go", "room": "garden"}),
        Action("take_lantern", {"at": "cellar", "has_lantern": False},
               {"has_lantern": True}, args={"_do": "take_lantern"}),
    ]


def run_object_tracking_task(brain_factory, max_cycles: int = 40) -> dict:
    """The lantern moves through >= 3 transitions: taken in the cellar,
    carried through the garden unobserved, re-observed in the study.
    Identity is preserved (one take, same key/entity throughout) and
    the history order is correct."""
    env = LanternWorld()
    brain = brain_factory(lantern_world_actions(), {"compute", "environment"})
    brain.add_goal("bring the lantern to the study",
                   {"has_lantern": True, "at": "study"})
    cycle_before = brain.cycle
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[-1]
    cycles = brain.cycle - cycle_before
    # Transition order from the env log: take < return through garden
    # < arrive in the study.
    log = env.log
    take_idx = next((i for i, e in enumerate(log)
                     if e.startswith("take_lantern")), None)
    garden_return_idx = next(
        (i for i, e in enumerate(log)
         if i > (take_idx or -1) and "go{'room': 'garden'}" in e), None)
    study_idx = next(
        (i for i, e in enumerate(log)
         if i > (garden_return_idx or -1) and "go{'room': 'study'}" in e), None)
    order_ok = (take_idx is not None and garden_return_idx is not None
                and study_idx is not None
                and take_idx < garden_return_idx < study_idx)
    # The unobserved interval: an observation in the garden after the
    # take that omits has_lantern, followed later by an observation in
    # the study that shows it True again.
    take_obs = next((i for i, o in enumerate(env.observations)
                     if o.get("has_lantern") is True), None)
    gap_obs = [i for i, o in enumerate(env.observations)
               if o.get("at") == "garden" and "has_lantern" not in o]
    gap_after_take = [i for i in gap_obs
                      if take_obs is not None and i > take_obs]
    reobserved = any(o.get("at") == "study" and o.get("has_lantern") is True
                     for o in env.observations)
    identity_state = (
        brain.world.get("has_lantern") is True
        and sum(1 for e in log if e.startswith("take_lantern")) == 1)
    # World-model path when Worker 2's model exists: the lantern entity
    # keeps one id, its history is ordered, its last location is the study.
    wm = getattr(brain, "world_model", None)
    wm_path = wm is not None and hasattr(wm, "entity") and hasattr(wm, "history")
    wm_evidence: dict = {"world_model_path": bool(wm_path)}
    identity_wm = None
    if wm_path:
        try:
            ent = wm.entity("lantern")
            # History is keyed by env key: has_lantern False -> True is
            # the take transition; location comes from believed_location
            # (the located_in relation, refreshed as the agent moves).
            hist = wm.history("lantern", "has_lantern") if ent is not None else []
            believed = (wm.believed_location("lantern")
                        if hasattr(wm, "believed_location") else (None, 0.0, None))
            wm_evidence.update({
                "entity_found": ent is not None,
                "entity_id": getattr(ent, "id", None) if ent is not None else None,
                "history_length": len(hist),
                "history": [str(h) for h in hist],
                "believed_location": (believed[0], believed[1], str(believed[2])),
            })
            identity_wm = bool(
                ent is not None and len(hist) >= 2
                and believed[0] == "study")
        except Exception as exc:  # a broken model read is evidence, not a crash
            wm_evidence["error"] = f"{type(exc).__name__}: {exc}"
            wm_path = False
    legs = {
        "goal_completed": goal.status == "done",
        "history_order_correct": bool(order_ok),
        "unobserved_interval_survived": bool(gap_after_take and reobserved),
        "identity_preserved": bool(identity_wm if wm_path else identity_state),
    }
    passed = all(legs.values())
    evidence = {
        "stop_reason": stop, "goal_status": goal.status,
        "take_index": take_idx, "garden_return_index": garden_return_idx,
        "study_index": study_idx,
        "observations_total": len(env.observations),
        "observations_missing_has_lantern": len(gap_obs),
        "reobserved_in_study": reobserved,
        "final_has_lantern": brain.world.get("has_lantern"),
        "take_events": sum(1 for e in log if e.startswith("take_lantern")),
        "env_log": list(log),
    }
    evidence.update(wm_evidence)
    return _result(
        passed, cycles, legs, evidence,
        f"lantern: goal {goal.status} in {cycles} cycles; take->garden->"
        f"study order {'ok' if order_ok else 'WRONG'}; unobserved gap "
        f"{'survived' if gap_after_take and reobserved else 'NOT survived'}; "
        f"identity {'preserved' if legs['identity_preserved'] else 'NOT preserved'}"
        f" (world-model path: {wm_path})")


# ------------------------------------------------------------------ prediction revision
def run_prediction_revision_task(brain_factory, max_cycles: int = 30) -> dict:
    """In SurpriseWorld the brain predicts ``heat_vapour`` makes the
    vapour hot; the environment contradicts it. The model must be
    revised: a revision event exists, the old value is retained in
    the revision trail, and the next plan does not re-predict the old
    value (it warms the coil first, or otherwise does not start with
    a bare ``heat_vapour`` while the coil is cold)."""
    env = SurpriseWorld()
    brain = brain_factory(surprise_world_actions(), {"compute", "environment"})
    brain.add_goal("heat the vapour", {"vapour_hot": True})
    cycle_before = brain.cycle
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[-1]
    cycles = brain.cycle - cycle_before
    # Revision event: Phase 5 "model_revised" when the world model
    # exists; the Phase 4 "revision" event is the same behaviour on
    # the older surface.
    revision_events = [e for e in brain.events
                       if e.get("kind") in ("model_revised", "revision")]
    revision_kind = revision_events[0].get("kind") if revision_events else None
    first_revision_cycle = (revision_events[0].get("cycle")
                            if revision_events else None)
    # Old value retained: world-model revisions list when present;
    # otherwise the violated-prediction event trail, which records
    # "predicted vapour_hot=True actual=False".
    wm = getattr(brain, "world_model", None)
    # Worker 1's WorldModel exposes revisions() as a METHOD returning
    # the list; accept a plain list attribute too, for robustness.
    _revs = getattr(wm, "revisions", None) if wm is not None else None
    wm_revisions = _revs() if callable(_revs) else _revs
    wm_path = isinstance(wm_revisions, list)
    old_retained = False
    old_retained_via = None
    if wm_path:
        old_retained = any(
            isinstance(r, dict)
            and str(r.get("key", "")).endswith("vapour_hot")
            and r.get("old_value") is True for r in wm_revisions)
        old_retained_via = "world_model.revisions" if old_retained else None
    if not old_retained:
        violated = [e for e in brain.events
                    if e.get("kind") == "prediction_checked"
                    and "violated" in e.get("detail", "")]
        old_retained = any("predicted vapour_hot=True" in e.get("detail", "")
                           and "actual=False" in e.get("detail", "")
                           for e in violated)
        if old_retained:
            old_retained_via = "prediction_checked violated event trail"
    # Next plan after the first revision must not re-predict the old
    # value: its first step is not a bare heat_vapour, or warm_coil
    # comes before heat_vapour in it.
    next_plan = (_first_plan_after(brain, first_revision_cycle)
                 if first_revision_cycle is not None else None)
    plan_revised = False
    if next_plan:
        plan_revised = bool(
            next_plan[0] != "heat_vapour"
            or ("warm_coil" in next_plan
                and next_plan.index("warm_coil") < next_plan.index("heat_vapour")))
    legs = {
        "revision_event_recorded": len(revision_events) > 0,
        "old_value_retained_in_revisions": bool(old_retained),
        "next_plan_does_not_repredict_old_value": bool(plan_revised),
    }
    passed = all(legs.values())
    return _result(
        passed, cycles, legs,
        {"stop_reason": stop, "goal_status": goal.status,
         "goal_completed": goal.status == "done",
         "revision_events": len(revision_events),
         "revision_event_kind": revision_kind,
         "first_revision_cycle": first_revision_cycle,
         "old_value_retained_via": old_retained_via,
         "world_model_path": bool(wm_path),
         "next_plan_after_revision": next_plan,
         "env_log": list(env.log)},
        f"prediction revision: {len(revision_events)} revision event(s)"
        f" ({revision_kind}); old value "
        f"{'retained' if old_retained else 'NOT retained'}"
        f" via {old_retained_via}; next plan {next_plan} "
        f"{'revised' if plan_revised else 'RE-PREDICTS the old value'}")


# ------------------------------------------------------------------ partial observability
def run_partial_observability_task(brain_factory, max_cycles: int = 80) -> dict:
    """Chamber world A: ``door_open`` is unobservable from the start
    room. The goal must complete while no plan depends on assuming
    ``door_open == False`` -- checked on the plans themselves (every
    step's preconditions, from the believed action model)."""
    rec, brain = _run_with_brain(WORLD_A, brain_factory, max_cycles)
    actions = {a.name: a for a in chamber_world_actions(WORLD_A)}
    # door_open is unobservable from the start room by construction.
    start_obs = ChamberWorld(WORLD_A).observe()
    door_unobserved = "door_open" not in start_obs
    plan_events = [e for e in brain.events if e.get("kind") == "plan"]
    first_steps = _plan_steps(plan_events[0].get("detail", "")) if plan_events else []

    def _assumes_closed(steps: list[str]) -> list[str]:
        bad = []
        for name in steps:
            a = actions.get(name)
            if a is not None and a.pre.get("door_open") is False:
                bad.append(name)
        return bad

    first_bad = _assumes_closed(first_steps)
    all_bad = []
    for e in plan_events:
        all_bad.extend(_assumes_closed(_plan_steps(e.get("detail", ""))))
    legs = {
        "goal_completed": bool(rec["completed"]),
        "door_unobserved_from_start_room": bool(door_unobserved),
        "first_plan_does_not_assume_door_closed": not first_bad,
        "no_plan_assumes_door_closed": not all_bad,
    }
    passed = all(legs.values())
    return _result(
        passed, rec["cycles"], legs,
        {"stop_reason": rec["stop_reason"], "goal_status": rec["goal_status"],
         "start_room_observation_keys": sorted(start_obs),
         "first_plan_steps": first_steps,
         "first_plan_assuming_closed": first_bad,
         "all_plans_assuming_closed": sorted(set(all_bad)),
         "plan_events": len(plan_events)},
        f"partial observability: goal {rec['goal_status']} in "
        f"{rec['cycles']} cycles; door unobserved at start: "
        f"{door_unobserved}; first plan {first_steps} "
        f"{'assumes nothing about the door' if not first_bad else 'ASSUMES door closed: ' + str(first_bad)}")


# ------------------------------------------------------------------ unknown vs false
UNKNOWN_SUBJECT = "quorble"
UNKNOWN_RELATION = "is_a"
FALSE_SUBJECT = "morvain"
FALSE_RELATION = "melts_at"
FALSE_VALUES = ("50c", "40c")


def run_unknown_vs_false_task(brain_factory) -> dict:
    """``ask()`` on something never taught returns not-known (honest
    'I don't know'); a claim contradicted by a verified fact returns
    disputed -- both values, neither silently picked -- not unknown."""
    brain = brain_factory([], {"compute"})
    cycle_before = brain.cycle
    unknown = brain.ask(UNKNOWN_SUBJECT, UNKNOWN_RELATION)
    brain.teach_fact(FALSE_SUBJECT, FALSE_RELATION, FALSE_VALUES[0], "teacher")
    brain.teach_fact(FALSE_SUBJECT, FALSE_RELATION, FALSE_VALUES[1], "teacher")
    false_answer = brain.ask(FALSE_SUBJECT, FALSE_RELATION)
    facts = brain.ltm.query(FALSE_SUBJECT, FALSE_RELATION,
                            statuses=("verified", "unverified", "disputed",
                                      "retracted"))
    disputed = sorted(f.value for f in facts if f.status == "disputed")
    asked_values = sorted(v for v, _, _, _ in false_answer.answers)
    asked_statuses = sorted({s for _, _, s, _ in false_answer.answers})
    legs = {
        "unknown_returns_not_known": bool(
            not unknown.known and unknown.answers == []),
        "false_claim_returns_disputed_not_unknown": bool(
            false_answer.known
            and disputed == sorted(FALSE_VALUES)
            and asked_values == sorted(FALSE_VALUES)
            and "disputed" in asked_statuses),
    }
    passed = all(legs.values())
    return _result(
        passed, brain.cycle - cycle_before, legs,
        {"unknown_ask": {"known": unknown.known,
                         "answers": unknown.answers,
                         "message": unknown.message},
         "false_ask": {"known": false_answer.known,
                       "answers": [(v, c, s) for v, c, s, _ in false_answer.answers],
                       "message": false_answer.message},
         "disputed_values": disputed,
         "fact_statuses": sorted((f.value, f.status) for f in facts)},
        f"unknown vs false: unknown ask known={unknown.known} "
        f"({unknown.message}); false ask known={false_answer.known}, "
        f"values {asked_values}, statuses {asked_statuses}")


# ------------------------------------------------------------------ all tasks, all variants
TASKS = (
    ("transfer_task", run_transfer_task,
     "Transfer experiment numbers (Phase 5 transfer): pass = continued "
     "B completes and needs fewer failed attempts than fresh B."),
    ("object_tracking", run_object_tracking_task,
     "Lantern world (study/garden/cellar): pass = goal completes, the "
     "lantern's history order is correct (take, unobserved transit, "
     "re-observed), identity preserved."),
    ("prediction_revision", run_prediction_revision_task,
     "SurpriseWorld contradiction: pass = revision event recorded, old "
     "value retained in revisions, next plan does not re-predict it."),
    ("partial_observability", run_partial_observability_task,
     "Chamber world A: pass = goal completes and no plan assumes "
     "door_open == False (door unobservable from the start room)."),
    ("unknown_vs_false", run_unknown_vs_false_task,
     "ask(): pass = unknown returns not-known; a claim contradicted "
     "by a verified fact returns disputed, not unknown."),
)


def run_world_tasks(brain_factories=None, transfer_record=None) -> dict:
    """Run the five acceptance tasks for each variant and return the
    record section (per-variant numbers, no winner declared). A brain
    failure in one task/variant is recorded, never raised. When the
    harness passes its transfer record, the ``full`` variant's
    transfer task reuses it instead of re-running the experiment."""
    factories = brain_factories or make_variant_factories()
    started = time.monotonic()
    tasks: dict = {}
    for task_id, runner, criterion in TASKS:
        variants: dict = {}
        for name, factory in factories.items():
            if task_id == "transfer_task" and name == "full" \
                    and transfer_record is not None:
                variants[name] = _safe(runner, factory, transfer_record)
            else:
                variants[name] = _safe(runner, factory)
        tasks[task_id] = {"criterion": criterion, "variants": variants}
    return {
        "mechanism": (
            "Phase 5 acceptance tasks (design SS6), defined in "
            "ab/eval/world_tasks.py only; the brain was never taught "
            "these worlds or names. Variants are built by "
            "monkeypatching/subclassing Brain (brain.py NOT edited), "
            "as in the harness baselines: no_learning replaces "
            "brain.learner with a no-op; non_integrated overrides "
            "_reason_from_wm with a no-op. No winner is declared -- "
            "the numbers below are the comparison."),
        "tasks": tasks,
        "duration_s": round(time.monotonic() - started, 3),
        "note": ("A task the full brain fails is a finding about the "
                 "brain, not a harness crash; the harness still exits "
                 "0 when this record is written."),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 world-model acceptance tasks")
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help="path of the JSON record to write")
    args = parser.parse_args(argv)
    try:
        section = run_world_tasks()
    except Exception as exc:                       # harness error, not a brain failure
        print(f"world tasks harness error: {type(exc).__name__}: {exc}")
        return 1
    record = {
        "experiment": "phase5-world-tasks",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "world_tasks": section,
    }
    try:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        with open(args.record, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    except OSError as exc:
        print(f"world tasks: cannot write record: {exc}")
        return 1
    for task_id, task in section["tasks"].items():
        print(f"{task_id}: {task['criterion']}")
        for name, res in task["variants"].items():
            print(f"  {name}: {'PASS' if res['passed'] else 'FAIL'} "
                  f"({res.get('cycles')} cycles) -- {res['detail']}")
    print("No winner is declared -- the numbers above are the comparison.")
    print(f"record written to {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
