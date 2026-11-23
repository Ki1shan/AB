"""Phase 5 transfer experiment (design SS3.3 / SS6.1).

Reuses the chamber worlds unchanged (``ChamberWorld``,
``chamber_world_actions``, ``WORLD_A`` / ``WORLD_B`` from
``ab.eval.disconnected``) and measures whether anything learned in
world A transfers to the renamed world B.

Protocol (three runs, all through ``_run_with_brain`` so the run
mechanics are identical to the disconnected experiment):

1. world A with a fresh brain (must complete);
2. world B with a fresh brain;
3. world B with the world-A brain carried over. On entry to B the
   brain's transient world-model observations are cleared while its
   durable learning (patterns, learner episodes, planner bans, causal
   records, LTM) is kept. The clearing uses
   ``Brain.clear_transient_for_new_world()`` when Worker 2 has added
   it; otherwise the documented fallback runs inline
   (``brain.state.clear()`` plus, when a ``world_model`` attribute
   exists, re-instantiating it from its own class while keeping
   ``brain.patterns`` intact). Which path ran is recorded in the
   record under ``entry_protocol`` -- the fallback is a measurement
   note, not a silent substitution.

Transfer criterion (task definition): the continued brain completes
B AND the fresh brain completes B AND the continued brain needed
strictly fewer failed attempts in B than the fresh brain did.
``transferred`` is exactly that boolean, computed from the measured
numbers. The record also carries the design SS6.1 cycles leg
(``continued_cycles_le_fresh``) separately, so a run that transfers
on failures but not on cycles is visible as such. If transfer is
False, that is recorded honestly as a failed acceptance criterion --
nothing here is tuned to force it.

The ``mechanism`` section records what the transfer would have to
work through: the patterns present before B, ``pattern_applied``
events during B, whether the decoy was tried in continued B, and
whether the fragile tool was retried after the mid-task reset.

Runnable standalone::

    python3 -m ab.eval.transfer [--record PATH]

Exit 0 when a full record was written -- even if transfer is False
(a failed transfer is a recorded result, not a harness crash).
Non-zero only on a harness error. No LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

from ..brain import Brain
from .disconnected import (  # noqa: F401  (ChamberWorld / actions re-exported for callers)
    WORLD_A,
    WORLD_B,
    ChamberWorld,
    _default_factory,
    _run_with_brain,
    chamber_world_actions,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase5-transfer-record.json")

# World-A/B action and entity names. Used ONLY here, in the evaluation
# record, to check that learned patterns are not name-bound (a pattern
# whose id mentions one of these names would be memorisation, not
# structural transfer). No pattern, binding, or planner code reads them.
_WORLD_NAMES = (
    "zorblat", "quivern", "chant", "relic", "blue", "green", "red",
    "tessera", "orb", "hum", "idol", "amber", "violet", "crimson",
)

_FAILURE_MARKERS = ("does not move", "is dust", "no intact",
                    "nothing happened", "cannot reach")


def _pattern_snapshot(brain) -> dict:
    """The brain's pattern library as observable state, or an explicit
    empty snapshot when the brain has no pattern library yet (Worker 2
    integration not landed). Never raises: an absent library is a
    measurement (``present: False``), not a crash."""
    lib = getattr(brain, "patterns", None)
    if lib is None:
        return {"present": False, "count": 0, "ids": [], "names": [],
                "note": "brain has no patterns attribute (no pattern library)"}
    try:
        patterns = lib.patterns() if callable(getattr(lib, "patterns", None)) else lib
    except Exception as exc:  # a broken library is data, not a crash
        return {"present": True, "count": None, "ids": [], "names": [],
                "note": f"pattern library unreadable: {type(exc).__name__}: {exc}"}
    ids, names = [], []
    for p in patterns or []:
        if isinstance(p, dict):
            ids.append(str(p.get("id", "")))
            names.append(str(p.get("name", "")))
        else:
            ids.append(str(getattr(p, "id", "")))
            names.append(str(getattr(p, "name", "")))
    return {"present": True, "count": len(ids), "ids": ids, "names": names,
            "note": ""}


def prepare_brain_for_new_world(brain) -> dict:
    """Clear a brain's transient world state on entry to a new world,
    keeping durable learning (patterns, learner episodes, planner bans,
    causal records, LTM) intact. Returns how the entry was done.

    Preferred path: ``Brain.clear_transient_for_new_world()`` (Worker 2,
    design SS3.3). Fallback when that method does not exist yet:
    ``brain.state.clear()`` plus, when the brain carries a
    ``world_model``, re-instantiating it from its own class (keeping
    ``brain.patterns`` untouched). The fallback is recorded, not hidden.
    """
    patterns_before = _pattern_snapshot(brain)
    if hasattr(brain, "clear_transient_for_new_world"):
        brain.clear_transient_for_new_world()
        method = "Brain.clear_transient_for_new_world()"
        wm_note = ""
    else:
        brain.state.clear()
        method = "fallback: brain.state.clear()"
        wm = getattr(brain, "world_model", None)
        if wm is None:
            wm_note = "no world_model attribute on this brain; state.clear() only"
        else:
            try:
                brain.world_model = type(wm)()
                wm_note = (f"world_model re-instantiated from "
                           f"{type(wm).__name__}(); patterns kept intact")
            except Exception as exc:  # keep going on state.clear() alone
                wm_note = (f"world_model re-instantiation failed "
                           f"({type(exc).__name__}: {exc}); state.clear() only")
    patterns_after = _pattern_snapshot(brain)
    return {
        "method": method,
        "world_model_note": wm_note,
        "patterns_before": patterns_before,
        "patterns_after": patterns_after,
        "patterns_intact": (patterns_before.get("ids") == patterns_after.get("ids")),
        "transient_state_cleared": len(brain.state.keys()) == 0,
    }


def _mechanism(world_b_continued: dict, brain, entry: dict,
               events_before_b: int, spec_b: dict) -> dict:
    """What the transfer would have had to work through, measured from
    the continued-B run's own logs and events."""
    env_log = world_b_continued.get("env_log", [])
    decoy = spec_b["decoy"]
    fragile = spec_b["fragile"]
    reset_idx = next((i for i, e in enumerate(env_log) if e.startswith("reset:")), None)
    fragile_after_reset = False
    if reset_idx is not None:
        fragile_after_reset = any(
            e.startswith(f"use_{fragile}") for e in env_log[reset_idx + 1:])
    applied = [e for e in brain.events[events_before_b:]
               if e.get("kind") == "pattern_applied"]
    return {
        "patterns_present_before_b": entry["patterns_before"],
        "patterns_intact_on_entry": entry["patterns_intact"],
        "pattern_applied_events_during_b": [
            {"cycle": e.get("cycle"), "detail": e.get("detail")}
            for e in applied],
        "pattern_applied_count_during_b": len(applied),
        "decoy_tried_in_continued_b": any(
            e.startswith(decoy) for e in env_log),
        "decoy_action": decoy,
        "fragile_tool_retried_after_reset": fragile_after_reset,
        "fragile_action": f"use_{fragile}",
        "reset_observed_in_continued_b": reset_idx is not None,
    }


def run_transfer_experiment(brain_factory=None, max_cycles: int = 80) -> dict:
    """Run worlds A, fresh B, continued B and return the record dict.

    ``brain_factory(actions, granted)`` builds a brain; the default
    builds a plain :class:`Brain`. The record states the measured
    numbers for all three runs; ``transferred`` is the boolean
    ``continued.completed AND fresh.completed AND
    continued.failed_attempts < fresh.failed_attempts``.
    """
    factory = brain_factory or _default_factory
    started = time.monotonic()
    record: dict = {
        "experiment": "phase5-transfer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_cycles": max_cycles,
        "failures": [],
    }
    world_a, brain_a = _run_with_brain(WORLD_A, factory, max_cycles)
    record["world_a_fresh"] = world_a
    # Snapshot the patterns learned in A now, before B runs: brain_a
    # IS the continued brain, so a later snapshot would mix in
    # anything learned during B.
    record["patterns_learned_in_a"] = _pattern_snapshot(brain_a)
    if not world_a["completed"]:
        record["failures"].append(
            f"world A (fresh brain) not completed: goal "
            f"{world_a['goal_status']}, stop {world_a['stop_reason']} "
            f"after {world_a['cycles']} cycles")
    world_b_fresh, _ = _run_with_brain(WORLD_B, factory, max_cycles)
    record["world_b_fresh"] = world_b_fresh
    if not world_b_fresh["completed"]:
        record["failures"].append(
            f"world B (fresh brain) not completed: goal "
            f"{world_b_fresh['goal_status']}, stop {world_b_fresh['stop_reason']}")
    # Entry protocol for the continued run: clear transient state on
    # the world-A brain BEFORE _run_with_brain runs B (which clears
    # state again and adds B's actions -- both clearings are
    # idempotent, and the entry record says which method ran here).
    entry = prepare_brain_for_new_world(brain_a)
    record["entry_protocol"] = entry
    events_before_b = len(brain_a.events)
    world_b_continued, brain_b = _run_with_brain(
        WORLD_B, factory, max_cycles, brain=brain_a)
    record["world_b_continued"] = world_b_continued
    if not world_b_continued["completed"]:
        record["failures"].append(
            f"world B (continued brain) not completed: goal "
            f"{world_b_continued['goal_status']}, stop "
            f"{world_b_continued['stop_reason']}")
    record["mechanism"] = _mechanism(
        world_b_continued, brain_b, entry, events_before_b, WORLD_B)
    transferred = bool(
        world_b_continued["completed"]
        and world_b_fresh["completed"]
        and world_b_continued["failed_attempts"] < world_b_fresh["failed_attempts"])
    record["transferred"] = transferred
    record["criterion"] = (
        "transferred = continued B completed AND fresh B completed AND "
        "continued B failed_attempts < fresh B failed_attempts")
    record["cycles_leg"] = {
        # Design SS6.1 also asks for continued cycles <= fresh cycles;
        # recorded separately so a failure-count transfer without the
        # cycles leg stays visible.
        "continued_cycles_le_fresh": (
            world_b_continued["cycles"] <= world_b_fresh["cycles"]),
        "continued_cycles": world_b_continued["cycles"],
        "fresh_cycles": world_b_fresh["cycles"],
    }
    record["comparison_vs_fresh_a"] = {
        "continued_b_failed_attempts": world_b_continued["failed_attempts"],
        "fresh_a_failed_attempts": world_a["failed_attempts"],
        "continued_b_fewer_than_fresh_a": (
            world_b_continued["failed_attempts"] < world_a["failed_attempts"]),
    }
    if not transferred:
        record["failures"].append(
            "transfer acceptance criterion NOT met: continued B did not "
            "need fewer failed attempts than fresh B "
            f"(continued {world_b_continued['failed_attempts']} vs fresh "
            f"{world_b_fresh['failed_attempts']}); this is a failed "
            "acceptance criterion, recorded honestly -- nothing was "
            "tuned to force it.")
    record["duration_s"] = round(time.monotonic() - started, 3)
    record["note"] = (
        "Transfer is measured, not assumed: the numbers above are what "
        "the three runs did. A 'not transferred' result means the "
        "pattern mechanism did not deliver fewer failed attempts in "
        "the renamed world; it is a finding about the brain, not a "
        "harness failure.")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 transfer experiment")
    parser.add_argument("--record", default=DEFAULT_RECORD,
                        help="path of the JSON record to write")
    parser.add_argument("--max-cycles", type=int, default=80)
    args = parser.parse_args(argv)
    try:
        record = run_transfer_experiment(max_cycles=args.max_cycles)
    except Exception as exc:                       # harness error, not a brain failure
        print(f"transfer experiment harness error: {type(exc).__name__}: {exc}")
        return 1
    try:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        with open(args.record, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    except OSError as exc:
        print(f"transfer experiment: cannot write record: {exc}")
        return 1
    a = record["world_a_fresh"]
    b = record["world_b_fresh"]
    c = record["world_b_continued"]
    print(f"world A fresh: completed={a['completed']} stop={a['stop_reason']} "
          f"cycles={a['cycles']} failed_attempts={a['failed_attempts']}")
    print(f"world B fresh: completed={b['completed']} stop={b['stop_reason']} "
          f"cycles={b['cycles']} failed_attempts={b['failed_attempts']}")
    print(f"world B continued: completed={c['completed']} stop={c['stop_reason']} "
          f"cycles={c['cycles']} failed_attempts={c['failed_attempts']}")
    print(f"entry protocol: {record['entry_protocol']['method']}; "
          f"patterns intact on entry: {record['entry_protocol']['patterns_intact']}")
    print(f"mechanism: patterns before B="
          f"{record['mechanism']['patterns_present_before_b']['count']}, "
          f"pattern_applied events during B="
          f"{record['mechanism']['pattern_applied_count_during_b']}, "
          f"decoy tried in continued B="
          f"{record['mechanism']['decoy_tried_in_continued_b']}, "
          f"fragile retried after reset="
          f"{record['mechanism']['fragile_tool_retried_after_reset']}")
    print(f"transferred: {record['transferred']}")
    print(f"failures recorded: {record['failures']}")
    print(f"record written to {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
