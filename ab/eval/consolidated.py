"""Consolidated evaluation (introduced in Phase 9; design docs/phase9-design.md SS3.1).

One entry point runs the whole evaluation programme and writes one
machine-readable record plus one human-readable summary::

    python -m ab.eval.consolidated [--seed N] [--record PATH] [--summary PATH]

and ``run_consolidated_evaluation(seed=...) -> dict``.

Common result format for every suite::

    {suite, experiments: {name: {passed, cycles, detail, evidence,
                                 failure_cause}},
     n_passed, n_failed, n_skipped, duration_s, seed}

Runs, in order: discriminating tasks (Phase 4); NEW reasoning tasks
(``ab.eval.reasoning_tasks2`` -- Worker 1; imported defensively, an
absent module fails that suite with the reason, it never crashes the
runner); transfer (Phase 5) + transfer on newly generated
environments (``ab.eval.envgen`` -- Worker 1; same defensive
import); world tasks (Phase 5); learning eval (Phase 6) + learning
curves (SS3.4); execution eval (Phase 7); Phase 8 eval (incl. the
P-D probe, which must pass after the SS4.1 fix); long-horizon task
(``ab.eval.long_horizon`` -- Worker 4; defensive import); perception
eval (consolidated OCR/document/image checks incl. the noisy-OCR
result recorded by Worker 4); memory eval (retention after restart,
retrieval, consolidation -- Phase 6 mechanisms, named suite);
tool-use eval (execution legs + RLIMIT_CPU burner result);
planning eval (named suite reusing discriminating + world-task
numbers, no new planner); calibration consolidation (SS3.6); failure
analysis (SS3.7); regression tracking (SS3.8); leakage audit (SS3.9).

Exit code: non-zero if ANY suite fails, and the summary says so
visibly (design P9-4). A successful report-writing process must not
be confused with a successful evaluation. Exit 0 only when
everything passes.

The unit suite is NOT re-run inside (the harness owns it; the
record cites the harness record's test counts -- see
``_cited_test_counts``). Every experiment records its seed and the
persistence version; config, line/test counts, tesseract version
and Python version are recorded per suite and at record level.
``record_path=None`` writes nothing (the harness path: a fast
harness run must never overwrite the canonical Phase 9 record).
No LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from ..brain import Brain
from ..core.learning import Learner
from ..imagination.simulator import laplace_rate
from ..reasoning.causal import NoOpCausal
from .regression import build_regression_record, write_regression_record

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Parent-verification fix (Phase 10, 2026-11-21): the defaults used to
# point at docs/phase9-eval-record.json / docs/phase9-eval-summary.txt,
# so a bare `python3 -m ab.eval.consolidated --seed 0` overwrote the
# canonical Phase 9 record with current-run numbers during the Phase 10
# gate (see docs/phase9-record-clobber-note.md). The defaults are now
# phase-neutral current-run files; a phase's canonical record is
# written only when its path is passed explicitly.
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "consolidated-eval-record.json")
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "docs", "consolidated-eval-summary.txt")
NOISY_OCR_SIDECAR = os.path.join(REPO_ROOT, "docs", "phase9-noisy-ocr.json")
CPU_BURNER_SIDECAR = os.path.join(REPO_ROOT, "docs", "phase9-cpu-burner.json")

# Evaluation-world/task names that must never appear in ab/ outside
# ab/eval/ (design SS3.9). The invented names from the Phase 4/5
# held-out tasks and worlds; new-task names from reasoning_tasks2
# are added at audit time from that module's UPPER_CASE constants.
LEAK_NAMES = ("velkra", "torvin", "sarnak", "kelmor", "lyssan",
              "zorblat", "quivern", "tessera")
FAILURE_MARKERS = ("does not move", "is dust", "no intact",
                   "nothing happened", "cannot reach")


# ------------------------------------------------------------------ environment meta
def _tesseract_version() -> str:
    try:
        proc = subprocess.run(["tesseract", "--version"], capture_output=True,
                              text=True, timeout=10)
        first = (proc.stdout or "").strip().splitlines()
        return first[0] if first else "tesseract present, version unreadable"
    except (OSError, subprocess.SubprocessError):
        return "not available"


def _line_counts() -> dict:
    """Python line/file counts for ab/ + tests/ (the code under
    evaluation). __pycache__ is skipped; unreadable files count 0."""
    files = lines = 0
    for base in ("ab", "tests"):
        for root, dirs, names in os.walk(os.path.join(REPO_ROOT, base)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                if not name.endswith(".py"):
                    continue
                files += 1
                try:
                    with open(os.path.join(root, name), encoding="utf-8") as fh:
                        lines += sum(1 for _ in fh)
                except OSError:
                    pass
    return {"python_files": files, "python_lines": lines}


def _cited_test_counts() -> dict:
    """Test counts cited from the harness record -- the unit suite is
    NOT re-run inside this runner. Reads
    docs/phase8-harness-record.json when present; otherwise records
    honestly that the suite was not run in this process."""
    path = os.path.join(REPO_ROOT, "docs", "phase8-harness-record.json")
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        t = rec.get("tests") or {}
        return {"ran": t.get("ran"), "failures": t.get("failures"),
                "errors": t.get("errors"), "skipped": t.get("skipped"),
                "ok": t.get("ok"),
                "mode": rec.get("mode"),
                "source": "docs/phase8-harness-record.json (cited, not re-run)",
                "status": ("cited from harness record"
                           + (f" (mode: {rec.get('mode')})"
                              if rec.get("mode") else ""))}
    except (OSError, ValueError) as exc:
        return {"ran": None, "failures": None, "errors": None,
                "skipped": None, "ok": None, "mode": None,
                "source": "docs/phase8-harness-record.json unreadable: "
                          f"{type(exc).__name__}: {exc}",
                "status": "not run in this process"}


def build_environment_meta(seed: int) -> dict:
    """Seed, config, persistence version, line/test counts,
    tesseract version, Python version -- recorded once per run and
    referenced by every suite (each experiment also carries its seed
    and the persistence version)."""
    return {
        "seed": int(seed),
        "config": ("full consolidated programme; per-phase evals run "
                   "in full mode unless the runner was invoked with "
                   "--fast (see the per-suite mode fields)"),
        "persistence_version": Brain.PERSISTENCE_VERSION,
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "tesseract_version": _tesseract_version(),
        "line_counts": _line_counts(),
        "test_counts": _cited_test_counts(),
    }


# ------------------------------------------------------------------ JSON hygiene
def _jsonable(value, _depth: int = 0):
    """Coerce an evidence value to plain JSON types (str() fallback
    for anything else, e.g. a Proof object carried in a task
    record). Depth-capped so a pathological structure cannot recurse
    forever."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth > 6:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, _depth + 1) for v in value]
    return str(value)


# ------------------------------------------------------------------ common result format
def _new_suite(name: str, seed: int, mode: str, meta: dict) -> dict:
    return {"suite": name, "experiments": {}, "n_passed": 0,
            "n_failed": 0, "n_skipped": 0, "duration_s": 0.0,
            "seed": int(seed), "mode": mode,
            "environment": {"persistence_version": meta["persistence_version"],
                            "python_version": meta["python_version"],
                            "tesseract_version": meta["tesseract_version"],
                            "line_counts": meta["line_counts"],
                            "test_counts": meta["test_counts"]}}


def _infer_failure_cause(name: str, detail: str) -> str:
    """A failure_cause for a failed experiment: mechanism name +
    one-line cause where known, else the honest default. Call sites
    pass an explicit cause when they know the mechanism better."""
    if "pd_overgeneralisation" in name or "pd_probe" in name:
        return ("simulation revision / causal variant resolution: the "
                "simulated ranking did not separate the "
                "press-with-mould route from the press-without-mould "
                "route (Phase 8 SS11 exp 12; Phase 9 SS4.1 fix)")
    if "retirement" in name:
        return ("pattern retirement: Brain.retire_pattern / the "
                "contradicting-evidence trigger did not fire in the "
                "workshop run (Phase 9 SS4.2)")
    if "long_horizon" in name:
        return ("long-horizon task: goal not completed, a stage out "
                "of order, or the resume leg did not complete")
    return "unknown -- needs investigation"


def _add_exp(suite: dict, name: str, passed: bool | None,
             cycles=None, detail: str = "", evidence=None,
             failure_cause: str | None = None, skipped: bool = False,
             skip_reason: str | None = None, meta: dict | None = None) -> None:
    """Add one experiment in the common format. ``passed=None`` with
    ``skipped=True`` records a skip (a result that was not run, never
    presented as a pass); a failed experiment always carries a
    failure_cause."""
    if skipped:
        suite["experiments"][name] = {
            "passed": None, "skipped": True, "cycles": cycles,
            "detail": skip_reason or detail,
            "evidence": _jsonable(evidence or {}),
            "failure_cause": None,
            "seed": suite["seed"],
            "persistence_version": (meta or {}).get("persistence_version"),
            "python_version": (meta or {}).get("python_version"),
            "tesseract_version": (meta or {}).get("tesseract_version")}
        suite["n_skipped"] += 1
        return
    ok = bool(passed)
    suite["experiments"][name] = {
        "passed": ok, "cycles": cycles,
        "detail": detail,
        "evidence": _jsonable(evidence or {}),
        "failure_cause": (None if ok else
                          (failure_cause or _infer_failure_cause(name, detail))),
        "seed": suite["seed"],
        "persistence_version": (meta or {}).get("persistence_version"),
        "python_version": (meta or {}).get("python_version"),
        "tesseract_version": (meta or {}).get("tesseract_version")}
    if ok:
        suite["n_passed"] += 1
    else:
        suite["n_failed"] += 1


def _finalise_suite(suite: dict, started: float) -> dict:
    suite["duration_s"] = round(time.monotonic() - started, 3)
    return suite


def _suite_error(name: str, seed: int, mode: str, meta: dict,
                 exc: Exception, started: float) -> dict:
    """A suite whose runner raised: recorded as a failed suite, never
    a runner crash. The error is the suite's failure, with its cause."""
    suite = _new_suite(name, seed, mode, meta)
    _add_exp(suite, "suite_error", False, detail=f"suite runner raised: {exc}",
             evidence={"error": f"{type(exc).__name__}: {exc}"},
             failure_cause=f"harness error in suite {name}: "
                           f"{type(exc).__name__}: {exc}", meta=meta)
    return _finalise_suite(suite, started)


# ------------------------------------------------------------------ task-module adapter
def _tasks_to_suite(name: str, tasks: dict, seed: int, mode: str,
                    meta: dict, started: float) -> dict:
    """Convert a per-task/per-variant record (the shape
    ``run_discriminating_tasks`` / ``run_world_tasks`` return) to the
    common format. One experiment per task for the FULL variant (the
    brain as built must pass the task), plus a variant-separation
    experiment (the tasks must separate the variants -- a task every
    variant passes identically discriminates nothing). Per-variant
    numbers live in the evidence; no winner is declared."""
    suite = _new_suite(name, seed, mode, meta)
    separation: dict = {}
    for task_id, task in (tasks or {}).items():
        variants = task.get("variants") or {}
        full = variants.get("full") or {}
        _add_exp(suite, f"{task_id}/full",
                 passed=bool(full.get("passed")),
                 cycles=full.get("cycles"),
                 detail=full.get("detail", ""),
                 evidence={"criterion": task.get("criterion"),
                           "variants": variants},
                 meta=meta)
        outcomes = {v: bool((r or {}).get("passed"))
                    for v, r in variants.items()}
        separation[task_id] = outcomes
    separates = any(len(set(outcomes.values())) > 1
                    for outcomes in separation.values())
    _add_exp(suite, "variant_separation", passed=separates,
             detail=("at least one task has differing outcomes across "
                     "variants" if separates else
                     "no task separates the variants -- every variant "
                     "passed/failed identically"),
             evidence={"per_task_variant_outcomes": separation},
             failure_cause=("discriminating tasks: no task separates "
                            "the variants -- the tasks do not "
                            "discriminate (see Phase 4 design)") if not separates else None,
             meta=meta)
    return _finalise_suite(suite, started)


def _run_task_module(module, factories: dict, seed: int, mode: str,
                     meta: dict, suite_name: str) -> dict:
    """Run every task in a reasoning-tasks-style module (``TASKS`` of
    (id, runner, criterion)) for each variant factory, and convert
    to the common format. A brain failure in one task/variant is
    recorded, never raised."""
    started = time.monotonic()
    tasks: dict = {}
    for task_id, runner, criterion in getattr(module, "TASKS", ()):
        variants: dict = {}
        for vname, factory in factories.items():
            try:
                variants[vname] = runner(factory)
            except Exception as exc:  # a brain failure is data, not a crash
                variants[vname] = {
                    "passed": False, "cycles": None, "legs": {},
                    "evidence": {"error": f"{type(exc).__name__}: {exc}"},
                    "detail": f"task errored: {type(exc).__name__}: {exc}"}
        tasks[task_id] = {"criterion": criterion, "variants": variants}
    return _tasks_to_suite(suite_name, tasks, seed, mode, meta, started)


def _adapt_unknown_result(name: str, result, seed: int, mode: str,
                          meta: dict, started: float) -> dict:
    """Convert an unknown-shaped result from a defensively imported
    module (long-horizon, reasoning_tasks2) to the common format.
    Recognised shapes: a dict with ``passed``; a dict with an
    ``experiments`` mapping of {name: {passed, ...}}; a dict with a
    ``tasks`` mapping (per-task/per-variant). Anything else is a
    recorded failure with the reason, never a crash."""
    suite = _new_suite(name, seed, mode, meta)
    if isinstance(result, dict) and "passed" in result:
        _add_exp(suite, name, passed=bool(result.get("passed")),
                 cycles=result.get("cycles"),
                 detail=str(result.get("detail", "")),
                 evidence={k: v for k, v in result.items()
                           if k not in ("passed", "cycles", "detail")},
                 meta=meta)
    elif isinstance(result, dict) and isinstance(result.get("experiments"), dict):
        for exp_name, section in result["experiments"].items():
            if not isinstance(section, dict):
                continue
            if section.get("skipped"):
                _add_exp(suite, exp_name, None, skipped=True,
                         skip_reason=str(section.get("reason", "")),
                         evidence=section, meta=meta)
            elif "passed" in section:
                _add_exp(suite, exp_name, passed=bool(section.get("passed")),
                         cycles=section.get("cycles"),
                         detail=str(section.get("detail", "")),
                         evidence=section, meta=meta)
    elif isinstance(result, dict) and isinstance(result.get("tasks"), dict):
        return _tasks_to_suite(name, result["tasks"], seed, mode, meta, started)
    else:
        _add_exp(suite, f"{name}/result_shape", False,
                 detail="defensively imported module returned an "
                        "unrecognised result shape",
                 evidence={"result_type": type(result).__name__,
                           "result_keys": (list(result.keys())
                                          if isinstance(result, dict) else None)},
                 failure_cause="unknown -- needs investigation: the "
                               "module's result shape is not one this "
                               "runner knows how to read",
                 meta=meta)
    return _finalise_suite(suite, started)


def _call_no_arg(fn, seed: int):
    """Call a defensively imported entry point: no args first, then
    with ``seed=`` on TypeError. Returns the result."""
    try:
        return fn()
    except TypeError:
        return fn(seed=seed)


# ------------------------------------------------------------------ ablations (monkeypatching only; brain.py untouched)
class _NoOpRevision:
    """A SimulationRevision-shaped no-op: ``predicted_success`` always
    falls back to the causal Laplace rate (never an adjusted rate),
    ``observe`` records nothing and moves nothing, ``compare``
    attributes mismatches without observing them. Replacing a
    brain's ``simulation_revision`` with this removes exactly one
    mechanism -- the simulation-revision loop -- and nothing else:
    causal rates, planner costs, patterns and the world model are
    untouched. ``adjusted_success`` / ``history`` stay present (and
    empty) so any code reading them keeps working."""

    def __init__(self):
        self.alpha = 0.5
        self.adjusted_success: dict = {}
        self.history: list = []

    def predicted_success(self, action: str, effect: str, causal=None,
                          variant=None):
        # variant accepted (Phase 9 SS4.1: the real revision keys by
        # (action, effect, variant)) and ignored: the no-op never
        # adjusts for any variant.
        rate = laplace_rate(causal, action, effect)
        return float(rate) if rate is not None else None

    def observe(self, action: str, effect: str, expected, actual,
                cycle: int = 0, world_model=None, causal=None,
                variant=None) -> dict:
        prior = laplace_rate(causal, action, effect)
        prior = float(prior) if prior is not None else 1.0
        return {"action": str(action), "effect": str(effect),
                "variant": variant,
                "expected": expected, "actual": actual,
                "observed": 1.0 if expected == actual else 0.0,
                "prior": round(prior, 4), "adjusted": round(prior, 4),
                "moved": 0.0, "cycle": int(cycle),
                "note": "no-op revision: observed outcome never adjusts "
                        "predicted success"}

    def to_json(self) -> dict:
        return {"alpha": self.alpha, "adjusted_success": {},
                "history": []}


def _no_revision_factory(actions, granted):
    """A full brain with the simulation-revision loop disabled (see
    ``_NoOpRevision``). ``Brain._get_simulator`` re-syncs
    ``simulator.revision`` from ``brain.simulation_revision`` on
    every ``imagine()``, so replacing both attributes covers the
    simulator path and the ``_check_predictions`` path."""
    brain = Brain(actions, granted=granted)
    noop = _NoOpRevision()
    brain.simulation_revision = noop
    sim = getattr(brain, "simulator", None)
    if sim is not None:
        sim.revision = noop
    return brain


def _no_causal_factory(actions, granted):
    """A full brain with the causal model disabled (see
    ``NoOpCausal`` in ``ab.reasoning.causal``). The no-op is
    injected at ``Brain`` construction, so the brain, the reasoning
    engine and the simulator all see the same no-op from the
    start -- never a half-replaced live model (design P10-5)."""
    return Brain(actions, granted=granted, causal_model=NoOpCausal())


ABLATION_REMOVED = {
    "no_revision": (
        "SimulationRevision adjustments only: per-(action, effect) "
        "predicted success never moves toward observed rates "
        "(observe() is a no-op, predicted_success() always falls "
        "back to the causal Laplace rate). Causal records, planner "
        "costs and bans, patterns, concepts and the world model are "
        "untouched."),
    "no_causal": (
        "The causal model only: NoOpCausal records nothing "
        "(record() is a no-op), reports no evidence (counts() / "
        "success_rate() / causes() always None) and claims nothing "
        "(claims() always []). Predicted rates fall back to the "
        "uninformative prior everywhere the model is read "
        "(planner confidence, engine abduction, simulator rates). "
        "Learner episodes, planner costs and bans, patterns, "
        "concepts and the world model are untouched."),
}
# Phase 9 recorded no_causal as NOT added: the causal model is
# threaded through the planner, the reasoning engine and the
# simulator at construction (Brain.causal / reasoning.causal /
# simulator.causal), so replacing it post-construction leaves stale
# references on the other two. Phase 10 (P10-5) resolves that by
# construction-time injection instead: Brain accepts the causal
# model as a constructor argument and hands the same object to the
# engine and the simulator, so the ablated brain is consistent
# from the start and no post-construction swap is ever needed.
# The comment above is kept so the Phase 9 reasoning stays
# auditable alongside the Phase 10 resolution.


# ------------------------------------------------------------------ suite runners (ctx: seed, fast, meta, shared)
def _mode(fast: bool) -> str:
    return "fast" if fast else "full"


def _run_discriminating_suite(ctx: dict) -> dict:
    """Suite (2): the three Phase 4 discriminating tasks."""
    started = time.monotonic()
    from .reasoning_tasks import run_discriminating_tasks

    try:
        section = run_discriminating_tasks()
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("discriminating", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    suite = _tasks_to_suite("discriminating", section.get("tasks") or {},
                            ctx["seed"], _mode(ctx["fast"]), ctx["meta"], started)
    ctx["discriminating_tasks"] = section.get("tasks") or {}
    return suite


def _run_reasoning2_suite(ctx: dict) -> dict:
    """Suite (3): the NEW reasoning tasks (Worker 1,
    ``ab.eval.reasoning_tasks2``). Defensive import: an absent
    module fails this suite with the reason, it never crashes the
    runner."""
    started = time.monotonic()
    try:
        import importlib

        module = importlib.import_module("ab.eval.reasoning_tasks2")
    except ImportError as exc:
        suite = _new_suite("reasoning_tasks2", ctx["seed"],
                           _mode(ctx["fast"]), ctx["meta"])
        _add_exp(suite, "reasoning_tasks2/module_present", False,
                 detail="ab.eval.reasoning_tasks2 could not be imported",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="Worker 1 module absent: "
                               "ab.eval.reasoning_tasks2 not landed yet",
                 meta=ctx["meta"])
        return _finalise_suite(suite, started)
    try:
        from .reasoning_tasks import make_variant_factories

        factories = make_variant_factories()
        for fname in ("run_new_reasoning_tasks", "run_reasoning_tasks2",
                      "run_tasks"):
            fn = getattr(module, fname, None)
            if fn is not None:
                result = _call_no_arg(fn, ctx["seed"])
                if isinstance(result, dict) and "tasks" in result:
                    return _tasks_to_suite(
                        "reasoning_tasks2", result["tasks"], ctx["seed"],
                        _mode(ctx["fast"]), ctx["meta"], started)
                return _adapt_unknown_result(
                    "reasoning_tasks2", result, ctx["seed"],
                    _mode(ctx["fast"]), ctx["meta"], started)
        if getattr(module, "TASKS", None):
            return _run_task_module(module, factories, ctx["seed"],
                                    _mode(ctx["fast"]), ctx["meta"],
                                    "reasoning_tasks2")
        raise AttributeError(
            "ab.eval.reasoning_tasks2 has no run_* entry point and no TASKS")
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("reasoning_tasks2", ctx["seed"],
                            _mode(ctx["fast"]), ctx["meta"], exc, started)


def _action_identity(action) -> tuple:
    """Full action identity (name + pre + eff + args), canonicalised
    so dict values compare by content. Two actions are the same
    action for planner purposes iff their identities match (cost /
    tool / store_as do not distinguish them -- see the double-add
    fix in :func:`_run_generic_env`)."""
    def _canon(value) -> str:
        return json.dumps(value, sort_keys=True, default=str)

    return (action.name, _canon(action.pre), _canon(action.eff),
            _canon(action.args))


def _run_generic_env(env, actions, goal, brain_factory, brain=None,
                     max_cycles: int = 80) -> dict:
    """One run of a generic (env, actions, goal) triple through a
    brain, mirroring the learning-eval run mechanics: a carried brain
    gets its actions added and ``brain.cycle + max_cycles`` cycles,
    a fresh brain gets ``max_cycles``.

    Phase 10 (P10-5x, double-add fix): a carried brain already holds
    the actions it was built with, and ``Planner.add_action`` appends
    blindly -- re-adding the same list doubled every A action in the
    planner (workshop: 15 actions became 30 after the A leg, 42 after
    the B leg). Actions are now added only when no action with the
    same full identity (name + pre + eff + args, see
    :func:`_action_identity`) is already present; the numbers added
    and skipped are recorded (``actions_added`` /
    ``actions_skipped_duplicate``).

    Failed attempts are counted with Worker 2's
    :func:`ab.eval.benchmarks.count_failures` (direct import -- the
    module has landed): the env-log count AND the learner-episode
    count (``ok=False``), with the episode count authoritative when
    episodes are available (the real :class:`Learner` records them;
    a no-op learner never does, so there the log count is the
    authority, exactly as in benchmarks). Both counts and the source
    are recorded in ``failures``; the env log alone undercounts real
    failures (the P10-2 audit: the surprise cold-coil line and the
    workshop press-no-mould line match no FAILURE_MARKER)."""
    # Direct import (benchmarks has landed): benchmarks imports
    # FAILURE_MARKERS from here, but only at its own module load; by
    # the time this function runs this module is fully loaded, so
    # there is no import cycle.
    from .benchmarks import count_failures

    if brain is None:
        brain = brain_factory(actions, {"compute", "environment"})
        limit = max_cycles
        actions_added = len(actions)
        actions_skipped = 0
    else:
        present = {_action_identity(a) for a in brain.planner.actions}
        actions_added = 0
        actions_skipped = 0
        for action in actions:
            ident = _action_identity(action)
            if ident in present:
                actions_skipped += 1
                continue
            brain.planner.add_action(action)
            present.add(ident)
            actions_added += 1
        brain.state.clear()
        limit = brain.cycle + max_cycles
    # Episodes are available iff the brain's learner is the real
    # Learner (a no-op learner never records episodes, so there the
    # log count is the authority -- see count_failures).
    episodes_available = isinstance(
        getattr(brain, "learner", None), Learner)
    episodes_before = len(list(
        getattr(getattr(brain, "learner", None), "episodes", []) or []))
    target = dict(goal) if isinstance(goal, dict) else {"goal": goal}
    brain.add_goal("generated transfer goal", target)
    cycle_before = brain.cycle
    stop = brain.run(env, limit)
    goal_obj = brain.goals.all()[-1]
    log = list(getattr(env, "log", []) or [])
    failures = count_failures(log, brain, episodes_before,
                              episodes_available=episodes_available)
    return {"completed": goal_obj.status == "done",
            "goal_status": goal_obj.status, "stop_reason": stop,
            "cycles": brain.cycle - cycle_before,
            "failed_attempts": failures["authoritative"],
            "failures": failures,
            "actions_added": actions_added,
            "actions_skipped_duplicate": actions_skipped,
            "env_log": log}


def _run_transfer_suite(ctx: dict) -> dict:
    """Suite (4): transfer (Phase 5) + transfer on newly generated
    environments (Worker 1, ``ab.eval.envgen``). The envgen leg is
    genuinely held-out for transfer: the generator seed is recorded
    and the brain never sees it."""
    started = time.monotonic()
    suite = _new_suite("transfer", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        from .reasoning_tasks import make_variant_factories
        from .transfer import (prepare_brain_for_new_world,
                               run_transfer_experiment)

        factories = make_variant_factories()
        section = run_transfer_experiment(brain_factory=factories["full"])
        ctx["transfer"] = section
        _add_exp(suite, "transfer_handbuilt", passed=bool(section.get("transferred")),
                 cycles=((section.get("world_b_continued") or {}).get("cycles")),
                 detail=(f"continued B failed "
                         f"{(section.get('world_b_continued') or {}).get('failed_attempts')} "
                         f"vs fresh B "
                         f"{(section.get('world_b_fresh') or {}).get('failed_attempts')}"),
                 evidence={"transferred": section.get("transferred"),
                           "world_a_fresh": section.get("world_a_fresh"),
                           "world_b_fresh": section.get("world_b_fresh"),
                           "world_b_continued": section.get("world_b_continued"),
                           "entry_protocol": section.get("entry_protocol")},
                 meta=ctx["meta"])
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        _add_exp(suite, "transfer_handbuilt", False,
                 detail=f"transfer experiment raised: {exc}",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="unknown -- needs investigation",
                 meta=ctx["meta"])
    # --- generated-environment transfer (defensive import) ---
    # One leg per family that generates (treasure / workshop /
    # surprise): train a carried brain on seed S, enter a fresh
    # instance of the same family at seed S+1, compare against a
    # fresh brain on its own fresh instance. Each family is its own
    # experiment, so a family with no transfer signal (e.g. the
    # fresh brain already solves it with zero failures) is visible
    # as such instead of hiding behind another family's result.
    try:
        import importlib

        envgen = importlib.import_module("ab.eval.envgen")
        generate = getattr(envgen, "generate", None)
        if generate is None:
            raise AttributeError("ab.eval.envgen has no generate()")
        from .reasoning_tasks import make_variant_factories

        factory = make_variant_factories()["full"]
        for family in ("treasure", "workshop", "surprise"):
            try:
                env_a, actions_a, goal_a, _shape_a = generate(
                    ctx["seed"], family)
                env_b, actions_b, goal_b, _shape_b = generate(
                    ctx["seed"] + 1, family)
                # Train on A with a carried brain, then enter B
                # (transient cleared, durable learning kept).
                carried_brain = factory(actions_a, {"compute", "environment"})
                rec_a = _run_generic_env(env_a, actions_a, goal_a, factory,
                                         brain=carried_brain)
                prepare_brain_for_new_world(carried_brain)
                rec_b_cont = _run_generic_env(env_b, actions_b, goal_b,
                                              factory, brain=carried_brain)
                # The fresh comparison needs its OWN env instance:
                # the env is stateful and single-use, and re-running
                # the continued brain's already-solved env would let
                # the fresh brain "complete" a goal the env already
                # satisfies (that bug read 2 cycles for a 5-action
                # task in an earlier run). The generator is
                # deterministic, so the same seed yields the same
                # layout with fresh state.
                env_b_fresh, actions_b_fresh, goal_b_fresh, _s = generate(
                    ctx["seed"] + 1, family)
                rec_b_fresh = _run_generic_env(env_b_fresh, actions_b_fresh,
                                               goal_b_fresh, factory)
            except Exception as exc:  # noqa: BLE001 - per-family, recorded
                _add_exp(suite, f"transfer_generated_envgen/{family}", False,
                         detail=f"generated-environment transfer for family "
                                f"{family} raised: {exc}",
                         evidence={"family": family,
                                   "error": f"{type(exc).__name__}: {exc}"},
                         failure_cause="unknown -- needs investigation",
                         meta=ctx["meta"])
                continue
            fa, fb = (rec_b_cont["failed_attempts"],
                      rec_b_fresh["failed_attempts"])
            if fa is not None and fb is not None:
                transferred = bool(rec_b_cont["completed"]
                                   and rec_b_fresh["completed"] and fa < fb)
                criterion = ("continued completes and needs strictly "
                             "fewer failed attempts than fresh")
                if fa == 0 and fb == 0:
                    transfer_note = (" (no transfer signal: fresh already "
                                     "solves it with 0 failures)")
                elif fa > fb:
                    transfer_note = (" (NEGATIVE transfer: continued did "
                                     f"worse than fresh, {fa} vs {fb} "
                                     "failed attempts)")
                else:
                    transfer_note = ""
            else:
                transferred = bool(rec_b_cont["completed"]
                                   and rec_b_fresh["completed"]
                                   and rec_b_cont["cycles"]
                                   < rec_b_fresh["cycles"])
                criterion = ("no env failure log: continued completes "
                             "and needs strictly fewer cycles than fresh")
                transfer_note = ""
            _add_exp(suite, f"transfer_generated_envgen/{family}",
                     passed=transferred, cycles=rec_b_cont["cycles"],
                     detail=(f"family={family} seeds={ctx['seed']}/"
                             f"{ctx['seed'] + 1}; continued B failed {fa} "
                             f"in {rec_b_cont['cycles']} cycles vs fresh B "
                             f"failed {fb} in {rec_b_fresh['cycles']} cycles; "
                             f"A completed={rec_a['completed']}"),
                     evidence={"family": family,
                               "seeds": [ctx["seed"], ctx["seed"] + 1],
                               "criterion": criterion,
                               "transferred": transferred,
                               "world_a": rec_a, "world_b_fresh": rec_b_fresh,
                               "world_b_continued": rec_b_cont,
                               "held_out_note": ("the envgen-generated "
                                                "transfer task counts as "
                                                "held-out for transfer: the "
                                                "brain never sees the "
                                                "generator seed")},
                     failure_cause=("transfer: continued brain did not "
                                    "beat the fresh brain on the generated "
                                    f"{family} environment{transfer_note}")
                                    if not transferred else None,
                     meta=ctx["meta"])
    except ImportError as exc:
        _add_exp(suite, "transfer_generated_envgen", False,
                 detail="ab.eval.envgen could not be imported",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="Worker 1 module absent: ab.eval.envgen "
                               "not landed yet",
                 meta=ctx["meta"])
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        _add_exp(suite, "transfer_generated_envgen", False,
                 detail=f"generated-environment transfer raised: {exc}",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="unknown -- needs investigation",
                 meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_world_tasks_suite(ctx: dict) -> dict:
    """Suite (5): the five Phase 5 world-model acceptance tasks."""
    started = time.monotonic()
    from .world_tasks import run_world_tasks

    try:
        section = run_world_tasks(transfer_record=ctx.get("transfer"))
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("world_tasks", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    suite = _tasks_to_suite("world_tasks", section.get("tasks") or {},
                            ctx["seed"], _mode(ctx["fast"]), ctx["meta"], started)
    ctx["world_tasks"] = section.get("tasks") or {}
    return suite


def _run_learning_suite(ctx: dict) -> dict:
    """Suite (6): the Phase 6 learning evaluation (full mode; ~8 s).
    ``record_path=None`` so this run never writes the canonical
    Phase 6 record."""
    started = time.monotonic()
    suite = _new_suite("learning", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        from .learning_eval import run_learning_evaluation

        record = run_learning_evaluation(record_path=None)
        ctx["learning_record"] = record
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("learning", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    full = (record.get("variants") or {}).get("full") or {}
    summary = full.get("summary") or {}
    _add_exp(suite, "variants/full_success",
             passed=bool(summary.get("n_completed") is not None
                         and summary.get("n_tasks")
                         and summary["n_completed"] == summary["n_tasks"]),
             detail=f"full variant success {summary.get('n_completed')}/"
                    f"{summary.get('n_tasks')} "
                    f"(transfer {summary.get('transfer_tasks')})",
             evidence={"full_summary": summary}, meta=ctx["meta"])
    for key, label in (("acceptance_experiment", "acceptance"),
                       ("two_family_transfer", "two_family_transfer"),
                       ("procedural_learning", "procedural"),
                       ("calibration", "calibration"),
                       ("success_based_learning", "success_withheld")):
        section = record.get(key) or {}
        _add_exp(suite, f"sub/{label}", passed=bool(section.get("passed")),
                 detail=str(section.get("criterion", ""))[:200],
                 evidence=section, meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_learning_curves_suite(ctx: dict) -> dict:
    """Suite (6b): learning curves (SS3.4) -- cycles / failed /
    success per experience level 0, 1, 2, 4, 8. A level's experiment
    passes when the point was measured (the curve is data: level 0
    is the no-experience control, so low success there is expected,
    not a failure). The curve itself is also stored top-level in the
    consolidated record under ``learning_curves``."""
    started = time.monotonic()
    suite = _new_suite("learning_curves", ctx["seed"], _mode(ctx["fast"]),
                       ctx["meta"])
    try:
        from .learning_eval import run_learning_curves

        curves = run_learning_curves(seed=ctx["seed"])
        ctx["learning_curves"] = curves
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("learning_curves", ctx["seed"],
                            _mode(ctx["fast"]), ctx["meta"], exc, started)
    for point in curves.get("points") or []:
        level = point.get("experience_level_requested")
        _add_exp(suite, f"level_{level}",
                 passed=bool(point.get("n") and point.get("runs")),
                 cycles=(point.get("cycles") or {}).get("mean"),
                 detail=(f"prior tasks {point.get('n_prior_tasks')}; "
                         f"success {point.get('n_completed')}/{point.get('n')} "
                         f"(rate {point.get('success_rate')}); cycles mean "
                         f"{(point.get('cycles') or {}).get('mean')} "
                         f"(std {(point.get('cycles') or {}).get('std')}); "
                         f"failed mean "
                         f"{(point.get('failed_attempts') or {}).get('mean')} "
                         f"(std {(point.get('failed_attempts') or {}).get('std')})"),
                 evidence={"point": point}, meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_retirement_world() -> dict:
    """Phase 9 SS4.2: the contradicting-evidence world, run here (not
    only in the unit test) so the consolidated record carries the
    firing with counts. Same shape as tests/test_phase9_retirement.py:
    a P-D pattern (press requires has_mould) is promoted in world X
    through the normal consolidation path (press fails 2x without
    has_mould in 2 distinct situations, succeeds 1x with it), then
    press SUCCEEDS 3x without has_mould in world Y -- each success a
    real recorded episode fed to record_pattern_contradiction. At the
    brain's own threshold (3) the pattern retires: it leaves the
    library, the planner multiplier drops 6.0 -> 1.0, and a
    pattern_retired event exists. Returns the experiment record."""
    from ..core.learning import Episode
    from ..planning.planner import Action

    actions = [
        Action("press", {"at": "bench"}, {"pressed": True}, cost=1.0),
        Action("fetch_mould", {"at": "bench"}, {"has_mould": True}, cost=1.0),
    ]
    brain = Brain(actions, granted={"compute", "environment"})
    brain.learner.record(Episode(1, "press", {"at": "bench"}, False))
    brain.learner.record(
        Episode(2, "press", {"at": "bench", "has_mould": False}, False))
    brain.learner.record(
        Episode(3, "press", {"at": "bench", "has_mould": True}, True))
    brain._consolidation_pass()
    promoted = [p for p in brain.patterns.patterns()
                if p.kind == "requires_precondition"]
    pid = promoted[0].id if promoted else None
    mult_before = brain.patterns.match(
        "press", {"at": "bench"}, {"pressed": True}, brain.world_model,
        [(a.name, a.pre, a.eff) for a in brain.planner.actions],
        {"at": "bench"})
    contradictions = 0
    for cycle in (4, 5, 6):
        brain.learner.record(
            Episode(cycle, "press", {"at": "bench"}, True))
        outcome = brain.record_pattern_contradiction(
            pid, f"press succeeded in world Y without has_mould "
                 f"(cycle {cycle})")
        contradictions = outcome.get("contradictions", contradictions)
    mult_after = brain.patterns.match(
        "press", {"at": "bench"}, {"pressed": True}, brain.world_model,
        [(a.name, a.pre, a.eff) for a in brain.planner.actions],
        {"at": "bench"})
    retired_events = sum(1 for e in brain.events
                         if e.get("kind") == "pattern_retired")
    retired = [r for r in brain.retired_patterns()]
    passed = bool(promoted and retired_events > 0 and retired
                  and mult_before == 6.0 and mult_after == 1.0)
    return {
        "passed": passed,
        "cycles": None,
        "detail": (f"P-D pattern promoted in world X, contradicted "
                   f"{contradictions}x in world Y, retired: "
                   f"{retired_events} pattern_retired event(s), "
                   f"multiplier {mult_before} -> {mult_after}"),
        "evidence": {
            "promoted_pattern": (promoted[0].name if promoted else None),
            "contradictions_fed": contradictions,
            "retired_events": retired_events,
            "retired_records": len(retired),
            "multiplier_before": mult_before,
            "multiplier_after": mult_after,
            "pattern_still_in_library": bool(
                pid and any(p.id == pid
                            for p in brain.patterns.patterns())),
        },
        "failure_cause": (None if passed else
                          "retirement did not fire in the "
                          "contradicting-evidence world"),
    }


def _run_execution_suite(ctx: dict) -> dict:
    """Suite (7): the Phase 7 execution evaluation. Fast mode runs
    the quick subset only (the record says so); full mode runs all
    twelve experiments. ``record_path=None`` so this run never
    writes the canonical Phase 7 record."""
    started = time.monotonic()
    suite = _new_suite("execution", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        from .execution_eval import run_execution_evaluation

        record = run_execution_evaluation(fast=ctx["fast"], record_path=None)
        ctx["execution_record"] = record
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("execution", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    for name, section in (record.get("experiments") or {}).items():
        if not isinstance(section, dict):
            continue
        if section.get("skipped"):
            _add_exp(suite, name, None, skipped=True,
                     skip_reason=str(section.get("reason", "")),
                     evidence=section, meta=ctx["meta"])
        elif "passed" in section:
            _add_exp(suite, name, passed=bool(section.get("passed")),
                     cycles=section.get("cycles"),
                     detail=str(section.get("detail", "")),
                     evidence=section, meta=ctx["meta"])
    carried = record.get("carried_items") or {}
    retire = carried.get("wrong_promoted_pattern_retirement") or {}
    status = str(retire.get("status", ""))
    _add_exp(suite, "carried/wrong_promoted_pattern_retirement",
             passed=status.upper() in ("FIRED", "PASS", "PASSED"),
             detail=f"carried-item status: {status or 'absent from record'}",
             evidence={"carried_item": retire},
             failure_cause=("pattern retirement: the workshop worlds "
                            "generate only 1 contradicting observation "
                            "and the brain's threshold is 3 -- see the "
                            "dedicated retirement world below, where it "
                            "FIRES (Phase 9 SS4.2)")
             if status.upper() not in ("FIRED", "PASS", "PASSED") else None,
             meta=ctx["meta"])
    # Dedicated contradicting-evidence world (Phase 9 SS4.2): the
    # retirement mechanism firing with counts, independent of the
    # workshop worlds' thin evidence.
    try:
        world = _run_retirement_world()
        _add_exp(suite, "retirement/contradicting_evidence_world",
                 passed=bool(world["passed"]),
                 detail=str(world["detail"]),
                 evidence=world["evidence"],
                 failure_cause=world.get("failure_cause"),
                 meta=ctx["meta"])
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        _add_exp(suite, "retirement/contradicting_evidence_world",
                 passed=False,
                 detail=f"retirement world errored: {type(exc).__name__}: {exc}",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="retirement world errored",
                 meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_phase8_suite(ctx: dict) -> dict:
    """Suite (8): the Phase 8 perception/imagination evaluation,
    including the P-D probe re-run after the SS4.1 fix -- the probe
    must PASS now, and a FAIL here is recorded honestly (it fails
    this suite and the consolidated run). Fast mode skips the probe
    (recorded as skipped, not passed)."""
    started = time.monotonic()
    suite = _new_suite("phase8", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        from .phase8_eval import run_phase8_evaluation

        record = run_phase8_evaluation(fast=ctx["fast"], record_path=None)
        ctx["phase8_record"] = record
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("phase8", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    for name, section in (record.get("experiments") or {}).items():
        if not isinstance(section, dict):
            continue
        if section.get("skipped"):
            _add_exp(suite, name, None, skipped=True,
                     skip_reason=str(section.get("reason", "")),
                     evidence=section, meta=ctx["meta"])
        elif "passed" in section:
            _add_exp(suite, name, passed=bool(section.get("passed")),
                     cycles=section.get("cycles"),
                     detail=str(section.get("detail", "")),
                     evidence=section, meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_long_horizon_suite(ctx: dict) -> dict:
    """Suite (9): the long-horizon task (Worker 4,
    ``ab.eval.long_horizon``). Defensive import: an absent module
    fails this suite with the reason, it never crashes the runner."""
    started = time.monotonic()
    try:
        import importlib

        module = importlib.import_module("ab.eval.long_horizon")
    except ImportError as exc:
        suite = _new_suite("long_horizon", ctx["seed"], _mode(ctx["fast"]),
                           ctx["meta"])
        _add_exp(suite, "long_horizon/module_present", False,
                 detail="ab.eval.long_horizon could not be imported",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="Worker 4 module absent: "
                               "ab.eval.long_horizon not landed yet",
                 meta=ctx["meta"])
        return _finalise_suite(suite, started)
    try:
        for fname in ("run_long_horizon_task", "run_long_horizon",
                      "run_long_horizon_evaluation", "run_task"):
            fn = getattr(module, fname, None)
            if fn is not None:
                return _adapt_unknown_result(
                    "long_horizon", _call_no_arg(fn, ctx["seed"]),
                    ctx["seed"], _mode(ctx["fast"]), ctx["meta"], started)
        raise AttributeError(
            "ab.eval.long_horizon has no run_* entry point")
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("long_horizon", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)


def _read_sidecar(path: str):
    """Read a Worker 4 JSON sidecar; returns (record, error). A
    missing sidecar is a normal outcome (the leg is skipped with the
    reason), never a crash."""
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        return (rec if isinstance(rec, dict) else {"value": rec}), None
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_perception_suite(ctx: dict) -> dict:
    """Suite (10): perception eval -- the Phase 8 document/image legs
    reused where clean (their numbers come from the Phase 8 suite
    run above, cited as the source, not re-run), plus the noisy-OCR
    result recorded by Worker 4 (sidecar
    docs/phase9-noisy-ocr.json when present; skipped with the reason
    when absent -- degradation is a result, not a failure to hide)."""
    started = time.monotonic()
    suite = _new_suite("perception", ctx["seed"], _mode(ctx["fast"]),
                       ctx["meta"])
    p8 = ctx.get("phase8_record") or {}
    p8_exps = p8.get("experiments") or {}
    for name in ("document_perception", "image_percept"):
        section = p8_exps.get(name) or {}
        if section.get("skipped"):
            _add_exp(suite, name, None, skipped=True,
                     skip_reason=str(section.get("reason", "")),
                     evidence={"source": "phase8 suite (this run)",
                               "section": section}, meta=ctx["meta"])
        elif "passed" in section:
            _add_exp(suite, name, passed=bool(section.get("passed")),
                     detail=str(section.get("detail", "")),
                     evidence={"source": "phase8 suite (this run); "
                                         "number reused, not re-run",
                               "section": section}, meta=ctx["meta"])
        else:
            _add_exp(suite, name, None, skipped=True,
                     skip_reason="Phase 8 suite did not produce this leg",
                     evidence={"source": "phase8 suite (this run)"},
                     meta=ctx["meta"])
    sidecar, error = _read_sidecar(NOISY_OCR_SIDECAR)
    if sidecar is None:
        _add_exp(suite, "noisy_ocr", None, skipped=True,
                 skip_reason="Worker 4 sidecar docs/phase9-noisy-ocr.json "
                             f"absent ({error})",
                 evidence={"sidecar": NOISY_OCR_SIDECAR}, meta=ctx["meta"])
    else:
        passed = sidecar.get("passed")
        if passed is None:
            # No explicit verdict in the sidecar: pass iff an exact
            # read rate was measured and reported (degradation is a
            # result, not a failure to hide). The sidecar reports
            # exact_reads / n_samples (and mean_confidence).
            n = sidecar.get("n_samples")
            exact = sidecar.get("exact_reads")
            passed = (isinstance(n, int) and n > 0
                      and isinstance(exact, int))
            rate = (round(exact / n, 4) if passed else None)
        else:
            n = sidecar.get("n_samples")
            exact = sidecar.get("exact_reads")
            rate = (round(exact / n, 4) if isinstance(n, int) and n
                    and isinstance(exact, int) else None)
        detail = str(sidecar.get("detail", sidecar.get("note", "")))
        if rate is not None:
            detail = (f"exact reads {exact}/{n} (rate {rate}); mean "
                      f"confidence {sidecar.get('mean_confidence')}"
                      + (f"; {detail}" if detail else ""))
        _add_exp(suite, "noisy_ocr", passed=bool(passed),
                 detail=detail,
                 evidence={"source": "docs/phase9-noisy-ocr.json (Worker 4)",
                           "exact_read_rate": rate,
                           "sidecar": sidecar}, meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_memory_suite(ctx: dict) -> dict:
    """Suite (11): memory eval -- retention after restart, retrieval,
    consolidation -- reusing the Phase 6 mechanisms (the numbers come
    from the learning suite's full run above, cited as the source,
    not re-run: the retention / retrieval / cap experiments are part
    of that run)."""
    started = time.monotonic()
    suite = _new_suite("memory", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    learn = ctx.get("learning_record") or {}
    retention = learn.get("retention_after_restart") or {}
    _add_exp(suite, "retention_after_restart",
             passed=bool(retention.get("retained_transfer")
                         and retention.get("episodes_preserved")
                         and retention.get("pattern_ids_preserved")),
             detail=("loaded brain still transfers to D better than "
                     "fresh D; patterns, episodes, concepts and "
                     "calibration survive save/load"
                     if retention else "learning record has no retention section"),
             evidence={"source": "learning suite (this run); number reused",
                       "retention": retention}, meta=ctx["meta"])
    retrieval = learn.get("retrieval_precision") or {}
    _add_exp(suite, "retrieval",
             passed=bool(retrieval.get("improved")),
             detail=("retrieval precision improves with more training "
                     "(late > early)" if retrieval.get("improved") else
                     "retrieval precision did not improve with training"),
             evidence={"source": "learning suite (this run); number reused",
                       "retrieval": retrieval}, meta=ctx["meta"])
    cap = learn.get("cap_measurement") or {}
    settings = cap.get("settings") or {}
    tail = settings.get("cap_100_tail") or {}
    prio = settings.get("cap_100_prioritized") or {}
    tail_frac = tail.get("pattern_evidence_retained_fraction")
    prio_frac = prio.get("pattern_evidence_retained_fraction")
    measured = tail_frac is not None and prio_frac is not None
    _add_exp(suite, "consolidation_bounded_memory",
             passed=bool(measured and prio_frac > tail_frac),
             detail=(f"prioritized pruning retains "
                     f"{prio_frac} of pattern-evidence episodes at cap 100 "
                     f"vs tail pruning {tail_frac}; transfer after pruning "
                     f"{prio.get('transfer_vs_fresh_d1')}"
                     if measured else
                     "cap measurement settings unreadable in learning record"),
             evidence={"source": "learning suite (this run); SS7.5 cap "
                                "measurement reused as the bounded-memory "
                                "consolidation mechanism",
                       "cap_100_tail": tail, "cap_100_prioritized": prio},
             failure_cause=("bounded memory: prioritized pruning did not "
                            "retain more pattern-evidence episodes than "
                            "tail pruning at cap 100")
             if measured else None,
             meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_tool_use_suite(ctx: dict) -> dict:
    """Suite (12): tool-use eval -- execution-eval legs reused (their
    numbers come from the execution suite run above, cited as the
    source) plus the RLIMIT_CPU burner result (Worker 4 sidecar
    docs/phase9-cpu-burner.json when present; the leg is skipped
    with the reason when absent -- the test module cannot be
    imported from here, so the sidecar is the honest channel)."""
    started = time.monotonic()
    suite = _new_suite("tool_use", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    exe = ctx.get("execution_record") or {}
    exe_exps = exe.get("experiments") or {}
    for name in ("budget_exceeding_tool", "permission_denial", "provenance"):
        section = exe_exps.get(name) or {}
        if section.get("skipped"):
            _add_exp(suite, name, None, skipped=True,
                     skip_reason=str(section.get("reason", "")),
                     evidence={"source": "execution suite (this run)",
                               "section": section}, meta=ctx["meta"])
        elif "passed" in section:
            _add_exp(suite, name, passed=bool(section.get("passed")),
                     detail=str(section.get("detail", "")),
                     evidence={"source": "execution suite (this run); "
                                         "number reused, not re-run",
                               "section": section}, meta=ctx["meta"])
        else:
            _add_exp(suite, name, None, skipped=True,
                     skip_reason="execution suite did not produce this leg",
                     evidence={"source": "execution suite (this run)"},
                     meta=ctx["meta"])
    sidecar, error = _read_sidecar(CPU_BURNER_SIDECAR)
    if sidecar is None:
        _add_exp(suite, "rlimit_cpu_burner", None, skipped=True,
                 skip_reason="Worker 4 sidecar docs/phase9-cpu-burner.json "
                             f"absent ({error})",
                 evidence={"sidecar": CPU_BURNER_SIDECAR}, meta=ctx["meta"])
    else:
        passed = sidecar.get("passed")
        if passed is None:
            # No explicit verdict in the sidecar: pass iff the burner
            # was terminated by the CPU limit as the sidecar
            # attributes it -- observed dead, exitcode -9 (SIGKILL),
            # no wall-time timeout, no parent kill. Note ``killed``
            # in the sidecar is the PARENT-kill flag (False when the
            # kernel enforced the limit itself), not the outcome.
            passed = bool(sidecar.get("observed_dead")
                          and sidecar.get("exitcode") == -9
                          and not sidecar.get("timed_out"))
        _add_exp(suite, "rlimit_cpu_burner", passed=bool(passed),
                 detail=str(sidecar.get("cause_as_reported",
                                        sidecar.get("detail",
                                                    sidecar.get("note", "")))),
                 evidence={"source": "docs/phase9-cpu-burner.json (Worker 4)",
                           "sidecar": sidecar}, meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_planning_suite(ctx: dict) -> dict:
    """Suite (13): planning eval -- a named suite reusing the
    discriminating + world-task numbers (plan cost / optimality on
    those tasks, as cycles to completion for the full variant). No
    new planner is built; every experiment says so in its evidence."""
    started = time.monotonic()
    suite = _new_suite("planning", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    for source, tasks in (("discriminating", ctx.get("discriminating_tasks") or {}),
                          ("world_tasks", ctx.get("world_tasks") or {})):
        for task_id, task in tasks.items():
            full = (task.get("variants") or {}).get("full") or {}
            _add_exp(suite, f"{source}/{task_id}",
                     passed=bool(full.get("passed")),
                     cycles=full.get("cycles"),
                     detail=(f"full variant: {full.get('detail', '')}"),
                     evidence={"source": f"{source} suite (this run); "
                                         "number reused, no new planner",
                               "criterion": task.get("criterion"),
                               "variants": task.get("variants")},
                     meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _imagination_calibration(ctx: dict) -> dict:
    """The imagination stream for SS3.6: a fresh brain imagines the
    treasure route, then executes it, and the calibration tracker
    reads the execution prediction stream (the same
    ``compute_calibration`` the learning/workshop streams use).
    Labelled as the imagine-then-execute stream, not as a pure
    simulation stream -- simulation alone makes no checkable
    predictions."""
    from ..body.environment import RoomWorld, room_world_actions
    from .learning_eval import compute_calibration

    env = RoomWorld()
    brain = Brain(room_world_actions(), granted={"compute", "environment"})
    brain.add_goal("get the treasure", {"has_treasure": True})
    # Imagination simulates from the brain's current world, so put
    # the believed state at the env's start state first (the same
    # whole-world setter the Phase 8 experiments use); without it
    # imagine() plans from an empty world and finds no route.
    brain.world = dict(env.observe())
    imagined = brain.imagine({"has_treasure": True})
    brain.run(env, 60)
    cal = compute_calibration(brain)
    cal["stream"] = ("imagination: imagine-then-execute on the treasure "
                     "task (Phase 8 exp-5/6 shape)")
    cal["scenarios_imagined"] = len((imagined or {}).get("scenarios") or [])
    return cal


def _run_calibration_suite(ctx: dict) -> dict:
    """Suite (14): calibration consolidation (SS3.6). One experiment
    per stream (learning / workshop / imagination), each with n, ECE
    default-1.0 vs calibrated, Brier default-1.0 vs calibrated. The
    stream-dependence finding is preserved: per-stream numbers are
    reported and the summary says where calibration helps and where
    it does not; streams are NEVER averaged into one smoothed
    number. A stream experiment passes when the stream was measured
    (n > 0) and both ECE/Brier pairs are reported -- the
    consolidation's job is reporting, and whether calibration helps
    on that stream is data in the evidence (``ece_improved`` /
    ``brier_improved``), not a hidden pass criterion."""
    started = time.monotonic()
    suite = _new_suite("calibration", ctx["seed"], _mode(ctx["fast"]),
                       ctx["meta"])
    streams: dict = {}
    learn = ctx.get("learning_record") or {}
    streams["learning"] = ((learn.get("calibration") or {}).get("overall") or {})
    exe = ctx.get("execution_record") or {}
    streams["workshop"] = ((exe.get("calibration") or {}).get("overall") or {})
    try:
        streams["imagination"] = _imagination_calibration(ctx)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        streams["imagination"] = {"error": f"{type(exc).__name__}: {exc}"}
    ctx["calibration_streams"] = streams
    for name, cal in streams.items():
        n = cal.get("n_predictions")
        measured = isinstance(n, int) and n > 0
        _add_exp(suite, f"stream/{name}", passed=bool(measured),
                 detail=(f"n={n}; ECE default-1.0 "
                         f"{cal.get('ece_default_1.0')} vs calibrated "
                         f"{cal.get('ece_calibrated')} "
                         f"(improved={cal.get('ece_improved')}); Brier "
                         f"{cal.get('brier_default_1.0')} vs "
                         f"{cal.get('brier_calibrated')} "
                         f"(improved={cal.get('brier_improved')})"
                         if measured else
                         f"stream not measured: {cal.get('error', 'no predictions')}"),
                 evidence={"stream": name, "calibration": cal},
                 failure_cause=(f"calibration stream {name}: no "
                                "predictions measured, so no ECE/Brier "
                                "could be reported")
                 if not measured else None,
                 meta=ctx["meta"])
    return _finalise_suite(suite, started)


def _run_ablations_suite(ctx: dict) -> dict:
    """Ablations beyond full / no_learning / non_integrated: the
    ``no_revision`` variant (SimulationRevision disabled -- see
    ``_NoOpRevision``) and the ``no_causal`` variant (causal model
    disabled -- see ``NoOpCausal`` and ``_no_causal_factory``),
    each run on the discriminating tasks and the new reasoning
    tasks. Each task experiment passes when the ablation run
    executed cleanly and its outcome was recorded (the ablation
    suite measures the mechanism's contribution; whether the variant
    passes the task is data in the evidence, with a ``differed``
    flag against the full variant). The mechanisms each ablation
    removes are recorded in the suite and top-level in the record.
    Phase 10: no_causal separates on causal_reasoning -- the full
    brain claims the reliable forge variant, the ablated brain
    claims nothing (see the ``no_causal/separation`` experiment)."""
    started = time.monotonic()
    suite = _new_suite("ablations", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        from .reasoning_tasks import TASKS as TASKS1, make_variant_factories

        factories = make_variant_factories()
        factories["no_revision"] = _no_revision_factory
        factories["no_causal"] = _no_causal_factory
        task_lists = [("discriminating", TASKS1)]
        try:
            import importlib

            module2 = importlib.import_module("ab.eval.reasoning_tasks2")
            if getattr(module2, "TASKS", None):
                task_lists.append(("reasoning_tasks2", module2.TASKS))
        except ImportError:
            pass
        outcomes: dict = {}
        for ablation in ("no_revision", "no_causal"):
            for source, tasks in task_lists:
                for task_id, runner, criterion in tasks:
                    per_variant: dict = {}
                    for vname in ("full", ablation):
                        try:
                            per_variant[vname] = runner(factories[vname])
                        except Exception as exc:  # noqa: BLE001 - data, not a crash
                            per_variant[vname] = {
                                "passed": False, "cycles": None,
                                "evidence": {"error": f"{type(exc).__name__}: {exc}"},
                                "detail": f"task errored: {type(exc).__name__}: {exc}"}
                    full = per_variant["full"]
                    ablated = per_variant[ablation]
                    differed = bool(full.get("passed")) != bool(ablated.get("passed"))
                    outcome = outcomes.setdefault(
                        f"{source}/{task_id}",
                        {"full_passed": bool(full.get("passed")),
                         "full_cycles": full.get("cycles"),
                         "full_detail": full.get("detail")})
                    outcome[f"{ablation}_passed"] = bool(ablated.get("passed"))
                    outcome[f"{ablation}_differed"] = differed
                    outcome[f"{ablation}_cycles"] = ablated.get("cycles")
                    outcome[f"{ablation}_detail"] = ablated.get("detail")
                    ran_cleanly = ("error" not in (ablated.get("evidence") or {})
                                   and "error" not in (full.get("evidence") or {}))
                    _add_exp(suite, f"{ablation}/{source}/{task_id}",
                             passed=bool(ran_cleanly),
                             cycles=ablated.get("cycles"),
                             detail=(f"{ablation} task "
                                     f"{'PASSED' if ablated.get('passed') else 'FAILED'} "
                                     f"(full {'PASSED' if full.get('passed') else 'FAILED'}; "
                                     f"{'outcome DIFFERED from full' if differed else 'same outcome as full'})"),
                             evidence={"task": f"{source}/{task_id}",
                                       "criterion": criterion,
                                       "full": full, ablation: ablated,
                                       "differed": differed},
                             failure_cause=("ablation run errored; see evidence")
                             if not ran_cleanly else None,
                             meta=ctx["meta"])
            n_differed = sum(1 for o in outcomes.values()
                             if o.get(f"{ablation}_differed"))
            _add_exp(suite, f"{ablation}/separation", passed=True,
                     detail=(f"{ablation} differed from full on "
                             f"{n_differed}/{len(outcomes)} task(s); "
                             f"the removed mechanism is "
                             f"{'load-bearing on at least one task' if n_differed else 'not load-bearing on these tasks'}"),
                     evidence={"outcomes": outcomes,
                               "mechanisms_removed": ABLATION_REMOVED},
                     meta=ctx["meta"])
        ctx["ablation_outcomes"] = outcomes
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("ablations", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    return _finalise_suite(suite, started)


# ------------------------------------------------------------------ benchmarks (Phase 10, suite 19)
def _run_benchmarks_suite(ctx: dict) -> dict:
    """Suite (19): the Phase 10 benchmark tiers (Worker 2,
    ``ab.eval.benchmarks``). Direct import -- the module has landed,
    so the wiring imports ``run_benchmarks`` by name; an import
    failure still fails this suite with the reason, it never crashes
    the runner. ``run_benchmarks(seed=0)``
    returns ``{"tasks": {name: {"passed", "cycles", "detail",
    "evidence", "failure_cause", "variants"}}}``; each task's
    ``variants`` (full / no_learning / non_integrated / fresh) are
    converted to the common per-task/per-variant format, so the
    suite records one experiment per task for the FULL variant plus
    the variant-separation experiment, exactly like the
    discriminating suite. A task whose result carries no variants
    is recorded as its own full-variant result instead (the
    separation experiment is then skipped for that task set with
    the reason recorded). The consolidated runner's exit-code
    contract (non-zero on any failing suite, design P9-4) applies
    here as everywhere: a benchmark task the full brain fails is a
    failing experiment, recorded honestly."""
    started = time.monotonic()
    try:
        from .benchmarks import run_benchmarks
    except ImportError as exc:
        suite = _new_suite("benchmarks", ctx["seed"], _mode(ctx["fast"]),
                           ctx["meta"])
        _add_exp(suite, "benchmarks/module_present", False,
                 detail="ab.eval.benchmarks could not be imported",
                 evidence={"error": f"{type(exc).__name__}: {exc}"},
                 failure_cause="benchmarks: ab.eval.benchmarks import "
                               "failed",
                 meta=ctx["meta"])
        return _finalise_suite(suite, started)
    try:
        result = _call_no_arg(run_benchmarks, ctx["seed"])
        if not isinstance(result, dict) or not isinstance(
                result.get("tasks"), dict):
            raise ValueError(
                "run_benchmarks returned an unrecognised result shape: "
                "expected {\"tasks\": {name: {...}}}")
        tasks = result["tasks"]
        ctx["benchmark_tasks"] = tasks
        ctx["benchmarks"] = result
        converted: dict = {}
        for task_id, task in tasks.items():
            if not isinstance(task, dict):
                continue
            variants = task.get("variants")
            if isinstance(variants, dict) and variants.get("full") is not None:
                converted[task_id] = {
                    "criterion": (task.get("criterion")
                                  or task.get("detail", "")),
                    "variants": variants,
                    "difficulty": task.get("difficulty"),
                    "tier": task.get("tier")}
            else:
                # No per-variant breakdown: the task result itself
                # is the full-variant result.
                converted[task_id] = {
                    "criterion": (task.get("criterion")
                                  or task.get("detail", "")),
                    "variants": {"full": task},
                    "difficulty": task.get("difficulty"),
                    "tier": task.get("tier"),
                    "note": "no per-variant breakdown in this task "
                            "result; recorded as the full-variant result"}
        if not converted:
            suite = _new_suite("benchmarks", ctx["seed"], _mode(ctx["fast"]),
                               ctx["meta"])
            _add_exp(suite, "benchmarks/tasks_present", False,
                     detail="run_benchmarks returned no tasks",
                     evidence={"result_keys": list(result.keys())},
                     failure_cause="benchmarks: run_benchmarks returned "
                                   "an empty tasks mapping",
                     meta=ctx["meta"])
            return _finalise_suite(suite, started)
        suite = _tasks_to_suite("benchmarks", converted, ctx["seed"],
                                _mode(ctx["fast"]), ctx["meta"], started)
        # _tasks_to_suite is shared with the discriminating suite;
        # its separation failure text names that suite -- reword it
        # here so a benchmarks separation failure says so.
        sep = (suite.get("experiments") or {}).get("variant_separation")
        if isinstance(sep, dict) and sep.get("passed") is False:
            sep["failure_cause"] = (
                "benchmarks: no task separates the variants -- the "
                "benchmark tasks do not discriminate (see Phase 10 "
                "design P10-3)")
        return suite
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("benchmarks", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)


# ------------------------------------------------------------------ failure analysis (SS3.7)
def build_failure_analysis(suites: dict) -> dict:
    """Every failure in every suite, with failure_cause as far as
    known, and the top failure modes by count (grouped by the
    mechanism prefix of the cause -- the text before the first
    ':'). A suite's own ``suite_error`` experiment counts as a
    failure of the harness-error mode."""
    failures: list[dict] = []
    for suite_name, suite in suites.items():
        if not isinstance(suite, dict):
            continue
        for exp_name, exp in (suite.get("experiments") or {}).items():
            if not isinstance(exp, dict) or exp.get("skipped"):
                continue
            if exp.get("passed") is False:
                cause = exp.get("failure_cause") or "unknown -- needs investigation"
                failures.append({"suite": suite_name, "experiment": exp_name,
                                 "detail": exp.get("detail"),
                                 "failure_cause": cause})
    modes: dict[str, int] = {}
    for failure in failures:
        cause = failure["failure_cause"] or "unknown -- needs investigation"
        mode = cause.split(":", 1)[0].strip() or "unknown -- needs investigation"
        modes[mode] = modes.get(mode, 0) + 1
    top = sorted(modes.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"failures": failures, "n_failures": len(failures),
            "top_failure_modes": [{"mode": mode, "count": count}
                                  for mode, count in top],
            "note": ("failure_cause is mechanism + one-line cause "
                     "where known, else 'unknown -- needs investigation'")}


def _run_failure_analysis_suite(ctx: dict) -> dict:
    """Suite (15): failure analysis. Passes when the analysis was
    computed over the suites run so far (it is analysis, not a
    criterion); the failures themselves are data, listed in full in
    the evidence. The runner recomputes it once more after the
    regression and leakage suites, so the final record covers them
    too."""
    started = time.monotonic()
    suite = _new_suite("failure_analysis", ctx["seed"], _mode(ctx["fast"]),
                       ctx["meta"])
    analysis = build_failure_analysis(ctx.get("suites") or {})
    ctx["failure_analysis"] = analysis
    _add_exp(suite, "failure_analysis_complete", passed=True,
             detail=(f"{analysis['n_failures']} failure(s) across the "
                     f"programme; top mode: "
                     f"{analysis['top_failure_modes'][0]['mode']} "
                     f"({analysis['top_failure_modes'][0]['count']})"
                     if analysis["top_failure_modes"] else
                     "no failures in the programme"),
             evidence={"analysis": analysis}, meta=ctx["meta"])
    return _finalise_suite(suite, started)


# ------------------------------------------------------------------ regression (SS3.8)
def _rows_summary(regression_record: dict) -> list[dict]:
    """Per-row summary for a regression record (used both by the
    regression suite's evidence and by the final rebuild)."""
    return [{"phase": r.get("phase"),
             "record_present": r.get("record_present"),
             "tests_ran": (r.get("tests") or {}).get("ran"),
             "tests_mode": (r.get("tests") or {}).get("mode"),
             "eval_passes": (r.get("eval_passes_summary")
                             if isinstance(r.get("eval_passes_summary"), dict)
                             else None)}
            for r in regression_record.get("rows") or []]


def _run_regression_suite(ctx: dict) -> dict:
    """Suite (16): regression tracking. Builds the regression record
    (per-phase key numbers from the on-disk records plus this run's
    numbers); the file itself is written by
    ``run_consolidated_evaluation`` when a record path is given, and
    only then -- a harness run with ``record_path=None`` never
    overwrites the canonical regression record. NOTE: this suite
    runs before the leakage audit, so its in-suite build misses the
    leakage experiments; ``run_consolidated_evaluation`` rebuilds
    the record from the final suites after the loop (same builder,
    complete numbers) and that rebuilt record is what the top-level
    ``regression`` key and the file carry."""
    started = time.monotonic()
    suite = _new_suite("regression", ctx["seed"], _mode(ctx["fast"]), ctx["meta"])
    try:
        current_numbers = _current_numbers_for_regression(ctx)
        record = build_regression_record(current=current_numbers,
                                         current_phase=ctx.get("phase"))
        ctx["regression_record"] = record
        ctx["current_numbers"] = current_numbers
        p8_row = next((r for r in record.get("rows") or []
                       if r.get("phase") == "8"), None)
        p8_ok = bool(p8_row and p8_row.get("record_present"))
        labels = ", ".join(r.get("phase") or "?" for r in record.get("rows") or [])
        _add_exp(suite, "regression_record", passed=p8_ok,
                 detail=(f"{len(record.get('rows') or [])} phase rows "
                         f"({labels}); Phase 8 row "
                         f"{'present' if p8_ok else 'MISSING'}"),
                 evidence={"rows_summary": _rows_summary(record)},
                 failure_cause=("regression tracking: Phase 8 row missing "
                                "-- docs/phase8-harness-record.json absent")
                 if not p8_ok else None,
                 meta=ctx["meta"])
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("regression", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    return _finalise_suite(suite, started)


def _current_numbers_for_regression(ctx: dict) -> dict:
    """This run's numbers in the regression-row shape: tests (cited),
    demo (not run by this runner -- None, honestly), transfer,
    behaviours (from the transfer record's mechanism section when
    present), and eval passes per suite (every experiment's pass
    value, keyed suite/experiment). Labelled with the running
    phase (``ctx["phase"]``) by the caller, never hard-coded."""
    passes: dict = {}
    for suite_name, suite in (ctx.get("suites") or {}).items():
        if not isinstance(suite, dict):
            continue
        for exp_name, exp in (suite.get("experiments") or {}).items():
            if isinstance(exp, dict) and "passed" in exp and not exp.get("skipped"):
                passes[f"{suite_name}/{exp_name}"] = bool(exp.get("passed"))
    transfer = ctx.get("transfer") or {}
    return {"tests": ctx["meta"]["test_counts"],
            "demo": None,
            "transfer": ({"transferred": transfer.get("transferred")}
                         if transfer else None),
            "behaviours": None,
            "eval_passes": passes}


# ------------------------------------------------------------------ leakage audit (SS3.9)
def _new_task_names_from_module(filename: str) -> list[str]:
    """Invented entity names from one eval module, if present:
    string constants assigned to UPPER_CASE module names, filtered
    to plausible held-out entity names. The filter matters: the
    module also holds criterion sentences, task ids, relation names
    (``is_a``, ``opens``, ``made_of``), action names (``take_garnik``)
    and state keys (``door_open``), and those are the brain's own
    generic vocabulary -- flagging them would be a false positive.
    So a candidate must be a single lowercase word (>= 4 letters,
    which drops state keys, action names, relations with ``_`` and
    criterion sentences), must not be a word from a TASKS id
    (``analogy`` / ``abstraction`` name the reasoning mechanism
    itself, which the brain legitimately owns), and must not be in
    the small stoplist of generic relation/verb words below. What
    survives is the invented entities (``vortan``, ``mellor``,
    ``norbex``, ``garnik``, ``trellin``, ``vossar``, ``krelm``,
    ``forvik``) -- the names whose appearance in ab/ outside
    ab/eval/ would be a real leak."""
    # Generic relation/verb words that appear as string constants in
    # the module but are the brain's own vocabulary, not held-out
    # entity names. Phase 10 adds the benchmarks module's own
    # evaluation vocabulary: "full" / "fresh" are the variant names
    # the harness itself uses (not world entities); "partial" is an
    # observability value; "observability" is a difficulty-dimension
    # name (and a world-model concept in ab/world/model.py);
    # "focused" is attention vocabulary (ab/core/attention.py).
    # Without these, the audit reports false-positive leaks for
    # words the brain legitimately owns.
    GENERIC_WORDS = frozenset({"opens", "made_of", "forged",
                               "full", "fresh", "partial",
                               "observability", "focused"})
    try:
        path = os.path.join(REPO_ROOT, "ab", "eval", filename)
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        task_id_words: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "TASKS"
                    for t in node.targets):
                for sub in ast.walk(node.value):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        task_id_words.update(sub.value.lower().split("_"))
        names: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id.isupper()
                       for t in node.targets):
                continue
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    word = sub.value.strip().lower()
                    if (re.fullmatch(r"[a-z]{4,}", word)
                            and word not in LEAK_NAMES
                            and word not in task_id_words
                            and word not in GENERIC_WORDS):
                        names.append(word)
        return sorted(set(names))
    except (OSError, SyntaxError, ValueError):
        return []


def _new_task_names_from_reasoning2() -> list[str]:
    """Invented entity names from reasoning_tasks2, if present (see
    ``_new_task_names_from_module``)."""
    return _new_task_names_from_module("reasoning_tasks2.py")


def _new_task_names_from_benchmarks() -> list[str]:
    """Invented entity names from the Phase 10 benchmarks module
    (Worker 2), if present (see ``_new_task_names_from_module``).
    An absent module contributes no names -- the audit never fails
    on a module that has not landed yet."""
    return _new_task_names_from_module("benchmarks.py")


def run_leakage_audit() -> dict:
    """Audit (a): grep that no evaluation-world/task name appears in
    ab/ outside ab/eval/. Audit (b): for each mechanism, state
    whether any task evaluating it was also used to construct/tune
    it. A leak fails the audit suite. The (b) statements rest on the
    per-phase design docs (the tasks live only in ab/eval/ by
    construction) plus the grep in (a), which is the check that the
    names did not leak into ab/."""
    names = (list(LEAK_NAMES) + _new_task_names_from_reasoning2()
             + _new_task_names_from_benchmarks())
    hits: dict[str, list[str]] = {}
    hit_lines: dict[str, dict[str, str]] = {}
    ab_root = os.path.join(REPO_ROOT, "ab")
    eval_root = os.path.join(ab_root, "eval")
    for root, dirs, files in os.walk(ab_root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if os.path.abspath(root).startswith(os.path.abspath(eval_root)):
            continue
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            for name in names:
                match = re.search(r"\b" + re.escape(name) + r"\b", text)
                if match:
                    rel = os.path.relpath(path, REPO_ROOT)
                    hits.setdefault(name, []).append(rel)
                    line = next((ln.strip() for ln in text.splitlines()
                                 if re.search(r"\b" + re.escape(name) + r"\b", ln)),
                                "")
                    hit_lines.setdefault(name, {})[rel] = line[:200]
    mechanisms = {
        "reasoning (deduction / contradiction / abduction)":
            ("held-out by construction: the discriminating tasks are "
             "defined only in ab/eval/reasoning_tasks.py and were not "
             "used to construct the reasoning engine (Phase 4 design); "
             "the grep above is the check that their names did not "
             "leak into ab/ outside ab/eval/"),
        "new reasoning tasks (causal / analogy / abstraction / counterfactual)":
            ("held-out by construction: defined only in "
             "ab/eval/reasoning_tasks2.py (Worker 1); not used to "
             "construct any mechanism" if os.path.isfile(
                 os.path.join(REPO_ROOT, "ab", "eval", "reasoning_tasks2.py"))
             else "module ab.eval.reasoning_tasks2 absent -- no statement possible yet"),
        "transfer":
            ("hand-built A/B defined only in ab/eval/ (disconnected.py, "
             "transfer.py); the envgen-generated transfer task counts "
             "as held-out for transfer -- the generator seed is "
             "recorded in the transfer suite and the brain never sees "
             "it"),
        "learning (chamber worlds C/D)":
            ("defined only in ab/eval/learning_eval.py; not used to "
             "construct the learning mechanism (Phase 6 design)"),
        "execution (workshop worlds W1/W2)":
            ("defined only in ab/eval/execution_eval.py; not used to "
             "construct the execution mechanisms (Phase 7 design)"),
        "perception / imagination (treasure / gem worlds)":
            ("defined only in ab/eval/phase8_eval.py (treasure reuses "
             "the demo RoomWorld); not used to construct the "
             "perception / imagination mechanisms (Phase 8 design)"),
        "benchmarks (Tier 2/3 worlds, Phase 10)":
            ("defined only in ab/eval/benchmarks.py (Worker 2); not "
             "used to construct any mechanism (Phase 10 design P10-3)"
             if os.path.isfile(
                 os.path.join(REPO_ROOT, "ab", "eval", "benchmarks.py"))
             else "module ab.eval.benchmarks absent -- no statement possible yet"),
    }
    return {"names_checked": names,
            "scope": "ab/ outside ab/eval/ (python files)",
            "hits": hits, "n_leaks": sum(len(v) for v in hits.values()),
            "hit_lines": hit_lines,
            "mechanisms": mechanisms,
            "note": ("a leak is any hit: a name appearing in ab/ "
                     "outside ab/eval/ fails the audit suite -- the "
                     "grep is the check, and no occurrence is excused "
                     "here (even a docstring example counts as a hit; "
                     "whether it is a functional dependence is a "
                     "separate question, answered per mechanism above)")}


def _run_leakage_suite(ctx: dict) -> dict:
    """Suite (17): leakage audit. One experiment per name (passed =
    no hits outside ab/eval/) plus the mechanism-construction
    audit experiment (passed = statements recorded for every
    mechanism)."""
    started = time.monotonic()
    suite = _new_suite("leakage_audit", ctx["seed"], _mode(ctx["fast"]),
                       ctx["meta"])
    try:
        audit = run_leakage_audit()
        ctx["leakage_audit"] = audit
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return _suite_error("leakage_audit", ctx["seed"], _mode(ctx["fast"]),
                            ctx["meta"], exc, started)
    for name in audit["names_checked"]:
        hits = audit["hits"].get(name) or []
        _add_exp(suite, f"name/{name}", passed=not hits,
                 detail=(f"no occurrences in ab/ outside ab/eval/"
                         if not hits else f"LEAK: found in {hits}"),
                 evidence={"hits": hits,
                           "hit_lines": (audit.get("hit_lines") or {}).get(name, {})},
                 failure_cause=(f"leakage: eval name '{name}' appears in "
                                f"ab/ outside ab/eval/: {hits}")
                 if hits else None,
                 meta=ctx["meta"])
    _add_exp(suite, "mechanism_construction_audit", passed=True,
             detail="per-mechanism held-out statements recorded",
             evidence={"mechanisms": audit["mechanisms"]}, meta=ctx["meta"])
    return _finalise_suite(suite, started)


# ------------------------------------------------------------------ pipeline
# Ordered mapping suite name -> runner. Tests replace entries with
# stubs to exercise the exit-code contract without running the
# programme; production runs every runner in this order.
_SUITE_RUNNERS = {
    "discriminating": _run_discriminating_suite,
    "reasoning_tasks2": _run_reasoning2_suite,
    "transfer": _run_transfer_suite,
    "world_tasks": _run_world_tasks_suite,
    "learning": _run_learning_suite,
    "learning_curves": _run_learning_curves_suite,
    "execution": _run_execution_suite,
    "phase8": _run_phase8_suite,
    "long_horizon": _run_long_horizon_suite,
    "perception": _run_perception_suite,
    "memory": _run_memory_suite,
    "tool_use": _run_tool_use_suite,
    "planning": _run_planning_suite,
    "calibration": _run_calibration_suite,
    "ablations": _run_ablations_suite,
    "benchmarks": _run_benchmarks_suite,
    "failure_analysis": _run_failure_analysis_suite,
    "regression": _run_regression_suite,
    "leakage_audit": _run_leakage_suite,
}


def run_consolidated_evaluation(seed: int = 0, fast: bool = False,
                                record_path: str | None = DEFAULT_RECORD,
                                summary_path: str | None = DEFAULT_SUMMARY,
                                phase: str | None = None) -> dict:
    """Run the whole consolidated programme and return the record
    dict. Writes the record to ``record_path`` and the summary to
    ``summary_path`` when given; ``None`` writes nothing (the
    harness path -- a fast harness run must never overwrite any
    phase's canonical record or summary, or the regression record).
    The regression record is written next to the record (same
    directory, ``phase9-regression-record.json`` -- the cumulative
    per-phase history file introduced in Phase 9, not a Phase 9
    snapshot) only when a record path is given. ``phase`` names the
    phase this run belongs to (e.g. ``"10.5"``); only then does the
    run rebuild that phase's row in the regression record from its
    own numbers -- a run with no phase adds no row and relabels
    nothing (Phase 10.5 fix: the old builder wrote every run's
    numbers into the "9" row)."""
    started = time.monotonic()
    meta = build_environment_meta(seed)
    ctx: dict = {"seed": int(seed), "fast": bool(fast), "meta": meta,
                 "suites": {}, "phase": phase}
    suites: dict = {}
    for name, runner in _SUITE_RUNNERS.items():
        try:
            suite = runner(ctx)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            suite = _suite_error(name, ctx["seed"], _mode(fast), meta, exc,
                                 time.monotonic())
        suites[name] = suite
        ctx["suites"] = suites
    # Failure analysis runs at position 15 but must also cover the
    # regression and leakage suites that run after it: recompute it
    # over all suites now and replace the entry in place.
    analysis = build_failure_analysis(suites)
    ctx["failure_analysis"] = analysis
    fa_suite = suites.get("failure_analysis")
    if isinstance(fa_suite, dict):
        fa_suite["experiments"] = {}
        fa_suite["n_passed"] = fa_suite["n_failed"] = fa_suite["n_skipped"] = 0
        _add_exp(fa_suite, "failure_analysis_complete", passed=True,
                 detail=(f"{analysis['n_failures']} failure(s) across the "
                         f"programme; top mode: "
                         f"{analysis['top_failure_modes'][0]['mode']} "
                         f"({analysis['top_failure_modes'][0]['count']})"
                         if analysis["top_failure_modes"] else
                         "no failures in the programme"),
                 evidence={"analysis": analysis}, meta=meta)
    # The regression record is rebuilt here from the FINAL suites:
    # the in-suite build ran before the leakage audit, so it missed
    # that suite's experiments. The rebuilt record is what the
    # top-level ``regression`` key and the file carry; the regression
    # suite's own experiment evidence is refreshed to match.
    try:
        ctx["current_numbers"] = _current_numbers_for_regression(ctx)
        ctx["regression_record"] = build_regression_record(
            current=ctx["current_numbers"],
            current_phase=ctx.get("phase"))
        reg_suite = suites.get("regression")
        if isinstance(reg_suite, dict):
            exp = (reg_suite.get("experiments") or {}).get("regression_record")
            if isinstance(exp, dict):
                exp["evidence"] = {"rows_summary": _rows_summary(
                    ctx["regression_record"])}
    except Exception:  # noqa: BLE001 - the in-suite record stands
        pass
    total = time.monotonic() - started
    n_failed_suites = sum(1 for s in suites.values()
                          if isinstance(s, dict) and s.get("n_failed"))
    n_passed_suites = sum(1 for s in suites.values()
                          if isinstance(s, dict) and not s.get("n_failed"))
    all_passed = n_failed_suites == 0
    record = {
        "evaluation": "ab.eval.consolidated (consolidated evaluation)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "mode": _mode(fast),
        "environment": meta,
        "suites": suites,
        "learning_curves": ctx.get("learning_curves"),
        "calibration": {
            "streams": ctx.get("calibration_streams"),
            "note": ("Per-stream ECE default-1.0 vs calibrated and "
                     "Brier default-1.0 vs calibrated; streams are "
                     "reported separately and NEVER averaged into one "
                     "smoothed number (stream-dependence finding "
                     "preserved: see the per-stream ece_improved / "
                     "brier_improved flags).")},
        "failure_analysis": analysis,
        "leakage_audit": ctx.get("leakage_audit"),
        "ablations": {
            "variants": {"no_revision": ABLATION_REMOVED["no_revision"],
                         "no_causal": ABLATION_REMOVED["no_causal"]},
            "outcomes": ctx.get("ablation_outcomes")},
        "regression": ctx.get("regression_record"),
        "overall": {
            "n_suites": len(suites),
            "n_suites_passed": n_passed_suites,
            "n_suites_failed": n_failed_suites,
            "all_passed": all_passed,
            "total_duration_s": round(total, 3),
            "verdict": ("ALL SUITES PASSED" if all_passed else
                        "EVALUATION FAILED -- one or more suites failed; "
                        "see failure_analysis and the per-suite sections")},
    }
    if record_path is not None:
        try:
            os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
            with open(record_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        except OSError as exc:
            record.setdefault("write_errors", []).append(
                f"could not write record to {record_path}: {exc}")
            return record
        summary = summary_text(record, record_path=record_path)
        if summary_path is not None:
            try:
                with open(summary_path, "w", encoding="utf-8") as fh:
                    fh.write(summary)
            except OSError as exc:
                record.setdefault("write_errors", []).append(
                    f"could not write summary to {summary_path}: {exc}")
        # The regression record lives next to the consolidated record
        # (one file, one writer); only written on a real run, never
        # from a harness run with record_path=None.
        try:
            regression_path = os.path.join(
                os.path.dirname(record_path) or ".",
                "phase9-regression-record.json")
            write_regression_record(current=ctx.get("current_numbers"),
                                    current_phase=ctx.get("phase"),
                                    path=regression_path)
            record["regression_record_path"] = regression_path
        except (OSError, ValueError) as exc:
            record.setdefault("write_errors", []).append(
                f"could not write regression record: {exc}")
    return record


def summary_text(record: dict, record_path: str | None = None) -> str:
    """Human-readable summary, derived from the record dict only
    (the summary is derived from the machine-readable record, never
    the other way round). ``record_path`` names the file the record
    was written to, so the "Full record" line names the actual file
    (defaults to DEFAULT_RECORD when not given)."""
    env = record.get("environment") or {}
    overall = record.get("overall") or {}
    lines = [
        f"Consolidated evaluation -- {record.get('timestamp')} "
        f"(mode: {record.get('mode')}, seed: {record.get('seed')})",
        "",
        f"VERDICT: {overall.get('verdict')}",
        f"Suites: {overall.get('n_suites_passed')} passed, "
        f"{overall.get('n_suites_failed')} failed "
        f"(of {overall.get('n_suites')}), "
        f"{overall.get('total_duration_s')}s total",
        f"Environment: Python {env.get('python_version')}, "
        f"tesseract {env.get('tesseract_version')}, persistence "
        f"v{env.get('persistence_version')}, "
        f"{(env.get('line_counts') or {}).get('python_lines')} python lines "
        f"in {(env.get('line_counts') or {}).get('python_files')} files; "
        f"unit tests: {(env.get('test_counts') or {}).get('status')} "
        f"(ran {(env.get('test_counts') or {}).get('ran')})",
        "",
    ]
    for name, suite in (record.get("suites") or {}).items():
        if not isinstance(suite, dict):
            continue
        lines.append(
            f"[{name}] {suite.get('n_passed')} passed, "
            f"{suite.get('n_failed')} failed, "
            f"{suite.get('n_skipped')} skipped "
            f"({suite.get('duration_s')}s, mode {suite.get('mode')})")
        for exp_name, exp in (suite.get("experiments") or {}).items():
            if not isinstance(exp, dict):
                continue
            if exp.get("skipped"):
                lines.append(f"  {exp_name}: SKIPPED ({exp.get('detail')})")
            elif exp.get("passed") is False:
                lines.append(f"  {exp_name}: FAIL -- {exp.get('detail')}")
                if exp.get("failure_cause"):
                    lines.append(f"    cause: {exp['failure_cause']}")
            else:
                lines.append(f"  {exp_name}: PASS -- {exp.get('detail')}")
    cal = record.get("calibration") or {}
    streams = cal.get("streams") or {}
    if streams:
        lines.append("")
        lines.append("Calibration (per stream -- never averaged):")
        for name, stream in streams.items():
            if not isinstance(stream, dict):
                continue
            lines.append(
                f"  {name}: n={stream.get('n_predictions')}; ECE "
                f"{stream.get('ece_default_1.0')} -> "
                f"{stream.get('ece_calibrated')} "
                f"(improved={stream.get('ece_improved')}); Brier "
                f"{stream.get('brier_default_1.0')} -> "
                f"{stream.get('brier_calibrated')} "
                f"(improved={stream.get('brier_improved')})")
        lines.append("  Stream-dependence preserved: calibration helps "
                     "where the stream is unreliable and can hurt where "
                     "the stream is deterministic-reliable; see the "
                     "per-stream improved flags above.")
    fa = record.get("failure_analysis") or {}
    if fa.get("top_failure_modes"):
        lines.append("")
        lines.append("Failure analysis (top modes by count):")
        for mode in fa["top_failure_modes"]:
            lines.append(f"  {mode['mode']}: {mode['count']}")
    else:
        lines.append("")
        lines.append("Failure analysis: no failures in the programme.")
    leak = record.get("leakage_audit") or {}
    if leak:
        lines.append("")
        lines.append(f"Leakage audit: {leak.get('n_leaks')} leak(s) across "
                     f"{len(leak.get('names_checked') or [])} names checked "
                     f"(ab/ outside ab/eval/)")
    abl = record.get("ablations") or {}
    if abl.get("outcomes"):
        lines.append("")
        lines.append("Ablations (no_revision / no_causal vs full):")
        for task, outcome in (abl["outcomes"] or {}).items():
            lines.append(
                f"  {task}: full={'PASS' if outcome.get('full_passed') else 'FAIL'}, "
                f"no_revision={'PASS' if outcome.get('no_revision_passed') else 'FAIL'} "
                f"({'differed' if outcome.get('no_revision_differed') else 'same'}), "
                f"no_causal={'PASS' if outcome.get('no_causal_passed') else 'FAIL'} "
                f"({'differed' if outcome.get('no_causal_differed') else 'same'})")
    if not overall.get("all_passed"):
        lines.append("")
        lines.append("EVALUATION FAILED -- this is a failed evaluation, "
                     "not a successful report-writing process. See "
                     "failure_analysis above and the per-suite FAIL "
                     "lines.")
    lines.append("")
    _record_display = record_path or DEFAULT_RECORD
    try:
        _rel = os.path.relpath(_record_display, REPO_ROOT)
        # Only relativise paths inside the repo; a temp/elsewhere path
        # stays absolute instead of a "../../../../tmp/..." climb.
        if not _rel.startswith(".."):
            _record_display = _rel
    except ValueError:
        pass
    _regression_display = os.path.join(
        os.path.dirname(_record_display) or ".",
        "phase9-regression-record.json")
    lines.append(f"Full record: {_record_display}; regression "
                 f"record: {_regression_display}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolidated evaluation (all suites, seedable)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for generated environments / curves")
    parser.add_argument("--fast", action="store_true",
                        help="fast mode: execution + Phase 8 quick subsets")
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--phase", default=None,
                        help="phase label this run belongs to "
                             "(e.g. 10.5); only then does the run "
                             "rebuild that phase's row in the "
                             "regression record from its own numbers")
    args = parser.parse_args(argv)
    try:
        record = run_consolidated_evaluation(
            seed=args.seed, fast=args.fast,
            record_path=args.record, summary_path=args.summary,
            phase=args.phase)
    except Exception as exc:  # noqa: BLE001 - harness error, not a brain failure
        print(f"consolidated evaluation harness error: {type(exc).__name__}: {exc}")
        return 1
    print(summary_text(record, record_path=args.record))
    print(f"record written to {args.record}")
    print(f"summary written to {args.summary}")
    if record.get("write_errors"):
        for err in record["write_errors"]:
            print(f"write error: {err}")
        return 1
    if not (record.get("overall") or {}).get("all_passed"):
        # A failed suite is a failed evaluation, not a successful
        # report-writing process (design P9-4).
        print("EVALUATION FAILED -- one or more suites failed; see the "
              "record for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
