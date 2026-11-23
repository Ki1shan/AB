"""Phase 3 evaluation harness (mandate SS3 + SS10).

Runnable with ONE command::

    python3 -m ab.eval.harness [--fast] [--record PATH] [--summary PATH]

It runs, in order:

(a) the full unittest suite (subprocess, counts captured);
(b) ``python -m ab.demo`` (subprocess, exit code + treasure outcome);
(c) the knowledge-disconnected experiment (``ab.eval.disconnected``,
    subprocess, record read back);
(d) resource measurement: wall time per phase, and peak child memory
    via ``resource.getrusage(RUSAGE_CHILDREN)`` when measurable --
    otherwise the honest string ``"not measured"``, never an invented
    number;
(e) baseline comparisons on the treasure task and (full mode only) on
    disconnected world A, for three variants of the brain:

    - ``full``: the brain as built;
    - ``no_learning``: the learner replaced after construction by a
      no-op object with the same public API (``record`` does nothing,
      ``propose_rules`` returns ``[]``). Planner costs therefore never
      adapt from experience. Implemented by monkeypatching, not by
      editing ``brain.py``;
    - ``non_integrated``: a :class:`Brain` subclass overriding
      ``_reason_from_wm`` with a no-op, so no in-cycle WM read /
      inference runs. Same monkeypatching approach.

    The harness reports the numbers (task completion + cycles) and
    does NOT declare a winner without data.

Outputs (defaults under ``docs/``):
- ``phase3-eval-record.json`` -- machine-readable record with
  ``timestamp``, ``mode``, ``tests``, ``demo``, ``disconnected``,
  ``baselines``, ``resources`` and ``repo_state`` fields.
- ``phase3-eval-summary.txt`` -- human-readable summary.
- Phase 5 adds, in parallel and without touching the Phase 3 files:
  ``phase5-eval-record.json`` / ``phase5-eval-summary.txt`` -- the
  full new record (everything above PLUS the ``transfer`` and
  ``world_tasks`` sections) and its summary. The harness also runs
  the transfer experiment and the world-model acceptance tasks
  in-process (see ``_run_transfer`` / ``_run_world_tasks``).
- Phase 6 adds, in parallel and without touching the Phase 3/5
  files: ``phase6-eval-record.json`` / ``phase6-eval-summary.txt``
  -- the full new record (everything above PLUS the
  ``learning_evaluation`` section) and its summary. The harness
  runs the learning evaluation in-process (see ``_run_learning``);
  the evaluation itself also writes
  ``docs/phase6-learning-record.json``. A learning-eval failure is
  recorded in the section (``error``), never hidden.
- Phase 7 adds, in parallel and without touching the Phase 3/5/6
  files: ``phase7-eval-record.json`` / ``phase7-eval-summary.txt``
  -- the full new record (everything above PLUS the
  ``execution_evaluation`` section) and its summary. The harness
  runs the execution evaluation in-process (see
  ``_run_execution``); an execution-eval failure is recorded in
  the section (``error``), never hidden. The execution evaluation
  itself (``python -m ab.eval.execution_eval``) writes
  ``docs/phase7-execution-record.json`` when run standalone; the
  harness does NOT write that file -- it passes
  ``record_path=None`` -- so a fast harness run never overwrites
  the canonical execution record with a fast-mode run (the
  Phase 6 SS14.8 quirk, fixed the same way for all records: the
  eval tests pass explicit temp paths for every record).
- Phase 8 adds, in parallel and without touching the Phase 3/5/6/7
  files: ``phase8-harness-record.json`` / ``phase8-harness-summary.txt``
  -- the full new record (everything above PLUS the
  ``phase8_evaluation`` section) and its summary. (Named
  ``phase8-harness-*`` rather than ``phase8-eval-*`` because the
  Phase 8 evaluation itself owns ``docs/phase8-eval-record.json``
  when run standalone; one file, one writer.) The harness
  runs the Phase 8 evaluation in-process (see ``_run_phase8``);
  a Phase 8 failure is recorded in the section (``error``), never
  hidden. The Phase 8 evaluation itself
  (``python -m ab.eval.phase8_eval``) writes
  ``docs/phase8-eval-record.json`` when run standalone; the
  harness does NOT write that file -- it passes
  ``record_path=None`` -- so a fast harness run never overwrites
  the canonical Phase 8 record with a fast-mode run.
- Phase 9 adds, in parallel and without touching the Phase 3/5/6/7/8
  files: ``phase9-harness-record.json`` / ``phase9-harness-summary.txt``
  -- the full new record (everything above PLUS the
  ``phase9_evaluation`` section) and its summary. (Named
  ``phase9-harness-*`` because the consolidated evaluation itself
  owns ``docs/consolidated-eval-record.json`` when run standalone; one
  file, one writer.) The harness runs the consolidated evaluation
  in-process (see ``_run_phase9``); a Phase 9 failure is recorded
  in the section (``error``), never hidden. The consolidated
  evaluation is called with ``record_path=None`` /
  ``summary_path=None`` so a harness run -- fast or full -- never
  overwrites the consolidated evaluation's canonical record,
  summary, or regression record; those files are written only by
  the standalone ``python -m ab.eval.consolidated``. Note the contract difference:
  the consolidated runner exits non-zero when ANY suite fails
  (design P9-4); the harness keeps its own documented contract
  (non-zero only on a failing test suite or a harness error), so a
  failing Phase 9 suite is recorded in the harness record, not
  turned into a harness failure.
- Phase 10 adds, in parallel and without touching the Phase 3/5/6/7/8/9
  files: ``phase10-harness-record.json`` / ``phase10-harness-summary.txt``
  -- the full new record (everything above PLUS the
  ``phase10_evaluation`` section) and its summary. (Named
  ``phase10-harness-*`` because the consolidated evaluation itself
  owns ``docs/consolidated-eval-record.json`` when run standalone; one
  file, one writer.) The harness runs the Phase 10 benchmarks
  in-process (see ``_run_phase10``); a benchmark failure is
  recorded in the section (``error`` / per-task PASS-FAIL), never
  hidden. In fast mode the benchmarks are skipped with the reason
  recorded (there is no quick subset for them) -- a fast harness
  run never claims benchmark numbers it did not measure. The
  consolidated evaluation (Phase 9 section) also runs the
  benchmarks as its suite 19 in full mode, so a full harness run
  measures them twice (once standalone here, once inside the
  consolidated programme) -- the same duplication the Phase 8
  evaluation already has, and both numbers are recorded.
- Phase 11 adds, in parallel and without touching the Phase 3/5/6/7/8/9/10
  files: ``phase11-harness-record.json`` / ``phase11-harness-summary.txt``
  -- the full new record (everything above PLUS the
  ``phase11_evaluation`` section) and its summary. (Named
  ``phase11-harness-*`` because the research-tasks evaluation itself
  owns ``docs/phase11-research-tasks-record.json`` when run standalone;
  one file, one writer.) The harness runs the held-out research tasks
  in-process (see ``_run_phase11``); a task failure is recorded in
  the section (per-task PASS-FAIL with numbers), never hidden. The
  tasks are cheap, so they run in fast mode too, with the mode
  recorded. No consolidated suite is added (design §9).

``--fast`` runs a small test subset, the demo, the disconnected
experiment, and treasure-only baselines, so the test suite can invoke
the harness without going slow; the record says ``"mode": "fast"``
and notes what was skipped. Re-runs produce the same counts
(timestamps differ, which the record notes).

Exit code is 0 when the record was written AND the test suite passed.
A failed discriminating task is a recorded result, not a harness
crash, so the harness still exits 0 when only a task fails. But a
failing test suite is a failed evaluation, not a successful
report-writing process: when the suite fails the harness still writes
the record and summary, prints an explicit ``SUITE FAIL`` in the
summary, and exits non-zero. Non-zero also on a harness error (e.g.
the record cannot be written). No LLM, stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase3-eval-record.json")
DEFAULT_SUMMARY = os.path.join(REPO_ROOT, "docs", "phase3-eval-summary.txt")
# Phase 5 parallel outputs (the Phase 3 filenames are left untouched).
DEFAULT_RECORD5 = os.path.join(REPO_ROOT, "docs", "phase5-eval-record.json")
DEFAULT_SUMMARY5 = os.path.join(REPO_ROOT, "docs", "phase5-eval-summary.txt")
# Phase 6 parallel outputs (the Phase 3/5 filenames are left untouched).
DEFAULT_RECORD6 = os.path.join(REPO_ROOT, "docs", "phase6-eval-record.json")
DEFAULT_SUMMARY6 = os.path.join(REPO_ROOT, "docs", "phase6-eval-summary.txt")
# Phase 7 parallel outputs (the Phase 3/5/6 filenames are left untouched).
DEFAULT_RECORD7 = os.path.join(REPO_ROOT, "docs", "phase7-eval-record.json")
DEFAULT_SUMMARY7 = os.path.join(REPO_ROOT, "docs", "phase7-eval-summary.txt")
# Phase 8 parallel outputs. NOTE the naming break from the Phase
# 5/6/7 convention (phaseN-eval-record.json): the Phase 8
# evaluation itself writes docs/phase8-eval-record.json when run
# standalone, so the harness's full record (everything PLUS the
# Phase 8 section) lives at docs/phase8-harness-record.json
# instead -- one file, one writer, no silent overwrite either way.
DEFAULT_RECORD8 = os.path.join(REPO_ROOT, "docs", "phase8-harness-record.json")
DEFAULT_SUMMARY8 = os.path.join(REPO_ROOT, "docs", "phase8-harness-summary.txt")
# Phase 9 parallel outputs. Named phase9-harness-* because the
# consolidated evaluation itself owns docs/consolidated-eval-record.json
# when run standalone -- one file, one writer, no silent overwrite
# either way (same naming break as Phase 8, for the same reason).
DEFAULT_RECORD9 = os.path.join(REPO_ROOT, "docs", "phase9-harness-record.json")
DEFAULT_SUMMARY9 = os.path.join(REPO_ROOT, "docs", "phase9-harness-summary.txt")
# Phase 10 parallel outputs. Named phase10-harness-* for the same
# reason as Phase 9: the consolidated evaluation owns
# docs/consolidated-eval-record.json, so the harness's full record
# (everything PLUS the Phase 10 benchmarks section) lives here --
# one file, one writer, no silent overwrite either way.
DEFAULT_RECORD10 = os.path.join(REPO_ROOT, "docs", "phase10-harness-record.json")
DEFAULT_SUMMARY10 = os.path.join(REPO_ROOT, "docs", "phase10-harness-summary.txt")
# Phase 11 parallel outputs. Named phase11-harness-* for the same
# reason as Phase 9/10: the Phase 11 research-tasks evaluation itself
# owns docs/phase11-research-tasks-record.json when run standalone,
# so the harness's full record (everything PLUS the Phase 11
# research-tasks section) lives here -- one file, one writer, no
# silent overwrite either way.
DEFAULT_RECORD11 = os.path.join(REPO_ROOT, "docs", "phase11-harness-record.json")
DEFAULT_SUMMARY11 = os.path.join(REPO_ROOT, "docs", "phase11-harness-summary.txt")

# The suite's timeout test leaves a 10 s sleeper thread lingering at
# interpreter exit (Phase 1 report SS7); the subprocess therefore takes
# ~11 s even though the tests themselves run in under a second.
# Phase 9 coordinator: the suite itself now takes ~122 s (742 tests),
# so the 120 s timeout expired mid-run and the harness reported
# SUITE FAIL with ran=None. 300 s leaves headroom for the current
# suite plus growth; the timeout is a harness guard, not a test
# result, and must stay above the suite's real wall time.
# Phase 10 (Worker 4, measured 2026-11-21): the suite is now 833
# tests and takes 551 s with light contention (and 1150 s when four
# other full-suite runs share the same 1.5 shared CPUs), so 300 s
# expires mid-run and the harness reports SUITE FAIL with ran=None.
# 1200 s keeps ~2x headroom over the light-contention measurement;
# under heavy contention a run may still time out, and that is a
# contention finding, not a suite failure -- re-run it alone.
SUITE_TIMEOUT_S = 1200
DEMO_TIMEOUT_S = 120
DISCONNECTED_TIMEOUT_S = 120


# ------------------------------------------------------------------ subprocess phases
def _run(argv, cwd=REPO_ROOT, timeout=120):
    """Run a subprocess, capturing output. Returns (rc, stdout, stderr,
    wall_s). ``rc`` is None when the run itself failed (timeout / spawn
    error), with the reason in stderr."""
    started = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started
    except subprocess.TimeoutExpired:
        return None, "", f"timeout after {timeout}s", time.monotonic() - started
    except OSError as exc:
        return None, "", f"{type(exc).__name__}: {exc}", time.monotonic() - started


def _parse_unittest_output(stderr: str, stdout: str) -> dict:
    """Parse ``python -m unittest`` counts from its output (it writes to
    stderr). Missing numbers are reported as None, never guessed."""
    text = stderr + "\n" + stdout
    ran = re.search(r"Ran (\d+) tests?", text)
    skipped = re.search(r"skipped=(\d+)", text)
    failures = re.search(r"failures=(\d+)", text)
    errors = re.search(r"errors=(\d+)", text)
    ok = "\nOK" in text or text.startswith("OK")
    return {
        "ran": int(ran.group(1)) if ran else None,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "failures": int(failures.group(1)) if failures else 0,
        "errors": int(errors.group(1)) if errors else 0,
        "ok": bool(ok),
        "raw_tail": text.strip().splitlines()[-3:],
    }


def _run_suite(fast: bool) -> dict:
    if fast:
        argv = [sys.executable, "-m", "unittest",
                "tests.test_phase1_vocab", "tests.test_phase1_state"]
        scope = "fast subset (tests.test_phase1_vocab, tests.test_phase1_state)"
    else:
        argv = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
        scope = "full suite (tests/)"
    rc, out, err, wall = _run(argv, timeout=SUITE_TIMEOUT_S)
    counts = _parse_unittest_output(err, out)
    counts.update({"command": " ".join(argv), "scope": scope,
                   "exit_code": rc, "duration_s": round(wall, 3)})
    return counts


def _run_demo() -> dict:
    rc, out, err, wall = _run([sys.executable, "-m", "ab.demo"],
                              timeout=DEMO_TIMEOUT_S)
    m = re.search(r"outcome:\s*(\w+)", out)
    return {"command": "python -m ab.demo", "exit_code": rc,
            "outcome": m.group(1) if m else None,
            "treasure_line": next((l for l in out.splitlines()
                                   if l.startswith("outcome:")), None),
            "duration_s": round(wall, 3),
            "stderr_tail": err.strip().splitlines()[-3:] if err.strip() else []}


def _run_disconnected() -> dict:
    record_path = os.path.join(REPO_ROOT, "docs", "phase3-disconnected-record.json")
    rc, out, err, wall = _run(
        [sys.executable, "-m", "ab.eval.disconnected", "--record", record_path],
        timeout=DISCONNECTED_TIMEOUT_S)
    result = {"command": "python -m ab.eval.disconnected",
              "exit_code": rc, "record_path": record_path,
              "duration_s": round(wall, 3),
              "stdout": out.strip().splitlines()}
    try:
        with open(record_path, encoding="utf-8") as fh:
            rec = json.load(fh)
        result["result"] = {
            "world_a": {k: rec["world_a"][k] for k in
                        ("completed", "stop_reason", "cycles", "failed_attempts")},
            "world_b_fresh": {k: rec["world_b_fresh"][k] for k in
                              ("completed", "stop_reason", "cycles", "failed_attempts")},
            "world_b_continued": {k: rec["world_b_continued"][k] for k in
                                  ("completed", "stop_reason", "cycles", "failed_attempts")},
            "transfer": rec["transfer"]["transferred"],
            "behaviours_observed": sum(
                1 for v in rec["behaviours"].values() if v["observed"]),
            "behaviours_total": len(rec["behaviours"]),
            "failures": rec["failures"],
        }
    except (OSError, ValueError, KeyError) as exc:
        result["result"] = None
        result["record_error"] = f"{type(exc).__name__}: {exc}"
    if err.strip():
        result["stderr_tail"] = err.strip().splitlines()[-3:]
    return result


# ------------------------------------------------------------------ baselines (in-process)
class _NoOpLearner:
    """Learner-shaped no-op: the same public API Brain uses
    (``record``, ``propose_rules``, ``to_json``), but nothing is ever
    learned -- planner costs never adapt from experience."""

    def __init__(self, *args, **kwargs):
        self.episodes = []

    def record(self, ep):
        pass

    def propose_rules(self):
        return []

    def episodes_for_induction(self):
        # Phase 4: Brain._reason_from_wm reads the learner's episodes
        # for induction every cycle. The no-op learner records none,
        # so this is faithfully [].
        return []

    def failures_for(self, action, situation, args=None):
        # Phase 10.5 (P10.5-7): the no-op learner records no
        # failures, so no plan is ever a loop plan for it.
        return 0

    def to_json(self):
        return {"episodes": [], "counts": []}


def _treasure_run(brain_factory, max_cycles: int = 60) -> dict:
    """One treasure-task run (the demo's RoomWorld) for one variant."""
    from ..body.environment import RoomWorld, room_world_actions

    env = RoomWorld()
    brain = brain_factory(room_world_actions(), {"compute", "environment"})
    brain.add_goal("get the treasure", {"has_treasure": True})
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[0]
    return {"completed": goal.status == "done", "goal_status": goal.status,
            "stop_reason": stop, "cycles": brain.cycle,
            "pick_lock_failure_rate": brain.planner.failure_rate("pick_lock")}


def _run_baselines(fast: bool) -> dict:
    """Run the three variants on the treasure task and (full mode) on
    disconnected world A. Returns numbers only -- no winner is
    declared here."""
    from ..brain import Brain
    from .disconnected import run_experiment

    def full_factory(actions, granted):
        return Brain(actions, granted=granted)

    def no_learning_factory(actions, granted):
        brain = Brain(actions, granted=granted)
        brain.learner = _NoOpLearner()      # monkeypatched, brain.py untouched
        return brain

    class NonIntegratedBrain(Brain):
        def _reason_from_wm(self):          # no in-cycle WM read / inference
            return []

    def non_integrated_factory(actions, granted):
        return NonIntegratedBrain(actions, granted=granted)

    factories = {"full": full_factory, "no_learning": no_learning_factory,
                 "non_integrated": non_integrated_factory}
    baselines = {
        "mechanism": (
            "Variants built in-process by monkeypatching/subclassing "
            "Brain (brain.py NOT edited): no_learning replaces "
            "brain.learner with a no-op object of the same public API; "
            "non_integrated subclasses Brain and overrides "
            "_reason_from_wm with a no-op. No winner is declared -- "
            "the numbers below are the comparison."),
        "treasure": {name: _treasure_run(fn) for name, fn in factories.items()},
    }
    if not fast:
        disc = {}
        for name, fn in factories.items():
            rec = run_experiment(brain_factory=fn, worlds=("A",))
            disc[name] = {k: rec["world_a"][k] for k in
                          ("completed", "stop_reason", "cycles", "failed_attempts")}
        baselines["disconnected_world_a"] = disc
    else:
        baselines["disconnected_world_a"] = (
            "skipped in fast mode (the subprocess disconnected run "
            "above still covers the full experiment)")
    return baselines


# ------------------------------------------------------------------ discriminating tasks (Phase 4)
def _run_discriminating() -> dict:
    """The three Phase 4 discriminating held-out tasks, in-process, for
    all three variants (see ab.eval.reasoning_tasks). A failed task is
    a recorded result, never a harness crash."""
    from .reasoning_tasks import run_discriminating_tasks

    started = time.monotonic()
    section = run_discriminating_tasks()
    section["duration_s"] = round(time.monotonic() - started, 3)
    return section


# ------------------------------------------------------------------ transfer + world tasks (Phase 5)
def _run_transfer() -> dict:
    """The Phase 5 transfer experiment, in-process (see
    ab.eval.transfer). A failed transfer is a recorded result, never
    a harness crash."""
    from .transfer import run_transfer_experiment

    started = time.monotonic()
    section = run_transfer_experiment()
    section["duration_s"] = round(time.monotonic() - started, 3)
    return section


def _run_world_tasks(transfer_record: dict | None) -> dict:
    """The five Phase 5 world-model acceptance tasks, in-process, for
    all three variants (see ab.eval.world_tasks). The harness's own
    transfer record is passed in so the ``full`` variant's transfer
    task reuses it instead of re-running the experiment. A failed
    task is a recorded result, never a harness crash."""
    from .world_tasks import run_world_tasks

    started = time.monotonic()
    section = run_world_tasks(transfer_record=transfer_record)
    section["duration_s"] = round(time.monotonic() - started, 3)
    return section


# ------------------------------------------------------------------ learning evaluation (Phase 6)
def _run_learning() -> dict:
    """The Phase 6 learning evaluation, in-process (see
    ab.eval.learning_eval). A learning-eval failure is recorded,
    never hidden: on exception the section carries the error and
    the harness continues (existing sections are unaffected)."""
    from .learning_eval import run_learning_evaluation

    started = time.monotonic()
    try:
        section = run_learning_evaluation()
        section["duration_s"] = round(time.monotonic() - started, 3)
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "note": ("learning evaluation failed; recorded here, "
                         "not hidden -- see the error above")}


# ------------------------------------------------------------------ execution evaluation (Phase 7)
def _run_execution(fast: bool) -> dict:
    """The Phase 7 execution evaluation, in-process (see
    ab.eval.execution_eval). Fast mode runs the quick subset only
    (multi-tool + carried items + calibration + pruning, ~2-3 s);
    full mode runs all twelve experiments (~28 s). An
    execution-eval failure is recorded, never hidden: on exception
    the section carries the error and the harness continues
    (existing sections are unaffected). ``record_path=None`` so the
    harness never writes the evaluation's own canonical record
    (docs/phase7-execution-record.json); that file is written only
    by the standalone ``python -m ab.eval.execution_eval``."""
    from .execution_eval import run_execution_evaluation

    started = time.monotonic()
    try:
        section = run_execution_evaluation(fast=fast, record_path=None)
        section["duration_s"] = round(time.monotonic() - started, 3)
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "note": ("execution evaluation failed; recorded here, "
                         "not hidden -- see the error above")}


# ------------------------------------------------------------------ phase 8 evaluation
def _run_phase8(fast: bool) -> dict:
    """The Phase 8 perception/imagination evaluation, in-process (see
    ab.eval.phase8_eval). Fast mode runs the cheap subset only (the
    workshop P-D probe is skipped, not passed); full mode runs all
    thirteen experiments. A Phase 8 failure is recorded, never hidden:
    on exception the section carries the error and the harness
    continues (existing sections are unaffected). ``record_path=None``
    so the harness never writes the evaluation's own canonical record
    (docs/phase8-eval-record.json); that file is written only by the
    standalone ``python -m ab.eval.phase8_eval``."""
    from .phase8_eval import run_phase8_evaluation

    started = time.monotonic()
    try:
        section = run_phase8_evaluation(fast=fast, record_path=None)
        section["duration_s"] = round(time.monotonic() - started, 3)
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "note": ("phase 8 evaluation failed; recorded here, "
                         "not hidden -- see the error above")}


# ------------------------------------------------------------------ phase 9 consolidated evaluation
def _run_phase9(fast: bool) -> dict:
    """The Phase 9 consolidated evaluation, in-process (see
    ab.eval.consolidated). Fast mode runs the execution + Phase 8
    quick subsets inside it (the consolidated record says so);
    full mode runs the whole programme. A Phase 9 failure is
    recorded, never hidden: on exception the section carries the
    error and the harness continues (existing sections are
    unaffected). ``record_path=None`` / ``summary_path=None`` so
    the harness never writes the consolidated evaluation's own
    canonical record (docs/consolidated-eval-record.json), summary, or
    regression record; those files are written only by the
    standalone ``python -m ab.eval.consolidated``. The consolidated
    runner's exit-code contract (non-zero on any failing suite,
    design P9-4) does NOT apply here: the harness keeps its own
    documented contract, so a failing Phase 9 suite is a recorded
    result in this section, not a harness failure."""
    from .consolidated import run_consolidated_evaluation

    started = time.monotonic()
    try:
        section = run_consolidated_evaluation(
            seed=0, fast=fast, record_path=None, summary_path=None)
        section["duration_s"] = round(time.monotonic() - started, 3)
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "note": ("phase 9 consolidated evaluation failed; "
                         "recorded here, not hidden -- see the error "
                         "above")}


# ------------------------------------------------------------------ phase 10 benchmarks
def _run_phase10(fast: bool) -> dict:
    """The Phase 10 benchmark tiers, in-process (see
    ab.eval.benchmarks, Worker 2). Runs ``run_benchmarks(seed=0)``
    and records the per-task numbers (pass/fail, cycles, detail,
    per-variant results, difficulty vector) in the section. A
    benchmark failure is recorded, never hidden: on exception the
    section carries the error and the harness continues (existing
    sections are unaffected). An absent benchmarks module is the
    same: the section records the absence with the reason, it never
    crashes the harness. Fast mode skips the benchmarks with the
    reason recorded (there is no quick subset for them) -- a fast
    run never claims numbers it did not measure."""
    if fast:
        return {"skipped": True,
                "reason": ("fast mode: the Phase 10 benchmarks have no "
                           "quick subset; run in full mode to measure them"),
                "mode": "fast",
                "duration_s": 0.0}
    started = time.monotonic()
    try:
        import importlib

        module = importlib.import_module("ab.eval.benchmarks")
        fn = getattr(module, "run_benchmarks", None)
        if fn is None:
            raise AttributeError(
                "ab.eval.benchmarks has no run_benchmarks entry point")
        try:
            section = fn(seed=0)
        except TypeError:
            section = fn()
        if not isinstance(section, dict) or not isinstance(
                section.get("tasks"), dict):
            raise ValueError(
                "run_benchmarks returned an unrecognised result shape: "
                "expected {\"tasks\": {name: {...}}}")
        section["duration_s"] = round(time.monotonic() - started, 3)
        section["mode"] = "full"
        section["seed"] = 0
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "mode": "full",
                "note": ("phase 10 benchmarks failed; recorded here, "
                         "not hidden -- see the error above")}


# ------------------------------------------------------------------ phase 11 research tasks
def _run_phase11(fast: bool) -> dict:
    """The Phase 11 held-out research tasks, in-process (see
    ab.eval.research_tasks, Worker 3). Runs ``run_research_tasks(seed=0)``
    and records the per-task pass/fail with numbers (cycles, failed
    attempts, sources). A task failure is recorded, never hidden: on
    exception the section carries the error and the harness continues
    (existing sections are unaffected). The tasks are cheap (~0.1 s),
    so they run in fast mode too, with the mode recorded — a fast run
    never claims full-mode numbers it did not measure (the tasks
    themselves are seed-fixed and mode-independent). The harness
    never writes the evaluation's own canonical record
    (docs/phase11-research-tasks-record.json); that file is written
    only by the standalone ``python -m ab.eval.research_tasks``.
    No consolidated suite is added (design §9): the consolidated
    programme stays 19 suites / 160 experiments."""
    started = time.monotonic()
    try:
        import importlib

        module = importlib.import_module("ab.eval.research_tasks")
        fn = getattr(module, "run_research_tasks", None)
        if fn is None:
            raise AttributeError(
                "ab.eval.research_tasks has no run_research_tasks entry point")
        try:
            section = fn(seed=0)
        except TypeError:
            section = fn()
        if not isinstance(section, dict) or not isinstance(
                section.get("tasks"), dict):
            raise ValueError(
                "run_research_tasks returned an unrecognised result shape: "
                "expected {\"tasks\": {name: {...}}}")
        section["duration_s"] = round(time.monotonic() - started, 3)
        section["mode"] = "fast" if fast else "full"
        section["seed"] = 0
        return section
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"error": f"{type(exc).__name__}: {exc}",
                "duration_s": round(time.monotonic() - started, 3),
                "mode": "fast" if fast else "full",
                "note": ("phase 11 research tasks failed; recorded here, "
                         "not hidden -- see the error above")}


# ------------------------------------------------------------------ resources / repo state
def _resources(suite_s: float, demo_s: float, disc_s: float,
               total_s: float) -> dict:
    """Peak child memory via getrusage when measurable; the honest
    string 'not measured' otherwise. ru_maxrss is KiB on Linux."""
    try:
        peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        peak = int(peak_kb) if peak_kb else "not measured"
    except (ValueError, OSError, AttributeError):
        peak = "not measured"
    return {"suite_wall_s": round(suite_s, 3), "demo_wall_s": round(demo_s, 3),
            "disconnected_wall_s": round(disc_s, 3),
            "total_wall_s": round(total_s, 3),
            "peak_child_rss_kb": peak,
            "peak_child_rss_note": (
                "ru_maxrss of waited-for child processes (KiB, Linux); "
                "'not measured' when the platform does not report it -- "
                "no number is ever invented here")}


def _repo_state() -> dict:
    py_files = test_files = 0
    for root, _dirs, files in os.walk(REPO_ROOT):
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                py_files += 1
                if root.endswith("tests") or f.startswith("test_"):
                    test_files += 1
    return {"vcs": ("git" if os.path.isdir(os.path.join(REPO_ROOT, ".git"))
                    else "none (no .git directory in this workspace)"),
            "python_files": py_files, "test_files": test_files,
            "note": ("git-less repo: no commit to cite, so the state is "
                     "described by file counts; re-runs should produce "
                     "the same test/demo/disconnected counts, timestamps "
                     "excepted")}


# ------------------------------------------------------------------ main
def run(fast: bool = False) -> dict:
    started = time.monotonic()
    tests = _run_suite(fast)
    demo = _run_demo()
    disconnected = _run_disconnected()
    baselines = _run_baselines(fast)
    discriminating = _run_discriminating()
    transfer = _run_transfer()
    world_tasks = _run_world_tasks(transfer)
    learning = _run_learning()
    execution = _run_execution(fast)
    phase8 = _run_phase8(fast)
    phase9 = _run_phase9(fast)
    phase10 = _run_phase10(fast)
    phase11 = _run_phase11(fast)
    total = time.monotonic() - started
    record = {
        "harness": "ab.eval.harness (Phase 3; Phase 5 sections added; "
                   "Phase 6 learning-evaluation section added; "
                   "Phase 7 execution-evaluation section added; "
                   "Phase 8 perception/imagination section added; "
                   "Phase 9 consolidated-evaluation section added; "
                   "Phase 10 benchmarks section added)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "fast" if fast else "full",
        "tests": tests,
        "demo": demo,
        "disconnected": disconnected,
        "baselines": baselines,
        "discriminating_tasks": discriminating,
        "transfer": transfer,
        "world_tasks": world_tasks,
        "learning_evaluation": learning,
        "execution_evaluation": execution,
        "phase8_evaluation": phase8,
        "phase9_evaluation": phase9,
        "phase10_evaluation": phase10,
        "phase11_evaluation": phase11,
        "resources": _resources(tests["duration_s"], demo["duration_s"],
                                disconnected["duration_s"], total),
        "repo_state": _repo_state(),
        "notes": [
            "Re-running this harness should produce the same counts; "
            "timestamps will differ.",
            "Exit code 0 means the record was written and the test "
            "suite passed. A failed discriminating task is a recorded "
            "result, not a harness crash -- the harness still exits 0 "
            "when only a task fails. A failing test suite is a failed "
            "evaluation: the record is still written, but the summary "
            "says SUITE FAIL and the exit code is non-zero.",
            "Timed-out tool threads linger (Phase 1 honest limitation): "
            "a timeout abandons the thread, it does not kill it.",
            "No live-network HTTPS proof is claimed here.",
        ],
    }
    return record


def _summary_text(record: dict) -> str:
    t, d, x = record["tests"], record["demo"], record["disconnected"]
    lines = [
        f"Phase 3 evaluation record -- {record['timestamp']} (mode: {record['mode']})",
        "",
        f"Tests: ran {t['ran']}, failures {t['failures']}, errors {t['errors']}, "
        f"skipped {t['skipped']}, ok={t['ok']} ({t['scope']}, "
        f"{t['duration_s']}s, exit {t['exit_code']})",
        f"Demo: exit {d['exit_code']}, treasure outcome {d['outcome']} "
        f"({d['duration_s']}s)",
    ]
    xr = x.get("result") or {}
    if xr:
        lines.append(
            f"Disconnected: A completed={xr['world_a']['completed']} "
            f"({xr['world_a']['cycles']} cycles, "
            f"{xr['world_a']['failed_attempts']} failed attempts); "
            f"B fresh completed={xr['world_b_fresh']['completed']}; "
            f"B continued completed={xr['world_b_continued']['completed']}; "
            f"transfer={xr['transfer']}; behaviours "
            f"{xr['behaviours_observed']}/{xr['behaviours_total']} observed; "
            f"failures recorded: {xr['failures']}")
    else:
        lines.append(f"Disconnected: record error: {x.get('record_error')}")
    lines.append("Baselines (treasure task):")
    for name, res in record["baselines"]["treasure"].items():
        lines.append(
            f"  {name}: completed={res['completed']} "
            f"({res['goal_status']}, {res['cycles']} cycles, "
            f"pick_lock failure rate {res['pick_lock_failure_rate']:.2f})")
    disc_b = record["baselines"].get("disconnected_world_a")
    if isinstance(disc_b, dict):
        lines.append("Baselines (disconnected world A):")
        for name, res in disc_b.items():
            lines.append(
                f"  {name}: completed={res['completed']} "
                f"({res['stop_reason']}, {res['cycles']} cycles, "
                f"{res['failed_attempts']} failed attempts)")
    else:
        lines.append(f"Baselines (disconnected world A): {disc_b}")
    lines.append("Discriminating tasks (Phase 4 held-out tasks, per variant -- "
                 "no winner is declared):")
    disc_tasks = record.get("discriminating_tasks", {}).get("tasks", {})
    for task_id, task in disc_tasks.items():
        parts = []
        for name, res in task["variants"].items():
            parts.append(f"{name}={'PASS' if res.get('passed') else 'FAIL'}"
                         f" ({res.get('cycles')} cycles)")
        lines.append(f"  {task_id}: " + " | ".join(parts))
        for name, res in task["variants"].items():
            lines.append(f"    {name}: {res.get('detail')}")
    transfer = record.get("transfer") or {}
    if transfer:
        a, b, c = (transfer.get("world_a_fresh", {}),
                   transfer.get("world_b_fresh", {}),
                   transfer.get("world_b_continued", {}))
        lines.append(
            f"Transfer (Phase 5): fresh A {a.get('failed_attempts')} failed "
            f"attempt(s)/{a.get('cycles')} cycles; fresh B "
            f"{b.get('failed_attempts')}/{b.get('cycles')}; continued B "
            f"{c.get('failed_attempts')}/{c.get('cycles')}; "
            f"transferred={transfer.get('transferred')}; entry via "
            f"{(transfer.get('entry_protocol') or {}).get('method')}; "
            f"patterns before B="
            f"{((transfer.get('mechanism') or {}).get('patterns_present_before_b') or {}).get('count')}")
    world_tasks = record.get("world_tasks") or {}
    wt_tasks = world_tasks.get("tasks", {}) if isinstance(world_tasks, dict) else {}
    if wt_tasks:
        lines.append("World tasks (Phase 5 acceptance tasks, per variant -- "
                     "no winner is declared):")
        for task_id, task in wt_tasks.items():
            parts = []
            for name, res in task["variants"].items():
                parts.append(f"{name}={'PASS' if res.get('passed') else 'FAIL'}"
                             f" ({res.get('cycles')} cycles)")
            lines.append(f"  {task_id}: " + " | ".join(parts))
    learn = record.get("learning_evaluation") or {}
    if learn:
        if learn.get("error"):
            lines.append("Learning evaluation (Phase 6): ERROR -- "
                         f"{learn['error']} (recorded, not hidden)")
        else:
            lines.append("Learning evaluation (Phase 6, held-out C1-C5 / "
                         "D1-D5, per variant -- no winner is declared):")
            for name, res in (learn.get("variants") or {}).items():
                s = res.get("summary", {})
                lines.append(
                    f"  {name}: success {s.get('success_rate')} "
                    f"({s.get('n_completed')}/{s.get('n_tasks')}), cycles "
                    f"mean {((s.get('cycles') or {}).get('mean'))} "
                    f"(std {((s.get('cycles') or {}).get('std'))}), failed "
                    f"attempts mean "
                    f"{((s.get('failed_attempts') or {}).get('mean'))} "
                    f"(std {((s.get('failed_attempts') or {}).get('std'))}), "
                    f"transfer {s.get('transfer_tasks')}")
            acc = learn.get("acceptance_experiment") or {}
            if acc:
                lines.append(
                    f"  Six-step acceptance experiment: "
                    f"{'PASS' if acc.get('passed') else 'FAIL'}")
            two = learn.get("two_family_transfer") or {}
            if two:
                lines.append(
                    f"  Two-family transfer: "
                    f"{two.get('n_families_transferred')} families "
                    f"transferred ({', '.join(two.get('families_transferred') or [])})")
            cal = (learn.get("calibration") or {}).get("overall") or {}
            if cal:
                lines.append(
                    f"  Calibration: ECE default-1.0 "
                    f"{cal.get('ece_default_1.0')} vs calibrated "
                    f"{cal.get('ece_calibrated')}; Brier "
                    f"{cal.get('brier_default_1.0')} vs "
                    f"{cal.get('brier_calibrated')}")
            if learn.get("failures"):
                lines.append("  Learning-evaluation failures recorded: "
                             + "; ".join(learn["failures"]))
    exe = record.get("execution_evaluation") or {}
    if exe:
        if exe.get("error"):
            lines.append("Execution evaluation (Phase 7): ERROR -- "
                         f"{exe['error']} (recorded, not hidden)")
        else:
            lines.append("Execution evaluation (Phase 7, workshop worlds "
                         "W1/W2, 12 acceptance experiments -- a failed "
                         "criterion is a recorded result, not a harness "
                         "crash):")
            for name, res in (exe.get("experiments") or {}).items():
                if isinstance(res, dict):
                    lines.append(f"  {name}: "
                                 f"{'PASS' if res.get('passed') else 'FAIL'}")
                else:
                    lines.append(f"  {name}: {res}")
            carried = exe.get("carried_items") or {}
            if carried:
                parts = []
                for key in ("pc_pd_transfer", "consolidation_firing",
                            "repeated_mistake_firing",
                            "concept_preference_firing",
                            "wrong_promoted_pattern_retirement"):
                    item = carried.get(key) or {}
                    parts.append(f"{key}={item.get('status')}")
                lines.append("  Carried Phase 6 items: " + ", ".join(parts))
            cal = (exe.get("calibration") or {}).get("overall") or {}
            if cal:
                lines.append(
                    f"  Calibration (workshop stream, "
                    f"{cal.get('n_predictions')} predictions): ECE "
                    f"default-1.0 {cal.get('ece_default_1.0')} vs "
                    f"calibrated {cal.get('ece_calibrated')}; Brier "
                    f"{cal.get('brier_default_1.0')} vs "
                    f"{cal.get('brier_calibrated')}")
            pru = exe.get("pruning") or {}
            if pru:
                lines.append(
                    f"  Pruning (pattern-evidence retention at cap "
                    f"{pru.get('cap')}): "
                    f"{pru.get('pattern_evidence_retained')}/"
                    f"{pru.get('pattern_evidence_episodes')} retained")
            if exe.get("failures"):
                lines.append("  Execution-evaluation failures recorded: "
                             + "; ".join(exe["failures"]))
            if exe.get("worker_dependencies"):
                lines.append("  Worker follow-ups recorded: "
                             + "; ".join(exe["worker_dependencies"]))
    p8 = record.get("phase8_evaluation") or {}
    if p8:
        if p8.get("error"):
            lines.append("Phase 8 evaluation: ERROR -- "
                         f"{p8['error']} (recorded, not hidden)")
        else:
            lines.append("Phase 8 evaluation (perception + imagination, "
                         "13 acceptance experiments -- a failed criterion "
                         "is a recorded result, not a harness crash):")
            for name, res in (p8.get("experiments") or {}).items():
                if isinstance(res, dict) and res.get("skipped"):
                    lines.append(f"  {name}: SKIPPED ({res.get('reason')})")
                elif isinstance(res, dict) and "passed" in res:
                    lines.append(f"  {name}: "
                                 f"{'PASS' if res.get('passed') else 'FAIL'}")
                else:
                    lines.append(f"  {name}: {res}")
            lines.append(
                f"  Phase 8: {p8.get('n_passed')} passed, "
                f"{p8.get('n_failed')} failed, "
                f"{p8.get('n_skipped')} skipped")
            if p8.get("worker_dependencies"):
                lines.append("  Worker follow-ups recorded: "
                             + "; ".join(p8["worker_dependencies"]))
    p9 = record.get("phase9_evaluation") or {}
    if p9:
        if p9.get("error"):
            lines.append("Phase 9 consolidated evaluation: ERROR -- "
                         f"{p9['error']} (recorded, not hidden)")
        else:
            overall = p9.get("overall") or {}
            lines.append(
                f"Phase 9 consolidated evaluation: "
                f"{overall.get('verdict', 'no verdict recorded')} "
                f"({overall.get('n_suites_passed')} suites passed, "
                f"{overall.get('n_suites_failed')} failed, "
                f"{p9.get('duration_s')}s; mode {p9.get('mode')})")
            for sname, suite in (p9.get("suites") or {}).items():
                if not isinstance(suite, dict):
                    continue
                lines.append(
                    f"  [{sname}] {suite.get('n_passed')} passed, "
                    f"{suite.get('n_failed')} failed, "
                    f"{suite.get('n_skipped')} skipped")
            leak = p9.get("leakage_audit") or {}
            if leak:
                lines.append(f"  Leakage audit: {leak.get('n_leaks')} leak(s)")
            lines.append("  (The consolidated runner's own contract is "
                         "non-zero on any failing suite; the harness keeps "
                         "its own contract, so a failing Phase 9 suite is "
                         "a recorded result here, not a harness failure.)")
    p10 = record.get("phase10_evaluation") or {}
    if p10:
        if p10.get("skipped"):
            lines.append("Phase 10 benchmarks: SKIPPED -- "
                         f"{p10.get('reason')}")
        elif p10.get("error"):
            lines.append("Phase 10 benchmarks: ERROR -- "
                         f"{p10['error']} (recorded, not hidden)")
        else:
            tasks = p10.get("tasks") or {}
            n_pass = sum(1 for tsk in tasks.values()
                         if isinstance(tsk, dict) and tsk.get("passed"))
            lines.append(
                f"Phase 10 benchmarks (Tier 2/3 tasks, seed "
                f"{p10.get('seed')}, mode {p10.get('mode')}): "
                f"{n_pass}/{len(tasks)} tasks passed by the full brain "
                f"({p10.get('duration_s')}s; no winner is declared):")
            for task_id, tsk in tasks.items():
                if not isinstance(tsk, dict):
                    lines.append(f"  {task_id}: {tsk}")
                    continue
                variants = tsk.get("variants") or {}
                if variants:
                    parts = []
                    for name, res in variants.items():
                        if isinstance(res, dict):
                            parts.append(
                                f"{name}={'PASS' if res.get('passed') else 'FAIL'}"
                                f" ({res.get('cycles')} cycles)")
                    lines.append(f"  {task_id}: " + " | ".join(parts))
                else:
                    lines.append(
                        f"  {task_id}: "
                        f"{'PASS' if tsk.get('passed') else 'FAIL'} "
                        f"({tsk.get('cycles')} cycles)")
                lines.append(f"    {tsk.get('detail')}")
                difficulty = tsk.get("difficulty")
                if difficulty:
                    lines.append(f"    difficulty: {difficulty}")
    p11 = record.get("phase11_evaluation") or {}
    if p11:
        if p11.get("error"):
            lines.append("Phase 11 research tasks: ERROR -- "
                         f"{p11['error']} (recorded, not hidden)")
        else:
            tasks = p11.get("tasks") or {}
            n_pass = sum(1 for tsk in tasks.values()
                         if isinstance(tsk, dict) and tsk.get("passed"))
            lines.append(
                f"Phase 11 research tasks (held-out, seed "
                f"{p11.get('seed')}, mode {p11.get('mode')}): "
                f"{n_pass}/{len(tasks)} tasks passed "
                f"({p11.get('duration_s')}s; a failed task is a recorded "
                f"result, not a harness failure):")
            for task_id, tsk in tasks.items():
                if not isinstance(tsk, dict):
                    lines.append(f"  {task_id}: {tsk}")
                    continue
                lines.append(
                    f"  {task_id}: "
                    f"{'PASS' if tsk.get('passed') else 'FAIL'} "
                    f"({tsk.get('cycles')} cycles, "
                    f"{tsk.get('failed_attempts')} failed attempts)")
                lines.append(f"    {tsk.get('detail')}")
    if not t["ok"]:
        lines.append("SUITE FAIL -- the test suite failed; this evaluation "
                     "did not pass even though the record was written.")
    r = record["resources"]
    lines.append(
        f"Resources: total wall {r['total_wall_s']}s "
        f"(suite {r['suite_wall_s']}s, demo {r['demo_wall_s']}s, "
        f"disconnected {r['disconnected_wall_s']}s); "
        f"peak child RSS {r['peak_child_rss_kb']} KiB")
    lines.append(f"Repo state: {record['repo_state']['vcs']}; "
                 f"{record['repo_state']['python_files']} python files, "
                 f"{record['repo_state']['test_files']} test files")
    lines.append("")
    lines.append("No winner is declared from the baselines -- see the "
                 "numbers above. Full record: docs/phase3-eval-record.json; "
                 "Phase 5 record (same content plus transfer + world_tasks): "
                 "docs/phase5-eval-record.json; Phase 6 record (same content "
                 "plus the learning evaluation): "
                 "docs/phase6-eval-record.json; learning-evaluation record: "
                 "docs/phase6-learning-record.json; Phase 7 record (same "
                 "content plus the execution evaluation): "
                 "docs/phase7-eval-record.json; execution-evaluation record: "
                 "docs/phase7-execution-record.json; Phase 8 harness record "
                 "(same content plus the Phase 8 evaluation): "
                 "docs/phase8-harness-record.json; Phase 8 evaluation record: "
                 "docs/phase8-eval-record.json; Phase 9 harness record "
                 "(same content plus the Phase 9 consolidated evaluation): "
                 "docs/phase9-harness-record.json; Consolidated "
                 "evaluation record: docs/consolidated-eval-record.json; "
                 "Phase 10 harness record (same content plus the "
                 "Phase 10 benchmarks): "
                 "docs/phase10-harness-record.json; Phase 11 harness "
                 "record (same content plus the Phase 11 research "
                 "tasks): docs/phase11-harness-record.json; Phase 11 "
                 "research-tasks record: "
                 "docs/phase11-research-tasks-record.json")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 evaluation harness")
    parser.add_argument("--fast", action="store_true",
                        help="small test subset, treasure-only baselines")
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--record5", default=DEFAULT_RECORD5,
                        help="Phase 5 record path (full record incl. "
                             "transfer + world_tasks sections)")
    parser.add_argument("--summary5", default=DEFAULT_SUMMARY5,
                        help="Phase 5 summary path")
    parser.add_argument("--record6", default=DEFAULT_RECORD6,
                        help="Phase 6 record path (full record incl. "
                             "the learning-evaluation section)")
    parser.add_argument("--summary6", default=DEFAULT_SUMMARY6,
                        help="Phase 6 summary path")
    parser.add_argument("--record7", default=DEFAULT_RECORD7,
                        help="Phase 7 record path (full record incl. "
                             "the execution-evaluation section)")
    parser.add_argument("--summary7", default=DEFAULT_SUMMARY7,
                        help="Phase 7 summary path")
    parser.add_argument("--record8", default=DEFAULT_RECORD8,
                        help="Phase 8 record path (full record incl. "
                             "the Phase 8 evaluation section; named "
                             "phase8-harness-record.json because the "
                             "Phase 8 evaluation itself owns "
                             "phase8-eval-record.json)")
    parser.add_argument("--summary8", default=DEFAULT_SUMMARY8,
                        help="Phase 8 summary path")
    parser.add_argument("--record9", default=DEFAULT_RECORD9,
                        help="Phase 9 record path (full record incl. "
                             "the Phase 9 consolidated-evaluation "
                             "section; named phase9-harness-record.json "
                             "because the consolidated evaluation itself "
                             "owns consolidated-eval-record.json)")
    parser.add_argument("--summary9", default=DEFAULT_SUMMARY9,
                        help="Phase 9 summary path")
    parser.add_argument("--record10", default=DEFAULT_RECORD10,
                        help="Phase 10 record path (full record incl. "
                             "the Phase 10 benchmarks section; named "
                             "phase10-harness-record.json because the "
                             "consolidated evaluation itself owns "
                             "consolidated-eval-record.json)")
    parser.add_argument("--summary10", default=DEFAULT_SUMMARY10,
                        help="Phase 10 summary path")
    parser.add_argument("--record11", default=DEFAULT_RECORD11,
                        help="Phase 11 record path (full record incl. "
                             "the Phase 11 research-tasks section; named "
                             "phase11-harness-record.json because the "
                             "research-tasks evaluation itself owns "
                             "phase11-research-tasks-record.json)")
    parser.add_argument("--summary11", default=DEFAULT_SUMMARY11,
                        help="Phase 11 summary path")
    args = parser.parse_args(argv)
    try:
        record = run(fast=args.fast)
    except Exception as exc:
        print(f"harness error: {type(exc).__name__}: {exc}")
        return 1
    try:
        for path, content in ((args.record, json.dumps(record, indent=1)),
                              (args.summary, _summary_text(record)),
                              (args.record5, json.dumps(record, indent=1)),
                              (args.summary5, _summary_text(record)),
                              (args.record6, json.dumps(record, indent=1)),
                              (args.summary6, _summary_text(record)),
                              (args.record7, json.dumps(record, indent=1)),
                              (args.summary7, _summary_text(record)),
                              (args.record8, json.dumps(record, indent=1)),
                              (args.summary8, _summary_text(record)),
                              (args.record9, json.dumps(record, indent=1)),
                              (args.summary9, _summary_text(record)),
                              (args.record10, json.dumps(record, indent=1)),
                              (args.summary10, _summary_text(record)),
                              (args.record11, json.dumps(record, indent=1)),
                              (args.summary11, _summary_text(record))):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
    except OSError as exc:
        print(f"harness error: cannot write output: {exc}")
        return 1
    print(_summary_text(record))
    print(f"record written to {args.record}")
    print(f"summary written to {args.summary}")
    print(f"Phase 5 record written to {args.record5}")
    print(f"Phase 5 summary written to {args.summary5}")
    print(f"Phase 6 record written to {args.record6}")
    print(f"Phase 6 summary written to {args.summary6}")
    print(f"Phase 7 record written to {args.record7}")
    print(f"Phase 7 summary written to {args.summary7}")
    print(f"Phase 8 record written to {args.record8}")
    print(f"Phase 8 summary written to {args.summary8}")
    print(f"Phase 9 record written to {args.record9}")
    print(f"Phase 9 summary written to {args.summary9}")
    print(f"Phase 10 record written to {args.record10}")
    print(f"Phase 10 summary written to {args.summary10}")
    print(f"Phase 11 record written to {args.record11}")
    print(f"Phase 11 summary written to {args.summary11}")
    if not record["tests"]["ok"]:
        # A failing suite is a failed evaluation, not a successful
        # report-writing process: the record is written, but the exit
        # code says so visibly. A failed discriminating task alone
        # does NOT fail the harness (it is a recorded result).
        print("SUITE FAIL -- test suite failed; see the record for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
