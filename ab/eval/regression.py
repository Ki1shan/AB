"""Regression tracking (design SS3.8; introduced in Phase 9).

``docs/phase9-regression-record.json``: per-phase key numbers
(tests, demo cycles, transfer, behaviours, eval passes per phase)
plus the current run's numbers when the consolidated runner is
invoked with ``--phase <label>``. A future phase diffs against it.
The Phase 8 numbers come from ``docs/phase8-harness-record.json`` /
``docs/phase8-eval-record.json`` as the previous row. A missing
on-disk record is reported as ``None`` with a note, never invented.

Coordinator fix (Phase 9 integration): the historical test counts
are read from the phase implementation reports (see
``_HISTORICAL_TESTS``), NOT from the on-disk per-phase record
files -- the harness rewrites those files identically on every run,
so their "tests" sections describe the latest run, not the phase's
own era. Demo/transfer/behaviours/eval-passes per phase are not
recoverable from those files for the same reason; they are None
with the source named, and the phase's implementation report is the
era evidence.

Parent fix (Phase 10.5 verification): the builder used to append
the CURRENT run's numbers under the hard-coded label ``"9"``
(``build_regression_record(phase9=...)``), so every consolidated
run after Phase 9 overwrote the Phase 9 row with its own numbers
-- the Phase 10 gate wrote 850 tests / 155/160 into the "9" row,
the Phase 10.5 gate wrote 929 / 156/160. The true Phase 9 row
(742 tests, 102/106) was lost from this file; it is restored here
from PHASE-9-IMPLEMENTATION-REPORT.md, and the per-experiment map
for Phase 9 stays None -- it was lost with the clobbered Phase 9
record (see docs/phase9-record-clobber-note.md), never
reconstructed. The current run now contributes a row ONLY when
invoked with an explicit phase label (``--phase``); a bare run
rebuilds the file from the tables below and the on-disk
consolidated records, and never relabels another phase's row.

Parent fix (Phase 12 verification): the same rebuild dropped the
Phase 11 row -- the Phase 11 gate's ``--phase 11`` run had appended
it to the file, but "11" was not in the history tables, so the
Phase 12 gate's rebuild lost it (1,057 tests, 156/160; the numbers
survive in PHASE-11-IMPLEMENTATION-REPORT.md and the Phase 11
addendum). "11" is now in the tables, and ``build_regression_record``
additionally carries forward any row already on disk whose phase
it does not regenerate, so a rebuild can never silently drop a
phase's row again.
"""
from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(REPO_ROOT, "docs", "phase9-regression-record.json")

# (phase label, record file) in order. The harness records carry the
# full per-phase sections; the Phase 8 eval record carries the
# experiment counts directly.
_PHASE_FILES = (
    ("3", "phase3-eval-record.json"),
    ("5", "phase5-eval-record.json"),
    ("6", "phase6-eval-record.json"),
    ("7", "phase7-eval-record.json"),
    ("8", "phase8-harness-record.json"),
)

# Consolidated (not harness) record files carrying the per-experiment
# pass map for a phase. A consolidated record is written only by an
# explicit standalone consolidated run, never by the harness, so it
# is stable era evidence for its phase -- unlike the harness records
# above, which are rewritten identically on every harness run.
_CONSOLIDATED_FILES = {
    "10": "phase10-eval-record.json",
    "10.5": "phase10.5-eval-record.json",
    "11": "phase11-eval-record.json",
}


def _load(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        return rec if isinstance(rec, dict) else None
    except (OSError, ValueError):
        return None


# Historical test counts, grounded in the phase implementation
# reports (the verified evidence), NOT in the on-disk per-phase
# record files: the harness rewrites phase3/5/6/7-eval-record.json
# (and phase8-harness-record.json) identically on every run, so
# their "tests" sections describe the LATEST run, not the phase's
# own era. Reading them as history would report every phase as
# having the current test count. The numbers below are the
# coordinator-verified counts from each PHASE-N-IMPLEMENTATION-
# REPORT.md (ran / failures / errors / skipped / ok). Phases 9, 10
# and 10.5 are restored here from their reports (the Phase 9 row
# was lost from this file by the pre-10.5 builder -- see the module
# docstring -- and is rebuilt from PHASE-9-IMPLEMENTATION-REPORT.md,
# not from any on-disk record).
_HISTORICAL_TESTS = {
    "3": {"ran": 224, "failures": 0, "errors": 0, "skipped": 1,
          "ok": True, "mode": "full",
          "source": "PHASE-3-IMPLEMENTATION-REPORT.md"},
    "5": {"ran": 376, "failures": 0, "errors": 0, "skipped": 1,
          "ok": True, "mode": "full",
          "source": "PHASE-5-IMPLEMENTATION-REPORT.md"},
    "6": {"ran": 476, "failures": 0, "errors": 0, "skipped": 1,
          "ok": True, "mode": "full",
          "source": "PHASE-6-IMPLEMENTATION-REPORT.md"},
    "7": {"ran": 573, "failures": 0, "errors": 0, "skipped": 1,
          "ok": True, "mode": "full",
          "source": "PHASE-7-IMPLEMENTATION-REPORT.md"},
    "8": {"ran": 669, "failures": 0, "errors": 0, "skipped": 0,
          "ok": True, "mode": "full",
          "source": "PHASE-8-IMPLEMENTATION-REPORT.md"},
    "9": {"ran": 742, "failures": 0, "errors": 0, "skipped": 0,
          "ok": True, "mode": "full",
          "source": "PHASE-9-IMPLEMENTATION-REPORT.md"},
    "10": {"ran": 850, "failures": 0, "errors": 0, "skipped": 0,
           "ok": True, "mode": "full",
           "source": "PHASE-10-IMPLEMENTATION-REPORT.md"},
    "10.5": {"ran": 929, "failures": 0, "errors": 0, "skipped": 0,
             "ok": True, "mode": "full",
             "source": "PHASE-10.5-IMPLEMENTATION-REPORT.md"},
    "11": {"ran": 1057, "failures": 0, "errors": 0, "skipped": 0,
           "ok": True, "mode": "full",
           "source": "PHASE-11-IMPLEMENTATION-REPORT.md"},
}

# Per-phase eval summaries that are report-grounded only (the
# per-experiment pass map is NOT recoverable for these phases).
# Phase 9: the per-experiment machine-readable record was clobbered
# during the Phase 10 gate (see docs/phase9-record-clobber-note.md);
# the counts below are from PHASE-9-IMPLEMENTATION-REPORT.md and the
# map stays None, never reconstructed.
_HISTORICAL_EVAL_SUMMARY = {
    "9": {"n_passed": 102, "n_failed": 4, "n_total": 106,
          "source": "PHASE-9-IMPLEMENTATION-REPORT.md"},
}


def _tests_of(rec: dict | None) -> dict:
    if not rec:
        return {"ran": None, "failures": None, "errors": None,
                "skipped": None, "ok": None, "mode": None,
                "source": "record absent"}
    t = rec.get("tests") or {}
    return {"ran": t.get("ran"), "failures": t.get("failures"),
            "errors": t.get("errors"), "skipped": t.get("skipped"),
            "ok": t.get("ok"), "mode": rec.get("mode"),
            "source": "harness record tests section"}


def _eval_passes_from_consolidated(rec: dict | None) -> dict | None:
    """Per-experiment pass map (``suite/experiment`` -> bool) from a
    consolidated record, or None when the record is absent."""
    if not rec:
        return None
    passes: dict = {}
    for suite_name, suite in (rec.get("suites") or {}).items():
        if not isinstance(suite, dict):
            continue
        for exp_name, exp in (suite.get("experiments") or {}).items():
            if isinstance(exp, dict) and "passed" in exp and not exp.get("skipped"):
                passes[f"{suite_name}/{exp_name}"] = bool(exp.get("passed"))
    return passes


def _row_from_harness_record(phase: str, rec: dict | None) -> dict:
    """One regression row from a harness record (Phases 3/5/6/7/8).

    Every field is either read from the record or None with the
    source named; nothing is guessed. Eval passes per phase are the
    counts the record itself carries.
    """
    row: dict = {"phase": phase, "record_present": rec is not None}
    if rec is None:
        row.update({"tests": _tests_of(None), "demo": None,
                    "transfer": None, "behaviours": None,
                    "eval_passes": None,
                    "note": "on-disk record absent; nothing invented"})
        return row
    # Tests: the phase's own verified count (see _HISTORICAL_TESTS).
    # The other fields are NOT recoverable per phase from the on-disk
    # records -- the harness rewrites them identically on every run,
    # so they describe the latest run, not this phase's era. They are
    # None with the source named, never back-filled from the latest
    # run; the phase's implementation report is the era evidence.
    row["tests"] = dict(_HISTORICAL_TESTS.get(
        phase, {"ran": None, "failures": None, "errors": None,
                "skipped": None, "ok": None, "mode": None,
                "source": "no historical count recorded"}))
    row["demo"] = None
    row["transfer"] = None
    row["behaviours"] = None
    row["eval_passes"] = None
    row["note"] = ("tests: phase's verified count from its "
                   "implementation report; demo/transfer/behaviours/"
                   "eval_passes: not recoverable per phase from the "
                   "on-disk records (rewritten identically each "
                   "harness run) -- see PHASE-%s-IMPLEMENTATION-"
                   "REPORT.md" % phase)
    return row


def _row_from_history(phase: str, repo_root: str) -> dict:
    """One regression row for Phases 9, 10, 10.5 and 11, grounded in the
    phase's implementation report (test count) and its on-disk
    consolidated record (per-experiment pass map, where one exists).
    Nothing is back-filled from a later run."""
    tests = dict(_HISTORICAL_TESTS.get(
        phase, {"ran": None, "failures": None, "errors": None,
                "skipped": None, "ok": None, "mode": None,
                "source": "no historical count recorded"}))
    row: dict = {"phase": phase, "record_present": True, "tests": tests,
                 "demo": None, "behaviours": None}
    # Transfer: the disconnected-world result recorded in each phase
    # report (continued world B beats fresh) -- True for 9, 10, 10.5.
    row["transfer"] = {"transferred": True}
    summary = _HISTORICAL_EVAL_SUMMARY.get(phase)
    filename = _CONSOLIDATED_FILES.get(phase)
    rec = _load(os.path.join(repo_root, "docs", filename)) if filename else None
    passes = _eval_passes_from_consolidated(rec)
    if passes is not None:
        row["eval_passes"] = passes
        row["eval_passes_summary"] = {
            "n_passed": sum(1 for v in passes.values() if v is True),
            "n_failed": sum(1 for v in passes.values() if v is False),
            "n_total": len(passes)}
        row["source"] = ("consolidated record %s (this phase's own "
                         "explicit run)" % filename)
    elif summary is not None:
        row["eval_passes"] = None
        row["eval_passes_summary"] = {
            "n_passed": summary["n_passed"],
            "n_failed": summary["n_failed"],
            "n_total": summary["n_total"]}
        row["source"] = summary["source"]
        row["note"] = ("per-experiment pass map not recoverable: the "
                       "Phase 9 machine-readable record was clobbered "
                       "during the Phase 10 gate (see "
                       "docs/phase9-record-clobber-note.md); counts "
                       "are from the phase report, the map is None, "
                       "never reconstructed")
    else:
        row["eval_passes"] = None
        row["note"] = "no consolidated record on disk; nothing invented"
    return row


def _row_from_current_run(phase: str, current: dict) -> dict:
    """One regression row from a consolidated run's own numbers,
    labelled with the phase that actually ran. ``current`` is a dict
    with the keys ``tests``, ``demo``, ``transfer``, ``behaviours``
    and ``eval_passes`` (each already in the row shape); missing keys
    become None."""
    row: dict = {"phase": phase, "record_present": current is not None}
    if current is None:
        row.update({"tests": _tests_of(None), "demo": None,
                    "transfer": None, "behaviours": None,
                    "eval_passes": None,
                    "note": "no current-run numbers passed in"})
        return row
    row["tests"] = current.get("tests")
    row["demo"] = current.get("demo")
    row["transfer"] = current.get("transfer")
    row["behaviours"] = current.get("behaviours")
    passes = current.get("eval_passes") or {}
    row["eval_passes"] = passes
    row["eval_passes_summary"] = {
        "n_passed": sum(1 for v in passes.values() if v is True),
        "n_failed": sum(1 for v in passes.values() if v is False),
        "n_total": len(passes)}
    row["source"] = "consolidated run --phase %s (this run)" % phase
    return row


def build_regression_record(current: dict | None = None,
                            current_phase: str | None = None,
                            repo_root: str = REPO_ROOT) -> dict:
    """Build the regression record: one row per phase -- Phases
    3/5/6/7/8 from the on-disk harness records, Phases 9/10/10.5/11
    from the verified history tables -- plus, ONLY when
    ``current_phase`` names the phase that ran, that phase's row
    rebuilt from this run's numbers (``current``). A run with no
    phase label contributes no row: the file is rebuilt from
    history alone, and no other phase's row is ever relabelled.
    Returns the record dict (does not write it)."""
    rows: list[dict] = []
    for phase, filename in _PHASE_FILES:
        rec = _load(os.path.join(repo_root, "docs", filename))
        rows.append(_row_from_harness_record(phase, rec))
    # Phase 8's own eval record (experiment counts, incl. the P-D
    # probe) is the previous row's direct evidence; attach it to the
    # Phase 8 row rather than inventing a second Phase 8 row.
    p8_eval = _load(os.path.join(repo_root, "docs", "phase8-eval-record.json"))
    if p8_eval is not None:
        rows[-1]["phase8_eval_record"] = {
            "n_passed": p8_eval.get("n_passed"),
            "n_failed": p8_eval.get("n_failed"),
            "n_skipped": p8_eval.get("n_skipped"),
            "mode": p8_eval.get("mode"),
            "pd_probe_passed": ((p8_eval.get("experiments") or {})
                                .get("pd_overgeneralisation_probe") or {})
                                .get("passed")}
    for phase in ("9", "10", "10.5", "11"):
        rows.append(_row_from_history(phase, repo_root))
    if current_phase is not None:
        # The running phase's row holds this run's numbers, replaced
        # in place so the table order is kept; any other phase's row
        # is untouched.
        new_row = _row_from_current_run(current_phase, current)
        for i, r in enumerate(rows):
            if r.get("phase") == current_phase:
                rows[i] = new_row
                break
        else:
            rows.append(new_row)
    # Parent fix (Phase 12 verification): the rebuild above drops any
    # row whose phase is not in the tables -- that is how the Phase 11
    # row (appended by the Phase 11 gate's --phase 11 run, 1,057
    # tests) was lost when the Phase 12 gate rebuilt the file. Carry
    # forward any row already on disk whose phase this builder does
    # not regenerate, in file order, so a rebuild can never silently
    # drop a phase's row again. (Phase 11 is now in the tables above,
    # so this is a safety net for future phases, not the restore
    # mechanism for 11 itself.)
    existing = _load(os.path.join(repo_root, "docs",
                                  "phase9-regression-record.json"))
    if existing is not None:
        known = {r.get("phase") for r in rows}
        for old_row in existing.get("rows") or []:
            if old_row.get("phase") not in known:
                rows.append(old_row)
                known.add(old_row.get("phase"))
    return {"record": "phase9-regression-record",
            "note": ("Per-phase key numbers: Phases 3/5/6/7/8 from the "
                     "on-disk harness records (test counts from the "
                     "phase reports), Phases 9/10/10.5/11 from the "
                     "verified history tables and each phase's own "
                     "consolidated record. A run invoked with --phase "
                     "<label> rebuilds that phase's row from its own "
                     "numbers; a run with no --phase adds no row and "
                     "relabels nothing. Missing records are None, "
                     "never invented."),
            "rows": rows}


def write_regression_record(current: dict | None = None,
                            current_phase: str | None = None,
                            path: str | None = DEFAULT_RECORD,
                            repo_root: str = REPO_ROOT) -> dict:
    """Build the regression record and write it to ``path`` (default
    docs/phase9-regression-record.json). Returns the record dict."""
    record = build_regression_record(current=current,
                                     current_phase=current_phase,
                                     repo_root=repo_root)
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1)
    return record
