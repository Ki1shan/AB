"""Parent-verification regression tests for the consolidated record paths.

Phase 10 gate finding (2026-11-21): the consolidated runner's default
record/summary paths pointed at ``docs/phase9-eval-record.json`` and
``docs/phase9-eval-summary.txt``, so a bare
``python3 -m ab.eval.consolidated --seed 0`` overwrote the canonical
Phase 9 record with current-run numbers (see
``docs/phase9-record-clobber-note.md``). The defaults are now
phase-neutral current-run files (``docs/consolidated-eval-record.json``
/ ``docs/consolidated-eval-summary.txt``); a phase's canonical record
is written only when its path is passed explicitly. These tests pin
that: a bare run writes the neutral files and leaves every phase
record byte-identical.

All runs here use tiny stub suites patched into ``_SUITE_RUNNERS`` --
never the real brain -- so each test takes seconds, not a minute.
No existing test file is touched; no test is weakened.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest import mock

from ab.eval import consolidated
from ab.eval.consolidated import (
    _add_exp,
    summary_text,
)

DOCS = os.path.join(consolidated.REPO_ROOT, "docs")

# Files a bare consolidated run must never change. The Phase 9 record
# and summary already hold Phase 10 numbers from the gate clobber
# (see docs/phase9-record-clobber-note.md); they are evidence and must
# come out of any run byte-identical.
IDENTITY_GUARDED = (
    "phase9-eval-record.json",
    "phase9-eval-summary.txt",
)

# The regression record is the cumulative per-phase history, written
# next to the record on every real run by design -- a run legitimately
# rewrites it. The test saves and restores its bytes so the tree is
# left exactly as found, but does NOT assert identity after the run.
RESTORE_ONLY = ("phase9-regression-record.json",)

NEUTRAL = (
    "consolidated-eval-record.json",
    "consolidated-eval-summary.txt",
)


def _passing_suite(name):
    # Same plain-dict stub shape as tests/test_phase9_eval.py.
    suite = {"suite": name, "experiments": {}, "n_passed": 0,
             "n_failed": 0, "n_skipped": 0, "duration_s": 0.0,
             "seed": 0, "mode": "full", "environment": {}}
    _add_exp(suite, f"{name}/check", passed=True, cycles=1,
             detail="stub pass", evidence={}, meta={})
    return suite


def _stub_runners():
    return {name: (lambda ctx, _n=name: _passing_suite(_n))
            for name in consolidated._SUITE_RUNNERS}


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestDefaultRecordPaths(unittest.TestCase):
    def test_default_record_paths_are_phase_neutral(self):
        # The defect was a phase name in the default path: a bare run
        # then overwrote that phase's canonical record. The defaults
        # must name no phase at all.
        for path in (consolidated.DEFAULT_RECORD,
                     consolidated.DEFAULT_SUMMARY):
            name = os.path.basename(path)
            self.assertNotIn("phase", name)
            self.assertTrue(name.startswith("consolidated-eval-"),
                            name)

    def test_summary_names_the_neutral_record(self):
        # The summary's "Full record" line used to hardcode the Phase 9
        # path; it must name the file the record was actually written to.
        with mock.patch.dict(consolidated._SUITE_RUNNERS,
                             _stub_runners(), clear=True):
            record = consolidated.run_consolidated_evaluation(
                seed=7, record_path=None, summary_path=None)
        text = summary_text(record)
        self.assertIn("Consolidated evaluation --", text)
        self.assertIn("consolidated-eval-record.json", text)
        self.assertNotIn("phase9-eval-record.json", text)

    def test_bare_run_writes_neutral_files_and_leaves_phase_records_untouched(self):
        # The exact gate scenario, with stub suites: a bare main()
        # (no --record/--summary) writes the neutral files and leaves
        # the guarded files byte-identical. Everything the run writes
        # is restored afterwards so the test leaves the tree exactly
        # as found.
        saved = {}
        for name in IDENTITY_GUARDED + RESTORE_ONLY + NEUTRAL:
            path = os.path.join(DOCS, name)
            saved[name] = _read_bytes(path) if os.path.exists(path) else None
        try:
            with mock.patch.dict(consolidated._SUITE_RUNNERS,
                                 _stub_runners(), clear=True):
                rc = consolidated.main(argv=["--seed", "7"])
            self.assertEqual(rc, 0)
            for name in NEUTRAL:
                path = os.path.join(DOCS, name)
                self.assertTrue(os.path.exists(path), name)
                self.assertIn(b"consolidated",
                              _read_bytes(path), name)
            for name in IDENTITY_GUARDED:
                self.assertEqual(
                    _sha(_read_bytes(os.path.join(DOCS, name))),
                    _sha(saved[name]), name)
        finally:
            # Restore only files whose bytes actually changed, so a
            # passing test does not bump mtimes on evidence files.
            for name, data in saved.items():
                path = os.path.join(DOCS, name)
                if data is None:
                    if os.path.exists(path):
                        os.remove(path)
                elif not os.path.exists(path) or \
                        _read_bytes(path) != data:
                    with open(path, "wb") as fh:
                        fh.write(data)

    def test_explicit_paths_leave_phase_records_untouched(self):
        # Explicit --record/--summary (the Phase 9 fix's contract):
        # the run writes only the given files.
        before = {name: _sha(_read_bytes(os.path.join(DOCS, name)))
                  for name in IDENTITY_GUARDED}
        with tempfile.TemporaryDirectory() as folder:
            record_path = os.path.join(folder, "r.json")
            summary_path = os.path.join(folder, "s.txt")
            with mock.patch.dict(consolidated._SUITE_RUNNERS,
                                 _stub_runners(), clear=True):
                rc = consolidated.main(argv=[
                    "--seed", "7", "--record", record_path,
                    "--summary", summary_path])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(record_path))
            self.assertTrue(os.path.exists(summary_path))
        for name in IDENTITY_GUARDED:
            self.assertEqual(
                _sha(_read_bytes(os.path.join(DOCS, name))),
                before[name], name)


if __name__ == "__main__":
    unittest.main()
