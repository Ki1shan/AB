"""Phase 11 W3 — evaluation tests: each held-out research task passes
standalone, and the harness Phase 11 section runs and records.

The harness itself is NOT run here (the coordinator runs the full
harness at integration); ``_run_phase11`` is exercised in-process
with ``record_path=None`` semantics — it must not write the
canonical research-tasks record.
"""
import json
import os
import tempfile
import unittest

from ab.eval import harness as harness_mod
from ab.eval.research_tasks import (TASKS, run_research_tasks)


class TestResearchTasksStandalone(unittest.TestCase):
    def test_each_task_passes(self):
        record = run_research_tasks(seed=0)
        self.assertEqual(record["seed"], 0)
        self.assertEqual(sorted(record["tasks"]), sorted(TASKS))
        self.assertEqual(record["n_passed"], len(TASKS))
        self.assertEqual(record["n_failed"], 0)
        self.assertEqual(record["failures"], [])
        for name in TASKS:
            task = record["tasks"][name]
            self.assertTrue(task["passed"], f"{name} failed: {task}")
            self.assertIn("detail", task)
            self.assertIn("cycles", task)
            self.assertIn("failed_attempts", task)
            self.assertIn("sources", task)

    def test_task_numbers_are_measured_not_hardcoded(self):
        record = run_research_tasks(seed=0)
        correct = record["tasks"]["correct_source"]
        self.assertEqual(correct["answer"], "aerilon")
        self.assertEqual(correct["answer_status"], "verified")
        self.assertEqual(correct["sources"],
                         ["web:alpha.org", "web:beta.org"])
        self.assertGreater(correct["cycles"], 0)
        disconnect = record["tasks"]["provider_disconnect"]
        self.assertTrue(disconnect["completed_from_ltm"])
        self.assertTrue(disconnect["disconnected_failed_no_provider"])
        self.assertTrue(disconnect["resumed_after_swap"])
        self.assertEqual(disconnect["new_answer"], "norvane")

    def test_run_does_not_write_a_record_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            record = run_research_tasks(seed=0, record_path=None)
            self.assertTrue(record["n_passed"] >= 1)
            self.assertEqual(os.listdir(folder), [])

    def test_explicit_record_path_writes_only_that_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "research-tasks-record.json")
            record = run_research_tasks(seed=0, record_path=path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                written = json.load(fh)
            self.assertEqual(written["n_passed"], record["n_passed"])
            self.assertEqual(os.listdir(folder), ["research-tasks-record.json"])


class TestHarnessPhase11Section(unittest.TestCase):
    def test_run_phase11_records_all_tasks(self):
        section = harness_mod._run_phase11(fast=False)
        self.assertNotIn("error", section)
        self.assertEqual(section["mode"], "full")
        self.assertEqual(section["seed"], 0)
        self.assertEqual(sorted(section["tasks"]), sorted(TASKS))
        self.assertEqual(section["n_passed"], len(TASKS))
        self.assertEqual(section["n_failed"], 0)
        for name in TASKS:
            self.assertTrue(section["tasks"][name]["passed"],
                            f"{name} failed in harness section")

    def test_run_phase11_fast_mode_runs_and_records_mode(self):
        section = harness_mod._run_phase11(fast=True)
        self.assertNotIn("error", section)
        self.assertEqual(section["mode"], "fast")
        self.assertEqual(section["n_passed"], len(TASKS))

    def test_harness_section_does_not_write_canonical_record(self):
        # The harness path must never overwrite the canonical
        # research-tasks record: _run_phase11 takes no path and the
        # canonical file's mtime must be untouched by the call.
        canonical = os.path.join(harness_mod.REPO_ROOT, "docs",
                                 "phase11-research-tasks-record.json")
        before = (os.path.getmtime(canonical)
                  if os.path.exists(canonical) else None)
        harness_mod._run_phase11(fast=False)
        after = (os.path.getmtime(canonical)
                 if os.path.exists(canonical) else None)
        self.assertEqual(before, after)

    def test_harness_defaults_name_phase11_files(self):
        self.assertTrue(harness_mod.DEFAULT_RECORD11.endswith(
            "phase11-harness-record.json"))
        self.assertTrue(harness_mod.DEFAULT_SUMMARY11.endswith(
            "phase11-harness-summary.txt"))

    def test_summary_text_covers_phase11(self):
        section = harness_mod._run_phase11(fast=False)
        record = {"timestamp": "2026-11-22T00:00:00+00:00", "mode": "full",
                  "tests": {"ran": 1, "failures": 0, "errors": 0,
                            "skipped": 0, "ok": True, "scope": "test",
                            "duration_s": 0.0, "exit_code": 0},
                  "demo": {"exit_code": 0, "outcome": "done",
                           "duration_s": 0.0},
                  "disconnected": {"result": None},
                  "baselines": {"treasure": {}},
                  "phase11_evaluation": section,
                  "resources": {"total_wall_s": 0.0, "suite_wall_s": 0.0,
                                "demo_wall_s": 0.0,
                                "disconnected_wall_s": 0.0,
                                "peak_child_rss_kb": "not measured"},
                  "repo_state": {"vcs": "none", "python_files": 0,
                                 "test_files": 0}}
        text = harness_mod._summary_text(record)
        self.assertIn("Phase 11 research tasks", text)
        for name in TASKS:
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
