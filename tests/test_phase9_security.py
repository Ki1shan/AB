"""Phase 9 W4 §4.8: RLIMIT_CPU live test.

A busy-loop tool in the isolated subprocess path with a small CPU
limit must be terminated by the kernel's RLIMIT_CPU enforcement.
The observed outcome is recorded in ``docs/phase9-cpu-burner.json``;
this test re-runs the burner live and asserts the live outcome
matches the sidecar (no hardcoded exit code in the test — the
sidecar is the record, the live run is the check). It also asserts
the control leg: a fast tool under the same limits completes.

Honest attribution note (see the sidecar's ``cause_as_reported``):
the sandbox reports only the exit code; it does not label the cause.
The kill is attributed to RLIMIT_CPU by elimination — no parent-side
kill (``killed`` is False), no wall-time timeout (``timed_out`` is
False, timeout 30 s), and the tool was still burning CPU when it
died. With soft == hard limit the kernel sends SIGXCPU and SIGKILL
together and SIGKILL is what terminates the process, so the exit
code is -9 (SIGKILL), not -24 (SIGXCPU).
"""
import json
import os
import signal
import unittest

from ab.body.sandbox import IsolatedToolRunner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDECAR = os.path.join(ROOT, "docs", "phase9-cpu-burner.json")
CPU_SECONDS = 2


def _burner():
    x = 0
    while True:
        x += 1
        if x % 1000000 == 0:
            x = 0


def _fast():
    return 42


def _load_sidecar() -> dict:
    with open(SIDECAR, encoding="utf-8") as fh:
        return json.load(fh)


class TestRlimitCpuLive(unittest.TestCase):
    def test_burner_is_terminated_and_matches_sidecar(self):
        sidecar = _load_sidecar()
        self.assertEqual(sidecar["cpu_seconds"], CPU_SECONDS)
        runner = IsolatedToolRunner(cpu_seconds=CPU_SECONDS)
        res = runner.run(_burner, {}, timeout=30)
        # Live outcome matches the recorded outcome.
        self.assertFalse(res.ok)
        self.assertFalse(res.timed_out)
        self.assertFalse(res.killed)          # parent did not kill it
        self.assertTrue(res.observed_dead)
        self.assertEqual(res.exitcode, sidecar["exitcode"])
        self.assertEqual(res.exitcode, -signal.SIGKILL)
        self.assertIn("RLIMIT_CPU=2s", res.limits_applied)
        # The sidecar's two recorded runs agree with each other and
        # with this live run.
        for run in sidecar["runs"]:
            self.assertEqual(run["exitcode"], res.exitcode)
            self.assertEqual(run["observed_dead"], True)
            self.assertEqual(run["timed_out"], False)

    def test_fast_tool_under_same_limits_completes(self):
        sidecar = _load_sidecar()
        runner = IsolatedToolRunner(cpu_seconds=CPU_SECONDS)
        res = runner.run(_fast, {}, timeout=30)
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 42)
        self.assertEqual(res.exitcode, 0)
        self.assertEqual(sidecar["control_fast_tool"]["ok"], True)
        self.assertEqual(sidecar["control_fast_tool"]["exitcode"], 0)


if __name__ == "__main__":
    unittest.main()
