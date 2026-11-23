"""Phase 7, Worker 1: enforceable tool isolation (D7-1/D7-3, D7-6).

Covers ``ab/body/sandbox.py`` (``IsolatedToolRunner``) and the
isolated path of ``ab/body/tools.py`` (``ToolRegistry.call`` with
``isolated=True``, ``ToolRegistry.cancel()``).

The central claim -- a runaway tool is actually KILLED, not merely
timed out -- is demonstrated, not asserted: a tool that sleeps 30 s
and would write a marker file afterwards is run with a 0.5 s budget;
the test asserts the result carries ``timed_out=True`` AND
``killed=True`` with ``exitcode == -9`` (SIGKILL), the process was
observed dead, and the marker file is absent (the tool never got to
run past the kill).
"""
import os
import tempfile
import threading
import time
import unittest

from ab.body.sandbox import (
    MAX_TOOL_OUTPUT_BYTES,
    RLIMIT_AS_BYTES,
    RLIMIT_CPU_SECONDS,
    IsolatedToolRunner,
)
from ab.body.tools import Tool, ToolRegistry, ToolSpec

# The exact bounds the design pins (D7-3/D7-6); the runner must record
# these literals in every isolated result and audit entry.
EXPECTED_LIMITS = ["RLIMIT_CPU=60s", "RLIMIT_AS=512MiB"]


class TestRunnerAvailability(unittest.TestCase):
    def test_fork_and_resource_available_on_linux(self):
        r = IsolatedToolRunner()
        self.assertTrue(r.available, r.unavailable_reason)
        self.assertIsNone(r.unavailable_reason)

    def test_bounds_are_the_design_values(self):
        self.assertEqual(MAX_TOOL_OUTPUT_BYTES, 64 * 1024)
        self.assertEqual(RLIMIT_CPU_SECONDS, 60)
        self.assertEqual(RLIMIT_AS_BYTES, 512 * 1024 * 1024)

    def test_limits_applied_records_the_exact_bounds(self):
        r = IsolatedToolRunner()
        self.assertEqual(r.limits_applied, EXPECTED_LIMITS)
        res = r.run(lambda: 1, {})
        self.assertEqual(res.limits_applied, EXPECTED_LIMITS)


class TestForkSemantics(unittest.TestCase):
    def test_lambda_tool_runs_in_child(self):
        r = IsolatedToolRunner()
        res = r.run(lambda x: x + 1, {"x": 41})
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 42)
        self.assertTrue(res.isolated)
        self.assertTrue(res.observed_dead)
        self.assertEqual(res.exitcode, 0)

    def test_closure_tool_runs_in_child(self):
        base = 40

        def add_two(x):
            return base + x

        r = IsolatedToolRunner()
        res = r.run(add_two, {"x": 2})
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 42)

    def test_raising_tool_returns_typed_error_not_a_crash(self):
        def boom():
            raise ValueError("kaboom")

        r = IsolatedToolRunner()
        res = r.run(boom, {})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "ValueError: kaboom")
        self.assertFalse(res.timed_out)
        self.assertFalse(res.killed)          # it died by itself; we did not kill it
        self.assertTrue(res.observed_dead)
        self.assertEqual(res.exitcode, 0)     # the child handled the error and exited cleanly


class TestObservedKill(unittest.TestCase):
    def test_sleeping_tool_is_killed_and_observed_dead(self):
        """The demonstrated kill (design experiment 3): a tool that
        sleeps 30 s and would write a marker file if it kept running,
        run with a 0.5 s budget."""
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")

            def sleeper():
                time.sleep(30)
                with open(marker, "w", encoding="utf-8") as fh:
                    fh.write("kept running")
                return "done"

            r = IsolatedToolRunner()
            started = time.monotonic()
            res = r.run(sleeper, {}, timeout=0.5)
            elapsed = time.monotonic() - started
            self.assertFalse(res.ok)
            self.assertTrue(res.timed_out)
            self.assertIn("timeout", res.error)
            self.assertTrue(res.killed, "kill must be observed, not assumed")
            self.assertTrue(res.observed_dead)
            self.assertEqual(res.exitcode, -9)       # SIGKILL to the process group
            self.assertLess(elapsed, 5.0)            # returned promptly, not after 30 s
            self.assertFalse(os.path.exists(marker),
                             "marker absent: the tool never ran past the kill")
            self.assertIsNone(r._proc)               # no child left behind

    def test_registry_isolated_kill_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")

            def sleeper():
                time.sleep(30)
                with open(marker, "w", encoding="utf-8") as fh:
                    fh.write("kept running")

            reg = ToolRegistry()
            reg.register(Tool("sleeper", sleeper, "compute"))
            started = time.monotonic()
            res = reg.call("sleeper", {}, {"compute"}, timeout=0.5, isolated=True)
            elapsed = time.monotonic() - started
            self.assertFalse(res.ok)
            self.assertTrue(res.timed_out)
            self.assertTrue(res.killed)
            self.assertTrue(res.isolated)
            self.assertEqual(list(res.limits_applied), EXPECTED_LIMITS)
            self.assertLess(elapsed, 5.0)
            self.assertFalse(os.path.exists(marker))
            entry = reg.audit[-1]
            self.assertTrue(entry["timed_out"])
            self.assertTrue(entry["killed"])
            self.assertTrue(entry["isolated"])
            self.assertEqual(entry["limits_applied"], EXPECTED_LIMITS)
            self.assertIsNone(entry["isolation_unavailable"])

    def test_registry_default_path_is_still_the_thread_path(self):
        """Default ``isolated=False`` preserves the legacy contract:
        a timeout returns quickly, the (unkillable) thread is NOT
        reported killed, and the audit says so."""
        reg = ToolRegistry()
        reg.register(Tool("slow", lambda: time.sleep(0.5), "compute"))
        res = reg.call("slow", {}, {"compute"}, timeout=0.1)
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertFalse(res.killed)
        self.assertFalse(res.isolated)
        entry = reg.audit[-1]
        self.assertFalse(entry["killed"])
        self.assertFalse(entry["isolated"])
        self.assertEqual(entry["limits_applied"], [])

    def test_fast_tool_via_registry_isolated_path(self):
        reg = ToolRegistry()
        res = reg.call("calc", {"expression": "6*7"}, {"compute"}, isolated=True)
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 42)
        self.assertTrue(res.isolated)
        self.assertFalse(res.killed)
        self.assertEqual(reg.audit[-1]["isolated"], True)


class TestCancel(unittest.TestCase):
    def test_cancel_kills_running_isolated_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")

            def sleeper():
                time.sleep(30)
                with open(marker, "w", encoding="utf-8") as fh:
                    fh.write("kept running")

            reg = ToolRegistry()
            reg.register(Tool("sleeper", sleeper, "compute"))
            results = []

            def _call():
                results.append(reg.call("sleeper", {}, {"compute"},
                                        timeout=30, isolated=True))

            worker = threading.Thread(target=_call)
            worker.start()
            # Wait until the child is actually running, then cancel.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                proc = reg._runner._proc
                if proc is not None and proc.is_alive():
                    break
                time.sleep(0.02)
            self.assertTrue(reg.cancel(), "cancel() must observe the child dead")
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive(), "the isolated call must return after cancel")
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertFalse(res.ok)
            self.assertTrue(res.cancelled)
            self.assertTrue(res.killed)
            self.assertFalse(res.timed_out)
            self.assertFalse(os.path.exists(marker))
            # The registry is still usable after a kill.
            self.assertEqual(reg.call("calc", {"expression": "1+1"},
                                      {"compute"}).output, 2)

    def test_cancel_with_nothing_running_returns_false(self):
        reg = ToolRegistry()
        self.assertFalse(reg.cancel())


class TestChildCrashAndOutput(unittest.TestCase):
    def test_child_exiting_by_itself_reports_exit_code(self):
        r = IsolatedToolRunner()
        res = r.run(lambda: os._exit(3), {})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "tool process exited with code 3")
        self.assertEqual(res.exitcode, 3)
        self.assertTrue(res.observed_dead)
        self.assertFalse(res.killed)     # it exited by itself; we did not kill it

    def test_oversized_output_is_refused_not_trusted(self):
        r = IsolatedToolRunner()
        res = r.run(lambda: "x" * 70_000, {})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "output_too_large")

    def test_unpicklable_output_becomes_its_repr(self):
        # A lambda return value cannot be pickled through the pipe;
        # the child sends its repr instead of crashing the protocol.
        r = IsolatedToolRunner()
        res = r.run(lambda: (lambda: 1), {})
        self.assertTrue(res.ok)
        self.assertIsInstance(res.output, str)
        self.assertIn("function", res.output)


class TestResourceLimitsApplied(unittest.TestCase):
    def test_cpu_and_address_space_limits_are_recorded(self):
        r = IsolatedToolRunner()
        res = r.run(lambda: 1, {})
        self.assertEqual(res.limits_applied, EXPECTED_LIMITS)
        self.assertIn("RLIMIT_CPU=60s", res.limits_applied)
        self.assertIn("RLIMIT_AS=512MiB", res.limits_applied)

    def test_address_space_limit_is_actually_enforced(self):
        """A 1 GiB allocation inside the 512 MiB RLIMIT_AS child fails
        with MemoryError instead of succeeding -- the limit is real,
        not just a recorded string."""
        r = IsolatedToolRunner()
        res = r.run(lambda: bytearray(1024 ** 3), {})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("MemoryError"), res.error)
        self.assertTrue(res.observed_dead)


class TestBoundMethodTools(unittest.TestCase):
    """A bound-method tool mutates its object's in-process state, so
    forking it would silently discard its effects while reporting
    success (the Brain's own ``research`` tool is the only such tool
    in the tree: it admits claims to LTM and writes state). With
    ``isolated=True`` requested, such a tool runs on the thread path
    instead, and the audit record says so explicitly."""

    def test_bound_method_runs_in_process_with_effects_visible(self):
        class Counter:
            def __init__(self):
                self.n = 0

            def inc(self, by):
                self.n += by
                return self.n

        counter = Counter()
        reg = ToolRegistry()
        reg.register(ToolSpec(
            "inc", [{"name": "by", "type": "int", "required": True}],
            return_type="int", required_grant="compute"),
            counter.inc)
        res = reg.call("inc", {"by": 3}, {"compute"}, isolated=True)
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 3)
        self.assertEqual(counter.n, 3,
                         "the bound method's mutation must be visible")
        self.assertFalse(res.isolated)
        entry = reg.audit[-1]
        self.assertFalse(entry["isolated"])
        self.assertIsNotNone(entry["isolation_skipped"])
        self.assertIn("bound method", entry["isolation_skipped"])

    def test_plain_function_still_runs_isolated(self):
        reg = ToolRegistry()
        res = reg.call("calc", {"expression": "6*7"}, {"compute"}, isolated=True)
        self.assertTrue(res.isolated)
        self.assertIsNone(reg.audit[-1]["isolation_skipped"])

    def test_brain_research_goal_completes_through_isolated_path(self):
        """End-to-end regression: Worker 2's brain path passes
        ``isolated=True`` for every tool call; the research tool's
        LTM/state effects must still land, or the goal fails.

        Phase 11 (W5) fixture update (§9 precedent, same intent, new
        semantics): the research tool is now an isolated
        ``functools.partial`` over ``research_fetch`` — the child
        materializes its provider from the spec, so the provider
        must survive a spec round-trip. A ``WebFetcher`` with a fake
        transport does NOT (the transport is a closure, never in the
        spec), so the fixture uses a ``MemoryKnowledgeProvider``
        carrying the same two pages. The second source's sentence is
        also reworded ("capital city") per the W1 mirror rule: the
        identical sentence from both domains no longer corroborates.
        The audit now asserts the research calls ran ISOLATED
        (``isolated=True``, ``isolation_skipped=None``)."""
        from ab.body.providers import MemoryKnowledgeProvider
        from ab.body.research import Researcher
        from ab.brain import Brain

        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
                 "https://b.org/p": "<p>Paris is the capital city of France.</p>"}

        b = Brain(researcher=Researcher(MemoryKnowledgeProvider(pages), None),
                  granted={"compute", "network"})
        b.add_research_goal("paris", "capital_of",
                            ["https://a.org/p", "https://b.org/p"])
        self.assertEqual(b.run(None, 30), "done")
        self.assertEqual(b.goals.all()[0].status, "done")
        self.assertEqual(b.ask("paris", "capital_of").best[2], "verified")
        # The research calls ran in the forked child (isolated tool),
        # recorded explicitly -- not on the thread path, not silently lost.
        research_entries = [e for e in b.tools.audit if e["tool"] == "research"]
        self.assertTrue(research_entries)
        self.assertTrue(all(e["isolated"] for e in research_entries))
        self.assertTrue(all(e["isolation_skipped"] is None for e in research_entries))


if __name__ == "__main__":
    unittest.main()
