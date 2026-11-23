"""Phase 7, Worker 1: security tests (design experiments 5/8/12, security half).

Covers, at the registry / sandbox / fetcher layer (Brain-level
containment and ``Brain(fs_scope=...)`` are Worker 2's job; the
fs-scope test below exercises ``ScopedFilesystem`` directly and
exercises the Brain API only if it exists yet):

(a) a runaway tool is actually killed -- see
    ``tests/test_phase7_isolation.py::TestObservedKill`` for the
    demonstrated kill; here the killed tool's marker is absent AND
    the registry recovers and keeps working afterwards.
(b) tool permission enforcement: a tool without its required grant is
    refused BEFORE execution (a side-effect marker proves it never ran).
(c) filesystem scope escapes refused: ``../`` escape, absolute path
    outside the scope, symlink pointing outside the scope.
(d) network allow-list: disallowed host refused, including via a
    redirect to a disallowed host (local test server shape reused
    from ``tests/test_phase1_security.py``).
(e) input validation (malformed args rejected before execution, tool
    never runs) and output validation (oversized output bounded /
    truncated, malformed output not trusted).
(f) secret redaction end to end at save level: a secret through a
    tool call never reaches saved events (Phase 3 shape).
(g) failure containment: a tool that raises, returns garbage, or
    hangs cannot corrupt brain state or take down the cycle.
(h) resource limits actually applied: ``limits_applied`` recorded and
    the address-space limit enforced (see also the isolation tests).
"""
import http.server
import inspect
import json
import os
import tempfile
import threading
import unittest

from ab.body.sandbox import ScopedFilesystem
from ab.body.tools import Tool, ToolRegistry, ToolSpec
from ab.body.web import FetchError, WebFetcher
from ab.brain import Brain

FAKE_SECRET = "sk-test-SECRET-VALUE-12345"


def _transport(pages):
    """Fake transport keyed by URL (Phase 1 shape). Values may be
    3-tuples or 4-tuples (status, ctype, body, headers)."""
    calls = []

    def t(url, timeout, max_bytes):
        calls.append(url)
        return pages[url]

    t.calls = calls
    return t


class TestKillRecovery(unittest.TestCase):
    def test_killed_tool_leaves_no_marker_and_registry_recovers(self):
        # (a) from the security side: the kill is demonstrated in
        # test_phase7_isolation.py; here the registry must keep
        # working afterwards (safe recovery after a kill).
        import time

        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "marker.txt")

            def sleeper():
                time.sleep(30)
                with open(marker, "w", encoding="utf-8") as fh:
                    fh.write("kept running")

            reg = ToolRegistry()
            reg.register(Tool("sleeper", sleeper, "compute"))
            res = reg.call("sleeper", {}, {"compute"}, timeout=0.5, isolated=True)
            self.assertFalse(res.ok)
            self.assertTrue(res.timed_out)
            self.assertTrue(res.killed)
            self.assertFalse(os.path.exists(marker))
            # Recovery: the next call works and the audit is intact.
            self.assertEqual(reg.call("calc", {"expression": "2+2"},
                                      {"compute"}).output, 4)
            self.assertTrue(all("tool" in e for e in reg.audit))


class TestToolPermissionEnforcement(unittest.TestCase):
    def test_tool_without_grant_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "ran.txt")

            def spy():
                with open(marker, "w", encoding="utf-8") as fh:
                    fh.write("ran")
                return "ran"

            for isolated in (False, True):
                reg = ToolRegistry()
                reg.register(ToolSpec(
                    "net_tool", [], return_type="str", required_grant="network"),
                    spy)
                res = reg.call("net_tool", {}, {"compute"}, isolated=isolated)
                self.assertFalse(res.ok, msg=f"isolated={isolated}")
                self.assertIn("permission", res.error)
                self.assertFalse(os.path.exists(marker),
                                 "marker absent: the tool never ran")
                self.assertIn("network", reg.audit[-1]["error"])

    def test_tool_with_grant_runs(self):
        reg = ToolRegistry()
        reg.register(ToolSpec("net_tool", [], return_type="str",
                              required_grant="network"), lambda: "ok")
        self.assertEqual(reg.call("net_tool", {}, {"network"}).output, "ok")


class TestFilesystemScope(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = os.path.join(self._tmp.name, "scope")
        self.outside = os.path.join(self._tmp.name, "outside")
        os.makedirs(self.root)
        os.makedirs(self.outside)
        with open(os.path.join(self.outside, "secret.txt"), "w", encoding="utf-8") as fh:
            fh.write("outside-secret")
        self.fs = ScopedFilesystem(self.root)

    def test_read_write_inside_scope_work(self):
        self.assertTrue(self.fs.write("notes/hello.txt", "hello")["ok"])
        self.assertTrue(self.fs.write("top.txt", "top")["ok"])
        self.assertEqual(self.fs.read("notes/hello.txt"),
                         {"ok": True, "content": "hello"})
        self.assertEqual(self.fs.read("top.txt")["content"], "top")

    def test_dotdot_escape_refused(self):
        res = self.fs.read("../outside/secret.txt")
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"].startswith("scope_denied:"), res["error"])
        res = self.fs.write("../outside/pwned.txt", "x")
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"].startswith("scope_denied:"), res["error"])
        self.assertFalse(os.path.exists(os.path.join(self.outside, "pwned.txt")))

    def test_absolute_path_outside_scope_refused(self):
        target = os.path.join(self.outside, "secret.txt")
        res = self.fs.read(target)
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"].startswith("scope_denied:"), res["error"])
        self.assertNotIn("outside-secret", json.dumps(res))

    def test_symlink_pointing_outside_scope_refused(self):
        link = os.path.join(self.root, "escape-link")
        os.symlink(self.outside, link)
        res = self.fs.read("escape-link/secret.txt")
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"].startswith("scope_denied:"), res["error"])
        res = self.fs.write("escape-link/pwned.txt", "x")
        self.assertFalse(res["ok"])
        self.assertTrue(res["error"].startswith("scope_denied:"), res["error"])
        self.assertFalse(os.path.exists(os.path.join(self.outside, "pwned.txt")))

    def test_symlink_inside_scope_still_works(self):
        self.fs.write("real.txt", "data")
        os.symlink(os.path.join(self.root, "real.txt"),
                   os.path.join(self.root, "alias.txt"))
        self.assertEqual(self.fs.read("alias.txt")["content"], "data")

    def test_scope_denied_events_recorded(self):
        self.fs.read("../outside/secret.txt")
        self.fs.read(os.path.join(self.outside, "secret.txt"))
        self.assertEqual(len(self.fs.events), 2)
        self.assertTrue(all(e["kind"] == "scope_denied" for e in self.fs.events))

    def test_brain_fs_scope_if_available(self):
        """Worker 2's ``Brain(fs_scope=...)`` API (design §1.3): file
        tools exist only with a scope. If the API does not exist yet,
        this skips -- the class-level tests above are the coverage."""
        params = inspect.signature(Brain.__init__).parameters
        if "fs_scope" not in params:
            self.skipTest("Brain(fs_scope=...) not yet implemented "
                          "(Worker 2); ScopedFilesystem tested directly above")
        b = Brain(fs_scope=self.root)
        self.assertIn("read_file", b.tools.names())
        self.assertIn("write_file", b.tools.names())
        res = b.tools.call("write_file",
                           {"path": "a.txt", "data": "hi"},
                           {"compute", "filesystem"})
        self.assertTrue(res.ok, res.error)
        self.assertEqual(b.tools.call("read_file", {"path": "a.txt"},
                                      {"compute", "filesystem"}).output, "hi")
        denied = b.tools.call("read_file", {"path": "../outside/secret.txt"},
                              {"compute", "filesystem"})
        self.assertFalse(denied.ok)
        self.assertIn("scope_denied", denied.error)


class TestNetworkAllowList(unittest.TestCase):
    def test_direct_private_ip_refused_by_real_server(self):
        """A real local HTTP server is running, so the fetch would
        succeed if it were attempted -- the refusal happens before
        any connection (zero hits on the server)."""
        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                body = b"should-not-be-reached"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            f = WebFetcher(["good.org"])
            with self.assertRaises(FetchError):
                f.fetch(f"http://127.0.0.1:{server.server_address[1]}/secret")
            self.assertEqual(hits, [])
        finally:
            server.shutdown()
            server.server_close()

    def test_redirect_to_private_ip_refused(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "",
                                   {"Location": "http://127.0.0.1/secret"}),
            "http://127.0.0.1/secret": (200, "text/html", "secret"),
        })
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertEqual(t.calls, ["https://good.org/a"])

    def test_redirect_to_disallowed_host_refused(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "",
                                   {"Location": "https://evil.com/x"}),
            "https://evil.com/x": (200, "text/html", "evil"),
        })
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertEqual(t.calls, ["https://good.org/a"])


class TestInputAndOutputValidation(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        self.reg.register(ToolSpec(
            "add", [{"name": "a", "type": "int", "required": True},
                    {"name": "b", "type": "int", "required": True}],
            return_type="int", required_grant="compute"),
            lambda a, b: a + b)

    def test_malformed_args_rejected_before_execution(self):
        seen = []
        self.reg.register(ToolSpec(
            "spy", [{"name": "x", "type": "int", "required": True}]),
            lambda x: seen.append(x))
        for isolated in (False, True):
            for bad_args in ({"x": "not an int"}, {}, {"x": 1, "y": 2}):
                res = self.reg.call("spy", bad_args, {"compute"}, isolated=isolated)
                self.assertFalse(res.ok, msg=f"isolated={isolated} args={bad_args}")
                self.assertTrue(res.error.startswith("arg_error:"), res.error)
        self.assertEqual(seen, [], "the tool never ran on any malformed call")

    def test_oversized_string_output_truncated_with_flag(self):
        self.reg.register(Tool("big", lambda: "y" * 70_000, "compute"))
        res = self.reg.call("big", {}, {"compute"})
        self.assertTrue(res.ok)
        self.assertLessEqual(len(res.output.encode("utf-8")), 64 * 1024)
        self.assertTrue(res.output_truncated)
        entry = self.reg.audit[-1]
        self.assertTrue(entry["output_truncated"])
        self.assertLessEqual(len(entry["result"].encode("utf-8")), 64 * 1024)

    def test_small_string_output_not_truncated(self):
        res = self.reg.call("add", {"a": 1, "b": 2}, {"compute"})
        self.assertFalse(res.output_truncated)
        self.assertFalse(self.reg.audit[-1]["output_truncated"])

    def test_non_jsonable_output_stored_as_truncated_repr(self):
        self.reg.register(Tool("weird", lambda: object(), "compute"))
        res = self.reg.call("weird", {}, {"compute"})
        self.assertTrue(res.ok)
        self.assertIsInstance(res.output, str)
        self.assertIn("object object", res.output)
        self.assertIsInstance(self.reg.audit[-1]["result"], str)

    def test_jsonable_dict_output_kept_as_dict(self):
        self.reg.register(Tool("d", lambda: {"k": [1, 2]}, "compute"))
        res = self.reg.call("d", {}, {"compute"})
        self.assertEqual(res.output, {"k": [1, 2]})
        self.assertEqual(self.reg.audit[-1]["result"], {"k": [1, 2]})


class TestSecretRedactionAtSaveLevel(unittest.TestCase):
    def test_secret_through_tool_call_never_reaches_saved_events(self):
        """Phase 3 shape: the secret travels through a tool call (as a
        secret-marked argument and as a tool result), the brain records
        an event mentioning it, and the save is scanned end to end --
        no saved file may contain the secret value."""
        b = Brain()
        b.tools.register(ToolSpec(
            "lookup", [{"name": "api_key", "type": "str", "required": True,
                        "secret": True},
                       {"name": "query", "type": "str", "required": True}]),
            lambda api_key, query: {"api_key": api_key, "echo": query})
        res = b.tools.call("lookup", {"api_key": FAKE_SECRET, "query": "hello"},
                           {"compute"})
        self.assertTrue(res.ok)
        self.assertEqual(res.output["api_key"], FAKE_SECRET)  # caller still gets it
        self.assertNotIn(FAKE_SECRET, json.dumps(b.tools.audit))
        # The event path redacts too (Brain._event -> redact_secrets).
        b._event("tool", f"lookup returned api_key={FAKE_SECRET}")
        with tempfile.TemporaryDirectory() as folder:
            b.save(folder)
            for name in sorted(os.listdir(folder)):
                with open(os.path.join(folder, name), encoding="utf-8") as fh:
                    blob = fh.read()
                self.assertNotIn(FAKE_SECRET, blob, msg=name)
            with open(os.path.join(folder, "events.json"), encoding="utf-8") as fh:
                events_blob = fh.read()
            self.assertIn("[REDACTED]", events_blob)


class TestFailureContainment(unittest.TestCase):
    def test_raising_tool_cannot_take_down_registry(self):
        reg = ToolRegistry()
        reg.register(Tool("boom", lambda: (_ for _ in ()).throw(ValueError("kaboom")),
                          "compute"))
        res = reg.call("boom", {}, {"compute"})
        self.assertFalse(res.ok)
        self.assertIn("ValueError", res.error)
        self.assertEqual(reg.call("calc", {"expression": "3*3"},
                                  {"compute"}).output, 9)

    def test_garbage_returning_tool_cannot_corrupt_registry(self):
        reg = ToolRegistry()
        reg.register(Tool("garbage", lambda: object(), "compute"))
        res = reg.call("garbage", {}, {"compute"})
        self.assertTrue(res.ok)
        self.assertIsInstance(res.output, str)   # stored as repr, never trusted
        self.assertEqual(reg.call("calc", {"expression": "3*3"},
                                  {"compute"}).output, 9)

    def test_hanging_tool_in_thread_path_cannot_stall_caller(self):
        import time

        reg = ToolRegistry()
        reg.register(Tool("slow", lambda: time.sleep(0.5), "compute"))
        started = time.monotonic()
        res = reg.call("slow", {}, {"compute"}, timeout=0.1)
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(reg.call("calc", {"expression": "3*3"},
                                  {"compute"}).output, 9)

    def test_hanging_tool_in_isolated_path_is_killed_and_registry_survives(self):
        import time

        reg = ToolRegistry()
        reg.register(Tool("slow", lambda: time.sleep(30), "compute"))
        res = reg.call("slow", {}, {"compute"}, timeout=0.5, isolated=True)
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertTrue(res.killed)
        self.assertEqual(reg.call("calc", {"expression": "3*3"},
                                  {"compute"}).output, 9)


class TestResourceLimitsRecorded(unittest.TestCase):
    def test_isolated_audit_records_cpu_and_address_space_limits(self):
        reg = ToolRegistry()
        res = reg.call("calc", {"expression": "1+1"}, {"compute"}, isolated=True)
        self.assertTrue(res.ok)
        self.assertEqual(list(res.limits_applied),
                         ["RLIMIT_CPU=60s", "RLIMIT_AS=512MiB"])
        self.assertEqual(reg.audit[-1]["limits_applied"],
                         ["RLIMIT_CPU=60s", "RLIMIT_AS=512MiB"])

    def test_address_space_limit_enforced_via_registry(self):
        reg = ToolRegistry()
        reg.register(Tool("hog", lambda: bytearray(1024 ** 3), "compute"))
        res = reg.call("hog", {}, {"compute"}, isolated=True)
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("MemoryError"), res.error)


if __name__ == "__main__":
    unittest.main()
