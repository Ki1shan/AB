"""Phase 1, Workstream 7: immediate security remediation.

Acceptance:
- Web: literal private IPs rejected; private DNS results rejected
  (including mixed results); redirects to private destinations rejected;
  redirects outside the allow-list rejected; redirect loops capped;
  a pinned connection reaches the validated IP while speaking for the
  original hostname (Host header).
- Tools: a hanging tool returns a timeout error within budget instead of
  stalling the caller; safe_calc caps length and node count.
- Environment: actions are denied without the "environment" grant (and
  never executed), allowed with it, and allowed by the default grants.
"""
import http.server
import threading
import time
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.body.tools import Tool, ToolRegistry, safe_calc
from ab.body.web import FetchError, WebFetcher, _PinnedConnection


def _transport(pages):
    """Fake transport keyed by URL. Values may be 3-tuples or 4-tuples
    (status, ctype, body, headers)."""
    calls = []

    def t(url, timeout, max_bytes):
        calls.append(url)
        return pages[url]

    t.calls = calls
    return t


class TestSsrfDns(unittest.TestCase):
    def test_literal_private_ips_rejected(self):
        f = WebFetcher(["good.org"], transport=_transport({}))
        for bad in ["http://127.0.0.1/a", "http://10.0.0.5/a", "http://169.254.169.254/x",
                    "http://[::1]/a", "http://0.0.0.0/a", "http://192.168.1.1/a"]:
            with self.assertRaises(FetchError, msg=bad):
                f.fetch(bad)

    def test_private_dns_result_rejected_and_transport_never_called(self):
        t = _transport({"https://good.org/a": (200, "text/html", "<p>hi</p>")})
        f = WebFetcher(["good.org"], transport=t, resolver=lambda h, p: ["10.0.0.1"])
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertEqual(t.calls, [])

    def test_mixed_dns_result_rejected(self):
        f = WebFetcher(["good.org"], transport=_transport({}),
                       resolver=lambda h, p: ["93.184.216.34", "10.0.0.1"])
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")

    def test_public_dns_result_allowed(self):
        t = _transport({"https://good.org/a": (200, "text/html", "<p>hi</p>")})
        f = WebFetcher(["good.org"], transport=t, resolver=lambda h, p: ["93.184.216.34"])
        self.assertEqual(f.fetch("https://good.org/a").text, "hi")


class TestSsrfRedirects(unittest.TestCase):
    def test_redirect_to_private_ip_rejected(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "", {"Location": "http://127.0.0.1/secret"}),
            "http://127.0.0.1/secret": (200, "text/html", "secret"),
        })
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertEqual(t.calls, ["https://good.org/a"])

    def test_redirect_to_allowlisted_host_resolving_privately_rejected(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "",
                                   {"Location": "https://internal.good.org/secret"}),
            "https://internal.good.org/secret": (200, "text/html", "secret"),
        })

        def resolver(host, port):
            return ["93.184.216.34"] if host == "good.org" else ["10.0.0.9"]

        f = WebFetcher(["good.org"], transport=t, resolver=resolver)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertEqual(t.calls, ["https://good.org/a"])

    def test_redirect_outside_allowlist_rejected(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "", {"Location": "https://evil.com/x"}),
        })
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")

    def test_redirect_loop_is_capped(self):
        t = _transport({
            "https://good.org/a": (302, "text/html", "", {"Location": "https://good.org/b"}),
            "https://good.org/b": (302, "text/html", "", {"Location": "https://good.org/a"}),
        })
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")
        self.assertLessEqual(len(t.calls), 4)   # initial + at most 3 redirects

    def test_redirect_without_location_is_an_error(self):
        t = _transport({"https://good.org/a": (302, "text/html", "")})
        f = WebFetcher(["good.org"], transport=t)
        with self.assertRaises(FetchError):
            f.fetch("https://good.org/a")


class TestPinnedConnection(unittest.TestCase):
    def test_connects_to_validated_ip_speaking_for_hostname(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["host"] = self.headers.get("Host")
                body = b"pinned-ok"
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
            conn = _PinnedConnection("example.test", server.server_address[1],
                                     "127.0.0.1", timeout=5)
            conn.request("GET", "/")
            resp = conn.getresponse()
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.read(), b"pinned-ok")
            conn.close()
            # http.client appends the non-default port; the point is the
            # header names the original hostname, not the validated IP.
            self.assertEqual(seen["host"].split(":")[0], "example.test")
        finally:
            server.shutdown()
            server.server_close()


class TestToolTimeout(unittest.TestCase):
    def test_hanging_tool_times_out_without_stalling_caller(self):
        r = ToolRegistry()
        r.register(Tool("slow", lambda: time.sleep(10), "compute"))
        started = time.monotonic()
        res = r.call("slow", {}, {"compute"}, timeout=0.1)
        elapsed = time.monotonic() - started
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertIn("timeout", res.error)
        self.assertLess(elapsed, 2.0)
        self.assertTrue(r.audit[-1]["timed_out"])

    def test_fast_tool_still_works_with_default_timeout(self):
        r = ToolRegistry()
        self.assertEqual(r.call("calc", {"expression": "6*7"}, {"compute"}).output, 42)

    def test_safe_calc_caps(self):
        with self.assertRaises(ValueError):
            safe_calc("1" * 201)
        with self.assertRaises(ValueError):
            safe_calc("+".join(["1"] * 60) + "+1")   # far over 100 AST nodes


class TestEnvironmentPermissions(unittest.TestCase):
    def test_env_action_denied_without_environment_grant_and_never_executes(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.run(env, 60)
        self.assertFalse(env.s["has_treasure"])
        self.assertEqual(env.log, [])   # env.act was never called
        self.assertTrue(any(e["kind"] == "deny" for e in b.events))

    def test_env_action_denied_with_empty_grants(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted=set())
        b.add_goal("get the treasure", {"has_treasure": True})
        b.run(env, 60)
        self.assertEqual(env.log, [])

    def test_env_action_allowed_with_explicit_grant(self):
        # Phase 2 decision (resolves Phase 1 blocker #3 / ADR-2): the
        # default grant set is compute-only, so the "allowed" case must
        # pass the environment grant explicitly. The assertion itself
        # (treasure obtained with the grant) is unchanged, not weakened.
        for granted in ({"compute", "environment"},
                        {"compute", "environment", "network"}):
            env = RoomWorld()
            b = Brain(room_world_actions(), granted=granted)
            b.add_goal("get the treasure", {"has_treasure": True})
            b.run(env, 60)
            self.assertTrue(env.s["has_treasure"], msg=str(granted))

    def test_env_action_denied_by_default_grants(self):
        # Phase 2: Brain() with no granted argument gets ("compute",)
        # only, so environment actions are denied by default.
        env = RoomWorld()
        b = Brain(room_world_actions())
        self.assertEqual(b.granted, {"compute"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.run(env, 60)
        self.assertFalse(env.s["has_treasure"])
        self.assertEqual(env.log, [])
        self.assertTrue(any(e["kind"] == "deny" for e in b.events))


if __name__ == "__main__":
    unittest.main()
