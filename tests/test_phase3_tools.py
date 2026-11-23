"""Phase 3 tests: typed tool contracts, safe_calc depth bound, audit trail.

Covers design SS3.4-3.5 and audit gaps B5/B9. Existing tests are
untouched; these only add coverage for the new ToolSpec/audit API.
"""
import json
import time
import unittest

from ab.body.tools import Tool, ToolRegistry, ToolSpec, safe_calc
from ab.core.audit import (
    EVENT_FIELDS,
    EVENT_SCHEMA_VERSION,
    MAX_EVENTS,
    new_event,
    prune_events,
    redact_secrets,
    validate_event,
)

FAKE_SECRET = "sk-test-SECRET-VALUE-12345"


def _nested_add(depth: int) -> str:
    """'1+(1+(1+...))' with ``depth`` additions; value is depth + 1."""
    expr = "1"
    for _ in range(depth):
        expr = f"1+({expr})"
    return expr


class TestToolSpecRegistration(unittest.TestCase):
    def test_duplicate_spec_registration_raises(self):
        r = ToolRegistry()
        spec = ToolSpec("echo", [{"name": "x", "type": "str", "required": True}])
        r.register(spec, lambda x: x)
        with self.assertRaises(ValueError):
            r.register(spec, lambda x: x)
        with self.assertRaises(ValueError):
            r.register(ToolSpec("echo", []), lambda: None)

    def test_duplicate_legacy_tool_registration_raises(self):
        r = ToolRegistry()
        r.register(Tool("echo", lambda x: x, "compute", "", ("x",)))
        with self.assertRaises(ValueError):
            r.register(Tool("echo", lambda x: x, "compute"))

    def test_duplicate_calc_registration_raises(self):
        r = ToolRegistry()          # calc is registered by __init__
        with self.assertRaises(ValueError):
            r.register(ToolSpec("calc", []), lambda: None)

    def test_unknown_arg_type_raises(self):
        r = ToolRegistry()
        with self.assertRaises(ValueError):
            r.register(ToolSpec("bad", [{"name": "x", "type": "weird",
                                         "required": True}]), lambda x: x)

    def test_spec_without_function_raises(self):
        r = ToolRegistry()
        with self.assertRaises(ValueError):
            r.register(ToolSpec("nofn", []))


class TestToolArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.r = ToolRegistry()
        self.r.register(ToolSpec(
            "add", [{"name": "a", "type": "int", "required": True},
                    {"name": "b", "type": "int", "required": True},
                    {"name": "label", "type": "str", "required": False,
                     "default": "sum"}],
            return_type="int", required_grant="compute"),
            lambda a, b, label="sum": a + b)

    def test_valid_call_runs_and_fills_default(self):
        res = self.r.call("add", {"a": 2, "b": 3}, {"compute"})
        self.assertTrue(res.ok)
        self.assertEqual(res.output, 5)

    def test_missing_required_arg_is_typed_arg_error(self):
        res = self.r.call("add", {"a": 2}, {"compute"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("arg_error:"), res.error)
        self.assertIn("b", res.error)
        self.assertFalse(res.timed_out)

    def test_wrong_type_is_typed_arg_error(self):
        res = self.r.call("add", {"a": "two", "b": 3}, {"compute"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("arg_error:"), res.error)
        self.assertIn("int", res.error)

    def test_bool_is_not_an_int(self):
        res = self.r.call("add", {"a": True, "b": 3}, {"compute"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("arg_error:"), res.error)

    def test_unknown_arg_is_typed_arg_error(self):
        res = self.r.call("add", {"a": 2, "b": 3, "c": 4}, {"compute"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("arg_error:"), res.error)
        self.assertIn("c", res.error)

    def test_invalid_call_never_runs_the_tool(self):
        seen = []
        self.r.register(ToolSpec(
            "spy", [{"name": "x", "type": "int", "required": True}]),
            lambda x: seen.append(x))
        self.r.call("spy", {"x": "not an int"}, {"compute"})
        self.assertEqual(seen, [])

    def test_audit_records_spec_version_result_and_duration(self):
        self.r.call("add", {"a": 1, "b": 2}, {"compute"})
        entry = self.r.audit[-1]
        self.assertEqual(entry["spec_version"], 1)
        self.assertEqual(entry["result"], 3)
        self.assertIn("duration_s", entry)
        self.assertTrue(entry["ok"])

    def test_permission_still_enforced(self):
        res = self.r.call("add", {"a": 1, "b": 2}, {"network"})
        self.assertFalse(res.ok)
        self.assertIn("permission", res.error)

    def test_unknown_tool_still_errors(self):
        res = self.r.call("ghost", {}, {"compute"})
        self.assertFalse(res.ok)
        self.assertIn("unknown tool", res.error)


class TestToolAuditRedaction(unittest.TestCase):
    def test_secret_arg_redacted_in_registry_audit(self):
        r = ToolRegistry()
        r.register(ToolSpec(
            "lookup", [{"name": "api_key", "type": "str", "required": True,
                        "secret": True},
                       {"name": "query", "type": "str", "required": True}]),
            lambda api_key, query: f"got {query}")
        res = r.call("lookup", {"api_key": FAKE_SECRET, "query": "hello"},
                     {"compute"})
        self.assertTrue(res.ok)
        blob = json.dumps(r.audit)
        self.assertNotIn(FAKE_SECRET, blob)
        entry = r.audit[-1]
        self.assertEqual(entry["args"]["api_key"], "[REDACTED]")
        self.assertEqual(entry["args"]["query"], "hello")   # key kept, value kept

    def test_secret_in_tool_result_redacted_in_audit(self):
        r = ToolRegistry()
        r.register(ToolSpec(
            "leak", [{"name": "x", "type": "str", "required": True}]),
            lambda x: {"api_key": FAKE_SECRET, "echo": x})
        res = r.call("leak", {"x": "hi"}, {"compute"})
        self.assertTrue(res.ok)                    # caller still gets the real result
        self.assertEqual(res.output["api_key"], FAKE_SECRET)
        self.assertNotIn(FAKE_SECRET, json.dumps(r.audit))
        self.assertEqual(r.audit[-1]["result"]["api_key"], "[REDACTED]")

    def test_bare_secret_string_result_redacted_in_audit(self):
        r = ToolRegistry()
        r.register(ToolSpec(
            "token_out", [{"name": "x", "type": "str", "required": True}]),
            lambda x: FAKE_SECRET)
        self.r = r
        res = r.call("token_out", {"x": "hi"}, {"compute"})
        self.assertTrue(res.ok)
        self.assertNotIn(FAKE_SECRET, json.dumps(r.audit))
        self.assertEqual(r.audit[-1]["result"], "[REDACTED]")


class TestToolTimeout(unittest.TestCase):
    def test_hanging_tool_times_out_without_stalling_caller(self):
        # Short sleeper (not the 10 s one in test_phase1_security):
        # the timeout must return quickly and flag timed_out. The
        # thread itself is NOT killed -- it lingers; nothing here
        # claims otherwise.
        r = ToolRegistry()
        r.register(Tool("slow", lambda: time.sleep(0.5), "compute"))
        started = time.monotonic()
        res = r.call("slow", {}, {"compute"}, timeout=0.1)
        elapsed = time.monotonic() - started
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertIn("timeout", res.error)
        self.assertLess(elapsed, 0.4)
        self.assertTrue(r.audit[-1]["timed_out"])


class TestSafeCalcDepth(unittest.TestCase):
    def test_deeply_nested_expression_raises_value_error(self):
        # 30 nested additions: 62 AST nodes, 124 chars -- under the
        # length and node caps, over the depth cap of 20.
        with self.assertRaises(ValueError):
            safe_calc(_nested_add(30))

    def test_moderately_nested_expression_still_works(self):
        self.assertEqual(safe_calc(_nested_add(8)), 9)

    def test_existing_caps_still_hold(self):
        self.assertEqual(safe_calc("2 + 3 * 4"), 14)
        with self.assertRaises(ValueError):
            safe_calc("1" * 201)
        with self.assertRaises(ValueError):
            safe_calc("+".join(["1"] * 60) + "+1")


class TestAuditModule(unittest.TestCase):
    def test_redact_secrets_str_keeps_key_redacts_value(self):
        out = redact_secrets(f"api_key={FAKE_SECRET}")
        self.assertNotIn(FAKE_SECRET, out)
        self.assertIn("api_key", out)
        self.assertIn("[REDACTED]", out)

    def test_redact_secrets_dict_keeps_keys(self):
        out = redact_secrets({"api_key": FAKE_SECRET, "name": "x",
                              "nested": {"token": FAKE_SECRET}})
        self.assertNotIn(FAKE_SECRET, json.dumps(out))
        self.assertEqual(out["api_key"], "[REDACTED]")
        self.assertEqual(out["name"], "x")
        self.assertEqual(out["nested"]["token"], "[REDACTED]")

    def test_redact_secrets_list(self):
        out = redact_secrets([FAKE_SECRET, "ok", {"password": FAKE_SECRET}])
        self.assertNotIn(FAKE_SECRET, json.dumps(out))
        self.assertEqual(out[1], "ok")
        self.assertEqual(out[2]["password"], "[REDACTED]")

    def test_redact_secrets_never_raises(self):
        self.assertEqual(redact_secrets(42), 42)
        self.assertIsNone(redact_secrets(None))
        obj = object()
        self.assertIs(redact_secrets(obj), obj)

    def test_validate_event_accepts_valid(self):
        # Coordinator correction (Phase 3 integration): the v1 schema is
        # Brain's established event shape {"cycle", "kind", "detail"}.
        # An earlier draft of audit.py described a different shape;
        # the schema was aligned to Brain, not the other way round,
        # so these tests now assert the corrected shape.
        self.assertTrue(validate_event(
            {"cycle": 3, "kind": "act", "detail": "did it"}))
        self.assertTrue(validate_event(new_event(3, "act", "did it")))

    def test_validate_event_rejects_unknown_fields(self):
        ev = {"cycle": 3, "kind": "act", "detail": "y", "extra": 1}
        self.assertFalse(validate_event(ev))

    def test_validate_event_rejects_version_field_as_unknown(self):
        # The schema version is module-level (EVENT_SCHEMA_VERSION),
        # not a per-event field: an event carrying "version" is invalid.
        ev = {"version": 1, "cycle": 3, "kind": "act", "detail": "y"}
        self.assertFalse(validate_event(ev))

    def test_validate_event_rejects_missing_field_and_non_dict(self):
        ev = {"cycle": 3, "kind": "act"}
        self.assertFalse(validate_event(ev))
        self.assertFalse(validate_event(None))
        self.assertFalse(validate_event("not an event"))

    def test_event_fields_and_version_constants(self):
        self.assertEqual(EVENT_FIELDS, {"cycle", "kind", "detail"})
        self.assertEqual(EVENT_SCHEMA_VERSION, 1)
        self.assertEqual(MAX_EVENTS, 1000)

    def test_prune_events_keeps_last_1000(self):
        events = [{"version": 1, "cycle": i, "kind": "k", "reason": "r",
                   "details": "d"} for i in range(1005)]
        pruned = prune_events(events)
        self.assertEqual(len(pruned), 1000)
        self.assertEqual(pruned[0]["cycle"], 5)
        self.assertEqual(pruned[-1]["cycle"], 1004)
        self.assertEqual([e["cycle"] for e in pruned], list(range(5, 1005)))

    def test_prune_events_short_list_unchanged_copy(self):
        events = [{"version": 1, "cycle": i, "kind": "k", "reason": "r",
                   "details": "d"} for i in range(3)]
        pruned = prune_events(events)
        self.assertEqual(pruned, events)
        self.assertIsNot(pruned, events)


if __name__ == "__main__":
    unittest.main()
