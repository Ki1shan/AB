"""Tool execution with typed contracts, permissions, and an audit trail.

Phase 3 changes (design SS3.4-3.5, audit gaps B5 and B9):
- Every tool registers a :class:`ToolSpec`: name, typed argument schema
  (name, type, required, default, optional ``secret`` flag), return type,
  timeout, required grant, version.
- :meth:`ToolRegistry.call` validates arguments against the spec BEFORE
  execution and returns a typed error (``"arg_error: ..."``) on
  mismatch -- missing required arg, wrong type, unknown arg -- instead
  of crashing inside the tool.
- Duplicate registration raises :class:`ValueError` instead of silently
  overwriting.
- Every call is recorded in :attr:`ToolRegistry.audit` with its spec
  version, arguments (redacted where marked secret, via
  ``ab.core.audit.redact_secrets``), result, and duration.
- ``safe_calc`` caps expression length (200), AST node count (100), and
  AST depth (20), all raising :class:`ValueError` like any other
  unsupported expression.

Backwards compatibility: the legacy :class:`Tool` dataclass and
``register(Tool(...))`` still work (existing tests and ``Brain`` use
them). A legacy tool is converted to a spec whose arguments are
untyped (``"any"``) and required exactly as its ``required_args``
said, so its validation behaviour is unchanged.

Timeout honesty (unchanged from Phase 1, audit #12): the budget is
enforced with a thread pool. A timed-out Python thread is NOT killed
-- CPython cannot terminate a running thread -- it lingers in the
background. The timeout guarantees only that the *cycle* is not
stalled and that the call is recorded as failed (``timed_out=True``).

Phase 7 (D7-1/D7-2/D7-6): true isolation is now available as an
opt-in path. ``ToolRegistry.call(..., isolated=True)`` runs the tool
in a forked child via :class:`ab.body.sandbox.IsolatedToolRunner`;
on expiry the child's process group is SIGKILLed and the kill is
observed before the result returns (``killed=True`` only when the
process was observed dead). ``ToolRegistry.cancel()`` kills the
currently running isolated child. The default ``isolated=False``
preserves the legacy thread path for existing callers. Tool outputs
are bounded: a string longer than 64 KiB is truncated (audit flag
``output_truncated``) and a non-JSON-able output is stored as its
truncated repr. Audit records gain ``isolated``, ``killed``,
``limits_applied`` and ``output_truncated`` fields.

One deliberate exception to isolation: a tool implemented as a bound
method operates on its object's in-process state, so forking it would
silently discard its effects while reporting success. The Brain's own
``research`` tool is the only such tool in the tree (it admits claims
to LTM and writes state); it runs on the thread path when
``isolated=True`` is requested, and the audit record says so
(``isolation_skipped``). See :meth:`ToolRegistry.call`.
"""
from __future__ import annotations

import ast
import inspect
import json
import operator
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Callable

from ..core.audit import redact_secrets
from .sandbox import MAX_TOOL_OUTPUT_BYTES, IsolatedToolRunner

DEFAULT_TOOL_TIMEOUT = 5.0
_MAX_EXPR_LEN = 200
_MAX_AST_NODES = 100
_MAX_AST_DEPTH = 20

_ARG_TYPES = ("str", "int", "float", "bool", "list", "dict", "any")


@dataclass
class Tool:
    """Legacy untyped tool descriptor. Kept so existing callers
    (``Brain``, older tests) keep working; new tools should register a
    :class:`ToolSpec` instead."""
    name: str
    fn: Callable
    permission: str                     # e.g. "compute", "network", "environment"
    description: str = ""
    required_args: tuple = ()


@dataclass
class ToolSpec:
    """Typed contract for one tool.

    ``args`` is a list of dicts, each with:
    - ``name`` (str): argument name.
    - ``type`` (str): one of ``"str"``, ``"int"``, ``"float"``,
      ``"bool"``, ``"list"``, ``"dict"`` (``"any"`` is accepted for
      legacy conversions and means "no type check").
    - ``required`` (bool).
    - ``default`` (any, optional): filled in when an optional argument
      is omitted.
    - ``secret`` (bool, optional): when True, the argument's value is
      redacted in the audit record.
    """
    name: str
    args: list = field(default_factory=list)
    return_type: str = "any"
    timeout: float = DEFAULT_TOOL_TIMEOUT
    required_grant: str = "compute"
    version: int = 1
    description: str = ""


@dataclass
class ToolResult:
    ok: bool
    output: object = None
    error: str = ""
    timed_out: bool = False             # True only when the budget expired
    # Phase 7: isolation fields. ``killed`` is True ONLY when a child
    # process was observed dead (timeout or cancel); the thread path
    # never sets it, because a thread cannot be killed.
    killed: bool = False
    isolated: bool = False              # True when the tool ran in a child process
    limits_applied: tuple = ()          # e.g. ("RLIMIT_CPU=60s", "RLIMIT_AS=512MiB")
    output_truncated: bool = False      # True when the output was cut to the 64 KiB bound
    cancelled: bool = False             # True when cancel() killed the run


_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _ast_depth(node) -> int:
    """Depth of an AST: 1 for a leaf, 1 + max child depth otherwise."""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)


def safe_calc(expression: str):
    """Arithmetic only: no names, calls, or attribute access.

    Length, node-count, and depth caps keep a hostile expression from
    exhausting the parser or the recursion budget; all three raise
    ValueError like any other unsupported expression.
    """
    if len(expression) > _MAX_EXPR_LEN:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("expression too complex")
    if _ast_depth(tree) > _MAX_AST_DEPTH:
        raise ValueError("expression too deep")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        raise ValueError("unsupported expression")
    return ev(tree)


def _check_type(value, type_name: str) -> bool:
    """Strict type check. ``bool`` is checked before ``int`` (a bool is
    an int in Python, but it is not an int argument), and ``float``
    accepts an int but not a bool. ``"any"`` always passes."""
    if type_name == "any":
        return True
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "list":
        return isinstance(value, list)
    if type_name == "dict":
        return isinstance(value, dict)
    return False


def _validate_args(spec: ToolSpec, args: dict):
    """Validate ``args`` against ``spec`` before execution.

    Returns ``(error, effective_args)``: on success ``error`` is None
    and ``effective_args`` is ``args`` with defaults filled in for
    omitted optional arguments; on failure ``error`` is a typed message
    starting with ``"arg_error:"`` and ``effective_args`` is None.
    """
    if not isinstance(args, dict):
        return f"arg_error: arguments for tool '{spec.name}' must be a dict", None
    by_name = {entry["name"]: entry for entry in spec.args}
    for entry in spec.args:
        if entry.get("required") and entry["name"] not in args:
            return (f"arg_error: missing required argument '{entry['name']}' "
                    f"for tool '{spec.name}'"), None
    for key in args:
        if key not in by_name:
            return (f"arg_error: unknown argument '{key}' "
                    f"for tool '{spec.name}'"), None
    for entry in spec.args:
        name = entry["name"]
        if name in args:
            type_name = entry.get("type", "any")
            if not _check_type(args[name], type_name):
                return (f"arg_error: argument '{name}' for tool '{spec.name}' "
                        f"must be of type '{type_name}', "
                        f"got '{type(args[name]).__name__}'"), None
    effective = dict(args)
    for entry in spec.args:
        if entry["name"] not in effective and "default" in entry:
            effective[entry["name"]] = entry["default"]
    return None, effective


def _spec_from_tool(tool: Tool) -> ToolSpec:
    """Convert a legacy :class:`Tool` to a spec. Its arguments are
    untyped (``"any"``) and required exactly as ``required_args``
    said, so validation behaviour for legacy tools is unchanged."""
    return ToolSpec(
        name=tool.name,
        args=[{"name": a, "type": "any", "required": True}
              for a in tool.required_args],
        return_type="any",
        timeout=DEFAULT_TOOL_TIMEOUT,
        required_grant=tool.permission,
        version=1,
        description=tool.description,
    )


class ToolRegistry:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._fns: dict[str, Callable] = {}
        self._tools: dict[str, Tool] = {}   # legacy view, kept for compatibility
        self.audit: list[dict] = []
        # Phase 7: one runner for all isolated calls; it reports
        # ``available == False`` on platforms without fork/resource.
        self._runner = IsolatedToolRunner()
        self.register(ToolSpec(
            name="calc",
            args=[{"name": "expression", "type": "str", "required": True}],
            return_type="float",
            timeout=DEFAULT_TOOL_TIMEOUT,
            required_grant="compute",
            version=1,
            description="Evaluate an arithmetic expression",
        ), lambda expression: safe_calc(expression))

    def register(self, spec_or_tool, fn: Callable | None = None) -> None:
        """Register a tool.

        ``register(spec, fn)`` for a :class:`ToolSpec`, or
        ``register(Tool(...))`` for a legacy tool. Registering a name
        that already exists raises :class:`ValueError` -- there is no
        silent overwrite. A spec whose argument type is not one of the
        known types also raises :class:`ValueError`.
        """
        if isinstance(spec_or_tool, Tool):
            spec = _spec_from_tool(spec_or_tool)
            fn = spec_or_tool.fn
        elif isinstance(spec_or_tool, ToolSpec):
            spec = spec_or_tool
            if fn is None:
                raise ValueError(f"register('{spec.name}'): a ToolSpec needs a function")
        else:
            raise TypeError("register() takes a ToolSpec (plus fn) or a Tool")
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool registration: '{spec.name}' is already registered")
        for entry in spec.args:
            if entry.get("type", "any") not in _ARG_TYPES:
                raise ValueError(f"register('{spec.name}'): unknown arg type "
                                 f"'{entry.get('type')}' for argument '{entry.get('name')}'")
        self._specs[spec.name] = spec
        self._fns[spec.name] = fn
        self._tools[spec.name] = Tool(spec.name, fn, spec.required_grant,
                                      spec.description,
                                      tuple(e["name"] for e in spec.args if e.get("required")))

    def names(self) -> list[str]:
        return sorted(self._specs)

    def replace_fn(self, name: str, fn: Callable) -> None:
        """Replace the function behind an already-registered tool,
        keeping its spec (name, args, grant, timeout, version).

        Phase 11 (W5): the Brain re-registers the ``research`` tool
        with a fresh ``functools.partial`` whenever its provider (and
        hence its provider spec) changes — ``register`` refuses a
        duplicate name, and there is no silent overwrite. Raises
        ``KeyError`` for an unknown tool name."""
        if name not in self._specs:
            raise KeyError(f"replace_fn: no such tool {name!r}")
        spec = self._specs[name]
        self._fns[name] = fn
        self._tools[name] = Tool(name, fn, spec.required_grant,
                                 spec.description,
                                 tuple(e["name"] for e in spec.args
                                       if e.get("required")))

    def spec(self, name: str) -> ToolSpec | None:
        """Return the registered spec for ``name``, or None."""
        return self._specs.get(name)

    def call(self, name: str, args: dict, granted: set[str],
             timeout: float | None = DEFAULT_TOOL_TIMEOUT,
             isolated: bool = False) -> ToolResult:
        """Call a tool with a permission check, argument validation, and
        an execution budget.

        Validation runs BEFORE execution: a missing required argument,
        a wrong type, or an unknown argument returns
        ``ok=False`` with a typed ``"arg_error: ..."`` message and the
        tool never runs.

        ``timeout`` is in seconds (None = no budget; only callers that
        deliberately opt out should pass None). On timeout the result is
        ``ok=False, timed_out=True`` and the audit record is marked so.

        ``isolated`` (Phase 7, D7-2): when True the tool runs in a
        forked child process via :class:`IsolatedToolRunner`. On expiry
        the child is SIGKILLed and the kill is observed; the result
        carries ``killed=True`` only when the process was observed
        dead. When isolation is unavailable on the platform the call
        falls back to the thread path and the audit record carries
        ``isolation_unavailable`` with the reason -- never silently.
        One exception, recorded in the audit as ``isolation_skipped``:
        a tool implemented as a *bound method* (``inspect.ismethod``)
        operates on its object's in-process state -- the Brain's own
        ``research`` tool admits claims to LTM and writes state -- and
        a forked child would run on a copy of that state, so the tool
        would report success while its effects were silently
        discarded. Such a tool runs on the thread path instead; the
        skip is explicit in the audit record, never silent.
        Default False preserves the legacy thread path: the underlying
        thread is NOT killed on timeout -- CPython cannot terminate a
        running thread -- it may linger after the call returns. The
        cycle is never blocked waiting for it.
        """
        started = time.monotonic()
        isolation_unavailable: str | None = None
        isolation_skipped: str | None = None
        spec = self._specs.get(name)
        if spec is None:
            res = ToolResult(False, error=f"unknown tool '{name}'")
            redacted_args = redact_secrets(args)
            result = None
        elif spec.required_grant not in granted:
            res = ToolResult(False, error=f"permission '{spec.required_grant}' not granted for '{name}'")
            redacted_args = self._redacted_args(spec, args)
            result = None
        else:
            error, effective = _validate_args(spec, args)
            if error is not None:
                res = ToolResult(False, error=error)
                redacted_args = self._redacted_args(spec, args)
                result = None
            else:
                fn = self._fns[name]
                if isolated and self._runner.available:
                    if inspect.ismethod(fn):
                        # Bound-method tools mutate their object's
                        # in-process state (see the call() docstring);
                        # isolating one would silently discard its
                        # effects while reporting success. Thread path,
                        # with the skip recorded in the audit entry.
                        isolation_skipped = (
                            "bound method: in-process state changes "
                            "would be lost in a forked child")
                        res = self._run_with_timeout(fn, effective, timeout)
                    else:
                        res = self._run_isolated(fn, effective, timeout)
                elif isolated:
                    # Isolation requested but unavailable: explicit
                    # fallback, recorded in the audit entry below.
                    isolation_unavailable = self._runner.unavailable_reason
                    res = self._run_with_timeout(fn, effective, timeout)
                else:
                    res = self._run_with_timeout(fn, effective, timeout)
                redacted_args = self._redacted_args(spec, args)
                if res.ok:
                    # Phase 7 (D7-6): bound the output before anything
                    # stores it (audit, store_as). A string over 64 KiB
                    # is truncated; a non-JSON-able value becomes its
                    # truncated repr and is never trusted blindly.
                    bounded, truncated = self._bound_output(res.output)
                    res.output = bounded
                    res.output_truncated = truncated
                result = redact_secrets(res.output) if res.ok else None
        self.audit.append({"tool": name,
                           "spec_version": spec.version if spec is not None else None,
                           "args": redacted_args, "result": result,
                           "ok": res.ok, "error": res.error,
                           "timed_out": res.timed_out,
                           "isolated": res.isolated,
                           "killed": res.killed,
                           "limits_applied": list(res.limits_applied),
                           "output_truncated": res.output_truncated,
                           "isolation_unavailable": isolation_unavailable,
                           "isolation_skipped": isolation_skipped,
                           "duration_s": round(time.monotonic() - started, 4),
                           "granted": sorted(granted)})
        return res

    def cancel(self) -> bool:
        """Kill the currently running isolated child, if any.

        Returns True only when a process was observed dead afterwards;
        False when nothing is running (or the kill could not be
        observed). The in-flight isolated call then returns a result
        with ``cancelled=True``. Thread-path calls are unaffected:
        a thread cannot be killed, so there is nothing to cancel.
        """
        return self._runner.cancel()

    @staticmethod
    def _bound_output(output):
        """Bound a tool output to :data:`MAX_TOOL_OUTPUT_BYTES`.

        Returns ``(bounded_output, truncated)``: a string longer than
        64 KiB (UTF-8 bytes) is cut to 64 KiB; a JSON-able value whose
        JSON form exceeds 64 KiB becomes its truncated repr; a
        non-JSON-able value becomes its (truncated) repr. ``truncated``
        is True only when bytes were actually cut.
        """
        if isinstance(output, str):
            encoded = output.encode("utf-8")
            if len(encoded) > MAX_TOOL_OUTPUT_BYTES:
                return (encoded[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", "ignore"),
                        True)
            return output, False
        try:
            encoded = json.dumps(output).encode("utf-8")
        except (TypeError, ValueError):
            text = repr(output)
            cut = text.encode("utf-8")
            if len(cut) > MAX_TOOL_OUTPUT_BYTES:
                return cut[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", "ignore"), True
            return text, False
        if len(encoded) > MAX_TOOL_OUTPUT_BYTES:
            text = repr(output)
            cut = text.encode("utf-8")
            return cut[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", "ignore"), True
        return output, False

    @staticmethod
    def _redacted_args(spec: ToolSpec, args: dict) -> dict:
        """Audit copy of ``args``: secret-marked arguments become
        ``"[REDACTED]"``; every other value still passes through
        :func:`redact_secrets` in case it carries a secret substring."""
        if not isinstance(args, dict):
            return redact_secrets(args)
        secret_names = {e["name"] for e in spec.args if e.get("secret")}
        return {k: ("[REDACTED]" if k in secret_names else redact_secrets(v))
                for k, v in args.items()}

    def _run_isolated(self, fn: Callable, args: dict, timeout: float | None) -> ToolResult:
        """Run ``fn`` in a forked child via :class:`IsolatedToolRunner`
        and map its result onto :class:`ToolResult`."""
        r = self._runner.run(fn, args, timeout)
        return ToolResult(ok=r.ok, output=r.output, error=r.error,
                          timed_out=r.timed_out, killed=r.killed,
                          isolated=r.isolated,
                          limits_applied=tuple(r.limits_applied),
                          cancelled=r.cancelled)

    @staticmethod
    def _run_with_timeout(fn: Callable, args: dict, timeout: float | None) -> ToolResult:
        if timeout is None:
            try:
                return ToolResult(True, fn(**args))
            except Exception as exc:                      # tool failures never crash the brain
                return ToolResult(False, error=f"{type(exc).__name__}: {exc}")
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(fn, **args)
            try:
                return ToolResult(True, fut.result(timeout=timeout))
            except FuturesTimeout:
                # The worker thread keeps running; it is abandoned, not killed.
                return ToolResult(False, error=f"timeout after {timeout}s", timed_out=True)
            except Exception as exc:                      # tool failures never crash the brain
                return ToolResult(False, error=f"{type(exc).__name__}: {exc}")
        finally:
            # wait=False: do not block the cycle on a hung worker thread.
            ex.shutdown(wait=False, cancel_futures=True)
