"""Phase 7 execution subsystem (D7-1): session, monitor, ledger.

``ExecutionSession`` is one managed run of one goal's plan: a stable
task id, budgets (cycles / tool calls / wall seconds), a status, and
per-step records. ``ExecutionMonitor`` owns the session for the active
goal and is checked from ``Brain.step()`` before the rule fires and
after each step; on a budget violation it logs
``execution_budget_exceeded``, sets the session status, and asks the
brain to stop via the existing ``_set_stop`` path (no second stop
mechanism). On cancellation it logs ``execution_cancelled`` and the
brain suspends the goal with plan and position preserved.
``ExecutionLedger`` is the auditable trail: one record per executed
action, bounded at 1,000 records (oldest-first pruning), persisted in
``execution.json``.

Stdlib only, deterministic. Arguments and effects in the plan
snapshot and ledger are redacted via ``ab.core.audit.redact_secrets``
before they are stored, so a secret fed through an executed tool
never reaches the saved record.
"""
from __future__ import annotations

import hashlib
import json
import time

from ..core.audit import redact_secrets

# D7-3 defaults (design §5): generous so existing tasks never trip
# them — max_cycles comes from the brain's own control budget.
DEFAULT_MAX_TOOL_CALLS = 50
DEFAULT_MAX_WALL_S = 60.0

# Same bound as the Phase 3 audit trail (ab/core/audit.py MAX_EVENTS).
MAX_LEDGER_RECORDS = 1000

SESSION_STATUSES = ("running", "done", "failed", "cancelled",
                    "budget_exceeded", "interrupted")

LEDGER_OUTCOMES = ("ok", "failed", "timed_out", "denied", "cancelled")


class DependencyError(Exception):
    """Raised when an action is blocked by unsatisfied preconditions.

    The brain uses the return value of ``Brain.check_dependencies``
    for its recoverable ``dependency_blocked`` path, so this exception
    is for direct callers that prefer to raise rather than inspect a
    list; it is never raised inside the cycle itself.
    """


def _plan_signature(plan_steps) -> str:
    """Canonical signature of a plan: action names + redacted args +
    effects, in order. Part of the stable task id."""
    parts = []
    for step in plan_steps or []:
        if isinstance(step, dict):
            name = step.get("name", "")
            args = step.get("args", {})
            eff = step.get("eff", {})
        else:
            name = getattr(step, "name", "")
            args = getattr(step, "args", {})
            eff = getattr(step, "eff", {})
        parts.append(json.dumps([str(name), redact_secrets(args),
                                 redact_secrets(eff)],
                                sort_keys=True, default=str))
    return "|".join(parts)


def _task_id(goal_id, plan_steps, started_cycle: int) -> str:
    """Stable SHA-256 of goal id + plan signature + start cycle."""
    blob = json.dumps([int(goal_id), _plan_signature(plan_steps),
                       int(started_cycle)], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ExecutionSession:
    """One managed run of one goal's plan."""

    def __init__(self, goal_id: int, plan, *, budgets: dict | None = None,
                 started_cycle: int = 0, started_wall: float | None = None,
                 task_id: str | None = None, redact_args=None):
        # Coordinator integration fix (Phase 7 gate): the plan snapshot
        # must honour ToolSpec ``secret`` markings, not just the
        # secret-name / token-shape heuristics in ``redact_secrets``.
        # ``redact_args`` is a callable ``(tool, args) -> dict``
        # supplied by the Brain (which owns the tool registry); the
        # default redacts via ``redact_secrets`` only, so direct
        # construction keeps working.
        self.redact_args = (redact_args
                            or (lambda tool, args: redact_secrets(args)))
        self.goal_id = int(goal_id)
        self.plan_snapshot = [
            {"name": (s.get("name", "") if isinstance(s, dict)
                      else getattr(s, "name", "")),
             "args": self.redact_args(
                 s.get("tool") if isinstance(s, dict)
                 else getattr(s, "tool", None),
                 s.get("args", {}) if isinstance(s, dict)
                 else getattr(s, "args", {})),
             "eff": redact_secrets(s.get("eff", {}) if isinstance(s, dict)
                                   else getattr(s, "eff", {}))}
            for s in (plan or [])]
        self.steps_executed = 0
        self.steps_total = len(self.plan_snapshot)
        self.status = "running"
        self.stop_reason: str | None = None
        self.budgets = dict(budgets or {})
        self.budgets.setdefault("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
        self.budgets.setdefault("max_wall_s", DEFAULT_MAX_WALL_S)
        self.started_cycle = int(started_cycle)
        self.started_wall = (time.monotonic() if started_wall is None
                             else float(started_wall))
        self.tool_calls = 0
        self.cancellations = 0
        self.cancel_requested = False
        self.cancel_reason: str | None = None
        self.steps: list[dict] = []      # per-step records
        self.task_id = (task_id if task_id is not None
                        else _task_id(self.goal_id, plan, self.started_cycle))

    # -- progress ----------------------------------------------------
    def record_step(self, action, outcome: str, cost: dict | None = None) -> dict:
        """Record one executed step. ``action`` is an Action or a plain
        name; ``outcome`` is one of LEDGER_OUTCOMES; ``cost`` carries
        ``tool_calls`` / ``wall_s`` / ``cycles`` (defaults 0). Returns
        the per-step record."""
        if outcome not in LEDGER_OUTCOMES:
            raise ValueError(f"unknown ledger outcome {outcome!r}")
        name = action if isinstance(action, str) else getattr(action, "name", str(action))
        cost = dict(cost or {})
        self.steps_executed += 1
        self.tool_calls += int(cost.get("tool_calls", 0))
        rec = {"action": str(name), "outcome": outcome,
               "tool_calls": int(cost.get("tool_calls", 0)),
               "wall_s": float(cost.get("wall_s", 0.0)),
               "cycles": int(cost.get("cycles", 0))}
        self.steps.append(rec)
        return rec

    # -- budgets ------------------------------------------------------
    def check_budget(self, now_cycle: int, now_wall: float) -> str | None:
        """The violated budget name, or None. Checked in the order
        cycles, tool calls, wall time; only the first violation is
        named (a run that trips several budgets at once reports the
        first, and the ledger still shows the counts that prove it)."""
        max_cycles = self.budgets.get("max_cycles")
        if max_cycles is not None and int(now_cycle) - self.started_cycle > int(max_cycles):
            return "max_cycles"
        max_calls = self.budgets.get("max_tool_calls")
        if max_calls is not None and self.tool_calls >= int(max_calls):
            return "max_tool_calls"
        max_wall = self.budgets.get("max_wall_s")
        if max_wall is not None and float(now_wall) - self.started_wall > float(max_wall):
            return "max_wall_s"
        return None

    # -- cancellation -------------------------------------------------
    def request_cancel(self, reason: str = "user_cancelled") -> None:
        """Flag the session for cancellation. The monitor (checked from
        ``Brain.step()``) turns the flag into a ``cancelled`` status
        and the brain suspends the goal; a tool running in isolated
        mode is killed by ``Brain.cancel_execution`` before this flag
        is acted on."""
        self.cancel_requested = True
        self.cancel_reason = str(reason)
        self.cancellations += 1

    # -- persistence --------------------------------------------------
    def to_json(self) -> dict:
        return {"version": 1, "task_id": self.task_id,
                "goal_id": self.goal_id,
                "plan_snapshot": [dict(s) for s in self.plan_snapshot],
                "steps_executed": self.steps_executed,
                "steps_total": self.steps_total,
                "status": self.status, "stop_reason": self.stop_reason,
                "budgets": dict(self.budgets),
                "started_cycle": self.started_cycle,
                "started_wall": self.started_wall,
                "tool_calls": self.tool_calls,
                "cancellations": self.cancellations,
                "cancel_requested": self.cancel_requested,
                "cancel_reason": self.cancel_reason,
                "steps": [dict(s) for s in self.steps]}

    @classmethod
    def from_json(cls, data: dict) -> "ExecutionSession":
        if not isinstance(data, dict):
            raise ValueError("execution session data must be a dict")
        session = cls(int(data.get("goal_id", 0)),
                      data.get("plan_snapshot", []),
                      budgets=dict(data.get("budgets", {})),
                      started_cycle=int(data.get("started_cycle", 0)),
                      started_wall=data.get("started_wall"),
                      task_id=data.get("task_id"))
        # The plan snapshot is already redacted dicts; keep it verbatim
        # (the constructor above re-derived it from the same dicts).
        session.plan_snapshot = [dict(s) for s in data.get("plan_snapshot", [])]
        session.steps_executed = int(data.get("steps_executed", 0))
        session.steps_total = int(data.get("steps_total",
                                           len(session.plan_snapshot)))
        session.status = str(data.get("status", "interrupted"))
        if session.status not in SESSION_STATUSES:
            raise ValueError(f"unknown session status {session.status!r}")
        session.stop_reason = data.get("stop_reason")
        session.tool_calls = int(data.get("tool_calls", 0))
        session.cancellations = int(data.get("cancellations", 0))
        session.cancel_requested = bool(data.get("cancel_requested", False))
        session.cancel_reason = data.get("cancel_reason")
        session.steps = [dict(s) for s in data.get("steps", [])]
        return session


class ExecutionLedger:
    """The auditable trail: one record per executed action.

    A record carries the cycle, action name, redacted args, tool (if
    any), outcome, predicted vs actual effects, cost (cycles / wall /
    tool calls) and the decision that chose it (rule id + plan cost +
    alternative considered, when known). Bounded at 1,000 records,
    oldest-first pruning — the same shape as the Phase 3 audit trail.
    """

    def __init__(self, records: list | None = None):
        self.records: list[dict] = [dict(r) for r in (records or [])]

    def __len__(self) -> int:
        return len(self.records)

    def record(self, *, cycle: int, action: str, args=None, tool=None,
               outcome: str, predicted=None, actual=None, cost=None,
               decision=None, verified: bool | None = None) -> dict:
        """Append one record (redacting args/effects first) and prune
        oldest-first to :data:`MAX_LEDGER_RECORDS`. Returns the record."""
        if outcome not in LEDGER_OUTCOMES:
            raise ValueError(f"unknown ledger outcome {outcome!r}")
        cost = dict(cost or {})
        entry = {"cycle": int(cycle), "action": str(action),
                 "args": redact_secrets(args if args is not None else {}),
                 "tool": tool,
                 "outcome": outcome,
                 "predicted": redact_secrets(predicted if predicted is not None else {}),
                 "actual": redact_secrets(actual if actual is not None else {}),
                 "cost": {"cycles": int(cost.get("cycles", 0)),
                          "wall_s": float(cost.get("wall_s", 0.0)),
                          "tool_calls": int(cost.get("tool_calls", 0))},
                 "decision": decision if decision is not None else {},
                 "verified": verified}
        self.records.append(entry)
        if len(self.records) > MAX_LEDGER_RECORDS:
            self.records = self.records[-MAX_LEDGER_RECORDS:]
        return entry

    def to_json(self) -> dict:
        return {"version": 1, "records": [dict(r) for r in self.records]}

    @classmethod
    def from_json(cls, data: dict) -> "ExecutionLedger":
        if not isinstance(data, dict):
            raise ValueError("execution ledger data must be a dict")
        records = data.get("records", [])
        if not isinstance(records, list):
            raise ValueError("execution ledger 'records' must be a list")
        return cls([dict(r) for r in records])


class ExecutionMonitor:
    """Owns the session (and ledger) for the active goal.

    Checked from ``Brain.step()`` before the rule fires and after each
    step. ``check(brain)`` returns ``"budget_exceeded"``,
    ``"cancelled"``, or None. On a budget violation it logs an
    ``execution_budget_exceeded`` event, sets the session status, and
    asks the brain to stop via the existing ``_set_stop`` path. On a
    cancelled session it logs ``execution_cancelled`` and sets the
    status; the brain suspends the goal (plan + position preserved)
    in response to the returned value.
    """

    def __init__(self, ledger: ExecutionLedger | None = None,
                 redact_args=None):
        self.session: ExecutionSession | None = None
        self.ledger = ledger if ledger is not None else ExecutionLedger()
        # Coordinator integration fix (Phase 7 gate): see
        # ExecutionSession — the Brain supplies a spec-aware redactor
        # (ToolSpec ``secret`` markings); the default is
        # ``redact_secrets`` only.
        self.redact_args = (redact_args
                            or (lambda tool, args: redact_secrets(args)))

    # -- session lifecycle -------------------------------------------
    def open_session(self, goal_id: int, plan, budgets: dict,
                     cycle: int, now_wall: float | None = None) -> ExecutionSession:
        """Open a fresh session for ``goal_id`` (replacing any closed
        session for a previous goal). Emits nothing itself — the brain
        logs the ``execution_started`` event, which names the task id,
        steps and budgets."""
        self.session = ExecutionSession(
            goal_id, plan, budgets=dict(budgets), started_cycle=int(cycle),
            started_wall=now_wall, redact_args=self.redact_args)
        return self.session

    def reopen_session(self, plan) -> ExecutionSession | None:
        """Re-open an ``interrupted`` (or ``cancelled``) session (after
        a save/load, an interruption, or an explicit resume after a
        cancellation) with its plan snapshot refreshed. Returns None
        when there is no such session to re-open. A cancelled session
        re-opens only through an explicit resume — it never restarts
        by itself."""
        session = self.session
        if session is None or session.status not in ("interrupted", "cancelled"):
            return None
        session.status = "running"
        session.cancel_requested = False
        session.cancel_reason = None
        session.plan_snapshot = [
            {"name": (s.get("name", "") if isinstance(s, dict)
                      else getattr(s, "name", "")),
             "args": session.redact_args(
                 s.get("tool") if isinstance(s, dict)
                 else getattr(s, "tool", None),
                 s.get("args", {}) if isinstance(s, dict)
                 else getattr(s, "args", {})),
             "eff": redact_secrets(s.get("eff", {}) if isinstance(s, dict)
                                   else getattr(s, "eff", {}))}
            for s in (plan or [])]
        session.steps_total = len(session.plan_snapshot)
        return session

    def close(self, status: str, reason: str | None = None) -> None:
        """Close the live session with its final status. No-op when
        there is no live (``running``) session."""
        session = self.session
        if session is None or session.status != "running":
            return
        session.status = status
        session.stop_reason = reason

    # -- enforcement ---------------------------------------------------
    def check(self, brain, now_cycle: int | None = None,
              now_wall: float | None = None) -> str | None:
        """Enforce budgets / cancellation for the live session.

        Returns ``"budget_exceeded"`` (session closed, brain stopped
        with reason ``budget_exceeded``), ``"cancelled"`` (session
        closed as cancelled; the caller suspends the goal), or None.
        """
        session = self.session
        if session is None or session.status != "running":
            return None
        cycle = self._brain_cycle(brain, now_cycle)
        wall = time.monotonic() if now_wall is None else float(now_wall)
        if session.cancel_requested:
            session.status = "cancelled"
            session.stop_reason = session.cancel_reason or "user_cancelled"
            brain._event("execution_cancelled",
                         f"task {session.task_id[:12]} cancelled "
                         f"(reason={session.stop_reason}; "
                         f"steps {session.steps_executed}/{session.steps_total} "
                         f"executed; plan and position preserved)")
            return "cancelled"
        violated = session.check_budget(cycle, wall)
        if violated is not None:
            session.status = "budget_exceeded"
            session.stop_reason = violated
            brain._event("execution_budget_exceeded",
                         f"task {session.task_id[:12]} exceeded budget "
                         f"'{violated}' (cycles {cycle - session.started_cycle}/"
                         f"{session.budgets.get('max_cycles')}, tool calls "
                         f"{session.tool_calls}/"
                         f"{session.budgets.get('max_tool_calls')}, wall "
                         f"{wall - session.started_wall:.2f}s/"
                         f"{session.budgets.get('max_wall_s')}s)")
            brain._set_stop("budget_exceeded")
            return "budget_exceeded"
        return None

    @staticmethod
    def _brain_cycle(brain, now_cycle):
        if now_cycle is not None:
            return int(now_cycle)
        return int(getattr(brain, "cycle", 0))

    # -- persistence ---------------------------------------------------
    def to_json(self) -> dict:
        return {"version": 1,
                "session": (self.session.to_json()
                            if self.session is not None else None),
                "ledger": self.ledger.to_json()}

    @classmethod
    def from_json(cls, data: dict) -> "ExecutionMonitor":
        if not isinstance(data, dict):
            raise ValueError("execution data must be a dict")
        monitor = cls()
        session_data = data.get("session")
        if isinstance(session_data, dict):
            monitor.session = ExecutionSession.from_json(session_data)
        ledger_data = data.get("ledger")
        if isinstance(ledger_data, dict):
            monitor.ledger = ExecutionLedger.from_json(ledger_data)
        return monitor
