"""Goals: what the brain is trying to make true."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

from ..core.vocab import normalize_text, tokenize

_gid = itertools.count(1)


def _reserve_goal_id(n: int) -> None:
    """Advance the global goal-id counter past ``n`` (used after loading
    persisted goals, so a fresh goal never reuses a restored id)."""
    while next(_gid) <= n:
        pass


@dataclass
class Goal:
    description: str
    target: dict                       # state key -> required value
    priority: float = 0.5
    deadline_cycle: int | None = None
    parent: int | None = None
    status: str = "pending"            # pending | active | done | failed
    meta: dict = field(default_factory=dict)
    id: int = field(default_factory=lambda: next(_gid))
    fail_reason: str = ""
    # Phase 3: the active plan lives HERE, in the goal record, and nowhere
    # else. ``Brain.plan`` is a read-only property over this list.
    # Entries are plain action dicts (see SuspendedGoal.plan).
    plan: list = field(default_factory=list)

    @property
    def keywords(self) -> set[str]:
        words = tokenize(self.description)
        return words | {normalize_text(k) for k in self.target} | {
            normalize_text(v) for v in self.target.values() if isinstance(v, str)}


@dataclass
class SuspendedGoal:
    """Phase 1 foundation for goal suspension (audit A4 / Eval C2).

    This is a *data structure only*: it records everything needed to
    resume an interrupted goal later. Full interruption / resumption
    behaviour is Phase 2 — do not claim it is complete merely because
    this record exists. ``plan`` holds the *remaining* plan steps as
    plain action dicts; ``plan_position`` is the number of steps already
    executed from the original plan.
    """
    goal_id: int
    original_status: str
    plan: list = field(default_factory=list)      # list of Action dicts
    plan_position: int = 0
    # Phase 3: the world-snapshot history list is gone (Brain.history was
    # removed; the event log is the only history). A suspension record
    # keeps a copy of the event log at suspension time instead, so the
    # context of the interruption is still inspectable after a load.
    events: list = field(default_factory=list)    # list of event dicts
    goal_start: dict = field(default_factory=dict)
    reason: str = ""
    suspended_cycle: int = 0
    # Shallow copy of the CognitiveState entries that were preconditions of
    # the next planned action at suspension time, so resumption can check
    # whether the world changed while suspended (Phase 2, design §3.2).
    # Values are plain JSON values.
    world_snapshot: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SuspendedGoal":
        if not isinstance(d, dict) or "goal_id" not in d:
            raise ValueError("suspended-goal record needs 'goal_id'")
        # Legacy Phase 1–2 records carried world-snapshot strings under
        # "history": keep them readable as snapshot events.
        events = list(d.get("events", []))
        if not events and "history" in d:
            events = [{"cycle": int(d.get("suspended_cycle", 0)), "kind": "world_snapshot",
                       "detail": str(s)} for s in d.get("history", [])]
        return cls(goal_id=int(d["goal_id"]), original_status=str(d.get("original_status", "active")),
                   plan=list(d.get("plan", [])), plan_position=int(d.get("plan_position", 0)),
                   events=events, goal_start=dict(d.get("goal_start", {})),
                   reason=str(d.get("reason", "")), suspended_cycle=int(d.get("suspended_cycle", 0)),
                   world_snapshot=dict(d.get("world_snapshot", {})))


class GoalManager:
    def __init__(self):
        self._goals: dict[int, Goal] = {}
        self._suspended: dict[int, SuspendedGoal] = {}
        # Phase 12 (P12-O5): mutation counter for Brain.world's
        # version-gated cache. Bumped on every mutator (runtime-only,
        # never persisted); a fresh object starts at 0.
        self.version = 0

    # ---- the plan lives in the goal record (Phase 3) ---------------------
    def set_plan(self, gid: int, steps: list) -> None:
        """Store ``steps`` (Action objects or plain dicts) as the goal's
        plan. This is the ONLY place a plan is stored."""
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        from .planner import Action  # local import: planner imports nothing from goals
        self._goals[gid].plan = [asdict(a) if isinstance(a, Action) else dict(a) for a in steps]

    def get_plan(self, gid: int) -> list:
        """The goal's plan as Action objects (empty list when none)."""
        from .planner import Action
        return [Action(x["name"], dict(x.get("pre", {})), dict(x.get("eff", {})),
                       float(x.get("cost", 1.0)), dict(x.get("args", {})),
                       x.get("tool"), x.get("store_as"))
                for x in self._goals[gid].plan]

    # ---- suspension (Phase 1: structure + serialization only) ------------
    def suspend(self, goal: Goal, plan: list, plan_position: int, events: list,
                goal_start: dict, reason: str, cycle: int = 0,
                *, world_snapshot: dict | None = None) -> SuspendedGoal:
        """Record an interrupted goal. The goal itself returns to ``pending``;
        its original status, remaining plan, position, event log at
        suspension time and start state are preserved in the record.
        Resumption is Phase 2. (Phase 3: the ``events`` argument replaces
        the old world-snapshot ``history`` list.)"""
        from .planner import Action  # local import: planner imports nothing from goals
        rec = SuspendedGoal(goal_id=goal.id, original_status=goal.status,
                            plan=[asdict(a) if isinstance(a, Action) else dict(a) for a in plan],
                            plan_position=int(plan_position), events=[dict(e) for e in events],
                            goal_start=dict(goal_start), reason=reason,
                            suspended_cycle=int(cycle),
                            world_snapshot=dict(world_snapshot) if world_snapshot is not None else {})
        self._suspended[goal.id] = rec
        goal.status = "pending"
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        return rec

    def suspended(self, gid: int) -> SuspendedGoal | None:
        return self._suspended.get(gid)

    def all_suspended(self) -> list[SuspendedGoal]:
        return list(self._suspended.values())

    def restore_suspended(self, gid: int) -> SuspendedGoal | None:
        """Return the suspension record for a goal (does not reactivate it;
        activation / resumption is Phase 2)."""
        return self._suspended.get(gid)

    def unsuspend(self, gid: int) -> SuspendedGoal | None:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        """Remove the suspension record for a goal and return it
        (None if absent). Unlike ``restore_suspended`` (peek), this
        consumes the record — the caller owns resumption from here."""
        return self._suspended.pop(gid, None)

    def add(self, goal: Goal) -> Goal:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        self._goals[goal.id] = goal
        return goal

    def get(self, gid: int) -> Goal:
        return self._goals[gid]

    def all(self) -> list[Goal]:
        return list(self._goals.values())

    def pending(self) -> list[Goal]:
        return [g for g in self._goals.values() if g.status == "pending"]

    def active(self) -> Goal | None:
        for g in self._goals.values():
            if g.status == "active":
                return g
        return None

    def activate(self, gid: int) -> None:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        cur = self.active()
        if cur and cur.id != gid:
            cur.status = "pending"
        self._goals[gid].status = "active"

    def complete(self, gid: int) -> None:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        self._goals[gid].status = "done"

    def fail(self, gid: int, reason: str) -> None:
        self.version += 1  # Phase 12 (P12-O5): world-cache invalidation
        g = self._goals[gid]
        g.status, g.fail_reason = "failed", reason

    def decompose(self, parent_id: int, subgoals: list[Goal]) -> list[Goal]:
        for sg in subgoals:
            sg.parent = parent_id
            self.add(sg)
        return subgoals

    @staticmethod
    def satisfied(goal: Goal, state: dict) -> bool:
        return all(state.get(k) == v for k, v in goal.target.items())

    # ---- persistence (Phase 1) -------------------------------------------
    def to_json(self) -> dict:
        return {"goals": [asdict(g) for g in self._goals.values()],
                "suspended": [s.to_json() for s in self._suspended.values()]}

    @classmethod
    def from_json(cls, data: dict) -> "GoalManager":
        if not isinstance(data, dict) or "goals" not in data:
            raise ValueError("goals file has no 'goals' list")
        gm = cls()
        for d in data["goals"]:
            g = Goal(d["description"], dict(d["target"]), float(d.get("priority", 0.5)),
                     d.get("deadline_cycle"), d.get("parent"),
                     str(d.get("status", "pending")), dict(d.get("meta", {})),
                     int(d["id"]), str(d.get("fail_reason", "")),
                     list(d.get("plan", [])))
            gm._goals[g.id] = g
        if gm._goals:
            _reserve_goal_id(max(gm._goals))
        for d in data.get("suspended", []):
            s = SuspendedGoal.from_json(d)
            gm._suspended[s.goal_id] = s
        return gm
