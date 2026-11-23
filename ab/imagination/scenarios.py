"""Scenario generation and comparison over simulated futures (Phase 8, P8-5).

``ScenarioGenerator`` turns a goal into two or more distinct imagined
routes — candidate plans from ``Planner.plan_all`` (plus, when that
yields only one route, replans that ban the first route's opening move
and, as a last resort, orderings of the plan's steps) — each simulated
with the ``Simulator`` so every scenario carries its own predicted
future state. ``ScenarioComparator`` ranks scenarios by predicted
success, then predicted cost, then risk, and records the ranking with
a reason per scenario — never a bare choice.

Deterministic, stdlib only, CPU-first, no LLM.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field


@dataclass
class Scenario:
    """One imagined route: its steps and its simulated future."""

    name: str
    steps: list = field(default_factory=list)       # action names
    goal: dict = field(default_factory=dict)
    predicted_state: dict = field(default_factory=dict)
    per_step: list = field(default_factory=list)
    predicted_success: float = 1.0
    predicted_cost: float = 0.0
    risk: float = 0.0
    assumptions: list = field(default_factory=list)
    met_goal: bool = False
    violations: list = field(default_factory=list)
    # Phase 9 §4.3: True when this route is imaginable but uses a
    # pair the planner bans for execution (a learned (action,
    # situation) avoidance). A banned scenario is shown, never
    # selected for execution — see Brain.imagine.
    banned: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": list(self.steps),
            "goal": dict(self.goal),
            "predicted_state": dict(self.predicted_state),
            "per_step": [dict(p) for p in self.per_step],
            "predicted_success": float(self.predicted_success),
            "predicted_cost": float(self.predicted_cost),
            "risk": float(self.risk),
            "assumptions": list(self.assumptions),
            "met_goal": bool(self.met_goal),
            "violations": list(self.violations),
            "banned": bool(self.banned),
            "simulated": True,
        }


def _freeze(state: dict) -> str:
    return json.dumps(state, sort_keys=True, default=str)


class ScenarioGenerator:
    """Generate distinct simulated scenarios for a goal.

    ``planner`` supplies candidate plans; ``simulator`` simulates
    them. Bounds (design P8-6): at most ``max_scenarios`` scenarios,
    at most ``max_steps`` simulated steps each.
    """

    def __init__(self, planner, simulator, max_scenarios: int = 3,
                 max_steps: int = 20, include_banned: bool = False):
        self.planner = planner
        self.simulator = simulator
        self.max_scenarios = int(max_scenarios)
        self.max_steps = int(max_steps)
        # Phase 9 §4.3: when True, routes the planner bans for
        # execution are ALSO surfaced, flagged ``banned`` (never
        # selected — see generate). Default False: imagination lists
        # executable routes only, as before.
        self.include_banned = bool(include_banned)

    # ---- candidate routes ----------------------------------------------

    def _plan_names(self, state, goal, banned=None, limit=4,
                    allow_banned_pairs=False):
        try:
            plans = self.planner.plan_all(dict(state), dict(goal),
                                          banned=set(banned or ()), limit=limit,
                                          allow_banned_pairs=allow_banned_pairs)
        except Exception:
            return []
        out = []
        for p in (plans or []):
            steps = [getattr(a, "name", str(a)) for a in (p.steps or [])]
            out.append([str(s) for s in steps])
        return out

    def _sequence_is_banned(self, seq, state) -> bool:
        """True when any step of ``seq`` (action names) is banned in
        the situation it would run in, per the planner's banned
        pairs. The check simulates the state forward through the
        steps' effects, exactly like Planner.sequence_banned."""
        banned_pairs = getattr(self.planner, "banned_pairs", None)
        if not banned_pairs:
            return False
        by_name = {}
        for a in (getattr(self.planner, "actions", None) or []):
            by_name.setdefault(getattr(a, "name", str(a)), a)
        steps = []
        for name in seq:
            action = by_name.get(name)
            if action is None:
                return False  # unknown step: cannot verify a ban
            steps.append({"name": getattr(action, "name", str(action)),
                          "pre": dict(getattr(action, "pre", {}) or {}),
                          "eff": dict(getattr(action, "eff", {}) or {}),
                          # Phase 10.5: carry the step's args (and
                          # tool, so the redactor can apply the right
                          # spec), so an args-aware ban is visible to
                          # this check — a ban on one variant must
                          # flag only sequences running that variant.
                          "args": dict(getattr(action, "args", {}) or {}),
                          "tool": getattr(action, "tool", None)})
        try:
            from ..planning.planner import Planner
            return bool(Planner.sequence_banned(
                steps, dict(state), banned_pairs,
                redact_args=getattr(self.planner, "redact_args", None)))
        except Exception:
            return False

    def _candidate_sequences(self, state, goal):
        """Distinct candidate step sequences, in planner order:
        plan_all variants first, then replans that ban the first
        route's opening move, then orderings of the first plan.
        Returns ``[(sequence, banned_flag)]`` — with
        ``include_banned`` a final relaxed pass (``allow_banned_pairs``)
        adds routes the planner bans for execution, flagged True and
        only when the ban actually applies to that sequence (verified
        with ``sequence_banned``; an unbanned route the relaxed pass
        happens to surface is not mislabelled banned)."""
        seen: set[tuple] = set()
        candidates: list[tuple[list[str], bool]] = []

        def _add(seq, banned=False):
            key = tuple(seq)
            if key and key not in seen:
                seen.add(key)
                candidates.append((list(seq), bool(banned)))

        for seq in self._plan_names(state, goal, limit=4):
            _add(seq)
        # A second route the planner ranked below the first: ban the
        # first route's opening move and replan.
        if candidates:
            for seq in self._plan_names(state, goal,
                                        banned={candidates[0][0][0]}, limit=2):
                _add(seq)
        # Last resort: orderings of the first plan's steps (bounded,
        # deterministic). Only orderings that still reach the goal
        # survive the simulation filter below.
        if len(candidates) < 2 and candidates:
            base = candidates[0][0]
            if 1 < len(base) <= 5:
                perms = sorted(set(itertools.permutations(base)))[:12]
            elif len(base) > 5:
                perms = []
                for i in range(len(base) - 1):
                    swapped = list(base)
                    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                    perms.append(tuple(swapped))
            else:
                perms = []
            for perm in perms:
                _add(list(perm))
        # Phase 9 §4.3: the pair ban prunes banned routes from every
        # strict plan_all call above (the generator never sees them),
        # so a relaxed pass is the only way to surface them. Only
        # sequences the ban actually applies to are flagged banned.
        if self.include_banned:
            for seq in self._plan_names(state, goal, limit=4,
                                        allow_banned_pairs=True):
                if tuple(seq) in seen:
                    continue
                if self._sequence_is_banned(seq, state):
                    _add(seq, banned=True)
        return candidates

    # ---- generation ------------------------------------------------------

    def generate(self, goal, start_state, banned=None) -> list[Scenario]:
        """Simulate candidate routes for ``goal`` from ``start_state``.
        Returns at most ``max_scenarios`` scenarios, each with a
        distinct predicted future state, in planner order (the
        comparator, not the generator, ranks them). A route whose
        simulation does not reach the goal is dropped, never kept as
        a scenario that pretends to succeed."""
        goal = dict(goal or {})
        start = dict(start_state or {})
        scenarios: list[Scenario] = []
        seen_states: set[str] = set()
        for seq, is_banned in self._candidate_sequences(start, goal):
            if len(scenarios) >= self.max_scenarios:
                break
            result = self.simulator.simulate(seq, start, goal=goal,
                                             max_steps=self.max_steps)
            if not result.met_goal:
                continue
            frozen = _freeze(result.predicted_state)
            if frozen in seen_states:
                continue
            seen_states.add(frozen)
            scenarios.append(Scenario(
                name=f"route-{len(scenarios) + 1}",
                steps=list(seq), goal=goal,
                predicted_state=dict(result.predicted_state),
                per_step=[dict(p) for p in result.per_step],
                predicted_success=float(result.predicted_success),
                predicted_cost=float(result.predicted_cost),
                risk=float(result.risk),
                assumptions=list(result.assumptions),
                met_goal=True,
                violations=list(result.violations),
                banned=bool(is_banned)))
        return scenarios


class ScenarioComparator:
    """Rank scenarios by predicted success, then predicted cost, then
    risk. Every ranked entry carries a reason naming the numbers it
    was ranked on — never a bare choice."""

    def __init__(self):
        self.last_ranking: list[dict] = []

    def rank(self, scenarios) -> list[dict]:
        """Returns ``[{scenario, rank, reason, predicted_success,
        predicted_cost, risk}, ...]`` in ranked order (best first).
        Ties on success fall to lower cost, then lower risk, then
        name, so the order is deterministic."""
        ordered = sorted(
            list(scenarios or []),
            key=lambda s: (-float(s.predicted_success),
                           float(s.predicted_cost),
                           float(s.risk), str(s.name)))
        ranking = []
        best = ordered[0] if ordered else None
        for i, sc in enumerate(ordered):
            if i == 0:
                reason = (f"rank 1: highest predicted success "
                          f"{sc.predicted_success:.2f}; predicted cost "
                          f"{sc.predicted_cost:.2f}, risk {sc.risk:.2f}")
            elif best is not None and abs(sc.predicted_success - best.predicted_success) < 1e-9:
                reason = (f"rank {i + 1}: same predicted success as best "
                          f"({sc.predicted_success:.2f}) but higher predicted "
                          f"cost {sc.predicted_cost:.2f} vs {best.predicted_cost:.2f}")
            else:
                reason = (f"rank {i + 1}: lower predicted success "
                          f"{sc.predicted_success:.2f} vs best "
                          f"{best.predicted_success:.2f}; predicted cost "
                          f"{sc.predicted_cost:.2f}, risk {sc.risk:.2f}")
            ranking.append({
                "scenario": sc,
                "rank": i + 1,
                "reason": reason,
                "predicted_success": float(sc.predicted_success),
                "predicted_cost": float(sc.predicted_cost),
                "risk": float(sc.risk),
                "steps": list(sc.steps),
                "name": sc.name,
            })
        self.last_ranking = ranking
        return ranking

