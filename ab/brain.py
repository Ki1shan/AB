"""AB: the brain. One cycle = sense -> attend -> control -> reason -> act -> evaluate -> learn."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import time
from collections import Counter
from urllib.parse import urlparse

from .body.ocr import OcrEngine
from .body.providers import provider_from_spec
from .body.research import ClaimExtractor, Researcher, research_fetch
from .body.tools import Tool, ToolRegistry, ToolSpec
from .perception.bus import PerceptionBus
from .execution import (DEFAULT_MAX_TOOL_CALLS, DEFAULT_MAX_WALL_S,
                        ExecutionMonitor)
from .core.attention import Attention
from .core.control import CognitiveControl
from .core.learning import Episode, Learner
from .core.memory import Answer, Fact, LongTermMemory
from .core.state import CognitiveState
from .core.thought import Thought
from .core.vocab import (FUNCTIONAL_RELATIONS, normalize_relation,
                         normalize_text, tokenize)
from .core.working_memory import FOCUS_K, WorkingMemory
from .planning.evaluator import Evaluator
from .planning.goals import Goal, GoalManager, SuspendedGoal
from .planning.hierarchy import HierarchicalPlanner
from .planning.planner import Action, Planner, action_identity, canonical_args, canonical_variant
from .reasoning.causal import CausalModel
from .reasoning.decision import (FEASIBILITY_MIN, Decision, DecisionMaker,
                                Option, feasibility_for_cost)
from .reasoning.engine import Contradiction, ReasoningEngine
from .reasoning.hypothesis import Hypothesis
from .reasoning.inference import Inference, Proof
from .reasoning.rules import Rule, RuleEngine

# Phase 5: the world model and pattern library (ab/world, Worker 1).
# Imported here so a missing/broken ab/world fails loudly at Brain
# construction with a clear message, never silently disables the
# model — the brain owns exactly one WorldModel / PatternLibrary.
try:
    from .world.model import EpistemicStatus, WorldModel
    from .world.patterns import AbstractPattern, PatternLibrary
    _WORLD_IMPORT_ERROR: Exception | None = None
except ImportError as _world_import_exc:  # pragma: no cover - Worker 1 landed
    EpistemicStatus = None  # type: ignore[assignment]
    WorldModel = None  # type: ignore[assignment]
    AbstractPattern = None  # type: ignore[assignment]
    PatternLibrary = None  # type: ignore[assignment]
    _WORLD_IMPORT_ERROR = _world_import_exc

# Phase 6: the learning subsystem (ab/learning, Worker 1). Imported
# here so a missing/broken ab/learning fails loudly at Brain
# construction with a clear message, never silently disables
# learning — the brain owns exactly one of each component.
try:
    from .learning import (CalibrationTracker, ConceptStore, Consolidator,
                           Prioritizer, Retriever)
    _LEARNING_IMPORT_ERROR: Exception | None = None
except ImportError as _learning_import_exc:  # pragma: no cover - Worker 1 landed
    CalibrationTracker = None  # type: ignore[assignment]
    ConceptStore = None  # type: ignore[assignment]
    Consolidator = None  # type: ignore[assignment]
    Prioritizer = None  # type: ignore[assignment]
    Retriever = None  # type: ignore[assignment]
    _LEARNING_IMPORT_ERROR = _learning_import_exc

# Phase 8: imagination (ab/imagination, Worker 2). Imported here so a
# missing/broken imagination package fails loudly at Brain construction
# with a clear message, never silently disables simulation — the brain
# owns exactly one Simulator / SimulationRevision.
try:
    from .imagination import (
        ScenarioComparator,
        ScenarioGenerator,
        SimulationRevision,
        Simulator,
    )
    _IMAGINATION_IMPORT_ERROR: Exception | None = None
except ImportError as _imagination_import_exc:  # pragma: no cover - Worker 2 landed
    ScenarioComparator = None  # type: ignore[assignment]
    ScenarioGenerator = None  # type: ignore[assignment]
    SimulationRevision = None  # type: ignore[assignment]
    Simulator = None  # type: ignore[assignment]
    _IMAGINATION_IMPORT_ERROR = _imagination_import_exc

# Phase 3 §3.5 (interface contract with the tools worker, who owns
# ab/core/audit.py): _event redacts secrets via audit.redact_secrets
# and save prunes events to audit.MAX_EVENTS. audit.py has landed in
# this checkout, so this is a direct import (the defensive fallback
# was removed once the module existed, per the task).
from .core.audit import MAX_EVENTS, redact_secrets


def default_rules() -> list[dict]:
    t = lambda k, v: {"key": k, "op": "==", "value": v}          # noqa: E731
    return [
        {"id": "complete_when_satisfied", "priority": 1.0, "source": "builtin",
         "when": [t("has_goal", True), t("goal_satisfied", True)], "then": [{"do": "complete_goal"}]},
        {"id": "give_up_when_stalled", "priority": 0.99, "source": "builtin",
         "when": [t("has_goal", True), t("stalled", True)], "then": [{"do": "fail_goal", "args": {"reason": "stalled"}}]},
        {"id": "give_up_no_plan", "priority": 0.95, "source": "builtin",
         "when": [t("has_goal", True), t("plan_failed", True)], "then": [{"do": "fail_goal", "args": {"reason": "no plan found"}}]},
        {"id": "replan_on_surprise", "priority": 0.9, "source": "builtin",
         "when": [t("has_goal", True), t("surprise", True), t("goal_satisfied", False)], "then": [{"do": "make_plan"}]},
        {"id": "plan_when_none", "priority": 0.8, "source": "builtin",
         "when": [t("has_goal", True), t("has_plan", False), t("goal_satisfied", False)], "then": [{"do": "make_plan"}]},
        {"id": "execute_plan", "priority": 0.7, "source": "builtin",
         "when": [t("has_plan", True), t("surprise", False)], "then": [{"do": "execute_step"}]},
        {"id": "select_goal", "priority": 0.5, "source": "builtin",
         "when": [t("has_goal", False)], "then": [{"do": "select_goal"}]},
    ]


class Brain:
    # Phase 7 (D7-8): after this many failed attempts at the same step
    # with the same strategy, the goal stops with reason
    # "strategy_exhausted" and its partial results preserved — a
    # documented, observable stop instead of a silent retry loop.
    MAX_SAME_STRATEGY_RETRIES = 2

    # Phase 9 §4.2: contradicting observations of one promoted
    # pattern's prediction needed to retire it automatically (see
    # record_pattern_contradiction).
    PATTERN_CONTRADICTION_THRESHOLD = 3

    def __init__(self, actions: list[Action] | None = None, researcher: Researcher | None = None,
                 ocr: OcrEngine | None = None, granted=("compute",),
                 wm_capacity: int = 7,
                 max_cycles: int = 200, ocr_min_confidence: float = 0.6,
                 fs_scope: str | None = None, specialist=None,
                 causal_model=None):
        # Phase 2 decision (resolves Phase 1 blocker #3 / ADR-2): the
        # default grant set is now ("compute",) only. Environment actions
        # require an explicit "environment" grant — callers that need to
        # act on the world (demo, RoomWorld tests) pass
        # granted={"compute", "environment"} explicitly. A brain built
        # with the default grants denies environment actions; see
        # _do_execute_step. That is the authorization boundary, and it
        # is tested (tests/test_phase1_security.py).
        self.cycle = 0
        self.state = CognitiveState()
        self.wm = WorkingMemory(wm_capacity)
        self.attention = Attention()
        self.control = CognitiveControl(max_cycles=max_cycles)
        self.ltm = LongTermMemory()
        self.inference = Inference(self.ltm)
        self.rules = RuleEngine()
        for r in default_rules():
            self.rules.add(r)
        # Phase 4: the unified reasoning engine (deduction / induction /
        # abduction / analogy / simulation / contradiction detection) and
        # the causal model. The engine reasons over the same Inference
        # and RuleEngine the brain already owns; the causal model is
        # attached to the engine so abduction can use its claims.
        # Phase 10 (P10-5): the causal model is injectable at
        # construction (``causal_model=``) so the no_causal ablation
        # can build a brain with NoOpCausal from the start -- the
        # same object is handed to the engine and the simulator
        # below, so no stale references exist. Default (None) is the
        # real CausalModel, exactly as before.
        self.reasoning = ReasoningEngine(self.inference, self.rules)
        self.causal = (causal_model if causal_model is not None
                       else CausalModel())
        self.reasoning.causal = self.causal
        self.decider = DecisionMaker({"priority": 0.4, "urgency": 0.2, "feasibility": 0.4})
        self.goals = GoalManager()
        # Phase 12 (P12-O5): version-gated cache for the ``world``
        # property (see the property). ``_world_cache`` holds
        # (version-key, built view); ``_world_token`` is bumped by
        # Brain-side invalidations the store versions cannot see
        # (load() replacing whole stores, _knowledge_keys changes).
        self._world_cache: tuple | None = None
        self._world_token = 0
        self.planner = Planner(actions)
        self.evaluator = Evaluator()
        self.learner = Learner(self.planner, self.rules)
        # Phase 5: the brain owns ONE world model and ONE pattern
        # library. The model is the structured view over the same
        # observations the shared CognitiveState holds — never a
        # parallel store that can diverge. The library is attached to
        # the planner as a cost prior (see Planner.attach_patterns).
        if _WORLD_IMPORT_ERROR is not None:
            raise ImportError(
                "ab/world is required by Brain (Phase 5) but could not "
                f"be imported: {_WORLD_IMPORT_ERROR}")
        self.world_model = WorldModel()
        self.patterns = PatternLibrary()
        self.planner.attach_patterns(
            self.patterns,
            lambda: (self.world_model, self.planner.actions, dict(self.world)))
        # Phase 6: the brain owns ONE of each learning component.
        # The pattern context carries the current state as its third
        # element so P-D can check its required key (planner
        # pattern_multiplier accepts a 2- or 3-tuple context).
        if _LEARNING_IMPORT_ERROR is not None:
            raise ImportError(
                "ab/learning is required by Brain (Phase 6) but could "
                f"not be imported: {_LEARNING_IMPORT_ERROR}")
        self.consolidator = Consolidator()
        self.concepts = ConceptStore()
        self.calibration = CalibrationTracker()
        self.retriever = Retriever()
        self.prioritizer = Prioritizer()
        self.planner.attach_concepts(self.concepts)
        # Phase 8 (P8-5/P8-6): imagination — simulation over a copy of
        # the world model, scenario comparison, and the revision loop
        # that feeds checked outcomes back into predicted success.
        # The simulator is rebuilt lazily by _get_simulator() when the
        # world model is replaced (e.g. by load()); see there.
        if _IMAGINATION_IMPORT_ERROR is not None:
            raise ImportError(
                "ab/imagination is required by Brain (Phase 8) but could "
                f"not be imported: {_IMAGINATION_IMPORT_ERROR}")
        self.simulation_revision = SimulationRevision()
        self.simulator = Simulator(self.world_model, causal=self.causal,
                                   planner=self.planner,
                                   revision=self.simulation_revision)
        self.last_imagination = None            # last imagine() result dict
        self.last_simulation_comparison = None  # last _rank_plans_by_simulation record
        self._last_imagined = None              # best imagined scenario (steps + predicted state)
        # Phase 6 cycle state: one-cycle alternative preference from
        # a repeated mistake, the failed action it names, the
        # (action, feature-set) pairs already reported, pattern ids
        # already announced, and the retriever's episode list (the
        # in-memory mirror of the learner's episodes).
        self._prefer_alternative = False
        self._repeated_mistake_action: str | None = None
        self._repeated_mistake_seen: set = set()
        self._announced_patterns: set = set()
        self._retriever_episodes: list = []
        # Phase 6 concept-learning dedupe: (tool, broke) pairs C-1
        # already recorded and (action, key) pairs C-2 already
        # recorded, so the per-cycle concept pass never re-adds the
        # same example.
        self._c1_seen: set = set()
        self._c2_seen: set = set()
        # Phase 3: the planner's stats are a derived view over this
        # learner's episodes, and its plan_all consults the RuleEngine's
        # ACTIVE avoid_action rules (never Learner internals).
        self.planner.attach_learner(self.learner)
        self.planner.attach_rules(self.rules)
        # Phase 6: every recorded episode also feeds the retriever
        # index and the prioritizer (duck-typed hook on the learner;
        # ab/core/learning.py never imports ab.learning).
        self.learner.on_record = self._on_episode_recorded
        self.tools = ToolRegistry()
        self.granted = set(granted)
        self.researcher, self.ocr = researcher, ocr
        # Phase 8 (P8-1/P8-4): one perception bus — every source goes
        # through it. ``specialist`` is an optional replaceable
        # perception model (absent by default); its output arrives as
        # a percept with provenance, never as reasoning.
        self.specialist = specialist
        self.perception_bus = PerceptionBus(self)
        self.extractor = researcher.extractor if researcher else ClaimExtractor()
        self.ocr_min_confidence = ocr_min_confidence
        if researcher:
            researcher.ltm = self.ltm                      # research always writes to THIS brain's memory
            self._register_research_tool()
        # Phase 7 (D7-1/D7-2): the execution subsystem — one monitor
        # (session + ledger) for the active goal, and the hierarchical
        # planner for multi-key goals. The monitor is checked from
        # step() before the rule fires and after each step.
        self.execution_monitor = ExecutionMonitor()
        # Coordinator integration fix (Phase 7 gate): the session's
        # plan snapshot must honour ToolSpec ``secret`` markings (the
        # registry audit already does; the snapshot did not). The
        # monitor/session use this bound redactor whenever they build
        # a snapshot from live plan steps.
        self.execution_monitor.redact_args = self._redact_tool_args
        # Phase 10.5: the planner derives ban keys and failure-rate
        # identities from action args — it gets the same redactor, so
        # those identities are computed on redacted args and no
        # persisted record (learner.json, plan.json) can leak a
        # secret-marked tool argument. Load re-sets this below (the
        # planner object itself survives load, but the assignment is
        # kept next to the monitor's for the same reason).
        self.planner.redact_args = self._redact_tool_args
        self.hierarchy = HierarchicalPlanner(self.goals)
        self._current_rule_id: str | None = None
        self._executing_step = False
        # Phase 7 strategy selection: the strategy of the last failed
        # step (a tool name, or a pattern-library name when the pattern
        # library names it), the consecutive-failure streak for it, and
        # the per-(action, strategy) failure counts (inspection).
        self._failed_strategy: str | None = None
        self._strategy_streak: dict = {"strategy": None, "action": None, "count": 0}
        self._strategy_failures: dict = {}
        # Phase 7 §4.2 (prioritized-pruning fix): pattern ids whose
        # evidence episodes have already had their usefulness marked
        # (+1 each), so a pattern learned across cycles is marked once.
        self._pattern_evidence_marked: set = set()
        # Phase 9 §4.2: contradicting observations per promoted
        # pattern id — reaching PATTERN_CONTRADICTION_THRESHOLD
        # retires the pattern (see record_pattern_contradiction).
        # Phase 10 (P10-8): the counts persist in revision.json
        # under "pattern_contradictions" (see save/load); the
        # retirement itself also persists via the pattern library
        # (world.json).
        self._pattern_contradictions: dict[str, list] = {}
        # Phase 10 (P10-7): need_percept requests already acted on,
        # keyed (subject, relation, source-kind, source) — so a
        # standing need_more_evidence refusal reads a document /
        # image once instead of re-reading it every cycle. Transient
        # (not persisted): after a load a document may be read once
        # more; the reader dedupes it as a duplicate claim.
        self._percept_requests_acted: set = set()
        # Phase 7 permission filtering: (tool, missing grant) pairs
        # rejected by the last _do_make_plan pass (decides the
        # no_permitted_plan failure); reset at each planning pass.
        self._last_permission_denials: list = []
        # Phase 7 (D7-7): filesystem scope. No file tool is registered
        # unless fs_scope is given; the scope is enforced by
        # ScopedFilesystem on realpaths (symlink-aware). The import is
        # guarded: a brain without fs_scope never touches the sandbox
        # module, and passing fs_scope without it raises a clear error
        # instead of silently running unscoped.
        self.fs_scope = fs_scope
        self._filesystem = None
        if fs_scope is not None:
            try:
                from .body.sandbox import (ScopedFilesystem, scoped_read_file,
                                           scoped_write_file)
            except ImportError as exc:
                raise ImportError(
                    "Brain(fs_scope=...) requires "
                    "ab.body.sandbox.ScopedFilesystem (Phase 7 Worker 1: "
                    "the isolation/sandbox module) but it could not be "
                    f"imported: {exc}") from exc
            self._filesystem = ScopedFilesystem(fs_scope)
            self.tools.register(ToolSpec(
                name="read_file",
                args=[{"name": "path", "type": "str", "required": True}],
                return_type="str", timeout=5.0, required_grant="filesystem",
                version=1, description="Read a text file inside the filesystem scope"),
                functools.partial(scoped_read_file, self._filesystem))
            self.tools.register(ToolSpec(
                name="write_file",
                args=[{"name": "path", "type": "str", "required": True},
                      {"name": "data", "type": "str", "required": True}],
                return_type="dict", timeout=5.0, required_grant="filesystem",
                version=1, description="Write a text file inside the filesystem scope"),
                functools.partial(scoped_write_file, self._filesystem))
        # Phase 3: Brain.plan is a read-only property over the active
        # goal's record (GoalManager) — there is no separate Brain.plan
        # store anymore. Brain.history is REMOVED: the event log is the
        # only history (suspension records keep an event-log copy).
        self._plan_started_len = 0      # steps in the plan when it was made
        self.plan_failed = False
        self.surprise = False
        # Phase 2: alternative plan retained from the last make_plan
        # (next-best distinct-first-action plan, if one exists).
        self.alternative_plan: list[Action] = []
        self.alternative_plan_cost: float = 0.0
        # Phase 2: last decision (persisted), terminal stop reason, and
        # progress tracking for the active goal. _steps_executed counts
        # only steps whose effects materialised (no surprise) and is
        # monotonic per goal — it is NOT reset by a replan, so
        # re-executing a failing step can never masquerade as progress
        # (and a recovery that takes a few cycles cannot look like a
        # stall either: the count only grows on real, sticky effects).
        self.last_decision: Decision | None = None
        self.last_stop_reason: str | None = None
        self._steps_executed = 0
        self._tie_deferred_cycle: int | None = None
        self._resuming = False
        self._queued_interrupts: list[dict] = []
        self.goal_start: dict = {}
        self.last_evaluation = None
        self.last_inferences: list[dict] = []   # derivations from the last cycle's WM read
        # Phase 4: the engine's last-cycle outputs, kept for inspection
        # and for the decision-confidence / contradiction paths.
        self.last_hypotheses: list[dict] = []
        self.last_contradictions: list[dict] = []
        self.last_considered_rules: list[str] = []  # ALL rules matched in the last cycle
        self._last_violations: list[dict] = []  # prediction violations awaiting abduction
        self.load_warnings: list[str] = []      # non-fatal problems found by load()
        self.events: list[dict] = []
        self._incoming: list[Thought] = []
        self._goal_thoughts: dict[int, Thought] = {}
        self._knowledge_keys: dict[str, tuple] = {}
        self._prev_obs: dict = {}
        # Phase 5: True once an environment action has executed since
        # _prev_obs was last refreshed (see _sense /
        # _do_execute_step). A key changing while this is False is an
        # exogenous change, not an action effect.
        self._acted_since_sense = False

    # ---- shared typed state (Phase 1) ------------------------------------
    @property
    def world(self) -> dict:
        """Compatibility view: plain values of the shared cognitive state.
        Read-only in the sense that callers must not mutate it; writes go
        through ``state.set`` (see ``_set_state``).

        Phase 3: this view also carries the DERIVED ``known:*`` entries.
        They are computed on read from LTM + inference — nothing is
        stored under a ``known:*`` key anywhere. Derived keys come
        from (a) registered knowledge keys, (b) ``known:*`` goal
        targets, and (c) every verified LTM fact.

        Phase 12 (P12-O5): the built view is cached behind a version
        key ``(state.version, ltm.version, goals.version,
        _world_token)`` — every mutation of the three stores bumps
        its version (see their ``version`` counters), and Brain-side
        invalidations (load(), _knowledge_keys changes) bump the
        token, so a cache hit means the inputs are byte-identical
        to the build's inputs and the returned view is identical.
        Callers always get a fresh copy of the cached view, so no
        caller can mutate the cached dict (no caller mutated the
        returned view before, either — the copy just makes that
        structural). The cache holds ONE view; memory is bounded."""
        key = (self.state.version, self.ltm.version,
               self.goals.version, self._world_token)
        cached = self._world_cache
        if cached is not None and cached[0] == key:
            return dict(cached[1])
        view = self.state.as_dict()
        derived: dict = {}
        for key2, (s, r) in self._knowledge_keys.items():
            derived[key2] = self.known(s, r)
        for g in self.goals.all():
            for k in g.target:
                m = re.match(r"known:(.+)\|(.+)", k)
                if m:
                    derived[k] = self.known(m.group(1), m.group(2))
        for f in self.ltm.all_facts():
            if f.status == "verified":
                derived[f"known:{f.subject}|{f.relation}"] = True
        view.update(derived)
        self._world_cache = (key, view)
        return dict(view)

    @world.setter
    def world(self, values: dict) -> None:
        # Used by tests and by v0.1 callers that replace the whole world.
        # Each entry becomes a state entry sourced from "external".
        # ``known:*`` keys are skipped: they are a derived view, not
        # something a caller can store (Phase 3 §2.1).
        self.state.clear()
        for k, v in dict(values).items():
            if str(k).startswith("known:"):
                continue
            self.state.set(k, v, source="external", confidence=1.0, cycle=self.cycle)

    def known(self, subject, relation) -> bool:
        """Derived view (Phase 3 §2.1): is (subject, relation) known?

        Computed on read, never stored: LTM verified facts first, then
        inferred state entries (``inferred:<subject>|<relation>``). No
        write in the codebase creates a ``known:*`` state entry; goal
        targets keep the ``known:subject|relation`` string only as a
        target-key format."""
        subject = normalize_text(subject)
        relation = normalize_relation(relation)
        if self.ltm.query(subject, relation, statuses=("verified",)):
            return True
        entry = self.state.get(f"inferred:{subject}|{relation}")
        return (entry is not None and entry.status == "active"
                and entry.value is not None and entry.value is not False)

    # ---- the plan lives in the goal record (Phase 3 §2.2) ----------------
    @property
    def plan(self) -> list[Action]:
        """Read-only view of the active goal's plan, from its goal
        record (GoalManager). There is no separate Brain.plan store;
        assigning to this property raises AttributeError. Writers use
        ``_set_plan``."""
        g = self.goals.active()
        if g is None:
            return []
        return self.goals.get_plan(g.id)

    def _set_plan(self, steps: list[Action], goal: Goal | None = None) -> None:
        """Write the plan into a goal record — the ONLY plan store.
        ``goal`` defaults to the active goal; pass it explicitly when
        the goal has just completed or failed (it is no longer active).
        No-op when no goal owns the plan."""
        g = goal if goal is not None else self.goals.active()
        if g is None:
            return
        self.goals.set_plan(g.id, list(steps))

    def _set_state(self, key, value, source: str, confidence: float = 1.0,
                   provenance: str = "", entry_type: str = "observation",
                   status: str = "active") -> None:
        # Phase 2 decision (resolves Phase 1 blocker #6): a state_change
        # event is logged only when the value changes, when the source
        # *category* changes (the part before ":", e.g. ltm -> inference),
        # or when confidence shifts by >= 0.2. A source flip within one
        # category at the same value is not an event anymore.
        prev = self.state.set(key, value, source, confidence, self.cycle,
                              provenance, entry_type, status=status)
        if prev is None:
            return
        value_changed = prev.value != value
        category_changed = prev.source.split(":")[0] != source.split(":")[0]
        confidence_changed = abs(prev.confidence - confidence) >= 0.2
        if value_changed or category_changed or confidence_changed:
            self._event("state_change",
                        f"{key}: {prev.source} -> {source} (was {prev.value!r}, now {value!r})")

    # ------------------------------------------------------------------ teaching & asking
    def teach_fact(self, subject, relation, value, source: str = "teacher"):
        result = self.ltm.add(subject, relation, value, source, self.cycle)
        # Phase 5: a taught fact also feeds the world model as a told
        # entry. Hierarchical relations (is_a / part_of / located_in)
        # additionally become model relations with status told, so the
        # model's is_a chains cover taught knowledge too (the
        # reasoning wrapper only presents observed/inferred relations
        # to the engine — told relations stay visible via LTM).
        subject = normalize_text(subject)
        relation = normalize_relation(relation)
        self.world_model.teach(subject, relation, value,
                               cycle=self.cycle, source=source)
        if relation in ("is_a", "part_of", "located_in") and isinstance(value, str):
            self.world_model.add_relation(
                subject, relation, value, status=EpistemicStatus.told,
                confidence=1.0, provenance=str(source))
        return result

    def teach_rule(self, rule: dict) -> Rule:
        rule = dict(rule)
        rule.setdefault("source", "teacher")
        return self.rules.add(rule)

    def teach_pattern(self, relation: str, regex: str) -> None:
        self.extractor.add_pattern(relation, regex)

    def ask(self, subject, relation) -> Answer:
        subject = normalize_text(subject)
        relation = normalize_relation(relation)
        # Phase 4: a disputed (subject, relation) is answered with BOTH
        # values and a disputed status — the brain never silently picks
        # one side of a contradiction it has flagged.
        disputed = self.state.get(f"disputed:{subject}|{relation}")
        if disputed is not None:
            vals = disputed.value if isinstance(disputed.value, list) else [disputed.value]
            seen: list = []
            for v in list(vals) + [f.value for f in self.ltm.query(subject, relation)]:
                if v not in seen:
                    seen.append(v)
            ans = [(v, 0.5, "disputed", ["reasoning"]) for v in seen]
            return Answer(True, ans, "Sources disagree; flagged as disputed.")
        direct = self.ltm.ask(subject, relation)
        if direct.known:
            return direct
        derived = self.inference.closure(subject, relation)
        if derived:
            ans = [(v, p.confidence, "inferred", [p.explain()]) for v, p in derived]
            return Answer(True, ans, "Derived by inference from verified facts.")
        return direct                         # honest "I don't know"

    def verify(self, subject, relation, value) -> str:
        proof = self.inference.prove(normalize_text(subject),
                                     normalize_relation(relation), value)
        if proof:
            return f"yes (proved: {proof.explain()})"
        held = self.ltm.query(subject, relation, value)
        if held:
            return f"unverified or disputed (status: {held[0].status})"
        return "unknown: not stored and not derivable"

    # Phase 11 §2.5 (integration fix): the design's literal template
    # "What is the {relation} of {subject}?" rendered "capital_of"
    # as "capital of of paris" — a double "of". Relations are
    # rendered as question phrases instead; unknown relations fall
    # back to the generic template with a trailing "of" stripped.
    RESEARCH_QUESTION_TEMPLATES = {
        "capital_of": "What is the capital of {subject}?",
        "boils_at": "What is the boiling point of {subject}?",
        "melts_at": "What is the melting point of {subject}?",
        "located_in": "Where is {subject} located?",
        "is_a": "What kind of thing is {subject}?",
        "founded_in": "When was {subject} founded?",
        "population_of": "What is the population of {subject}?",
        "larger_than": "What is {subject} larger than?",
    }

    def formulate_research_question(self, subject, relation) -> str:
        """Phase 11 §2.5: the research question for a (subject, relation)
        gap. The relation renders as a question phrase (see
        ``RESEARCH_QUESTION_TEMPLATES``), e.g. ``capital_of`` ->
        "What is the capital of paris?". Pure function of its inputs;
        no brain state is read or written."""
        subject = normalize_text(subject)
        template = self.RESEARCH_QUESTION_TEMPLATES.get(
            normalize_relation(relation))
        if template is not None:
            return template.format(subject=subject)
        rel_words = normalize_relation(relation).replace("_", " ")
        if rel_words.endswith(" of"):
            rel_words = rel_words[:-len(" of")]
        return f"What is the {rel_words} of {subject}?"

    def add_research_goal(self, subject, relation, urls: list[str], priority: float = 0.5) -> Goal:
        """Goal: make (subject, relation) a VERIFIED fact, by consulting the given sources.

        Phase 11 §2.5: the goal meta records ``research_question``
        (see :meth:`formulate_research_question`) and, when candidate
        URLs are given and a researcher exists, ``research_sources``
        (the researcher's ranked list) — with a ``research_question``
        event and, when selection ran, a ``research_sources_selected``
        event carrying the ranking reasons from
        ``researcher.last_selection``. Both creation paths (gap
        identification and the contradiction policy) route through
        here, so the trace is recorded once, in one place. A brain
        without a researcher still records the question; selection is
        skipped (no researcher to rank with) and no selection event
        is emitted. Ranking is delegated entirely to
        ``Researcher.select_sources`` — no second ranking
        implementation lives here."""
        subject = normalize_text(subject)
        relation = normalize_relation(relation)
        key = f"known:{subject}|{relation}".lower()
        self._knowledge_keys[key] = (subject, relation)
        # Phase 12 (P12-O5): _knowledge_keys feeds the derived
        # known:* part of the world view — invalidate its cache.
        self._world_token += 1
        self._world_cache = None
        self.planner.add_action(Action(f"research_{subject}_{relation}", {}, {key: True}, cost=3.0,
                                       args={"urls": list(urls)}, tool="research"))
        goal = self.goals.add(Goal(f"learn {relation} of {subject}", {key: True}, priority))
        question = self.formulate_research_question(subject, relation)
        goal.meta["research_question"] = question
        self._event("research_question",
                    f"research goal {goal.id}: {question} cycle={self.cycle}")
        researcher = getattr(self, "researcher", None)
        select = getattr(researcher, "select_sources", None) if researcher is not None else None
        if callable(select) and urls:
            ranked = select(question, list(urls))
            goal.meta["research_sources"] = list(ranked)
            reasons = "; ".join(
                f"{url} ({reason})"
                for url, reason in getattr(researcher, "last_selection", []) or [])
            self._event("research_sources_selected",
                        f"research goal {goal.id}: sources ranked "
                        f"{list(ranked)}"
                        + (f" -- {reasons}" if reasons else "")
                        + f" cycle={self.cycle}")
        return goal

    def add_goal(self, description: str, target: dict, priority: float = 0.5,
                 deadline_cycle: int | None = None) -> Goal:
        return self.goals.add(Goal(description, target, priority, deadline_cycle))

    # ------------------------------------------------------------------ perception (Phase 8, P8-1)
    def perceive(self, kind: str, payload, source: str = "user",
                 confidence: float = 1.0, urgency: float = 0.5):
        """Unified perception entry: every source goes through the bus.

        ``kind`` is one of ``text`` | ``document`` | ``ocr`` |
        ``image`` | ``observation`` | ``event`` | ``specialist``.
        Returns the routed :class:`Percept` (confidence, provenance
        and source identity preserved). Unknown kinds raise
        ``ValueError``.
        """
        return self.perception_bus.perceive(kind, payload, source=source,
                                            confidence=confidence,
                                            urgency=urgency)

    def perceive_text(self, text: str, source: str = "user", urgency: float = 0.5) -> Thought:
        # Thin wrapper over the bus (behaviour unchanged): text becomes
        # a Thought in the incoming queue, admitted to WM by _attend.
        percept = self.perception_bus.perceive("text", text, source=source,
                                               urgency=urgency)
        return percept.content

    def perceive_image(self, path: str) -> dict:
        # Thin wrapper over the bus (behaviour unchanged): the Phase 3
        # OCR gate still applies — per-word threshold 0.6, uncertain
        # words logged as ocr_uncertain events, never reaching WM/LTM.
        # Raises OcrUnavailable when no OCR engine is available.
        # (Structured image records go through perceive("image", ...)
        # instead; an image is never flattened to a string here.)
        percept = self.perception_bus.perceive("ocr", path, source=f"ocr:{path}")
        return percept.content

    def read_document(self, path: str) -> dict:
        """Read a local ``.txt`` / ``.md`` document through the bus
        and return the reading report (stored / corroborated /
        duplicate / conflict / ignored_sentences, sentence kept per
        stored claim). A missing, oversize, undecodable or
        unsupported file fails cleanly: the report carries ``error``
        and zero counts, and no partial claims are stored. Logs a
        brain event, so ``explain()`` shows what was read and learned.
        Documents are untrusted data, never instructions; a single
        document never verifies a fact (file trust 0.45, P8-2)."""
        from .body.reading import read_text_file
        try:
            text = read_text_file(path)
        except (FileNotFoundError, ValueError, OSError, UnicodeDecodeError) as exc:
            self._event("document_read",
                        f"failed {path}: {type(exc).__name__}: {exc}")
            return {"stored": 0, "corroborated": 0, "duplicate": 0,
                    "conflict": 0, "ignored_sentences": [],
                    "source": f"file:{os.path.basename(str(path))}",
                    "claims": [], "trust": 0.45,
                    "error": f"{type(exc).__name__}: {exc}"}
        return self.read_text(text, source=f"file:{os.path.basename(str(path))}")

    def read_text(self, text: str, source: str = "file:text") -> dict:
        """Read document text through the bus (same admission path as
        ``read_document``) and return the reading report. Logs a brain
        event."""
        percept = self.perception_bus.perceive(
            "document", {"text": text, "source": source}, source=source,
            confidence=0.45)
        report = percept.content
        self._event("document_read",
                    f"read {source}: stored={report['stored']} "
                    f"corroborated={report['corroborated']} "
                    f"duplicate={report['duplicate']} "
                    f"conflict={report['conflict']} "
                    f"ignored={len(report['ignored_sentences'])}")
        return report

    # ------------------------------------------------------------------ the cycle
    def step(self, env=None) -> str | None:
        self.cycle += 1
        # Phase 2: the cycle budget is enforced here, in step(), not just
        # in run(). When the budget is exhausted no further rule fires;
        # the stop reason is recorded and run() reports it.
        if self.control.over_budget(self.cycle):
            self._set_stop("budget_exhausted")
            return None
        self._sense(env)
        # Phase 3: check last cycle's stored predictions against the
        # freshly sensed world, before anything else reasons about it.
        self._check_predictions()
        self._attend()
        self._reason_from_wm()
        # Phase 3 (mandate decision): queued interruptions from a
        # resumption are executed now if their condition still persists,
        # or dropped with a dropped_interrupt event if it does not.
        self._process_queued_interrupts()
        self._check_interruption(env)
        self._maybe_switch_goal()
        # Phase 7: the monitor enforces budgets / cancellation DURING
        # execution, before any rule fires. A violation stops the task
        # here — no rule fires on a budget-exceeded or cancelled task.
        if self._monitor_check() is not None:
            return None
        # Phase 12 (P12-O4): one world build shared by the
        # cognitive state and the rule match (previously two builds
        # of the same state here).
        world = self.world
        cog = self._cognitive_state(world)
        matched = self.rules.match({**world, **cog})
        # Phase 4: ALL matched rules are recorded (not just the one
        # that fires) — the considered set, persisted in plan.json.
        # Which rule FIRES is unchanged: meta[0], as before.
        self.last_considered_rules = [r.id for r in matched]
        meta = [r for r in matched if not all(a["do"] == "avoid_action" for a in r.then)]
        if not meta:
            self._event("idle", "no rule applies")
            return None
        rule = meta[0]
        ok = True
        events_before = len(self.events)
        # Phase 7: the rule that chose this step is part of the ledger's
        # decision record (see _do_execute_step).
        self._current_rule_id = rule.id
        for act in rule.then:
            ok = getattr(self, "_do_" + act["do"])(act.get("args", {}), env) and ok
        # Phase 7: no step is in flight past the firing loop (each
        # _do_execute_step clears the flag on its own returns; this is
        # the belt-and-braces clear for any other action handler).
        self._executing_step = False
        # Phase 4: a need-more-evidence refusal is a DEFERMENT, not a
        # failure — the brain asked for more information instead of
        # guessing. Record it via record_deferred (fired only, never
        # success/fail, never retires) rather than record_outcome(False).
        if any(e["kind"] == "need_more_evidence" for e in self.events[events_before:]):
            self.rules.record_deferred(rule.id)
        else:
            self.rules.record_outcome(rule.id, ok)
        # Phase 3: Brain.history is gone — stall is measured on goal
        # progress (Phase 2), and the event log is the only history.
        if self.goals.active() is not None:
            self.control.record_progress(self.cycle, self._goal_progress())
        self.learner.propose_rules()
        # Phase 6: end-of-cycle consolidation — P-C/P-D candidates
        # from the episode set, promotion of those meeting the bar,
        # and concept learning C-1/C-2 (see _consolidation_pass).
        self._consolidation_pass()
        # Phase 7: the monitor is checked again after the step, so a
        # budget tripped by the step just executed (or a cancellation
        # requested during it) stops the task before the next cycle.
        self._monitor_check()
        self._event("rule", f"fired '{rule.id}' -> {'ok' if ok else 'not ok'}; "
                            f"considered {self.last_considered_rules}")
        return rule.id

    def run(self, env=None, max_cycles: int | None = None) -> str:
        """Run cycles until a terminal condition. Returns the stop reason:
        "done", "impossible", "budget_exhausted", "stalled",
        "no_feasible_goal" or "interrupted". The cycle count remains
        available as ``brain.cycle``. (Phase 1 returned the cycle count;
        no existing test asserted that return value.)
        """
        limit = max_cycles or self.control.max_cycles
        self.last_stop_reason = None
        while self.cycle < limit:
            if self.goals.active() is None and not self._selectable_pending():
                if self.last_stop_reason is None:
                    if self.goals.all_suspended():
                        self._set_stop("interrupted")
                    elif any(g.status == "failed" for g in self.goals.all()):
                        # The failing goal's mapped reason was recorded by
                        # _do_fail_goal in last_stop_reason already; this
                        # branch only fires when it was not set.
                        self._set_stop("impossible")
                    elif self.goals.all() and all(g.status == "done" for g in self.goals.all()):
                        self._set_stop("done")
                    elif not self.goals.all():
                        self._set_stop("done")
                    else:
                        self._set_stop("no_feasible_goal")
                break
            self.step(env)
            if self.last_stop_reason in ("budget_exhausted",
                                        "budget_exceeded"):
                break
        if self.cycle >= limit and self.last_stop_reason is None:
            if self.goals.all() and all(g.status == "done" for g in self.goals.all()):
                self._set_stop("done")
            else:
                self._set_stop("budget_exhausted")
        return self.last_stop_reason or "done"

    # ---- Phase 2 helpers -------------------------------------------------
    def _set_stop(self, reason: str) -> None:
        """Record the terminal stop reason (first one wins) with an event."""
        if self.last_stop_reason is None:
            self.last_stop_reason = reason
            self._event("stop", reason)

    def _selectable_pending(self) -> list[Goal]:
        """Pending goals that may be freshly selected. A suspended goal is
        pending in status but quarantined here: it can only re-enter via
        resume_goal(), so a fresh selection can never silently discard
        its saved plan position."""
        return [g for g in self.goals.pending() if self.goals.suspended(g.id) is None]

    @staticmethod
    def _goal_urgency(g: Goal, cycle: int) -> float:
        if g.deadline_cycle is None:
            return 0.3
        return max(0.0, min(1.0, 1 - (g.deadline_cycle - cycle) / 20))

    def _fact_confidence(self) -> float:
        """Confidence of the facts the current decision relies on: the
        weakest proof confidence from the last cycle's WM derivations,
        or 1.0 when no inference was involved."""
        if not self.last_inferences:
            return 1.0
        return min(d.get("confidence", 1.0) for d in self.last_inferences)

    def _goal_progress(self) -> float:
        """Phase 2 goal-progress measure (audit A5 fix): how far the
        active goal has actually advanced. It is the sum of (a) target
        keys currently satisfied, (b) steps of the current plan whose
        effects materialised, and (c) verified LTM facts relevant to
        the goal. World keys flipping irrelevantly do not move it, so
        a churning world can neither hide a stall nor fake progress.
        Returns 0.0 when no goal is active."""
        g = self.goals.active()
        if g is None:
            return 0.0
        satisfied = sum(1 for k, v in g.target.items() if self.world.get(k) == v)
        verified = 0
        keywords = g.keywords
        for f in self.ltm.all_facts():
            if f.status != "verified":
                continue
            if f.subject in keywords or f"known:{f.subject}|{f.relation}" in self._knowledge_keys:
                verified += 1
        return float(satisfied + self._steps_executed + verified)

    # ---- Phase 2: interruption and resumption ----------------------------
    def _check_interruption(self, env) -> bool:
        """Check interruption triggers, in priority order, after ATTEND
        and before rule matching. At most one fires per cycle.

        Goal-level preemption (the active goal is suspended):
        - urgent percept: a fresh percept in WM with urgency >= the
          attention interrupt threshold (0.9).
        Plan-level interruption (the plan is invalidated, the goal
        itself is NOT preempted — it replans via the replan rule):
        - env_change: the next planned action's preconditions no longer
          hold in the freshly sensed world.
        - surprise: the previous cycle ended in surprise, so the plan
          was already cleared; recorded here so the invalidation is a
          structured event, not a silent flag.
        The higher-priority-goal trigger lives in _maybe_switch_goal,
        which runs right after this check.
        Returns True when a trigger fired.
        """
        if self._resuming:
            return False
        for t in self.wm.contents():
            # Freshness is "perceived since the last cycle began":
            # step() increments the cycle before this check runs, so a
            # percept made between steps carries created_cycle == cycle-1.
            # An exact == test could never fire (found by Phase 2 tests).
            # A percept older than one cycle is stale and must not keep
            # re-triggering after a resumption.
            if (t.kind == "percept" and t.created_cycle >= self.cycle - 1
                    and t.urgency >= self.attention.interrupt_threshold):
                if self.goals.active() is None:
                    return False
                self._interrupt_current_goal(
                    "urgent_percept", f"urgent percept: {str(t.content)[:50]}")
                return True
        if self.plan and self.goals.active() is not None:
            nxt = self.plan[0]
            if not Planner.applicable(nxt, self.world):
                self._set_plan([])
                self.surprise = True
                self.alternative_plan, self.alternative_plan_cost = [], 0.0
                self._event("interrupt",
                            f"trigger=env_change reason=preconditions of '{nxt.name}' "
                            f"no longer hold cycle={self.cycle}")
                return True
        if self.surprise:
            self._event("interrupt",
                        f"trigger=surprise reason=current plan invalidated "
                        f"by surprise cycle={self.cycle}")
            return True
        return False

    def _interrupt_current_goal(self, trigger: str, reason: str) -> SuspendedGoal | None:
        """Suspend the active goal, preserving its plan position, and
        log a structured interrupt event. During a resumption the
        interruption is queued, not executed (Phase 2: no nested
        preemption); the queued trigger is recorded in
        ``_queued_interrupts`` for the next cycle to re-evaluate."""
        if self._resuming:
            self._queued_interrupts.append(
                {"trigger": trigger, "reason": reason, "cycle": self.cycle})
            self._event("interrupt",
                        f"trigger={trigger} reason=queued during resumption "
                        f"cycle={self.cycle}")
            return None
        g = self.goals.active()
        if g is None:
            return None
        rec = self.suspend_active_goal(reason)
        self._set_plan([])
        self.surprise, self.plan_failed = False, False
        self._plan_started_len = 0
        self._steps_executed = 0
        self.control.reset_progress()
        self._event("interrupt",
                    f"trigger={trigger} reason={reason} goal='{g.description}' "
                    f"position={rec.plan_position} cycle={self.cycle}")
        return rec

    def resume_goal(self, goal_id: int) -> bool:
        """Resume a suspended goal (Phase 2 behaviour for the Phase 1
        SuspendedGoal structure). Restores the plan and plan position,
        re-validates the FIRST remaining action's preconditions against
        the current CognitiveState — later actions' preconditions are
        satisfied by earlier actions' effects, so checking them against
        the current world would wrongly call them stale. If the first
        action is stale, the remaining plan is discarded and a fresh
        plan is made from the current state (logged as replan_on_resume);
        if no plan exists the goal fails with reason "no plan on
        resume". Returns True when the goal is active again."""
        rec = self.goals.suspended(goal_id)
        if rec is None:
            return False
        g = self.goals.get(goal_id)
        if g.status in ("done", "failed"):
            return False
        cur = self.goals.active()
        if cur is not None and cur.id != goal_id:
            self._interrupt_current_goal(
                "resumed_goal", f"preempted by resume of goal '{g.description}'")
        self._resuming = True
        try:
            self.goals.unsuspend(goal_id)
            self.goals.activate(goal_id)
            # Phase 4: refresh the cached goal Thought — the one cached
            # at selection time is stale (old created_cycle and
            # urgency). The resumption cycle and the goal's current
            # urgency are what the next cycle must see.
            self._goal_thoughts[goal_id] = Thought(
                g.description, "goal", "goal-manager", g.priority, 1.0,
                g.keywords, self.cycle)
            self._set_plan([Action(x["name"], dict(x.get("pre", {})),
                                   dict(x.get("eff", {})), float(x.get("cost", 1.0)),
                                   dict(x.get("args", {})), x.get("tool"), x.get("store_as"))
                            for x in rec.plan])
            # Phase 7: a loaded or interrupted execution session for
            # this goal re-opens here (same task id, ledger kept); the
            # next action is re-validated below, exactly like the
            # existing resume re-validation. When the plan is stale
            # and replanned, _do_make_plan re-opens it instead.
            # Coordinator integration fix (Phase 7 gate): the
            # re-open must also happen when the restored plan is
            # EMPTY (cancelled mid-step: the plan was cleared by
            # the failure path before suspension, so the goal resumes
            # with "will replan"). Without this, the session stayed
            # "cancelled" and the next _do_make_plan refused to
            # re-open it (a cancelled session never re-opens on an
            # ordinary replan — ADR P7-9), so the resumed run
            # suspended again immediately. Re-opening here is
            # correct: resume_goal IS the explicit resume.
            session = self.execution_monitor.session
            if (session is not None and session.goal_id == goal_id
                    and session.status in ("interrupted", "cancelled")):
                reopened = self.execution_monitor.reopen_session(self.plan)
                if reopened is not None:
                    self._event("execution_resumed",
                                f"task {reopened.task_id[:12]} resumed for "
                                f"'{g.description}' "
                                f"({reopened.steps_executed}/{reopened.steps_total} steps done)")
            self._plan_started_len = len(self.plan) + rec.plan_position
            self._steps_executed = rec.plan_position
            self.goal_start = dict(rec.goal_start)
            self.surprise = False
            self.plan_failed = False
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self.last_stop_reason = None
            if not self.plan:
                self._event("resume",
                            f"goal='{g.description}' resumed at position "
                            f"{rec.plan_position} (no remaining plan; will replan) "
                            f"cycle={self.cycle}")
                return True
            first = self.plan[0]
            stale = [k for k, v in first.pre.items() if self.world.get(k) != v]
            if stale:
                changed = {k: (rec.world_snapshot.get(k), self.world.get(k))
                           for k in rec.world_snapshot
                           if rec.world_snapshot.get(k) != self.world.get(k)}
                self._event("replan_on_resume",
                            f"goal='{g.description}' stale preconditions {stale} "
                            f"changed_snapshot={changed} cycle={self.cycle}")
                self._set_plan([])
                self._plan_started_len = 0
                if not self._do_make_plan({}, None):
                    self._do_fail_goal({"reason": "no plan on resume"}, None)
                    return False
                self._event("resume",
                            f"goal='{g.description}' resumed at position "
                            f"{rec.plan_position} (replanned) cycle={self.cycle}")
                return True
            self._event("resume",
                        f"goal='{g.description}' resumed at position "
                        f"{rec.plan_position} cycle={self.cycle}")
            return True
        finally:
            self._resuming = False
            if self._queued_interrupts:
                # Phase 3 (mandate decision): queued triggers are NOT
                # dropped here — they are kept for the next cycle, which
                # executes them if their condition still persists or
                # drops them with a dropped_interrupt event if not (see
                # _process_queued_interrupts).
                self._event("interrupt_queued",
                            f"{len(self._queued_interrupts)} interruption(s) queued "
                            f"during resumption; re-evaluated next cycle "
                            f"cycle={self.cycle}")

    def _record_need_percept(self, goal: Goal) -> list[dict]:
        """Phase 9 §4.9 (minimal active sensing): on the
        ``need_more_evidence`` path, record what percept could supply
        the missing information — one ``need_percept`` event plus one
        state entry per unmet target key, naming the missing
        (subject, relation) and the percept kind that could supply it
        (``document`` for a knowledge fact, ``observation`` for a
        physical state key). This is the brain REQUESTING a percept;
        it does not fetch anything and does not claim a full
        active-sensing loop (that stays open for Phase 10). Returns
        the recorded requests."""
        requests: list[dict] = []
        for k, v in goal.target.items():
            m = re.match(r"known:(.+)\|(.+)", k)
            if m:
                if self.known(m.group(1), m.group(2)) == v:
                    continue
                subject, relation, kind = m.group(1), m.group(2), "document"
            elif self.world.get(k) == v:
                continue
            else:
                subject, relation, kind = str(k), "value", "observation"
            self._set_state(f"need_percept:{subject}|{relation}", kind,
                            source="brain:need_percept", confidence=0.5,
                            provenance=f"need_more_evidence for '{goal.description}'",
                            entry_type="need_percept")
            self._event("need_percept",
                        f"missing {subject} {relation}; a {kind} percept "
                        f"could supply it (goal '{goal.description}')")
            requests.append({"subject": subject, "relation": relation,
                             "kind": kind, "goal": goal.description})
        # Phase 10 (P10-7): act on the requests instead of only
        # recording them (see _act_on_need_percept).
        for req in requests:
            self._act_on_need_percept(req, goal)
        return requests

    def _act_on_need_percept(self, request: dict, goal: Goal) -> None:
        """Phase 10 (P10-7): act on one ``need_percept`` request.

        - Knowledge fact (kind ``document``): when the goal names a
          readable source in ``goal.meta`` — ``document`` /
          ``documents`` (a ``.txt`` / ``.md`` path, read through
          ``read_document``) or ``image`` / ``images`` (read through
          the OCR perception path) — read it in-cycle through the
          perception bus and let the result enter WM/LTM by the
          existing path. A read failure is logged (``percept_failed``)
          and never crashes the cycle. With no source named, nothing
          is read: the ``need_percept`` record above stands.
        - Physical state (kind ``observation``): the brain already
          re-observes every cycle; the new part is the request
          driving a SPECIFIC perception action, recorded here as a
          ``percept_requested`` event naming the key.
        Each (subject, relation, source) acts at most once per Brain
        lifetime (``_percept_requests_acted``), so a standing
        need_more_evidence refusal does not re-read every cycle.
        Boundary (P10-7): no autonomous sensing loop beyond the
        need-driven request — nothing here polls, scans, or fetches
        on its own schedule."""
        subject = request["subject"]
        relation = request["relation"]
        kind = request["kind"]
        meta = goal.meta if isinstance(getattr(goal, "meta", None), dict) else {}
        if kind == "document":
            sources: list[tuple[str, str]] = []
            for key, skind in (("document", "document"),
                               ("documents", "document"),
                               ("image", "image"), ("images", "image")):
                value = meta.get(key)
                if isinstance(value, str) and value:
                    sources.append((value, skind))
                elif isinstance(value, (list, tuple)):
                    sources.extend((str(p), skind) for p in value if p)
            for path, skind in sources:
                marker = (subject, relation, skind, path)
                if marker in self._percept_requests_acted:
                    continue
                self._percept_requests_acted.add(marker)
                self._event("percept_requested",
                            f"need {subject} {relation}: reading {skind} {path} "
                            f"(goal '{goal.description}')")
                try:
                    if skind == "document":
                        self.read_document(path)
                    else:
                        self.perceive_image(path)
                except Exception as exc:  # unreadable image / no OCR engine: logged, cycle continues
                    self._event("percept_failed",
                                f"need {subject} {relation}: could not read {skind} {path}: "
                                f"{type(exc).__name__}: {exc}")
            return
        marker = (subject, relation, "observation", "")
        if marker in self._percept_requests_acted:
            return
        self._percept_requests_acted.add(marker)
        self._event("percept_requested",
                    f"need {subject} {relation}: re-observing '{subject}' "
                    f"(goal '{goal.description}'); no autonomous sensing "
                    f"loop beyond this need-driven request")

    def _request_missing_information(self, goal: Goal) -> Goal | None:
        """Create a research goal for the first unmet target key of
        ``goal`` when that key names a knowledge fact
        (``known:subject|relation``). Returns the research goal, or None
        when the missing information is not a researchable knowledge
        fact (physical goals have no web query to make), when there is
        no researcher, or when every target key is already satisfied.
        The brain asks for information instead of guessing; it does not
        fabricate a query for a physical precondition."""
        if self.researcher is None:
            return None
        for k, v in goal.target.items():
            # Phase 3: knowledge targets are checked through the derived
            # known() view, never through a stored known:* entry.
            m = re.match(r"known:(.+)\|(.+)", k)
            if m:
                if self.known(m.group(1), m.group(2)) == v:
                    continue
            elif self.world.get(k) == v:
                continue
            if not m:
                return None
            subject, relation = m.group(1), m.group(2)
            urls = list(goal.meta.get("urls", [])) if isinstance(goal.meta, dict) else []
            rg = self.add_research_goal(subject, relation, urls)
            self._event("research_request",
                        f"missing information for '{goal.description}': "
                        f"learn {relation} of {subject}")
            return rg
        return None

    def _try_alternative(self, failed: Action, situation: dict) -> bool:
        """Recovery after a failed step: ban the (action, situation)
        pair so the planner does not immediately re-propose it, then
        adopt the retained alternative plan if its first step is still
        applicable AND no LATER step in its sequence is banned in the
        situation it would run in (Phase 4 whole-sequence check,
        scoped to later steps by the Phase 7 integration fix: the
        first step is governed by the retrieved-failure count rule
        — rejected only after >= 2 failures in the same situation —
        so one flaky failure cannot permanently block a retry).
        When the alternative is rejected for a banned later step, the
        brain replans immediately (same cycle) instead of adopting it.
        Returns True when recovery was handled (alternative adopted or
        a fresh plan made), so the caller must not clear the plan;
        False when nothing was done and the caller should invalidate
        the plan as before."""
        self.planner.ban(failed.name, situation, failed.args, failed.tool)
        # Phase 6: retrieve relevant past experience for the failed
        # situation before choosing a recovery. The retrieved
        # episodes are the evidence the consolidation pass and the
        # repeated-mistake detector read; the count is logged even
        # when no alternative exists, so retrieval is visible.
        retrieved: list = []
        if self.retriever is not None:
            features = dict(situation)
            if failed.eff:
                features["effect"] = sorted(failed.eff)[0]
            retrieved = self.retriever.retrieve(features, k=5)
            self._event("retrieved_experience",
                        f"{len(retrieved)} relevant episode(s) retrieved "
                        f"for failed '{failed.name}' in situation "
                        f"{sorted(situation)}")
        # A retrieved failure is an exact (action, args, situation)
        # repeat (Phase 10.5 P10.5-3: the args identity joins the
        # key, so a retrieved failure of one variant — cut-zinc —
        # does not count against a sibling variant — cut-tin).
        # Coordinator integration fix (Phase 7 gate): a SINGLE
        # retrieved failure must not reject an alternative's first
        # step. One failure can be flaky (a tool that raises once and
        # succeeds afterwards); treating it as proof the step can
        # never work made the brain loop press-fail-replan until it
        # stalled, with the only working step (cut) permanently
        # blocked by its one flaky failure (failing-tool acceptance
        # experiment). The rejection now needs >= 2 retrieved
        # failures of the same (action, situation) — the same
        # threshold as the repeated-mistake detector — and the
        # strategy cap (2 same-strategy attempts) is the backstop if
        # the retry fails again. The whole-sequence ban check below
        # is likewise scoped to the LATER steps: the first step's
        # single-failure ban is what this count rule governs, while a
        # banned later step still rejects the alternative.
        retrieved_failed_counts = Counter(
            (str(ep.get("action", "")),
             canonical_args(ep.get("args", {}) or {}),
             tuple(sorted((str(k), str(v)) for k, v in
                          dict(ep.get("situation", {})).items())))
            for ep in retrieved
            if str(ep.get("outcome", "")).lower() in ("failure", "failed"))
        if self.alternative_plan and Planner.applicable(self.alternative_plan[0], self.world):
            first = self.alternative_plan[0]
            first_situation = tuple(sorted(
                (str(k), str(v)) for k, v in
                {k: self.world.get(k) for k in first.pre}.items()))
            first_failures = retrieved_failed_counts.get(
                (first.name,
                 canonical_args(self._redact_tool_args(
                     first.tool, dict(getattr(first, "args", None) or {}))),
                 first_situation), 0)
            if first_failures >= 2:
                self._event("alternative_rejected",
                            f"alternative plan rejected after '{failed.name}' failed: "
                            f"its first step '{first.name}' repeats a retrieved "
                            f"failure {first_failures} times in the same situation; replanning")
                return self._do_make_plan({}, None)
            later_steps = self.alternative_plan[1:]
            if later_steps and Planner.sequence_banned(
                    later_steps,
                    {**self.world, **dict(first.eff)},
                    self.planner.banned_pairs,
                    redact_args=self._redact_tool_args):
                self._event("alternative_rejected",
                            f"alternative plan rejected after '{failed.name}' failed: "
                            f"a later step is banned in its situation; replanning")
                # Phase 4: the brain already knows it must replan, so it
                # replans in THIS cycle rather than waiting for next
                # cycle's replan rule — a recovery cycle spent only
                # waiting would stall the goal (no step executes, so
                # goal progress stays flat across the stall window).
                # Scoped to this path only: when there is no alternative
                # at all, the existing interrupt/replan rhythm (and the
                # stall test's timing) is untouched. Returns True when
                # a fresh plan was made, so the caller does not clear it.
                return self._do_make_plan({}, None)
            adopted = self.alternative_plan
            self._set_plan(list(adopted))
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self._plan_started_len = len(self.plan)
            self.surprise = False
            self._event("alternative_plan",
                        f"adopted after '{failed.name}' failed: "
                        f"{' > '.join(a.name for a in adopted)}")
            return True
        return False

    # ---- Phase 6: learning integration ------------------------------
    def _on_episode_recorded(self, ep: Episode) -> None:
        """Learner.on_record hook (Phase 6 §3): every recorded episode
        also updates the retriever index and the prioritizer's
        usefulness counts. The retriever holds a mirror list of
        episode dicts (the learner stays the single store); the list
        is capped at the learner's cap."""
        self._retriever_episodes.append(
            {"action": ep.action, "situation": dict(ep.situation),
             "outcome": "success" if ep.ok else "failure",
             # Phase 10.5: the mirror carries the args the action ran
             # with, so _try_alternative's retrieved-failure counts
             # can key by (action, args, situation).
             "args": dict(getattr(ep, "args", None) or {}),
             "cycle": ep.cycle})
        cap = getattr(self.learner, "max_episodes", 500)
        if len(self._retriever_episodes) > cap:
            self._retriever_episodes = self._retriever_episodes[-cap:]
        self.retriever.index(self._retriever_episodes)
        self.prioritizer.note_use(ep.action, dict(ep.situation))

    def _announce_patterns(self, learned) -> None:
        """Emit one pattern_learned event per newly added P-A/P-B
        pattern (candidates are announced by the consolidation pass
        when promoted, never here)."""
        for pattern in (learned or []):
            if getattr(pattern, "candidate", False):
                continue
            if pattern.id in self._announced_patterns:
                continue
            self._announced_patterns.add(pattern.id)
            self._event("pattern_learned",
                        f"pattern '{pattern.name}' learned from episodes "
                        f"({pattern.kind}; confidence {pattern.confidence:.2f})")

    def _consolidation_pass(self) -> None:
        """Phase 6 end-of-cycle consolidation (design §3): after
        propose_rules(), learn P-C/P-D candidates from the episode
        set, promote those meeting the >= 3-episodes / >= 2-situations
        bar with a ``consolidated`` event, and learn concepts C-1/C-2
        from the same episodes. P-A/P-B still apply immediately
        (learn_from_episodes adds them); only P-C/P-D go through the
        queue. Promotion never writes to LTM."""
        episodes = list(self.learner.episodes)
        if not episodes:
            return
        action_effects = {a.name: dict(a.eff) for a in self.planner.actions}
        learned = self.patterns.learn_from_episodes(
            episodes, action_effects, world_model=self.world_model)
        # Phase 7 §4.2: pattern-evidence episodes earn usefulness +1
        # when the pattern is learned (see _mark_pattern_evidence_useful).
        self._mark_pattern_evidence_useful(learned)
        for pattern in learned:
            if pattern.candidate:
                self._register_candidate(pattern, episodes)
        self._announce_patterns(learned)
        for cand in self.consolidator.review():
            if len(cand["evidence"]) >= 3 and cand["situations"] >= 2:
                self._promote_candidate(cand["id"])
        self._learn_concepts(episodes)

    def _register_candidate(self, pattern, episodes) -> str:
        """Submit a P-C/P-D pattern as a consolidation candidate.
        The candidate content is the STABLE structure (kind, name,
        effect, roles, relations) — never the evidence list, so the
        stable candidate id does not change as more episodes arrive.
        Evidence is the supporting episodes (action + outcome) and
        the situation count is the distinct situations among them.
        Returns the candidate id."""
        actions = {str(e.get("action", "")) for e in pattern.evidence}
        supporting = [ep for ep in episodes if ep.action in actions]
        evidence = [{"action": ep.action,
                     "outcome": "success" if ep.ok else "failure"}
                    for ep in supporting]
        situations = len({tuple(sorted((str(k), str(v)) for k, v in
                                       dict(ep.situation).items()))
                          for ep in supporting})
        content = {"name": pattern.name, "kind": pattern.kind,
                   "effect": pattern.effect,
                   "roles": [dict(r) for r in pattern.roles],
                   "relations": [list(rel) for rel in pattern.relations]}
        return self.consolidator.add_candidate(
            "pattern", content, evidence, situations,
            provenance=(f"{pattern.kind} from {len(evidence)} episodes "
                        f"across {situations} situations"))

    def _promote_candidate(self, candidate_id: str, force: bool = False) -> bool:
        """Promote one candidate to durable knowledge (design §2):
        the Consolidator records the promotion (bar or force), the
        pattern joins the library, and a ``consolidated`` event names
        the candidate and its evidence. For a P-C pattern the durable
        tool is also marked in the world model
        (breaks_after_first_use False, inferred) so the pattern and
        the durable_tool concept can match it structurally. Returns
        False when the candidate is unknown, rejected, or below the
        bar without force."""
        cand = self.consolidator.get(candidate_id)
        if cand is None:
            return False
        if cand["status"] == "promoted":
            return True
        if not self.consolidator.promote(candidate_id, force=force,
                                         cycle=self.cycle):
            return False
        pattern = self._pattern_from_candidate(cand)
        if pattern is not None:
            self.patterns.add_pattern(pattern)
            if pattern.kind == "durable_tool":
                # Coordinator fix (Phase 7 gate): mark the tool of
                # EVERY action with the evidence name, not just the
                # first. In a continued brain the action list
                # accumulates the old world's actions, so first-match
                # marked W1's cutter and never W2's shears — the
                # durable_tool concept then could not match the new
                # world's tool and the 0.8 preference never fired
                # there, even though the concept is explicitly
                # name-free and transfer is its purpose. Each
                # same-named action's own tool is marked, so the
                # promotion transfers to the new world's tool.
                for action_name in {str(e.get("action", ""))
                                    for e in pattern.evidence}:
                    actions = [a for a in self.planner.actions
                               if a.name == action_name]
                    if not actions:
                        tool = _tool_entity_name(action_name)
                        if tool:
                            self.world_model.set_property(
                                tool, "breaks_after_first_use", False,
                                status="inferred",
                                confidence=pattern.confidence,
                                provenance=f"pattern:{pattern.name}",
                                cycle=self.cycle)
                        continue
                    for action in actions:
                        tool = self._action_tool_name(action)
                        if tool:
                            self.world_model.set_property(
                                tool, "breaks_after_first_use", False,
                                status="inferred",
                                confidence=pattern.confidence,
                                provenance=f"pattern:{pattern.name}",
                                cycle=self.cycle)
        content = cand.get("content", {}) or {}
        self._event("consolidated",
                    f"promoted candidate {candidate_id[:12]} "
                    f"({content.get('kind')}:{content.get('effect')}) "
                    f"with evidence {len(cand['evidence'])} episodes "
                    f"across {cand['situations']} situations")
        return True

    @staticmethod
    def _pattern_from_candidate(cand: dict):
        """Rebuild an AbstractPattern from a candidate's stored
        content + evidence (the candidate queue is the persistence
        form; the library holds the pattern). None when the content
        is not a pattern candidate."""
        if cand is None or cand.get("kind") != "pattern":
            return None
        content = cand.get("content", {}) or {}
        evidence = [{"world": "", "action": str(e.get("action", "")),
                     "outcome": str(e.get("outcome", ""))}
                    for e in (cand.get("evidence", []) or [])]
        return AbstractPattern(
            name=str(content.get("name", "")),
            kind=str(content.get("kind", "")),
            effect=str(content.get("effect", "")),
            roles=[dict(r) for r in content.get("roles", [])],
            relations=[tuple(rel) for rel in content.get("relations", [])],
            evidence=evidence,
            confidence=min(0.9, 0.5 + 0.1 * len(evidence)))

    @staticmethod
    def _requires_keys(failures: list, successes: list) -> list:
        """Keys missing (absent-or-falsy) in every failure situation
        and present-and-truthy in at least one success situation —
        the P-D / C-2 precondition shape (same rule as
        ab.world.patterns._requires_keys, over plain dicts)."""
        if len(failures) < 2 or not successes:
            return []
        keys: set[str] = set()
        for s in successes:
            keys.update(str(k) for k, v in (s or {}).items() if v)
        return sorted(k for k in keys
                      if all(not (s or {}).get(k) for s in failures))

    def _concept_keys(self, episodes):
        """The C-1 / C-2 examples derivable from ``episodes``.

        Returns ``(c1, c2)`` where ``c1`` is a list of
        ``(tool, broke, successes)`` for tool actions with a
        recordable durable_tool example (broke after first use, or
        >= 2 successes with no break), and ``c2`` is a list of
        ``(action, key, failed_situations, succeeded_situations)``
        for the P-D key shape. Shared by _learn_concepts (which
        records them) and _restore_learning (which pre-seeds the
        dedupe sets from the restored episodes)."""
        c1, c2 = [], []
        by_action: dict[str, list] = {}
        for ep in episodes:
            by_action.setdefault(ep.action, []).append(ep)
        for action in sorted(by_action):
            rows = by_action[action]
            successes = [ep for ep in rows if ep.ok]
            failures = [ep for ep in rows if not ep.ok]
            action_obj = next((a for a in self.planner.actions
                               if a.name == action), None)
            tool = (self._action_tool_name(action_obj)
                    if action_obj is not None else _tool_entity_name(action))
            if tool is not None:
                broke = False
                by_sit: dict[tuple, list[bool]] = {}
                for ep in rows:
                    key = tuple(sorted((str(k), str(v)) for k, v in
                                       dict(ep.situation).items()))
                    by_sit.setdefault(key, []).append(ep.ok)
                for outcomes in by_sit.values():
                    if True in outcomes and any(
                            not o for o in outcomes[outcomes.index(True) + 1:]):
                        broke = True
                if broke or len(successes) >= 2:
                    c1.append((tool, broke, len(successes)))
            if len(failures) >= 2 and successes:
                failed_sits = [dict(ep.situation) for ep in failures]
                succeeded_sits = [dict(ep.situation) for ep in successes]
                for key in self._requires_keys(failed_sits, succeeded_sits):
                    c2.append((action, key, failed_sits, succeeded_sits))
        return c1, c2

    def _learn_concepts(self, episodes) -> None:
        """Phase 6 rules C-1 / C-2 from the episode set (see
        _concept_keys for the shapes). Each example is recorded
        once (see _c1_seen / _c2_seen), so the per-cycle pass never
        re-adds the same example."""
        c1, c2 = self._concept_keys(episodes)
        for tool, broke, successes in c1:
            key = (tool, broke)
            if key in self._c1_seen:
                continue
            if self.concepts.learn_durable_tool(tool, successes, broke):
                self._c1_seen.add(key)
        for action, key, failed_sits, succeeded_sits in c2:
            if (action, key) in self._c2_seen:
                continue
            self.concepts.learn_requires(action, failed_sits, succeeded_sits)
            self._c2_seen.add((action, key))

    def consolidate(self, candidate_id: str) -> bool:
        """Explicit human promotion of one candidate (design §2):
        delegates to Consolidator.promote(force=True) via
        _promote_candidate, which also applies the promotion
        (pattern -> library, durable-tool marking in the world
        model). Returns True only when the candidate was promoted."""
        return self._promote_candidate(candidate_id, force=True)

    # ---- Phase 3: predictions, queued interruptions, auto-resume --------
    def _store_predictions(self, action: Action) -> None:
        """Store an action's predicted effects in state BEFORE it is
        executed, as ``predicted:<action>:<key>`` with confidence,
        cycle and source "planner". The next cycle checks them against
        the sensed world (see _check_predictions). The provenance
        carries the action's preconditions as the situation, so a
        violated prediction can ban the (action, situation) pair and
        feed the causal model (Phase 4)."""
        situation = {k: self.world.get(k) for k in action.pre}
        provenance = json.dumps({"plan": action.name, "situation": situation}, sort_keys=True)
        for k, v in action.eff.items():
            # Phase 6 (§2, §3): confidence comes from the
            # CalibrationTracker — the same Laplace formula as Phase
            # 5, (met + 1) / (total + 2), now with the tracker
            # recording the pair so ECE can be measured. With no
            # calibration record for this (action, effect) the
            # causal record is the fallback (same formula); with
            # neither it stays 1.0, labelled uncalibrated.
            if (self.calibration is not None
                    and self.calibration.is_calibrated(action.name, k)):
                confidence = self.calibration.calibrated_confidence(action.name, k)
            else:
                # Phase 5 (§2.12): confidence is the Laplace-smoothed
                # causal success rate (met + 1) / (total + 2) once at
                # least one causal record exists for this (action,
                # effect); with no record it stays 1.0 — the planner's
                # own model with no evidence yet, documented as
                # uncalibrated (full calibration is Phase 6 work).
                # Phase 9 (P9-1): the rate is this action VARIANT's
                # (its canonical precondition set), not the pooled
                # name rate — a variant that always failed must not
                # borrow another variant's successes for confidence.
                counts = self.causal.counts(
                    action.name, k,
                    variant=canonical_variant(action.pre))
                confidence = ((counts[0] + 1) / (counts[1] + 2)
                              if counts is not None else 1.0)
            self._set_state(f"predicted:{action.name}:{k}", v, source="planner",
                            confidence=confidence, provenance=provenance,
                            entry_type="inferred")

    def _predicted_actual(self, key):
        """The sensed value a prediction is checked against. Knowledge
        keys go through the derived known() view, everything else
        through the plain world view."""
        m = re.match(r"known:(.+)\|(.+)", key)
        if m:
            return self.known(m.group(1), m.group(2))
        return self.world.get(key)

    def _plan_uses_action(self, action_name: str) -> bool:
        """True when the CURRENT plan contains a step named
        ``action_name``. (Coordinator fix: the retained ALTERNATIVE
        plan is checked separately — an already-executed action whose
        prediction was later invalidated by an exogenous change, e.g.
        the Phase 2 sabotage teleport, may still sit in the
        alternative while the current plan has moved past it; dropping
        the current plan for that would destroy the env_change
        interruption path, which is the designed handler for exactly
        that situation.)"""
        return action_name in [a.name for a in self.plan]

    def _alternative_uses_action(self, action_name: str) -> bool:
        """True when the retained ALTERNATIVE plan contains a step
        named ``action_name``."""
        return action_name in [a.name for a in self.alternative_plan]

    def _resolve_executed_variant(self, action_name: str, entry):
        """Phase 9 (P9-1): the variant of the action actually executed,
        for the causal record / revision of this checked prediction.
        The prediction's provenance carries the situation — the
        executed variant's precondition values at execution time
        (``_store_predictions`` records exactly the executed action's
        precondition keys) — so the variant is resolved by matching
        name + the precondition set against the planner's action
        list. Returns ``(variant, resolved)``: the canonical
        precondition set, or ``(None, False)`` when no action
        matches (the caller then records ``variant=None`` — the
        documented fallback — and the prediction_checked event says
        so). An exact precondition-set match is required: a subset
        match would resolve a with-mould execution to the cheaper
        without-mould variant, which is the conflation this fix
        removes."""
        situation = {}
        try:
            provenance = json.loads(getattr(entry, "provenance", "") or "")
            if isinstance(provenance, dict):
                situation = provenance.get("situation", {}) or {}
        except (TypeError, ValueError):
            situation = {}
        if isinstance(situation, dict) and situation:
            for a in self.planner.actions:
                if a.name != action_name:
                    continue
                if set(a.pre) == set(situation) and all(
                        situation.get(k) == v for k, v in a.pre.items()):
                    return canonical_variant(a.pre), True
        return None, False

    def _check_predictions(self) -> None:
        """Check every stored prediction against the sensed world, log
        a prediction_checked event with the outcome (met / violated),
        and delete the entry. Runs at the start of a cycle, so a
        prediction stored in cycle N is checked in cycle N+1.

        Phase 4: every outcome is recorded in the causal model, and a
        violation is self-correction, not a silent miss — it is kept
        for abduction in this cycle's reasoning, the LTM belief the
        prediction depended on is retracted (or the claim is flagged
        disputed when there is no such belief to retract), the
        (action, situation) pair is banned so the planner avoids an
        action whose effect did not materialise, and a revision event
        is logged."""
        for key in [k for k in self.state.keys() if k.startswith("predicted:")]:
            entry = self.state.get(key)
            if entry is None:
                continue
            rest = key[len("predicted:"):]
            action, _, eff_key = rest.partition(":")
            actual = self._predicted_actual(eff_key)
            outcome = "met" if actual == entry.value else "violated"
            # Phase 9 (P9-1): the causal record and the simulation
            # revision are keyed by the variant actually executed
            # (resolved from the prediction's provenance situation);
            # variant=None is the recorded fallback when it cannot be
            # resolved.
            variant, _variant_resolved = self._resolve_executed_variant(
                action, entry)
            self._event("prediction_checked",
                        f"{action} predicted {eff_key}={entry.value!r} "
                        f"actual={actual!r} outcome={outcome} "
                        f"variant={variant!r} cycle={self.cycle}")
            # Phase 5: the causal record BEFORE the violation branch,
            # so the "previously succeeded" check below sees the
            # history without this failure (record() happens once).
            prior_counts = self.causal.counts(action, eff_key,
                                              variant=variant)
            self.causal.record(action, eff_key, outcome == "met",
                               variant=variant)
            if outcome == "violated":
                # The engine's abduction reasons over episode OUTCOME
                # labels ("success" / "failure"), not raw state values:
                # a violated prediction is a failed outcome for this
                # action. The raw values are kept alongside for the
                # revision trail.
                self._last_violations.append(
                    {"action": action, "expected": "success",
                     "actual": "failure", "met": False,
                     "effect": eff_key, "expected_value": entry.value,
                     "actual_value": actual})
                self._event("revision",
                            f"prediction violated: {action} predicted "
                            f"{eff_key}={entry.value!r} actual={actual!r}; "
                            f"flagging the belief it depended on")
                # Phase 5 (§2.12–§2.13, §4.2): a violated prediction
                # revises the world model, and the revision drives
                # recovery. The old value is retained in the model's
                # append-only revisions; the error is recorded with
                # its magnitude; bans keyed to the old model are
                # cleared for the revised keys. The stall progress
                # window restarts when this is the FIRST recorded
                # outcome for the (action, effect) pair: the failure
                # is news, the model and the plan change in response,
                # and the old window measured a different situation.
                # A REPEAT failure of the same pair is not news —
                # resetting on every repeated violation would defeat
                # stall detection outright (StuckWorld: the same
                # prediction fails forever, and the goal must
                # eventually fail honestly instead of retrying
                # forever). One reset per distinct pair, at most, so
                # the window always refills on a genuinely stuck goal.
                # When the action previously SUCCEEDED for this same
                # effect, the tool it used is marked single-use
                # suspect and the pattern library relearns from the
                # episode history — the structural evidence for a
                # one-use tool. Prior success is read from the causal
                # record OR the learner's success episodes: the
                # causal record alone misses a success whose
                # prediction was later invalidated by an exogenous
                # change (the chamber reset: the door opened, then
                # swung shut before the check), while the learner's
                # episode is the action's own success record.
                # Finally, a plan that still relies on the violated
                # action to produce this effect was built on the
                # falsified model, so it is dropped and remade from
                # the revised model instead of executing stale.
                self.world_model.record_prediction_error(
                    action, eff_key, entry.value, actual, self.cycle)
                self.world_model.revise(
                    eff_key, entry.value, actual,
                    reason="prediction violated", cycle=self.cycle)
                cleared = self.planner.clear_bans_for_keys([eff_key])
                first_outcome = prior_counts is None or prior_counts[1] == 0
                if first_outcome:
                    self.control.reset_progress()
                self._event("model_revised",
                            f"{eff_key}: {entry.value!r} -> {actual!r} "
                            f"(prediction violated by {action}); "
                            f"cleared {cleared} ban(s) keyed to the old "
                            f"model"
                            + ("; stall window reset (first outcome for "
                               "this action/effect)" if first_outcome else "")
                            + f" cycle={self.cycle}")
                # A plan that still relies on the violated action is
                # dropped and remade from the revised model — but only
                # the plan(s) that actually rely on it. The current
                # plan is dropped when it contains the action; the
                # retained alternative is dropped when IT contains it.
                # (Coordinator fix: previously either plan containing
                # the action dropped BOTH, so an already-executed
                # action lingering in the alternative — e.g. go_lab
                # after a sabotage teleport — destroyed the current
                # plan and with it the env_change interruption path.)
                in_current = self._plan_uses_action(action)
                in_alt = self._alternative_uses_action(action)
                if in_alt:
                    self.alternative_plan, self.alternative_plan_cost = [], 0.0
                if in_current:
                    self._set_plan([])
                    self._event("plan_invalidated",
                                f"plan dropped after {action}'s predicted "
                                f"{eff_key} was violated; replanning from "
                                f"the revised model cycle={self.cycle}")
                elif in_alt:
                    self._event("plan_invalidated",
                                f"alternative plan dropped after {action}'s "
                                f"predicted {eff_key} was violated; the "
                                f"current plan does not use it "
                                f"cycle={self.cycle}")
                previously_succeeded = (
                    (prior_counts is not None and prior_counts[0] > 0)
                    or any(getattr(ep, "action", None) == action
                           and getattr(ep, "ok", False)
                           for ep in getattr(self.learner, "episodes", [])))
                if previously_succeeded:
                    tool = _tool_entity_name(action)
                    if tool:
                        self.world_model.mark_single_use_suspect(
                            tool, self.cycle)
                    learned = self.patterns.learn_from_episodes(
                        list(self.learner.episodes),
                        action_effects={a.name: dict(a.eff)
                                        for a in self.planner.actions},
                        world_model=self.world_model)
                    # Phase 7 §4.2: pattern-evidence episodes earn
                    # usefulness +1 when the pattern is learned.
                    self._mark_pattern_evidence_useful(learned)
                    # Phase 6: P-C/P-D patterns come back as
                    # candidates, not learned knowledge — they go to
                    # the consolidation queue, never to this event.
                    for p in learned:
                        if getattr(p, "candidate", False):
                            self._register_candidate(
                                p, list(self.learner.episodes))
                    added = [p for p in learned
                             if not getattr(p, "candidate", False)]
                    new = [p for p in added
                           if p.id not in self._announced_patterns]
                    for p in new:
                        self._announced_patterns.add(p.id)
                    if new:
                        self._event(
                            "pattern_learned",
                            f"{len(new)} pattern(s) learned from "
                            f"episodes after {action} violated "
                            f"{eff_key}: "
                            + ", ".join(p.id[:12] for p in new))
                self._retract_failed_belief(eff_key, entry.value)
                # NOTE (Phase 4): a violated prediction does NOT ban the
                # (action, situation) pair here. A violation only proves
                # the world differs from the prediction — the world may
                # have moved exogenously (e.g. the Phase 2 sabotage
                # test, where the environment teleports the agent after
                # a successful move), so attributing the miss to the
                # action would wrongly ban the only way back. Genuine
                # action failure is already banned, situation-scoped,
                # by _try_alternative at execution time (surprise /
                # failure), which is the planner-avoidance mechanism
                # item 11 names; the causal record above is what feeds
                # abduction and the causal-claims path.
            # Phase 8 (P8-5): simulation revision — feed this checked
            # outcome back into the simulator's per-(action, effect)
            # predicted success, so the next similar task's simulation
            # moves toward the observed rate. The world-model
            # prediction error itself is NOT recorded here: the
            # violation branch above already records it (and only
            # violations), so passing the world model would change
            # the error trail's meaning; SimulationRevision.observe
            # records the error when it is given a world model (its
            # compare() path and direct callers do that, deduplicated
            # on (action, effect, cycle, expected, actual)).
            _rev = getattr(self, "simulation_revision", None)
            if _rev is not None:
                _adj = _rev.observe(action, eff_key, entry.value, actual,
                                    self.cycle, world_model=None,
                                    causal=self.causal, variant=variant)
                if _adj.get("moved"):
                    self._event("simulation_revised",
                                f"{action} {eff_key}: predicted success "
                                f"{_adj['prior']:.2f} -> {_adj['adjusted']:.2f} "
                                f"(observed {'met' if outcome == 'met' else 'missed'})")
            self.state.delete(key)

    def _retract_failed_belief(self, eff_key: str, predicted_value) -> None:
        """Self-correction for a failed prediction: if the predicted
        effect key names an LTM belief (``known:subject|relation`` or
        ``subject|relation``) whose stored value is the predicted
        (wrong) value, retract it — the belief the prediction depended
        on was false. Otherwise there is no LTM belief to retract, and
        the claim is flagged disputed in state instead."""
        m = re.match(r"(?:known:)?(.+)\|(.+)", eff_key)
        if not m:
            return
        subject, relation = normalize_text(m.group(1)), normalize_relation(m.group(2))
        for f in self.ltm.query(subject, relation):
            if normalize_text(str(f.value)) != normalize_text(str(predicted_value)):
                continue
            self.ltm.retract(f.id, reason="failed prediction", cycle=self.cycle)
            self._event("revision",
                        f"retracted belief {subject} {relation} {f.value} "
                        f"after failed prediction cycle={self.cycle}")
            return
        # No stored belief carries the predicted value. A direct
        # ``subject|relation`` prediction names a claim value, so flag
        # that claim disputed; a ``known:`` prediction only says
        # knowledge was not acquired (its value, True, is not a claim),
        # so there is nothing to flag.
        if eff_key.startswith("known:"):
            return
        self._set_state(f"disputed:{subject}|{relation}", [str(predicted_value)],
                        source="reasoning:revision", confidence=0.5,
                        provenance="failed prediction", entry_type="belief",
                        status="disputed")

    def unchecked_predictions(self) -> list[str]:
        """Stored predictions that have not been checked yet (Phase 3
        §2.7). A prediction that is never checked is a bug; tests use
        this to catch it."""
        return [k for k in self.state.keys() if k.startswith("predicted:")]

    def _queued_condition_persists(self, trigger: str, queued: dict) -> bool:
        """Does a queued interruption's trigger condition still hold?
        Only verifiable conditions may re-fire; anything else is
        dropped rather than fired late."""
        if trigger == "urgent_percept":
            return any(t.kind == "percept" and t.created_cycle >= self.cycle - 1
                       and t.urgency >= self.attention.interrupt_threshold
                       for t in self.wm.contents())
        if trigger == "higher_priority_goal":
            cur = self.goals.active()
            if cur is None:
                return False
            return any(g.priority > cur.priority + self.control.switch_margin
                       for g in self._selectable_pending())
        if trigger == "resumed_goal":
            m = re.search(r"resume of goal '([^']+)'", queued.get("reason", ""))
            if not m:
                return False
            g = self.goals.active()
            return g is not None and g.description == m.group(1)
        return False

    def _process_queued_interrupts(self) -> None:
        """Phase 3 (mandate decision): on the next cycle after a
        resumption, execute each queued trigger whose condition still
        persists, and drop the rest with a dropped_interrupt event."""
        if not self._queued_interrupts:
            return
        queued, self._queued_interrupts = self._queued_interrupts, []
        for q in queued:
            trigger = q.get("trigger", "")
            if self.goals.active() is None or not self._queued_condition_persists(trigger, q):
                self._event("dropped_interrupt",
                            f"trigger={trigger} reason=condition no longer persists "
                            f"cycle={self.cycle}")
                continue
            self._event("interrupt",
                        f"trigger={trigger} reason=queued condition persists "
                        f"cycle={self.cycle}")
            self._interrupt_current_goal(trigger, q.get("reason", "queued interruption"))

    def _maybe_auto_resume(self) -> None:
        """Phase 3 (mandate decision): when the preempting goal finishes
        and no higher-priority pending goal exists, auto-resume the
        most recently suspended goal with a resume event. Explicit
        resume_goal() stays available for choosing a different
        suspended goal. Only completion auto-resumes; a failed goal
        does not pull a suspended goal back in."""
        if self._selectable_pending():
            return
        suspended = self.goals.all_suspended()
        if not suspended:
            return
        rec = max(suspended, key=lambda r: r.suspended_cycle)
        self.resume_goal(rec.goal_id)

    # ------------------------------------------------------------------ cycle internals
    def _sense(self, env) -> None:
        obs = env.observe() if env is not None else {}
        # Phase 5: a key that changed since the last observation
        # refresh, with no environment action executed in between
        # (see _acted_since_sense, cleared on every _prev_obs
        # refresh), is an EXOGENOUS change — the world moved on its
        # own. It is logged as a world_changed brain event and
        # recorded in the model with cause="exogenous", never as an
        # action effect. (The mid-task door reset in the chamber
        # worlds lands here: the reset fires inside observe() itself,
        # after the post-action observation already refreshed
        # _prev_obs, so no action sits between the two observations.)
        exogenous: dict[str, tuple] = {}
        for k, v in obs.items():
            if self._prev_obs.get(k) != v:
                self._incoming.append(Thought(f"{k}={v}", "percept", "environment", 0.4, 1.0, {k}, self.cycle))
            if k in self._prev_obs and self._prev_obs.get(k) != v \
                    and not self._acted_since_sense:
                exogenous[k] = (self._prev_obs.get(k), v)
        self._prev_obs = dict(obs)
        self._acted_since_sense = False
        for k, v in obs.items():
            self._set_state(k, v, source=f"env:{k}", confidence=1.0,
                            provenance="env:observe", entry_type="observation")
            if k in exogenous:
                old, _new = exogenous[k]
                self._event("world_changed",
                            f"exogenous change: {k}: {old!r} -> {v!r} "
                            f"cycle={self.cycle}")
                self.world_model.observe(k, v, self.cycle, source="env",
                                         cause="exogenous")
            else:
                self.world_model.observe(k, v, self.cycle, source="env")
        # Phase 5: unobserved entities keep existing with decaying
        # confidence (object permanence); decay runs once per cycle.
        self.world_model.decay(self.cycle)
        # Phase 3: _refresh_knowledge is gone — known:* is a derived
        # view computed on read (Brain.known / Brain.world), so there
        # is nothing to refresh here.

    def _record_seen(self, signature: str) -> None:
        """Write one focused thought's signature to state as a seen
        entry (Phase 3 §2.5): key ``seen:<sha>``, value the signature
        verbatim, source "attention", type "observation", status
        "active". New keys log no state_change event (see _set_state),
        so this is quiet."""
        digest = hashlib.sha256(str(signature).encode()).hexdigest()[:16]
        key = f"seen:{digest}"
        if key in self.state:
            return
        self._set_state(key, str(signature), source="attention", confidence=1.0,
                        provenance="attention:focus", entry_type="observation")

    def _seed_attention_from_state(self) -> None:
        """Re-seed Attention._seen from the seen:* state entries
        (used on load; also picks up entries seeded from a legacy
        attention.json)."""
        sigs = [e.value for e in self.state.snapshot().values()
                if e.key.startswith("seen:") and isinstance(e.value, str)]
        if sigs:
            self.attention.restore_seen(sigs)

    def _attend(self) -> None:
        self.wm.tick()
        goal = self.goals.active()
        cands = list(self._incoming)
        self._incoming.clear()
        if goal:
            gt = self._goal_thoughts.setdefault(goal.id, Thought(goal.description, "goal", "goal-manager",
                                                                 goal.priority, 1.0, goal.keywords, self.cycle))
            cands.append(gt)
        kept, blocked = self.control.inhibit(cands)
        for t in blocked:
            self._event("inhibit", f"blocked thought: {str(t.content)[:50]}")
        # Phase 3 §2.5: every candidate Attention scores gets its
        # signature recorded in state (Attention.focus adds all
        # candidates to its seen-set, so recording `kept` keeps the
        # state view and the in-memory seen-set identical).
        for t in self.attention.focus(kept, goal.keywords if goal else set(), self.cycle, k=FOCUS_K):
            if self.attention.interrupts(t):
                self._event("interrupt", f"urgent: {str(t.content)[:50]}")
            self.wm.admit(t)
        for t in kept:
            self._record_seen(t.signature())

    # ---- Phase 4: reasoning-engine integration -------------------------
    def _facts_fn(self, subject, relation=None):
        """Verified-only facts for the engine (its injected knowledge
        source). Accepts fn(subject) or fn(subject, relation); the
        engine falls back to its own LTM when passed None, so this is
        also usable standalone."""
        return self.ltm.query(subject, relation, statuses=("verified",))

    def _world_facts_fn(self, subject, relation=None):
        """Phase 5: the engine's facts source, extended with
        world-model relations. Verified LTM facts come first (via
        ``_facts_fn``); then the model's ``is_a`` / ``part_of`` /
        ``located_in`` relations for the subject, with status observed
        or inferred only — never assumed or predicted.

        NOTE on the shape: ``ReasoningEngine._fetch_facts`` admits
        only facts whose status reads ``verified`` (its verified-only
        gate), and ``Proof`` reads ``.confidence`` / ``.subject`` /
        ``.relation`` / ``.value`` as attributes, so a plain dict
        cannot travel through the engine — it would crash
        ``proof_confidence``. Model relations are therefore presented
        as ``Fact`` objects with status ``verified`` (the brain
        vouches for its own direct observations), with the real model
        status preserved in the provenance string. Only relations the
        model itself observed or inferred are ever presented."""
        if subject is None:
            out = list(self._facts_fn(None, relation))
        else:
            out = list(self._facts_fn(subject, relation))
        if subject is not None:
            for rel_name in ("is_a", "part_of", "located_in"):
                for r in self.world_model.relations(subject=subject,
                                                    relation=rel_name):
                    status = r.status.value if hasattr(r.status, "value") else str(r.status)
                    if status not in ("observed", "inferred"):
                        continue
                    if not isinstance(r.object, str):
                        continue
                    conf = max(0.0, min(1.0, float(r.confidence)))
                    out.append(Fact(
                        subject=str(r.subject), relation=str(r.relation),
                        value=str(r.object),
                        sources={"world_model": conf}, status="verified",
                        added_cycle=self.cycle,
                        history=[f"world_model:{status}:{r.provenance}"]))
        # Phase 6: the concept store answers is_a queries for learned
        # concepts. An entity whose world-model properties satisfy a
        # concept's defining features is presented as a Fact with
        # status verified (the brain vouches for its own learned
        # concept, as it does for its own direct observations above),
        # with the real provenance — concept name and confidence —
        # preserved in the history, so reasoning can use a learned
        # concept (e.g. "X is_a durable_tool") and planning can
        # prefer it afterwards. Only is_a is answered this way, and
        # only when the caller asked for is_a (or for everything).
        if subject is not None and self.concepts is not None and relation in (None, "is_a"):
            ent = self.world_model.entity(subject)
            if ent is not None:
                features = {k: p.value for k, p in
                            (getattr(ent, "properties", {}) or {}).items()}
                for name in self.concepts.names():
                    ok, conf = self.concepts.is_a(features, name)
                    if not ok:
                        continue
                    out.append(Fact(
                        subject=str(subject), relation="is_a", value=name,
                        sources={"concept_store": conf}, status="verified",
                        added_cycle=self.cycle,
                        history=[f"concept:{name}:{conf:.2f}"]))
        return out

    def _inherited_derivations(self, derivations, facts_fn, constraints):
        """Property inheritance, supplementing the engine's deduction.

        The engine's BFS follows transitive relations only, so a
        functional property (``boils_at`` / ``melts_at`` /
        ``capital_of``) stated about a parent is never derived for a
        child. Here, for each transitive derivation ``S -> V``, the
        verified functional facts about ``V`` are inherited by ``S``
        (proof = the chain plus the parent's fact, confidence = the
        weakest link). Only verified facts are inherited, and inherited
        derivations go through the same constraint check as engine
        derivations. This is what makes a derived functional claim
        able to contradict a verified direct fact (see
        ``_handle_contradictions``)."""
        functional = {normalize_relation(r) for r in FUNCTIONAL_RELATIONS}
        out, seen = [], set()
        for d in derivations:
            if d.proof is None:
                continue
            for f in self._facts_fn(d.value):
                rel = normalize_relation(f.relation)
                if rel not in functional:
                    continue
                key = (normalize_text(d.subject), rel, normalize_text(f.value))
                if key in seen:
                    continue
                seen.add(key)
                conf = min(d.confidence, f.confidence)
                steps = list(d.proof.steps) + [f]
                nd = type(d)(subject=normalize_text(d.subject), relation=rel,
                             value=normalize_text(f.value),
                             proof=Proof(steps, Inference.proof_confidence(steps)),
                             confidence=conf, source="inheritance", status="derived")
                ok, _ = self.reasoning.check_constraints(nd, constraints)
                if ok:
                    out.append(nd)
        return out

    def _ltm_contradictions(self, subjects) -> list:
        """Conflicts among stored LTM facts for the given subjects:
        two distinct values for the same functional relation (verified,
        unverified or disputed — retracted facts do not count)."""
        functional = {normalize_relation(r) for r in FUNCTIONAL_RELATIONS}
        out = []
        for subj in sorted(subjects):
            by_rel: dict[str, list] = {}
            for f in self.ltm.query(subj, statuses=("verified", "unverified", "disputed")):
                rel = normalize_relation(f.relation)
                if rel not in functional:
                    continue
                val = normalize_text(str(f.value))
                if val not in by_rel.setdefault(rel, []):
                    by_rel[rel].append(val)
            for rel, vals in sorted(by_rel.items()):
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        out.append(Contradiction(
                            claim_a=(subj, rel, vals[i]),
                            claim_b=(subj, rel, vals[j]),
                            reason=(f"functional relation '{rel}' for '{subj}': "
                                    f"stored facts disagree ('{vals[i]}' vs '{vals[j]}')"),
                            policy="disputed+research"))
        return out

    def _test_hypotheses(self, action: str, outcome: str) -> None:
        """Test open hypotheses about ``action`` against a fresh
        outcome ("success" / "failure"). A hypothesis generated by
        abduction explains a surprise outcome; repeating the action
        runs its test — the same outcome corroborates it, a different
        outcome falsifies it (``evaluate_hypothesis`` emits the
        outcome event either way). A corroborated hypothesis is still
        never verified: no LTM fact is created from it."""
        for h in self.hypotheses:
            if h.get("status") != "open" or h.get("subject") != action:
                continue
            expected = h.get("outcome")
            if expected is None:
                continue
            self.evaluate_hypothesis(
                h["id"],
                {"met": outcome == expected,
                 "detail": (f"repeated '{action}' -> {outcome}; hypothesis "
                            f"predicted outcome '{expected}'")})

    def _handle_contradictions(self, contradictions) -> None:
        """Phase 4 contradiction policy: ``disputed+research``.

        A derived claim conflicting with a verified fact, or two
        derivations conflicting on a functional relation, is never
        silently resolved by picking a side. Both claims are marked
        disputed in state, a contradiction event and a revision event
        are logged, the resolution policy is recorded on the decision
        in progress (if any), and a research goal is created for the
        missing evidence. The order research goals are created in is
        decided by ``CognitiveControl.resolve_conflicts`` over the
        conflicting claims (name, confidence-as-priority, and the
        functional slot as the shared resource, so one research goal
        is created per disputed slot, highest-confidence claim first).
        Contradictions already recorded (same disputed values) are not
        re-emitted, so a standing contradiction does not spam events.
        """
        if self.last_decision is not None:
            self.last_decision.resolution_policy = "disputed+research"
        # BOTH claims of every contradiction are marked disputed in
        # state (all conflicting values are recorded, not just one
        # side) — the marking is not the resolution. A contradiction
        # already fully recorded emits no new event, so a standing
        # contradiction does not spam the log.
        for c in contradictions:
            subj_a, rel_a, val_a = (normalize_text(c.claim_a[0]), normalize_relation(c.claim_a[1]),
                                    normalize_text(str(c.claim_a[2])))
            subj_b, rel_b, val_b = (normalize_text(c.claim_b[0]), normalize_relation(c.claim_b[1]),
                                    normalize_text(str(c.claim_b[2])))
            key = f"disputed:{subj_a}|{rel_a}"
            entry = self.state.get(key)
            vals = list(entry.value) if entry is not None and isinstance(entry.value, list) else []
            new_vals = [v for v in (val_a, val_b) if v not in vals]
            if not new_vals:
                continue
            vals.extend(new_vals)
            self._set_state(key, vals, source="reasoning:contradiction",
                            confidence=0.5, provenance="contradiction",
                            entry_type="belief", status="disputed")
            self._event("contradiction",
                        f"{subj_a} {rel_a}: '{val_a}' vs '{val_b}' conflict on a "
                        f"functional relation; both values disputed, "
                        f"policy=disputed+research cycle={self.cycle}")
            self._event("revision",
                        f"contradiction on {subj_a} {rel_a}: neither side chosen; "
                        f"claim(s) flagged disputed, research goal created "
                        f"cycle={self.cycle}")
        # Research goals are created in the order resolve_conflicts
        # returns over the conflicting claims (claim, confidence as
        # priority, functional slot as the shared resource), so one
        # research goal is created per disputed slot, highest-confidence
        # claim first.
        conf_by_claim: dict[tuple, float] = {}
        for d in self.last_inferences:
            conf_by_claim[(d["subject"], d["relation"], d["value"])] = d.get("confidence", 0.5)
        proposals: list[tuple[str, float, frozenset]] = []
        for c in contradictions:
            for claim in (c.claim_a, c.claim_b):
                subj, rel, val = (normalize_text(claim[0]), normalize_relation(claim[1]),
                                  normalize_text(str(claim[2])))
                proposals.append((f"{subj}|{rel}={val}",
                                  float(conf_by_claim.get((subj, rel, val), 0.5)),
                                  frozenset({f"{subj}|{rel}"})))
        order = CognitiveControl.resolve_conflicts(proposals)
        for name in order:
            slot, _, _val = name.partition("=")
            subj, _, rel = slot.partition("|")
            target_key = f"known:{subj}|{rel}"
            if not any(target_key in g.target for g in self.goals.all()):
                self.add_research_goal(subj, rel, [])
                self._event("research_request",
                            f"missing evidence for disputed {subj} {rel}: "
                            f"research goal created cycle={self.cycle}")

    def _abstract_categories(self, subjects) -> None:
        """Phase 4 abstraction: subjects in working memory that share
        >= 3 verified (relation, value) properties propose a pending
        category. A proposal is knowledge about a possible grouping,
        never a fact: it is stored as a pending state entry (never in
        LTM) and needs human approval before anything may use it."""
        by_subject: dict[str, set] = {}
        for f in self.ltm.all_facts():
            if f.status == "verified" and f.subject in subjects:
                by_subject.setdefault(f.subject, set()).add((f.relation, f.value))
        subs = sorted(by_subject)
        for i in range(len(subs)):
            for j in range(i + 1, len(subs)):
                shared = sorted(by_subject[subs[i]] & by_subject[subs[j]])
                if len(shared) < 3:
                    continue
                payload = "|".join([subs[i], subs[j]] + [f"{r}={v}" for r, v in shared])
                cid = "cat_" + hashlib.sha256(payload.encode()).hexdigest()[:8]
                key = f"proposed_category:{cid}"
                if self.state.get(key) is not None:
                    continue
                self._set_state(key, {"members": [subs[i], subs[j]],
                                      "shared": [[r, v] for r, v in shared],
                                      "status": "pending"},
                                source="abstraction", confidence=0.5,
                                provenance="shared properties", entry_type="belief",
                                status="pending")
                self._event("category_proposed",
                            f"subjects {subs[i]} and {subs[j]} share {len(shared)} "
                            f"properties; proposed pending category {cid}")

    def _induced_pending(self) -> list[dict]:
        """Induced (success) rules awaiting human approval, read back
        from their pending state entries (so they survive save/load).
        Failure induction is the Learner's job and is not duplicated
        here (see _reason_from_wm)."""
        out = []
        for key in self.state.keys():
            if not key.startswith("induced_rule:"):
                continue
            entry = self.state.get(key)
            if not isinstance(entry.value, dict) or entry.value.get("status") != "pending":
                continue
            out.append(dict(entry.value))
        return out

    @property
    def hypotheses(self) -> list[dict]:
        """Hypotheses currently held (open / corroborated / falsified),
        read back from their state entries. Hypotheses never enter LTM
        and never drive an action. (A property, not a method, so
        ``brain.hypotheses`` is always an iterable list.)"""
        out = []
        for key in self.state.keys():
            if not key.startswith("hypothesis:"):
                continue
            entry = self.state.get(key)
            if isinstance(entry.value, dict):
                out.append(dict(entry.value))
        return out

    @property
    def proposed_categories(self) -> list[dict]:
        """Pending category proposals from abstraction (see
        _abstract_categories)."""
        out = []
        for key in self.state.keys():
            if not key.startswith("proposed_category:"):
                continue
            entry = self.state.get(key)
            if isinstance(entry.value, dict):
                out.append(dict(entry.value))
        return out

    def evaluate_hypothesis(self, hyp_id: str, observation: dict):
        """Test a held hypothesis against an observation
        (``{"met": bool, "detail": str}``). A passed test CORROBORATES
        (confidence +0.1, capped at 0.9) — it never verifies: no LTM
        fact is ever created from a hypothesis. A failed test FALSIFIES
        it, with a hypothesis_falsified event. Returns the updated
        Hypothesis, or None when no such hypothesis is held."""
        key = f"hypothesis:{hyp_id}"
        entry = self.state.get(key)
        if entry is None or not isinstance(entry.value, dict):
            return None
        d = dict(entry.value)
        hyp = Hypothesis(id=d["id"], subject=d["subject"], relation=d["relation"],
                         value=d["value"], confidence=d.get("confidence", 0.0),
                         source=d.get("source", ""),
                         evidence_for=list(d.get("evidence_for", [])),
                         evidence_against=list(d.get("evidence_against", [])),
                         status=d.get("status", "open"), test=d.get("test"))
        updated = self.reasoning.evaluate_hypothesis(hyp, observation)
        for kind, detail in self.reasoning.events:
            self._event(kind, detail)
        self.reasoning.events = []
        new_d = {**d, "status": updated.status, "confidence": updated.confidence,
                 "evidence_for": list(updated.evidence_for),
                 "evidence_against": list(updated.evidence_against)}
        self._set_state(key, new_d, source=entry.source, confidence=updated.confidence,
                        provenance=entry.provenance, entry_type="hypothesis",
                        status=updated.status)
        self.last_hypotheses = [new_d if h.get("id") == hyp_id else h
                                for h in self.last_hypotheses]
        return updated

    def simulate_plan(self, steps, goal: dict | None = None):
        """Counterfactual plan simulation over symbolic action effects
        only (``ReasoningEngine.simulate`` + ``Planner.effects_fn``):
        returns ``(simulated_state, met_goal, assumptions)`` without
        executing anything. Simulated outcomes are hypotheses, never
        observations. Steps may be Actions, plain names, or dicts."""
        start = dict(self.world)
        if goal is not None:
            start["goal"] = dict(goal)
        steps = [s.name if isinstance(s, Action) else s for s in (steps or [])]
        return self.reasoning.simulate(steps, start, self.planner.effects_fn)

    # ------------------------------------------------------------------ Phase 8: imagination
    def _get_simulator(self):
        """The brain's Simulator, rebuilt when the world model was
        replaced (e.g. by ``load()``) or the revision object is
        missing, so simulation always runs over a copy of the LIVE
        model, never a stale one."""
        rev = getattr(self, "simulation_revision", None)
        if rev is None:
            rev = SimulationRevision()
            self.simulation_revision = rev
        sim = getattr(self, "simulator", None)
        if sim is None or getattr(sim, "world_model", None) is not self.world_model:
            sim = Simulator(self.world_model, causal=self.causal,
                            planner=self.planner, revision=rev)
            self.simulator = sim
        else:
            sim.revision = rev
            sim.causal = self.causal
            sim.planner = self.planner
        return sim

    def _goal_target_dict(self, goal) -> dict:
        """A goal as a plain target dict: a dict passes through, a
        Goal contributes ``.target``, a string uses the active goal's
        target. Raises ValueError when no target can be determined."""
        if isinstance(goal, dict):
            return dict(goal)
        if goal is not None and hasattr(goal, "target"):
            return dict(goal.target)
        if isinstance(goal, str):
            active = self.goals.active()
            if active is not None:
                return dict(active.target)
        raise ValueError("imagine() needs a goal target dict (or a Goal)")

    def imagine(self, goal, include_banned: bool = False) -> dict:
        """Imagine routes to ``goal`` (a target dict or a Goal):
        generate >=2 distinct scenarios when the planner offers them,
        simulate each over a copy of the world model, and rank them
        with reasons. Nothing is executed and nothing is mutated —
        the result is labelled ``simulated`` throughout. The best
        scenario is kept as ``self._last_imagined`` so the revision
        loop can compare it with reality after execution.

        Phase 9 §4.3: with ``include_banned=True`` routes the
        planner bans for execution are ALSO surfaced, flagged
        ``banned`` — imaginable, never selected: the best scenario
        is always the top-ranked NON-banned one (None when every
        imagined route is banned). Default False keeps the previous
        contract (executable routes only)."""
        target = self._goal_target_dict(goal)
        sim = self._get_simulator()
        generator = ScenarioGenerator(self.planner, sim,
                                      max_scenarios=3, max_steps=20,
                                      include_banned=include_banned)
        scenarios = generator.generate(target, dict(self.world))
        comparator = ScenarioComparator()
        ranking = comparator.rank(scenarios)
        for entry in ranking:
            if entry["scenario"].banned:
                entry["reason"] += (" (banned: imaginable but not "
                                    "executable — uses a learned-banned "
                                    "(action, situation) pair)")
        best_entry = next((e for e in ranking
                           if not e["scenario"].banned), None)
        best = best_entry["scenario"] if best_entry is not None else None
        result = {
            "goal": dict(target),
            "include_banned": bool(include_banned),
            "scenarios": [s.to_dict() for s in scenarios],
            "ranking": [{k: (v.to_dict() if k == "scenario" else v)
                         for k, v in entry.items()}
                        for entry in ranking],
            "best": best.to_dict() if best is not None else None,
            "simulated": True,
            "assumptions": (list(best.assumptions) if best is not None
                            else ["simulated: no route to this goal was found "
                                  "from the current state"]),
            "note": ("" if scenarios else
                     "simulated: no plan reaches this goal from the current state"),
        }
        self.last_imagination = result
        self._last_imagined = best.to_dict() if best is not None else None
        if best is not None:
            detail = "; ".join(
                f"{e['name']}: {e['reason']}" for e in ranking)
            self._event("simulation_compared",
                        f"imagined {len(scenarios)} route(s) to {target}: "
                        f"best is {best.name} "
                        f"({' > '.join(best.steps)}) predicted success "
                        f"{best.predicted_success:.2f} cost "
                        f"{best.predicted_cost:.2f} — {detail} (simulated, "
                        f"not observed)")
        else:
            if scenarios:
                self._event("simulation_compared",
                            f"imagined {len(scenarios)} route(s) to {target} "
                            f"from the current state, but every route is "
                            f"banned for execution (no executable route "
                            f"imagined) (simulated, not observed)")
            else:
                self._event("simulation_compared",
                            f"imagined 0 routes to {target} from the current "
                            f"state (simulated, not observed)")
        return result

    def _parse_what_if(self, question: str):
        """Parse a "what if ...?" question into (action_name|None,
        state_edits dict). Action forms: "what if I use <action>?",
        "what if I try <tool>?", matching planner action names (with
        spaces or underscores) or the tool entity an action uses
        (``has_<tool>`` precondition / ``use_<tool>`` name). State
        forms: "what if the door were open/closed?" (door_open).
        Returns (None, {}) when nothing can be identified — the
        caller then says so honestly instead of inventing an answer."""
        q = (question or "").strip().lower()
        # Action forms first: "use/try/do (the) <phrase>".
        m = re.search(r"(?:use|try|do)\s+(?:the\s+)?([a-z][a-z _-]*?)(?:\?| to | instead| first|$)",
                      q)
        phrase = (m.group(1).strip(" -") if m else "")
        if phrase:
            phrase_us = phrase.replace(" ", "_").replace("-", "_")
            actions = sorted(self.planner.actions,
                             key=lambda a: -len(a.name))
            for a in actions:
                if phrase_us == a.name or phrase == a.name.replace("_", " "):
                    return a.name, {}
            for a in actions:
                if phrase_us and (phrase_us in a.name or a.name in phrase_us):
                    return a.name, {}
            for a in actions:
                tool = self._action_tool_name(a)
                if tool and (phrase_us == tool or phrase == tool.replace("_", " ")):
                    return a.name, {}
        # State forms: the door.
        if "door" in q:
            if "open" in q:
                return None, {"door_open": True}
            if "closed" in q or "shut" in q:
                return None, {"door_open": False}
        return None, {}

    def what_if(self, question: str) -> dict:
        """Answer a "what if ...?" question from simulation, with
        explicit assumptions labelled ``simulated`` — never as
        observed fact. "What if I use <action>?" simulates the best
        route that uses that action; "what if the door were open?"
        re-simulates the best route under that changed fact and
        reports the difference from the baseline simulation."""
        question = str(question or "")
        sim = self._get_simulator()
        action, state_edits = self._parse_what_if(question)
        base_assumption = "simulated: this answer comes from simulation, not observation"
        if action is not None:
            start = dict(self.world)
            goal = None
            active = self.goals.active()
            if active is not None:
                goal = dict(active.target)
            steps = None
            first = sim.simulate([action], start, max_steps=20)
            if first.per_step and first.per_step[0].get("confidence", 0.0) > 0.0 and goal:
                rest = self.planner.plan(first.predicted_state, goal, self._banned())
                if rest is not None:
                    steps = [action] + [a.name for a in rest.steps]
            if steps is None and goal is not None:
                # The action is not usable from the current state: find
                # an imagined route to the goal that uses it, so the
                # answer describes a real route instead of a no-op.
                generator = ScenarioGenerator(self.planner, sim,
                                              max_scenarios=3, max_steps=20)
                for sc in generator.generate(goal, start):
                    if action in sc.steps:
                        steps = list(sc.steps)
                        break
            if steps is None:
                steps = [action]
            result = sim.simulate(steps, start, goal=goal, max_steps=20)
            if steps[0] == action:
                route_text = (f"Using '{action}' then "
                              f"{' > '.join(steps[1:]) if len(steps) > 1 else 'nothing further'}")
            else:
                route_text = (f"Using '{action}' in the route "
                              f"{' > '.join(steps)}")
            answer = (f"(simulated) {route_text}: "
                      f"predicted state {result.predicted_state}, predicted "
                      f"success {result.predicted_success:.2f}, predicted cost "
                      f"{result.predicted_cost:.2f}.")
            if steps == [action] and (not first.per_step
                                      or first.per_step[0].get("confidence", 0.0) == 0.0):
                answer += (f" Note: '{action}' has no known effect in the "
                           f"current state, so the simulation predicts no change.")
            return {"question": question, "action": action, "steps": steps,
                    "predicted_state": dict(result.predicted_state),
                    "predicted_success": float(result.predicted_success),
                    "predicted_cost": float(result.predicted_cost),
                    "simulated": True,
                    "assumptions": [base_assumption] + list(result.assumptions),
                    "answer": answer}
        if state_edits:
            active = self.goals.active()
            goal = dict(active.target) if active is not None else None
            steps = [a.name for a in self.plan] if self.plan else None
            if steps is None and goal is not None:
                plan = self.planner.plan(dict(self.world), goal, self._banned())
                steps = [a.name for a in plan.steps] if plan is not None else []
            steps = steps or []
            baseline = sim.simulate(steps, dict(self.world), goal=goal, max_steps=20)
            changed = sim.simulate(steps, dict(self.world), goal=goal,
                                   max_steps=20, state_edits=state_edits)
            diff = {k: (baseline.predicted_state.get(k), changed.predicted_state.get(k))
                    for k in set(baseline.predicted_state) | set(changed.predicted_state)
                    if baseline.predicted_state.get(k) != changed.predicted_state.get(k)}
            edit_text = ", ".join(f"{k}={v!r}" for k, v in state_edits.items())
            answer = (f"(simulated) If {edit_text}, the same route "
                      f"{' > '.join(steps) or '(no steps)'} is predicted to end at "
                      f"{changed.predicted_state} (success "
                      f"{changed.predicted_success:.2f}) instead of "
                      f"{baseline.predicted_state} (success "
                      f"{baseline.predicted_success:.2f}).")
            return {"question": question, "action": None,
                    "state_edits": dict(state_edits),
                    "steps": steps,
                    "baseline": baseline.to_dict(),
                    "counterfactual": changed.to_dict(),
                    "difference": diff,
                    "simulated": True,
                    "assumptions": [base_assumption] + list(changed.assumptions),
                    "answer": answer}
        return {"question": question, "action": None,
                "simulated": True,
                "assumptions": [base_assumption,
                                "simulated: no action or changed fact could "
                                "be identified in the question, so no "
                                "simulation was run"],
                "answer": ("(simulated) I could not identify an action or a "
                           "changed fact in that question, so I did not "
                           "simulate anything rather than guess.")}

    def counterfactual(self, edits=None, goal=None, steps=None,
                       change=None, **kwargs) -> dict:
        """Counterfactual over the world model: edit a world-model
        fact (or state key) in the *copy*, re-simulate the same steps,
        and report the outcome difference, with the changed assumption
        named. ``edits`` may be ``{state_key: value}``,
        ``{"entity", "property", "value"}``, or a list of either;
        ``change={"key": ..., "value": ...}`` and keyword arguments
        (``counterfactual(door_open=True)``) are accepted too. The
        baseline is the same steps simulated from the current state
        with no edits — the actual-history simulation. Nothing is
        executed and the real world model is never edited."""
        sim = self._get_simulator()
        state_edits: dict = {}
        world_edits: list = []
        sources = []
        if change is not None:
            sources.append(change)
        if edits is not None:
            sources.append(edits)
        if kwargs:
            sources.append(dict(kwargs))
        for src in sources:
            if isinstance(src, dict) and "entity" in src and "property" in src:
                world_edits.append({"entity": str(src["entity"]),
                                    "property": str(src["property"]),
                                    "value": src.get("value")})
            elif isinstance(src, dict) and "key" in src:
                state_edits[str(src["key"])] = src.get("value")
            elif isinstance(src, dict):
                for k, v in src.items():
                    state_edits[str(k)] = v
            elif isinstance(src, (list, tuple)):
                for item in src:
                    if isinstance(item, dict) and "entity" in item and "property" in item:
                        world_edits.append(item)
                    elif isinstance(item, dict) and "key" in item:
                        state_edits[str(item["key"])] = item.get("value")
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        state_edits[str(item[0])] = item[1]
        if goal is None:
            active = self.goals.active()
            goal = dict(active.target) if active is not None else None
        else:
            goal = self._goal_target_dict(goal)
        if steps is None:
            if self.plan:
                steps = [a.name for a in self.plan]
            elif goal is not None:
                plan = self.planner.plan(dict(self.world), goal, self._banned())
                steps = [a.name for a in plan.steps] if plan is not None else []
            else:
                steps = []
        steps = [s.name if isinstance(s, Action) else s for s in (steps or [])]
        start = dict(self.world)
        baseline = sim.simulate(steps, start, goal=goal, max_steps=20)
        changed = sim.simulate(steps, start, goal=goal, max_steps=20,
                               state_edits=state_edits or None,
                               world_edits=world_edits or None)
        # Name the changed assumption(s) with their actual values.
        assumptions = ["simulated: this is a counterfactual simulation, not observed history"]
        for key, value in state_edits.items():
            assumptions.append(
                f"simulated: assumed {key}={value!r} (actual value was "
                f"{start.get(key)!r})")
        for item in world_edits:
            ent = self.world_model.entity(item["entity"])
            old = None
            if ent is not None and item["property"] in (ent.properties or {}):
                old = ent.properties[item["property"]].value
            assumptions.append(
                f"simulated: assumed world-model fact "
                f"{item['entity']}.{item['property']}={item['value']!r} "
                f"(actual value was {old!r})")
        # Phase 9 coordinator determinism fix: build the difference
        # dict in SORTED key order. It was built from set-union
        # iteration, whose order varies per process (hash
        # randomisation), so the counterfactual answer/detail strings
        # differed between identical runs -- the only non-timing,
        # non-temp-path difference in the same-seed reproducibility
        # check. Sorting changes no semantics (callers read it by key).
        difference = {k: (baseline.predicted_state.get(k),
                          changed.predicted_state.get(k))
                      for k in sorted(set(baseline.predicted_state)
                                      | set(changed.predicted_state))
                      if baseline.predicted_state.get(k) != changed.predicted_state.get(k)}
        outcome_changed = bool(difference) or baseline.met_goal != changed.met_goal
        edit_text = ", ".join(
            [f"{k}={v!r}" for k, v in state_edits.items()]
            + [f"{e['entity']}.{e['property']}={e['value']!r}" for e in world_edits])
        answer = (f"(simulated) If {edit_text or 'nothing changed'}, the route "
                  f"{' > '.join(steps) or '(no steps)'} is predicted to end at "
                  f"{changed.predicted_state} instead of the baseline "
                  f"{baseline.predicted_state}"
                  + (f" — the outcome changes: {difference}."
                     if outcome_changed else " — the outcome does not change."))
        result = {"edits": dict(state_edits), "world_edits": list(world_edits),
                  "steps": list(steps), "goal": goal,
                  "baseline": baseline.to_dict(),
                  "counterfactual": changed.to_dict(),
                  "difference": difference,
                  "outcome_changed": bool(outcome_changed),
                  "assumptions": assumptions,
                  "simulated": True,
                  "answer": answer}
        self._event("counterfactual",
                    f"counterfactual {edit_text or 'no edits'}: outcome "
                    f"{'changed' if outcome_changed else 'unchanged'} "
                    f"{difference} (simulated, not observed)")
        return result

    def _rank_plans_by_simulation(self, plans, target) -> list:
        """Phase 8 (P8-6): simulation-based planning. Simulate the
        candidate plans (bounded: at most 3 scenarios, at most 20
        simulated steps each) and prefer the best simulated one
        (predicted success, then predicted cost, then risk). The
        planner's cost ordering is the fallback when the simulations
        cannot discriminate (equal success and cost). The comparison
        is recorded in a ``simulation_compared`` event and in
        ``last_simulation_comparison``; the best simulation is kept
        as ``_last_imagined`` for the revision loop. A simulation
        failure never breaks planning — the planner order stands."""
        if len(plans or []) < 2:
            return list(plans or [])
        try:
            sim = self._get_simulator()
            sims = []
            for plan in list(plans)[:3]:
                steps = [a.name for a in plan.steps][:20]
                sims.append(sim.simulate(steps, dict(self.world),
                                         goal=dict(target), max_steps=20))
        except Exception as exc:  # noqa: BLE001 - simulation must never break planning
            self._event("simulation_failed",
                        f"simulation-based planning skipped: {exc!r}; "
                        f"keeping planner cost order")
            return list(plans)
        discriminates = (len({round(s.predicted_success, 9) for s in sims}) > 1
                         or len({round(s.predicted_cost, 9) for s in sims}) > 1)
        order = sorted(range(len(sims)),
                       key=lambda i: (-sims[i].predicted_success,
                                      sims[i].predicted_cost,
                                      sims[i].risk, i))
        ranked_plans = [list(plans)[i] for i in order]
        best = sims[order[0]]
        self.last_simulation_comparison = {
            "scenarios": [{"steps": list(s.steps),
                           "predicted_success": s.predicted_success,
                           "predicted_cost": s.predicted_cost,
                           "risk": s.risk,
                           "met_goal": s.met_goal}
                          for s in sims],
            "chosen_index": int(order[0]),
            "discriminates": bool(discriminates),
            "simulated": True,
        }
        self._last_imagined = best.to_dict()
        summary = "; ".join(
            f"{' > '.join(s.steps) or '(already true)'}: success "
            f"{s.predicted_success:.2f} cost {s.predicted_cost:.2f}"
            for s in sims)
        if discriminates:
            self._event("simulation_compared",
                        f"simulated {len(sims)} candidate plan(s): {summary}; "
                        f"chose {' > '.join(best.steps) or '(already true)'} "
                        f"by predicted success then cost "
                        f"(simulated, not observed)")
            return ranked_plans
        self._event("simulation_compared",
                    f"simulated {len(sims)} candidate plan(s): {summary}; "
                    f"simulations cannot discriminate, keeping planner cost "
                    f"order (simulated, not observed)")
        return list(plans)

    def _reason_from_wm(self) -> list[dict]:
        """Phase 4: working-memory reasoning goes through the unified
        ``ReasoningEngine`` (deduction over verified facts, induction
        from episodes, abduction for prediction violations, constraint
        checks, contradiction detection), plus Brain-level property
        inheritance for functional relations (see
        ``_inherited_derivations``).

        Derived conclusions (status "derived" only) become inferred
        thoughts in WM and ``inferred:<subject>|<relation>`` state
        entries (furthest derived value per subject/relation), with the
        proof chain recorded in ``self.last_inferences`` — exactly the
        contract the old method kept. Hypotheses go to
        ``hypothesis:*`` state entries and NEVER to LTM, and never
        drive an action (no ``inferred:*`` entry is written for one).
        Contradictions are handled by ``_handle_contradictions``
        (disputed + research). Engine events are propagated into the
        Brain's event log, and a spent reasoning budget emits
        ``reasoning_budget_exhausted``. Returns the derivation dicts
        (subject / relation / value / proof / confidence / status).
        """
        contents = self.wm.contents()
        subjects_known = {f.subject for f in self.ltm.all_facts()}
        subjects: set[str] = set()
        for t in contents:
            if t.kind not in ("percept", "goal") or not isinstance(t.content, str):
                continue
            candidates = {normalize_text(t.content)}
            candidates |= {w for w in tokenize(t.content) if w in subjects_known}
            subjects |= {c for c in candidates if c in subjects_known}
        goal = self.goals.active()
        goal_targets: list[dict] = []
        if goal is not None:
            for k in goal.target:
                m = re.match(r"known:(.+)\|(.+)", k)
                if m:
                    goal_targets.append({"subject": m.group(1), "relation": m.group(2)})
        episodes_fn = getattr(self.learner, "episodes_for_induction", None)
        if callable(episodes_fn):
            episodes = episodes_fn()
        else:
            # Learner-shaped stand-ins (e.g. the no-learning variant)
            # expose only .episodes; normalise the same way.
            episodes = [{"action": ep.action, "situation": dict(ep.situation),
                         "outcome": "success" if ep.ok else "failure"}
                        for ep in getattr(self.learner, "episodes", [])]
        predictions = list(self._last_violations)
        self._last_violations = []
        # Constraints on derivations: a minimum-confidence floor, plus
        # one exclusion per banned action (a derivation naming a banned
        # action must not drive planning). Action preconditions are
        # state-shaped, not derivation-shaped, so they are enforced by
        # the planner's applicability check, not here; the step budget
        # is the engine's own max_steps.
        constraints = [{"key": "confidence", "op": ">=", "value": self.control.min_confidence}]
        for name in self._banned():
            constraints.append({"key": "subject", "op": "!=", "value": name})
        result = self.reasoning.reason(sorted(subjects), self._world_facts_fn,
                                       goal_targets=goal_targets, episodes=episodes,
                                       predictions=predictions, constraints=constraints)
        for kind, detail in result.events:
            self._event(kind, detail)
        if result.budget_exhausted and not any(
                k == "reasoning_budget_exhausted" for k, _ in result.events):
            self._event("reasoning_budget_exhausted",
                        f"reasoning budget of {self.reasoning.max_steps} steps spent")
        derivations = list(result.derivations)
        derivations.extend(self._inherited_derivations(derivations, self._facts_fn, constraints))
        contradictions = self.reasoning.detect_contradictions(derivations, self._facts_fn)
        # LTM-level contradictions: the engine only checks derivations
        # against facts, but two stored facts can already conflict on a
        # functional relation (the LTM conflict path marks both
        # disputed at storage). When the brain attends to such a
        # subject in-cycle, that conflict is a contradiction too — it
        # must surface here (event + research goal), not stay silent
        # in storage. The non-integrated variant never runs this
        # method, so it still detects nothing in-cycle.
        contradictions.extend(self._ltm_contradictions(subjects))
        # Phase 5: contradictions recorded in the world model (via
        # note_contradiction — an explicit, same-cycle conflict, never
        # a sequential state change, which is an observation or an
        # exogenous event instead) go through the same
        # disputed+research policy as every other contradiction.
        for rec in self.world_model.contradictions():
            contradictions.append(Contradiction(
                claim_a=(str(rec.get("entity", "")), str(rec.get("property", "")),
                         str(rec.get("value_a", ""))),
                claim_b=(str(rec.get("entity", "")), str(rec.get("property", "")),
                         str(rec.get("value_b", ""))),
                reason=(f"world-model observations disagree on "
                        f"{rec.get('entity')} {rec.get('property')}: "
                        f"{rec.get('value_a')!r} vs {rec.get('value_b')!r}"),
                policy="disputed+research"))
        seen_pairs: set[tuple] = set()
        deduped = []
        for c in contradictions:
            pair = tuple(sorted([tuple(c.claim_a), tuple(c.claim_b)]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            deduped.append(c)
        contradictions = deduped
        # One dict per (subject, relation, value); disputed derivations
        # are reported but never written to WM or inferred:* state.
        derivations_out: list[dict] = []
        seen_keys: set[tuple] = set()
        for d in derivations:
            key = (normalize_text(d.subject), normalize_relation(d.relation),
                   normalize_text(str(d.value)))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            proof_str = d.proof.explain() if d.proof is not None else ""
            derivations_out.append({"subject": key[0], "relation": key[1],
                                    "value": key[2], "proof": proof_str,
                                    "confidence": d.confidence, "status": d.status})
            if d.status != "derived":
                continue
            content = f"{key[0]} {key[1]} {key[2]}"
            if not any(x.content == content for x in self.wm.contents()):
                inferred = Thought(content, "inferred", "inference",
                                   0.5, d.confidence,
                                   tokenize(content), self.cycle)
                inferred.salience = 0.5 + 0.5 * d.confidence
                self.wm.admit(inferred)
        # One state entry per (subject, relation): the furthest derived value.
        furthest: dict[tuple, dict] = {}
        for d in derivations_out:
            if d["status"] == "derived":
                furthest[(d["subject"], d["relation"])] = d
        for (subj, rel), d in sorted(furthest.items()):
            self._set_state(f"inferred:{subj}|{rel}", d["value"],
                            source=f"inference:{d['proof']}", confidence=d["confidence"],
                            provenance=d["proof"], entry_type="inferred")
        self.last_inferences = derivations_out
        # Phase 10 (P10-6): in-cycle analogy. For each goal target
        # whose (subject, relation) has no verified fact and no
        # derivation this cycle, and whose subject is in WM, ask the
        # engine's analogize over LTM + this cycle's derived facts.
        # The returned hypotheses go through the SAME hypothesis path
        # below (hypothesis:* state entry, test = observation /
        # research, confidence capped at 0.4 by analogize itself):
        # never verified alone, never driving an action (no
        # inferred:* entry, no plan step). analogize itself is
        # untouched.
        analogy_hypotheses = self._analogy_hypotheses(
            goal_targets, subjects, derivations_out)
        # Hypotheses: state entries only, never LTM, never inferred:*.
        # The surprise outcome each hypothesis explains is attached
        # ("outcome") so a later repeat of the action can test it (see
        # _test_hypotheses) — the stored dict is bookkeeping, not a fact.
        # A hypothesis the engine re-abduces on a later cycle NEVER
        # overwrites one already held: a corroborated/falsified
        # hypothesis keeps its tested status, confidence and evidence
        # (fresh evidence is merged in, not replaced), so the outcome
        # of a test the brain already ran is not silently reset to open.
        self.last_hypotheses = []
        surprise_outcome = predictions[0].get("actual") if predictions else None
        for h in list(result.hypotheses) + analogy_hypotheses:
            hd = {"id": h.id, "subject": h.subject, "relation": h.relation,
                  "value": h.value, "confidence": h.confidence, "source": h.source,
                  "evidence_for": list(h.evidence_for),
                  "evidence_against": list(h.evidence_against),
                  "status": h.status, "test": h.test,
                  "outcome": surprise_outcome}
            if h.source == "analogy":
                # An analogy hypothesis explains no surprise outcome:
                # it must never be corroborated/falsified by an action
                # outcome test (see _test_hypotheses, which skips
                # outcome None). Its test is observation/research.
                hd["outcome"] = None
            existing = self.state.get(f"hypothesis:{h.id}")
            if existing is not None and isinstance(existing.value, dict):
                prev = existing.value
                hd["status"] = prev.get("status", h.status)
                hd["confidence"] = prev.get("confidence", h.confidence)
                for key in ("evidence_for", "evidence_against"):
                    merged = list(prev.get(key, []))
                    merged.extend(x for x in hd[key] if x not in merged)
                    hd[key] = merged
                hd["outcome"] = prev.get("outcome", surprise_outcome)
                if hd == prev:
                    self.last_hypotheses.append(dict(prev))
                    continue
            self.last_hypotheses.append(hd)
            self._set_state(f"hypothesis:{h.id}", hd, source=f"hypothesis:{h.source}",
                            confidence=hd["confidence"], provenance=h.test or "",
                            entry_type="hypothesis", status=hd["status"])
        # Induction from repeated SUCCESSES proposes pending rules
        # (failure induction is the Learner's job and is not duplicated
        # here). Proposed rules are pending only — never auto-approved —
        # and surface through pending_rules()/approve_rule().
        for prop in self.reasoning.induce(episodes):
            if prop["outcome"] == "failure":
                continue
            self._set_state(f"induced_rule:{prop['id']}",
                            {**prop, "status": "pending"},
                            source="induction", confidence=0.5,
                            provenance="induction", entry_type="belief",
                            status="pending")
        self._abstract_categories(subjects)
        self.last_contradictions = [
            {"claim_a": list(c.claim_a), "claim_b": list(c.claim_b),
             "reason": c.reason, "policy": c.policy} for c in contradictions]
        if contradictions:
            self._handle_contradictions(contradictions)
        if result.insufficient_evidence and result.needed:
            self._event("insufficient_evidence", result.needed)
        self._event("reason", f"{len(contents)} WM thoughts, {len(derivations_out)} derivation(s), "
                              f"{len(self.last_hypotheses)} hypothesi(e)s, "
                              f"{len(contradictions)} contradiction(s)")
        return derivations_out

    def _analogy_hypotheses(self, goal_targets, subjects, derivations_out) -> list:
        """Phase 10 (P10-6): in-cycle analogy proposals, one per goal
        target with no verified fact and no derivation for its
        relation, whose subject is in working memory.

        The facts function covers LTM (verified facts + world-model
        relations via ``_world_facts_fn``) PLUS this cycle's derived
        facts. Derived facts are presented as dicts with status
        ``"verified"`` because the engine's facts gate admits only
        that status — with an explicit ``"derived": True`` marker and
        ``derived:<proof>`` provenance so the origin stays visible.
        That is a bounded exception, not a laundering path: the
        result is only ever a Hypothesis (confidence capped at 0.4
        by ``analogize``), which the hypothesis path below stores as
        a state entry and never as an LTM fact, and which never
        drives an action. A derived fact can therefore only suggest
        something to check, never establish it."""
        if not goal_targets:
            return []
        derived_by_subject: dict[str, list] = {}
        for d in derivations_out:
            if d.get("status") == "derived":
                derived_by_subject.setdefault(d["subject"], []).append(d)

        def _analogy_facts(subject, relation=None):
            def _as_dict(d):
                return {"subject": d["subject"], "relation": d["relation"],
                        "value": d["value"], "status": "verified",
                        "derived": True,
                        "provenance": f"derived:{d.get('proof', '')}"}
            if subject is None:
                facts = list(self._world_facts_fn(None, relation))
                facts.extend(_as_dict(d) for d in derivations_out
                             if d.get("status") == "derived")
                return facts
            facts = list(self._world_facts_fn(subject, relation))
            for d in derived_by_subject.get(normalize_text(subject), []):
                if (relation is not None
                        and normalize_relation(d["relation"]) != normalize_relation(relation)):
                    continue
                facts.append(_as_dict(d))
            return facts

        out = []
        for target in goal_targets:
            subj = normalize_text(target.get("subject", ""))
            rel = normalize_relation(target.get("relation", ""))
            if not subj or not rel or subj not in subjects:
                continue
            if self._facts_fn(subj, rel):
                continue
            if any(d["subject"] == subj and d["relation"] == rel
                   for d in derivations_out if d.get("status") == "derived"):
                continue
            hyp = self.reasoning.analogize(subj, rel, _analogy_facts)
            if hyp is not None:
                out.append(hyp)
        return out

    def _maybe_switch_goal(self) -> None:
        """Phase 2: a higher-priority goal preempts by SUSPENDING the
        current goal (its plan position is preserved for resume_goal),
        not by destroying it. The challenger must clear the switch
        margin (0.15) and have a feasible plan (feasibility > 0), and
        the switch itself goes through the same DecisionMaker path as
        goal selection, recorded in last_decision. Suspended goals are
        excluded from challenging."""
        cur = self.goals.active()
        if cur is None:
            return
        pend = [g for g in self._selectable_pending() if g.id != cur.id]
        if not pend:
            return
        challenger = max(pend, key=lambda g: g.priority)
        if not self.control.should_switch(cur.priority, challenger.priority):
            return
        # Phase 12 (P12-O4): one world build and one banned-set for
        # both plans — planner.plan does not mutate brain state, so
        # the second call would recompute identical inputs.
        world = self.world
        banned = self._banned()
        c_plan = self.planner.plan(world, challenger.target, banned)
        c_feas = feasibility_for_cost(c_plan.cost) if c_plan else 0.0
        if c_feas <= 0 or c_feas < FEASIBILITY_MIN:
            # Phase 3 literal feasibility gate: never preempt for a plan
            # too expensive to attempt (feasibility < 0.25).
            return
        cur_plan = self.planner.plan(world, cur.target, banned)
        cur_feas = feasibility_for_cost(cur_plan.cost) if cur_plan else 0.0
        opts = [
            Option(cur.description, {"priority": cur.priority,
                                     "urgency": self._goal_urgency(cur, self.cycle),
                                     "feasibility": cur_feas}, cur),
            Option(challenger.description, {"priority": challenger.priority,
                                            "urgency": self._goal_urgency(challenger, self.cycle),
                                            "feasibility": c_feas}, challenger),
        ]
        d = self.decider.decide_with_evidence(
            opts, feasibility=c_feas, fact_confidence=self._fact_confidence(),
            chosen_goal_id=challenger.id,
            reasoning_confidence=self._fact_confidence())
        self.last_decision = d
        if d.chosen is None or d.chosen.payload is not challenger:
            return
        self._interrupt_current_goal(
            "higher_priority_goal",
            f"'{challenger.description}' (priority {challenger.priority}) "
            f"outranks '{cur.description}' ({cur.priority})")
        self.goals.activate(challenger.id)
        self._set_plan([])
        self.surprise, self.plan_failed = False, False
        self._plan_started_len = 0
        self._steps_executed = 0
        self.control.reset_progress()
        self.goal_start = dict(self.world)
        self._event("switch", f"'{cur.description}' -> '{challenger.description}' (clearly higher priority)")

    def _cognitive_state(self, world: dict | None = None) -> dict:
        # Phase 12 (P12-O4): ``world`` may be passed in by callers
        # that already built the view (step(), _banned()) so the
        # world is not rebuilt here and again by the caller for the
        # same state. When omitted it is built exactly as before.
        g = self.goals.active()
        if world is None:
            world = self.world
        return {"has_goal": g is not None, "has_plan": bool(self.plan),
                "goal_satisfied": bool(g) and self.goals.satisfied(g, world),
                "surprise": self.surprise, "plan_failed": self.plan_failed,
                # Phase 2: stall is goal progress (see _goal_progress),
                # not world-string churn.
                "stalled": self.control.is_stalled_by_progress()}

    def _banned(self) -> set[str]:
        # Phase 4: match against the merged world + cognitive state, so
        # a rule conditioned on has_goal / stalled / surprise /
        # plan_failed can ban an action (matching on self.world alone
        # could never see those keys).
        # Phase 12 (P12-O4): the world view is built once and shared
        # with _cognitive_state — previously it was built twice here
        # (once inside _cognitive_state, once for the match dict)
        # from identical state, so this is the same match input.
        world = self.world
        cog = self._cognitive_state(world)
        banned = set()
        for r in self.rules.match({**world, **cog}):
            for a in r.then:
                if a["do"] == "avoid_action":
                    banned.add(a["args"]["name"])
        return banned

    # ---- action handlers (return True when the cognitive operation achieved its purpose)
    def _do_select_goal(self, args, env) -> bool:
        """Phase 3: ONE decision path. Every goal selection goes through
        DecisionMaker.decide_with_evidence and returns a Decision
        object recorded in last_decision; no parallel utility ranking
        or log-string decision remains (the best option for the
        missing-information request is read back from the Decision's
        own ranking)."""
        pend = self._selectable_pending()
        if not pend:
            return False
        # Phase 12 (P12-O4): one world build and one banned-set for
        # the whole loop — planner.plan does not mutate brain state,
        # so every iteration would recompute identical inputs.
        world = self.world
        banned = self._banned()
        opts = []
        for g in pend:
            plan = self.planner.plan(world, g.target, banned)
            opts.append(Option(g.description, {"priority": g.priority,
                                               "urgency": self._goal_urgency(g, self.cycle),
                                               "feasibility": feasibility_for_cost(plan.cost) if plan else 0.0}, g))
        d = self.decider.decide_with_evidence(
            opts, fact_confidence=self._fact_confidence(),
            reasoning_confidence=self._fact_confidence())
        # Evidence comes from the Decision itself, not a second ranking.
        if d.chosen is not None:
            d.feasibility = float(d.chosen.scores.get("feasibility", 0.0))
            d.chosen_goal_id = (d.chosen.payload.id if getattr(d.chosen.payload, "id", None) is not None
                                else None)
        elif d.ranking:
            top = next((o for o in opts if o.name == d.ranking[0][0]), opts[0])
            d.feasibility = float(top.scores.get("feasibility", 0.0))
        self.last_decision = d
        self._event("decide", d.reason + (" [near-tie]" if d.tie else ""))
        if d.chosen is None:
            # Phase 2 behaviour, kept: the best option scored below the
            # minimum utility — the brain refuses to act on insufficient
            # evidence. Instead of guessing, ask for the missing
            # information and try again next cycle.
            self._event("need_more_evidence", d.reason)
            top = next((o for o in opts if o.name == d.ranking[0][0]), opts[0]) if d.ranking else opts[0]
            self._record_need_percept(top.payload)
            self._request_missing_information(top.payload)
            return False
        # Phase 3 literal feasibility gate (mandate decision, rescaled
        # formula): a plan with feasibility < 0.25 is refused with
        # need_more_evidence even when its utility clears the bar.
        # With feasibility = 1/(1+cost/5) this refuses only plans
        # longer than 15 steps; the 4-step demo treasure (0.56) passes.
        # The gate applies only when a plan EXISTS (feasibility > 0):
        # a goal with no plan at all (feasibility 0.0) is not refused
        # here — it is selected and then fails cleanly as "impossible"
        # when make_plan finds nothing, which is the informative
        # terminal outcome (and the decider's own minimum-utility
        # refusal still applies to it, as in the atlantis case).
        feas = float(d.chosen.scores.get("feasibility", 0.0))
        if 0.0 < feas < FEASIBILITY_MIN:
            d.chosen = None
            d.chosen_goal_id = None
            d.reason = (f"best plan feasibility {feas:.2f} < {FEASIBILITY_MIN}: "
                        f"need more evidence")
            self.last_decision = d
            self._event("need_more_evidence", d.reason)
            _top = next((o for o in opts if o.name == d.ranking[0][0]), opts[0])
            self._record_need_percept(_top.payload)
            self._request_missing_information(_top.payload)
            return False
        if d.tie:
            if self._tie_deferred_cycle == self.cycle - 1:
                # Deferred last cycle; decide now rather than deferring
                # forever. The choice is still the decider's, recorded.
                self._tie_deferred_cycle = None
            else:
                self._tie_deferred_cycle = self.cycle
                self._event("near_tie",
                            f"deferred choice between '{d.ranking[0][0]}' and "
                            f"'{d.ranking[1][0] if len(d.ranking) > 1 else ''}'; "
                            f"requesting separating information")
                self._request_missing_information(d.chosen.payload)
                return False
        g = d.chosen.payload
        self.goals.activate(g.id)
        self._set_plan([])
        self.surprise, self.plan_failed = False, False
        self._plan_started_len = 0
        self._steps_executed = 0
        self.control.reset_progress()
        self.last_stop_reason = None
        self.goal_start = dict(self.world)
        return True

    def _validated_plans(self, plans) -> list:
        """Phase 5: keep only the plans the world model validates.
        ``check_plan`` simulates each plan's effects forward through
        the planner's effects function; every intermediate state must
        satisfy the model's constraints. A rejected plan is logged
        with the violated constraint, never silently dropped. Returns
        the surviving plans, in order.

        Phase 7: permission-aware filtering runs FIRST (defence in
        depth — the call-site refusal in ``ToolRegistry.call`` and
        ``_do_execute_step`` stays as the second line). A candidate
        plan is rejected when any step names a tool whose required
        grant is not in ``self.granted`` (read from the registry spec)
        or an env action when ``"environment"`` is not granted, with a
        ``plan_tool_denied`` event naming the tool and the missing
        grant; the rejection is also recorded in
        ``_last_permission_denials`` so ``_do_make_plan`` can fail the
        goal as ``no_permitted_plan`` when no permitted plan exists.
        An env-action denial also logs a ``deny`` event (the Phase 1
        security contract: a denial is recorded and the environment
        is never acted on)."""
        out = []
        for p in (plans or []):
            permitted, denied_tool, missing_grant = self._plan_permitted(p)
            if not permitted:
                self._last_permission_denials.append((denied_tool, missing_grant))
                self._event("plan_tool_denied",
                            f"plan {' > '.join(a.name for a in p.steps)} "
                            f"rejected: tool '{denied_tool}' requires grant "
                            f"'{missing_grant}' (granted={sorted(self.granted)})")
                if missing_grant == "environment":
                    self._event("deny",
                                f"env action '{denied_tool}' denied "
                                f"(granted={sorted(self.granted)})")
                continue
            ok, violated = self.world_model.check_plan(
                p.steps, self.world, self.planner.effects_fn)
            if ok:
                out.append(p)
            else:
                self._event("plan_rejected",
                            f"plan {' > '.join(a.name for a in p.steps)} "
                            f"rejected by world model: {violated}")
        return out

    def _plan_violation(self, plan) -> str | None:
        """Phase 6 §4.1: the first constraint a plan violates in the
        world model, or None when the plan is valid. ``check_plan``
        returns the violated constraint name; a plan that is valid
        under the model has none."""
        ok, violated = self.world_model.check_plan(
            plan.steps, self.world, self.planner.effects_fn)
        return None if ok else violated

    @staticmethod
    def _action_tool_name(action) -> str | None:
        """The tool entity an action uses, by precondition key shape
        (``has_<entity>``) first, else by the ``use_<entity>`` action
        name shape. None when the action names no tool."""
        if action is None:
            return None
        pre = getattr(action, "pre", None) or {}
        for k in pre:
            if str(k).startswith("has_") and len(str(k)) > 4:
                return str(k)[4:]
        return _tool_entity_name(getattr(action, "name", ""))

    def _log_pattern_applications(self, steps) -> None:
        """Phase 5: every pattern cost application in the chosen plan
        is visible — one ``pattern_applied`` event per penalised step,
        naming the pattern id(s) covering the step's effect and the
        multiplier. (Patterns covering an effect the step claims are
        exactly the ones ``match`` can fire for it.)"""
        for step in (steps or []):
            mult = self.planner.pattern_multiplier(step)
            if mult == 1.0:
                continue
            ids = [p.id for p in self.patterns.patterns()
                   if p.effect in step.eff]
            self._event("pattern_applied",
                        f"pattern(s) {','.join(i[:12] for i in ids)} "
                        f"penalised '{step.name}' x{mult} "
                        f"for effect(s) {sorted(set(step.eff) & {p.effect for p in self.patterns.patterns()})}")

    def _starts_with_banned_retry(self, plan) -> bool:
        """Phase 5: does ``plan`` start by retrying an action that is
        banned in the CURRENT situation (its preconditions' values)?
        Such a first step reruns the action in exactly the situation
        it just failed in — in a deterministic world that repeats the
        failure, it is not a recovery. The P4-9 relaxation exists so
        a plan can do other steps first (which may change the world)
        and retry the banned action afterwards."""
        if not plan.steps:
            return False
        first = plan.steps[0]
        return self.planner.is_banned(
            first.name, {k: self.world.get(k) for k in first.pre},
            getattr(first, "args", None) or {}, getattr(first, "tool", None))

    def _plan_is_loop_on_twice_failed_step(self, plan) -> bool:
        """Phase 10.5 (P10.5-7): True when some step of ``plan``
        would retry a (name, args, situation) triple that has
        already failed >= 2 times in the identical situation AND
        the plan's prefix leaves the world (net) exactly as it is
        now. Both halves matter. The first identifies the triple
        (precondition-restricted situation, as episodes record it).
        The second is the hidden-precondition guard: a retry is a
        genuine recovery attempt when an earlier step changed the
        world first — warming a cold coil before heating it — even
        though the coil is not one of heat's preconditions and so
        is invisible to the first half. A go-away-and-back prefix
        (the measured unrecoverable shape) changes the world and
        changes it back, so the net state is unchanged and the
        retry is still a loop. A step that is not applicable in
        the simulated state is a model issue, not a loop (False).
        The threshold 2 matches the repeated-mistake / strategy
        exhaustion thresholds. Args are redacted exactly as episode
        recording redacts them, so the lookup matches the stored
        counts (a secret-arg action's failures are filed under its
        redacted identity). The net-state half makes the rule
        conservative by design: a prefix that changes an
        irrelevant key lets one more retry through, and the stall
        detector is the backstop for that case — a false positive
        (failing a recoverable goal as impossible) would be worse
        than one extra failure."""
        now = dict(self.world)
        state = dict(now)
        for step in plan.steps:
            if not self.planner.applicable(step, state):
                return False
            situation = {k: state.get(k) for k in step.pre}
            args = self._redact_tool_args(
                getattr(step, "tool", None), dict(getattr(step, "args", None) or {}))
            if (state == now
                    and self.learner.failures_for(step.name, situation, args) >= 2):
                return True
            state.update(step.eff)
        return False

    def _retry_failed_subgoal(self, parent: Goal, subs, failed_subs) -> bool:
        """Retry one failed subgoal by re-decomposing its key into a
        fresh subgoal (at most once per key, tracked in the parent's
        meta). Returns True when the fresh subgoal got a plan and was
        activated (the caller returns True); False when no retry is
        available or the fresh subgoal is not plannable either."""
        retries = dict((parent.meta or {}).get("subgoal_retries", {}))
        for sg in failed_subs:
            key = next(iter(sg.target), None)
            if key is None or retries.get(key, 0) >= 1:
                continue
            retries[key] = retries.get(key, 0) + 1
            parent.meta["subgoal_retries"] = retries
            fresh = Goal(f"{parent.description} :: {key} (retry)",
                         dict(sg.target), priority=parent.priority)
            self.goals.decompose(parent.id, [fresh])
            self._event("subgoal_retry",
                        f"subgoal '{sg.description}' failed; retrying key "
                        f"'{key}' as '{fresh.description}'")
            sub_plans = self._validated_plans(
                self.planner.plan_all(self.world, fresh.target, self._banned(), limit=1))
            if not sub_plans:
                continue
            self.goals.activate(fresh.id)
            self._set_plan(list(sub_plans[0].steps))
            self.plan_failed = False
            self._plan_started_len = len(self.plan)
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self._event("plan", f"via retried subgoal '{fresh.description}': "
                                f"{' > '.join(a.name for a in sub_plans[0].steps) or '(already true)'} "
                                f"(cost {sub_plans[0].cost})")
            self._log_pattern_applications(sub_plans[0].steps)
            self._open_execution_session(fresh, sub_plans[0].steps)
            return True
        return False

    def _do_make_plan(self, args, env) -> bool:
        g = self.goals.active()
        # Phase 7: permission denials are per planning pass.
        self._last_permission_denials = []
        # Phase 6 §3.5: the repeated-mistake flag is one-cycle and is
        # consumed by THIS planning pass, whichever path it takes.
        prefer_alternative = self._prefer_alternative
        self._prefer_alternative = False
        failed_action = self._repeated_mistake_action
        self._repeated_mistake_action = None
        plans = self._validated_plans(
            self.planner.plan_all(self.world, g.target, self._banned(), limit=2))
        self.surprise = False
        if not plans:
            # Phase 4 last resort: no unbanned plan exists. Retry with
            # (action, situation) pair bans relaxed — a pair ban records
            # one failure in one situation, not a physical impossibility,
            # and an earlier step may since have changed the world (e.g.
            # warming a coil the model does not represent). Rule-based
            # name bans and approved avoid_action rules stay enforced.
            # The plan is flagged, so the retry is visible, not silent.
            raw = self.planner.plan_all(self.world, g.target, self._banned(), limit=2,
                                        allow_banned_pairs=True)
            # Phase 5: prefer relaxed plans that do something else
            # first over ones that start by retrying a banned action
            # in the unchanged situation (see
            # _starts_with_banned_retry). Fall back to the full
            # relaxed list when every candidate starts that way —
            # the old P4-9 behaviour as a last resort — but ORDERED
            # by first-step failure rate, then failure COUNT, then
            # cost: plan cost is order-independent (the same actions
            # in a different order cost the same), so cost order
            # cannot distinguish a first step that failed once
            # flakily from one that failed every time. The rate
            # alone cannot either (1/1 and 3/3 are both 1.0); the
            # count can — one failure is weak evidence, three
            # consistent failures are strong. (Coordinator fix,
            # Phase 7 gate: in the flaky-tool run the fallback kept
            # re-adopting the press-first plan — three press
            # failures — over the cut-first plan whose cut had failed
            # only once; the brain re-failed press until it stalled.)
            relaxed = [p for p in raw if not self._starts_with_banned_retry(p)]
            if not relaxed:
                # Phase 10.5 (P10.5-5): the first-step rate and count
                # are the VARIANT-aware ones (``failure_rate_for`` /
                # ``failure_counts_for``: exact variant, then the
                # P10.5-8 key-shape fallback) — a pooled rate would
                # rank a never-failed variant by its failed sibling's
                # record, which is the P9-7 defect in ordering form.
                relaxed = sorted(
                    raw,
                    key=lambda p: (
                        self.planner.failure_rate_for(p.steps[0])
                        if p.steps else 0.0,
                        self.planner.failure_counts_for(p.steps[0])[1]
                        if p.steps else 0,
                        p.cost))
            # Phase 6 §3.5(3): after a repeated mistake, prefer a
            # relaxed plan whose first step uses a DIFFERENT tool
            # from the failed action's tool — a different tool is a
            # different strategy; retrying the same tool is not.
            if prefer_alternative and failed_action is not None:
                failed_tool = self._action_tool_name(
                    next((a for a in self.planner.actions
                          if a.name == failed_action), None))
                relaxed = sorted(
                    relaxed,
                    key=lambda p: (0 if not p.steps or
                                   self._action_tool_name(p.steps[0]) != failed_tool
                                   else 1))
            plans = self._validated_plans(relaxed)
            if plans:
                self._event("plan_uses_banned_action",
                            f"no unbanned plan for '{g.description}'; retrying a "
                            f"banned action: {' > '.join(a.name for a in plans[0].steps)}")
            elif raw:
                # Phase 6 §4.1 (narrowing): EVERY candidate failed the
                # world-model check. That is model/planner
                # disagreement — EXCEPT when a violated constraint is
                # a LEARNED door-gated edge: the model learned the
                # gate from an observed failed entry (see
                # note_failed_entry), so executing a plan the model
                # knows is invalid is not a disagreement to paper
                # over. The goal fails as impossible instead, with
                # the reason naming the learned constraint. Built-in
                # constraints (containment) and checks failed on an
                # unobserved key keep the old fallback: the
                # planner's best plan is kept and flagged.
                violated = [self._plan_violation(p) for p in relaxed]
                if any(v == "door_blocks_entry" for v in violated):
                    self._event(
                        "model_disagreement_learned_constraint",
                        f"all plans failed the world-model check for "
                        f"'{g.description}' on a learned door-gated edge "
                        f"(violated: {sorted({v for v in violated if v})}); "
                        f"failing the goal instead of executing a plan "
                        f"the model knows is invalid")
                    self._do_fail_goal(
                        {"reason": "model_disagreement_learned_constraint"}, None)
                    return False
                # Phase 5: EVERY candidate failed the world-model
                # check. That is model/planner disagreement, not proof
                # that no plan exists: Worker 1's built-in
                # door_blocks_entry rejects ANY room change while a
                # door is closed, which is broader than design §2.7
                # (only a door-gated edge requires the door open) —
                # every RoomWorld treasure plan starts with a move
                # that needs no door. The planner's best plan is kept
                # and flagged, instead of failing the goal for a
                # disagreement between the two models.
                # Phase 7: the kept plan must still be permitted — a
                # permission-denied plan is never kept as the
                # disagreement fallback. When every candidate was
                # permission-denied, the goal fails as
                # no_permitted_plan instead.
                permitted_relaxed = [p for p in relaxed
                                     if self._plan_permitted(p)[0]]
                if permitted_relaxed:
                    plans = [permitted_relaxed[0]]
                    self._event("plan_model_disagreement",
                                f"all plans failed the world-model check "
                                f"for '{g.description}'; keeping the "
                                f"planner's best plan "
                                f"{' > '.join(a.name for a in permitted_relaxed[0].steps)} "
                                f"despite the disagreement")
                elif self._last_permission_denials:
                    self._do_fail_goal({"reason": "no_permitted_plan"}, None)
                    return False
                else:
                    plans = [relaxed[0]]
                    self._event("plan_model_disagreement",
                                f"all plans failed the world-model check "
                                f"for '{g.description}'; keeping the "
                                f"planner's best plan "
                                f"{' > '.join(a.name for a in relaxed[0].steps)} "
                                f"despite the disagreement")
        # Phase 10.5 (P10.5-7): loop-plan rule. Whether the plans
        # came from the normal path or the last-resort relaxation,
        # drop any plan that would retry a twice-failed (action,
        # args, situation) triple in the identical situation
        # (``_plan_is_loop_on_twice_failed_step`` simulates forward
        # through the prefix effects, so a plan whose earlier steps
        # change the world before the retry is NOT a loop). When
        # every available plan is a loop plan, executing one is not
        # a recovery attempt — the goal is unrecoverable, and it
        # fails as ``impossible`` instead of executing another
        # doomed retry and waiting for the stall detector. The
        # threshold 2 matches the repeated-mistake / strategy
        # exhaustion thresholds.
        if plans:
            live = [p for p in plans
                    if not self._plan_is_loop_on_twice_failed_step(p)]
            if live and len(live) != len(plans):
                self._event("loop_plan_dropped",
                            f"dropped {len(plans) - len(live)} loop plan(s) "
                            f"for '{g.description}' (retry of a twice-failed "
                            f"step in an unchanged situation); keeping "
                            f"{len(live)}")
                plans = live
            elif not live:
                self._event("repeated_failure_no_alternative",
                            f"every plan for '{g.description}' retries a "
                            f"twice-failed step in an unchanged situation; "
                            f"failing the goal as unrecoverable instead of "
                            f"executing another doomed retry")
                self._do_fail_goal(
                    {"reason": "repeated_failure_no_alternative"}, None)
                return False
        if not plans:
            self._set_plan([])
            self.plan_failed = True
            self._plan_started_len = 0
            # Phase 4: no direct plan — decompose a conjunctive goal
            # into single-key subgoals (GoalManager.decompose) and plan
            # for the first plannable subgoal. A goal whose target has
            # a single key cannot be decomposed further and fails exactly
            # as before; a decomposition that yields no plannable
            # subgoal fails the goal as impossible, with that reason
            # recorded. Subgoals created on an earlier attempt are
            # reused, never duplicated.
            subs = [ch for ch in self.goals.all() if ch.parent == g.id]
            if not subs and len(g.target) >= 2:
                # Phase 7: hierarchical decomposition — subgoals in
                # dependency order from the action-derived graph (see
                # ab/planning/hierarchy.py), not declaration order.
                subs, had_cycle = self.hierarchy.decompose(g, self.planner.actions)
                if had_cycle:
                    self._event("hierarchy_cycle",
                                f"dependency cycle in '{g.description}'; "
                                f"subgoals kept in declaration order")
                self._event("plan", f"no direct plan for '{g.description}'; "
                                    f"decomposed into {len(subs)} subgoal(s)")
            # A subgoal that was attempted (done or failed) proves the
            # decomposition made progress; a pending subgoal with no
            # plan at this point is a failed key, not a fresh
            # decomposition — the parent retries it once (below)
            # instead of failing as "no plannable subgoal".
            unplannable: list = []
            for sg in subs:
                if sg.status in ("done", "failed"):
                    continue
                sub_plans = self._validated_plans(
                    self.planner.plan_all(self.world, sg.target, self._banned(), limit=1))
                if not sub_plans:
                    unplannable.append(sg)
                    continue
                self.goals.activate(sg.id)
                self._set_plan(list(sub_plans[0].steps))
                self.plan_failed = False
                self._plan_started_len = len(self.plan)
                self.alternative_plan, self.alternative_plan_cost = [], 0.0
                self._event("plan", f"via subgoal '{sg.description}': "
                                    f"{' > '.join(a.name for a in sub_plans[0].steps) or '(already true)'} "
                                    f"(cost {sub_plans[0].cost})")
                self._log_pattern_applications(sub_plans[0].steps)
                # Phase 7: the subgoal's plan opens (or re-opens) its
                # execution session.
                self._open_execution_session(sg, sub_plans[0].steps)
                return True
            if subs:
                # Phase 7: a failed subgoal is never silent. The parent
                # retries once per failed key by re-decomposing it;
                # when the retry is exhausted the parent fails with
                # reason "subgoal_failed" and its partial results
                # preserved (see _retry_failed_subgoal). A pending
                # subgoal with no plan counts as a failed key for
                # this retry ONLY when a sibling was already attempted
                # (done or failed) — the decomposition made progress
                # and then stuck. When nothing was ever attempted the
                # failure below stays "no plannable subgoal".
                failed_subs = [sg for sg in subs if sg.status == "failed"]
                attempted = any(sg.status in ("done", "failed")
                                for sg in subs)
                retry_candidates = failed_subs + (
                    unplannable if attempted and not failed_subs else [])
                if retry_candidates:
                    if self._retry_failed_subgoal(g, subs, retry_candidates):
                        return True
                    self.alternative_plan, self.alternative_plan_cost = [], 0.0
                    self._event("plan", f"no plan found for '{g.description}' "
                                        f"(subgoal failed and retry exhausted)")
                    self._do_fail_goal({"reason": "subgoal_failed"}, None)
                    return False
                if self._last_permission_denials:
                    self.alternative_plan, self.alternative_plan_cost = [], 0.0
                    self._event("plan", f"no permitted plan for '{g.description}'")
                    self._do_fail_goal({"reason": "no_permitted_plan"}, None)
                    return False
                self.alternative_plan, self.alternative_plan_cost = [], 0.0
                self._event("plan", f"no plan found for '{g.description}' "
                                    f"(decomposition produced no plannable subgoal)")
                self._do_fail_goal({"reason": "no plannable subgoal"}, None)
                return False
            if self._last_permission_denials:
                self.alternative_plan, self.alternative_plan_cost = [], 0.0
                self._event("plan", f"no permitted plan for '{g.description}'")
                self._do_fail_goal({"reason": "no_permitted_plan"}, None)
                return False
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self._event("plan", f"no plan found for '{g.description}'")
            return False
        # Phase 8 (P8-6): simulation-based planning — simulate the
        # candidate plans (bounded: at most 3 scenarios, at most 20
        # simulated steps each) and prefer the best simulated one;
        # the planner's cost ordering remains the fallback when the
        # simulations cannot discriminate. The comparison is recorded
        # in a simulation_compared event (see
        # _rank_plans_by_simulation).
        plans = self._rank_plans_by_simulation(plans, g.target)
        # Phase 6 §3.5(2): after a repeated mistake the brain prefers
        # the retained alternative plan on the next planning pass —
        # the alternative becomes the plan, the failed plan becomes
        # the retained alternative. The change is visible in the
        # plan event below (the plan differs from the failed one).
        # Coordinator integration fix (Phase 7 gate): the swap is
        # CONDITIONAL, not blind. The old code swapped plans[0] and
        # plans[1] unconditionally, assuming plans[0] was the failed
        # plan — but the relaxation path above already reorders by
        # tool (Phase 6 §3.5(3)), so after a relaxation plans[0] can
        # ALREADY be the alternative. The blind swap then undid the
        # sort and put the failed plan back on top: in the workshop
        # flaky-tool run it re-adopted the twice-failed press-first
        # plan over the cut-first alternative at cycle 9, and the
        # brain looped press-fail-replan until it stalled. Swap only
        # when plans[0] actually starts with the failed action and
        # plans[1] starts with something else.
        if prefer_alternative and len(plans) > 1:
            _first0 = plans[0].steps[0] if plans[0].steps else None
            _first1 = plans[1].steps[0] if plans[1].steps else None
            if (_first0 is not None and _first0.name == failed_action
                    and _first1 is not None
                    and _first1.name != failed_action):
                plans = [plans[1], plans[0]]
                self._event("alternative_preferred",
                            f"repeated mistake on '{failed_action}': preferring "
                            f"the alternative plan "
                            f"{' > '.join(a.name for a in plans[0].steps)} "
                            f"over {' > '.join(a.name for a in plans[1].steps)}")
            elif _first0 is not None and _first0.name != failed_action:
                # The reordering above (relaxation sort / cost order)
                # already put a different first action on top: the
                # preference is honoured without a swap. Record it
                # the same way — the audit trail must show the
                # preference was honoured, not stay silent. (The
                # event fires at most once per planning pass: the
                # flag is consumed above.)
                self._event("alternative_preferred",
                            f"repeated mistake on '{failed_action}': preferring "
                            f"the alternative plan "
                            f"{' > '.join(a.name for a in plans[0].steps)} "
                            f"over {' > '.join(a.name for a in plans[1].steps)} "
                            f"(already first after reordering; no swap needed)")
        # Phase 7: strategy selection — after a strategy failure the
        # next plan must differ in its first-action tool when a
        # different-strategy candidate exists (the existing
        # _prefer_alternative shape, extended, not duplicated).
        if self._failed_strategy is not None and len(plans) > 1:
            first = plans[0].steps[0] if plans[0].steps else None
            if first is not None and self._strategy_name(first) == self._failed_strategy:
                different = [p for p in plans[1:]
                             if p.steps and self._strategy_name(p.steps[0]) != self._failed_strategy]
                if different:
                    plans = [different[0]] + [p for p in plans if p is not different[0]]
                    self._event("strategy_preferred",
                                f"after strategy '{self._failed_strategy}' failed, preferring plan "
                                f"{' > '.join(a.name for a in plans[0].steps)}")
        best = plans[0]
        self._set_plan(list(best.steps))
        self.plan_failed = False
        self._plan_started_len = len(self.plan)
        # NOTE: _steps_executed is deliberately NOT reset here — it is
        # monotonic per goal (see _goal_progress), so a replan does not
        # erase the successful steps already banked toward the goal.
        if len(plans) > 1:
            self.alternative_plan = list(plans[1].steps)
            self.alternative_plan_cost = plans[1].cost
        else:
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
        self._event("plan", f"{' > '.join(a.name for a in best.steps) or '(already true)'} "
                            f"(cost {best.cost})"
                            + (f"; alternative kept (cost {plans[1].cost})" if len(plans) > 1 else ""))
        self._log_pattern_applications(best.steps)
        # Phase 7: a plan whose first step uses a different strategy
        # clears the failed-strategy preference, and the session for
        # this goal opens (or re-opens) with the new plan.
        if best.steps and self._strategy_name(best.steps[0]) != self._failed_strategy:
            self._failed_strategy = None
        self._open_execution_session(g, best.steps)
        return True

    def _note_failed_entry_if_door(self, a: Action) -> None:
        """Phase 5 (coordinator fix, design §2.7): when a move action
        fails while the entry barrier is closed, the destination is a
        door-gated room — the model learns the gate from the failed
        traversal, never by assumption. Only ``go`` actions (env
        ``_do == "go"`` with a ``room`` arg) can teach a gate."""
        if a.args.get("_do") != "go":
            return
        dest = a.args.get("room")
        if dest is None or self.world.get("door_open") is not False:
            return
        self.world_model.note_failed_entry(dest, self.cycle)

    def _do_execute_step(self, args, env) -> bool:
        a = self.plan[0]
        # Phase 7: a step is in flight (cancel_execution defers the
        # suspension to the monitor check when this is True).
        self._executing_step = True
        if not self.planner.applicable(a, self.world):
            # Phase 7: explicit dependency check — the missing
            # precondition keys are named, the block is recorded, and
            # recovery goes through _try_alternative. The blocked
            # action never executes.
            missing = self.check_dependencies(a, self.world)
            self._event("dependency_blocked",
                        f"action '{a.name}' blocked: missing dependencies "
                        f"{missing} (preconditions {dict(a.pre)})")
            self._note_failed_entry_if_door(a)
            self._event("surprise", f"preconditions of '{a.name}' no longer hold")
            if self._try_alternative(a, {k: self.world.get(k) for k in a.pre}):
                self._executing_step = False
                return False
            self._set_plan([])
            self.surprise = True
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self._executing_step = False
            return False
        # Phase 3: store the predicted effects BEFORE execution, so the
        # next cycle can check them against the sensed world.
        self._store_predictions(a)
        before = dict(self.world)
        step_started_wall = time.monotonic()
        if a.tool:
            # Phase 7 (D7-2): the brain's execution path runs tools in
            # isolated mode. The TypeError fallback is only for a
            # registry without the Phase 7 keyword (Worker 1 not
            # landed); it never swallows a tool failure, which the
            # registry already converts into a ToolResult.
            # Coordinator integration fix (Phase 7 gate): pass the
            # tool's own spec timeout. Without it every tool ran
            # under the registry's 5 s default, so a ToolSpec that
            # declared timeout=30 silently got 5 s.
            _spec = self.tools.spec(a.tool)
            _timeout = _spec.timeout if _spec is not None else None
            try:
                res = self.tools.call(a.tool, a.args, self.granted,
                                      timeout=_timeout, isolated=True)
            except TypeError:
                res = self.tools.call(a.tool, a.args, self.granted)
            ok, info = res.ok, (res.error or "tool ok")
            if ok and a.tool == "research" and self.researcher is not None:
                # Phase 11 §2.2 (W5): the research tool ran isolated in
                # the forked child and returned a plain-data report
                # dict; admission to LTM + the untrusted state writes
                # happen here, in-process, via the merge entry point.
                # A non-dict output (e.g. the 64 KiB bound truncated
                # it to a repr) cannot merge: record it and skip the
                # merge rather than crashing the cycle.
                if isinstance(res.output, dict):
                    self._research_tool(
                        self.researcher.merge_dict(res.output, self.cycle))
                else:
                    self._event("research_merge_skipped",
                                f"research tool output was not a report dict "
                                f"({type(res.output).__name__}); merge skipped "
                                f"cycle={self.cycle}")
            # Phase 2 (audit A9): Action.store_as is honoured — a tool's
            # output is written to the shared state under that key, so
            # the next cycle can reason about it.
            if ok and a.store_as:
                self._set_state(a.store_as, res.output, source=f"tool:{a.tool}",
                                confidence=1.0, provenance=f"tool:{a.tool}",
                                entry_type="inferred")
            after = dict(self.world)
        elif env is not None:
            # Phase 1: environment actions go through the same authorization
            # boundary as registered tools — the ``granted`` set. A brain
            # built without the "environment" grant must not act on the
            # world, and the attempt is recorded, not silently ignored.
            action_name = a.args.get("_do", a.name)
            if "environment" not in self.granted:
                ok, info = False, f"permission 'environment' not granted for environment action '{action_name}'"
                self._event("deny", f"env action '{action_name}' denied (granted={sorted(self.granted)})")
                after = dict(self.world)
            else:
                ok, info = env.act(action_name, {k: v for k, v in a.args.items() if k != "_do"})
                # Phase 5: this action executed — mark it so _sense
                # can tell an action effect from an exogenous change —
                # then refresh _prev_obs from the post-action
                # observation (a change that only appears at the NEXT
                # sense, with no action between the two observations,
                # is exogenous; see _sense).
                self._acted_since_sense = True
                after = env.observe()
                for k, v in after.items():
                    self._set_state(k, v, source=f"env:{k}", confidence=1.0,
                                    provenance="env:observe", entry_type="observation")
                    self.world_model.observe(k, v, self.cycle, source="env",
                                             cause=f"action:{a.name}")
                self._prev_obs = dict(after)
                self._acted_since_sense = False
                # Phase 5: a successful move is a spatial transition
                # in the world model (the room graph is built from
                # observed moves, never assumed).
                if ok and before.get("at") != after.get("at"):
                    self.world_model.record_transition(
                        before.get("at"), after.get("at"), self.cycle,
                        cause=f"action:{a.name}")
        else:
            ok, info, after = False, "no environment to act on", dict(self.world)
        after = dict(self.world)
        step_wall_s = time.monotonic() - step_started_wall
        # Coordinator integration fix (Phase 7 gate): a failed tool's
        # RAW error text must never become a learner-episode surprise.
        # The raw text is the tool's own output (it can carry garbage,
        # markers, or secrets); the failure record belongs in the
        # events / tool audit, where it is already kept (the "act"
        # event below still carries ``info``). An episode records only
        # the TYPED failure kind and the tool name, so nothing a tool
        # prints can leak into learner.json. Non-tool (env) failures
        # keep the established ("action", "success", info) shape: env
        # messages are world-generated and bounded.
        tool_cancelled = bool(a.tool) and bool(getattr(res, "cancelled", False))
        tool_timed_out = bool(a.tool) and bool(getattr(res, "timed_out", False))
        if ok:
            surprises = self.evaluator.check_prediction(a.eff, after)
        elif a.tool:
            if tool_cancelled:
                surprises = [("tool", "cancelled", a.tool)]
            elif tool_timed_out:
                surprises = [("tool", "timeout", a.tool)]
            else:
                surprises = [("tool", "failure", a.tool)]
        else:
            surprises = [("action", "success", info)]
        # Phase 7: explicit action verification — a mismatch is a
        # surprise (the Phase 4/5 prediction machinery checks the
        # stored prediction next cycle; the surprise flag and the
        # failure path below already carry it, so nothing is appended
        # twice) AND a ledger record with verified=False. A
        # verification failure never silently passes as success.
        verified, mismatches = self._verify_action_outcome(a, before, after)
        if not verified:
            self._event("action_verification_failed",
                        f"action '{a.name}' executed but effects mismatched: "
                        f"{[(k, a.eff.get(k), after.get(k)) for k in mismatches]}")
        # Phase 7: the ledger records every executed action (one
        # outcome per started step), and the session counts the step.
        outcome_ok = ok and not surprises
        if tool_cancelled:
            ledger_outcome = "cancelled"
        elif tool_timed_out:
            ledger_outcome = "timed_out"
        elif not ok and (info or "").startswith("permission"):
            ledger_outcome = "denied"
        elif outcome_ok:
            ledger_outcome = "ok"
        else:
            ledger_outcome = "failed"
        decision = {"rule_id": self._current_rule_id,
                    "plan_cost": round(sum(float(s.cost) for s in self.plan), 4),
                    "alternative_considered": bool(self.alternative_plan),
                    "alternative_cost": self.alternative_plan_cost}
        tool_calls = 1 if a.tool else 0
        self.execution_monitor.ledger.record(
            cycle=self.cycle, action=a.name,
            args=self._redact_tool_args(a.tool, dict(a.args)), tool=a.tool,
            outcome=ledger_outcome, predicted=dict(a.eff),
            actual={k: after.get(k) for k in a.eff},
            cost={"cycles": 1, "wall_s": step_wall_s, "tool_calls": tool_calls},
            decision=decision, verified=verified)
        session = self.execution_monitor.session
        if session is not None and session.status == "running":
            session.record_step(a.name, ledger_outcome,
                                {"tool_calls": tool_calls, "wall_s": step_wall_s,
                                 "cycles": 1})
        # Phase 10.5: the episode carries the args the action ran
        # with — redacted with the same spec-aware redactor as the
        # ledger, so a secret-marked tool argument never reaches
        # learner.json (episodes or counts keys). P7-10: persistence
        # never carries secrets.
        self.learner.record(Episode(self.cycle, a.name, {k: before.get(k) for k in a.pre},
                                    ok and not surprises, surprises,
                                    args=self._redact_tool_args(a.tool, dict(a.args))))
        # Phase 6: the same outcome also updates the calibration
        # tracker (predicted confidence vs outcome, per effect) and
        # checks the repeated-mistake counter. The predicted
        # confidence is read back from the state entry
        # _store_predictions wrote before execution.
        outcome_ok = ok and not surprises
        if self.calibration is not None:
            for eff_key in a.eff:
                pred_entry = self.state.get(f"predicted:{a.name}:{eff_key}")
                pred_conf = (pred_entry.confidence if pred_entry is not None
                             else 1.0)
                self.calibration.record(a.name, eff_key, pred_conf, outcome_ok)
        situation = {k: before.get(k) for k in a.pre}
        if not outcome_ok:
            # Learner-shaped stand-ins (e.g. the harness's no-learning
            # variant) expose only the public API Brain used before
            # Phase 6; repeated_mistakes is optional there, as
            # episodes_for_induction already is in _reason_from_wm.
            rep_fn = getattr(self.learner, "repeated_mistakes", None)
            if callable(rep_fn):
                try:
                    # Phase 10.5: the failed action's args join the
                    # grouping, so one variant's repeated failures do
                    # not flag a sibling variant. The args are
                    # redacted first, so the grouping matches the
                    # episodes as recorded above (redacted).
                    rep = rep_fn(a.name, situation,
                                 self._redact_tool_args(a.tool, dict(a.args)))
                except TypeError:
                    # Learner-shaped stand-ins with the two-argument
                    # form (see the comment above).
                    rep = rep_fn(a.name, situation)
            else:
                rep = None
            if rep is not None:
                key = (a.name,
                       canonical_args(self._redact_tool_args(a.tool, dict(a.args))),
                       tuple(rep["features"]))
                if key not in self._repeated_mistake_seen:
                    self._repeated_mistake_seen.add(key)
                    self._prefer_alternative = True
                    self._repeated_mistake_action = a.name
                    self._event(
                        "repeated_mistake",
                        f"action '{a.name}' failed {rep['failures']} times in "
                        f"situations sharing features {rep['features']}; "
                        f"preferring an alternative plan next cycle")
        else:
            # A success in this feature set clears the marker, so a
            # later fresh pair of failures in the same features
            # re-fires the event instead of being swallowed.
            self._repeated_mistake_seen.discard(
                (a.name,
                 canonical_args(self._redact_tool_args(a.tool, dict(a.args))),
                 tuple(sorted(str(k) for k in situation))))
        # Phase 4: a fresh outcome tests any open hypothesis about this
        # action (repeat = the hypothesis's test; see _test_hypotheses).
        self._test_hypotheses(a.name, "success" if (ok and not surprises) else "failure")
        self._set_plan(self.plan[1:])
        # Phase 2, tightened Phase 10.5 (P10.5-5): only a step whose
        # effects materialised counts toward goal progress — and
        # "materialised" now means the world actually CHANGED on at
        # least one of the step's effect keys, or on its ``store_as``
        # key when it has one. A failed step never advances the goal,
        # and neither does an idempotent re-execution (a re-press
        # with ``pressed`` already True, a re-take of a held item):
        # ``ok and not surprises`` alone cannot tell those apart,
        # because a step whose predicted effect already holds is no
        # surprise. In-plan steps always change state in the
        # planner's model — ``plan_all`` now prunes no-change steps
        # outright (the visited set alone did NOT guarantee this: a
        # no-change FIRST step opens a fresh (state, first-action)
        # key, so a redundant re-cut could be planned and stall a
        # recovery mid-plan once it stopped counting as progress) —
        # so a step that changes nothing in execution is a model
        # mismatch or an idempotent retry — never progress. Counting
        # it as progress defeated the progress-based stall detector
        # (P10-1b: the 80-cycle budget exhaustion). ``before``/
        # ``after`` are the shared-state snapshots taken around the
        # execution above.
        _world_changed = (
            any(before.get(k) != after.get(k) for k in a.eff)
            or (a.store_as is not None
                and before.get(a.store_as) != after.get(a.store_as)))
        if ok and not surprises and _world_changed:
            self._steps_executed += 1
            # Phase 7: a success breaks the consecutive-failure streak
            # and clears the failed-strategy preference.
            self._strategy_streak = {"strategy": None, "action": None, "count": 0}
            self._failed_strategy = None
        self._event("act", f"{a.name}: {info}" + (f" | surprise {surprises}" if surprises else ""))
        if surprises or not ok:
            self._note_failed_entry_if_door(a)
            # Phase 7: record the failed strategy. After
            # MAX_SAME_STRATEGY_RETRIES failed attempts at the same
            # step with the same strategy the goal stops with reason
            # "strategy_exhausted" and partial results preserved —
            # never a silent retry loop. (Actions with no strategy —
            # plain env/state actions — never trip this cap; the
            # progress-based stall detection still governs them.)
            strategy = self._record_strategy_failure(a)
            if (strategy is not None
                    and self._strategy_streak.get("count", 0) >= self.MAX_SAME_STRATEGY_RETRIES):
                self._executing_step = False
                self._do_fail_goal({"reason": "strategy_exhausted"}, None)
                return False
            # Recovery: try the retained alternative plan before
            # re-planning from scratch. _try_alternative also bans the
            # failed (action, situation) pair for this goal.
            if self._try_alternative(a, {k: before.get(k) for k in a.pre}):
                self._executing_step = False
                # Phase 7: the monitor is enforced after the step, so
                # a budget tripped by it stops the task now.
                self._monitor_check()
                return False
            self._set_plan([])
            self.surprise = True
            self.alternative_plan, self.alternative_plan_cost = [], 0.0
            self._executing_step = False
            self._monitor_check()
            return False
        self._executing_step = False
        # Phase 7: the monitor is enforced after the step, so a budget
        # tripped by it stops the task now.
        self._monitor_check()
        return True

    def _do_complete_goal(self, args, env) -> bool:
        g = self.goals.active()
        # Phase 7: the execution session closes with its final status.
        self._close_execution_session(g, "done")
        self.last_evaluation = self.evaluator.evaluate_goal(g.target, self.goal_start, self.world)
        self.goals.complete(g.id)
        self._reset_after_goal(g)
        self._event("evaluate", f"goal done: '{g.description}' score {self.last_evaluation.score:.2f}")
        # Phase 3 (mandate decision): the preempting goal has finished —
        # auto-resume the most recently suspended goal, if any.
        self._maybe_auto_resume()
        return self.last_evaluation.success

    def _do_fail_goal(self, args, env) -> bool:
        g = self.goals.active()
        reason = args.get("reason", "unknown")
        # Phase 7: the execution session closes with its final status.
        self._close_execution_session(g, "failed", reason)
        # Phase 7: a failed subgoal is recorded on its parent, never
        # silent — the parent retries (once, by re-decomposition in
        # _do_make_plan) or fails with reason "subgoal_failed".
        if g.parent is not None:
            self._event("subgoal_failed",
                        f"subgoal '{g.description}' (id={g.id}) failed "
                        f"({reason}); parent goal id={g.parent} notified")
        self.last_evaluation = self.evaluator.evaluate_goal(g.target, self.goal_start, self.world)
        self.goals.fail(g.id, reason)
        self._reset_after_goal(g)
        # Phase 2: map the failure to its terminal stop reason. run()
        # reports it (unless a later goal clears it by being selected).
        stop = {"no plan found": "impossible", "no plan on resume": "impossible",
                "no plannable subgoal": "impossible",
                "model_disagreement_learned_constraint": "impossible",
                "repeated_failure_no_alternative": "impossible",
                "no_permitted_plan": "impossible",
                "subgoal_failed": "impossible",
                "strategy_exhausted": "impossible",
                "stalled": "stalled"}.get(reason)
        if stop is not None:
            self.last_stop_reason = stop
        self._event("evaluate", f"goal FAILED: '{g.description}' ({reason}); "
                                f"unmet: {self.last_evaluation.reasons}")
        return True

    def clear_transient_for_new_world(self) -> None:
        """Phase 5: entering a new world. Transient observations are
        cleared — the shared CognitiveState, the world model's
        transient entries, the incoming-thought queue, the
        goal-thought cache and the last-observation record — while
        durable learning is kept: LTM, rules, learner episodes,
        planner bans and actions, causal records, and the pattern
        library.

        Worker 1's WorldModel has no ``clear_transient`` method, so
        the model is cleared by re-instantiating it (the patterns
        object is KEPT, and the planner's pattern context reads the
        live ``self.world_model``, so the re-instantiation needs no
        re-attachment). The stall window is reset: progress measured
        in the old world says nothing about the new one. Phase 10.5
        (P10.5-9): the planner's action generation advances — actions
        added from here on are the new world's own, carried-over
        actions keep their generation, so exact cost ties break in
        favour of the current world."""
        self.planner.mark_generation()
        self.state.clear()
        self.world_model = WorldModel()
        self._incoming = []
        self._goal_thoughts = {}
        self._prev_obs = {}
        self._acted_since_sense = False
        self._plan_started_len = 0
        self._steps_executed = 0
        self.surprise, self.plan_failed = False, False
        self.alternative_plan, self.alternative_plan_cost = [], 0.0
        self.control.reset_progress()
        self._event("new_world",
                    "transient state cleared for a new world; patterns, "
                    "learner episodes, planner bans and causal records kept")

    def _reset_after_goal(self, goal: Goal | None = None) -> None:
        # Phase 12 (P12-W7): the cached goal Thought dies with the
        # goal. _goal_thoughts is read ONLY for the ACTIVE goal
        # (_attend's setdefault); a done/failed goal is never active
        # again, and resumption overwrites the entry (resume_goal),
        # so dropping it here is semantics-preserving. Without this,
        # one Thought (~0.7 KiB with keywords/tags) was retained per
        # completed goal (~91 KiB/1,000 cycles in the soak). This is
        # the single choke point: _reset_after_goal is called only
        # from _do_complete_goal / _do_fail_goal, the only paths that
        # mark goals done/failed (GoalManager.complete/fail have no
        # other production callers).
        if goal is not None:
            self._goal_thoughts.pop(goal.id, None)
        # The plan is cleared in the goal record itself (Phase 3): the
        # goal is passed explicitly because it is no longer active.
        self._set_plan([], goal)
        self.surprise, self.plan_failed = False, False
        self._plan_started_len = 0
        self._steps_executed = 0
        self.alternative_plan, self.alternative_plan_cost = [], 0.0
        # Phase 6: a repeated-mistake preference dies with the goal.
        self._prefer_alternative = False
        self._repeated_mistake_action = None
        # Phase 7: a strategy preference / streak dies with the goal.
        self._failed_strategy = None
        self._strategy_streak = {"strategy": None, "action": None, "count": 0}
        self.control.reset_progress()

    # ---- Phase 7: execution session lifecycle (D7-2) -------------------
    def _session_budgets(self) -> dict:
        """Per-task default budgets (D7-3): max_cycles from the brain's
        own control budget, max_tool_calls 50, max_wall_s 60.0."""
        return {"max_cycles": self.control.max_cycles,
                "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
                "max_wall_s": DEFAULT_MAX_WALL_S}

    def _redact_tool_args(self, tool, args) -> dict:
        """Redact a tool call's arguments for any persisted or
        auditable record (ledger, session snapshot, saved plans).

        Coordinator integration fix (Phase 7 gate): ``redact_secrets``
        alone keys only on secret-named keys and token-shaped values,
        so a short secret passed as a spec-marked ``secret`` argument
        (e.g. ``code``) leaked into the ledger, the session snapshot
        and ``plan.json``. This helper ALSO honours the ToolSpec's
        ``secret`` markings (the same rule the registry audit
        already uses): a secret-marked argument is always
        ``"[REDACTED]"`` regardless of its value's shape. With no
        spec (unknown tool, or a plain env action) it falls back to
        ``redact_secrets``. Never raises.
        """
        try:
            spec = None
            if isinstance(tool, str) and tool:
                spec_fn = getattr(self.tools, "spec", None)
                if callable(spec_fn):
                    spec = spec_fn(tool)
            if spec is not None:
                secret_names = {e["name"] for e in (spec.args or [])
                                if isinstance(e, dict) and e.get("secret")}
                return {k: ("[REDACTED]" if k in secret_names
                            else redact_secrets(v))
                        for k, v in dict(args or {}).items()}
            return redact_secrets(args)
        except Exception:
            return redact_secrets(args)

    def _redact_saved_actions(self, steps) -> list:
        """Redact one saved list of action dicts (``asdict`` shape)
        with the spec-aware redactor, for ``save()``. Returns new
        dicts; the in-memory plans are never mutated. A resumed
        secret-bearing step receives ``"[REDACTED]"`` as the argument
        and must re-supply the secret (see ADR P7-10) — persistence
        never carries secrets."""
        out = []
        for s in (steps or []):
            if not isinstance(s, dict):
                out.append(s)
                continue
            d = dict(s)
            d["args"] = self._redact_tool_args(d.get("tool"),
                                               d.get("args", {}))
            out.append(d)
        return out

    def _open_execution_session(self, goal: Goal, steps) -> None:
        """Open (or re-open) the execution session for ``goal`` when
        a plan was just made. A live session for the same goal is
        kept (its task id is stable across replans); an interrupted
        session for the goal is re-opened as running; otherwise a
        fresh session is opened and an ``execution_started`` event
        names its task id, steps and budgets.

        Coordinator integration fix (Phase 7 gate): a ``cancelled``
        session is NEVER re-opened here. Re-opening it on an ordinary
        replan let a cancelled task silently restart and retry the
        cancelled step (observed in the cancellation acceptance
        experiment: the sleeper was retried 3x after the kill until
        ``strategy_exhausted``). A cancelled session re-opens only
        through an explicit ``resume_goal()`` (which calls
        ``monitor.reopen_session`` directly)."""
        monitor = self.execution_monitor
        session = monitor.session
        if session is not None and session.goal_id == goal.id:
            if session.status == "running":
                session.plan_snapshot = [
                    {"name": a.name, "args": self._redact_tool_args(a.tool, dict(a.args)),
                     "eff": redact_secrets(dict(a.eff))} for a in (steps or [])]
                session.steps_total = len(session.plan_snapshot)
                return
            if session.status == "interrupted":
                reopened = monitor.reopen_session(steps)
                if reopened is not None:
                    self._event("execution_resumed",
                                f"task {reopened.task_id[:12]} resumed for "
                                f"'{goal.description}' "
                                f"({reopened.steps_executed}/{reopened.steps_total} steps done)")
                return
            if session.status == "cancelled":
                # The task was cancelled and has not been explicitly
                # resumed: leave the session cancelled. The monitor
                # check (see _monitor_check) suspends the goal, and
                # the task only continues via resume_goal().
                return
        budgets = self._session_budgets()
        session = monitor.open_session(goal.id, steps, budgets, self.cycle)
        self._event("execution_started",
                    f"task {session.task_id[:12]} for '{goal.description}': "
                    f"{len(steps)} step(s), budgets max_cycles="
                    f"{budgets['max_cycles']} max_tool_calls="
                    f"{budgets['max_tool_calls']} max_wall_s="
                    f"{budgets['max_wall_s']}")

    def _close_execution_session(self, goal: Goal, status: str, reason: str | None = None) -> None:
        """Close the live session for ``goal`` with its final status
        (and an ``execution_finished`` event). The session object is
        kept on the monitor for inspection and persistence."""
        monitor = self.execution_monitor
        session = monitor.session
        if session is None or session.goal_id != goal.id or session.status != "running":
            return
        monitor.close(status, reason)
        self._event("execution_finished",
                    f"task {session.task_id[:12]} for '{goal.description}' "
                    f"finished: status={status} reason={reason or '-'} "
                    f"steps {session.steps_executed}/{session.steps_total} "
                    f"tool_calls={session.tool_calls}")

    def _monitor_check(self) -> str | None:
        """Run the monitor enforcement for the live session. Returns
        the monitor's verdict (``"budget_exceeded"`` / ``"cancelled"``
        / None). On cancellation the goal is suspended here, with its
        plan and position preserved for ``resume_goal()``."""
        result = self.execution_monitor.check(self)
        if result == "cancelled":
            self._suspend_cancelled_session()
            return result
        # A session already closed as budget_exceeded must keep
        # blocking: the monitor only enforces a RUNNING session, so
        # without this a later step() would silently execute the
        # stopped task's plan (the goal is still active). The stop
        # reason is already recorded; this just refuses the step.
        session = self.execution_monitor.session
        active = self.goals.active()
        if (result is None and session is not None and active is not None
                and session.goal_id == active.id
                and session.status == "budget_exceeded"):
            return "budget_exceeded"
        # Coordinator integration fix (Phase 7 gate): the same guard
        # for a session sitting in "cancelled" for the still-active
        # goal. ``cancel_execution`` sets the status directly while a
        # step is mid-flight, and ``ExecutionMonitor.check`` only
        # enforces RUNNING sessions — without this guard the goal
        # would stay active with a cancelled session and the next
        # cycle would replan and retry the cancelled step. The
        # guard is goal-scoped (like the budget_exceeded one): a
        # cancelled session for a different, suspended goal must not
        # block the brain's work on another goal.
        if (result is None and session is not None and active is not None
                and session.goal_id == active.id
                and session.status == "cancelled"):
            self._suspend_cancelled_session()
            return "cancelled"
        return result

    def _suspend_cancelled_session(self) -> None:
        """Suspend the active goal after a cancellation, preserving
        its plan and position (the ``suspend_active_goal`` path, so
        ``resume_goal()`` can continue it). A cancelled task never
        restarts by itself."""
        g = self.goals.active()
        session = self.execution_monitor.session
        if g is None or session is None or session.goal_id != g.id:
            return
        reason = session.cancel_reason or "user_cancelled"
        rec = self.suspend_active_goal(reason)
        if rec is not None:
            self._set_plan([])
            self.surprise, self.plan_failed = False, False
            self._plan_started_len = 0
            self._steps_executed = 0
            self.control.reset_progress()

    # ---- Phase 7: public cancellation (D7-2) ---------------------------
    def cancel_execution(self, reason: str = "user_cancelled") -> dict:
        """Request cancellation of the live execution session.

        If a tool is currently running in isolated mode, the runner
        terminates it (``ToolRegistry.cancel()``) and this call
        returns only after the process is observed dead (the kill is
        polled with a short deadline; the observed state is recorded
        in the returned dict). If no tool is running, the flag takes
        effect before the next step. The task's plan and position are
        preserved and the goal is suspended via the existing
        ``suspend_active_goal`` path, so ``resume_goal()`` can
        continue it. A cancelled task never restarts by itself.
        Returns ``{"cancelled", "killed", "observed", "goal_suspended"}``.
        """
        monitor = self.execution_monitor
        session = monitor.session
        killed: bool | None = None
        observed: str
        if session is not None and session.status == "running":
            session.request_cancel(reason)
        cancel_fn = getattr(self.tools, "cancel", None)
        if callable(cancel_fn):
            try:
                killed = bool(cancel_fn())
            except Exception as exc:                      # a broken cancel never blocks the flag
                killed = False
                observed = f"tools.cancel() raised {type(exc).__name__}: {exc}"
            else:
                observed = ("isolated tool process observed dead after kill"
                            if killed else
                            "no isolated tool process was running to kill")
        else:
            observed = ("no isolated tool runner available "
                        "(ToolRegistry.cancel not present); a thread-path "
                        "tool cannot be killed, so the cancellation flag "
                        "takes effect before the next step")
        if session is not None and session.status == "running":
            session.status = "cancelled"
            session.stop_reason = reason
            self._event("execution_cancelled",
                        f"task {session.task_id[:12]} cancelled "
                        f"(reason={reason}; {observed}; steps "
                        f"{session.steps_executed}/{session.steps_total} "
                        f"executed; plan and position preserved)")
        suspended = False
        if not self._executing_step:
            # No step is mid-flight: suspend now. When a step IS
            # mid-flight (a tool call in another thread), the monitor
            # check at the end of step() suspends it instead — the
            # plan must not be captured while it is being advanced.
            before = self.goals.active()
            self._suspend_cancelled_session()
            suspended = before is not None and self.goals.active() is None
        return {"cancelled": True, "killed": killed, "observed": observed,
                "goal_suspended": suspended}

    # ---- Phase 7: dependency check + action verification (D7-2) --------
    def check_dependencies(self, action: Action, world: dict) -> list:
        """The unsatisfied precondition keys of ``action`` in ``world``
        (empty when every precondition holds). The explicit,
        named, recoverable dependency check: a missing dependency
        blocks execution (see ``_do_execute_step``), it never
        silently skips the step and never executes an action whose
        preconditions do not hold."""
        return [k for k, v in (getattr(action, "pre", {}) or {}).items()
                if (world or {}).get(k) != v]

    def _verify_action_outcome(self, action: Action, before: dict,
                               after: dict) -> tuple[bool, list]:
        """Verify an executed action's effects against the world
        afterwards. Returns ``(verified, mismatches)`` where
        ``mismatches`` is the list of effect keys whose actual value
        differs from the predicted one. A mismatch is a surprise
        (see ``_do_execute_step``): it goes through the Phase 4/5
        machinery and the ledger records ``verified=False``. A
        verification failure never silently passes as success."""
        mismatches = [k for k, v in (getattr(action, "eff", {}) or {}).items()
                      if (after or {}).get(k) != v]
        return (not mismatches, mismatches)

    # ---- Phase 7: strategy selection (D7-2) -----------------------------
    def _strategy_name(self, action) -> str | None:
        """The strategy an action belongs to: its tool name when it
        runs through a tool, else the pattern-library name when a
        pattern covers one of its effects, else None (a plain env /
        state action has no strategy to select against)."""
        if action is None:
            return None
        tool = getattr(action, "tool", None)
        if isinstance(tool, str) and tool:
            return tool
        eff = getattr(action, "eff", None) or {}
        for pattern in self.patterns.patterns():
            if getattr(pattern, "effect", None) in eff:
                return str(pattern.name)
        return None

    def _record_strategy_failure(self, action) -> str | None:
        """Record a failed strategy: a ``strategy_failed`` event
        naming it, the consecutive-failure streak for it, and the
        per-(action, strategy) count. Returns the strategy name, or
        None when the action has no strategy. A success, or a failure
        under a different strategy, resets the streak."""
        strategy = self._strategy_name(action)
        if strategy is None:
            self._strategy_streak = {"strategy": None, "action": None, "count": 0}
            return None
        streak = self._strategy_streak
        if streak.get("strategy") == strategy and streak.get("action") == action.name:
            streak["count"] = int(streak.get("count", 0)) + 1
        else:
            streak = {"strategy": strategy, "action": action.name, "count": 1}
            self._strategy_streak = streak
        key = (action.name, strategy)
        self._strategy_failures[key] = self._strategy_failures.get(key, 0) + 1
        self._failed_strategy = strategy
        self._event("strategy_failed",
                    f"strategy '{strategy}' failed on '{action.name}' "
                    f"(attempt {streak['count']} at this step with this "
                    f"strategy; {self._strategy_failures[key]} failure(s) total)")
        return strategy

    # ---- Phase 7: permission-aware planning (D7-2) ----------------------
    def _plan_permitted(self, plan) -> tuple[bool, str | None, str | None]:
        """Is every step of ``plan`` permitted under ``self.granted``?
        A step is refused when it names a tool whose required grant
        (read from the registry spec, never guessed from the action
        name) is not granted, or when it is an env action (``_do``)
        and ``"environment"`` is not granted. An unknown tool (no
        spec) is left to the call-site refusal — the second line of
        defence. Returns ``(True, None, None)`` or
        ``(False, tool_or_action_name, missing_grant)``."""
        for step in (plan.steps if hasattr(plan, "steps") else plan or []):
            tool = getattr(step, "tool", None)
            if isinstance(tool, str) and tool:
                spec = self.tools.spec(tool)
                if spec is not None and spec.required_grant not in self.granted:
                    return False, tool, spec.required_grant
            elif isinstance(getattr(step, "args", None), dict) and "_do" in step.args:
                if "environment" not in self.granted:
                    return False, str(step.args.get("_do", step.name)), "environment"
        return True, None, None

    # ---- Phase 7 §4.2: pattern-evidence usefulness -----------------------
    def _mark_pattern_evidence_useful(self, patterns) -> None:
        """Prioritized-pruning fix (Phase 6 §11.5: pattern-evidence
        episodes retained 0/3): when a pattern is learned from
        episodes, mark those episodes' usefulness +1 via the existing
        prioritizer usefulness counts (``Prioritizer.note_use``).
        Evidence carries only (action, outcome) — no situation — so
        every learner episode matching the action and outcome is
        marked. Each pattern is marked once (``_pattern_evidence_marked``).
        Every recorded episode already earned +1 at record time via
        ``_on_episode_recorded``; this is the second +1 that makes
        pattern evidence outrank filler under prioritized pruning."""
        for pattern in (patterns or []):
            pid = getattr(pattern, "id", None)
            if pid is None or pid in self._pattern_evidence_marked:
                continue
            self._pattern_evidence_marked.add(pid)
            for ev in (getattr(pattern, "evidence", None) or []):
                action = str(ev.get("action", "")) if isinstance(ev, dict) else ""
                outcome = str(ev.get("outcome", "")) if isinstance(ev, dict) else ""
                if not action:
                    continue
                for ep in list(self.learner.episodes):
                    if ep.action != action:
                        continue
                    # Evidence writes "failed"/"success"; learner
                    # episodes write "failure"/"success" — normalize
                    # both spellings of a failed outcome to False.
                    if outcome and (ep.ok != (outcome.lower() == "success")):
                        continue
                    self.prioritizer.note_use(ep.action, dict(ep.situation))

    # ---- goal suspension foundation (Phase 1: structure only) ------------
    def suspend_active_goal(self, reason: str) -> SuspendedGoal | None:
        """Record the active goal's state for a later resumption.

        Phase 1 only: this captures goal id, original status, remaining
        plan, plan position (steps already executed), history, goal-start
        state and reason, and returns the record. It does NOT change what
        the cycle does next — interruption / resumption behaviour is
        Phase 2. The goal returns to ``pending`` in the manager, as the
        record preserves its original status.
        """
        g = self.goals.active()
        if g is None:
            return None
        position = self._plan_started_len - len(self.plan)
        # Phase 7: an interruption during an execution session suspends
        # the goal WITH its live session (status -> interrupted, plan
        # position recorded in the suspension record). A session
        # already closed as cancelled keeps that status — cancellation
        # is terminal for the session, interruption is not.
        session = self.execution_monitor.session
        if session is not None and session.goal_id == g.id and session.status == "running":
            session.status = "interrupted"
        # Phase 2: snapshot the state entries that are preconditions of
        # the next planned action, so resume_goal can tell whether the
        # world moved while the goal was suspended.
        snapshot: dict = {}
        if self.plan:
            snapshot = {k: self.world.get(k) for k in self.plan[0].pre}
        # Phase 3: the suspension record keeps the event log, not a
        # world-snapshot history list (Brain.history is removed).
        return self.goals.suspend(g, list(self.plan), max(0, position),
                                  list(self.events), dict(self.goal_start),
                                  reason, self.cycle, world_snapshot=snapshot)

    @property
    def provider(self):
        """The Brain's current knowledge provider (Phase 11 §2.4, W5):
        the researcher's provider, or ``None`` when this brain has no
        researcher (or the researcher has no provider). This is the
        source of truth for the research tool's provider spec — the
        spec recorded in ``research.json`` is NOT applied on load
        (W2), so the provider the Brain holds at registration time is
        what the isolated tool fetches through."""
        researcher = getattr(self, "researcher", None)
        return researcher.provider if researcher is not None else None

    def _register_research_tool(self) -> None:
        """Register (or re-register) the isolated research tool
        (Phase 11 §2.2, W5): ``functools.partial(research_fetch,
        provider_spec=<current provider spec>)`` — a partial of a
        module-level function with picklable args IS picklable, and
        ``inspect.ismethod(partial)`` is False, so the isolated path
        runs it in the forked child. ``None`` (no provider) registers
        the tool with a ``None`` spec: the tool stays registered (the
        planner may still reference it) but its fetch fails gracefully
        via the no-provider guard in ``Researcher.collect``."""
        spec = None
        provider = self.provider
        if provider is not None and hasattr(provider, "spec"):
            spec = provider.spec()
        fn = functools.partial(research_fetch, provider_spec=spec)
        if "research" in self.tools.names():
            self.tools.replace_fn("research", fn)
        else:
            self.tools.register(Tool("research", fn, "network",
                                     "Fetch sources, extract claims, store with provenance",
                                     ("urls",)))

    def set_provider(self, provider_or_None) -> None:
        """Swap the Brain's knowledge provider (Phase 11 §2.4, W5,
        Brain level): swaps the researcher's provider
        (``Researcher.set_provider`` — a ``WebFetcher`` is wrapped,
        ``None`` disconnects) AND re-registers the isolated research
        tool with the new spec, so the next isolated research call
        fetches through the new provider. Disconnecting (``None``)
        does NOT unregister the tool: a research action then fails
        with ``no_provider`` and the goal defers, while the brain
        keeps operating on what it already knows. No-op when this
        brain has no researcher (no research tool exists)."""
        if self.researcher is None:
            return
        self.researcher.set_provider(provider_or_None)
        self._register_research_tool()

    def _research_tool(self, rep) -> dict:
        """Merge entry point for a research report (Phase 11 §2.2, W5).

        Takes a ``ResearchReport`` ALREADY merged into this brain's
        researcher (via ``Researcher.merge`` / ``merge_dict`` —
        admission to LTM happens there, never here) and runs the
        Brain-side half: ``injection_attempt`` events for
        instruction-like text in fetched pages, the
        ``research_admitted`` / ``research_held`` trace events, and
        the untrusted state writes for claims the report holds as
        untrusted. It performs NO fetch: the fetch ran in the
        isolated child (``research_fetch``) or, for direct callers,
        in ``Researcher.collect``. Returns the same
        ``{"fetched", "failed", "summary"}`` dict as before.
        """
        # Phase 11 §6 (W4): instruction-like text found in a fetched
        # page is data, never an instruction. Log it as an
        # injection_attempt event (source + matched phrase only) and
        # act on nothing in it — the page's claims still go through
        # the normal untrusted path below.
        for attempt in getattr(rep, "injection_attempts", []) or []:
            self._event("injection_attempt",
                        f"instruction-like text in page from "
                        f"{attempt.get('domain')}: "
                        f"{attempt.get('matched')!r} treated as data, "
                        f"not acted on ({attempt.get('url')}) "
                        f"cycle={self.cycle}")
        # Phase 11 §2.5 (W3): visibility only — admission itself stays
        # in Researcher.investigate; this loop only emits the trace.
        # Per distinct claim in the report: ``research_admitted`` when
        # the claim is (now) verified in LTM, ``research_held`` when it
        # stays untrusted (reason: single-domain / mirror / disputed).
        # The held reason is read off the report, never re-derived
        # from a second ranking: mirror_suspected provenance flags
        # mean "mirror"; < 2 distinct registrable domains mean
        # "single-domain"; a disputed LTM status means "disputed".
        # (An untrusted claim from >= 2 domains with no mirror flag
        # cannot occur — independent wording would have admitted it —
        # so the fallback there is "mirror".)
        for claim, _res, status, _url in getattr(rep, "results", []) or []:
            fact_text = f"{claim.subject} {claim.relation} = {claim.value}"
            provs = [p for p in getattr(rep, "claims", []) or []
                     if isinstance(p, dict)
                     and p.get("subject") == claim.subject
                     and p.get("relation") == claim.relation
                     and p.get("value") == claim.value]
            domains = sorted({p.get("registrable_domain") or p.get("domain", "")
                              for p in provs
                              if p.get("registrable_domain") or p.get("domain")})
            if status == "verified":
                self._event("research_admitted",
                            f"{fact_text} admitted to LTM (verified) "
                            f"from {domains} cycle={self.cycle}")
            elif status == "disputed":
                self._event("research_held",
                            f"{fact_text} held: disputed "
                            f"(sources {domains}) cycle={self.cycle}")
            elif status == "untrusted":
                if any(p.get("mirror_suspected") for p in provs):
                    reason = "mirror"
                elif len(domains) < 2:
                    reason = "single-domain"
                else:
                    reason = "mirror"
                self._event("research_held",
                            f"{fact_text} held: {reason} "
                            f"(sources {domains}) cycle={self.cycle}")
        # Phase 3 §3.2: research results are UNTRUSTED. Every extracted
        # claim the researcher reports as untrusted is written to state
        # as an untrusted entry (source web:<registrable-domain>,
        # status "untrusted", full provenance when the researcher
        # provides it) and NEVER directly to LTM here — LTM admission
        # is the researcher's corroboration path (2 independent
        # domains) or explicit human approval via approve_claim().
        # The Phase 3 researcher provides rep.claims (provenance dicts,
        # one per extracted claim occurrence); fall back to rep.results
        # when that format is absent.
        claims = list(getattr(rep, "claims", []) or [])
        if claims:
            for prov in claims:
                if not isinstance(prov, dict):
                    continue
                if prov.get("status", "untrusted") != "untrusted":
                    continue  # admitted to LTM already; not untrusted
                domain = prov.get("registrable_domain") or prov.get("domain") or ""
                key = f"untrusted:{prov.get('subject')}|{prov.get('relation')}"
                self._set_state(key, prov.get("value"), source=f"web:{domain}",
                                confidence=float(prov.get("confidence", 0.45)),
                                provenance=json.dumps(prov, sort_keys=True, default=str),
                                entry_type="belief", status="untrusted")
        else:
            for claim, _res, status, url in getattr(rep, "results", []) or []:
                if status != "untrusted":
                    continue
                domain = urlparse(url).hostname or ""
                key = f"untrusted:{claim.subject}|{claim.relation}"
                self._set_state(key, claim.value, source=f"web:{domain}",
                                confidence=0.45, provenance=url,
                                entry_type="belief", status="untrusted")
        return {"fetched": rep.fetched, "failed": rep.failed, "summary": rep.summary()}

    # ---- Phase 3 §3.1/§3.2: learning-activation wrappers (thin only) ----
    def pending_rules(self) -> list:
        """Proposed rules awaiting human approval, with their evidence.
        Delegates to Learner.pending_rules(); falls back to the
        RuleEngine's pending rules if that method is absent (the
        learning worker had not landed yet). Phase 4: engine-induced
        (success) rules are merged in from their pending state entries
        — both kinds wait for the same explicit human approval."""
        fn = getattr(self.learner, "pending_rules", None)
        if callable(fn):
            out = list(fn())
        else:
            out = []
            for r in self.rules.rules(status="pending"):
                action = None
                for a in r.then:
                    if a.get("do") == "avoid_action":
                        action = a.get("args", {}).get("name")
                out.append({"rule_id": r.id, "action": action,
                            "situation": {c.get("key"): c.get("value") for c in r.when},
                            "status": "pending"})
        seen = {d.get("rule_id") for d in out}
        for d in self._induced_pending():
            if d.get("id") not in seen:
                out.append({"rule_id": d["id"], "action": d.get("action"),
                            "situation": dict(d.get("situation", {})),
                            "outcome": d.get("outcome"),
                            "successes": d.get("successes"),
                            "failures": d.get("failures"),
                            "status": "pending"})
        return out

    def approve_rule(self, rule_id: str) -> bool:
        """Explicit human approval of a proposed rule: it becomes active
        in the RuleEngine and is persisted in rules.json by the next
        Brain.save. Delegates to Learner.approve_rule(); falls back to
        RuleEngine.approve when the learner method is absent. Returns
        True only when the rule existed and was pending."""
        fn = getattr(self.learner, "approve_rule", None)
        ok = fn(rule_id) if callable(fn) else None
        if ok is None:
            rule = self.rules.get(rule_id)
            if rule is None or rule.status != "pending":
                return False
            self.rules.approve(rule_id)
            ok = True
        if ok:
            rule = self.rules.get(rule_id)
            if rule is not None and rule.status == "pending":
                self.rules.approve(rule_id)
            self._event("rule_approved", f"rule '{rule_id}' approved")
        if not ok:
            # Phase 4: an engine-induced (success) rule is approved by
            # activating it in the RuleEngine (it records the learned
            # regularity as active knowledge; it has no action to fire,
            # so its ``then`` is empty and it can never hijack the
            # cycle) and marking its state entry approved.
            #
            # Phase 5 correction: the Phase 4 comment here used to
            # claim "the planner already prefers the action it names".
            # That was FALSE — Planner.effective_cost then used only
            # the failure rate, and nothing read induced rules. The
            # preference is made real here: the named action is
            # recorded in Planner.preferred_actions (factor 0.8), and
            # effective_cost multiplies by it. The factor applies per
            # action name: effective_cost takes no state, so the
            # rule's situation conditions cannot be re-checked there
            # (design §4.3's "when the conditions match" is not
            # implementable in effective_cost as specified); the
            # situation stays recorded in the state entry for audit,
            # and the preference is restored on load from the approved
            # induced_rule:* entries.
            induced = {d.get("id"): d for d in self._induced_pending()}
            if rule_id in induced:
                d = induced[rule_id]
                conds = [{"key": k, "op": "==", "value": v}
                         for k, v in sorted(d.get("situation", {}).items())]
                self.rules.add(Rule(id=rule_id, when=conds, then=[],
                                    priority=0.6, source="induction", status="active"))
                entry = self.state.get(f"induced_rule:{rule_id}")
                if entry is not None and isinstance(entry.value, dict):
                    new_d = {**entry.value, "status": "approved"}
                    self._set_state(f"induced_rule:{rule_id}", new_d,
                                    source="induction", confidence=0.5,
                                    provenance="induction", entry_type="belief",
                                    status="approved")
                action = d.get("action")
                if action:
                    self.planner.preferred_actions[str(action)] = 0.8
                self._event("rule_approved", f"rule '{rule_id}' approved")
                return True
        return bool(ok)

    def reject_rule(self, rule_id: str) -> bool:
        """Explicit human rejection of a proposed rule: it is retired so
        the same episodes never re-propose it. Delegates to
        Learner.reject_rule(); falls back to RuleEngine.reject when the
        learner method is absent. Returns True only when the rule
        existed and was pending."""
        fn = getattr(self.learner, "reject_rule", None)
        if callable(fn):
            return bool(fn(rule_id))
        rule = self.rules.get(rule_id)
        if rule is None or rule.status != "pending":
            return False
        self.rules.reject(rule_id)
        self._event("rule_rejected", f"rule '{rule_id}' rejected")
        return True

    # ---- Phase 9 §4.2: retirement of promoted knowledge ---------------
    def retire_pattern(self, pattern_id: str, reason: str = "") -> bool:
        """Explicit human retirement of a promoted pattern (Phase 9
        §4.2): the pattern leaves the library so the planner never
        applies it again, and the retirement is recorded as a
        ``pattern_retired`` event + a ``retired`` record in the
        library (never silent). The pattern's consolidation
        candidate is rejected too, so the next consolidation pass
        cannot re-promote it; and the library refuses to re-learn
        the same id, so the same evidence can never resurrect it.
        Returns True only when a pattern with that id was in the
        library (unknown / already-retired ids return False)."""
        pattern_id = str(pattern_id)
        record = self.patterns.retire_pattern(
            pattern_id, reason, cycle=self.cycle)
        if record is None:
            return False
        for cand in self.consolidator.all():
            if cand.get("kind") != "pattern":
                continue
            if cand.get("status") not in ("candidate", "promoted"):
                continue
            content = cand.get("content", {}) or {}
            if (content.get("name") == record["name"]
                    and content.get("kind") == record["kind"]
                    and content.get("effect") == record["effect"]):
                self.consolidator.reject(
                    cand["id"], reason or "pattern retired")
        self._event("pattern_retired",
                    f"pattern '{record['name']}' "
                    f"({record['kind']}:{record['effect']}) retired: "
                    f"{reason or 'human decision'}")
        return True

    def retired_patterns(self) -> list[dict]:
        """Retired-pattern records (copies) from the library."""
        return self.patterns.retired_patterns()

    @staticmethod
    def _restore_pattern_contradictions(data: dict) -> dict[str, list]:
        """Phase 10 (P10-8): restore per-pattern contradiction counts
        from revision.json's ``"pattern_contradictions"`` field
        (``{pattern_id: [{cycle, detail}, ...]}``). An absent field —
        a pre-P10-8 save — defaults to empty. Malformed entries are
        skipped, never fatal: a corrupt count is dropped, the rest
        of the revision table is not."""
        raw = data.get("pattern_contradictions") if isinstance(data, dict) else None
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list] = {}
        for pid, bucket in raw.items():
            if not isinstance(bucket, list):
                continue
            recs = []
            for rec in bucket:
                if not isinstance(rec, dict):
                    continue
                try:
                    recs.append({"cycle": int(rec.get("cycle", 0)),
                                 "detail": str(rec.get("detail", ""))})
                except (TypeError, ValueError):
                    continue
            out[str(pid)] = recs
        return out

    def record_pattern_contradiction(self, pattern_id: str,
                                     detail: str = "") -> dict:
        """Record one observation contradicting a promoted pattern's
        prediction (Phase 9 §4.2 trigger). Each observation is a
        ``pattern_contradicted`` event; when the count for one
        pattern reaches ``PATTERN_CONTRADICTION_THRESHOLD`` (3) the
        pattern is retired automatically with a reason naming the
        count. Returns ``{pattern_id, contradictions, retired}``.
        Observations are recorded even for an unknown id (they are
        evidence, not an error); retirement only happens when the
        pattern is actually in the library."""
        pattern_id = str(pattern_id)
        bucket = self._pattern_contradictions.setdefault(pattern_id, [])
        bucket.append({"cycle": self.cycle, "detail": str(detail or "")})
        n = len(bucket)
        self._event("pattern_contradicted",
                    f"pattern {pattern_id[:12]} contradicted "
                    f"({n}/{self.PATTERN_CONTRADICTION_THRESHOLD}): "
                    f"{detail or 'no detail'}")
        retired = False
        if n >= self.PATTERN_CONTRADICTION_THRESHOLD:
            retired = self.retire_pattern(
                pattern_id,
                reason=(f"contradicted by {n} observations"
                        + (f": {detail}" if detail else "")))
        return {"pattern_id": pattern_id, "contradictions": n,
                "retired": retired}

    def retire_rule(self, rule_id: str, reason: str = "") -> bool:
        """Explicit human retirement of a rule (Phase 9 §4.2): a
        thin wrapper over the learner approved set / RuleEngine.
        A pending proposed rule is rejected (retired in the engine);
        an approved learned rule leaves the learner's approved set
        and is retired in the engine; any other active engine rule
        is retired in the engine; an engine-induced (success) rule
        also has its state entry marked retired and its planner
        preference dropped. Every retirement is a ``rule_retired``
        event. Returns True only when something was retired
        (unknown / already-retired ids return False)."""
        rule_id = str(rule_id)
        retired = False
        fn = getattr(self.learner, "reject_rule", None)
        if callable(fn) and fn(rule_id):
            retired = True
        elif rule_id in getattr(self.learner, "_approved", {}):
            self.learner._approved.pop(rule_id, None)
            rule = self.rules.get(rule_id)
            if rule is not None:
                self.rules.reject(rule_id)
            retired = True
        else:
            rule = self.rules.get(rule_id)
            if rule is not None and rule.status in ("active", "pending"):
                self.rules.reject(rule_id)
                retired = True
        entry = self.state.get(f"induced_rule:{rule_id}")
        if (entry is not None and isinstance(entry.value, dict)
                and entry.value.get("status") == "approved"):
            new_d = {**entry.value, "status": "retired"}
            self._set_state(f"induced_rule:{rule_id}", new_d,
                            source="induction", confidence=0.5,
                            provenance="induction", entry_type="belief",
                            status="retired")
            action = entry.value.get("action")
            if action:
                self.planner.preferred_actions.pop(str(action), None)
            retired = True
        if retired:
            self._event("rule_retired",
                        f"rule '{rule_id}' retired: "
                        f"{reason or 'human decision'}")
        return retired

    def approve_claim(self, subject, relation, obj, source: str = "user"):
        """Explicit human approval of a claim as a fact (Phase 3 §3.2).
        Delegates to LongTermMemory.approve_claim (present in the Phase
        3 memory build) and returns the fact's status ("verified" for
        a fully trusted source with no functional conflict). If that
        method is absent, records the approval as a teacher-sourced
        fact via the existing LTM teach path instead."""
        memory = getattr(self, "memory", None) or self.ltm
        fn = getattr(memory, "approve_claim", None)
        if callable(fn):
            return fn(subject, relation, obj, source=source)
        _, fact = self.ltm.add(subject, relation, obj, source_id="teacher",
                               cycle=self.cycle)
        return fact.status

    # ------------------------------------------------------------------ persistence & inspection
    # Phase 3, versioned format (version 2). Durable knowledge (LTM, rules,
    # goals — including each goal's plan — state, WM, events, learner
    # episodes, last decision, stop reason, alternative plan) is
    # persisted; transient execution state (_incoming, _prev_obs,
    # _goal_thoughts, last_evaluation) is NOT — a loaded brain starts a
    # fresh execution context with its knowledge intact. Planner stats
    # are DERIVED from learner episodes (no planner.json is written)
    # and the attention seen-set lives in state (no attention.json).
    # meta.json carries a "files" manifest listing every file in the
    # save and its schema version. Old Phase 1–2 saves still load:
    # missing decision.json / learner.json default to empty, a missing
    # manifest is treated as v1 with a warning, and legacy planner.json
    # (actions/bans only — stats ignored) and attention.json (seen-set
    # seeded into state) are migrated silently. A v0.1 folder
    # (memory.json + rules.json only) still loads.
    # Phase 5: version 3 adds world.json (world model + pattern
    # library, schema v1). Old v2 saves still load: a missing
    # world.json means an empty model/library, silently unless the
    # manifest lists it (same treatment as other missing optional
    # files). Planner.preferred_actions is NOT stored — it is rebuilt
    # on load from the approved induced_rule:* state entries.
    # Phase 6: version 4 adds learning.json (consolidation queue,
    # concept store, calibration summary, prioritizer usefulness,
    # schema v1). Old v3 saves still load: a missing learning.json
    # means empty Phase 6 state, with a load_warnings entry (a v3
    # save is the only pre-v4 save with a versioned manifest to
    # tell it apart from a v0.1 folder, which keeps the silent
    # treatment of other missing optional files).
    # Phase 7: version 5 adds execution.json (live session + ledger,
    # schema v1). Old v4 saves still load: a missing execution.json
    # means empty execution state, with a load_warnings entry (a v4
    # save is the only pre-v5 save with a versioned manifest to tell
    # it apart from a v0.1 folder, which keeps the silent treatment
    # of other missing optional files). A session saved as "running"
    # loads as "interrupted" and re-validates on resume_goal.
    # Phase 9: version 6 adds revision.json (simulation-revision
    # table, schema v1; decision P9-2). Old v5 saves still load: a
    # missing revision.json means an empty revision table, with a
    # load_warnings entry.
    # Phase 11 (W2): version 7 adds research.json (researcher _seen
    # claims + source-quality history, schema v1). Old v6 saves still
    # load: a missing research.json means empty research state, with
    # a load_warnings entry.
    PERSISTENCE_VERSION = 7

    # Schema version per file in a Phase 3 save (the manifest values).
    FILE_SCHEMA_VERSIONS = {
        "meta.json": 2, "state.json": 1, "wm.json": 1, "goals.json": 1,
        "plan.json": 1, "memory.json": 1, "rules.json": 1, "learner.json": 1,
        "decision.json": 1, "events.json": 1, "world.json": 1,
        "learning.json": 1, "execution.json": 1, "revision.json": 1,
        "research.json": 1,
    }

    def save(self, folder: str) -> None:
        from dataclasses import asdict
        os.makedirs(folder, exist_ok=True)
        # Phase 3 §3.5: the event log is bounded at MAX_EVENTS, pruned
        # oldest-first, at save time.
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]
        self.ltm.save(os.path.join(folder, "memory.json"))
        self.rules.save_json(os.path.join(folder, "rules.json"))
        self._write_json(folder, "state.json", self.state.to_json())
        self._write_json(folder, "wm.json", self.wm.to_json())
        # goals.json carries each goal's plan (Phase 3 §2.2); plan.json
        # carries only the active goal's execution state, no steps.
        # Coordinator integration fix (Phase 7 gate): plans are
        # redacted at save time — a secret fed through an executed
        # tool lives in the plan's action args, and goals.json /
        # plan.json persisted it raw (provenance experiment: every
        # secret regardless of shape). See _redact_saved_actions.
        goals_data = self.goals.to_json()
        for _g in goals_data.get("goals", []):
            _g["plan"] = self._redact_saved_actions(_g.get("plan", []))
        self._write_json(folder, "goals.json", goals_data)
        planner_data = self.planner.to_json()
        planner_data["actions"] = self._redact_saved_actions(
            planner_data.get("actions", []))
        self._write_json(folder, "plan.json", {
            "goal_id": self.goals.active().id if self.goals.active() else None,
            "plan_started_len": self._plan_started_len,
            "plan_failed": self.plan_failed, "surprise": self.surprise,
            "goal_start": self.goal_start,
            "knowledge_keys": {k: list(v) for k, v in self._knowledge_keys.items()},
            "last_inferences": self.last_inferences,
            # Phase 2: stop reason, retained alternative plan, and the
            # monotonic successful-step count for the active goal.
            "last_stop_reason": self.last_stop_reason,
            "alternative_plan": self._redact_saved_actions(
                [asdict(a) for a in self.alternative_plan]),
            "alternative_plan_cost": self.alternative_plan_cost,
            "steps_executed": self._steps_executed,
            # Phase 4: the rules considered (matched, not just fired) in
            # the last cycle — execution state, restored on load.
            "last_considered_rules": list(self.last_considered_rules),
            # Phase 3 integration fix (2026-11-21): planner.json was
            # removed because planner *stats* are now derived from
            # learner episodes — but the planner's ACTIONS are not
            # duplicated anywhere else, and removing the file silently
            # dropped them on load (a brain loaded without constructor
            # actions could never plan again). Actions + banned pairs
            # therefore live in plan.json now; stats are still never
            # stored (Planner.to_json writes no stats).
            "planner": planner_data})
        self._write_json(folder, "events.json", self.events)
        # Phase 2: learner episodes (capped at 500 by Learner) and the
        # last decision. Grants are deliberately NOT persisted: the
        # loader chooses the authorization set explicitly.
        self._write_json(folder, "learner.json", self.learner.to_json())
        self._write_json(folder, "decision.json",
                         self.last_decision.to_json() if self.last_decision else None)
        # Phase 5: the world model + pattern library (schema v1).
        self._write_json(folder, "world.json",
                         {"model": self.world_model.to_json(),
                          "patterns": self.patterns.to_json()})
        # Phase 6: the learning subsystem (schema v1) — the
        # consolidation queue, concept store, calibration summary
        # and prioritizer usefulness. Episodes stay in learner.json
        # and patterns stay in world.json; this file holds only the
        # new Phase 6 state.
        self._write_json(folder, "learning.json",
                         {"version": 1,
                          "consolidation": self.consolidator.to_json(),
                          "concepts": self.concepts.to_json(),
                          "calibration": self.calibration.to_json(),
                          "prioritizer": self.prioritizer.to_json()})
        # Phase 7: the execution subsystem (schema v1) — the live
        # session (if any) and the ledger, plus the budget
        # configuration in the session record.
        self._write_json(folder, "execution.json",
                         self.execution_monitor.to_json())
        # Phase 9 §4.5 (decision P9-2): the simulation-revision
        # table (schema v1) — adjusted predicted success + the
        # adjustment history.
        # Phase 10 (P10-8): the per-pattern contradiction counts ride
        # inside revision.json under "pattern_contradictions" — no
        # persistence version bump (P8-7 precedent: new data on an
        # existing persisted structure). SimulationRevision ignores
        # the extra key on load; Brain.load restores it below.
        revision_data = self.simulation_revision.to_json()
        revision_data["pattern_contradictions"] = {
            pid: [dict(rec) for rec in bucket]
            for pid, bucket in sorted(self._pattern_contradictions.items())}
        self._write_json(folder, "revision.json", revision_data)
        # Phase 11 (W2): the researcher state (schema v1) — seen
        # claims (pending + admitted) and source-quality history. A
        # brain with no researcher writes the empty state, so the
        # manifest shape is the same either way.
        self._write_json(folder, "research.json",
                         self.researcher.to_json() if self.researcher is not None
                         else Researcher.empty_json())
        # meta.json last: its manifest lists every file in the save.
        self._write_json(folder, "meta.json",
                         {"version": self.PERSISTENCE_VERSION, "cycle": self.cycle,
                          "files": dict(self.FILE_SCHEMA_VERSIONS)})

    @staticmethod
    def _write_json(folder: str, name: str, data) -> None:
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

    def load(self, folder: str) -> None:
        """Load a folder saved by ``save`` (v2) or by Phase 1–2 (v1) or
        v0.1 (memory + rules).

        Missing Phase 1 files are skipped (v0.1 compatibility). A corrupt
        optional file is skipped with a warning recorded in
        ``self.load_warnings`` — the brain keeps working with the rest of
        its state. Corrupt ``memory.json`` / ``rules.json`` raise, because
        silently dropping durable knowledge would be worse than failing.
        A ``meta.json`` version newer than supported raises ValueError,
        as does an unknown per-file schema version in the manifest. A
        manifest-listed file that is missing is skipped with a warning.
        A missing manifest is treated as v1 with a warning.
        """
        # Phase 12 (P12-O5): load() replaces the state/goals stores
        # wholesale (fresh objects restart their version counters at
        # 0) and mutates LTM in place, so the world cache must be
        # invalidated here even though no store version can express it.
        self._world_token += 1
        self._world_cache = None
        self.load_warnings = []
        # Phase 6: the save's version / manifest, for the learning.json
        # warning below (a v3 save missing learning.json warns; a
        # v0.1 folder with no manifest stays silent).
        save_version = 0
        manifest = None
        meta = self._read_optional(folder, "meta.json")
        if meta is not None:
            version = meta.get("version", 0)
            save_version = int(version) if isinstance(version, int) else 0
            manifest = meta.get("files")
            if version > self.PERSISTENCE_VERSION:
                raise ValueError(f"unsupported persistence version {version} "
                                 f"(this build supports up to {self.PERSISTENCE_VERSION})")
            self.cycle = int(meta.get("cycle", 0))
            manifest = meta.get("files")
            if not manifest:
                self.load_warnings.append(
                    "meta.json: no files manifest; treating save as v1")
            else:
                for fname, fver in sorted(manifest.items()):
                    known = self.FILE_SCHEMA_VERSIONS.get(fname)
                    if known is None or fver != known:
                        raise ValueError(
                            f"unsupported schema version {fver!r} for {fname} "
                            f"(known: {known!r})")
                    if fname != "meta.json" and not os.path.exists(os.path.join(folder, fname)):
                        self.load_warnings.append(
                            f"{fname}: listed in manifest but missing; skipped")
        self.ltm.load(os.path.join(folder, "memory.json"))
        with open(os.path.join(folder, "rules.json"), encoding="utf-8") as fh:
            for d in json.load(fh):
                self.rules.upsert(d)

        def _restore(name, fn):
            data = self._read_optional(folder, name)
            if data is None:
                return
            try:
                fn(data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(f"{name}: skipped ({type(exc).__name__}: {exc})")

        _restore("state.json", lambda d: setattr(self, "state", CognitiveState.from_json(d)))
        _restore("wm.json", lambda d: setattr(self, "wm", WorkingMemory.from_json(d)))
        _restore("goals.json", lambda d: setattr(self, "goals", GoalManager.from_json(d)))
        _restore("attention.json", self._restore_attention_legacy)
        _restore("events.json", lambda d: setattr(self, "events", list(d)))

        # Phase 1–2 saves carry planner.json: migrate its actions and
        # bans; its stats are IGNORED (derived from learner episodes).
        # No warning: old saves load silently (see the Phase 3 report).
        _restore("planner.json", self._restore_planner_legacy)

        # Phase 2: learner episodes restore after the planner (they need
        # the restored planner); a missing learner.json means a Phase 1
        # save, whose episodes were transient by design.
        def _restore_learner(d):
            self.learner = Learner.from_json(d, self.planner, self.rules)
        _restore("learner.json", _restore_learner)

        def _restore_decision(d):
            self.last_decision = Decision.from_json(d) if isinstance(d, dict) else None
        _restore("decision.json", _restore_decision)

        # Phase 5: world.json (schema v1) — the world model + pattern
        # library. A missing file is a pre-v3 save: the brain starts
        # with an empty model and library, silently, like every other
        # missing optional file (a v3 save whose manifest LISTS
        # world.json but lacks the file already warned above, in the
        # manifest check). A corrupt world.json is skipped with a
        # warning and the brain keeps working with the rest.
        world_data = self._read_optional(folder, "world.json")
        if world_data is None:
            self.world_model = WorldModel()
            self.patterns = PatternLibrary()
        else:
            try:
                self._restore_world(world_data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(
                    f"world.json: skipped ({type(exc).__name__}: {exc})")
                self.world_model = WorldModel()
                self.patterns = PatternLibrary()

        # Phase 6: learning.json (schema v1) — the consolidation
        # queue, concept store, calibration summary and prioritizer
        # usefulness. A missing file in a v3 save means empty Phase 6
        # state, with a warning (design §2.2); a v0.1 folder or a
        # v1/v2 save keeps the silent treatment of other missing
        # optional files. A corrupt learning.json is skipped with a
        # warning by _read_optional and the brain keeps working with
        # the rest (the file exists, so no pre-v4 warning is added).
        learning_data = self._read_optional(folder, "learning.json")
        if learning_data is None:
            if (save_version == 3
                    and not os.path.exists(os.path.join(folder, "learning.json"))):
                self.load_warnings.append(
                    "learning.json: missing (pre-v4 save); Phase 6 state "
                    "starts empty")
        else:
            try:
                self._restore_learning(learning_data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(
                    f"learning.json: skipped ({type(exc).__name__}: {exc})")

        def _restore_plan(d):
            # Phase 3: the plan lives in the goal record (goals.json),
            # so plan.json carries execution state only. A legacy
            # plan.json that still has "steps" migrates them into the
            # owning goal's record when it has no plan yet.
            self._plan_started_len = int(d.get("plan_started_len", 0))
            self.plan_failed = bool(d.get("plan_failed", False))
            self.surprise = bool(d.get("surprise", False))
            self.goal_start = dict(d.get("goal_start", {}))
            self._knowledge_keys = {k: tuple(v) for k, v in d.get("knowledge_keys", {}).items()}
            self.last_inferences = list(d.get("last_inferences", []))
            # Phase 2 additions (defaults keep Phase 1 saves loadable).
            self.last_stop_reason = d.get("last_stop_reason")
            # Phase 3 integration fix: restore the planner's actions +
            # banned pairs from plan.json (see save). Without this, a
            # brain loaded without constructor actions has an empty
            # planner and can never plan. Legacy planner.json (Phase 1-2
            # saves) is migrated separately above; both paths merge.
            if isinstance(d.get("planner"), dict):
                self._restore_planner_legacy(d["planner"])
            self.alternative_plan = [Action(x["name"], dict(x.get("pre", {})), dict(x.get("eff", {})),
                                            float(x.get("cost", 1.0)), dict(x.get("args", {})),
                                            x.get("tool"), x.get("store_as"))
                                     for x in d.get("alternative_plan", [])]
            self.alternative_plan_cost = float(d.get("alternative_plan_cost", 0.0))
            self._steps_executed = int(d.get("steps_executed", 0))
            self.last_considered_rules = list(d.get("last_considered_rules", []))
            legacy_steps = d.get("steps", [])
            if legacy_steps:
                target = None
                if d.get("goal_id") is not None:
                    try:
                        target = self.goals.get(int(d["goal_id"]))
                    except (KeyError, TypeError, ValueError):
                        target = None
                if target is None:
                    target = self.goals.active()
                if target is not None and not target.plan:
                    self.goals.set_plan(target.id, [Action(
                        x["name"], dict(x.get("pre", {})), dict(x.get("eff", {})),
                        float(x.get("cost", 1.0)), dict(x.get("args", {})),
                        x.get("tool"), x.get("store_as")) for x in legacy_steps])
        _restore("plan.json", _restore_plan)
        # Phase 7: execution.json (schema v1) — the live session and
        # the ledger. A missing file in a v4 save means empty
        # execution state, with a warning (same treatment as
        # learning.json in a v3 save); a v0.1 folder or a v1–v3 save
        # keeps the silent treatment of other missing optional files.
        # A corrupt execution.json is skipped with a warning by
        # _read_optional and the brain keeps working with the rest
        # (the file exists, so no pre-v5 warning is added). A session
        # saved as "running" loads as "interrupted": the task did not
        # finish before the save, so it resumes as interrupted and
        # re-validates its next action on resume_goal.
        execution_data = self._read_optional(folder, "execution.json")
        if execution_data is None:
            if (save_version == 4
                    and not os.path.exists(os.path.join(folder, "execution.json"))):
                self.load_warnings.append(
                    "execution.json: missing (pre-v5 save); execution state "
                    "starts empty")
            self.execution_monitor = ExecutionMonitor()
        else:
            try:
                self.execution_monitor = ExecutionMonitor.from_json(execution_data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(
                    f"execution.json: skipped ({type(exc).__name__}: {exc})")
                self.execution_monitor = ExecutionMonitor()
        # Coordinator integration fix (Phase 7 gate): the restored
        # monitor/session must use the spec-aware redactor too, or a
        # resume after load would rebuild the session snapshot from
        # live plan steps with only redact_secrets (re-opening the
        # provenance leak for short secrets).
        self.execution_monitor.redact_args = self._redact_tool_args
        # Phase 10.5: same re-set for the planner (see __init__):
        # the planner object survives load, but a resumed brain must
        # keep deriving ban keys and failure-rate identities from
        # redacted args, or a secret-marked tool argument would leak
        # into plan.json on the next save.
        self.planner.redact_args = self._redact_tool_args
        if self.execution_monitor.session is not None:
            self.execution_monitor.session.redact_args = self._redact_tool_args
        session = self.execution_monitor.session
        if session is not None and session.status == "running":
            # No event is logged here: load() must not change the
            # event log (test_phase1_persistence asserts an exact
            # round-trip of events). The interruption is visible in
            # the session status / stop_reason, and resume_goal logs
            # execution_resumed when the task continues.
            session.status = "interrupted"
            session.stop_reason = "interrupted by load"
        # Phase 9 §4.5: revision.json (schema v1) — the
        # simulation-revision table. A missing file in a v5 save
        # means an empty revision table, with a warning (same
        # treatment as learning.json in a v3 save / execution.json
        # in a v4 save); a v0.1 folder or a v1–v4 save keeps the
        # silent treatment of other missing optional files. A
        # corrupt revision.json is skipped with a warning by
        # _read_optional and the table starts empty (the file
        # exists, so no pre-v6 warning is added).
        revision_data = self._read_optional(folder, "revision.json")
        if revision_data is None:
            if (save_version == 5
                    and not os.path.exists(os.path.join(folder, "revision.json"))):
                self.load_warnings.append(
                    "revision.json: missing (pre-v6 save); revision table "
                    "starts empty")
            self.simulation_revision = SimulationRevision()
            # Phase 10 (P10-8): no revision file means no stored
            # contradiction counts either.
            self._pattern_contradictions = {}
        else:
            try:
                self.simulation_revision = SimulationRevision.from_json(
                    revision_data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(
                    f"revision.json: skipped ({type(exc).__name__}: {exc})")
                self.simulation_revision = SimulationRevision()
                self._pattern_contradictions = {}
            else:
                # Phase 10 (P10-8): restore the per-pattern
                # contradiction counts; a missing field (a pre-P10-8
                # save) defaults to empty. The retirement threshold
                # logic reads these restored counts, so a
                # contradiction observed before a restart still counts
                # toward retirement after it.
                self._pattern_contradictions = (
                    self._restore_pattern_contradictions(revision_data))
        # The simulator (if already built) must see the restored
        # table; _get_simulator also re-links it lazily.
        if getattr(self, "simulator", None) is not None:
            self.simulator.revision = self.simulation_revision
        # Phase 11 (W2): research.json (schema v1) — the researcher's
        # seen claims and source-quality history. A missing file in a
        # v6 save means empty research state, with a warning (same
        # treatment as revision.json in a v5 save); a v0.1 folder or a
        # v1–v5 save keeps the silent treatment of other missing
        # optional files. A corrupt research.json is skipped with a
        # warning by _read_optional / the from_json guard below and
        # the researcher state starts empty (the file exists, so no
        # pre-v7 warning is added). A manifest-listed research.json
        # that is missing warns in the manifest check above. The
        # recorded provider_spec is NOT applied here — restoring the
        # researcher's provider is the Brain-level swap's job.
        research_data = self._read_optional(folder, "research.json")
        if research_data is None:
            if (save_version == 6
                    and not os.path.exists(os.path.join(folder, "research.json"))):
                self.load_warnings.append(
                    "research.json: missing (pre-v7 save); research state "
                    "starts empty")
            if self.researcher is not None:
                self.researcher.reset_state()
        elif self.researcher is not None:
            try:
                self.researcher.from_json(research_data)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.load_warnings.append(
                    f"research.json: skipped ({type(exc).__name__}: {exc})")
                self.researcher.reset_state()
        # Phase 11 (W5): re-register the isolated research tool with
        # the loaded provider. The recorded provider_spec is applied
        # ONLY when this brain's researcher has no provider — the
        # provider the Brain was constructed with is the source of
        # truth (W2 deliberately does not apply the spec in
        # Researcher.from_json). When no provider is recorded and the
        # researcher has none, the tool is registered with a None
        # spec: research then fails gracefully via the no-provider
        # guard in Researcher.collect.
        if self.researcher is not None:
            if (self.researcher.provider is None
                    and isinstance(research_data, dict)):
                spec = research_data.get("provider_spec")
                if isinstance(spec, dict):
                    try:
                        self.researcher.set_provider(provider_from_spec(spec))
                    except ValueError as exc:
                        self.load_warnings.append(
                            f"research provider_spec: skipped ({exc})")
            self._register_research_tool()
        # Phase 3: the attention seen-set lives in state — re-seed it
        # from the seen:* entries (including ones seeded from a legacy
        # attention.json above).
        self._seed_attention_from_state()
        # The learner may have been replaced by _restore_learner;
        # re-attach it (and the rules engine) so planner stats stay
        # derived from learner episodes.
        self.planner.attach_learner(self.learner)
        self.planner.attach_rules(self.rules)
        # Phase 5: the pattern library may have been replaced by the
        # world restore — re-attach it (the context lambda reads the
        # live world model, so it needs no update) — and rebuild the
        # induced-rule cost preferences from the approved
        # induced_rule:* state entries.
        self.planner.attach_patterns(
            self.patterns,
            lambda: (self.world_model, self.planner.actions, dict(self.world)))
        # Phase 6: the learner may have been replaced by
        # _restore_learner, so re-attach its on_record hook and
        # rebuild the retriever index from the restored episodes;
        # re-attach the (possibly restored) concept store too.
        self.learner.on_record = self._on_episode_recorded
        self._retriever_episodes = [
            {"action": ep.action, "situation": dict(ep.situation),
             "outcome": "success" if ep.ok else "failure",
             "cycle": ep.cycle}
            for ep in self.learner.episodes]
        self.retriever.index(self._retriever_episodes)
        self.planner.attach_concepts(self.concepts)
        self._rebuild_preferred_actions()
        # Transient execution state starts fresh after a load (the
        # Phase 7 execution session / ledger above are the exception:
        # they are restored, with a running session as interrupted).
        self._incoming, self._prev_obs, self._goal_thoughts = [], {}, {}
        self.last_evaluation = None

    def _restore_attention_legacy(self, d) -> None:
        """Migrate a Phase 1–2 attention.json: weights/threshold are
        restored, and each seen signature is seeded into state as a
        seen:* entry (Phase 3 §2.5) and into Attention._seen."""
        att = Attention.from_json(d)
        self.attention = att
        for sig in d.get("seen", []):
            digest = hashlib.sha256(str(sig).encode()).hexdigest()[:16]
            key = f"seen:{digest}"
            if key not in self.state:
                self.state.set(key, str(sig), source="attention", confidence=1.0,
                               cycle=self.cycle, provenance="attention:legacy",
                               entry_type="observation")
        self.attention.restore_seen(d.get("seen", []))

    def _restore_learning(self, data) -> None:
        """Restore a v4 learning.json: the consolidation queue, the
        concept store, the calibration summary and the prioritizer
        usefulness counts. Promoted pattern candidates are re-added
        to the library (their patterns also live in world.json, so
        this is a belt-and-braces re-admission, deduped by id). A
        malformed half raises and the caller falls back to fresh
        Phase 6 state (with a warning)."""
        if not isinstance(data, dict):
            raise ValueError("learning.json must be a dict")
        cons_data = data.get("consolidation")
        if isinstance(cons_data, dict):
            self.consolidator = Consolidator.from_json(cons_data)
        concepts_data = data.get("concepts")
        if isinstance(concepts_data, dict):
            self.concepts = ConceptStore.from_json(concepts_data)
        cal_data = data.get("calibration")
        if isinstance(cal_data, dict):
            self.calibration = CalibrationTracker.from_json(cal_data)
        pri_data = data.get("prioritizer")
        if isinstance(pri_data, dict):
            self.prioritizer = Prioritizer.from_json(pri_data)
        for cand in self.consolidator.promoted():
            pattern = self._pattern_from_candidate(cand)
            if pattern is not None:
                self.patterns.add_pattern(pattern)
        # The concept dedupe sets are pre-seeded from the restored
        # episodes: the per-cycle concept pass runs at the end of
        # every cycle, so every example derivable from these episodes
        # was already recorded before the save — pre-seeding keeps
        # the loaded brain from re-adding the same examples.
        c1_keys, c2_keys = self._concept_keys(list(self.learner.episodes))
        self._c1_seen = {(tool, broke) for tool, broke, _n in c1_keys}
        self._c2_seen = {(action, key) for action, key, _f, _s in c2_keys}

    def _restore_world(self, data) -> None:
        """Restore a v3 world.json: the world model and the pattern
        library. A missing "model"/"patterns" key means an empty half;
        a malformed half raises and the caller falls back to fresh
        (with a warning)."""
        if not isinstance(data, dict):
            raise ValueError("world.json must be a dict")
        model_data = data.get("model")
        if isinstance(model_data, dict):
            self.world_model = WorldModel.from_json(model_data)
        patterns_data = data.get("patterns")
        if isinstance(patterns_data, dict):
            self.patterns = PatternLibrary.from_json(patterns_data)

    def _rebuild_preferred_actions(self) -> None:
        """Phase 5: rebuild Planner.preferred_actions from the
        approved induced_rule:* state entries (factor 0.8 each).
        The preference itself is never stored in planner.json — the
        approved state entries are its persistence."""
        self.planner.preferred_actions = {}
        for key in self.state.keys():
            if not key.startswith("induced_rule:"):
                continue
            entry = self.state.get(key)
            if not isinstance(entry.value, dict):
                continue
            if entry.value.get("status") != "approved":
                continue
            action = entry.value.get("action")
            if action:
                self.planner.preferred_actions[str(action)] = 0.8

    def _restore_planner_legacy(self, d) -> None:
        """Migrate a Phase 1–2 planner.json: its actions and banned
        pairs are merged into this planner. Its stats are deliberately
        NOT restored — planner stats are derived from learner episodes
        (Phase 3 §2.4)."""
        legacy = Planner.from_json(d)
        have = {a.name for a in self.planner.actions}
        for a in legacy.actions:
            if a.name not in have:
                self.planner.add_action(a)
                # Phase 10.5 (P10.5-9): keep the restored action's
                # generation (add_action tags with the current one).
                self.planner._action_generation[action_identity(a)] = \
                    legacy.generation_of(a)
        self.planner.banned_pairs |= legacy.banned_pairs
        self.planner._generation = max(self.planner._generation,
                                       legacy._generation)

    def _read_optional(self, folder: str, name: str):
        """Read an optional Phase 1 file: None if missing, and on corrupt
        JSON record a warning and return None. Schema errors from the
        caller's ``from_json`` are caught by ``_restore`` in ``load``."""
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            self.load_warnings.append(f"{name}: skipped ({type(exc).__name__}: {exc})")
            return None

    def _event(self, kind: str, detail: str) -> None:
        # Phase 3 §3.5: every recorded detail is redacted via
        # audit.redact_secrets before it enters the trail. Events use
        # the schema-v1 shape {"cycle", "kind", "detail"} (see
        # ab/core/audit.py EVENT_FIELDS). Phase 12 (ADR P12-10): the
        # trail is capped at MAX_EVENTS in-cycle, oldest-first — the
        # cap used to be enforced only at save time, so the log grew
        # linearly with cycles during a run (10,000-cycle soak:
        # 123,567 events). The save-time prune stays as belt-and-braces.
        self.events.append({"cycle": self.cycle, "kind": kind,
                            "detail": redact_secrets(detail)})
        if len(self.events) > MAX_EVENTS:
            del self.events[:len(self.events) - MAX_EVENTS]

    def explain(self) -> list[str]:
        return [f"[{e['cycle']:>3}] {e['kind']:<9} {e['detail']}" for e in self.events]


def _tags(text: str) -> set[str]:
    """Backwards-compatible name: the single tokenizer (see ab/core/vocab.py)."""
    return tokenize(text)


def _tool_entity_name(action_name: str) -> str | None:
    """The entity a tool action operates with, by key/action shape
    only (never a world name): strip a leading ``use_`` prefix
    (``use_wrench`` -> ``wrench``). Returns None when the action
    names no tool — a bare action has no tool entity to suspect."""
    name = str(action_name or "")
    if name.startswith("use_") and len(name) > 4:
        return name[4:]
    return None
