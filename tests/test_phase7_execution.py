"""Phase 7, Worker 2: execution system + Brain integration (D7-1/D7-2).

Covers ``ab/execution/`` (session / monitor / ledger) and the Brain
hooks: session lifecycle, budget/cancel enforcement in step(),
dependency checks, action verification, permission-aware plan
filtering at both layers, strategy selection + retry cap,
cancellation, interruption mid-execution, and the §4.2
prioritized-pruning fix.

Test worlds live ONLY in this file (never in ``ab/``): a small
bench world with 3 tool calls + 3 env actions on the successful path.
Tool functions are module-level (fork-safe: the brain runs tools in
isolated mode, so a closure counter would not propagate — counts are
asserted via the registry audit / session / ledger instead).
"""
import os
import tempfile
import threading
import time
import unittest

from ab.brain import Brain
from ab.body.environment import Environment, RoomWorld, room_world_actions
from ab.body.tools import ToolSpec
from ab.planning.planner import Action


# ----------------------------------------------------------------------
# Bench world: 3 env actions + 3 tool calls on the successful path.
# The env observes only ITS OWN keys (at/has_part/assembled); tool
# effect keys (cut_done/drill_done/measured) live in the brain's
# state via store_as — if the env reported them it would clobber the
# tool results on every sense.
# ----------------------------------------------------------------------
class BenchEnv(Environment):
    def __init__(self):
        self.s = {"at": "store", "has_part": False, "assembled": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict):
        s = self.s
        if name == "go_bench" and s["at"] == "store":
            s["at"] = "bench"
            ok, info = True, "at bench"
        elif name == "take_part" and s["at"] == "bench" and not s["has_part"]:
            s["has_part"] = True
            ok, info = True, "took part"
        elif name == "assemble" and s["at"] == "bench":
            s["assembled"] = True
            ok, info = True, "assembled"
        else:
            ok, info = False, "nothing happened"
        self.log.append(f"{name}: {info}")
        return ok, info


def _cut() -> bool:
    return True


def _drill() -> bool:
    return True


def _measure() -> bool:
    return True


def _boom() -> bool:
    raise ValueError("bit broken")


def _drill_alt() -> bool:
    return True


def _slow() -> bool:
    # Writes a start marker (visible across the fork: a file, not a
    # closure counter) so the test can cancel exactly while it runs.
    if _SLOW_MARKER["path"] is not None:
        with open(_SLOW_MARKER["path"], "w", encoding="utf-8") as fh:
            fh.write("started")
    time.sleep(1.5)
    return True


_SLOW_MARKER: dict = {"path": None}


def _bench_actions(drill_cost: float = 1.0,
                   with_alt: bool = False) -> list[Action]:
    acts = [
        Action("go_bench", {"at": "store"}, {"at": "bench"},
               args={"_do": "go_bench"}),
        Action("take_part", {"at": "bench", "has_part": False},
               {"has_part": True}, args={"_do": "take_part"}),
        Action("cut_part", {"has_part": True}, {"cut_done": True},
               tool="cut", store_as="cut_done"),
        Action("drill_part", {"has_part": True}, {"drill_done": True},
               cost=drill_cost, tool="drill", store_as="drill_done"),
        Action("measure_part", {"cut_done": True, "drill_done": True},
               {"measured": True}, tool="measure", store_as="measured"),
        Action("assemble", {"cut_done": True, "drill_done": True,
                            "measured": True},
               {"assembled": True}, args={"_do": "assemble"}),
    ]
    if with_alt:
        # A dearer alternative way to drill (cost 5.0): the planner
        # prefers the cheap drill first, so the brain only adopts
        # this one after the cheap strategy fails (strategy
        # preference, not cost).
        acts.append(Action("drill_part_alt", {"has_part": True},
                           {"drill_done": True}, cost=5.0,
                           tool="drill2", store_as="drill_done"))
    return acts


def _bench_brain(drill_fn=_drill, drill_cost: float = 1.0,
                 with_alt: bool = False, **kwargs) -> Brain:
    b = Brain(_bench_actions(drill_cost=drill_cost, with_alt=with_alt),
              granted={"compute", "environment"}, **kwargs)
    b.tools.register(ToolSpec(name="cut", args=[], return_type="bool",
                              timeout=5.0, required_grant="compute",
                              version=1), _cut)
    b.tools.register(ToolSpec(name="drill", args=[], return_type="bool",
                              timeout=5.0, required_grant="compute",
                              version=1), drill_fn)
    b.tools.register(ToolSpec(name="measure", args=[], return_type="bool",
                              timeout=5.0, required_grant="compute",
                              version=1), _measure)
    if with_alt:
        b.tools.register(ToolSpec(name="drill2", args=[],
                                  return_type="bool", timeout=5.0,
                                  required_grant="compute", version=1),
                         _drill_alt)
    return b


def _events(brain, kind: str) -> list[dict]:
    return [e for e in brain.events if e["kind"] == kind]


# ----------------------------------------------------------------------
# Multi-step execution with monitoring
# ----------------------------------------------------------------------
class TestMultiToolExecution(unittest.TestCase):
    def test_multi_tool_task_completes_with_session_and_ledger(self):
        env = BenchEnv()
        b = _bench_brain()
        g = b.add_goal("assemble the part", {"assembled": True})
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["assembled"])
        session = b.execution_monitor.session
        self.assertIsNotNone(session)
        self.assertEqual(session.status, "done")
        self.assertEqual(session.goal_id, g.id)
        self.assertEqual(session.steps_executed, 6)
        self.assertEqual(session.steps_total, 6)
        self.assertEqual(session.tool_calls, 3)
        # Stable task id: 64 hex chars (SHA-256 of goal + plan + cycle).
        self.assertEqual(len(session.task_id), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in session.task_id))
        # One ledger record per executed step, in order, each ok and
        # verified, each naming the choosing rule and the plan cost.
        ledger = b.execution_monitor.ledger
        self.assertEqual(len(ledger.records), 6)
        self.assertEqual([r["action"] for r in ledger.records],
                         ["go_bench", "take_part", "cut_part", "drill_part",
                          "measure_part", "assemble"])
        self.assertTrue(all(r["outcome"] == "ok" for r in ledger.records))
        self.assertTrue(all(r["verified"] is True for r in ledger.records))
        self.assertEqual([r["tool"] for r in ledger.records],
                         [None, None, "cut", "drill", "measure", None])
        self.assertTrue(all(r["decision"].get("rule_id") == "execute_plan"
                            for r in ledger.records))
        self.assertTrue(all(r["decision"].get("plan_cost", 0) > 0
                            for r in ledger.records))
        self.assertTrue(any(e["kind"] == "execution_started"
                            for e in b.events))
        self.assertTrue(any(e["kind"] == "execution_finished"
                            for e in b.events))

    def test_session_budgets_default_to_generous_values(self):
        env = BenchEnv()
        b = _bench_brain()
        b.add_goal("assemble the part", {"assembled": True})
        b.step(env)   # select
        b.step(env)   # make plan -> session opens
        session = b.execution_monitor.session
        self.assertIsNotNone(session)
        self.assertEqual(session.budgets["max_tool_calls"], 50)
        self.assertEqual(session.budgets["max_wall_s"], 60.0)
        self.assertEqual(session.budgets["max_cycles"], b.control.max_cycles)


# ----------------------------------------------------------------------
# Failing tool + recovery / strategy selection + retry cap
# ----------------------------------------------------------------------
class TestFailingToolRecovery(unittest.TestCase):
    def test_failing_tool_recovers_via_different_strategy(self):
        # The cheap drill always raises; the dearer drill2 works. The
        # brain must record the failure, pick a different strategy
        # (first-action tool differs) and complete — the failing
        # tool's garbage never reaches state.
        env = BenchEnv()
        b = _bench_brain(drill_fn=_boom, with_alt=True)
        g = b.add_goal("assemble the part", {"assembled": True})
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["assembled"])
        self.assertTrue(any("drill" in e["detail"]
                            for e in _events(b, "strategy_failed")))
        # The recovery plan uses the DIFFERENT tool (drill2), not the
        # failed one. (The switch happens through the failure ban in
        # this flow — the preference sort itself is covered by
        # test_strategy_preference_prefers_different_tool.)
        ledger = b.execution_monitor.ledger.records
        failed = [r for r in ledger if r["action"] == "drill_part"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["outcome"], "failed")
        self.assertIs(failed[0]["verified"], False)
        self.assertTrue(any(r["action"] == "drill_part_alt"
                            and r["outcome"] == "ok" for r in ledger))
        # The failed drill never wrote state: no drill_done from it
        # (the successful alt wrote it exactly once, at the end).
        self.assertIs(b.world.get("drill_done"), True)

    def test_strategy_preference_prefers_different_tool(self):
        # Focused unit test for the _do_make_plan preference sort:
        # with a failed strategy recorded, the brain prefers a
        # candidate plan whose FIRST step uses a different tool,
        # even when the same-strategy plan is cheaper.
        # Phase 10.5 (P10.5-5, W2) pin update (Phase 2 §9
        # precedent — same intent, new intended semantics, not
        # weakened): the expected winner was ``cut_part`` (tool
        # ``cut``, cost 4.0 vs the drill plan's 3.0). But the test
        # world already has ``cut_done`` True, and ``cut_part``
        # is a tool action whose ``store_as`` (``cut_done``) is
        # the very key its effects declare — so in the model the
        # step changes nothing, and the P10.5-5 planner prune
        # removes the redundant cut-first candidate. The
        # preference sort then picks the next different-tool
        # candidate, ``drill_part_alt`` (tool ``drill2``, cost
        # 7.0 vs 3.0) — which is the candidate that does the
        # outstanding work (``drill_done`` is still False). The
        # mechanism under test is unchanged: a different tool is
        # preferred despite the higher cost.
        env = BenchEnv()
        b = _bench_brain(drill_cost=1.0, with_alt=True)
        b.add_goal("assemble the part", {"assembled": True})
        b.world = {"at": "bench", "has_part": True, "cut_done": True,
                   "drill_done": False, "measured": False,
                   "assembled": False}
        b.goals.activate(b.goals.pending()[0].id)
        b._failed_strategy = "drill"
        b._do_make_plan({}, env)
        # The cheapest plan starts with the failed tool (drill); the
        # preference sort adopts the other candidate instead — its
        # first step uses a different tool (drill2), even though it
        # costs more (7.0 vs 3.0).
        self.assertEqual(b.plan[0].name, "drill_part_alt")
        self.assertNotEqual(b._strategy_name(b.plan[0]), "drill")
        self.assertTrue(any(e["kind"] == "strategy_preferred"
                            for e in b.events))

    def test_strategy_exhausted_after_two_failures_same_strategy(self):
        acts = [Action("do_it", {}, {"done": True}, tool="boom",
                       store_as="done")]
        b = Brain(acts, granted={"compute"})
        b.tools.register(ToolSpec(name="boom", args=[], return_type="bool",
                                  timeout=5.0, required_grant="compute",
                                  version=1), _boom)
        g = b.add_goal("do it", {"done": True})
        self.assertEqual(b.run(None, 60), "impossible")
        self.assertEqual(g.status, "failed")
        self.assertEqual(g.fail_reason, "strategy_exhausted")
        failed = _events(b, "strategy_failed")
        self.assertEqual(len(failed), 2)
        self.assertTrue(all("boom" in e["detail"] for e in failed))
        ledger = b.execution_monitor.ledger.records
        self.assertEqual(len([r for r in ledger if r["action"] == "do_it"]), 2)
        self.assertTrue(all(r["outcome"] == "failed" for r in ledger))
        # Partial results preserved: the ledger is the record of what
        # was attempted, and the goal record keeps the failure reason.
        self.assertEqual(b.last_stop_reason, "impossible")


# ----------------------------------------------------------------------
# Dependency handling
# ----------------------------------------------------------------------
class TestDependencyHandling(unittest.TestCase):
    def test_check_dependencies_names_missing_keys(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        take_key = next(a for a in b.planner.actions if a.name == "take_key")
        self.assertEqual(b.check_dependencies(take_key, {"at": "hall",
                                                         "has_key": False}),
                         ["at"])
        self.assertEqual(
            b.check_dependencies(take_key, {"at": "lab", "has_key": False}), [])
        go = next(a for a in b.planner.actions if a.name == "go_vault")
        self.assertEqual(
            b.check_dependencies(go, {"at": "lab", "door_open": False}),
            ["door_open"])

    def test_dependency_blocked_never_executes_and_recovers(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        take_key = next(a for a in b.planner.actions if a.name == "take_key")
        b._set_plan([take_key])
        self.assertFalse(b._do_execute_step({}, env))
        blocked = _events(b, "dependency_blocked")
        self.assertEqual(len(blocked), 1)
        self.assertIn("take_key", blocked[0]["detail"])
        self.assertIn("at", blocked[0]["detail"])
        self.assertEqual(env.log, [])   # the blocked action never ran
        self.assertEqual(b.execution_monitor.ledger.records, [])


# ----------------------------------------------------------------------
# Permission-aware plan filtering, both layers
# ----------------------------------------------------------------------
class TestPermissionAwarePlanning(unittest.TestCase):
    def test_tool_without_grant_rejected_at_plan_layer_never_runs(self):
        marker = os.path.join(tempfile.mkdtemp(), "vip_ran.txt")

        def _vip() -> bool:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("ran")
            return True

        b = Brain([Action("do_vip", {}, {"vip_done": True}, tool="vip",
                          store_as="vip_done")],
                  granted={"compute"})
        b.tools.register(ToolSpec(name="vip", args=[], return_type="bool",
                                  timeout=5.0, required_grant="special",
                                  version=1), _vip)
        g = b.add_goal("do the vip thing", {"vip_done": True})
        self.assertEqual(b.run(None, 30), "impossible")
        self.assertEqual(g.status, "failed")
        self.assertEqual(g.fail_reason, "no_permitted_plan")
        denied = _events(b, "plan_tool_denied")
        self.assertTrue(denied)
        self.assertTrue(any("vip" in e["detail"] and "special" in e["detail"]
                            for e in denied))
        self.assertFalse(os.path.exists(marker))   # the tool never ran

    def test_env_without_grant_rejected_at_plan_layer(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        self.assertEqual(b.run(env, 60), "impossible")
        self.assertEqual(g.fail_reason, "no_permitted_plan")
        self.assertEqual(env.log, [])
        self.assertTrue(any(e["kind"] == "deny" for e in b.events))
        self.assertTrue(any(e["kind"] == "plan_tool_denied" for e in b.events))

    def test_call_site_refusal_is_second_line_of_defence(self):
        # The plan is forced past the planner (set directly), so the
        # call-site refusal in ToolRegistry.call / _do_execute_step
        # is what stops the tool — recorded as a denied ledger entry.
        marker = os.path.join(tempfile.mkdtemp(), "vip_ran.txt")

        def _vip() -> bool:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("ran")
            return True

        b = Brain([Action("do_vip", {}, {"vip_done": True}, tool="vip",
                          store_as="vip_done")],
                  granted={"compute"})
        b.tools.register(ToolSpec(name="vip", args=[], return_type="bool",
                                  timeout=5.0, required_grant="special",
                                  version=1), _vip)
        g = b.add_goal("do the vip thing", {"vip_done": True})
        b.goals.activate(g.id)
        b._set_plan([Action("do_vip", {}, {"vip_done": True}, tool="vip",
                            store_as="vip_done")])
        self.assertFalse(b._do_execute_step({}, None))
        self.assertFalse(os.path.exists(marker))
        ledger = b.execution_monitor.ledger.records
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["outcome"], "denied")
        self.assertEqual(ledger[0]["tool"], "vip")


# ----------------------------------------------------------------------
# Budgets
# ----------------------------------------------------------------------
class TestBudgets(unittest.TestCase):
    def _planned_brain(self):
        env = BenchEnv()
        b = _bench_brain()
        b.add_goal("assemble the part", {"assembled": True})
        b.step(env)   # select
        b.step(env)   # make plan -> session opens
        self.assertIsNotNone(b.execution_monitor.session)
        return env, b

    def test_tool_call_budget_stops_with_named_budget(self):
        env, b = self._planned_brain()
        b.execution_monitor.session.budgets["max_tool_calls"] = 1
        self.assertEqual(b.run(env, 60), "budget_exceeded")
        self.assertEqual(b.last_stop_reason, "budget_exceeded")
        session = b.execution_monitor.session
        self.assertEqual(session.status, "budget_exceeded")
        self.assertEqual(session.stop_reason, "max_tool_calls")
        self.assertEqual(session.tool_calls, 1)
        exceeded = _events(b, "execution_budget_exceeded")
        self.assertTrue(exceeded)
        self.assertIn("max_tool_calls", exceeded[-1]["detail"])
        self.assertFalse(env.s["assembled"])

    def test_wall_time_budget_stops_with_named_budget(self):
        env, b = self._planned_brain()
        b.execution_monitor.session.budgets["max_wall_s"] = 0.0
        self.assertEqual(b.run(env, 60), "budget_exceeded")
        self.assertEqual(b.last_stop_reason, "budget_exceeded")
        session = b.execution_monitor.session
        self.assertEqual(session.status, "budget_exceeded")
        self.assertEqual(session.stop_reason, "max_wall_s")
        exceeded = _events(b, "execution_budget_exceeded")
        self.assertTrue(exceeded)
        self.assertIn("max_wall_s", exceeded[-1]["detail"])
        # Nothing executed after the budget was set.
        self.assertEqual(b.execution_monitor.ledger.records, [])


# ----------------------------------------------------------------------
# Cancellation (between steps, and during a running isolated tool)
# ----------------------------------------------------------------------
class TestCancellation(unittest.TestCase):
    def test_cancel_between_steps_preserves_state_and_resumes(self):
        env = BenchEnv()
        b = _bench_brain()
        g = b.add_goal("assemble the part", {"assembled": True})
        b.step(env)   # select
        b.step(env)   # plan
        b.step(env)   # execute go_bench
        task_id = b.execution_monitor.session.task_id
        result = b.cancel_execution("user_cancelled")
        self.assertTrue(result["cancelled"])
        self.assertIsNone(b.goals.active())
        self.assertIsNotNone(b.goals.suspended(g.id))
        session = b.execution_monitor.session
        self.assertEqual(session.status, "cancelled")
        self.assertEqual(session.task_id, task_id)
        self.assertTrue(any(e["kind"] == "execution_cancelled"
                            for e in b.events))
        # Exactly one outcome per started step; the world is consistent.
        ledger = b.execution_monitor.ledger.records
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["action"], "go_bench")
        self.assertEqual(ledger[0]["outcome"], "ok")
        self.assertEqual(env.s["at"], "bench")
        # Resume continues the same task to completion.
        self.assertTrue(b.resume_goal(g.id))
        self.assertEqual(b.execution_monitor.session.task_id, task_id)
        self.assertEqual(b.execution_monitor.session.status, "running")
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["assembled"])
        self.assertEqual(b.execution_monitor.session.status, "done")
        self.assertEqual(len(b.execution_monitor.ledger.records), 6)

    def test_cancel_during_running_tool_kills_it_and_resumes(self):
        env = BenchEnv()
        b = Brain([
            Action("go_bench", {"at": "store"}, {"at": "bench"},
                   args={"_do": "go_bench"}),
            Action("take_part", {"at": "bench", "has_part": False},
                   {"has_part": True}, args={"_do": "take_part"}),
            Action("slow_part", {"has_part": True}, {"slow_done": True},
                   tool="slow", store_as="slow_done"),
            Action("assemble", {"slow_done": True}, {"assembled": True},
                   args={"_do": "assemble"}),
        ], granted={"compute", "environment"})
        b.tools.register(ToolSpec(name="slow", args=[], return_type="bool",
                                  timeout=5.0, required_grant="compute",
                                  version=1), _slow)
        g = b.add_goal("assemble the part", {"assembled": True})
        outcome: dict = {}

        def _run():
            outcome["result"] = b.run(env, 60)

        # Set the marker path BEFORE the thread starts: the slow tool
        # writes its start marker (visible across the fork) before
        # sleeping, so the test can cancel exactly while it runs.
        marker = os.path.join(tempfile.mkdtemp(), "slow_started.txt")
        _SLOW_MARKER["path"] = marker
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not os.path.exists(marker) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(os.path.exists(marker))
        _SLOW_MARKER["path"] = None
        result = b.cancel_execution("user_cancelled")
        self.assertTrue(result["cancelled"])
        # The isolated child was observed dead (Worker 1 kill).
        self.assertIs(result["killed"], True)
        thread.join(timeout=10.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome.get("result"), "interrupted")
        session = b.execution_monitor.session
        self.assertEqual(session.status, "cancelled")
        ledger = b.execution_monitor.ledger.records
        slow = [r for r in ledger if r["action"] == "slow_part"]
        self.assertEqual(len(slow), 1)   # exactly one outcome per started step
        self.assertEqual(slow[0]["outcome"], "cancelled")
        # No half-applied action: the killed tool never wrote its result.
        self.assertIsNot(b.world.get("slow_done"), True)
        self.assertFalse(env.s["assembled"])
        # Resume completes the task (the slow tool runs to completion).
        self.assertTrue(b.resume_goal(g.id))
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["assembled"])
        self.assertEqual(
            len([r for r in b.execution_monitor.ledger.records
                 if r["action"] == "slow_part"]), 2)


# ----------------------------------------------------------------------
# Interruption mid-execution + resume (no duplicates)
# ----------------------------------------------------------------------
class TestInterruptionMidExecution(unittest.TestCase):
    def test_interrupt_mid_task_then_resume_without_duplicates(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        b.step(env)   # plan
        b.step(env)   # execute go_lab_from_hall
        self.assertIsNone(b.goals.suspended(g.id))
        b.perceive_text("FIRE evacuate now", urgency=0.95)
        b.step(env)   # suspends the goal
        self.assertIsNone(b.goals.active())
        session = b.execution_monitor.session
        self.assertEqual(session.status, "interrupted")
        ledger = b.execution_monitor.ledger.records
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["action"], "go_lab_from_hall")
        self.assertEqual(ledger[0]["outcome"], "ok")
        self.assertTrue(b.resume_goal(g.id))
        self.assertEqual(b.execution_monitor.session.status, "running")
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(env.s["has_treasure"])
        # The interrupted step was not re-executed on resume.
        self.assertEqual(
            len([r for r in b.execution_monitor.ledger.records
                 if r["action"] == "go_lab_from_hall"]), 1)


# ----------------------------------------------------------------------
# Action verification + §4.2 prioritized-pruning fix
# ----------------------------------------------------------------------
class TestVerificationAndPruningFix(unittest.TestCase):
    def test_verify_action_outcome_detects_mismatch(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        take_key = next(a for a in b.planner.actions if a.name == "take_key")
        verified, mismatches = b._verify_action_outcome(
            take_key, {"at": "lab", "has_key": False},
            {"at": "lab", "has_key": True})
        self.assertTrue(verified)
        self.assertEqual(mismatches, [])
        verified, mismatches = b._verify_action_outcome(
            take_key, {"at": "lab", "has_key": False},
            {"at": "lab", "has_key": False})
        self.assertFalse(verified)
        self.assertEqual(mismatches, ["has_key"])

    def test_pattern_evidence_episodes_gain_usefulness(self):
        # End to end: after the treasure run the unreliable-opener
        # pattern is learned from the pick_lock failure episode, and
        # that episode's usefulness is +1 over the +1 every recorded
        # episode already earns (Phase 6 §11.5: 0/3 retained).
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("get the treasure", {"has_treasure": True})
        self.assertEqual(b.run(env, 60), "done")
        self.assertTrue(any(p.kind == "unreliable_toolless_opener"
                            for p in b.patterns.patterns()))
        self.assertGreaterEqual(
            b.prioritizer.uses("pick_lock", {"at": "lab"}), 2)

    def test_marked_evidence_survives_prioritized_pruning(self):
        # Mechanism check on the Phase 6 §11.5 construction shape:
        # old evidence episodes (cycle 12) vs recent filler (cycle 20,
        # usefulness 0). Unmarked, the evidence loses (0.38 < 0.50);
        # marked +1, it wins (0.55 > 0.50) and is retained.
        from types import SimpleNamespace

        from ab.core.learning import Episode

        b = Brain([], granted={"compute"})
        for cycle, room in ((12, "r1"), (13, "r2"), (14, "r3")):
            b.learner.record(Episode(cycle, "cut_part", {"room": room}, True))
        pattern = SimpleNamespace(
            id="test-pattern",
            evidence=[{"action": "cut_part", "outcome": "success"}])
        b._mark_pattern_evidence_useful([pattern])
        for room in ("r1", "r2", "r3"):
            self.assertEqual(b.prioritizer.uses("cut_part", {"room": room}), 2)
        episodes = [{"action": "cut_part", "situation": {"room": room},
                     "cycle": cycle}
                    for cycle, room in ((12, "r1"), (13, "r2"), (14, "r3"))]
        filler = [{"action": f"filler_{i}", "situation": {"slot": i},
                   "cycle": 20} for i in range(5)]
        kept = b.prioritizer.prune(episodes + filler, cap=3,
                                   current_cycle=20)
        kept_actions = sorted(e["action"] for e in kept)
        self.assertEqual(kept_actions, ["cut_part"] * 3)


# ----------------------------------------------------------------------
# Execution units: session / monitor / ledger in isolation
# ----------------------------------------------------------------------
class TestExecutionUnits(unittest.TestCase):
    def test_session_task_id_stable_and_budgets(self):
        from ab.execution import ExecutionSession

        plan = [Action("a", {}, {"x": True}), Action("b", {"x": True}, {"y": True})]
        s1 = ExecutionSession(7, plan, budgets={"max_cycles": 10},
                              started_cycle=3, started_wall=100.0)
        s2 = ExecutionSession(7, plan, budgets={"max_cycles": 10},
                              started_cycle=3, started_wall=100.0)
        self.assertEqual(s1.task_id, s2.task_id)
        s3 = ExecutionSession(8, plan, budgets={"max_cycles": 10},
                              started_cycle=3, started_wall=100.0)
        self.assertNotEqual(s1.task_id, s3.task_id)
        self.assertIsNone(s1.check_budget(5, 101.0))
        self.assertEqual(s1.check_budget(14, 101.0), "max_cycles")
        s1.record_step("a", "ok", {"tool_calls": 1})
        self.assertEqual(s1.tool_calls, 1)
        s1.budgets["max_tool_calls"] = 1
        self.assertEqual(s1.check_budget(5, 101.0), "max_tool_calls")
        s1.request_cancel("test")
        self.assertTrue(s1.cancel_requested)
        self.assertEqual(s1.cancel_reason, "test")
        # JSON round-trip preserves the task id and the counts.
        restored = ExecutionSession.from_json(s1.to_json())
        self.assertEqual(restored.task_id, s1.task_id)
        self.assertEqual(restored.tool_calls, 1)
        self.assertEqual(restored.steps_executed, 1)

    def test_ledger_bounds_at_1000_oldest_first(self):
        from ab.execution import MAX_LEDGER_RECORDS, ExecutionLedger

        ledger = ExecutionLedger()
        for i in range(MAX_LEDGER_RECORDS + 50):
            ledger.record(cycle=i, action=f"a{i}", outcome="ok")
        self.assertEqual(len(ledger), MAX_LEDGER_RECORDS)
        self.assertEqual(ledger.records[0]["action"], "a50")
        self.assertEqual(ledger.records[-1]["action"],
                         f"a{MAX_LEDGER_RECORDS + 49}")

    def test_ledger_redacts_secrets(self):
        from ab.execution import ExecutionLedger

        ledger = ExecutionLedger()
        rec = ledger.record(cycle=1, action="vault",
                            args={"api_key": "sk-test-SECRET-VALUE-12345"},
                            outcome="ok")
        self.assertEqual(rec["args"], {"api_key": "[REDACTED]"})


if __name__ == "__main__":
    unittest.main()
