"""Phase 7 coordinator integration tests.

These are the integration-level proofs for the fixes landed by the
coordinator during the Phase 7 gate (not by the workers): they pin
the behaviour of the recovery / cancellation / redaction paths as
one system, where a unit test of a single mechanism would not see
the interaction.

  1. cancel-during-tool stops without reopening the session, and the
     same goal resumes and completes afterwards (ADR P7-9).
  2. a flaky tool recovers and completes, and its garbage error text
     never reaches learner.json or any saved file (fixes B1/B2 and
     the (3a)-(3c) recovery-path defects).
  3. short and token-shaped secrets fed through an executed tool are
     absent from the tool audit, events, ledger and every saved file
     (ADR P7-10).
  4. the brain's tool path honours the ToolSpec timeout (not the
     registry's 5 s default): a 2 s sleeper with a 0.5 s spec
     timeout is killed at ~0.5 s (fix A3).
  5. the retrieved-failure count rule: an alternative's first step is
     rejected only after >= 2 retrieved failures in the same
     situation (fix B1 / ADR P7-9).

Fake secrets below are invented for these tests only.
"""

import json
import os
import tempfile
import threading
import time
import unittest

from ab.body.environment import Environment
from ab.body.tools import ToolSpec
from ab.brain import Brain
from ab.core.learning import Episode
from ab.eval.execution_eval import (
    MAX_CYCLES,
    WORKSHOP_W1,
    WorkshopWorld,
    _register_workshop_tool,
    workshop_actions,
)
from ab.planning.planner import Action

SHORT_SECRET = "INT_SHORT_SECRET_1"
TOKEN_SECRET = "INT_TOKEN_FAKE_9Z8X7W6V5U4T3S2R1Q0"
FLAKY_MARKER = "INT_FLAKY_GARBAGE_4KQ"


# ---- cancel / resume harness (mirrors test_phase7_execution.py) ----

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
        self.log.append(name)
        return ok, info


_SLOW_MARKER: dict = {"path": None}
_SLOW_FAST: dict = {"fast": False}


def _slow() -> bool:
    if _SLOW_FAST["fast"]:
        return True
    if _SLOW_MARKER["path"] is not None:
        with open(_SLOW_MARKER["path"], "w", encoding="utf-8") as fh:
            fh.write("started")
    time.sleep(30)
    return True


def _sleep_two() -> bool:
    time.sleep(2)
    return True


def _bench_brain():
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
    return b


def _events_of(brain, kind):
    return [e for e in brain.events if e.get("kind") == kind]


class TestCancelStopsWithoutReopenThenResumes(unittest.TestCase):
    def test_cancel_during_tool_then_resume_completes(self):
        env = BenchEnv()
        b = _bench_brain()
        goal = b.add_goal("assemble the part", {"assembled": True})
        outcome: dict = {}

        def _run():
            outcome["result"] = b.run(env, 60)

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
        self.assertIs(result["killed"], True)
        # "observed" is a detail string, not a bool: the kill is
        # only reported once the process is observed dead.
        self.assertIn("observed dead", result["observed"])
        thread.join(timeout=10.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcome.get("result"), "interrupted")
        # ADR P7-9: the cancelled session stays cancelled -- it must
        # NOT be silently re-opened as running by the next planning
        # pass, and the goal must not be retried behind the user's
        # back. The ledger holds exactly one outcome per started step.
        self.assertEqual(b.execution_monitor.session.status, "cancelled")
        slow = [r for r in b.execution_monitor.ledger.records
                if r["action"] == "slow_part"]
        self.assertEqual(len(slow), 1)
        self.assertEqual(slow[0]["outcome"], "cancelled")
        self.assertEqual(
            [e for e in b.events if e.get("kind") == "execution_resumed"],
            [])
        # Resume is explicit: resume_goal re-opens the cancelled
        # session (same task id) and the task completes. The slow
        # tool still takes 30 s, so flip it to fast mode first (the
        # forked child inherits the flag) -- the point here is the
        # resume path, not the sleeper.
        _SLOW_FAST["fast"] = True
        try:
            self.assertTrue(b.resume_goal(goal.id))
            self.assertEqual(b.run(env, 60), "done")
        finally:
            _SLOW_FAST["fast"] = False
        self.assertEqual(b.goals.all()[-1].status, "done")


class TestFlakyToolRecoveryIntegration(unittest.TestCase):
    def test_flaky_cut_recovers_and_garbage_stays_out_of_saved_files(self):
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = Brain(workshop_actions(spec),
                      granted={"compute", "environment"})
        with tempfile.TemporaryDirectory() as folder:
            counter_path = os.path.join(folder, "flaky_count.txt")
            _register_workshop_tool(brain, spec, env, flaky_first=True,
                                    flaky_marker=FLAKY_MARKER,
                                    flaky_counter_path=counter_path)
            brain.add_goal(f"build the {spec['treasure']}",
                           {spec["treasure_key"]: True})
            stop = brain.run(env, MAX_CYCLES)
            self.assertEqual(stop, "done")
            self.assertEqual(brain.goals.all()[-1].status, "done")
            # The failure is recorded where failure records belong.
            audit_failures = [
                a for a in brain.tools.audit
                if not a["ok"] and FLAKY_MARKER in str(a.get("error", ""))]
            self.assertTrue(audit_failures)
            # ... and the garbage never reaches shared state or saved
            # files (events.json is the audit trail, excluded here).
            self.assertNotIn(FLAKY_MARKER, json.dumps(brain.world, default=str))
            brain.save(folder)
            for fname in sorted(os.listdir(folder)):
                if fname in ("events.json", "flaky_count.txt"):
                    continue
                with open(os.path.join(folder, fname), encoding="utf-8") as fh:
                    self.assertNotIn(FLAKY_MARKER, fh.read(), msg=fname)
            # The learner file in particular carries no raw tool text.
            with open(os.path.join(folder, "learner.json"),
                      encoding="utf-8") as fh:
                learner_blob = fh.read()
            self.assertNotIn(FLAKY_MARKER, learner_blob)
            # The failed tool's episode surprises are the typed
            # triples, never the raw error text (Episode.surprises
            # is a list of tuples).
            surprises = [str(s) for ep in brain.learner.episodes
                         for s in ep.surprises]
            self.assertTrue(any("tool" in s and "failure" in s
                                for s in surprises))
            self.assertFalse(any(FLAKY_MARKER in s for s in surprises))


class TestSecretRedactionThroughExecutedTool(unittest.TestCase):
    def _run_stamp(self, value):
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = Brain(workshop_actions(spec),
                      granted={"compute", "environment"})
        _register_workshop_tool(brain, spec, env)

        def _stamper(code):
            return True

        brain.tools.register(ToolSpec(
            name="stamper",
            args=[{"name": "code", "type": "str", "required": True,
                   "secret": True}],
            return_type="str", timeout=5.0, required_grant="compute",
            description="Stamp with a secret code"), _stamper)
        brain.planner.add_action(Action(
            "stamp", {}, {"stamped": True}, tool="stamper",
            args={"code": value}, store_as="stamped"))
        brain.add_goal("stamp once", {"stamped": True})
        brain.run(env, 20)
        return brain, value

    def test_short_and_token_secrets_absent_everywhere(self):
        for value in (SHORT_SECRET, TOKEN_SECRET):
            brain, secret = self._run_stamp(value)
            self.assertEqual(brain.goals.all()[-1].status, "done")
            self.assertFalse(any(secret in json.dumps(a, default=str)
                                 for a in brain.tools.audit))
            self.assertFalse(any(secret in json.dumps(e, default=str)
                                 for e in brain.events))
            self.assertFalse(any(
                secret in json.dumps(r, default=str)
                for r in brain.execution_monitor.ledger.records))
            with tempfile.TemporaryDirectory() as folder:
                brain.save(folder)
                for fname in sorted(os.listdir(folder)):
                    with open(os.path.join(folder, fname),
                              encoding="utf-8") as fh:
                        self.assertNotIn(secret, fh.read(), msg=fname)


class TestSpecTimeoutHonouredOnBrainPath(unittest.TestCase):
    def test_tool_spec_timeout_kills_sleeper_not_registry_default(self):
        env = BenchEnv()
        b = Brain([
            Action("go_bench", {"at": "store"}, {"at": "bench"},
                   args={"_do": "go_bench"}),
            Action("slow_step", {"at": "bench"}, {"slow_done": True},
                   tool="sleeper", store_as="slow_done"),
        ], granted={"compute", "environment"})
        # Spec timeout 0.5 s; the tool sleeps 2 s; the registry
        # default is 5 s. If the brain path used either of the
        # larger budgets the run would take far longer.
        b.tools.register(ToolSpec(name="sleeper", args=[],
                                  return_type="bool", timeout=0.5,
                                  required_grant="compute", version=1),
                         _sleep_two)
        b.add_goal("finish the slow step", {"slow_done": True})
        started = time.monotonic()
        b.run(env, 40)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 10.0)
        timed_out = [a for a in b.tools.audit if a.get("timed_out")]
        self.assertTrue(timed_out)
        self.assertTrue(all(a.get("killed") for a in timed_out))
        outcomes = [r["outcome"]
                    for r in b.execution_monitor.ledger.records]
        self.assertIn("timed_out", outcomes)
        # The brain survives the kill and can still plan and act: a
        # second goal completes. (run() itself reports "impossible"
        # here because the FIRST goal failed -- its stop reason
        # prefers any failed goal over all-done -- so the assertion
        # is on the second goal's status, not run()'s return.)
        b.add_goal("go to the bench", {"at": "bench"})
        b.run(env, 40)
        self.assertEqual(b.goals.all()[-1].status, "done")


class TestRetrievedFailureCountRule(unittest.TestCase):
    def _gate_brain(self):
        b = Brain([
            Action("open_gate", {"at": "hall"}, {"gate_open": True},
                   cost=0.5, args={"_do": "open_gate"}),
            Action("take_key", {"at": "hall"}, {"has_key": True},
                   cost=1.0, args={"_do": "take_key"}),
            Action("use_key", {"at": "hall", "has_key": True},
                   {"gate_open": True}, cost=1.0,
                   args={"_do": "use_key"}),
            # §9 fixture change (Phase 10.5 coordinator): a third
            # route to the gate. The two-failure test records two
            # take_key failures, so after the alternative is rejected
            # the replan's only remaining route (take_key) is a loop
            # plan under the P10.5-7 rule — every plan retries a
            # twice-failed step — and _do_make_plan fails the goal
            # instead of returning a fresh plan. That is the loop
            # rule working as designed, not a failure of the
            # retrieved-failure count rule this test pins. The third
            # route (break_gate, never failed) gives the replan a
            # non-loop plan, so the test again isolates the count
            # rule: rejection fires, one event, no alternative adopted.
            # The one-failure test is unaffected (its alternative is
            # adopted directly, no replan).
            Action("break_gate", {"at": "hall"}, {"gate_open": True},
                   cost=2.0, args={"_do": "break_gate"}),
        ], granted={"compute", "environment"})
        b.world = {"at": "hall", "has_key": False, "gate_open": False}
        goal = b.add_goal("open the gate", {"gate_open": True})
        b.goals.activate(goal.id)
        return b

    def _set_alternative(self, b, name="take_key"):
        action = next(a for a in b.planner.actions if a.name == name)
        b.alternative_plan = [action]
        b.alternative_plan_cost = action.cost
        return action

    def test_one_retrieved_failure_does_not_reject_alternative(self):
        b = self._gate_brain()
        failed = next(a for a in b.planner.actions if a.name == "open_gate")
        self._set_alternative(b)
        b.learner.record(Episode(1, "take_key", {"at": "hall"}, False))
        self.assertTrue(b._try_alternative(failed, {"at": "hall"}))
        self.assertEqual(len(_events_of(b, "alternative_rejected")), 0)
        self.assertEqual(len(_events_of(b, "alternative_plan")), 1)
        self.assertEqual(b.plan[0].name, "take_key")

    def test_two_retrieved_failures_reject_alternative(self):
        b = self._gate_brain()
        failed = next(a for a in b.planner.actions if a.name == "open_gate")
        self._set_alternative(b)
        b.learner.record(Episode(1, "take_key", {"at": "hall"}, False))
        b.learner.record(Episode(2, "take_key", {"at": "hall"}, False))
        self.assertTrue(b._try_alternative(failed, {"at": "hall"}))
        rejected = _events_of(b, "alternative_rejected")
        self.assertEqual(len(rejected), 1)
        self.assertIn("2 times", rejected[0]["detail"])
        self.assertEqual(len(_events_of(b, "alternative_plan")), 0)


if __name__ == "__main__":
    unittest.main()
