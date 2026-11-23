"""Phase 7 tests: the execution evaluation (design SS3 + SS4).

Fast and deterministic: the full execution evaluation is run once
per module in FAST mode (lazy singleton, as in test_phase6_eval.py --
fast mode runs the multi-tool experiment plus the carried-item,
calibration and pruning measurements, ~2-3 s) and the tests assert
the MEASUREMENT. The slow experiments (sleeper, cancellation,
budgets) live only in ab/eval/execution_eval.py run by the harness
or standalone -- never here. Nothing in this file invokes the
harness with default paths (the fast record is built in-process
with record_path=None), so running the suite never overwrites a
canonical docs/ record. Where a criterion is not met, the record
says so -- these tests do not hide failed criteria behind weaker
assertions, and they do not assert a criterion the measurement
shows failing (see the test comments).
"""
import os
import tempfile
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ab.body.tools import ToolRegistry, ToolSpec  # noqa: E402
from ab.body.web import FetchError, WebFetcher  # noqa: E402
from ab.brain import Brain  # noqa: E402
from ab.eval.execution_eval import (  # noqa: E402
    MAX_CYCLES,
    WORKSHOP_W1,
    WorkshopWorld,
    _full_factory,
    _register_workshop_tool,
    run_execution_evaluation,
    workshop_actions,
)
from ab.planning.planner import Action  # noqa: E402

_RECORD = None


def _record():
    global _RECORD
    if _RECORD is None:
        _RECORD = run_execution_evaluation(fast=True, record_path=None)
    return _RECORD


class TestFastExecutionRecord(unittest.TestCase):
    def test_fast_record_has_required_sections(self):
        rec = _record()
        self.assertEqual(rec["mode"], "fast")
        for section in ("experiments", "carried_items", "calibration",
                        "pruning"):
            self.assertIn(section, rec, f"missing record section {section}")
        self.assertIn("multi_tool_task", rec["experiments"])
        self.assertIn("failures", rec)

    def test_multi_tool_task_completes_with_tool_and_env_actions(self):
        multi = _record()["experiments"]["multi_tool_task"]
        full = multi["variants_w1"]["full"]
        self.assertTrue(multi["passed"])
        self.assertTrue(full["completed"])
        self.assertGreaterEqual(full["tool_calls"], 3)
        self.assertGreaterEqual(full["env_actions"], 3)
        self.assertTrue(full["mid_task_reset_observed"])

    def test_carried_items_fired_with_counts(self):
        carried = _record()["carried_items"]
        # P-C/P-D transfer: P-D transfers by pattern id (4 applications
        # in W2); P-C forms in both worlds.
        # Phase 10.5 (P10.5-5/8, W2) pin update (Phase 2 §9
        # precedent — same intent, new intended semantics, not
        # weakened): the adopted variant-aware rate now does the
        # steering P-D used to do in W2 — the toolless press
        # carries key-shape rate 1.0, so the continued brain's
        # adopted W2 plans contain no toolless press for P-D to
        # penalise, and P-D applies 0 times there (P-C likewise
        # 0: the W2 cut steps carry no pattern penalty). The
        # patterns themselves still FORM in both worlds (both
        # flags below stay true), and the BEHAVIOURAL transfer
        # the patterns exist to produce is stronger than before
        # (continued 1 failed / 18 cycles vs fresh 4 / 27) — the
        # transfer moved from the pattern channel to the
        # variant-rate channel. The item is therefore NOT FIRED
        # in W2, with the reason recorded, exactly as the
        # retirement item below already expects NOT FIRED with a
        # reason. If a future change makes a toolless press
        # plannable in W2 again, P-D must fire there — the
        # formed flags and the behavioural assertions below are
        # the guardrail.
        pc_pd = carried["pc_pd_transfer"]
        self.assertEqual(pc_pd["status"], "NOT FIRED")
        self.assertEqual(pc_pd["pd_pattern_applied_in_w2"], 0)
        self.assertEqual(pc_pd["pc_pattern_applied_in_w2"], 0)
        self.assertTrue(pc_pd["pc_formed_w1"] and pc_pd["pd_formed_w1"])
        self.assertTrue(pc_pd["pc_formed_w2_continued"]
                        and pc_pd["pd_formed_w2_continued"])
        beh = pc_pd["behavioural_transfer"]
        self.assertLess(beh["w2_continued_failed_attempts"],
                        beh["w2_fresh_failed_attempts"])
        self.assertLess(beh["w2_continued_cycles"],
                        beh["w2_fresh_cycles"])
        # Consolidation fires in W1 (and W2 continued).
        consol = carried["consolidation_firing"]
        self.assertEqual(consol["status"], "FIRED")
        self.assertGreater(consol["consolidated_events_w1"], 0)
        # A repeated mistake fires in W1.
        rep = carried["repeated_mistake_firing"]
        self.assertEqual(rep["status"], "FIRED")
        self.assertGreater(rep["repeated_mistake_events_w1"], 0)
        # The 0.8 concept planner preference fires. Coordinator
        # integration change (Phase 7 gate, documented in the Phase 7
        # report): it used to fire in W1 (1 firing) and never in W2,
        # because _promote_candidate marked only the FIRST same-named
        # 'cut' action's tool -- W1's cutter, never W2's shears. The
        # fix marks every same-named action's tool, so the preference
        # now fires in W2 continued (4 firings, multiplier 0.8 there)
        # and the W1 count is 0 -- a timing artefact of this run's
        # path (promotion lands at cycle 18, the run completes at 25
        # with no replan after it), while the W1 multiplier is still
        # verified 0.8. The assertion is therefore on the total
        # firings and on both multipliers, not on the W1 count alone:
        # the criterion is that the preference fires in a world with
        # the 0.8 multiplier applied, which the numbers show.
        concept = carried["concept_preference_firing"]
        self.assertEqual(concept["status"], "FIRED")
        self.assertEqual(concept["concept_multiplier_on_w1_cut"], 0.8)
        self.assertEqual(concept["concept_multiplier_on_w2_cut"], 0.8)
        self.assertGreater(
            concept["concept_firings_w1"]
            + concept["concept_firings_w2_continued"], 0)

    def test_wrong_promoted_pattern_retirement_not_fired(self):
        # Phase 9 pin update (Phase 2 §9 precedent — same intent, new
        # intended semantics, not weakened): the Brain-level retirement
        # API exists now, so the old reason (no API) is gone; the item
        # is still NOT FIRED in the workshop worlds because only 1
        # contradicting observation exists there and the brain's own
        # threshold is 3. The mechanism itself FIRES in the dedicated
        # contradicting-evidence world (Phase 9 retirement test).
        item = _record()["carried_items"]["wrong_promoted_pattern_retirement"]
        self.assertEqual(item["status"], "NOT FIRED")
        self.assertTrue(item["retire_api_present"])
        # Two genuine contradictions, one per P-D pattern: the W1
        # pattern (requires has_mould) is contradicted by the W2
        # press success with has_cast, and the W2 pattern (requires
        # has_cast) by the W1 press success with has_mould. Neither
        # pattern reaches the threshold of 3 on its own.
        self.assertEqual(len(item["contradictions_fed"]), 2)
        self.assertEqual(item["retired_events"], 0)
        self.assertIn("reason", item)

    def test_calibration_improves_on_workshop_stream(self):
        cal = _record()["calibration"]["overall"]
        # Phase 10.5 (P10.5-5, W2) pin update (Phase 2 §9
        # precedent — same intent, new intended semantics, not
        # weakened): the P10-1 variant-aware pattern application
        # shortened the W2-continued workshop run from 22 to 21
        # cycles, so one fewer prediction was recorded. Now the
        # P10.5-5 stall fix + adopted variant rate change which
        # steps execute (the redundant re-cut is no longer planned;
        # the W2 cheap-press attempts are avoided), so the stream
        # is one prediction shorter again (36 -> 35). The
        # calibration claim itself is unchanged (ece_improved).
        self.assertEqual(cal["n_predictions"], 35)
        self.assertTrue(cal["ece_improved"])
        self.assertLess(cal["ece_calibrated"], cal["ece_default_1.0"])

    def test_pruning_fix_present_and_retains_evidence(self):
        pru = _record()["pruning"]
        self.assertTrue(pru["fix_present"])
        self.assertGreater(pru["pattern_evidence_retained"], 0)


class TestPermissionDenial(unittest.TestCase):
    def test_tool_without_grant_refused_at_plan_time(self):
        spec = WORKSHOP_W1
        ran = []

        def _fetcher():
            ran.append("ran")
            return "fetched"

        brain = Brain(workshop_actions(spec),
                      granted={"compute", "environment"})
        brain.tools.register(ToolSpec(
            name="fetcher", args=[], return_type="str", timeout=5.0,
            required_grant="network",
            description="A network tool the brain has no grant for"),
            _fetcher)
        brain.planner.add_action(Action("fetch", {}, {"fetched": True},
                                        tool="fetcher", store_as="fetched"))
        brain.add_goal("fetch once", {"fetched": True})
        brain.run(WorkshopWorld(spec), 20)
        self.assertEqual(ran, [])
        self.assertEqual(brain.tools.audit, [])
        self.assertTrue(any(e.get("kind") == "plan_tool_denied"
                            for e in brain.events))
        goal = brain.goals.all()[-1]
        self.assertEqual(goal.status, "failed")
        self.assertEqual(goal.fail_reason, "no_permitted_plan")

    def test_env_action_without_grant_never_acts(self):
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = Brain(workshop_actions(spec), granted={"compute"})
        brain.add_goal(f"build the {spec['treasure']}",
                       {spec["treasure_key"]: True})
        brain.run(env, 10)
        self.assertEqual(env.log, [])
        self.assertTrue(any(e.get("kind") == "plan_tool_denied"
                            for e in brain.events))


class TestBypassWeb(unittest.TestCase):
    def test_disallowed_host_refused_direct_and_via_redirect(self):
        fetched_urls = []

        def _transport(url, timeout, max_bytes):
            fetched_urls.append(url)
            if url == "https://good.org/start":
                return (302, "text/html", "",
                        {"location": "http://127.0.0.1/secret"})
            return (200, "text/html", "body", {})

        fetcher = WebFetcher(["good.org"], transport=_transport)
        with self.assertRaises(FetchError):
            fetcher.fetch("http://evil.org/x")
        with self.assertRaises(FetchError):
            fetcher.fetch("https://good.org/start")
        self.assertNotIn("http://127.0.0.1/secret", fetched_urls)


class TestSaveRestartResume(unittest.TestCase):
    def test_save_restart_resume_mid_task(self):
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = _full_factory(workshop_actions(spec),
                              {"compute", "environment"}, env=env, spec=spec)
        brain.add_goal(f"build the {spec['treasure']}",
                       {spec["treasure_key"]: True})
        for _ in range(8):
            brain.step(env)
        goal = brain.goals.active()
        suspended = brain.suspend_active_goal("test_save_restart")
        self.assertIsNotNone(suspended)
        self.assertGreater(suspended.plan_position, 0)
        session_at_save = brain.execution_monitor.session
        with tempfile.TemporaryDirectory() as folder:
            brain.save(folder)
            self.assertIn("execution.json", os.listdir(folder))
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(folder)
            _register_workshop_tool(loaded, spec, env)
            loaded_session = loaded.execution_monitor.session
            self.assertIsNotNone(loaded_session)
            self.assertEqual(loaded_session.task_id,
                             session_at_save.task_id)
            self.assertEqual(loaded_session.status, "interrupted")
            self.assertEqual(len(loaded.learner.episodes),
                             len(brain.learner.episodes))
            self.assertTrue(loaded.resume_goal(goal.id))
            stop = loaded.run(env, MAX_CYCLES + loaded.cycle)
            self.assertEqual(loaded.goals.all()[-1].status, "done")
            self.assertEqual(stop, "done")
            self.assertEqual(loaded.execution_monitor.session.status, "done")


class TestRegistryTimeout(unittest.TestCase):
    def test_thread_path_timeout_lingers(self):
        # Registry-level timeout behaviour: the thread path returns
        # timed_out after ~0.2 s, but the thread is NOT killed -- the
        # marker it writes at ~0.6 s still appears afterwards.
        marker = os.path.join(tempfile.mkdtemp(), "marker.txt")
        registry = ToolRegistry()

        def _sleeper():
            time.sleep(0.6)
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("done")
            return True

        registry.register(ToolSpec(
            name="sleeper", args=[], return_type="bool", timeout=5.0,
            required_grant="compute", description="slow tool"), _sleeper)
        started = time.monotonic()
        res = registry.call("sleeper", {}, {"compute"}, timeout=0.2,
                            isolated=False)
        elapsed = time.monotonic() - started
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertLess(elapsed, 0.6)
        self.assertFalse(os.path.exists(marker))
        deadline = started + 1.0
        while time.monotonic() < deadline and not os.path.exists(marker):
            time.sleep(0.05)
        self.assertTrue(os.path.exists(marker),
                        "thread-path sleeper kept running after timeout")

    def test_isolated_path_timeout_kills(self):
        # The isolated path kills the child on timeout and observes
        # it dead: the marker never appears.
        marker = os.path.join(tempfile.mkdtemp(), "marker.txt")
        registry = ToolRegistry()

        def _sleeper():
            time.sleep(0.6)
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write("done")
            return True

        registry.register(ToolSpec(
            name="sleeper", args=[], return_type="bool", timeout=5.0,
            required_grant="compute", description="slow tool"), _sleeper)
        started = time.monotonic()
        res = registry.call("sleeper", {}, {"compute"}, timeout=0.2,
                            isolated=True)
        elapsed = time.monotonic() - started
        self.assertFalse(res.ok)
        self.assertTrue(res.timed_out)
        self.assertTrue(res.killed)
        self.assertLess(elapsed, 0.6)
        deadline = started + 1.0
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if os.path.exists(marker):
                break
        self.assertFalse(os.path.exists(marker),
                         "isolated sleeper kept running after timeout")


class TestDependencyBlocking(unittest.TestCase):
    def test_blocked_action_refused_with_dependency_blocked(self):
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = _full_factory(workshop_actions(spec),
                              {"compute", "environment"}, env=env, spec=spec)
        brain.add_goal(f"build the {spec['treasure']}",
                       {spec["treasure_key"]: True})
        brain.step(env)  # select goal
        brain.step(env)  # make plan
        go_foundry = next(a for a in brain.planner.actions
                          if a.name == "go_foundry")
        log_before = len(env.log)
        brain._set_plan([go_foundry])
        executed = brain._do_execute_step({}, env)
        self.assertFalse(executed)
        self.assertEqual(len(env.log), log_before)
        dep = [e for e in brain.events
               if e.get("kind") == "dependency_blocked"]
        self.assertGreater(len(dep), 0)
        self.assertIn("door_open", dep[0].get("detail", ""))


if __name__ == "__main__":
    unittest.main()
