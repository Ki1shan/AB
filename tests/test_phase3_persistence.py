"""Phase 3: persistence — one manifest, one path.

Covers the Phase 2 save loading, corrupt/missing optional files,
unknown versions raising, no planner.json / attention.json written,
and the attention seen-set round-tripping through state.
"""
import json
import os
import tempfile
import unittest
from dataclasses import asdict

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions


def _mid_run_brain():
    env = RoomWorld()
    b = Brain(room_world_actions(), granted={"compute", "environment"})
    b.teach_fact("cat", "is_a", "mammal")
    b.add_goal("get the treasure", {"has_treasure": True})
    b.step(env)   # select goal
    b.step(env)   # make plan
    b.step(env)   # execute go_lab_from_hall
    return env, b


def _downgrade_to_phase2_shape(folder, brain):
    """Rewrite a Phase 3 save folder into Phase 2 shape: meta v1 with
    no manifest, goals.json without the plan field, plan.json with the
    legacy steps + history keys, planner.json and attention.json in
    their old formats, decision.json and learner.json removed."""
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "cycle": meta["cycle"]}, fh)
    with open(os.path.join(folder, "goals.json"), encoding="utf-8") as fh:
        goals = json.load(fh)
    for g in goals["goals"]:
        g.pop("plan", None)
    with open(os.path.join(folder, "goals.json"), "w", encoding="utf-8") as fh:
        json.dump(goals, fh)
    with open(os.path.join(folder, "plan.json"), encoding="utf-8") as fh:
        plan = json.load(fh)
    plan["steps"] = [asdict(a) for a in brain.plan]
    plan["history"] = ["legacy world snapshot"]
    with open(os.path.join(folder, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh)
    with open(os.path.join(folder, "planner.json"), "w", encoding="utf-8") as fh:
        json.dump({"stats": {"go_lab_from_hall": [1, 0]},
                   "actions": [asdict(a) for a in room_world_actions()],
                   "banned_pairs": []}, fh)
    with open(os.path.join(folder, "attention.json"), "w", encoding="utf-8") as fh:
        json.dump(brain.attention.to_json(), fh)
    for name in ("decision.json", "learner.json"):
        os.remove(os.path.join(folder, name))


class TestPhase2SaveLoads(unittest.TestCase):
    def test_phase2_save_loads_with_v1_warning(self):
        env, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        _downgrade_to_phase2_shape(d, b)
        b2 = Brain(granted={"compute", "environment"})
        b2.load(d)
        # Missing manifest is treated as v1, with a warning.
        self.assertTrue(any("v1" in w for w in b2.load_warnings))
        # The legacy plan.json steps migrate into the goal record.
        self.assertEqual([a.name for a in b2.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual([a["name"] for a in b2.goals.active().plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        # Legacy planner.json actions migrate in; its stats are ignored
        # (stats are derived from learner episodes, of which there are
        # none in this downgraded save).
        self.assertEqual(b2.planner.stats, {})
        self.assertTrue(any(a.name == "pick_lock" for a in b2.planner.actions))
        # Legacy attention.json seeds the seen-set into state.
        self.assertEqual(b2.attention._seen, b.attention._seen)
        self.assertTrue(any(k.startswith("seen:") for k in b2.state.keys()))
        # Missing decision.json / learner.json default to empty.
        self.assertIsNone(b2.last_decision)
        self.assertEqual(b2.learner.episodes, [])
        # Durable knowledge is intact, and the loaded brain finishes.
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")
        self.assertEqual(b2.run(env, 60), "done")
        self.assertTrue(env.s["has_treasure"])

    def test_v01_folder_still_loads(self):
        b = Brain()
        b.teach_fact("a", "is_a", "b")
        d = tempfile.mkdtemp()
        b.ltm.save(os.path.join(d, "memory.json"))
        b.rules.save_json(os.path.join(d, "rules.json"))
        b2 = Brain()
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertEqual(b2.ask("a", "is_a").best[0], "b")


class TestManifest(unittest.TestCase):
    def test_manifest_lists_every_file_and_no_removed_files_written(self):
        # Phase 7 pin update (following the Phase 2 §9 / Phase 3 §8 /
        # Phase 4 §14.1 / Phase 5 / Phase 6 precedent — same intent, new intended
        # semantics, not weakened): persistence v5 adds execution.json
        # (live session + ledger), so the manifest version is 5 and the
        # file set gains "execution.json". Removed files (planner.json, attention.json)
        # still must not be written.
        # Phase 9 pin update (same §9 precedent): persistence v6 adds
        # revision.json (simulation-revision table), so the manifest
        # version is 6 and the file set gains "revision.json".
        # Phase 11 pin update (same §9 precedent): persistence v7 adds
        # research.json (researcher _seen + source quality), so the
        # manifest version is 7 and the file set gains "research.json".
        _, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        names = set(os.listdir(d))
        self.assertNotIn("planner.json", names)
        self.assertNotIn("attention.json", names)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        # Phase 11 pin update (§9 precedent — same intent, new intended
        # semantics): persistence v7 (research.json added).
        self.assertEqual(meta["version"], 7)
        self.assertEqual(set(meta["files"]), names)
        self.assertEqual(set(meta["files"]), {
            "meta.json", "state.json", "wm.json", "goals.json", "plan.json",
            "memory.json", "rules.json", "learner.json", "decision.json",
            "events.json", "world.json", "learning.json", "execution.json",
            "revision.json", "research.json"})

    def test_missing_manifest_file_skipped_with_warning(self):
        _, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        os.remove(os.path.join(d, "learner.json"))
        b2 = Brain()
        b2.load(d)
        self.assertTrue(any("learner.json" in w for w in b2.load_warnings))
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")
        self.assertEqual(b2.learner.episodes, [])

    def test_corrupt_optional_file_skipped_with_warning(self):
        _, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "wm.json"), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        b2 = Brain()
        b2.load(d)
        self.assertTrue(any("wm.json" in w for w in b2.load_warnings))
        self.assertEqual(b2.ask("cat", "is_a").best[0], "mammal")

    def test_unknown_file_version_raises(self):
        _, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        meta["files"]["wm.json"] = 99
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        with self.assertRaises(ValueError):
            Brain().load(d)

    def test_unknown_meta_version_raises(self):
        _, b = _mid_run_brain()
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"version": 99, "cycle": 0}, fh)
        with self.assertRaises(ValueError):
            Brain().load(d)


class TestAttentionSeenSetInState(unittest.TestCase):
    def test_seen_set_round_trips_through_state_no_attention_file(self):
        _, b = _mid_run_brain()
        self.assertTrue(b.attention._seen)
        seen_keys = [k for k in b.state.keys() if k.startswith("seen:")]
        self.assertTrue(seen_keys)
        for k in seen_keys:
            self.assertEqual(b.state.get(k).source, "attention")
            self.assertEqual(b.state.get(k).entry_type, "observation")
            self.assertEqual(b.state.get(k).status, "active")
        d = tempfile.mkdtemp()
        b.save(d)
        self.assertNotIn("attention.json", os.listdir(d))
        b2 = Brain(granted={"compute", "environment"})
        b2.load(d)
        self.assertEqual(b2.load_warnings, [])
        self.assertEqual(b2.attention._seen, b.attention._seen)
        self.assertEqual(sorted(k for k in b2.state.keys() if k.startswith("seen:")),
                         sorted(seen_keys))


class TestPlanPersistenceViaGoalRecord(unittest.TestCase):
    def test_suspend_save_load_resume_round_trip(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)
        b.step(env)
        b.step(env)
        rec = b.suspend_active_goal("interrupted by test")
        self.assertEqual(rec.plan_position, 1)
        d = tempfile.mkdtemp()
        b.save(d)
        b2 = Brain(room_world_actions(), granted={"compute", "environment"})
        b2.load(d)
        rec2 = b2.goals.restore_suspended(rec.goal_id)
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2.plan_position, 1)
        self.assertEqual([a["name"] for a in rec2.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertTrue(rec2.events)
        self.assertTrue(b2.resume_goal(rec.goal_id))
        self.assertEqual([a.name for a in b2.plan],
                         ["pick_lock", "go_vault", "take_treasure"])
        self.assertEqual(b2.run(env, 60), "done")


class TestPlannerActionsSurviveSaveLoad(unittest.TestCase):
    def test_loaded_brain_without_constructor_actions_can_still_plan(self):
        # Integration regression (Phase 3, 2026-11-21): removing
        # planner.json dropped the planner's ACTIONS on load — a brain
        # loaded without constructor actions had an empty planner and
        # could never plan again (the Phase 1 round-trip test failed
        # with "no plan found" after a save/load). Actions now live in
        # plan.json; stats are still derived, never stored.
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.teach_fact("cat", "is_a", "mammal")
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)
        b.step(env)
        b.step(env)
        d = tempfile.mkdtemp()
        b.save(d)
        self.assertNotIn("planner.json", os.listdir(d))
        b2 = Brain(granted={"compute", "environment"})
        self.assertEqual(b2.planner.actions, [])
        b2.load(d)
        self.assertEqual([a.name for a in b2.planner.actions],
                         [a.name for a in room_world_actions()])
        self.assertIsNotNone(
            b2.planner.plan(b2.world, {"has_treasure": True}))
        b2.run(env, 60)
        self.assertTrue(env.s["has_treasure"])
        self.assertEqual(b2.goals.all()[0].status, "done")


class TestSavedEventLogRedactsSecrets(unittest.TestCase):
    def test_secret_in_event_never_reaches_saved_events_file(self):
        # Integration close-out (Phase 3 §3.5): redaction was tested at
        # the audit-module and ToolRegistry levels, but the acceptance
        # item is the SAVED event log. A secret fed through an event
        # must not appear in events.json on disk.
        secret = "sk-test-SECRET-VALUE-12345"
        b = Brain()
        b._event("act", f"tool returned api_key={secret}")
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "events.json"), encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn(secret, raw)
        self.assertIn("[REDACTED]", raw)


class TestEventLogBounded(unittest.TestCase):
    def test_save_prunes_events_to_1000_oldest_first(self):
        b = Brain()
        b.events = [{"cycle": i, "kind": "x", "detail": f"e{i}"} for i in range(1500)]
        d = tempfile.mkdtemp()
        b.save(d)
        with open(os.path.join(d, "events.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(len(saved), 1000)
        self.assertEqual(saved[0]["detail"], "e500")
        self.assertEqual(saved[-1]["detail"], "e1499")


if __name__ == "__main__":
    unittest.main()
