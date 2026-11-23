"""Phase 9 §4.3 / §4.5 / §4.6 integration tests (Worker 2).

- §4.3 ``Brain.imagine`` 1-scenario finding: the Phase 8 experiment-2
  setup is reproduced and the finding is pinned with numbers — the
  pair ban prunes the second candidate in ``Planner.plan_all``
  (which the generator calls), not in the generator's own fallbacks,
  so the generator never sees it. It is a REAL limitation, and the
  fix surfaces the banned route as a scenario flagged ``banned``
  (opt-in via ``include_banned=True``; the default keeps the
  previous contract, which the Phase 8 tests pin at 1 scenario
  post-learning). A banned scenario is never the best.
- §4.5 revision-table persistence (decision P9-2): v6 save round-trip
  plus v5-migration (old save loads with an empty revision table).
- §4.6 world-model public event API: the bus routes env events
  through ``WorldModel.record_event`` (no direct ``_events`` append).

Stdlib only, deterministic.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import room_world_actions
from ab.eval.phase8_eval import (
    KEY_ROUTE,
    PICK_LOCK_ROUTE,
    TREASURE_START,
    TREASURE_TARGET,
    _run_treasure,
    _set_world,
)
from ab.imagination.scenarios import ScenarioGenerator
from ab.world.model import WorldEvent, WorldModel


class TestImagineBannedRoutes(unittest.TestCase):
    """§4.3 — reproduce the Phase 8 exp-2 setup and pin the finding."""

    def _learned_brain(self) -> Brain:
        """A brain that ran the treasure task once (completed in 10
        cycles, 1 pick_lock attempt) and is reset to the treasure
        start state for imagination — exactly the exp-2 setup."""
        brain = Brain(room_world_actions(), granted={"compute", "environment"})
        run = _run_treasure(brain)
        self.assertTrue(run["completed"])
        self.assertEqual(run["cycles"], 10)
        self.assertEqual(run["pick_lock_attempts"], 1)
        _set_world(brain, TREASURE_START)
        return brain

    def test_pair_ban_prunes_second_candidate_in_plan_all(self):
        brain = self._learned_brain()
        # The learned (pick_lock, situation) pair ban — one pair.
        # Phase 10.5 (§9 pin update): the ban key gained the
        # canonical-args identity as its middle element; pick_lock
        # runs with no semantic args, so it is "" and the intent —
        # exactly one pair ban for pick_lock at the lab — is
        # unchanged.
        self.assertEqual(
            sorted(brain.planner.banned_pairs),
            [("pick_lock", "", (("at", "lab"),))])
        # Strict plan_all sees ONE route (the key route); the
        # relaxed call sees both. The pruning site is plan_all's
        # banned-pair filter — the generator's own fallbacks (ban
        # first move, permutations) all go through strict plan_all,
        # so they cannot recover the route either.
        strict = brain.planner.plan_all(
            dict(TREASURE_START), dict(TREASURE_TARGET), limit=4)
        self.assertEqual(
            [[a.name for a in p.steps] for p in strict], [KEY_ROUTE])
        relaxed = brain.planner.plan_all(
            dict(TREASURE_START), dict(TREASURE_TARGET), limit=4,
            allow_banned_pairs=True)
        self.assertEqual(
            [[a.name for a in p.steps] for p in relaxed],
            [KEY_ROUTE, PICK_LOCK_ROUTE])

    def test_imagine_default_returns_one_scenario_post_learning(self):
        brain = self._learned_brain()
        result = brain.imagine(dict(TREASURE_TARGET))
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["steps"], KEY_ROUTE)
        self.assertFalse(result["scenarios"][0]["banned"])
        self.assertEqual(result["best"]["steps"], KEY_ROUTE)
        self.assertFalse(result["include_banned"])

    def test_imagine_include_banned_surfaces_pick_lock_flagged(self):
        brain = self._learned_brain()
        result = brain.imagine(dict(TREASURE_TARGET), include_banned=True)
        self.assertEqual(len(result["scenarios"]), 2)
        routes = {tuple(s["steps"]): s for s in result["scenarios"]}
        self.assertEqual(set(routes), {tuple(KEY_ROUTE), tuple(PICK_LOCK_ROUTE)})
        key_scen, lock_scen = routes[tuple(KEY_ROUTE)], routes[tuple(PICK_LOCK_ROUTE)]
        self.assertFalse(key_scen["banned"])
        self.assertTrue(lock_scen["banned"])
        self.assertTrue(lock_scen["met_goal"])
        self.assertTrue(lock_scen["simulated"])
        # The banned route simulates WORSE (the learned failure is
        # in its predicted success) and is never the best.
        self.assertLess(lock_scen["predicted_success"],
                        key_scen["predicted_success"])
        self.assertEqual(result["best"]["steps"], KEY_ROUTE)
        self.assertFalse(result["best"]["banned"])
        banned_entries = [e for e in result["ranking"]
                          if e["scenario"]["banned"]]
        self.assertEqual(len(banned_entries), 1)
        self.assertIn("banned", banned_entries[0]["reason"])
        self.assertTrue(result["include_banned"])

    def test_generator_include_banned_flag(self):
        brain = self._learned_brain()
        sim = brain._get_simulator()
        default = ScenarioGenerator(brain.planner, sim).generate(
            dict(TREASURE_TARGET), dict(TREASURE_START))
        self.assertEqual(len(default), 1)
        self.assertFalse(default[0].banned)
        with_banned = ScenarioGenerator(
            brain.planner, sim, include_banned=True).generate(
                dict(TREASURE_TARGET), dict(TREASURE_START))
        self.assertEqual(len(with_banned), 2)
        self.assertEqual([s.banned for s in with_banned], [False, True])
        self.assertEqual(with_banned[1].steps, PICK_LOCK_ROUTE)
        self.assertEqual(with_banned[1].to_dict()["banned"], True)


class TestRevisionPersistence(unittest.TestCase):
    """§4.5 (decision P9-2): revision table in Brain.save/load, v6."""

    def _adjusted_brain(self) -> Brain:
        brain = Brain(room_world_actions(), granted={"compute", "environment"})
        brain.simulation_revision.observe(
            "pick_lock", "door_open", True, False, cycle=1)
        brain.simulation_revision.observe(
            "pick_lock", "door_open", True, False, cycle=2)
        return brain

    def test_revision_table_round_trips_in_v6_save(self):
        brain = self._adjusted_brain()
        folder = tempfile.mkdtemp(prefix="ab-phase9-rev-")
        brain.save(folder)
        with open(os.path.join(folder, "meta.json")) as fh:
            meta = json.load(fh)
        # Phase 11 pin update (§9 precedent): persistence v7.
        self.assertEqual(meta["version"], 7)
        self.assertIn("revision.json", meta["files"])
        self.assertEqual(meta["files"]["revision.json"], 1)
        with open(os.path.join(folder, "revision.json")) as fh:
            saved = json.load(fh)
        self.assertEqual(
            saved["adjusted_success"], [["pick_lock", "door_open", 0.25]])
        self.assertEqual(len(saved["history"]), 2)

        loaded = Brain(room_world_actions(), granted={"compute", "environment"})
        loaded.load(folder)
        self.assertEqual(loaded.load_warnings, [])
        self.assertEqual(
            loaded.simulation_revision.adjusted_success,
            {("pick_lock", "door_open"): 0.25})
        self.assertEqual(len(loaded.simulation_revision.history), 2)
        self.assertEqual(
            loaded.simulation_revision.history[0]["adjusted"], 0.5)
        self.assertEqual(
            loaded.simulation_revision.history[1]["adjusted"], 0.25)
        # The restored table drives predictions: the simulator sees it.
        self.assertEqual(
            loaded.simulation_revision.predicted_success(
                "pick_lock", "door_open"), 0.25)

    def test_v5_save_loads_with_empty_revision_table(self):
        brain = self._adjusted_brain()
        brain.teach_fact("mould", "is_a", "tool")
        folder = tempfile.mkdtemp(prefix="ab-phase9-rev-v5-")
        brain.save(folder)
        # Downgrade the save to the v5 shape: version 5, no
        # revision.json in the manifest or on disk.
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["version"] = 5
        del meta["files"]["revision.json"]
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        os.remove(os.path.join(folder, "revision.json"))

        loaded = Brain(room_world_actions(), granted={"compute", "environment"})
        loaded.load(folder)
        self.assertEqual(loaded.simulation_revision.adjusted_success, {})
        self.assertEqual(loaded.simulation_revision.history, [])
        self.assertTrue(any("revision.json" in w and "pre-v6" in w
                            for w in loaded.load_warnings))
        # Everything else in the save still loads.
        self.assertTrue(loaded.ask("mould", "is_a").known)


class TestPublicEventApi(unittest.TestCase):
    """§4.6: WorldModel.record_event + bus routing."""

    def test_record_event_appends_and_returns_the_event(self):
        model = WorldModel()
        event = model.record_event(
            kind="door_slam", entity="door", key="door_open",
            old_value=True, new_value=False, cause="observation", cycle=3)
        self.assertIsInstance(event, WorldEvent)
        self.assertEqual(
            (event.kind, event.entity, event.key, event.new_value,
             event.cause, event.cycle),
            ("door_slam", "door", "door_open", False, "observation", 3))
        self.assertIn(event, model.events())
        self.assertEqual(len(model.events()), 1)

    def test_bus_env_event_arrives_via_public_api(self):
        brain = Brain(room_world_actions(), granted={"compute", "environment"})
        percept = brain.perceive("event", {
            "kind": "door_slam", "entity": "door", "key": "door_open",
            "old_value": True, "new_value": False, "cause": "observation"})
        self.assertEqual(percept.status, "admitted")
        self.assertIsInstance(percept.content, WorldEvent)
        events = brain.world_model.events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "door_slam")
        self.assertEqual(events[0].entity, "door")
        self.assertEqual(events[0].key, "door_open")
        self.assertEqual(events[0].new_value, False)
        # The env event also reaches WM as a Thought (P8-1 behaviour
        # unchanged).
        self.assertTrue(any("door_slam" in str(t.content)
                            for t in brain._incoming))

    def test_bus_no_longer_appends_to_private_events(self):
        import ab.perception.bus as bus_mod
        path = bus_mod.__file__
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn("_events.append", source)
        self.assertIn("record_event", source)


if __name__ == "__main__":
    unittest.main()
