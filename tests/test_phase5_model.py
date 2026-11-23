"""World model core (Phase 5, Worker 1): entities, relations, spatial
graph, temporal log, constraints, epistemic status, permanence, partial
observability, contradictions, prediction errors, revision history.

All names here are invented for the tests (alpha/beta/gamma objects,
hall/lab/store rooms) — no world specs are imported.
"""
import json
import unittest

from ab.planning.planner import Action, Planner
from ab.reasoning.causal import CausalModel
from ab.world.model import (
    PERMANENCE_DECAY,
    EpistemicStatus,
    WorldModel,
)


def _planner():
    """Small invented action set over hall/lab/store rooms."""
    return Planner(actions=[
        Action("go_hall_lab", {"at": "hall"}, {"at": "lab"}),
        Action("go_lab_store", {"at": "lab"}, {"at": "store"}),
        Action("close_door", {}, {"door_open": False}),
        Action("open_door", {}, {"door_open": True}),
        Action("take_alpha_key", {"at": "lab"}, {"has_alpha_key": True}),
    ])


class TestEntitiesIdentity(unittest.TestCase):
    def test_identity_stable_across_move_change_and_unobserved(self):
        m = WorldModel()
        ent = m.observe("at", "hall", cycle=1)
        self.assertEqual(ent.id, "self")
        key_ent = m.observe("has_alpha_key", True, cycle=1)
        self.assertEqual(key_ent.id, "alpha_key")
        self.assertEqual(key_ent.kind, "object")
        self.assertEqual(key_ent.first_seen_cycle, 1)
        # Same entity object is returned on every later observation.
        again = m.observe("has_alpha_key", True, cycle=2)
        self.assertIs(again, key_ent)
        self.assertIs(m.entity("alpha_key"), key_ent)
        self.assertEqual(key_ent.first_seen_cycle, 1)
        # Move: identity survives, location follows.
        m.observe("at", "lab", cycle=3)
        self.assertIs(m.entity("alpha_key"), key_ent)
        self.assertEqual(m.believed_location("alpha_key")[0], "lab")
        # Change state: the door entity keeps its identity too.
        door = m.observe("door_open", False, cycle=3)
        self.assertEqual(door.id, "door")
        m.observe("door_open", True, cycle=4)
        self.assertIs(m.entity("door"), door)
        # Unobserved for a cycle: still the same entity, still present.
        m.decay(5)
        self.assertIs(m.entity("alpha_key"), key_ent)
        self.assertFalse(m.is_gone("alpha_key"))

    def test_key_shape_mapping(self):
        m = WorldModel()
        m.observe("at", "hall", cycle=1)
        self.assertEqual(m.entity("self").kind, "agent")
        self.assertEqual(m.entity("self").properties["location"].value, "hall")
        m.observe("door_open", True, cycle=1)
        self.assertEqual(m.entity("door").properties["open"].value, True)
        m.observe("has_beta_tool", True, cycle=1)
        self.assertEqual(m.entity("beta_tool").properties["held"].value, True)
        # Generic key: prefix before the first "_" is the entity.
        m.observe("coil_hot", True, cycle=1)
        self.assertEqual(m.entity("coil").properties["hot"].value, True)
        # Key with no underscore: entity is the key, property "value".
        m.observe("temperature", 21, cycle=1)
        self.assertEqual(m.entity("temperature").properties["value"].value, 21)


class TestEpistemicStatus(unittest.TestCase):
    def test_six_statuses_distinct(self):
        m = WorldModel()
        m.observe("coil_hot", True, cycle=1)
        self.assertEqual(m.status_of("coil", "hot"), EpistemicStatus.observed)
        m.teach("alpha_key", "held", True, cycle=1)
        self.assertEqual(m.status_of("alpha_key", "held"), EpistemicStatus.told)
        m.set_property("beta_tool", "held", True,
                       status=EpistemicStatus.inferred, confidence=0.8,
                       provenance="inference:proof-1", cycle=2)
        self.assertEqual(m.status_of("beta_tool", "held"), EpistemicStatus.inferred)
        m.set_property("gamma_tool", "held", False,
                       status=EpistemicStatus.assumed, confidence=0.5,
                       provenance="planning-default", cycle=2)
        self.assertEqual(m.status_of("gamma_tool", "held"), EpistemicStatus.assumed)
        m.set_property("delta_tool", "held", True,
                       status=EpistemicStatus.predicted, confidence=0.9,
                       provenance="predicted:take_delta", cycle=2)
        self.assertEqual(m.status_of("delta_tool", "held"), EpistemicStatus.predicted)
        # Unknown: absent entity and absent property alike.
        self.assertEqual(m.status_of("no_such_thing", "held"), EpistemicStatus.unknown)
        self.assertEqual(m.status_of("coil", "no_such_property"), EpistemicStatus.unknown)
        self.assertEqual(
            set(EpistemicStatus),
            {EpistemicStatus.observed, EpistemicStatus.inferred,
             EpistemicStatus.assumed, EpistemicStatus.predicted,
             EpistemicStatus.told, EpistemicStatus.unknown})

    def test_unknown_is_not_false(self):
        m = WorldModel()
        # Explicitly observed False is a known False, not unknown.
        m.observe("door_open", False, cycle=1)
        self.assertEqual(m.status_of("door", "open"), EpistemicStatus.observed)
        self.assertIs(m.entity("door").properties["open"].value, False)
        # A key never observed at all is unknown — never read as False.
        self.assertEqual(m.status_of("coil", "hot"), EpistemicStatus.unknown)
        self.assertNotIn("coil_hot", m.observed_keys())
        self.assertIn("coil_hot", m.unobserved_keys(["coil_hot", "door_open"]))


class TestRelationsHierarchySpace(unittest.TestCase):
    def test_relations_and_nary(self):
        m = WorldModel()
        rel = m.add_relation("alpha_key", "located_in", "lab", strength=1.0)
        self.assertEqual(rel.subject, "alpha_key")
        self.assertEqual(rel.relation, "located_in")
        self.assertEqual(rel.object, "lab")
        self.assertEqual(rel.status, EpistemicStatus.observed)
        self.assertEqual(m.relations(subject="alpha_key"),
                         [rel])
        self.assertEqual(m.relations(relation="located_in"), [rel])
        self.assertEqual(m.relations(subject="nobody"), [])
        # N-ary relation: the object is a list (a route through rooms).
        route = m.add_relation("hall", "route", ["hall", "lab", "store"],
                               provenance="observed-transitions")
        self.assertEqual(route.object, ["hall", "lab", "store"])
        self.assertIn(route, m.relations(relation="route"))

    def test_located_in_derived_from_observations(self):
        m = WorldModel()
        m.observe("has_alpha_key", True, cycle=1)
        # No location known yet: no located_in relation is invented.
        self.assertEqual(m.believed_location("alpha_key")[0], None)
        m.observe("at", "hall", cycle=2)
        room, conf, status = m.believed_location("alpha_key")
        self.assertEqual((room, conf, status),
                         ("hall", 1.0, EpistemicStatus.observed))
        # Moving moves what is held; dropping it drops the relation.
        m.observe("at", "lab", cycle=3)
        self.assertEqual(m.believed_location("alpha_key")[0], "lab")
        m.observe("has_alpha_key", False, cycle=4)
        self.assertEqual(m.believed_location("alpha_key")[0], None)
        self.assertEqual(m.believed_location("alpha_key")[2],
                         EpistemicStatus.unknown)

    def test_ancestors_walk_is_a_chains(self):
        m = WorldModel()
        m.add_relation("alpha_cat", "is_a", "mammal")
        m.add_relation("mammal", "is_a", "animal")
        m.add_relation("animal", "is_a", "living_thing")
        self.assertEqual(m.ancestors("alpha_cat"),
                         ["mammal", "animal", "living_thing"])
        self.assertEqual(m.ancestors("mammal"), ["animal", "living_thing"])
        self.assertEqual(m.ancestors("living_thing"), [])
        self.assertEqual(m.ancestors("unknown_thing"), [])
        # A cycle in the chain terminates instead of looping forever.
        m.add_relation("living_thing", "is_a", "mammal")
        self.assertEqual(m.ancestors("alpha_cat"),
                         ["mammal", "animal", "living_thing"])

    def test_spatial_graph_and_reachable(self):
        m = WorldModel()
        m.record_transition("hall", "lab", cycle=1)
        m.record_transition("lab", "store", cycle=2)
        self.assertEqual(m.reachable("hall"), {"hall", "lab", "store"})
        self.assertEqual(m.reachable("lab"), {"lab", "store"})
        self.assertEqual(m.reachable("store"), {"store"})
        self.assertEqual(m.reachable("nowhere"), {"nowhere"})
        # Edges are directed: no edge back was observed.
        m.record_transition("store", "lab", cycle=3)
        self.assertEqual(m.reachable("store"), {"store", "lab"})
        self.assertNotIn("hall", m.reachable("store"))


class TestTemporal(unittest.TestCase):
    def test_ordering_and_history(self):
        m = WorldModel()
        m.observe("at", "hall", cycle=1)
        m.observe("door_open", False, cycle=2)
        m.observe("at", "lab", cycle=3)
        m.observe("door_open", True, cycle=4)
        self.assertTrue(m.before("at", "door_open"))
        self.assertFalse(m.before("door_open", "at"))
        self.assertTrue(m.after("door_open", "at"))
        self.assertFalse(m.before("at", "never_seen_key"))
        self.assertFalse(m.before("never_seen_key", "at"))
        self.assertEqual(m.history("self", "at"), [(1, "hall"), (3, "lab")])
        self.assertEqual(m.history("door", "door_open"),
                         [(2, False), (4, True)])
        self.assertEqual(m.history("door", "nope"), [])
        # events_since is inclusive of the given cycle.
        self.assertEqual([e.cycle for e in m.events_since(3)], [3, 4])
        self.assertEqual([e.cycle for e in m.events_since(1)], [1, 2, 3, 4])
        self.assertEqual(m.events_since(99), [])


class TestConstraints(unittest.TestCase):
    def test_check_state_with_custom_constraint(self):
        m = WorldModel()
        m.add_constraint("no_hot_coil",
                         lambda s: s.get("coil_hot") is not True,
                         "coil must not be hot")
        self.assertEqual(m.check_state({"coil_hot": True}), ["no_hot_coil"])
        self.assertEqual(m.check_state({"coil_hot": False}), [])
        self.assertEqual(m.check_state({}), [])
        self.assertIn("door_blocks_entry", m.constraints())
        self.assertIn("containment", m.constraints())
        self.assertIn("no_hot_coil", m.constraints())

    def test_check_plan_door_blocks_entry_mid_sequence(self):
        m = WorldModel()
        planner = _planner()
        start = {"at": "hall", "door_open": True}
        # Design §2.7 (coordinator fix): only a room the model has
        # LEARNED is door-gated counts here — the gate is established
        # by a failed entry while the door was closed. The worker
        # build's blanket rule (any move while any door is closed)
        # false-positived on the standard demo plan (hall -> lab needs
        # no door) and forced every plan through the
        # plan_model_disagreement fallback; the gated-edge rule is
        # what the design specified, and the mid-sequence violation
        # assertion below is unchanged, with the gate established.
        m.note_failed_entry("store", cycle=0)
        # Step 1 moves (door open: fine), step 2 closes the door,
        # step 3 tries to move into the gated store with the door
        # closed: violation at the THIRD step, not the first — the
        # check runs through the plan.
        steps = [planner.actions[0], planner.actions[2], planner.actions[1]]
        ok, violated = m.check_plan(steps, start, planner.effects_fn)
        self.assertFalse(ok)
        self.assertEqual(violated, "door_blocks_entry")
        # The same plan with the door left open is valid throughout.
        steps_ok = [planner.actions[0], planner.actions[1]]
        self.assertEqual(m.check_plan(steps_ok, start, planner.effects_fn),
                         (True, None))

    def test_check_plan_ungated_move_while_door_closed_is_valid(self):
        # Coordinator fix: a move into a room never observed to need a
        # door must NOT violate, even with a door closed elsewhere
        # (the demo case: hall -> lab while the vault door is closed).
        m = WorldModel()
        planner = _planner()
        start = {"at": "hall", "door_open": False}
        self.assertEqual(m.door_gated_rooms(), set())
        ok, violated = m.check_plan([planner.actions[0]], start,
                                    planner.effects_fn)
        self.assertEqual((ok, violated), (True, None))
        # ... and the gate, once learned from a failed entry, persists
        # across a save/load round-trip.
        m.note_failed_entry("store", cycle=1)
        m2 = WorldModel.from_json(m.to_json())
        self.assertEqual(m2.door_gated_rooms(), {"store"})
        ok, violated = m2.check_plan(
            [planner.actions[0], planner.actions[2], planner.actions[1]],
            {"at": "hall", "door_open": True}, planner.effects_fn)
        self.assertEqual((ok, violated), (False, "door_blocks_entry"))

    def test_check_plan_door_unknown_is_not_assumed_closed(self):
        m = WorldModel()
        planner = _planner()
        # door_open never observed: the move must NOT be rejected —
        # unknown is never assumed False.
        start = {"at": "hall"}
        ok, violated = m.check_plan([planner.actions[0]], start,
                                    planner.effects_fn)
        self.assertEqual((ok, violated), (True, None))

    def test_check_plan_containment(self):
        m = WorldModel()
        m.add_relation("alpha_key", "located_in", "lab")
        planner = _planner()
        take = [a for a in planner.actions if a.name == "take_alpha_key"][0]
        # Taking from the wrong room violates containment.
        ok, violated = m.check_plan([take], {"at": "hall", "door_open": True},
                                    planner.effects_fn)
        self.assertFalse(ok)
        self.assertEqual(violated, "containment")
        # Taking from the object's own room is fine.
        self.assertEqual(
            m.check_plan([take], {"at": "lab", "door_open": True},
                         planner.effects_fn),
            (True, None))
        # Unknown location never violates.
        m2 = WorldModel()
        self.assertEqual(
            m2.check_plan([take], {"at": "hall", "door_open": True},
                          planner.effects_fn),
            (True, None))

    def test_check_plan_accepts_action_objects_dicts_and_names(self):
        m = WorldModel()
        planner = _planner()
        start = {"at": "hall", "door_open": True}
        self.assertEqual(
            m.check_plan([{"name": "go_hall_lab"}], start, planner.effects_fn),
            (True, None))
        self.assertEqual(
            m.check_plan(["go_hall_lab"], start, planner.effects_fn),
            (True, None))


class TestPermanence(unittest.TestCase):
    def test_decay_only_for_unobserved_entities(self):
        m = WorldModel()
        m.set_property("alpha_key", "held", True,
                       status=EpistemicStatus.inferred, confidence=0.8,
                       provenance="inference", cycle=1)
        m.observe("at", "hall", cycle=2)  # "self" observed in cycle 2
        m.decay(2)
        # alpha_key was not observed in cycle 2: confidence decays once.
        prop = m.entity("alpha_key").properties["held"]
        self.assertAlmostEqual(prop.confidence, 0.8 * PERMANENCE_DECAY)
        # "self" was observed in cycle 2: no decay.
        self.assertEqual(m.entity("self").properties["location"].confidence, 1.0)
        # Decaying the same cycle again is a no-op (no double decay).
        m.decay(2)
        self.assertAlmostEqual(
            m.entity("alpha_key").properties["held"].confidence,
            0.8 * PERMANENCE_DECAY)
        # A further unobserved cycle decays again.
        m.decay(3)
        self.assertAlmostEqual(
            m.entity("alpha_key").properties["held"].confidence,
            0.8 * PERMANENCE_DECAY ** 2)
        self.assertEqual(PERMANENCE_DECAY, 0.9)

    def test_gone_only_on_explicit_evidence(self):
        m = WorldModel()
        m.observe("has_alpha_key", True, cycle=1)
        m.observe("at", "hall", cycle=1)
        # Unobserved for many cycles: still believed present, never gone.
        for cycle in (2, 3, 4, 5):
            m.decay(cycle)
        self.assertFalse(m.is_gone("alpha_key"))
        room, conf, _status = m.believed_location("alpha_key")
        self.assertEqual(room, "hall")
        self.assertLess(conf, 1.0)  # decayed, but still believed present
        self.assertFalse(m.is_gone("no_such_thing"))
        # Only an explicit mark_gone makes it gone.
        m.mark_gone("alpha_key", "no intact alpha_key remains", cycle=6)
        self.assertTrue(m.is_gone("alpha_key"))
        prop = m.entity("alpha_key").properties["exists"]
        self.assertIs(prop.value, False)
        self.assertEqual(prop.status, EpistemicStatus.observed)
        self.assertEqual(prop.confidence, 1.0)
        # Decay never erodes the gone flag itself.
        m.decay(7)
        self.assertEqual(
            m.entity("alpha_key").properties["exists"].confidence, 1.0)


class TestPartialObservability(unittest.TestCase):
    def test_observed_vs_unobserved_keys(self):
        m = WorldModel()
        m.observe("at", "hall", cycle=1)
        m.observe("has_alpha_key", True, cycle=1)
        self.assertEqual(m.observed_keys(), {"at", "has_alpha_key"})
        self.assertEqual(
            m.unobserved_keys(["at", "door_open", "coil_hot"]),
            ["door_open", "coil_hot"])
        self.assertEqual(m.unobserved_keys(["at"]), [])
        # Observing a key later removes it from the unobserved set.
        m.observe("door_open", True, cycle=2)
        self.assertEqual(m.unobserved_keys(["at", "door_open", "coil_hot"]),
                         ["coil_hot"])


class TestExogenousVsActionEvents(unittest.TestCase):
    def test_cause_distinguishes_exogenous_from_action_effect(self):
        m = WorldModel()
        m.observe("door_open", True, cycle=1)
        # The door changes with no action in between: exogenous.
        m.observe("door_open", False, cycle=2, cause="exogenous")
        # The door changes because an action ran: action effect.
        m.observe("door_open", True, cycle=3, cause="action:open_door")
        events = [e for e in m.events() if e.key == "door_open"]
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1].cause, "exogenous")
        self.assertEqual(events[1].kind, "exogenous")
        self.assertEqual(events[1].old_value, True)
        self.assertEqual(events[1].new_value, False)
        self.assertEqual(events[2].cause, "action:open_door")
        self.assertEqual(events[2].kind, "observation")
        exogenous = [e for e in events if e.cause == "exogenous"]
        self.assertEqual(len(exogenous), 1)


class TestPrediction(unittest.TestCase):
    def test_predict_without_causal_records_is_confidence_1(self):
        m = WorldModel()
        planner = _planner()
        state = {"at": "hall", "door_open": True}
        expected, conf = m.predict("go_hall_lab", state, planner.effects_fn)
        self.assertEqual(expected["at"], "lab")
        self.assertEqual(expected["door_open"], True)
        self.assertEqual(conf, 1.0)
        # An inapplicable action predicts no change, still at 1.0.
        expected, conf = m.predict("go_lab_store", state, planner.effects_fn)
        self.assertEqual(expected, state)
        self.assertEqual(conf, 1.0)

    def test_predict_uses_laplace_smoothed_causal_rate(self):
        m = WorldModel()
        planner = _planner()
        causal = CausalModel()
        for _ in range(3):
            causal.record("go_hall_lab", "at", True)
        causal.record("go_hall_lab", "at", False)  # 3 met / 4 total
        state = {"at": "hall", "door_open": True}
        _expected, conf = m.predict("go_hall_lab", state, planner.effects_fn,
                                    causal=causal)
        # Laplace: (3 + 1) / (4 + 2) = 2/3.
        self.assertAlmostEqual(conf, 4 / 6)
        # No record for this (action, effect): back to 1.0.
        _expected, conf = m.predict("open_door", state, planner.effects_fn,
                                    causal=causal)
        self.assertEqual(conf, 1.0)

    def test_prediction_error_tracking_and_rate(self):
        m = WorldModel()
        rec = m.record_prediction_error("open_door", "door_open",
                                        True, False, cycle=3)
        self.assertEqual(rec["error"], 1)
        m.record_prediction_error("open_door", "door_open", True, True, cycle=4)
        self.assertEqual(m.error_rate("open_door", "door_open"), 0.5)
        # Numeric errors use absolute difference, not 0/1.
        rec = m.record_prediction_error("go_hall_lab", "distance", 5, 3, cycle=5)
        self.assertEqual(rec["error"], 2)
        self.assertEqual(m.error_rate("go_hall_lab", "distance"), 2.0)
        self.assertIsNone(m.error_rate("nope", "door_open"))
        self.assertEqual(len(m.prediction_errors()), 3)


class TestContradictionAndRevision(unittest.TestCase):
    def test_contradiction_keeps_both_values(self):
        m = WorldModel()
        m.observe("door_open", True, cycle=1)
        record = m.note_contradiction("door", "open", True, False, cycle=2)
        self.assertEqual(record["value_a"], True)
        self.assertEqual(record["value_b"], False)
        self.assertEqual(record["status"], "disputed")
        self.assertEqual(len(m.contradictions()), 1)
        # The original entry is never overwritten by the dispute.
        self.assertIs(m.entity("door").properties["open"].value, True)
        self.assertEqual(m.status_of("door", "open"), EpistemicStatus.observed)

    def test_revision_retains_old_value_and_new_model_is_used(self):
        m = WorldModel()
        planner = _planner()
        m.observe("door_open", False, cycle=1)
        # Coordinator fix (gated-edge rule, design §2.7): the lab is
        # established as door-gated first (a failed entry while the
        # door was closed), so under the old model moving there IS
        # blocked by the closed door. Without the learned gate the
        # move is valid — the blanket rule this test relied on was
        # the worker build's deviation, not the design.
        m.note_failed_entry("lab", cycle=1)
        # Under the old model, moving is blocked by the closed door.
        ok, _v = m.check_plan(
            [planner.actions[0]], {"at": "hall", "door_open": False},
            planner.effects_fn)
        self.assertFalse(ok)
        # Revise: the old value is retained, the new value takes over.
        key = m.revise("door_open", False, True, "sensor recalibration",
                       cycle=2)
        self.assertEqual(key, "door_open")
        revisions = m.revisions()
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]["old_value"], False)
        self.assertEqual(revisions[0]["new_value"], True)
        self.assertEqual(revisions[0]["reason"], "sensor recalibration")
        self.assertEqual(revisions[0]["cycle"], 2)
        self.assertEqual(m.entity("door").properties["open"].value, True)
        self.assertEqual(m.status_of("door", "open"), EpistemicStatus.observed)
        # The NEXT plan uses the new model: the move is now allowed.
        ok, violated = m.check_plan(
            [planner.actions[0]], {"at": "hall", "door_open": True},
            planner.effects_fn)
        self.assertEqual((ok, violated), (True, None))
        # Revisions are append-only: a second revision keeps the first.
        m.revise("door_open", True, False, "door slammed", cycle=3)
        self.assertEqual(len(m.revisions()), 2)
        self.assertEqual(m.revisions()[0]["old_value"], False)


class TestSingleUseSuspect(unittest.TestCase):
    def test_mark_single_use_suspect(self):
        m = WorldModel()
        m.observe("has_gamma_tool", True, cycle=1)
        ent = m.mark_single_use_suspect("gamma_tool", cycle=2)
        prop = ent.properties["single_use_suspect"]
        self.assertIs(prop.value, True)
        self.assertEqual(prop.status, EpistemicStatus.inferred)
        self.assertEqual(prop.confidence, 0.7)
        self.assertIs(m.entity("gamma_tool"), ent)


class TestPersistence(unittest.TestCase):
    def test_json_round_trip_preserves_everything(self):
        m = WorldModel()
        m.observe("at", "hall", cycle=1)
        m.observe("has_alpha_key", True, cycle=1)
        m.observe("door_open", False, cycle=2)
        m.observe("at", "lab", cycle=3)
        m.record_transition("hall", "lab", cycle=3)
        m.add_relation("alpha_cat", "is_a", "mammal")
        m.add_relation("hall", "route", ["hall", "lab", "store"])
        m.note_contradiction("door", "open", False, True, cycle=2)
        m.record_prediction_error("open_door", "door_open", True, False, cycle=2)
        m.revise("door_open", False, True, "sensor recalibration", cycle=3)
        m.mark_single_use_suspect("gamma_tool", cycle=3)
        m.add_constraint("no_hot_coil", lambda s: True, "custom-kept-by-name")
        m.decay(4)
        # Coordinator fix (gated-edge rule, design §2.7): the store is
        # established as door-gated, so the mid-sequence check below
        # still fails on door_blocks_entry after the round-trip (the
        # gate itself is persisted in world.json).
        m.note_failed_entry("store", cycle=4)

        blob = m.to_json()
        json.dumps(blob)  # must be JSON-serialisable
        loaded = WorldModel.from_json(json.loads(json.dumps(blob)))

        self.assertEqual(loaded.entity("alpha_key").properties["held"].value, True)
        self.assertEqual(loaded.entity("alpha_key").first_seen_cycle, 1)
        self.assertEqual(loaded.believed_location("alpha_key")[0], "lab")
        self.assertEqual(loaded.ancestors("alpha_cat"), ["mammal"])
        self.assertEqual(loaded.reachable("hall"), {"hall", "lab"})
        # The move to lab is logged twice: once by the observation and
        # once by the recorded transition — both are real records.
        self.assertEqual(loaded.history("self", "at"),
                         [(1, "hall"), (3, "lab"), (3, "lab")])
        self.assertEqual(loaded.history("door", "door_open"),
                         [(2, False), (3, True)])
        self.assertEqual(len(loaded.contradictions()), 1)
        self.assertEqual(loaded.error_rate("open_door", "door_open"), 1.0)
        self.assertEqual(len(loaded.revisions()), 1)
        self.assertEqual(loaded.revisions()[0]["new_value"], True)
        self.assertEqual(loaded.observed_keys(),
                         {"at", "has_alpha_key", "door_open"})
        self.assertEqual(
            loaded.entity("gamma_tool").properties["single_use_suspect"].value,
            True)
        # Constraints survive by name; built-ins are live again.
        self.assertIn("door_blocks_entry", loaded.constraints())
        self.assertIn("containment", loaded.constraints())
        self.assertIn("no_hot_coil", loaded.constraints())
        planner = _planner()
        ok, violated = loaded.check_plan(
            [planner.actions[0], planner.actions[2], planner.actions[1]],
            {"at": "hall", "door_open": True}, planner.effects_fn)
        self.assertFalse(ok)
        self.assertEqual(violated, "door_blocks_entry")
        # A second round-trip is stable.
        self.assertEqual(WorldModel.from_json(loaded.to_json()).to_json(),
                         loaded.to_json())


if __name__ == "__main__":
    unittest.main()
