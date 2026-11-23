"""Phase 12 W6: memory-bounds regression tests (ADR P12-10 / P12-11).

The 10,000-cycle soak FAILED boundedness (docs/phase12-soak-record-
prefixed.json): ``brain.events`` reached 123,567 (cap 1,000, enforced
only at save) and the world-model log reached 13,410 (no cap at all).
Both logs are now capped in-cycle, oldest-first, at MAX_EVENTS=1,000.

These tests pin the caps:
- >1,000 brain events -> exactly 1,000, the NEWEST 1,000 retained.
- >1,000 world-model events (public + internal append paths) ->
  exactly 1,000, newest retained.
- Save/load round-trip after pruning: events.json holds <=1,000,
  newest retained; an oversized pre-fix world.json prunes on load.
- ``before()``/``after()`` on a key whose first event was pruned
  returns False (the honest bounded-log semantics).
- A 2,000-cycle soak run: both stores stay <=1,000 throughout.
"""
import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import room_world_actions
from ab.core.audit import MAX_EVENTS
from ab.eval.soak import Soak
from ab.world.model import WorldModel


def _drive_brain_events(brain, n):
    """Append ``n`` distinct events, one per cycle number."""
    for i in range(n):
        brain.cycle = i
        brain._event("probe", f"event-{i}")


class TestBrainEventCap(unittest.TestCase):
    def test_over_cap_keeps_newest_1000(self):
        b = Brain()
        _drive_brain_events(b, MAX_EVENTS + 500)
        self.assertEqual(len(b.events), MAX_EVENTS)
        # Oldest-first pruning: the first retained event is #500.
        self.assertEqual(b.events[0]["detail"], "event-500")
        self.assertEqual(b.events[0]["cycle"], 500)
        self.assertEqual(b.events[-1]["detail"], "event-1499")
        self.assertEqual(b.events[-1]["cycle"], 1499)

    def test_under_cap_untouched(self):
        b = Brain()
        _drive_brain_events(b, 10)
        self.assertEqual(len(b.events), 10)
        self.assertEqual(b.events[0]["detail"], "event-0")

    def test_explain_renders_only_retained_events(self):
        b = Brain()
        _drive_brain_events(b, MAX_EVENTS + 5)
        lines = b.explain()
        self.assertEqual(len(lines), MAX_EVENTS)
        self.assertNotIn("event-0", " ".join(lines))
        self.assertIn("event-1004", " ".join(lines))


class TestWorldModelEventCap(unittest.TestCase):
    def test_record_event_over_cap_keeps_newest_1000(self):
        m = WorldModel()
        for i in range(MAX_EVENTS + 500):
            m.record_event(kind="probe", entity="e", key=f"k{i}",
                           new_value=i, cycle=i)
        events = m.events()
        self.assertEqual(len(events), MAX_EVENTS)
        self.assertEqual(events[0].cycle, 500)
        self.assertEqual(events[0].key, "k500")
        self.assertEqual(events[-1].cycle, 1499)
        self.assertEqual(events[-1].key, "k1499")

    def test_internal_appends_capped(self):
        # set_property / observe / transitions are internal append
        # sites, not record_event: they must go through the same cap.
        m = WorldModel()
        for i in range(MAX_EVENTS + 200):
            m.set_property("e", "k", i, cycle=i)
        events = m.events()
        self.assertEqual(len(events), MAX_EVENTS)
        self.assertEqual(events[0].cycle, 200)
        self.assertEqual(events[-1].cycle, 1199)
        m2 = WorldModel()
        for i in range(MAX_EVENTS + 200):
            m2.observe("at", f"room{i}", cycle=i)
        self.assertEqual(len(m2.events()), MAX_EVENTS)
        self.assertEqual(m2.events()[-1].cycle, 1199)

    def test_oversized_save_prunes_on_load(self):
        # A pre-fix world.json can hold >1,000 events (the log was
        # never capped): from_json keeps the newest 1,000.
        m = WorldModel()
        data = m.to_json()
        data["events"] = [
            {"cycle": i, "kind": "probe", "entity": "e", "key": f"k{i}",
             "old_value": None, "new_value": i, "cause": "probe"}
            for i in range(MAX_EVENTS + 500)]
        loaded = WorldModel.from_json(data)
        events = loaded.events()
        self.assertEqual(len(events), MAX_EVENTS)
        self.assertEqual(events[0].cycle, 500)
        self.assertEqual(events[-1].cycle, 1499)

    def test_before_after_on_pruned_key_returns_false(self):
        m = WorldModel()
        m.record_event(kind="probe", entity="e", key="first", cycle=0)
        m.record_event(kind="probe", entity="e", key="second", cycle=1)
        self.assertTrue(m.before("first", "second"))
        for i in range(2, MAX_EVENTS + 100):
            m.record_event(kind="probe", entity="e", key=f"k{i}", cycle=i)
        # Both early keys were pruned: no first event is recorded,
        # so ordering is unknowable and both queries answer False.
        self.assertFalse(m.before("first", "second"))
        self.assertFalse(m.after("first", "second"))
        self.assertFalse(m.before("first", "k1100"))
        # Keys still in the window keep working.
        self.assertTrue(m.before("k1050", "k1099"))


class TestSaveLoadRoundTrip(unittest.TestCase):
    def test_events_json_capped_and_newest_retained(self):
        b = Brain(room_world_actions(),
                  granted={"compute", "environment"})
        _drive_brain_events(b, MAX_EVENTS + 500)
        for i in range(MAX_EVENTS + 200):
            b.world_model.set_property("e", "k", i, cycle=i)
        with tempfile.TemporaryDirectory() as folder:
            b.save(folder)
            with open(os.path.join(folder, "events.json")) as fh:
                saved = json.load(fh)
            self.assertEqual(len(saved), MAX_EVENTS)
            self.assertEqual(saved[0]["detail"], "event-500")
            self.assertEqual(saved[-1]["detail"], "event-1499")
            with open(os.path.join(folder, "world.json")) as fh:
                world = json.load(fh)["model"]
            self.assertEqual(len(world["events"]), MAX_EVENTS)
            self.assertEqual(world["events"][0]["cycle"], 200)
            self.assertEqual(world["events"][-1]["cycle"], 1199)
            b2 = Brain()
            b2.load(folder)
            self.assertEqual(len(b2.events), MAX_EVENTS)
            self.assertEqual(b2.events[0]["detail"], "event-500")
            self.assertEqual(b2.events[-1]["detail"], "event-1499")
            self.assertEqual(len(b2.world_model.events()), MAX_EVENTS)
            self.assertEqual(b2.world_model.events()[-1].cycle, 1199)


class TestSoakStoresBounded(unittest.TestCase):
    def test_2000_cycle_run_keeps_both_stores_capped(self):
        # Fast version of the soak assertion: the 10,000-cycle soak
        # is the proof; this pins the regression in the suite.
        record = Soak(target=2000).run()
        samples = record["samples"]
        self.assertGreaterEqual(record["cycles_run"], 2000)
        self.assertLessEqual(
            max(s["audit_events"] for s in samples), MAX_EVENTS,
            "brain.events exceeded MAX_EVENTS during the run")
        self.assertLessEqual(
            max(s["wm_events"] for s in samples), MAX_EVENTS,
            "world-model _events exceeded MAX_EVENTS during the run")
        self.assertEqual(record["stores"]["audit_events"]["end_value"],
                         MAX_EVENTS)
        self.assertEqual(record["stores"]["wm_events"]["end_value"],
                         MAX_EVENTS)


if __name__ == "__main__":
    unittest.main()
