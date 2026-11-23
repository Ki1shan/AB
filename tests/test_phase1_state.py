"""Phase 1, Workstream 1: shared typed cognitive state.

Acceptance: one typed store with stable ids, source, confidence, cycle,
provenance, type and status; same-source overwrites update in place,
different-source overwrites supersede with history kept; JSON round-trip
with schema validation; Brain.world stays a compatible plain-dict view.
"""
import unittest

from ab.brain import Brain
from ab.core.state import CognitiveState, StateEntry


class TestCognitiveState(unittest.TestCase):
    def test_set_get_as_dict_and_metadata(self):
        st = CognitiveState()
        st.set("at", "lab", source="env", confidence=1.0, cycle=3,
               provenance="env:observe", entry_type="observation")
        e = st.get("at")
        self.assertEqual(e.value, "lab")
        self.assertEqual(e.id, "at")
        self.assertEqual(e.source, "env")
        self.assertEqual(e.confidence, 1.0)
        self.assertEqual(e.cycle, 3)
        self.assertEqual(e.provenance, "env:observe")
        self.assertEqual(e.entry_type, "observation")
        self.assertEqual(e.status, "active")
        self.assertEqual(st.as_dict(), {"at": "lab"})

    def test_same_source_overwrite_updates_in_place(self):
        st = CognitiveState()
        st.set("at", "hall", source="env", confidence=1.0, cycle=1)
        created = st.get("at").created_at
        st.set("at", "lab", source="env", confidence=1.0, cycle=2)
        e = st.get("at")
        self.assertEqual(e.value, "lab")
        self.assertEqual(e.created_at, created)
        self.assertEqual(st.history(), [])

    def test_different_source_overwrite_supersedes_with_history(self):
        st = CognitiveState()
        st.set("x", 1, source="env", confidence=1.0, cycle=1)
        st.set("x", 2, source="ltm:fact-9", confidence=0.9, cycle=2,
               entry_type="belief")
        self.assertEqual(st.get("x").value, 2)
        self.assertEqual(len(st.history()), 1)
        self.assertEqual(st.history()[0].status, "superseded")
        self.assertEqual(st.history()[0].value, 1)

    def test_json_round_trip_and_schema_validation(self):
        st = CognitiveState()
        st.set("at", "lab", source="env", confidence=1.0, cycle=2,
               provenance="env:observe")
        st2 = CognitiveState.from_json(st.to_json())
        self.assertEqual(st2.as_dict(), {"at": "lab"})
        self.assertEqual(st2.get("at").provenance, "env:observe")
        self.assertEqual(st2.get("at").cycle, 2)
        with self.assertRaises(ValueError):
            CognitiveState.from_json({"no_entries": []})
        with self.assertRaises(ValueError):
            StateEntry.from_json({"value": 1})

    def test_confidence_is_clamped(self):
        st = CognitiveState()
        st.set("x", 1, source="env", confidence=99.0)
        self.assertEqual(st.get("x").confidence, 1.0)


class TestBrainWorldView(unittest.TestCase):
    def test_world_is_a_plain_dict_view_of_state(self):
        b = Brain()
        b._set_state("at", "hall", source="env", confidence=1.0)
        self.assertEqual(b.world, {"at": "hall"})
        self.assertEqual(b.state.get("at").source, "env")

    def test_world_assignment_keeps_compatibility(self):
        b = Brain()
        b.world = {"at": "lab"}
        self.assertEqual(b.world, {"at": "lab"})
        self.assertEqual(b.state.get("at").source, "external")

    def test_known_is_derived_view_with_ltm_provenance(self):
        # Phase 3 §2.1 (mandated): known:* is a derived view, not a
        # stored entry — _refresh_knowledge was deleted. Provenance now
        # lives in LTM, and known() reads it from there.
        b = Brain()
        b.teach_fact("paris", "capital_of", "france")
        b.add_research_goal("paris", "capital_of", ["https://a.org/p"])
        self.assertTrue(b.known("paris", "capital_of"))
        self.assertIsNone(b.state.get("known:paris|capital_of"))
        fact = b.ltm.query("paris", "capital_of", "france")[0]
        self.assertEqual(fact.sources, {"teacher": 1.0})
        self.assertEqual(fact.status, "verified")

    def test_state_change_is_logged_on_source_switch(self):
        b = Brain()
        b._set_state("x", 1, source="env")
        b._set_state("x", 2, source="ltm:fact-1")
        self.assertTrue(any(e["kind"] == "state_change" for e in b.events))


if __name__ == "__main__":
    unittest.main()
