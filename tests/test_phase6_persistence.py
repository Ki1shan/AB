"""Phase 6 persistence tests (Worker 2): v4 round-trip, v3 backwards
compatibility, corrupt learning.json.

All names here are invented for the tests (alpha/beta tools, gate_open
effect, hall/lab rooms) — no world specs are imported.
"""
import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.core.learning import Episode

from tests.test_phase6_integration import _FakeEnv, gate_actions


def _rich_brain():
    """A brain with promoted P-C + P-D patterns, a durable_tool
    concept, a requires_has_key concept, calibration records and
    prioritizer uses — the full Phase 6 state."""
    brain = Brain(gate_actions(alpha_cost=1.0, beta_cost=1.0, decoy_cost=5.0),
                  granted={"compute", "environment"})
    brain.world = {"at": "hall", "has_alpha_tool": True,
                   "has_beta_tool": True, "gate_open": False}
    # P-C evidence: the durable tool succeeds 3x across 2 situations.
    for cycle, sit in ((1, {"at": "hall", "has_alpha_tool": True}),
                       (2, {"at": "lab", "has_alpha_tool": True}),
                       (3, {"at": "hall", "has_alpha_tool": True})):
        brain.learner.record(Episode(cycle, "use_alpha_tool", sit, True))
    # P-D evidence: open_gate fails twice lacking has_key, succeeds once with it.
    brain.learner.record(Episode(4, "open_gate", {"at": "hall"}, False))
    brain.learner.record(Episode(5, "open_gate", {"at": "lab"}, False))
    brain.learner.record(
        Episode(6, "open_gate", {"at": "hall", "has_key": True}, True))
    brain._consolidation_pass()
    # Calibration records (predicted 1.0, mixed outcomes).
    brain.calibration.record("open_gate", "gate_open", 1.0, False)
    brain.calibration.record("open_gate", "gate_open", 1.0, True)
    return brain


class TestPersistenceV4RoundTrip(unittest.TestCase):
    def test_learning_json_round_trip(self):
        brain = _rich_brain()
        # The P-A contrast (tool-less open_gate fails, tooled
        # use_alpha_tool succeeds) also fires, immediately: three
        # pattern kinds in the library, two of them promoted.
        self.assertEqual(
            {p.kind for p in brain.patterns.patterns()},
            {"unreliable_toolless_opener", "durable_tool",
             "requires_precondition"})
        self.assertIsNotNone(brain.concepts.get("durable_tool"))
        self.assertIn("requires_has_key", brain.concepts.names())
        self.assertEqual(brain.calibration.counts("open_gate", "gate_open"),
                         (1, 2))
        uses_before = brain.prioritizer.uses(
            "use_alpha_tool", {"at": "hall", "has_alpha_tool": True})
        self.assertGreater(uses_before, 0)
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            self.assertIn("learning.json", os.listdir(d))
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            # Phase 7 pin update (Phase 2 §9 precedent — same intent,
            # new intended semantics, not weakened): persistence v5.
            # Phase 9 pin update (same §9 precedent): persistence v6
            # (revision.json added).
            # Phase 11 pin update (same §9 precedent): persistence v7
            # (research.json added).
            self.assertEqual(meta["version"], 7)
            self.assertIn("learning.json", meta["files"])
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)
            self.assertEqual(loaded.load_warnings, [])
            # Candidates survive, with their promotion records.
            promoted = loaded.consolidator.promoted()
            self.assertEqual(len(promoted), 2)
            self.assertEqual(
                {c["content"]["kind"] for c in promoted},
                {"durable_tool", "requires_precondition"})
            # The promoted patterns survive.
            self.assertEqual(
                {p.kind for p in loaded.patterns.patterns()},
                {"unreliable_toolless_opener", "durable_tool",
                 "requires_precondition"})
            # The concepts survive.
            concept = loaded.concepts.get("durable_tool")
            self.assertIsNotNone(concept)
            self.assertTrue(concept.positives)
            self.assertIn("requires_has_key", loaded.concepts.names())
            # The calibration data survives, with the same confidence.
            self.assertEqual(
                loaded.calibration.counts("open_gate", "gate_open"), (1, 2))
            self.assertAlmostEqual(
                loaded.calibration.calibrated_confidence("open_gate", "gate_open"),
                brain.calibration.calibrated_confidence("open_gate", "gate_open"))
            # The prioritizer usefulness survives.
            self.assertEqual(
                loaded.prioritizer.uses(
                    "use_alpha_tool", {"at": "hall", "has_alpha_tool": True}),
                uses_before)

    def test_loaded_brain_completes_goal_with_preference_intact(self):
        brain = _rich_brain()
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)
            # The learned preference is intact: the durable tool is
            # cheaper than the fragile one (0.8 pattern x 0.8 concept).
            alpha = next(a for a in loaded.planner.actions
                         if a.name == "use_alpha_tool")
            beta = next(a for a in loaded.planner.actions
                        if a.name == "use_beta_tool")
            self.assertLess(loaded.planner.effective_cost(alpha),
                            loaded.planner.effective_cost(beta))
            # ... and the loaded brain completes the goal with it.
            def _open_alpha(env, args):
                env.s["gate_open"] = True
                return True, "the gate opens"

            def _open_beta(env, args):
                env.s["gate_open"] = True
                return True, "the gate opens"

            env = _FakeEnv({"at": "hall", "has_alpha_tool": True,
                            "has_beta_tool": True, "gate_open": False},
                           {"use_alpha_tool": _open_alpha,
                            "use_beta_tool": _open_beta,
                            "open_gate": lambda e, a: (False, "no move")})
            loaded.add_goal("open the gate", {"gate_open": True})
            stop = loaded.run(env, 60)
            self.assertEqual(stop, "done")
            self.assertIn("use_alpha_tool", env.log)
            self.assertNotIn("use_beta_tool", env.log)

    def test_second_round_trip_is_stable(self):
        brain = _rich_brain()
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)
            d2 = tempfile.mkdtemp()
            loaded.save(d2)
            reloaded = Brain(granted={"compute", "environment"})
            reloaded.load(d2)
            self.assertEqual(reloaded.load_warnings, [])
            self.assertEqual(
                {p.kind for p in reloaded.patterns.patterns()},
                {"unreliable_toolless_opener", "durable_tool",
                 "requires_precondition"})
            self.assertEqual(
                reloaded.calibration.counts("open_gate", "gate_open"), (1, 2))
            self.assertEqual(
                len(reloaded.consolidator.promoted()), 2)


class TestPersistenceV3BackwardsCompatible(unittest.TestCase):
    def test_v3_save_loads_with_warning_and_empty_phase6_state(self):
        brain = _rich_brain()
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            # Simulate a v3 save: drop learning.json and downgrade
            # the manifest to version 3 without it.
            os.remove(os.path.join(d, "learning.json"))
            meta_path = os.path.join(d, "meta.json")
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            # Phase 7 pin update (same §9 precedent): the save is now v5.
            # Phase 9 pin update (same §9 precedent): the save is now v6.
            # Phase 11 pin update (same §9 precedent): the save is now v7.
            self.assertEqual(meta["version"], 7)
            meta["version"] = 3
            del meta["files"]["learning.json"]
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)  # must not raise
            self.assertTrue(any("learning.json" in w
                                for w in loaded.load_warnings))
            # Empty Phase 6 state, everything else intact. (The
            # patterns survive via world.json, which a real v3 save
            # also carries — only the learning.json state is gone.)
            self.assertEqual(loaded.consolidator.all(), [])
            self.assertEqual(loaded.concepts.names(), [])
            self.assertEqual(loaded.calibration.total_records(), 0)
            self.assertTrue(loaded.learner.episodes)
            self.assertEqual(
                {p.kind for p in loaded.patterns.patterns()},
                {"unreliable_toolless_opener", "durable_tool",
                 "requires_precondition"})

    def test_corrupt_learning_json_skipped_with_warning(self):
        brain = _rich_brain()
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            with open(os.path.join(d, "learning.json"), "w",
                      encoding="utf-8") as fh:
                fh.write("{not json")
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)  # must not raise
            self.assertTrue(any("learning.json" in w
                                for w in loaded.load_warnings))
            # The brain keeps working with the rest of its state...
            self.assertTrue(loaded.learner.episodes)
            # ... and with fresh (empty) Phase 6 state.
            self.assertEqual(loaded.consolidator.all(), [])
            self.assertEqual(loaded.concepts.names(), [])
            self.assertEqual(loaded.calibration.total_records(), 0)


if __name__ == "__main__":
    unittest.main()
