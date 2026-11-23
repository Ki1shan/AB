"""Phase 5 persistence tests (Worker 2).

Persistence v3 (``world.json``): the world model and the pattern
library survive a save/load round-trip, and a loaded brain still
completes a goal in a new world. Old v2 saves (no ``world.json``)
still load, with an empty model/library.
"""

import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.eval.disconnected import (
    WORLD_A,
    WORLD_B,
    _run_with_brain,
    make_variant_factories,
)


def _full_factory():
    return make_variant_factories()["full"]


class TestPersistenceV3RoundTrip(unittest.TestCase):
    def test_world_model_patterns_and_revisions_survive(self):
        rec, brain = _run_with_brain(WORLD_A, _full_factory(), 80)
        self.assertTrue(rec["completed"])
        self.assertTrue(brain.patterns.patterns())
        self.assertTrue(brain.world_model.revisions())
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            self.assertIn("world.json", os.listdir(d))
            # Phase 7 pin update (following the Phase 2 §9 / Phase 3 §8 /
            # Phase 4 §14.1 / Phase 5 / Phase 6 precedent — same intent, new
            # intended semantics, not weakened): persistence v5 adds
            # execution.json (live session + ledger), so the manifest version is 5
            # and the file set gains "execution.json".
            # Phase 9 pin update (same §9 precedent): persistence v6
            # adds revision.json, so the manifest version is 6.
            self.assertIn("execution.json", os.listdir(d))
            with open(os.path.join(d, "meta.json"),
                      encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["version"], 7)
            self.assertIn("world.json", meta["files"])
            self.assertIn("learning.json", meta["files"])
            self.assertIn("execution.json", meta["files"])
            with open(os.path.join(d, "world.json"),
                      encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertIn("model", saved)
            self.assertIn("patterns", saved)
            loaded = Brain(granted={"compute", "environment"})
            loaded.load(d)
            self.assertEqual(loaded.load_warnings, [])
        # The model survives: same entities, same door belief, same
        # revision trail (the door_open revisions from the reset).
        self.assertEqual(
            {e.id for e in loaded.world_model.entities()},
            {e.id for e in brain.world_model.entities()})
        self.assertEqual(
            loaded.world_model.status_of("door", "open"),
            brain.world_model.status_of("door", "open"))
        self.assertEqual(
            [(r["key"], r["old_value"], r["new_value"])
             for r in loaded.world_model.revisions()],
            [(r["key"], r["old_value"], r["new_value"])
             for r in brain.world_model.revisions()])
        # The patterns survive: same count, same kinds and effects.
        self.assertEqual(
            [(p.kind, p.effect) for p in loaded.patterns.patterns()],
            [(p.kind, p.effect) for p in brain.patterns.patterns()])
        # The loaded brain still completes a goal in a new world
        # (transfer protocol: transient state cleared, patterns kept).
        # Grants are a deployment authorization, not saved state
        # (default is compute-only), so the loaded brain above was
        # built with the environment grant, like every acting test.
        loaded.clear_transient_for_new_world()
        self.assertTrue(loaded.patterns.patterns())
        rec_b, _ = _run_with_brain(WORLD_B, _full_factory(), 80,
                                   brain=loaded)
        self.assertTrue(rec_b["completed"])

    def test_corrupt_world_json_skipped_with_warning(self):
        _, brain = _run_with_brain(WORLD_A, _full_factory(), 80)
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            with open(os.path.join(d, "world.json"), "w",
                      encoding="utf-8") as fh:
                fh.write("{not json")
            loaded = Brain()
            loaded.load(d)
            self.assertTrue(any("world.json" in w
                                for w in loaded.load_warnings))
            self.assertEqual(loaded.world_model.entities(), [])
            self.assertEqual(loaded.patterns.patterns(), [])


class TestPersistenceV2BackwardsCompatible(unittest.TestCase):
    def test_v2_save_still_loads_with_warning_and_empty_model(self):
        # Build a v2-shaped folder exactly as specified: save (v3),
        # delete world.json, downgrade the manifest version to 2. The
        # manifest still lists world.json, so the load warns about the
        # missing file — and the brain starts with an empty model and
        # library while everything else loads.
        _, brain = _run_with_brain(WORLD_A, _full_factory(), 80)
        with tempfile.TemporaryDirectory() as d:
            brain.save(d)
            os.remove(os.path.join(d, "world.json"))
            meta_path = os.path.join(d, "meta.json")
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            # Phase 7 pin update (same §9 precedent as above): the
            # save is now v5; the downgrade to 2 still simulates a v2
            # Phase 9 pin update (same §9 precedent): the save is now v6.
            # save, which is the intent of this test.
            # Phase 11 pin update (same §9 precedent): the save is now v7
            # (research.json added).
            self.assertEqual(meta["version"], 7)
            meta["version"] = 2
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            loaded = Brain()
            loaded.load(d)  # must not raise
            self.assertTrue(any("world.json" in w
                                for w in loaded.load_warnings))
            self.assertEqual(loaded.world_model.entities(), [])
            self.assertEqual(loaded.world_model.observed_keys(), set())
            self.assertEqual(loaded.patterns.patterns(), [])
            # The rest of the save still loads: the goal record and
            # the learner's episodes survived.
            self.assertTrue(loaded.goals.all())
            self.assertTrue(loaded.learner.episodes)


if __name__ == "__main__":
    unittest.main()
