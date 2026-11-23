"""Phase 11 §5 (worker W2): research persistence, v7.

Covers the v7 manifest, Researcher._seen round-trip, corroboration
across a restart, and old v6 saves loading with a warning + empty
research state. All providers here are in-memory — no real network.
"""
import json
import os
import tempfile
import unittest

from ab.body.providers import MemoryKnowledgeProvider
from ab.body.research import Researcher
from ab.brain import Brain
from ab.core.memory import LongTermMemory

WATER_A = "<p>Water boils at 100 °C.</p>"
WATER_B = "<p>Water boils at 100 degrees C.</p>"
PARIS_A = "<p>Paris is the capital of France.</p>"
PARIS_B = "<p>Paris is the capital city of France.</p>"


def _brain_with_pages(pages, **kwargs):
    return Brain(researcher=Researcher(MemoryKnowledgeProvider(pages), None),
                 granted={"compute", "network"}, **kwargs)


class TestResearchJsonRoundTrip(unittest.TestCase):
    def test_to_json_from_json_preserves_seen_and_quality(self):
        pages = {"https://a.org/w": WATER_A, "https://b.org/w": WATER_B}
        r = Researcher(MemoryKnowledgeProvider(pages), LongTermMemory())
        r.investigate(["https://a.org/w", "https://b.org/w"])
        r.investigate(["https://c.org/w"])  # failed fetch, quality only
        data = r.to_json()
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["provider_spec"]["kind"], "memory")

        r2 = Researcher(MemoryKnowledgeProvider(pages), LongTermMemory())
        r2.from_json(data)
        key = ("water", "boils_at", "100C")
        self.assertIn(key, r2._seen)
        entry = r2._seen[key]
        self.assertEqual(entry["domains"], {"a.org", "b.org"})
        self.assertTrue(entry["admitted"])
        self.assertEqual(len(entry["provenances"]), 2)
        self.assertEqual(entry["claim"].sentence, "Water boils at 100 °C.")
        # W1 fields ride along when present (here: the defensive
        # defaults, since W1 has not landed — the shape is what the
        # contract pins, not the values).
        self.assertIn("extraction_confidence", data["seen"][0]["claim"])
        self.assertIn("negated", data["seen"][0]["claim"])
        # Quality history rides in the same payload.
        self.assertEqual(r2.quality.stats("a.org")["claims_admitted"], 1)
        self.assertEqual(r2.quality.stats("c.org")["fetch_success_rate"], 0.0)
        # A second round-trip is stable.
        self.assertEqual(r2.to_json(), data)

    def test_from_json_rejects_unknown_version(self):
        r = Researcher(MemoryKnowledgeProvider({}), LongTermMemory())
        with self.assertRaises(ValueError):
            r.from_json({"version": 99, "seen": []})

    def test_from_json_with_w1_fields(self):
        # A save written after W1 lands carries extraction_confidence
        # + negated on the claim; from_json restores them onto the
        # Claim object (defensively — it works with or without them).
        r = Researcher(MemoryKnowledgeProvider({}), LongTermMemory())
        r.from_json({"version": 1, "seen": [{
            "claim": {"subject": "paris", "relation": "capital_of",
                      "value": "france",
                      "sentence": "Paris is the capital of France.",
                      "extraction_confidence": 0.95, "negated": True},
            "domains": ["a.org"], "provenances": [], "admitted": False}],
            "quality": {"version": 1, "domains": {}}})
        claim = r._seen[("paris", "capital_of", "france")]["claim"]
        self.assertEqual(claim.extraction_confidence, 0.95)
        self.assertTrue(claim.negated)


class TestV7SaveLoad(unittest.TestCase):
    def test_v7_manifest_and_clean_load(self):
        brain = _brain_with_pages({"https://a.org/p": PARIS_A})
        brain.researcher.investigate(["https://a.org/p"])
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)
        with open(os.path.join(folder, "meta.json")) as fh:
            meta = json.load(fh)
        self.assertEqual(meta["version"], 7)
        self.assertEqual(meta["files"]["research.json"], 1)
        with open(os.path.join(folder, "research.json")) as fh:
            saved = json.load(fh)
        self.assertEqual(saved["version"], 1)
        self.assertEqual(len(saved["seen"]), 1)
        self.assertFalse(saved["seen"][0]["admitted"])

        loaded = _brain_with_pages({"https://a.org/p": PARIS_A,
                                    "https://b.org/p": PARIS_B})
        loaded.load(folder)
        self.assertEqual(loaded.load_warnings, [])
        key = ("paris", "capital_of", "france")
        self.assertIn(key, loaded.researcher._seen)
        entry = loaded.researcher._seen[key]
        self.assertEqual(entry["domains"], {"a.org"})
        self.assertFalse(entry["admitted"])
        self.assertEqual(len(entry["provenances"]), 1)
        self.assertEqual(entry["provenances"][0]["url"], "https://a.org/p")

    def test_pending_claim_survives_save_load(self):
        brain = _brain_with_pages({"https://a.org/p": PARIS_A})
        brain.researcher.investigate(["https://a.org/p"])
        self.assertEqual(len(brain.researcher.pending_claims()), 1)
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)
        loaded = _brain_with_pages({"https://a.org/p": PARIS_A})
        loaded.load(folder)
        pending = loaded.researcher.pending_claims()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["value"], "france")
        self.assertTrue(pending[0]["untrusted"])
        # And the loaded brain still has nothing in LTM for it.
        self.assertEqual(loaded.ltm.query("paris", "capital_of"), [])

    def test_corroboration_spans_restart(self):
        # Save after domain A only (differently-worded same-fact
        # sentences per the mirror rule: "100 °C" vs "100 degrees C").
        brain = _brain_with_pages({"https://a.org/w": WATER_A})
        brain.researcher.investigate(["https://a.org/w"])
        self.assertEqual(brain.ltm.query("water", "boils_at"), [])
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)

        loaded = _brain_with_pages({"https://b.org/w": WATER_B})
        loaded.load(folder)
        self.assertEqual(loaded.ltm.query("water", "boils_at"), [])
        rep = loaded.researcher.investigate(["https://b.org/w"])
        self.assertEqual(rep.summary()["verified"], 1)
        facts = loaded.ltm.query("water", "boils_at", "100C")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].status, "verified")
        self.assertEqual(sorted(facts[0].sources), ["web:a.org", "web:b.org"])
        # The restored _seen entry now carries both domains.
        entry = loaded.researcher._seen[("water", "boils_at", "100C")]
        self.assertEqual(entry["domains"], {"a.org", "b.org"})
        self.assertTrue(entry["admitted"])

    def test_brain_without_researcher_saves_empty_research_state(self):
        brain = Brain(granted={"compute"})
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)
        with open(os.path.join(folder, "research.json")) as fh:
            saved = json.load(fh)
        self.assertEqual(saved, {"version": 1, "seen": [],
                                 "quality": {"version": 1, "domains": {}},
                                 "provider_spec": None})
        loaded = Brain(granted={"compute"})
        loaded.load(folder)
        self.assertEqual(loaded.load_warnings, [])


class TestV6SaveLoadsWithWarning(unittest.TestCase):
    def _v6_folder(self):
        brain = _brain_with_pages({"https://a.org/p": PARIS_A})
        brain.researcher.investigate(["https://a.org/p"])
        brain.teach_fact("mould", "is_a", "tool")
        folder = tempfile.mkdtemp(prefix="ab-phase11-v6-")
        brain.save(folder)
        # Downgrade the save to the v6 shape: version 6, no
        # research.json in the manifest or on disk.
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["version"] = 6
        del meta["files"]["research.json"]
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        os.remove(os.path.join(folder, "research.json"))
        return folder

    def test_v6_save_loads_with_empty_research_state(self):
        loaded = _brain_with_pages({"https://a.org/p": PARIS_A})
        loaded.load(self._v6_folder())
        self.assertEqual(loaded.researcher._seen, {})
        self.assertEqual(loaded.researcher.quality.domains(), [])
        self.assertTrue(any("research.json" in w and "pre-v7" in w
                            for w in loaded.load_warnings))
        # Everything else in the save still loads.
        self.assertTrue(loaded.ask("mould", "is_a").known)

    def test_missing_manifest_listed_research_json_warns(self):
        brain = _brain_with_pages({"https://a.org/p": PARIS_A})
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)
        os.remove(os.path.join(folder, "research.json"))
        loaded = _brain_with_pages({"https://a.org/p": PARIS_A})
        loaded.load(folder)
        self.assertTrue(any("research.json" in w and "missing" in w
                            for w in loaded.load_warnings))
        self.assertEqual(loaded.researcher._seen, {})

    def test_unknown_research_schema_version_raises(self):
        brain = _brain_with_pages({"https://a.org/p": PARIS_A})
        folder = tempfile.mkdtemp(prefix="ab-phase11-v7-")
        brain.save(folder)
        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["files"]["research.json"] = 99
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        loaded = _brain_with_pages({"https://a.org/p": PARIS_A})
        with self.assertRaises(ValueError):
            loaded.load(folder)


if __name__ == "__main__":
    unittest.main()
