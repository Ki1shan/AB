"""Phase 11 §2.4 (worker W2): KnowledgeProvider backends, spec round-trip,
source-quality tracking + selection, Researcher-level disconnect/swap.

All providers here are fakes / in-memory — no real network.
"""
import os
import tempfile
import unittest

from ab.body.providers import (FileKnowledgeProvider, KnowledgeProvider,
                               MemoryKnowledgeProvider, SourceQualityTracker,
                               WebKnowledgeProvider, provider_from_spec)
from ab.body.research import Researcher
from ab.body.web import FetchError, WebFetcher
from ab.core.memory import LongTermMemory


def fake_transport(pages):
    def t(url, timeout, max_bytes):
        if url in pages:
            return 200, "text/html", pages[url]
        return 404, "text/html", ""
    return t


PARIS = "<p>Paris is the capital of France.</p>"
# Phase 11 §4.1 (W1 mirror rule): the second source must state the
# same fact in DIFFERENT words, or the two domains count as a
# verbatim mirror and never corroborate.
PARIS2 = "<p>Paris is the capital city of France.</p>"
WATER_A = "<p>Water boils at 100 °C.</p>"
WATER_B = "<p>Water boils at 100 degrees C.</p>"


class TestWebKnowledgeProvider(unittest.TestCase):
    def test_wraps_web_fetcher(self):
        pages = {"https://a.org/p": PARIS}
        fetcher = WebFetcher(["a.org"], transport=fake_transport(pages))
        prov = WebKnowledgeProvider(fetcher)
        self.assertEqual(prov.kind, "web")
        doc = prov.fetch("https://a.org/p")
        self.assertEqual(doc.url, "https://a.org/p")
        self.assertEqual(doc.domain, "a.org")
        self.assertIn("Paris is the capital of France", doc.text)
        self.assertTrue(doc.untrusted)
        with self.assertRaises(FetchError):
            prov.fetch("https://evil.com/p")

    def test_spec_round_trip(self):
        pages = {"https://a.org/p": PARIS, "https://b.org/p": PARIS}
        prov = WebKnowledgeProvider(
            WebFetcher(["a.org", "b.org"], transport=fake_transport(pages)))
        spec = prov.spec()
        self.assertEqual(spec["kind"], "web")
        self.assertEqual(spec["allowed_domains"], ["a.org", "b.org"])
        rebuilt = provider_from_spec(spec)
        self.assertIsInstance(rebuilt, WebKnowledgeProvider)
        self.assertEqual(rebuilt.spec(), spec)
        # A rebuilt web provider builds a default WebFetcher: the
        # allow-list, not the transport, survives the spec — so a
        # fetch outside the allow-list still fails, from the spec.
        with self.assertRaises(FetchError):
            rebuilt.fetch("https://evil.com/p")

    def test_web_spec_with_no_fetcher_builds_default(self):
        prov = provider_from_spec({"kind": "web", "name": "web",
                                   "allowed_domains": ["a.org"]})
        self.assertIsInstance(prov, WebKnowledgeProvider)
        self.assertEqual(prov.spec()["allowed_domains"], ["a.org"])


class TestFileKnowledgeProvider(unittest.TestCase):
    def _write(self, folder, name, text):
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_fetch_by_bare_name_and_file_url(self):
        folder = tempfile.mkdtemp(prefix="ab-phase11-file-")
        self._write(folder, "notes.txt", PARIS)
        prov = FileKnowledgeProvider(root=folder)
        self.assertEqual(prov.kind, "file")
        for url in ("file:notes.txt", "file://" + os.path.join(folder, "notes.txt")):
            doc = prov.fetch(url)
            self.assertEqual(doc.domain, "file:notes.txt")
            self.assertIn("Paris is the capital of France", doc.text)
            self.assertTrue(doc.untrusted)

    def test_missing_file_raises_and_spec_round_trip(self):
        folder = tempfile.mkdtemp(prefix="ab-phase11-file-")
        self._write(folder, "notes.txt", PARIS)
        prov = FileKnowledgeProvider(root=folder)
        with self.assertRaises(FileNotFoundError):
            prov.fetch("file:gone.txt")
        spec = prov.spec()
        self.assertEqual(spec["kind"], "file")
        self.assertEqual(spec["root"], folder)
        rebuilt = provider_from_spec(spec)
        self.assertIsInstance(rebuilt, FileKnowledgeProvider)
        self.assertEqual(rebuilt.fetch("file:notes.txt").domain, "file:notes.txt")
        self.assertEqual(rebuilt.spec(), spec)

    def test_file_and_web_domains_never_match(self):
        # file:<basename> is its own registrable domain: it can never
        # equal a web registrable domain, so files never corroborate
        # web claims and vice versa (the Researcher sees distinct
        # domains for the same claim from a file and from the web).
        from ab.body.research import registrable_domain
        self.assertNotEqual(registrable_domain("file:notes.txt"),
                            registrable_domain("notes.txt"))
        self.assertEqual(registrable_domain("file:notes.txt"), "file:notes.txt")


class TestMemoryKnowledgeProvider(unittest.TestCase):
    def test_fetch_and_unknown_url(self):
        pages = {"https://a.org/p": PARIS, "book://x": "Water boils at 100 °C."}
        prov = MemoryKnowledgeProvider(pages)
        self.assertEqual(prov.kind, "memory")
        doc = prov.fetch("https://a.org/p")
        self.assertEqual(doc.domain, "a.org")
        self.assertTrue(doc.untrusted)
        self.assertEqual(prov.fetch("book://x").domain, "book://x")
        with self.assertRaises(FetchError):
            prov.fetch("https://a.org/missing")

    def test_spec_round_trip_preserves_pages(self):
        pages = {"https://a.org/p": PARIS}
        prov = MemoryKnowledgeProvider(pages, name="book")
        spec = prov.spec()
        self.assertEqual(spec["kind"], "memory")
        self.assertEqual(spec["pages"], pages)
        rebuilt = provider_from_spec(spec)
        self.assertIsInstance(rebuilt, MemoryKnowledgeProvider)
        self.assertEqual(rebuilt.spec(), spec)
        # The rebuilt provider converts the stored HTML to text, as
        # WebFetcher does for text/html pages.
        self.assertEqual(rebuilt.fetch("https://a.org/p").text,
                         "Paris is the capital of France.")

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            provider_from_spec({"kind": "smoke-signal"})
        with self.assertRaises(ValueError):
            provider_from_spec("not a dict")


class TestResearcherAcceptsProviderOrFetcher(unittest.TestCase):
    def test_old_web_fetcher_positional_still_works(self):
        pages = {"https://a.org/p": PARIS}
        fetcher = WebFetcher(["a.org"], transport=fake_transport(pages))
        r = Researcher(fetcher, LongTermMemory())
        self.assertIsInstance(r.provider, WebKnowledgeProvider)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)

    def test_provider_positional_and_keyword(self):
        pages = {"https://a.org/p": PARIS}
        prov = MemoryKnowledgeProvider(pages)
        r1 = Researcher(prov, LongTermMemory())
        r2 = Researcher(provider=prov, ltm=LongTermMemory())
        self.assertIs(r1.provider, prov)
        self.assertIs(r2.provider, prov)
        self.assertEqual(r1.investigate(["https://a.org/p"]).summary()["untrusted"], 1)

    def test_web_fetcher_as_provider_keyword_is_wrapped(self):
        fetcher = WebFetcher(["a.org"], transport=fake_transport(
            {"https://a.org/p": PARIS}))
        r = Researcher(provider=fetcher, ltm=LongTermMemory())
        self.assertIsInstance(r.provider, WebKnowledgeProvider)
        self.assertEqual(r.investigate(["https://a.org/p"]).summary()["untrusted"], 1)

    def test_non_provider_raises(self):
        with self.assertRaises(TypeError):
            Researcher("not a provider", LongTermMemory())


class TestSourceQualityTracker(unittest.TestCase):
    def _good_bad(self):
        t = SourceQualityTracker()
        for _ in range(10):
            t.record_fetch("good.org", ok=True)
            t.record_claim("good.org")
            t.record_admitted("good.org")
        for _ in range(10):
            t.record_fetch("bad.org", ok=False)
            t.record_claim("bad.org")
        for _ in range(5):
            t.record_disputed("bad.org")
        t.record_fetch("bad.org", ok=True)
        return t

    def test_rates_and_score_ordering(self):
        t = self._good_bad()
        good = t.stats("good.org")
        self.assertEqual(good["fetch_success_rate"], 1.0)
        self.assertEqual(good["corroboration_rate"], 1.0)
        self.assertEqual(good["contradiction_rate"], 0.0)
        bad = t.stats("bad.org")
        self.assertAlmostEqual(bad["fetch_success_rate"], 1 / 11)
        self.assertEqual(bad["corroboration_rate"], 0.0)
        self.assertEqual(bad["contradiction_rate"], 0.5)
        self.assertGreater(t.score("good.org"), t.score("bad.org"))
        # No history is neutral: 0.5, between a good and a bad source.
        self.assertEqual(t.score("never-seen.org"), 0.5)
        self.assertGreater(t.score("good.org"), 0.5)
        self.assertLess(t.score("bad.org"), 0.5)
        self.assertIsNone(t.stats("never-seen.org")["fetch_success_rate"])

    def test_json_round_trip(self):
        t = self._good_bad()
        rebuilt = SourceQualityTracker.from_json(t.to_json())
        self.assertEqual(rebuilt.stats("good.org"), t.stats("good.org"))
        self.assertEqual(rebuilt.stats("bad.org"), t.stats("bad.org"))
        self.assertEqual(rebuilt.score("good.org"), t.score("good.org"))
        with self.assertRaises(ValueError):
            SourceQualityTracker.from_json({"version": 99, "domains": {}})

    def test_quality_hook_feeds_from_investigate(self):
        # Two domains corroborate -> both count as admitted; a failed
        # fetch counts against its domain.
        pages = {"https://a.org/w": WATER_A, "https://b.org/w": WATER_B}
        r = Researcher(MemoryKnowledgeProvider(pages), LongTermMemory())
        r.investigate(["https://a.org/w", "https://b.org/w", "https://c.org/w"])
        for domain in ("a.org", "b.org"):
            st = r.quality.stats(domain)
            self.assertEqual(st["claims_seen"], 1)
            self.assertEqual(st["claims_admitted"], 1)
            self.assertEqual(st["claims_disputed"], 0)
            self.assertEqual(st["fetch_success_rate"], 1.0)
        self.assertEqual(r.quality.stats("c.org")["fetch_success_rate"], 0.0)
        self.assertEqual(r.quality.stats("c.org")["claims_seen"], 0)

    def test_disputed_claim_feeds_contradiction_rate(self):
        # A teacher-verified fact, then a rival claim from TWO domains:
        # LTM's functional-conflict path disputes the rival, and the
        # hook records the contradiction against the rival domains.
        ltm = LongTermMemory()
        ltm.add("water", "boils_at", "100C", source_id="teacher")
        pages = {"https://d.org/w": "<p>Water boils at 90 °C.</p>",
                 "https://e.org/w": "<p>Water boils at 90 degrees C.</p>"}
        r = Researcher(MemoryKnowledgeProvider(pages), ltm)
        rep = r.investigate(["https://d.org/w", "https://e.org/w"])
        self.assertEqual(rep.summary()["disputed"], 1)
        for domain in ("d.org", "e.org"):
            st = r.quality.stats(domain)
            self.assertEqual(st["claims_seen"], 1)
            self.assertEqual(st["claims_disputed"], 1)
            self.assertEqual(st["claims_admitted"], 0)


class TestSelectSources(unittest.TestCase):
    def test_prefers_better_source_on_evidence(self):
        r = Researcher(MemoryKnowledgeProvider({}), LongTermMemory())
        for _ in range(10):
            r.quality.record_fetch("good.org", ok=True)
            r.quality.record_claim("good.org")
            r.quality.record_admitted("good.org")
        for _ in range(10):
            r.quality.record_fetch("bad.org", ok=False)
            r.quality.record_claim("bad.org")
            r.quality.record_disputed("bad.org")
        candidates = ["https://bad.org/p", "https://good.org/p",
                      "https://new.org/p"]
        ranked = r.select_sources("What is the capital of France?", candidates)
        self.assertEqual(ranked, ["https://good.org/p", "https://new.org/p",
                                  "https://bad.org/p"])
        # last_selection: [(url, reason), ...] in ranked order; the
        # reason names the domain and its quality score.
        self.assertEqual([u for u, _ in r.last_selection], ranked)
        reasons = dict(r.last_selection)
        self.assertIn("good.org", reasons["https://good.org/p"])
        self.assertIn("bad.org", reasons["https://bad.org/p"])
        self.assertIn("quality=", reasons["https://good.org/p"])
        good_score = float(reasons["https://good.org/p"]
                           .split("quality=")[1].split()[0])
        bad_score = float(reasons["https://bad.org/p"]
                          .split("quality=")[1].split()[0])
        self.assertGreater(good_score, bad_score)

    def test_tiebreak_is_shorter_url_first_then_deterministic(self):
        r = Researcher(MemoryKnowledgeProvider({}), LongTermMemory())
        candidates = ["https://b.org/longer-path", "https://a.org/x",
                      "https://c.org/x"]
        ranked = r.select_sources("q", candidates)
        # All scores 0.5 (no history): shorter first, then URL order.
        self.assertEqual(ranked, ["https://a.org/x", "https://c.org/x",
                                  "https://b.org/longer-path"])
        self.assertEqual(r.select_sources("q", list(reversed(candidates))),
                         ranked)

    def test_select_sources_does_not_fetch(self):
        r = Researcher(MemoryKnowledgeProvider({}), LongTermMemory())
        self.assertEqual(r.select_sources("q", ["https://a.org/p"]), ["https://a.org/p"])
        self.assertEqual(r.quality.domains(), [])


class TestResearcherDisconnectSwap(unittest.TestCase):
    def _researcher(self):
        pages = {"https://a.org/p": PARIS, "https://b.org/p": PARIS2}
        return Researcher(MemoryKnowledgeProvider(pages), LongTermMemory())

    def test_no_provider_investigate_fails_gracefully(self):
        r = Researcher(provider=None, ltm=LongTermMemory())
        self.assertIsNone(r.provider)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(rep.fetched, [])
        self.assertEqual(rep.results, [])
        self.assertEqual(rep.claims, [])
        self.assertEqual(len(rep.failed), 2)
        for url, error in rep.failed:
            self.assertIn("no_provider", error)
        # Nothing was admitted and nothing leaked into LTM.
        self.assertEqual(r.ltm.query("paris", "capital_of"), [])

    def test_set_provider_none_then_swap_resumes(self):
        r = self._researcher()
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        r.set_provider(None)
        rep = r.investigate(["https://b.org/p"])
        self.assertEqual(len(rep.failed), 1)
        self.assertIn("no_provider", rep.failed[0][1])
        self.assertEqual(r.ltm.query("paris", "capital_of"), [])
        # Swap in a fresh provider: research resumes, and the pending
        # claim from before the disconnect corroborates across it.
        r.set_provider(MemoryKnowledgeProvider(
            {"https://b.org/p": PARIS2}))
        rep = r.investigate(["https://b.org/p"])
        self.assertEqual(rep.summary()["verified"], 1)
        self.assertEqual(r.ltm.query("paris", "capital_of", "france")[0].status,
                         "verified")

    def test_set_provider_accepts_web_fetcher(self):
        r = self._researcher()
        r.set_provider(WebFetcher(
            ["a.org"], transport=fake_transport({"https://a.org/p": PARIS})))
        self.assertIsInstance(r.provider, WebKnowledgeProvider)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)

    def test_file_provider_failure_is_a_failed_fetch_not_a_crash(self):
        r = Researcher(FileKnowledgeProvider(root=tempfile.mkdtemp()),
                       LongTermMemory())
        rep = r.investigate(["file:gone.txt"])
        self.assertEqual(rep.fetched, [])
        self.assertEqual(len(rep.failed), 1)
        self.assertIn("FileNotFoundError", rep.failed[0][1])


if __name__ == "__main__":
    unittest.main()
