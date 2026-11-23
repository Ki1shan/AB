"""Phase 11 §4 (W1): public-suffix-correct registrable_domain and
the mirror rule (corroboration needs independent WORDING, not just
independent domains).

Run: python -m unittest tests.test_phase11_domains -v
"""
import unittest

from ab.body.research import (PUBLIC_SUFFIXES, ClaimExtractor, Researcher,
                              is_independent_corroboration,
                              registrable_domain, sentence_fingerprint)
from ab.body.web import WebFetcher
from ab.core.memory import LongTermMemory


def fake_transport(pages):
    def t(url, timeout, max_bytes):
        if url in pages:
            return 200, "text/html", pages[url]
        return 404, "text/html", ""
    return t


def researcher_for(pages, ltm=None):
    domains = sorted({u.split("/")[2].split(":")[0] for u in pages})
    fetcher = WebFetcher(domains, transport=fake_transport(pages))
    return Researcher(fetcher, ltm if ltm is not None else LongTermMemory())


class TestRegistrableDomainPSL(unittest.TestCase):
    def test_every_table_entry(self):
        # Every curated suffix, in three shapes: bare, one label
        # before it, two labels before it. The table is curated, not
        # the full PSL — the test enumerates the table itself, so
        # adding a suffix adds its cases automatically.
        self.assertEqual(len(PUBLIC_SUFFIXES), 14)
        for suffix in sorted(PUBLIC_SUFFIXES):
            with self.subTest(suffix=suffix):
                self.assertEqual(registrable_domain(f"example.{suffix}"),
                                 f"example.{suffix}")
                self.assertEqual(registrable_domain(f"www.example.{suffix}"),
                                 f"example.{suffix}")
                self.assertEqual(registrable_domain(f"a.b.{suffix}"),
                                 f"b.{suffix}")

    def test_design_examples(self):
        self.assertEqual(registrable_domain("a.b.co.uk"), "b.co.uk")
        self.assertEqual(registrable_domain("foo.bar.com.au"), "bar.com.au")
        self.assertEqual(registrable_domain("sub.alpha.org"), "alpha.org")

    def test_fallback_and_old_cases(self):
        self.assertEqual(registrable_domain("www.alpha.org"), "alpha.org")
        self.assertEqual(registrable_domain("alpha.org"), "alpha.org")
        self.assertEqual(registrable_domain("a.b.example.com"), "example.com")
        self.assertEqual(registrable_domain("localhost"), "localhost")
        self.assertEqual(registrable_domain(""), "")
        # Bare suffix itself, and a suffix used as a registrable
        # domain (e.g. a GitHub Pages site), return as-is.
        self.assertEqual(registrable_domain("co.uk"), "co.uk")
        self.assertEqual(registrable_domain("github.io"), "github.io")
        self.assertEqual(registrable_domain("a.github.io"), "a.github.io")
        self.assertEqual(registrable_domain("x.y.github.io"), "y.github.io")
        # Case and trailing dot are normalized; suffix matching is
        # label-boundary aware ("notco.uk" is not "co.uk").
        self.assertEqual(registrable_domain("WWW.ALPHA.ORG."), "alpha.org")
        self.assertEqual(registrable_domain("a.b.notco.uk"), "notco.uk")
        self.assertEqual(registrable_domain("a.b.co.uk."), "b.co.uk")

    def test_co_uk_sources_are_one_domain_for_corroboration(self):
        # The point of the table: two hosts under the same co.uk
        # registrable domain do NOT corroborate each other.
        pages = {"https://www.example.co.uk/p": "<p>Paris is the capital of France.</p>",
                 "https://news.example.co.uk/p": "<p>Paris is the capital city of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://www.example.co.uk/p", "https://news.example.co.uk/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertFalse(r.is_corroborated("paris", "capital_of", "france"))
        # ...while two DIFFERENT co.uk registrable domains do.
        pages2 = {"https://www.example.co.uk/p": "<p>Paris is the capital of France.</p>",
                  "https://www.other.co.uk/p": "<p>Paris is the capital city of France.</p>"}
        ltm2 = LongTermMemory()
        r2 = researcher_for(pages2, ltm2)
        rep2 = r2.investigate(["https://www.example.co.uk/p", "https://www.other.co.uk/p"])
        self.assertEqual(rep2.summary()["verified"], 1)
        self.assertEqual(ltm2.query("paris", "capital_of", "france")[0].status, "verified")


class TestSentenceFingerprint(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(sentence_fingerprint("Paris is the capital of France."),
                         "paris is the capital of france")
        self.assertEqual(sentence_fingerprint("  Paris   is the capital of France! "),
                         "paris is the capital of france")
        self.assertEqual(sentence_fingerprint("Water boils at 100 \u00b0C."),
                         "water boils at 100 \u00b0c")
        self.assertNotEqual(sentence_fingerprint("Water boils at 100 \u00b0C."),
                            sentence_fingerprint("Water boils at 100 degrees C."))
        self.assertEqual(sentence_fingerprint(""), "")

    def test_predicate_on_synthetic_entries(self):
        two_domains_two_wordings = {
            "domains": {"a.org", "b.org"},
            "provenances": [{"sentence": "Paris is the capital of France."},
                            {"sentence": "Paris is the capital city of France."}]}
        self.assertTrue(is_independent_corroboration(two_domains_two_wordings))
        mirror = {
            "domains": {"a.org", "b.org"},
            "provenances": [{"sentence": "Paris is the capital of France."},
                            {"sentence": "Paris is the capital of France."}]}
        self.assertFalse(is_independent_corroboration(mirror))
        # Same wording up to case / whitespace / trailing punctuation
        # is still the same wording.
        mirror_noisy = {
            "domains": {"a.org", "b.org"},
            "provenances": [{"sentence": "Paris is the capital of France."},
                            {"sentence": "  paris is the capital of france!"}]}
        self.assertFalse(is_independent_corroboration(mirror_noisy))
        single_domain = {
            "domains": {"a.org"},
            "provenances": [{"sentence": "Paris is the capital of France."},
                            {"sentence": "Paris is the capital city of France."}]}
        self.assertFalse(is_independent_corroboration(single_domain))
        self.assertFalse(is_independent_corroboration({"domains": set(), "provenances": []}))


class TestMirrorRule(unittest.TestCase):
    def test_same_sentence_two_domains_is_a_mirror_not_corroboration(self):
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
                 "https://b.org/p": "<p>Paris is the capital of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        # The claim stays held: untrusted, never admitted.
        self.assertEqual(rep.summary()["verified"], 0)
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(rep.summary()["mirror_held"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertFalse(r.is_corroborated("paris", "capital_of", "france"))
        # The reason is recorded on the report AND flagged on every
        # provenance dict of the claim.
        self.assertEqual(len(rep.claims), 2)
        for prov in rep.claims:
            self.assertTrue(prov["mirror_suspected"])
            self.assertEqual(prov["status"], "untrusted")
        # Held, not lost: the claim is still pending outside LTM.
        self.assertEqual(len(r.pending_claims()), 2)

    def test_different_sentences_two_domains_still_corroborate(self):
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
                 "https://b.org/p": "<p>Paris is the capital city of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(rep.summary()["verified"], 1)
        self.assertEqual(rep.summary()["mirror_held"], 0)
        self.assertEqual(ltm.query("paris", "capital_of", "france")[0].status, "verified")
        self.assertTrue(r.is_corroborated("paris", "capital_of", "france"))
        for prov in rep.claims:
            self.assertFalse(prov["mirror_suspected"])

    def test_third_domain_with_new_wording_rescues_a_mirror(self):
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
                 "https://b.org/p": "<p>Paris is the capital of France.</p>",
                 "https://c.org/p": "<p>Paris is the capital city of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep1 = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(rep1.summary()["mirror_held"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        rep2 = r.investigate(["https://c.org/p"])
        self.assertEqual(rep2.summary()["verified"], 1)
        self.assertEqual(rep2.summary()["mirror_held"], 0)
        self.assertEqual(ltm.query("paris", "capital_of", "france")[0].status, "verified")
        # Admission settles the mirror question on the provenances.
        entry = r._seen[("paris", "capital_of", "france")]
        self.assertTrue(all(not p["mirror_suspected"] for p in entry["provenances"]))
        self.assertTrue(all(p["corroborated"] for p in entry["provenances"]))
        self.assertEqual(r.pending_claims(), [])

    def test_one_domain_two_wordings_is_not_a_mirror_just_single_domain(self):
        pages = {"https://a.org/p1": "<p>Paris is the capital of France.</p>",
                 "https://a.org/p2": "<p>Paris is the capital city of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p1", "https://a.org/p2"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(rep.summary()["mirror_held"], 0)
        self.assertEqual(ltm.query("paris", "capital_of"), [])

    def test_subdomains_with_different_wordings_still_do_not_corroborate(self):
        pages = {"https://www.alpha.org/p": "<p>Water boils at 100 \u00b0C.</p>",
                 "https://news.alpha.org/p": "<p>Water boils at 100 degrees C.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://www.alpha.org/p", "https://news.alpha.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(rep.summary()["mirror_held"], 0)
        self.assertFalse(r.is_corroborated("water", "boils_at", "100C"))


class TestFixtureBehaviour(unittest.TestCase):
    """The §4.1 fixture behaviour: the demo and test_brain.py wordings
    (second source "capital city") corroborate; the old identical
    wordings do not."""

    def test_demo_wording_corroborates(self):
        pages = {"https://alpha.org/f": "<p>Paris is the capital of France.</p>",
                 "https://beta.org/f": "<html><script>x()</script><p>Paris is the capital city of France.</p></html>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://alpha.org/f", "https://beta.org/f"])
        self.assertEqual(rep.summary()["verified"], 1)
        self.assertEqual(ltm.query("paris", "capital_of", "france")[0].status, "verified")

    def test_old_identical_wording_would_be_a_mirror(self):
        pages = {"https://alpha.org/f": "<p>Paris is the capital of France.</p>",
                 "https://beta.org/f": "<p>Paris is the capital of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://alpha.org/f", "https://beta.org/f"])
        self.assertEqual(rep.summary()["verified"], 0)
        self.assertEqual(rep.summary()["mirror_held"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])

    def test_test_body_corroboration_wordings_unchanged(self):
        # test_body.py already used different sentences and needs NO
        # change — verify the behaviour it relies on still holds.
        pages = {"https://a.org/w": "<p>Water boils at 100 \u00b0C.</p>",
                 "https://b.org/w": "<p>Water boils at 100 degrees C.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/w", "https://b.org/w"])
        self.assertEqual(rep.summary()["verified"], 1)
        self.assertEqual(rep.summary()["mirror_held"], 0)


if __name__ == "__main__":
    unittest.main()
