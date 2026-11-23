"""Phase 3 §3.2 (research untrusted path) and §3.3 (OCR gate).

All web/OCR sources here are fakes — no real network, no real tesseract.
"""
import unittest
from datetime import datetime

from ab.body.ocr import FakeWordOcr, OcrSensor, StubOcr, words_above_threshold
from ab.body.research import ClaimExtractor, Researcher, registrable_domain
from ab.body.web import WebFetcher
from ab.core.memory import TRUST, LongTermMemory
from ab.reasoning.inference import Inference


def fake_transport(pages):
    def t(url, timeout, max_bytes):
        if url in pages:
            return 200, "text/html", pages[url]
        return 404, "text/html", ""
    return t


PARIS = "<p>Paris is the capital of France.</p>"
# Phase 11 §4.1 (W1, §9 precedent): the differently-worded same fact
# for the second domain — the identical sentence from both domains is
# a mirror under the mirror rule and no longer corroborates.
PARIS2 = "<p>Paris is the capital city of France.</p>"


def researcher_for(pages, ltm=None):
    domains = sorted({u.split("/")[2].split(":")[0] for u in pages})
    fetcher = WebFetcher(domains, transport=fake_transport(pages))
    return Researcher(fetcher, ltm if ltm is not None else LongTermMemory())


class TestRegistrableDomain(unittest.TestCase):
    def test_registrable_domain(self):
        self.assertEqual(registrable_domain("sub.alpha.org"), "alpha.org")
        self.assertEqual(registrable_domain("www.alpha.org"), "alpha.org")
        self.assertEqual(registrable_domain("alpha.org"), "alpha.org")
        self.assertEqual(registrable_domain("a.b.example.com"), "example.com")
        self.assertEqual(registrable_domain("localhost"), "localhost")


class TestResearchUntrustedPath(unittest.TestCase):
    def test_single_domain_claim_untrusted_and_not_in_ltm(self):
        ltm = LongTermMemory()
        r = researcher_for({"https://a.org/p": PARIS}, ltm)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(rep.summary()["verified"], 0)
        claim, res, status, url = rep.results[0]
        self.assertEqual(status, "untrusted")
        self.assertIsNone(res)
        self.assertEqual(url, "https://a.org/p")
        self.assertEqual(rep.claims[0]["untrusted"], True)
        # Held as pending, outside LTM, for the core worker to surface.
        self.assertEqual(len(r.pending_claims()), 1)

    def test_two_independent_domains_corroborated(self):
        ltm = LongTermMemory()
        # Phase 11 §4.1 (W1, §9 precedent): the second domain words
        # the fact differently (PARIS2) — identical sentences are a
        # mirror and no longer corroborate. Same intent, new semantics.
        r = researcher_for({"https://a.org/p": PARIS, "https://b.org/p": PARIS2}, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(rep.summary()["verified"], 1)
        self.assertEqual(rep.summary()["untrusted"], 0)
        facts = ltm.query("paris", "capital_of", "france")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].status, "verified")
        self.assertEqual(sorted(facts[0].sources), ["web:a.org", "web:b.org"])
        self.assertTrue(r.is_corroborated("paris", "capital_of", "france"))
        self.assertEqual(r.pending_claims(), [])
        # Corroboration also spans separate investigate() calls.
        ltm2 = LongTermMemory()
        r2 = researcher_for({"https://a.org/p": PARIS, "https://b.org/p": PARIS2}, ltm2)
        r2.investigate(["https://a.org/p"])
        self.assertEqual(ltm2.query("paris", "capital_of"), [])
        rep2 = r2.investigate(["https://b.org/p"])
        self.assertEqual(rep2.summary()["verified"], 1)
        self.assertEqual(ltm2.query("paris", "capital_of", "france")[0].status, "verified")

    def test_two_subdomains_of_same_registrable_domain_do_not_corroborate(self):
        ltm = LongTermMemory()
        pages = {"https://www.alpha.org/p": PARIS, "https://news.alpha.org/p": PARIS}
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://www.alpha.org/p", "https://news.alpha.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertFalse(r.is_corroborated("paris", "capital_of", "france"))
        grouped = r.claims_for("paris", "capital_of")
        self.assertEqual(set(grouped), {"alpha.org"})
        self.assertEqual(len(grouped["alpha.org"]), 2)

    def test_teacher_source_admits_alone(self):
        ltm = LongTermMemory()
        res, fact = ltm.add("paris", "capital_of", "france", source_id="teacher")
        self.assertEqual(fact.status, "verified")

    def test_approve_claim_admits(self):
        ltm = LongTermMemory()
        status = ltm.approve_claim("paris", "capital_of", "france")
        self.assertEqual(status, "verified")
        self.assertEqual(ltm.ask("paris", "capital_of").best[2], "verified")
        # Explicit approval is the only path that admits a single claim:
        # research itself never calls approve_claim (a single-domain fetch
        # stays untrusted even though approve_claim exists).
        r = researcher_for({"https://a.org/p": PARIS}, ltm)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.results[0][2], "verified")  # already admitted
        self.assertEqual(len(ltm.query("paris", "capital_of", "france")), 1)

    def test_full_provenance_present_on_every_claim(self):
        ltm = LongTermMemory()
        r = researcher_for({"https://a.org/p": PARIS, "https://b.org/p": PARIS}, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(len(rep.claims), 2)
        for prov in rep.claims:
            for key in ("url", "sentence", "fetched_at", "confidence",
                        "domain", "registrable_domain"):
                self.assertIn(key, prov, f"missing provenance key {key}")
            self.assertEqual(prov["sentence"], "Paris is the capital of France.")
            self.assertIsInstance(prov["confidence"], float)
            self.assertEqual(prov["domain"], prov["url"].split("/")[2])
            self.assertEqual(prov["registrable_domain"], prov["domain"])
            datetime.fromisoformat(prov["fetched_at"])  # parses as ISO
        self.assertEqual({p["url"] for p in rep.claims},
                         {"https://a.org/p", "https://b.org/p"})

    def test_claims_for_grouped_by_registrable_domain(self):
        ltm = LongTermMemory()
        r = researcher_for({"https://a.org/p": PARIS, "https://b.org/p": PARIS}, ltm)
        r.investigate(["https://a.org/p", "https://b.org/p"])
        grouped = r.claims_for("paris", "capital_of")
        self.assertEqual(set(grouped), {"a.org", "b.org"})
        self.assertEqual(grouped["a.org"][0]["url"], "https://a.org/p")

    def test_untrusted_single_domain_fact_drives_no_derivation(self):
        # Control: teacher-taught verified facts DO derive.
        ltm = LongTermMemory()
        ltm.add("cat", "is_a", "mammal", source_id="teacher")
        ltm.add("mammal", "is_a", "animal", source_id="teacher")
        inf = Inference(ltm)
        self.assertIsNotNone(inf.prove("cat", "is_a", "animal"))
        # A single-domain web claim is not in LTM, so it drives nothing.
        r = researcher_for({"https://a.org/p": "<p>Fluffy is a cat.</p>"}, ltm)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.results[0][2], "untrusted")
        self.assertEqual(ltm.query("fluffy", "is_a"), [])
        self.assertIsNone(inf.prove("fluffy", "is_a", "mammal"))
        self.assertEqual(inf.closure("fluffy", "is_a"), [])
        # The same claim corroborated by a second domain DOES derive.
        # Phase 11 §4.1 (W1, §9 precedent): the second domain words
        # the same claim differently (appositive form) — identical
        # sentences are a mirror and no longer corroborate.
        r2 = researcher_for({"https://a.org/p": "<p>Fluffy is a cat.</p>",
                             "https://b.org/p": "<p>Fluffy, a cat, sleeps all day.</p>"}, ltm)
        r2.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertIsNotNone(inf.prove("fluffy", "is_a", "mammal"))


class TestOcrGate(unittest.TestCase):
    def test_low_confidence_word_never_admitted_recorded_uncertain(self):
        engine = FakeWordOcr({"img.png": [
            ("Paris", 0.95), ("is", 0.92), ("the", 0.90),
            ("capital", 0.20, {"left": 10, "top": 5, "width": 40, "height": 12}),
            ("of", 0.93), ("France", 0.96)]})
        sensor = OcrSensor(engine)  # default threshold 0.6
        gated = sensor.gate("img.png")
        admitted_texts = [w["text"] for w in gated["admitted"]]
        self.assertNotIn("capital", admitted_texts)
        self.assertEqual([w["text"] for w in gated["uncertain"]], ["capital"])
        uncertain = gated["uncertain"][0]
        self.assertEqual(uncertain["confidence"], 0.20)
        self.assertEqual(uncertain["bbox"], {"left": 10, "top": 5, "width": 40, "height": 12})
        self.assertEqual(uncertain["image"], "img.png")
        # The admitted text alone yields no capital_of claim: the
        # low-confidence word never becomes a claim.
        claims = ClaimExtractor().extract(gated["admitted_text"])
        self.assertEqual(claims, [])

    def test_words_above_threshold_default_and_custom(self):
        words = [{"text": "a", "confidence": 0.9, "bbox": None, "image": "i.png"},
                 {"text": "b", "confidence": 0.6, "bbox": None, "image": "i.png"},
                 {"text": "c", "confidence": 0.59, "bbox": None, "image": "i.png"}]
        admitted, uncertain = words_above_threshold(words)
        self.assertEqual([w["text"] for w in admitted], ["a", "b"])  # 0.6 admits
        self.assertEqual([w["text"] for w in uncertain], ["c"])
        admitted, uncertain = words_above_threshold(words, threshold=0.9)
        self.assertEqual([w["text"] for w in admitted], ["a"])
        self.assertEqual([w["text"] for w in uncertain], ["b", "c"])

    def test_ocr_read_words_per_word_dicts(self):
        sensor = OcrSensor(StubOcr({"p.png": "Mars is a planet."}, confidence=0.9))
        words = sensor.read_words("p.png")
        self.assertEqual([w["text"] for w in words], ["Mars", "is", "a", "planet."])
        for w in words:
            self.assertEqual(set(w), {"text", "confidence", "bbox", "image"})
            self.assertEqual(w["confidence"], 0.9)
            self.assertIsNone(w["bbox"])
            self.assertEqual(w["image"], "p.png")

    def test_ocr_alone_never_verifies_a_fact(self):
        self.assertEqual(TRUST["ocr"], 0.5)
        ltm = LongTermMemory()
        res, fact = ltm.add("mars", "is_a", "planet", source_id="ocr:img.png")
        self.assertEqual(fact.status, "unverified")
        self.assertEqual(ltm.ask("mars", "is_a").best[2], "unverified")
        # Corroboration rules still apply: a second, independent source
        # (here a teacher) is what verifies it — OCR alone never does.
        ltm.add("mars", "is_a", "planet", source_id="teacher")
        self.assertEqual(ltm.query("mars", "is_a", "planet")[0].status, "verified")


if __name__ == "__main__":
    unittest.main()
