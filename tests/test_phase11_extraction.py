"""Phase 11 §3 (W1): claim extraction beyond 5 patterns, negation,
per-pattern extraction confidence, teach_pattern validation, and
the held-out extraction measurement.

Run: python -m unittest tests.test_phase11_extraction -v
"""
import unittest

from ab.body.research import (ClaimExtractor, Researcher,
                              _provenance, registrable_domain)
from ab.body.web import WebFetcher
from ab.core.memory import LongTermMemory
from ab.eval.extraction_set import (EXTRACTION_SET, build_extraction_set,
                                    five_pattern_extractor,
                                    measure_extraction, run_measurement)


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


def one(sentence, extractor=None):
    claims = (extractor or ClaimExtractor()).extract(sentence)
    assert len(claims) == 1, f"expected 1 claim from {sentence!r}, got {claims}"
    return claims[0]


class TestNewPatterns(unittest.TestCase):
    def test_founded_in(self):
        c = one("Microsoft was founded in 1975.")
        self.assertEqual((c.subject, c.relation, c.value), ("microsoft", "founded_in", "1975"))
        self.assertFalse(c.negated)
        # Year only: a trailing place defeats the formulaic shape.
        self.assertEqual(ClaimExtractor().extract("Microsoft was founded in 1975 in California."), [])

    def test_population_of_both_forms(self):
        c = one("Tokyo has a population of 14 million.")
        self.assertEqual((c.subject, c.relation, c.value), ("tokyo", "population_of", "14 million"))
        c = one("The population of Paris is 2 million.")
        self.assertEqual((c.subject, c.relation, c.value), ("paris", "population_of", "2 million"))

    def test_located_in_lies_in(self):
        c = one("The Thames lies in England.")
        self.assertEqual((c.subject, c.relation, c.value), ("thames", "located_in", "england"))
        c = one("Oxford is situated in England.")
        self.assertEqual((c.subject, c.relation, c.value), ("oxford", "located_in", "england"))

    def test_larger_than_is_a_new_non_functional_relation(self):
        c = one("Russia is larger than France.")
        self.assertEqual((c.subject, c.relation, c.value), ("russia", "larger_than", "france"))
        c = one("The lakes are larger than the ponds.")
        self.assertEqual((c.subject, c.relation, c.value), ("lakes", "larger_than", "ponds"))
        # NOT functional: no dispute machinery engages — two rival
        # larger_than values coexist as ordinary facts.
        from ab.core.vocab import FUNCTIONAL_RELATIONS
        self.assertNotIn("larger_than", FUNCTIONAL_RELATIONS)

    def test_appositive_is_a_single_word_only(self):
        c = one("Paris, a city, is the capital of France.")
        self.assertEqual((c.subject, c.relation, c.value), ("paris", "is_a", "city"))
        c = one("Mars, a planet, orbits the Sun.")
        self.assertEqual((c.subject, c.relation, c.value), ("mars", "is_a", "planet"))
        # Multi-word appositives yield NO claim rather than a wrong one.
        self.assertEqual(ClaimExtractor().extract("Paris, a city in France, is beautiful."), [])
        self.assertEqual(ClaimExtractor().extract("In 1990, a year of change, the wall had fallen."), [])
        # The appositive never produces a dirty subject for the
        # specific patterns: the capital claim is not also extracted
        # with subject "paris, a city".
        claims = ClaimExtractor().extract("Paris, a city, is the capital of France.")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].relation, "is_a")

    def test_old_patterns_unchanged(self):
        claims = ClaimExtractor().extract(
            "Paris is the capital of France. Water boils at 100 \u00b0C at sea level. "
            "Mars is a planet. Click here to buy now!")
        self.assertEqual({(x.subject, x.relation, x.value) for x in claims},
                         {("paris", "capital_of", "france"), ("water", "boils_at", "100C"),
                          ("mars", "is_a", "planet")})

    def test_question_yields_no_claim_and_cap_stays(self):
        # Held-out measurement caught this: a question matches the
        # capital_of shape with subject "what". Questions assert
        # nothing, so they yield no claim.
        self.assertEqual(ClaimExtractor().extract("What is the capital of France?"), [])
        self.assertEqual(ClaimExtractor().extract("Is Paris the capital of France?"), [])
        long_sent = "Paris is the capital of France" + " and it is grand" * 40 + "."
        self.assertGreater(len(long_sent), 300)
        self.assertEqual(ClaimExtractor().extract(long_sent), [])


class TestNegation(unittest.TestCase):
    NEGATED = [
        ("Paris is not the capital of Germany.", ("paris", "capital_of", "germany")),
        ("London is not the capital of France.", ("london", "capital_of", "france")),
        ("Paris is not the capital city of Germany.", ("paris", "capital_of", "germany")),
        ("Oxford is not located in France.", ("oxford", "located_in", "france")),
        ("The Nile does not lie in Asia.", ("nile", "located_in", "asia")),
        ("Water does not boil at 90 \u00b0C.", ("water", "boils_at", "90C")),
        ("Ice does not melt at 100 degrees C.", ("ice", "melts_at", "100C")),
        ("The Moon is not a planet.", ("moon", "is_a", "planet")),
        ("Rome was not founded in 1900.", ("rome", "founded_in", "1900")),
        ("Tokyo does not have a population of 2 million.", ("tokyo", "population_of", "2 million")),
        ("The population of London is not 9 million.", ("london", "population_of", "9 million")),
        ("Russia is not larger than Canada.", ("russia", "larger_than", "canada")),
    ]

    def test_negated_sentences_yield_negated_claims_only(self):
        for sent, (s, r, v) in self.NEGATED:
            with self.subTest(sentence=sent):
                claims = ClaimExtractor().extract(sent)
                self.assertEqual(len(claims), 1)
                c = claims[0]
                self.assertTrue(c.negated)
                self.assertEqual((c.subject, c.relation, c.value), (s, r, v))

    def test_negated_claims_are_dropped_and_counted(self):
        pages = {"https://a.org/p": "<p>Paris is not the capital of Germany.</p>",
                 "https://b.org/p": "<p>Paris is not the capital of Germany.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        # Even from TWO domains, a negated claim is never admitted.
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertEqual(rep.summary()["ignored_negated"], 2)
        self.assertEqual(rep.results, [])
        self.assertEqual(r._seen, {})
        self.assertEqual(r.pending_claims(), [])
        # Provenance is kept (audit trail), marked ignored_negated.
        self.assertEqual(len(rep.claims), 2)
        for prov in rep.claims:
            self.assertTrue(prov["negated"])
            self.assertEqual(prov["status"], "ignored_negated")

    def test_negated_does_not_block_or_pose_as_the_positive_claim(self):
        # Same key, opposite polarity: the positive claim (one domain)
        # stays held untrusted; the negated one is counted, not held.
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
                 "https://b.org/p": "<p>Paris is not the capital of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p", "https://b.org/p"])
        self.assertEqual(rep.summary()["ignored_negated"], 1)
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertEqual(len(r.pending_claims()), 1)
        self.assertFalse(r.pending_claims()[0]["negated"])

    def test_negated_claim_never_becomes_untrusted_state(self):
        # Brain-level guard: the Researcher's "ignored_negated"
        # provenance status keeps _research_tool from surfacing the
        # negated claim as an untrusted:paris|capital_of entry.
        from ab.brain import Brain
        pages = {"https://a.org/p": "<p>Paris is not the capital of France.</p>"}
        fetcher = WebFetcher(["a.org"], transport=fake_transport(pages))
        b = Brain(researcher=Researcher(fetcher, None), granted={"compute", "network"})
        out = b._research_tool(b.researcher.investigate(["https://a.org/p"], b.cycle))
        self.assertEqual(out["summary"]["ignored_negated"], 1)
        self.assertIsNone(b.state.get("untrusted:paris|capital_of"))
        self.assertEqual(b.ltm.query("paris", "capital_of"), [])


class TestExtractionConfidence(unittest.TestCase):
    EXPECTED = {
        ("Paris is the capital of France.", "capital_of"): 0.95,
        ("Water boils at 100 \u00b0C.", "boils_at"): 0.95,
        ("Ice melts at 0 degrees C.", "melts_at"): 0.95,
        ("Oxford is located in England.", "located_in"): 0.9,
        ("The Thames lies in England.", "located_in"): 0.9,
        ("Mars is a planet.", "is_a"): 0.8,
        ("Paris, a city, is the capital of France.", "is_a"): 0.8,
        ("Microsoft was founded in 1975.", "founded_in"): 0.95,
        ("Tokyo has a population of 14 million.", "population_of"): 0.75,
        ("Russia is larger than France.", "larger_than"): 0.85,
        ("Paris is not the capital of Germany.", "capital_of"): 0.95,
    }

    def test_per_pattern_confidence(self):
        for sent, rel in self.EXPECTED:
            with self.subTest(sentence=sent):
                c = one(sent)
                self.assertEqual(c.relation, rel)
                self.assertEqual(c.extraction_confidence, self.EXPECTED[(sent, rel)])

    def test_provenance_carries_extraction_confidence(self):
        c = one("Tokyo has a population of 14 million.")
        prov = _provenance(c, "https://a.org/p", "a.org", "2026-11-22T00:00:00+00:00")
        self.assertEqual(prov["extraction_confidence"], 0.75)
        self.assertEqual(prov["confidence"], 0.45)  # source trust, unchanged
        self.assertFalse(prov["negated"])
        self.assertFalse(prov["mirror_suspected"])

    def test_extraction_confidence_never_verifies_alone(self):
        # A single-domain claim at 0.95 extraction confidence stays
        # untrusted: admission still uses source trust (0.45), and
        # one domain is never enough however confident the match.
        pages = {"https://a.org/p": "<p>Paris is the capital of France.</p>"}
        ltm = LongTermMemory()
        r = researcher_for(pages, ltm)
        rep = r.investigate(["https://a.org/p"])
        self.assertEqual(rep.summary()["untrusted"], 1)
        self.assertEqual(rep.summary()["verified"], 0)
        self.assertEqual(ltm.query("paris", "capital_of"), [])
        self.assertEqual(rep.claims[0]["extraction_confidence"], 0.95)
        self.assertEqual(rep.claims[0]["confidence"], 0.45)


class TestTeachPatternValidation(unittest.TestCase):
    def test_missing_groups_rejected(self):
        e = ClaimExtractor()
        for bad in [r"^(?P<s>.+?) likes stuff$",
                    r"^(?P<v>.+?) is liked$",
                    r"^(.+?) likes (.+?)$"]:
            with self.subTest(regex=bad), self.assertRaises(ValueError):
                e.add_pattern("likes", bad)
        self.assertEqual(len(e.patterns), len(ClaimExtractor().patterns))

    def test_uncompilable_and_empty_rejected(self):
        e = ClaimExtractor()
        with self.assertRaises(ValueError):
            e.add_pattern("likes", r"^(?P<s>.+?) likes (?P<v>$")
        with self.assertRaises(ValueError):
            e.add_pattern("likes", "")
        with self.assertRaises(ValueError):
            e.add_pattern("", r"^(?P<s>.+?) likes (?P<v>.+?)$")

    def test_redos_nested_quantifier_rejected(self):
        e = ClaimExtractor()
        for bad in [r"^(?P<s>(.+)+) likes (?P<v>.+?)$",
                    r"^(?P<s>(\w*)*) likes (?P<v>.+?)$",
                    r"^(?P<s>.+?) likes (?P<v>(.+?)*)$"]:
            with self.subTest(regex=bad), self.assertRaises(ValueError):
                e.add_pattern("likes", bad)
        # The guard is a heuristic: safe shapes it must NOT flag.
        e.add_pattern("likes", r"^(?P<s>.+?) likes (?P<v>-?\d+(?:\.\d+)?)\.?$")
        self.assertEqual(e.extract("Tom likes 2.5.")[0].value, "2.5")

    def test_too_long_rejected(self):
        e = ClaimExtractor()
        with self.assertRaises(ValueError):
            e.add_pattern("likes", "x" * 501 + r"^(?P<s>.+?) likes (?P<v>.+?)$")

    def test_confidence_validated(self):
        e = ClaimExtractor()
        for bad in (0.0, -0.5, 1.5, "high"):
            with self.subTest(confidence=bad), self.assertRaises(ValueError):
                e.add_pattern("likes", r"^(?P<s>.+?) likes (?P<v>.+?)$", confidence=bad)
        e.add_pattern("likes", r"^(?P<s>.+?) likes (?P<v>.+?)$", confidence=0.6)
        self.assertEqual(e.extract("Tom likes tea.")[0].extraction_confidence, 0.6)
        # Taught patterns default to the loose-pattern level (0.8).
        e.add_pattern("hates", r"^(?P<s>.+?) hates (?P<v>.+?)$")
        self.assertEqual(e.extract("Tom hates rain.")[0].extraction_confidence, 0.8)

    def test_taught_pattern_works_and_can_be_removed(self):
        e = ClaimExtractor()
        e.add_pattern("likes", r"^(?P<s>.+?) likes (?P<v>.+?)$")
        self.assertEqual(e.extract("Tom likes tea.")[0].relation, "likes")
        self.assertEqual(e.remove_pattern("likes"), 1)
        self.assertEqual(e.extract("Tom likes tea."), [])
        self.assertEqual(e.remove_pattern("likes"), 0)
        # Removing a built-in relation removes its patterns (both
        # positive and negated forms) and only that relation.
        self.assertEqual(e.remove_pattern("capital_of"), 2)
        self.assertEqual(e.extract("Paris is the capital of France."), [])
        self.assertEqual(e.extract("Paris is not the capital of France."), [])
        self.assertEqual(e.extract("Mars is a planet.")[0].relation, "is_a")


class TestHeldOutMeasurement(unittest.TestCase):
    def test_set_is_deterministic_and_fixed_size(self):
        self.assertEqual(build_extraction_set(), build_extraction_set())
        self.assertEqual(build_extraction_set(seed=11), EXTRACTION_SET)
        self.assertEqual(len(EXTRACTION_SET), 84)
        # No eval-task invented names in the set (leakage audit stays
        # clean): ordinary nouns / real-world facts only.
        forbidden = ("zorblax", "velkra", "torvin", "sarnak", "kelmor",
                     "lyssan", "fluffy", "atlantis")
        for sentence, _gold in EXTRACTION_SET:
            for name in forbidden:
                self.assertNotIn(name, sentence.lower(), sentence)

    def test_full_extractor_measured_numbers(self):
        # Measured 2026-11-22 (W1): full extractor 73/73 gold, 0 FP,
        # 0 FN on this 84-item set. The set is construction-matched
        # to the pattern shapes, so 1.0/1.0 is a coverage statement
        # about the set, not about arbitrary text — the meaningful
        # number is the delta vs baseline below.
        res = run_measurement()
        full = res["full"]
        self.assertEqual(full["true_positives"], 73)
        self.assertEqual(full["false_positives"], 0)
        self.assertEqual(full["false_negatives"], 0)
        self.assertEqual(full["precision"], 1.0)
        self.assertEqual(full["recall"], 1.0)
        self.assertEqual(res["n_items"], 84)

    def test_baseline_and_delta(self):
        res = run_measurement()
        base = res["baseline_5_pattern"]
        self.assertEqual(base["true_positives"], 21)
        self.assertEqual(base["false_positives"], 1)
        self.assertEqual(base["false_negatives"], 52)
        self.assertAlmostEqual(base["precision"], 21 / 22)
        self.assertAlmostEqual(base["recall"], 21 / 73)
        # The delta is the new-pattern + negation win, measured on
        # the same set: recall +0.71 (52 newly-covered gold claims),
        # precision +0.05 (the baseline's one FP is the dirty-subject
        # appositive capital_of the comma guard removes).
        self.assertAlmostEqual(res["delta_recall"], 52 / 73, places=4)
        self.assertGreater(res["delta_precision"], 0.0)
        # 52 miss items: one per missed gold claim (the dirty-subject
        # item is both an FP and an FN, so it is one of the 52, not
        # an extra).
        self.assertEqual(len(base["misses"]), 52)

    def test_baseline_is_really_five_patterns(self):
        e = five_pattern_extractor()
        self.assertEqual({rel for rel, _, _ in e.patterns},
                         {"capital_of", "located_in", "boils_at", "melts_at", "is_a"})
        self.assertEqual(e.negated_patterns, [])
        self.assertEqual(len(e.patterns), 5)


if __name__ == "__main__":
    unittest.main()
