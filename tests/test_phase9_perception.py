"""Phase 9 W4: perception carried items — noisy OCR + per-word OCR
confidence (§4.7), perception decides to look (§4.9), fragile-tool
what-if phrasing (§4.10).

Noisy OCR (§4.7): the numbers live in ``docs/phase9-noisy-ocr.json``
(real tesseract on rendered text at increasing degradation; see the
sidecar's ``method``). These tests do NOT hardcode the numbers: they
recompute the aggregates from the sidecar's per-sample rows and
assert they match, and they run one live degraded read to confirm
the pipeline still behaves (blurred confidence below clean).

Per-word OCR confidence (§4.7, decision P9-3): investigated, NOT
implemented — there is no clean seam, and the gate is the mechanism.
Evidence, pinned by ``test_ocr_claim_confidence_is_flat_source_trust``:
``PerceptionBus._route_ocr`` stores OCR claims via
``ltm.add(..., source=f"ocr:{basename}")``; ``LongTermMemory._trust_of``
splits the source id on ":" and looks up the ``ocr`` prefix in TRUST
(flat 0.5); ``Fact.confidence`` is computed from the sources dict
only; ``LTM.add`` takes no per-claim confidence parameter. Scaling a
claim by mean word confidence would require per-source-instance trust
or a new claim-confidence field on Fact plus persistence changes — a
change to the LTM trust model, which P9-3 forbids hacking. What DOES
carry per-word confidence today: the WM Thought (``mean_admitted``)
and the percept/bus result (mean word confidence). So the gate
(threshold 0.6, uncertain words never reaching WM/LTM) stays the
mechanism, and the claim confidence stays flat source trust.

Perception decides to look (§4.9, minimal): ``Brain._record_need_percept``
records a ``need_percept`` event + state entry on the
``need_more_evidence`` path. The test uses the atlantis setup (goal
with an unknown target and no plan).

Fragile-tool what-if (§4.10): the decline is CORRECT behaviour, pinned
here. ``Action`` (``ab/planning/planner.py``) has no properties field;
no action or tool in any workshop world carries a ``fragile``
property — "fragile" exists only as an eval-spec key and as prose
("the fragile tool" = snips/clippers/pick_lock, the tool that breaks
or never works). ``Brain._parse_what_if`` matches action names and
tool-entity names, so "the fragile tool" matches nothing and
``what_if`` honestly declines rather than guessing. No fragile tool
was invented to make a test pass.
"""
import json
import os
import tempfile
import unittest

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ab.body.environment import room_world_actions
from ab.body.ocr import FakeWordOcr, TesseractOcr, words_above_threshold
from ab.body.research import Researcher
from ab.body.web import WebFetcher
from ab.brain import Brain
from ab.eval.phase8_eval import LAB_STATE, TREASURE_TARGET, _set_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISY_SIDECAR = os.path.join(ROOT, "docs", "phase9-noisy-ocr.json")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _render(text, path, size=44):
    img = Image.new("RGB", (1100, 120), "white")
    font = ImageFont.truetype(FONT, size)
    ImageDraw.Draw(img).text((20, 30), text, fill="black", font=font)
    img.save(path)
    return img


def _transport(pages):
    return lambda url, t, m: (200, "text/html", pages[url]) if url in pages else (404, "text/html", "")


PAGES = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
         "https://b.org/p": "<p>Paris is the capital of France.</p>",
         "https://c.org/b": "<p>Berlin is the capital of Germany.</p>",
         "https://d.org/b": "<p>Berlin is the capital of Germany.</p>"}


class TestNoisyOcrSidecar(unittest.TestCase):
    def _load_sidecar(self):
        with open(NOISY_SIDECAR, encoding="utf-8") as fh:
            return json.load(fh)

    def test_sidecar_aggregates_match_per_sample_rows(self):
        rec = self._load_sidecar()
        rows = rec["per_sample"]
        self.assertEqual(rec["n_samples"], len(rows))
        self.assertEqual(rec["exact_reads"],
                         sum(1 for r in rows if r["exact_read"]))
        self.assertEqual(rec["gate_exact_reads"],
                         sum(1 for r in rows if r["gate_exact_read"]))
        mean = round(sum(r["confidence"] for r in rows) / len(rows), 3)
        self.assertAlmostEqual(rec["mean_confidence"], mean)
        for kind, agg in rec["mean_confidence_by_degradation"].items():
            sub = [r for r in rows if r["degradation"] == kind]
            self.assertEqual(agg["n"], len(sub))
            self.assertEqual(agg["exact_reads"],
                             sum(1 for r in sub if r["exact_read"]))
        # Degradation is a recorded result: the sidecar must show it
        # (blurred reads strictly worse than clean ones).
        clean = rec["mean_confidence_by_degradation"]["clean"]
        blur8 = rec["mean_confidence_by_degradation"]["blur8"]
        self.assertLess(blur8["mean_confidence"], clean["mean_confidence"])
        self.assertLess(blur8["exact_reads"], clean["exact_reads"])

    def test_gate_rejects_garbage_words(self):
        # At least one degraded sample has uncertain words the gate
        # kept out of the admitted text (the gate is the mechanism).
        rec = self._load_sidecar()
        self.assertTrue(any(r["uncertain_words"] for r in rec["per_sample"]))
        # And the gate never admits a word the sentence does not
        # contain without it being visible in misread_admitted.
        for r in rec["per_sample"]:
            sent = {w.strip(".,").lower() for w in r["sentence"].split()}
            admitted_bad = [w["text"] for w in r["admitted_words"]
                            if w["text"].strip(".,").lower() not in sent]
            self.assertEqual(sorted(admitted_bad), sorted(r["misread_admitted"]))

    @unittest.skipUnless(TesseractOcr().available(), "tesseract not installed")
    def test_live_blurred_read_is_worse_than_clean(self):
        # One live spot check (fast: two reads): blur degrades the
        # real tesseract read, as the sidecar records.
        d = tempfile.mkdtemp()
        text = "Mars is a planet."
        base = _render(text, os.path.join(d, "clean.png"))
        clean_path = os.path.join(d, "clean.png")
        blurred = base.filter(ImageFilter.GaussianBlur(5))
        blur_path = os.path.join(d, "blur5.png")
        blurred.save(blur_path)
        ocr = TesseractOcr()
        clean_conf = ocr.read(clean_path).confidence
        blur_conf = ocr.read(blur_path).confidence
        self.assertGreater(clean_conf, 0.5)
        self.assertLess(blur_conf, clean_conf)
        # And the gate admits fewer words from the blurred image.
        clean_words = ocr.read_words(clean_path)
        blur_words = ocr.read_words(blur_path)
        clean_admitted, _ = words_above_threshold(clean_words, 0.6)
        blur_admitted, _ = words_above_threshold(blur_words, 0.6)
        self.assertLessEqual(len(blur_admitted), len(clean_admitted))


class TestPerWordOcrConfidenceDecision(unittest.TestCase):
    def test_ocr_claim_confidence_is_flat_source_trust(self):
        # Words admitted at different confidences (0.95 down to 0.65)
        # still store the claim at flat ocr source trust 0.5: the
        # per-word confidences do NOT scale the LTM claim. That is
        # the P9-3 finding (gate is the mechanism), pinned here so a
        # future change to it is a deliberate decision, not drift.
        words = [("Mars", 0.95), ("is", 0.65), ("a", 0.9), ("planet", 0.9)]
        b = Brain(ocr=FakeWordOcr({"sign.png": words}))
        out = b.perceive_image("sign.png")
        self.assertEqual(out["text"], "Mars is a planet")
        fact = b.ltm.query("mars", "is_a", "planet")[0]
        self.assertEqual(fact.sources, {"ocr:sign.png": 0.5})
        self.assertEqual(fact.status, "unverified")

    def test_wm_thought_carries_mean_admitted_word_confidence(self):
        # The per-word confidence is not lost entirely: the WM Thought
        # carries the mean admitted-word confidence (0.85 here: mean
        # of 0.95, 0.65, 0.9, 0.9), and the bus result carries the mean
        # over all words (the same 0.85 here — nothing was uncertain).
        words = [("Mars", 0.95), ("is", 0.65), ("a", 0.9), ("planet", 0.9)]
        b = Brain(ocr=FakeWordOcr({"sign.png": words}))
        out = b.perceive_image("sign.png")
        thought = out["thought"]
        self.assertAlmostEqual(thought.confidence, 0.85)
        self.assertAlmostEqual(out["confidence"], 0.85)


class TestNeedPercept(unittest.TestCase):
    def test_unknown_target_goal_records_need_percept(self):
        fetcher = WebFetcher(["a.org", "b.org", "c.org", "d.org"],
                             transport=_transport(PAGES))
        b = Brain(researcher=Researcher(fetcher, None),
                  granted={"compute", "network"})
        # No research action exists for atlantis: no plan, utility
        # below the bar — the need_more_evidence path.
        b.add_goal("learn atlantis", {"known:atlantis|capital_of": True},
                   priority=0.1)
        b.step(None)
        self.assertTrue(any(e["kind"] == "need_more_evidence" for e in b.events))
        events = [e for e in b.events if e["kind"] == "need_percept"]
        self.assertTrue(events)
        self.assertIn("atlantis", events[0]["detail"])
        self.assertIn("capital_of", events[0]["detail"])
        self.assertIn("document", events[0]["detail"])
        entry = b.state.get("need_percept:atlantis|capital_of")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "document")

    def test_physical_target_names_observation_percept(self):
        # A 20-step physical plan is refused by the feasibility gate;
        # the missing key is physical, so the requested percept kind
        # is observation, not document.
        from ab.planning.planner import Action
        acts = [Action(f"s{i}", {"n": i}, {"n": i + 1}) for i in range(20)]
        b = Brain(acts, granted={"compute"})
        b.world = {"n": 0}
        b.add_goal("count to twenty", {"n": 20}, priority=0.9)
        b.step(None)
        self.assertTrue(any(e["kind"] == "need_more_evidence" for e in b.events))
        events = [e for e in b.events if e["kind"] == "need_percept"]
        self.assertTrue(events)
        self.assertIn("observation", events[0]["detail"])
        entry = b.state.get("need_percept:n|value")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "observation")


class TestFragileToolWhatIf(unittest.TestCase):
    def _lab_brain(self):
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        _set_world(b, LAB_STATE)
        goal = b.add_goal("get the treasure", dict(TREASURE_TARGET))
        b.goals.activate(goal.id)
        return b

    def test_literal_fragile_tool_phrasing_is_honestly_declined(self):
        # No action or tool bears the name "fragile tool" and no
        # action carries a fragile property (see module docstring):
        # the decline is the correct behaviour, pinned here.
        b = self._lab_brain()
        out = b.what_if("what if I use the fragile tool?")
        self.assertIsNone(out["action"])
        self.assertIn("could not identify", out["answer"])
        self.assertTrue(out["simulated"])

    def test_named_fragile_action_still_parses(self):
        # The parser itself works for the action that IS the fragile
        # tool in this world (pick_lock never works): the gap is the
        # descriptive phrasing only, not the simulation.
        b = self._lab_brain()
        out = b.what_if("what if I use pick_lock?")
        self.assertEqual(out["action"], "pick_lock")
        self.assertIsNotNone(out["predicted_success"])


if __name__ == "__main__":
    unittest.main()
