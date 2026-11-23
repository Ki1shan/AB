"""Phase 10 (P10-6): in-cycle analogy.

Phase 9 proposed analogy engine-level only (``ReasoningEngine.analogize``
called directly by the eval task). Phase 10 wires it into
``Brain._reason_from_wm``: when the active goal names a (subject,
relation) with no verified fact and no derivation this cycle, and the
subject is in working memory, the engine proposes by analogy over LTM
+ this cycle's derived facts. The hypothesis goes through the EXISTING
hypothesis path (``hypothesis:*`` state entry, test = observation /
research, confidence capped at 0.4 by ``analogize`` itself), is never
verified alone, and never drives an action (no ``inferred:*`` entry,
no plan step, no LTM fact).

The task facts are imported from the Phase 9 held-out analogy task
(``ab/eval/reasoning_tasks2.py``) — the same task, now run in-cycle
instead of engine-level. Nothing about the task is re-stated here.

Stdlib only, deterministic.
"""
import unittest

from ab.brain import Brain
from ab.eval.reasoning_tasks2 import (ANALOGY_FACTS, ANALOGY_RELATION,
                                       ANALOGY_SUBJECT, ANALOGY_VALUE)


def _analogy_brain():
    """Brain taught the Phase 9 analogy facts, with the goal that
    makes the missing relation a live target (``norbex opens ?``),
    activated, and the subject attended to in working memory."""
    b = Brain([], granted={"compute"})
    for s, r, v in ANALOGY_FACTS:
        b.teach_fact(s, r, v, "teacher")
    goal = b.add_goal("learn opens of norbex",
                      {"known:norbex|opens": True}, priority=0.5)
    b.goals.activate(goal.id)
    b.perceive_text(ANALOGY_SUBJECT)
    return b


def _analogy_hypotheses(brain):
    return [h for h in brain.hypotheses if h.get("source") == "analogy"]


class TestInCycleAnalogy(unittest.TestCase):
    def test_analogy_hypothesis_appears_in_cycle_in_state(self):
        # The Phase 9 analogy task, run in-cycle: after one step the
        # hypothesis is in state with source "analogy" — the brain
        # proposed it itself, not the test calling the engine.
        b = _analogy_brain()
        b.step()
        hyps = _analogy_hypotheses(b)
        self.assertEqual(len(hyps), 1)
        hyp = hyps[0]
        self.assertEqual(hyp["subject"], ANALOGY_SUBJECT)
        self.assertEqual(hyp["relation"], ANALOGY_RELATION)
        self.assertEqual(hyp["value"], ANALOGY_VALUE)
        self.assertEqual(hyp["status"], "open")
        self.assertEqual(hyp["confidence"], 0.4)  # capped: min(0.4, 0.2 * 2 shared)
        self.assertIn("quivern", hyp["test"])
        entry = b.state.get(f"hypothesis:{hyp['id']}")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value["source"], "analogy")

    def test_brain_still_answers_i_dont_know_until_test_corroborates(self):
        b = _analogy_brain()
        b.step()
        self.assertTrue(_analogy_hypotheses(b))
        answer = b.ask(ANALOGY_SUBJECT, ANALOGY_RELATION)
        self.assertFalse(answer.known)
        self.assertIn("don't know", answer.message)

    def test_analogy_never_drives_an_action(self):
        b = _analogy_brain()
        b.step()
        self.assertTrue(_analogy_hypotheses(b))
        self.assertEqual(b.plan, [])
        self.assertIsNone(
            b.state.get(f"inferred:{ANALOGY_SUBJECT}|{ANALOGY_RELATION}"))
        self.assertFalse(any(ANALOGY_SUBJECT in str(g.target)
                             for g in b.goals.all()
                             if g.description != "learn opens of norbex"))

    def test_no_verified_fact_created_by_analogy_alone(self):
        b = _analogy_brain()
        b.step()
        self.assertTrue(_analogy_hypotheses(b))
        self.assertEqual(
            b.ltm.query(ANALOGY_SUBJECT, ANALOGY_RELATION,
                        statuses=("verified",)), [])
        self.assertEqual(b.ltm.query(ANALOGY_SUBJECT, ANALOGY_RELATION), [])
        self.assertFalse(b.known(ANALOGY_SUBJECT, ANALOGY_RELATION))

    def test_no_analogy_when_verified_fact_exists(self):
        b = Brain([], granted={"compute"})
        for s, r, v in ANALOGY_FACTS:
            b.teach_fact(s, r, v, "teacher")
        # The target relation is already a verified teacher fact:
        # there is nothing to hypothesise about.
        b.teach_fact(ANALOGY_SUBJECT, ANALOGY_RELATION, ANALOGY_VALUE,
                     "teacher")
        goal = b.add_goal("learn opens of norbex",
                          {"known:norbex|opens": True}, priority=0.5)
        b.goals.activate(goal.id)
        b.perceive_text(ANALOGY_SUBJECT)
        b.step()
        self.assertEqual(_analogy_hypotheses(b), [])
        self.assertTrue(b.ask(ANALOGY_SUBJECT, ANALOGY_RELATION).known)

    def test_no_analogy_without_wm_subject(self):
        # The goal names a subject the brain has never attended to:
        # its facts are in LTM, but nothing in WM mentions it — not
        # even the goal description (which is deliberately generic
        # here). Analogy only fires for a WM subject (design P10-6).
        # NOTE: in the sibling test the goal description itself
        # ("learn opens of norbex") attends the subject into WM via
        # the goal thought, so that case DOES analogize — correctly.
        b = Brain([], granted={"compute"})
        b.teach_fact("quivern", "is_a", "mellor", "teacher")
        b.teach_fact("quivern", "located_in", "vortan", "teacher")
        b.teach_fact("quivern", "opens", "garnik", "teacher")
        b.teach_fact("zorblat", "is_a", "mellor", "teacher")
        b.teach_fact("zorblat", "located_in", "vortan", "teacher")
        goal = b.add_goal("learn the missing relation",
                          {"known:zorblat|opens": True}, priority=0.5)
        b.goals.activate(goal.id)
        b.step()
        self.assertEqual(_analogy_hypotheses(b), [])
        self.assertFalse(b.ask("zorblat", "opens").known)

    def test_analogy_hypothesis_not_duplicated_across_cycles(self):
        b = _analogy_brain()
        b.step()
        b.step()
        hyps = _analogy_hypotheses(b)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0]["value"], ANALOGY_VALUE)
        self.assertEqual(hyps[0]["status"], "open")
        # One state entry, not one per cycle.
        keys = [k for k in b.state.keys()
                if k.startswith("hypothesis:") and
                b.state.get(k).value.get("source") == "analogy"]
        self.assertEqual(len(keys), 1)

    def test_analogy_hypothesis_test_names_observation_or_research(self):
        b = _analogy_brain()
        b.step()
        hyp = _analogy_hypotheses(b)[0]
        self.assertIn(ANALOGY_VALUE, hyp["test"])
        self.assertIn("verify", hyp["test"])
        # It explains no surprise outcome, so no action-outcome test
        # can corroborate/falsify it (see _test_hypotheses).
        self.assertIsNone(hyp["outcome"])

    def test_analogy_uses_derived_facts_as_shared_features(self):
        # The ONLY feature the target shares with its sibling is a
        # DERIVED one: norbex is_a mellor follows by deduction from
        # norbex is_a krelm + krelm is_a mellor. Without derived facts
        # in the analogy facts function, the siblings share nothing
        # and no hypothesis is proposed.
        b = Brain([], granted={"compute"})
        b.teach_fact("quivern", "is_a", "mellor", "teacher")
        b.teach_fact("quivern", "opens", "garnik", "teacher")
        b.teach_fact("norbex", "is_a", "krelm", "teacher")
        b.teach_fact("krelm", "is_a", "mellor", "teacher")
        goal = b.add_goal("learn opens of norbex",
                          {"known:norbex|opens": True}, priority=0.5)
        b.goals.activate(goal.id)
        b.perceive_text("norbex")
        b.step()
        # The derivation itself exists this cycle...
        derived = [d for d in b.last_inferences
                   if d["subject"] == "norbex" and d["relation"] == "is_a"
                   and d["value"] == "mellor"]
        self.assertTrue(derived)
        # ...and the analogy used it as the shared feature.
        hyps = _analogy_hypotheses(b)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0]["value"], "garnik")
        self.assertIn("norbex is_a mellor", hyps[0]["evidence_for"])


if __name__ == "__main__":
    unittest.main()
