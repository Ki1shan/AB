"""Phase 11 W3 — research workflow, in-cycle (design §2.5).

Question formulation, goal meta + events, source-selection wiring,
admitted/held events, and goal completion. Uses the eval-task
invented names (lunvar/aerilon/drakovia) only as test data — the
brain itself contains none of them.
"""
import unittest

from ab.body.providers import MemoryKnowledgeProvider
from ab.body.research import Researcher
from ab.brain import Brain
from ab.reasoning.engine import Contradiction

PAGES = {"https://alpha.org/p": "Lunvar is the capital of Aerilon.",
         "https://beta.org/p": "Lunvar is the capital city of Aerilon.",
         "https://gamma.org/p": "Lunvar is the capital of Drakovia.",
         "https://delta.org/p": "Lunvar is the capital city of Drakovia."}


def brain_for(pages=None):
    researcher = Researcher(MemoryKnowledgeProvider(dict(pages or PAGES)), None)
    return Brain(researcher=researcher, granted={"compute", "network"})


def events(brain, kind):
    return [e for e in brain.events if e.get("kind") == kind]


class TestFormulateResearchQuestion(unittest.TestCase):
    def test_capital_of_template_renders_correctly(self):
        # Phase 11 integration fix: the design's literal template
        # rendered "capital_of" as "capital of of paris" (double
        # "of"). Relations now render as question phrases, so the
        # question reads correctly.
        b = Brain()
        self.assertEqual(b.formulate_research_question("paris", "capital_of"),
                         "What is the capital of paris?")
        self.assertEqual(b.formulate_research_question("lunvar", "capital_of"),
                         "What is the capital of lunvar?")

    def test_other_relations_and_normalisation(self):
        b = Brain()
        self.assertEqual(b.formulate_research_question("water", "boils_at"),
                         "What is the boiling point of water?")
        # Subject is normalised (case/whitespace/article); relation
        # aliases go through the vocab normaliser.
        self.assertEqual(b.formulate_research_question("  The Lunvar ", "capital_of"),
                         "What is the capital of lunvar?")
        self.assertEqual(b.formulate_research_question("paris", "capital"),
                         "What is the capital of paris?")

    def test_pure_function_of_inputs(self):
        b = Brain()
        q1 = b.formulate_research_question("lunvar", "capital_of")
        b.cycle = 99
        b.teach_fact("lunvar", "capital_of", "aerilon")
        self.assertEqual(b.formulate_research_question("lunvar", "capital_of"), q1)


class TestResearchGoalMetaAndEvents(unittest.TestCase):
    def test_goal_meta_question_and_sources_and_events(self):
        b = brain_for()
        goal = b.add_research_goal("lunvar", "capital_of",
                                   ["https://alpha.org/p", "https://beta.org/p"])
        self.assertEqual(goal.meta["research_question"],
                         "What is the capital of lunvar?")
        # Selection is delegated to Researcher.select_sources: no
        # history, so the shorter URL ranks first (cost tiebreak).
        self.assertEqual(goal.meta["research_sources"],
                         ["https://beta.org/p", "https://alpha.org/p"])
        self.assertEqual(goal.meta["research_sources"],
                         b.researcher.select_sources(
                             goal.meta["research_question"],
                             ["https://alpha.org/p", "https://beta.org/p"]))
        q_events = events(b, "research_question")
        self.assertEqual(len(q_events), 1)
        self.assertIn("What is the capital of lunvar?", q_events[0]["detail"])
        sel = events(b, "research_sources_selected")
        self.assertEqual(len(sel), 1)
        self.assertIn("https://beta.org/p", sel[0]["detail"])
        self.assertIn("quality=0.50", sel[0]["detail"])
        self.assertIn("beta.org", sel[0]["detail"])

    def test_no_researcher_records_question_but_no_selection(self):
        b = Brain()
        goal = b.add_research_goal("lunvar", "capital_of",
                                   ["https://alpha.org/p", "https://beta.org/p"])
        self.assertEqual(goal.meta["research_question"],
                         "What is the capital of lunvar?")
        self.assertNotIn("research_sources", goal.meta)
        self.assertEqual(events(b, "research_question"), events(b, "research_question"))
        self.assertEqual(len(events(b, "research_question")), 1)
        self.assertEqual(events(b, "research_sources_selected"), [])

    def test_empty_urls_no_selection_event(self):
        b = brain_for()
        goal = b.add_research_goal("lunvar", "capital_of", [])
        self.assertEqual(goal.meta["research_question"],
                         "What is the capital of lunvar?")
        self.assertNotIn("research_sources", goal.meta)
        self.assertEqual(events(b, "research_sources_selected"), [])

    def test_contradiction_path_goal_carries_question_meta(self):
        # The contradiction policy routes through add_research_goal,
        # so its goals carry the same meta + events (one helper, both
        # paths). A direct _handle_contradictions call with a
        # functional-slot conflict is the unit-level trigger.
        b = brain_for()
        b._handle_contradictions([
            Contradiction(("lunvar", "capital_of", "aerilon"),
                          ("lunvar", "capital_of", "drakovia"),
                          "functional conflict")])
        goals = [g for g in b.goals.all()
                 if "known:lunvar|capital_of" in g.target]
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0].meta["research_question"],
                         "What is the capital of lunvar?")
        self.assertEqual(len(events(b, "research_question")), 1)
        self.assertEqual(events(b, "research_sources_selected"), [])


class TestAdmittedHeldEvents(unittest.TestCase):
    def test_verified_claim_emits_research_admitted(self):
        b = brain_for()
        b._research_tool(b.researcher.investigate(["https://alpha.org/p", "https://beta.org/p"], b.cycle))
        admitted = events(b, "research_admitted")
        self.assertEqual(len(admitted), 1)
        self.assertIn("lunvar capital_of = aerilon", admitted[0]["detail"])
        self.assertIn("alpha.org", admitted[0]["detail"])
        self.assertIn("beta.org", admitted[0]["detail"])
        self.assertEqual(events(b, "research_held"), [])

    def test_single_domain_claim_emits_research_held_single_domain(self):
        b = brain_for()
        b._research_tool(b.researcher.investigate(["https://alpha.org/p"], b.cycle))
        held = events(b, "research_held")
        self.assertEqual(len(held), 1)
        self.assertIn("lunvar capital_of = aerilon", held[0]["detail"])
        self.assertIn("single-domain", held[0]["detail"])
        self.assertEqual(events(b, "research_admitted"), [])
        self.assertEqual(b.ltm.query("lunvar", "capital_of"), [])

    def test_mirror_claim_emits_research_held_mirror(self):
        pages = {"https://alpha.org/p": "Lunvar is the capital of Aerilon.",
                 "https://beta.org/p": "Lunvar is the capital of Aerilon."}
        b = brain_for(pages)
        b._research_tool(b.researcher.investigate(["https://alpha.org/p", "https://beta.org/p"], b.cycle))
        held = events(b, "research_held")
        self.assertEqual(len(held), 1)
        self.assertIn("mirror", held[0]["detail"])
        self.assertEqual(b.ltm.query("lunvar", "capital_of"), [])

    def test_disputed_claim_emits_research_held_disputed(self):
        b = brain_for()
        b._research_tool(b.researcher.investigate(["https://alpha.org/p", "https://beta.org/p"], b.cycle))
        b._research_tool(b.researcher.investigate(["https://gamma.org/p", "https://delta.org/p"], b.cycle))
        held = [e for e in events(b, "research_held")
                if "disputed" in e["detail"]]
        self.assertTrue(held, "no research_held event with reason disputed")
        self.assertTrue(any("drakovia" in e["detail"] for e in held))


class TestResearchGoalCompletion(unittest.TestCase):
    def test_demo_research_goal_still_completes(self):
        # Existing completion semantics unchanged: the goal completes
        # when its target key becomes known (verified).
        b = brain_for()
        goal = b.add_research_goal("lunvar", "capital_of",
                                   ["https://alpha.org/p", "https://beta.org/p"])
        stop = b.run(None, 30)
        self.assertEqual(goal.status, "done")
        self.assertEqual(stop, "done")
        self.assertEqual(b.ask("lunvar", "capital_of").best[2], "verified")
        self.assertEqual(len(events(b, "research_admitted")), 1)

    def test_single_source_goal_fails_honestly_with_held_trace(self):
        b = brain_for()
        goal = b.add_research_goal("lunvar", "capital_of",
                                   ["https://alpha.org/p"])
        b.run(None, 30)
        self.assertEqual(goal.status, "failed")
        self.assertIsNone(b.ask("lunvar", "capital_of").best)
        self.assertTrue(events(b, "research_held"))


if __name__ == "__main__":
    unittest.main()
