"""Phase 2: the cognitive cycle — alternative plans, store_as, near-tie /
need-more-evidence decisions, and persistence of decisions and learner
episodes.
"""
import tempfile
import unittest

from ab.brain import Brain
from ab.body.environment import RoomWorld, room_world_actions
from ab.body.research import Researcher
from ab.body.tools import Tool
from ab.body.web import WebFetcher
from ab.core.learning import Episode, Learner, stable_rule_id
from ab.planning.planner import Action, Planner
from ab.reasoning.rules import RuleEngine


def _transport(pages):
    return lambda url, t, m: (200, "text/html", pages[url]) if url in pages else (404, "text/html", "")


PAGES = {"https://a.org/p": "<p>Paris is the capital of France.</p>",
         "https://b.org/p": "<p>Paris is the capital of France.</p>",
         "https://c.org/b": "<p>Berlin is the capital of Germany.</p>",
         "https://d.org/b": "<p>Berlin is the capital of Germany.</p>"}


def _research_brain():
    fetcher = WebFetcher(["a.org", "b.org", "c.org", "d.org"], transport=_transport(PAGES))
    return Brain(researcher=Researcher(fetcher, None), granted={"compute", "network"})


class TestAlternativePlans(unittest.TestCase):
    def test_failed_action_tries_alternative_before_replanning(self):
        env = RoomWorld()
        env.s["at"] = "lab"   # start in the lab: two first actions exist
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        g = b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select
        b.step(env)   # plan
        self.assertEqual([a.name for a in b.alternative_plan],
                         ["take_key", "unlock", "go_vault", "take_treasure"])
        b.step(env)   # execute pick_lock: it fails
        self.assertTrue(any(e["kind"] == "alternative_plan" for e in b.events))
        self.assertEqual([a.name for a in b.plan],
                         ["take_key", "unlock", "go_vault", "take_treasure"])
        self.assertTrue(b.planner.is_banned("pick_lock", {"at": "lab"}))
        self.assertEqual(b.run(env, 60), "done")
        self.assertEqual(g.status, "done")
        pick_lock_acts = [e for e in b.events
                          if e["kind"] == "act" and "pick_lock" in e["detail"]]
        self.assertEqual(len(pick_lock_acts), 1)


class TestStoreAs(unittest.TestCase):
    def test_store_as_writes_tool_output_to_state(self):
        # NOTE on the scaffolding (verified against ab/body/tools.py and
        # ab/brain.py _do_execute_step, not assumed):
        #  - ToolRegistry calls fn(**args), so a tool fn must accept
        #    keyword args; a `lambda args: ...` fn fails with TypeError.
        #  - A tool action's declared effects are never applied to state
        #    by the brain; only store_as (or _refresh_knowledge) can
        #    materialise them. So the effect key, the store_as key and
        #    the goal key must agree, or the goal can never complete.
        #  - The tool output here is a hashable scalar. A dict output
        #    is covered separately below (test_dict_valued_store_as):
        #    it used to crash Planner.plan_all in _freeze ("unhashable
        #    type: 'dict'") until _freeze learned to encode unhashable
        #    values canonically.
        b = Brain(granted={"compute"})
        b.tools.register(Tool("echo7", lambda **kwargs: 7, "compute", "returns seven"))
        b.planner.add_action(Action("get_seven", {}, {"seven": 7},
                                    args={}, tool="echo7", store_as="seven"))
        g = b.add_goal("get seven", {"seven": 7})
        self.assertEqual(b.run(None, 30), "done")
        self.assertEqual(g.status, "done")
        self.assertEqual(b.world.get("seven"), 7)
        entry = b.state.get("seven")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source, "tool:echo7")

    def test_dict_valued_store_as_does_not_crash_planner(self):
        # Regression: a dict tool output stored via store_as used to
        # make the next Planner.plan_all raise TypeError in _freeze.
        # The planner now encodes unhashable values canonically, so
        # the goal completes and the dict is in state verbatim.
        b = Brain(granted={"compute"})
        b.tools.register(Tool("echo_dict", lambda **kwargs: {"value": 7},
                              "compute", "returns a dict"))
        b.planner.add_action(Action("get_dict", {}, {"seven": {"value": 7}},
                                    args={}, tool="echo_dict", store_as="seven"))
        g = b.add_goal("get dict", {"seven": {"value": 7}})
        self.assertEqual(b.run(None, 30), "done")
        self.assertEqual(g.status, "done")
        self.assertEqual(b.world.get("seven"), {"value": 7})
        entry = b.state.get("seven")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.source, "tool:echo_dict")


class TestUncertainDecisions(unittest.TestCase):
    def test_near_tie_creates_research_goal_instead_of_guessing(self):
        b = _research_brain()
        b.add_goal("learn paris", {"known:paris|capital_of": True}, priority=0.5)
        b.add_goal("learn berlin", {"known:berlin|capital_of": True}, priority=0.5)
        b.step(None)
        self.assertIsNone(b.goals.active())   # deferred, not guessed
        self.assertTrue(any(e["kind"] == "near_tie" for e in b.events))
        self.assertTrue(any(g.description.startswith("learn capital_of of")
                            for g in b.goals.pending()))
        self.assertIsNotNone(b.last_decision)
        self.assertTrue(b.last_decision.tie)

    def test_low_feasibility_plan_refused_with_need_more_evidence(self):
        b = _research_brain()
        # No research action exists for atlantis, so feasibility is 0.0
        # and utility is 0.4*0.1 + 0.2*0.3 + 0.4*0.0 = 0.10 < 0.25.
        g = b.add_goal("learn atlantis", {"known:atlantis|capital_of": True},
                       priority=0.1)
        b.step(None)
        self.assertIsNot(b.goals.active(), g)
        self.assertIsNone(b.goals.active())
        self.assertTrue(any(e["kind"] == "need_more_evidence" for e in b.events))
        self.assertTrue(any(g.description == "learn capital_of of atlantis"
                            for g in b.goals.pending()))
        self.assertIsNotNone(b.last_decision)
        self.assertIsNone(b.last_decision.chosen)


class TestPhase2Persistence(unittest.TestCase):
    def test_decision_is_persisted_and_restored(self):
        env = RoomWorld()
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        b.add_goal("get the treasure", {"has_treasure": True})
        b.step(env)   # select: records last_decision
        folder = tempfile.mkdtemp()
        b.save(folder)
        b2 = Brain(granted={"compute", "environment"})
        b2.load(folder)
        self.assertIsNotNone(b2.last_decision)
        self.assertEqual(b2.last_decision.chosen.name, "get the treasure")
        self.assertEqual(b2.last_decision.ranking[0][0], "get the treasure")
        self.assertGreater(b2.last_decision.feasibility, 0)

    def test_learner_episodes_survive_save_load(self):
        # NOTE on ordering: propose_rules() adds the rule to the brain's
        # RuleEngine, and rules ARE persisted (rules.json). Proposing on
        # the original *before* save would therefore make the loaded
        # brain's propose_rules() correctly return [] (rule already
        # known). So: save first, then propose on the loaded brain and
        # on the original, and compare the ids — both must derive the
        # same stable rule id from the restored episodes.
        b = Brain(room_world_actions(), granted={"compute", "environment"})
        for i in range(3):
            b.learner.record(Episode(i, "pick_lock", {"at": "lab"}, False))
        folder = tempfile.mkdtemp()
        b.save(folder)
        b2 = Brain(granted={"compute", "environment"})
        b2.load(folder)
        self.assertEqual(len(b2.learner.episodes), 3)
        prop_loaded = b2.learner.propose_rules()
        prop_orig = b.learner.propose_rules()
        self.assertEqual(len(prop_loaded), 1)
        self.assertEqual(len(prop_orig), 1)
        self.assertEqual(prop_loaded[0].id, prop_orig[0].id)
        self.assertEqual(prop_loaded[0].id,
                         stable_rule_id("pick_lock", {"at": "lab"}))
        failures = sum(v[1] for (action, _args, _), v in b2.learner._counts.items()
                       if action == "pick_lock")
        self.assertEqual(failures, 3)

    def test_learner_prunes_episodes_to_500_on_record(self):
        learner = Learner(Planner(), RuleEngine())
        for i in range(600):
            learner.record(Episode(i, "a", {"x": 1}, False))
        self.assertEqual(len(learner.episodes), 500)


if __name__ == "__main__":
    unittest.main()
