"""Phase 10.5 (W1) — variant-aware action identity tests.

The defect: world A (envgen workshop, seed 0) cuts zinc at krelm and
world B (seed 1) cuts tin at krelm with the SAME action name, SAME
preconditions and SAME effects — they differ ONLY in ``args``. The
Phase 10 planner keyed bans, failure rates and learner counts by
(name, situation), so a cut-zinc failure in B banned cut-tin too and
the last-resort relaxation re-proposed zinc forever (continued 7
failed / 24 cycles, stalled, vs fresh 1 / 14).

These tests pin the fix: identity is (name, canonical precondition
variant, canonical semantic args). Stdlib only, deterministic.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from ab.brain import Brain
from ab.core.learning import Episode, Learner, stable_rule_id
from ab.eval.envgen import generate
from ab.planning.planner import (
    Action,
    Planner,
    action_identity,
    canonical_args,
    canonical_variant,
)
from ab.reasoning.rules import RuleEngine


# The two colliding variants, exactly as envgen builds them
# (workshop seed 0 vs seed 1, shared room krelm).
ZINC = Action("cut", {"at": "krelm", "has_awl": True}, {"cut_b": True},
              args={"material": "zinc"})
TIN = Action("cut", {"at": "krelm", "has_awl": True}, {"cut_b": True},
             args={"material": "tin"})
SITUATION = {"at": "krelm", "has_awl": True}
TARGET = {"cut_b": True}


def _cut(material: str) -> Action:
    return Action("cut", {"at": "krelm", "has_awl": True}, {"cut_b": True},
                  args={"material": material})


def _learner_with_episodes():
    """A learner whose planner has both cut variants attached, with
    2 cut-zinc failures and 1 cut-tin success recorded."""
    planner = Planner([ZINC, TIN])
    learner = Learner(planner, RuleEngine())
    planner.attach_learner(learner)
    for i in range(2):
        learner.record(Episode(i, "cut", dict(SITUATION), False,
                               args={"material": "zinc"}))
    learner.record(Episode(2, "cut", dict(SITUATION), True,
                           args={"material": "tin"}))
    return planner, learner


class TestCanonicalArgs(unittest.TestCase):
    def test_excludes_dispatch_keys_but_keeps_semantic_args(self):
        # go carries _do (dispatch) and room (semantic).
        self.assertEqual(canonical_args({"_do": "go", "room": "lab"}),
                         "room='lab'")
        self.assertEqual(canonical_args({"material": "zinc"}),
                         "material='zinc'")
        # press carries only _do: no semantic identity.
        self.assertEqual(canonical_args({"_do": "press"}), "")
        self.assertEqual(canonical_args({}), "")
        self.assertEqual(canonical_args(None), "")

    def test_sorted_and_stable(self):
        self.assertEqual(canonical_args({"b": 2, "a": 1}),
                         canonical_args({"a": 1, "b": 2}))
        self.assertEqual(canonical_args({"a": 1, "b": 2}),
                         "a=1;b=2")


class TestActionIdentity(unittest.TestCase):
    def test_zinc_vs_tin_differ_only_in_args(self):
        self.assertNotEqual(action_identity(ZINC), action_identity(TIN))
        self.assertEqual(action_identity(ZINC)[0], action_identity(TIN)[0])
        # Same canonical precondition variant: the P9-1 key cannot
        # tell them apart — that is the defect this phase fixes.
        self.assertEqual(canonical_variant(ZINC.pre),
                         canonical_variant(TIN.pre))
        self.assertEqual(Planner.variant_of(ZINC),
                         Planner.variant_of(TIN))
        self.assertEqual(action_identity(ZINC)[1],
                         action_identity(TIN)[1])
        self.assertNotEqual(action_identity(ZINC)[2],
                            action_identity(TIN)[2])

    def test_empty_args_action_identity(self):
        press = Action("press", {"at": "bench"}, {"pressed": True},
                       args={"_do": "press"})
        self.assertEqual(action_identity(press),
                         ("press", canonical_variant(press.pre), ""))


class TestVariantAwareBans(unittest.TestCase):
    def test_ban_on_zinc_does_not_block_tin(self):
        planner = Planner([ZINC, TIN])
        planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        self.assertTrue(planner.is_banned("cut", dict(SITUATION),
                                           {"material": "zinc"}))
        self.assertFalse(planner.is_banned("cut", dict(SITUATION),
                                            {"material": "tin"}))
        self.assertFalse(planner.is_banned("cut", dict(SITUATION)))
        self.assertFalse(planner.is_banned("cut", {"at": "drelm",
                                                    "has_awl": True},
                                            {"material": "zinc"}))

    def test_plan_all_still_finds_tin_after_zinc_ban(self):
        planner = Planner([ZINC, TIN])
        planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        plans = planner.plan_all(dict(SITUATION), dict(TARGET))
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].steps[0].args, {"material": "tin"})
        # Banning tin too leaves no unbanned plan — the ban is real,
        # just variant-scoped.
        planner.ban("cut", dict(SITUATION), {"material": "tin"})
        self.assertEqual(planner.plan_all(dict(SITUATION), dict(TARGET)), [])
        # The last-resort relaxation still recovers a plan.
        relaxed = planner.plan_all(dict(SITUATION), dict(TARGET),
                                   allow_banned_pairs=True)
        self.assertEqual(len(relaxed), 1)

    def test_sequence_banned_is_variant_aware(self):
        planner = Planner([ZINC, TIN])
        planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        self.assertTrue(Planner.sequence_banned(
            [ZINC], dict(SITUATION), planner.banned_pairs))
        self.assertFalse(Planner.sequence_banned(
            [TIN], dict(SITUATION), planner.banned_pairs))
        # Plain-dict steps (the imagination path) behave the same.
        tin_dict = {"name": "cut", "pre": dict(SITUATION),
                    "eff": {"cut_b": True}, "args": {"material": "tin"}}
        zinc_dict = dict(tin_dict, args={"material": "zinc"})
        self.assertFalse(Planner.sequence_banned(
            [tin_dict], dict(SITUATION), planner.banned_pairs))
        self.assertTrue(Planner.sequence_banned(
            [zinc_dict], dict(SITUATION), planner.banned_pairs))

    def test_clear_bans_for_keys_still_works(self):
        planner = Planner([ZINC, TIN])
        planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        planner.ban("cut", dict(SITUATION), {"material": "tin"})
        self.assertEqual(planner.clear_bans_for_keys(["at"]), 2)
        self.assertEqual(len(planner.banned_pairs), 0)


class TestVariantAwareFailureRate(unittest.TestCase):
    def test_zinc_failures_do_not_inflate_tin_rate(self):
        planner, _learner = _learner_with_episodes()
        self.assertEqual(planner.failure_rate_for(ZINC), 1.0)
        self.assertEqual(planner.failure_rate_for(TIN), 0.0)
        # The pooled rate still mixes the variants (the fallback).
        self.assertAlmostEqual(planner.failure_rate("cut"), 2 / 3)

    def test_no_episode_variant_returns_zero(self):
        # P10.5-8 note (coordinator): W2 adds a key-shape fallback
        # to failure_rate_for — exact (name, variant values, args),
        # then key-shape (name, precondition KEY set, args), then
        # 0.0. This pin stays valid under it: no episode at all
        # carries args material=steel, so neither the exact tier
        # nor the key-shape tier (which still keys on args) matches,
        # and the rate is 0.0. Do NOT extend this pin to a variant
        # whose args DO appear in same-keys episodes.
        planner, _learner = _learner_with_episodes()
        steel = _cut("steel")
        self.assertEqual(planner.failure_rate_for(steel), 0.0)

    def test_no_learner_falls_back_to_pooled_rate(self):
        planner = Planner([ZINC, TIN])
        planner.record("cut", True)
        planner.record("cut", False)
        self.assertAlmostEqual(planner.failure_rate_for(ZINC), 0.5)
        self.assertAlmostEqual(planner.failure_rate_for(TIN), 0.5)

    def test_effective_cost_uses_variant_rate(self):
        # Phase 10.5 (P10.5-5, W2) — §9-precedent update of the W1
        # pin ``test_effective_cost_untouched_by_variant_rate``.
        # Same intent (pin which failure rate ``effective_cost``
        # uses), new intended semantics: the adoption is DONE, so
        # the cost now uses the variant-aware ``failure_rate_for``
        # (exact variant, then the P10.5-8 key-shape fallback),
        # not the pooled rate. Zinc failed twice (rate 1.0) and tin
        # succeeded once (rate 0.0), so the two variants' costs now
        # differ — the pooled pin (equal costs) would be a silent
        # weakening, not a preservation.
        planner, _learner = _learner_with_episodes()
        self.assertAlmostEqual(
            planner.effective_cost(ZINC),
            ZINC.cost * (1.0 + 2.0 * planner.failure_rate_for(ZINC)))
        self.assertAlmostEqual(
            planner.effective_cost(TIN),
            TIN.cost * (1.0 + 2.0 * planner.failure_rate_for(TIN)))
        self.assertAlmostEqual(planner.failure_rate_for(ZINC), 1.0)
        self.assertAlmostEqual(planner.failure_rate_for(TIN), 0.0)
        self.assertNotAlmostEqual(
            planner.effective_cost(ZINC), planner.effective_cost(TIN))


class TestLearnerCountsByArgs(unittest.TestCase):
    def test_counts_keyed_by_args_identity(self):
        _planner, learner = _learner_with_episodes()
        keys = {(a, args_id) for (a, args_id, _sig) in learner.counts}
        self.assertIn(("cut", "material='zinc'"), keys)
        self.assertIn(("cut", "material='tin'"), keys)
        zinc = learner.counts[("cut", "material='zinc'",
                                (("at", "krelm"), ("has_awl", "True")))]
        tin = learner.counts[("cut", "material='tin'",
                              (("at", "krelm"), ("has_awl", "True")))]
        self.assertEqual(zinc, [0, 2])
        self.assertEqual(tin, [1, 0])

    def test_stable_rule_id_differs_across_args(self):
        zinc_id = stable_rule_id("cut", dict(SITUATION),
                                 "material='zinc'")
        tin_id = stable_rule_id("cut", dict(SITUATION),
                                "material='tin'")
        self.assertNotEqual(zinc_id, tin_id)
        self.assertTrue(zinc_id.startswith("avoid_cut_"))
        # Deterministic across calls.
        self.assertEqual(zinc_id, stable_rule_id(
            "cut", dict(SITUATION), "material='zinc'"))

    def test_repeated_mistakes_groups_by_args(self):
        _planner, learner = _learner_with_episodes()
        rep = learner.repeated_mistakes("cut", dict(SITUATION),
                                        {"material": "zinc"})
        self.assertIsNotNone(rep)
        self.assertEqual(rep["failures"], 2)
        self.assertEqual(rep["successes"], 0)
        self.assertEqual(rep["args"], "material='zinc'")
        # Tin has a success in the same features: no repeat, and
        # zinc's failures do not count against it.
        self.assertIsNone(learner.repeated_mistakes(
            "cut", dict(SITUATION), {"material": "tin"}))

    def test_propose_rules_separates_variants(self):
        planner = Planner([ZINC, TIN])
        learner = Learner(planner, RuleEngine())
        planner.attach_learner(learner)
        for i in range(3):
            learner.record(Episode(i, "cut", dict(SITUATION), False,
                                   args={"material": "zinc"}))
        learner.record(Episode(3, "cut", dict(SITUATION), True,
                               args={"material": "tin"}))
        proposed = learner.propose_rules()
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0].id,
                         stable_rule_id("cut", dict(SITUATION),
                                        "material='zinc'"))

    def test_episode_args_round_trip_through_learner_json(self):
        _planner, learner = _learner_with_episodes()
        data = learner.to_json()
        self.assertEqual(data["episodes"][0]["args"],
                         {"material": "zinc"})
        restored = Learner.from_json(data, Planner(), RuleEngine())
        self.assertEqual(restored.episodes[0].args,
                         {"material": "zinc"})
        self.assertEqual(restored.episodes[2].args, {"material": "tin"})
        self.assertEqual(restored.counts, learner.counts)


class TestPersistence(unittest.TestCase):
    def test_episode_args_survive_brain_save_load(self):
        brain = Brain([ZINC, TIN], granted={"compute", "environment"})
        brain.learner.record(Episode(0, "cut", dict(SITUATION), False,
                                     args={"material": "zinc"}))
        brain.learner.record(Episode(1, "cut", dict(SITUATION), True,
                                     args={"material": "tin"}))
        folder = tempfile.mkdtemp()
        brain.save(folder)
        brain2 = Brain(granted={"compute", "environment"})
        brain2.load(folder)
        self.assertEqual(len(brain2.learner.episodes), 2)
        self.assertEqual(brain2.learner.episodes[0].args,
                         {"material": "zinc"})
        self.assertEqual(brain2.learner.episodes[1].args,
                         {"material": "tin"})
        # The variant-aware counts survive too.
        self.assertEqual(
            brain2.learner.counts[("cut", "material='zinc'",
                                   (("at", "krelm"), ("has_awl", "True")))],
            [0, 1])
        # And the failure rates still attribute by args after load.
        self.assertEqual(brain2.planner.failure_rate_for(ZINC), 1.0)
        self.assertEqual(brain2.planner.failure_rate_for(TIN), 0.0)

    def test_legacy_learner_json_loads_with_empty_args(self):
        _planner, learner = _learner_with_episodes()
        data = learner.to_json()
        # Downgrade to the pre-10.5 shape: no "args" on episodes, and
        # ONE pooled counts entry per (action, sig) — the counts a
        # real pre-10.5 save would carry (it never split by args).
        for ep in data["episodes"]:
            del ep["args"]
        data["counts"] = [[["cut", [["at", "krelm"], ["has_awl", "True"]]],
                           [1, 2]]]
        restored = Learner.from_json(data, Planner(), RuleEngine())
        self.assertTrue(all(ep.args == {} for ep in restored.episodes))
        self.assertEqual(restored.counts[("cut", "",
                                          (("at", "krelm"),
                                           ("has_awl", "True")))], [1, 2])

    def test_planner_banned_pairs_new_and_legacy_forms(self):
        planner = Planner([ZINC, TIN])
        planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        data = planner.to_json()
        self.assertEqual(data["banned_pairs"],
                         [["cut", "material='zinc'",
                           [["at", "krelm"], ["has_awl", "True"]]]])
        restored = Planner.from_json(data)
        self.assertTrue(restored.is_banned("cut", dict(SITUATION),
                                           {"material": "zinc"}))
        self.assertFalse(restored.is_banned("cut", dict(SITUATION),
                                            {"material": "tin"}))
        # Legacy 2-element form loads as the empty-args identity.
        legacy = Planner.from_json(
            {"actions": [],
             "banned_pairs": [["cut", [["at", "krelm"],
                                       ["has_awl", "True"]]]]})
        self.assertTrue(legacy.is_banned("cut", dict(SITUATION)))
        self.assertFalse(legacy.is_banned("cut", dict(SITUATION),
                                          {"material": "zinc"}))


class TestTryAlternativeVariantAware(unittest.TestCase):
    """Brain level, on the real envgen worlds: after cut-zinc fails
    in B (seed 1), only cut-zinc is banned and the retained
    alternative starting with cut-tin is adopted."""

    def test_try_alternative_bans_only_the_failed_variant(self):
        _env_a, actions_a, _goal_a, _shape_a = generate(0, "workshop")
        _env_b, actions_b, _goal_b, _shape_b = generate(1, "workshop")
        zinc = next(a for a in actions_a
                    if a.name == "cut" and a.args.get("material") == "zinc")
        tin = next(a for a in actions_b
                   if a.name == "cut" and a.args.get("material") == "tin")
        self.assertEqual(zinc.pre, tin.pre)
        self.assertEqual(zinc.eff, tin.eff)
        brain = Brain(actions_b, granted={"compute", "environment"})
        brain.world = dict(SITUATION)
        goal = brain.add_goal("cut the tin", dict(TARGET))
        brain.goals.activate(goal.id)
        brain.alternative_plan = [tin]
        brain.alternative_plan_cost = 1.0
        adopted = brain._try_alternative(zinc, dict(SITUATION))
        self.assertTrue(adopted)
        self.assertEqual(brain.plan[0].args, {"material": "tin"})
        self.assertTrue(brain.planner.is_banned(
            "cut", dict(SITUATION), {"material": "zinc"}))
        self.assertFalse(brain.planner.is_banned(
            "cut", dict(SITUATION), {"material": "tin"}))
        # And the planner can still reach the goal from here without
        # the last-resort relaxation: a tin plan exists, and no plan
        # starts with the banned zinc variant. (The full B action
        # set also yields a go-away-and-back route, so the count is
        # not pinned — only the variant split is.)
        plans = brain.planner.plan_all(dict(SITUATION), dict(TARGET))
        self.assertTrue(any(p.steps and p.steps[0].args == {"material": "tin"}
                            for p in plans))
        self.assertFalse(any(p.steps and p.steps[0].args == {"material": "zinc"}
                             for p in plans))

    def test_starts_with_banned_retry_is_variant_aware(self):
        brain = Brain([ZINC, TIN], granted={"compute", "environment"})
        brain.world = dict(SITUATION)
        brain.planner.ban("cut", dict(SITUATION), {"material": "zinc"})
        from ab.planning.planner import Plan
        self.assertTrue(brain._starts_with_banned_retry(
            Plan([ZINC], 1.0)))
        self.assertFalse(brain._starts_with_banned_retry(
            Plan([TIN], 1.0)))


class TestSecretArgsRedacted(unittest.TestCase):
    """P7-10 meets P10.5: a secret-marked tool argument must never
    reach a persisted record, and the redacted identity must still
    match — the episode, the ban key and the failure rate all key on
    the REDACTED args, so nothing leaks and nothing silently stops
    matching. Uses the Phase 7 stamper setup (a ToolSpec-marked
    ``secret`` argument whose value is short, so only the
    spec-aware redactor catches it)."""

    def test_secret_arg_redacted_in_episodes_and_save(self):
        from ab.body.tools import ToolSpec
        from ab.eval.execution_eval import (
            WORKSHOP_W1,
            WorkshopWorld,
            _register_workshop_tool,
            workshop_actions,
        )
        # A fake value, invented for this test (not a real secret).
        code = "test-only-fake-stamp-code"
        spec = WORKSHOP_W1
        env = WorkshopWorld(spec)
        brain = Brain(workshop_actions(spec),
                      granted={"compute", "environment"})
        _register_workshop_tool(brain, spec, env)

        def _stamper(code):
            return True

        brain.tools.register(ToolSpec(
            name="stamper",
            args=[{"name": "code", "type": "str", "required": True,
                   "secret": True}],
            return_type="str", timeout=5.0, required_grant="compute",
            description="Stamp with a secret code"), _stamper)
        stamp = Action("stamp", {}, {"stamped": True}, tool="stamper",
                       args={"code": code}, store_as="stamped")
        brain.planner.add_action(stamp)
        brain.add_goal("stamp once", {"stamped": True})
        brain.run(env, 20)
        self.assertEqual(brain.goals.all()[-1].status, "done")
        # The recorded episode carries the redacted arg, not the code.
        self.assertEqual(len(brain.learner.episodes), 1)
        self.assertEqual(brain.learner.episodes[0].args,
                         {"code": "[REDACTED]"})
        # ... and the failure rate still attributes the episode to
        # the live action (the planner's hook redacts the live args
        # for the comparison, so the redacted episode matches).
        self.assertEqual(brain.planner.failure_rate_for(stamp), 0.0)
        # ... and no saved file carries the code.
        folder = tempfile.mkdtemp()
        brain.save(folder)
        for fname in sorted(os.listdir(folder)):
            with open(os.path.join(folder, fname), encoding="utf-8") as fh:
                self.assertNotIn(code, fh.read(), msg=fname)

    def test_ban_on_secret_arg_is_redacted_and_still_matches(self):
        from ab.body.tools import ToolSpec
        brain = Brain([], granted={"compute", "environment"})
        brain.tools.register(ToolSpec(
            name="stamper",
            args=[{"name": "code", "type": "str", "required": True,
                   "secret": True}],
            return_type="str", timeout=5.0, required_grant="compute",
            description="Stamp with a secret code"), lambda code: True)
        code = "test-only-fake-stamp-code"
        stamp = Action("stamp", {}, {"stamped": True}, tool="stamper",
                       args={"code": code})
        brain.planner.ban("stamp", {}, {"code": code}, "stamper")
        # The stored key carries the redacted identity only.
        self.assertEqual(brain.planner.banned_pairs,
                         {("stamp", "code='[REDACTED]'", ())})
        # ... and the live (raw) action still matches the ban,
        # because the check redacts the same way.
        self.assertTrue(brain.planner.is_banned(
            "stamp", {}, {"code": code}, "stamper"))
        self.assertTrue(Planner.sequence_banned(
            [stamp], {}, brain.planner.banned_pairs,
            redact_args=brain._redact_tool_args))
        # Persistence of the ban carries no secret either.
        data = brain.planner.to_json()
        self.assertNotIn(code, json.dumps(data))


class TestCurrentWorldTieBreak(unittest.TestCase):
    """Phase 10.5 (P10.5-9): on an exact cost tie, the planner
    prefers an action native to the current world over a
    carried-over one. Cut-zinc (world A) and cut-tin (world B) tie
    on every cost component (base 1.0, pooled rate 0.0, variant
    rate 0.0, concept 0.8 both, pattern 1.0 both); insertion order
    used to pick zinc by accident of list order. The tie-break is
    the generation tag: actions added after the brain entered a
    new world win ties over carried-over actions — and never
    override a cost difference of any size."""

    def _planner_with_worlds(self):
        p = Planner([ZINC])
        self.assertEqual(p.generation_of(ZINC), 0)
        p.mark_generation()
        p.add_action(TIN)
        self.assertEqual(p.generation_of(TIN), 1)
        return p

    def test_tie_prefers_current_world_action(self):
        p = self._planner_with_worlds()
        plan = p.plan(dict(SITUATION), {"cut_b": True})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].args, {"material": "tin"})

    def test_cost_difference_still_wins_over_tie_break(self):
        pricey_tin = Action("cut", dict(SITUATION), {"cut_b": True},
                            cost=5.0, args={"material": "tin"})
        p = Planner([ZINC])
        p.mark_generation()
        p.add_action(pricey_tin)
        plan = p.plan(dict(SITUATION), {"cut_b": True})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].args, {"material": "zinc"})

    def test_no_mark_means_insertion_order(self):
        # A single world never marks a generation: all actions are
        # generation 0 and the old tie behaviour (insertion order)
        # is preserved exactly.
        p = Planner([ZINC, TIN])
        self.assertEqual(p.generation_of(ZINC), 0)
        self.assertEqual(p.generation_of(TIN), 0)
        plan = p.plan(dict(SITUATION), {"cut_b": True})
        self.assertEqual(plan.steps[0].args, {"material": "zinc"})

    def test_generations_survive_save_load(self):
        p = self._planner_with_worlds()
        restored = Planner.from_json(p.to_json())
        self.assertEqual(restored._generation, 1)
        self.assertEqual(restored.generation_of(ZINC), 0)
        self.assertEqual(restored.generation_of(TIN), 1)
        plan = restored.plan(dict(SITUATION), {"cut_b": True})
        self.assertEqual(plan.steps[0].args, {"material": "tin"})

    def test_legacy_save_loads_with_generation_zero(self):
        data = {"actions": [{"name": "cut", "pre": dict(SITUATION),
                             "eff": {"cut_b": True}, "cost": 1.0,
                             "args": {"material": "zinc"}}],
                "banned_pairs": []}
        restored = Planner.from_json(data)
        self.assertEqual(restored._generation, 0)
        self.assertEqual(restored.generation_of(ZINC), 0)

    def test_brain_first_plan_in_new_world_uses_tin(self):
        # Brain level, real envgen worlds: after the A leg and
        # entry to B, the first plan the carried brain makes for
        # the B goal cuts tin, not zinc — the zinc failure is never
        # incurred. (The full transfer leg is measured by the
        # consolidated suite; this pins the mechanism.)
        from ab.eval.consolidated import _action_identity, _run_generic_env
        from ab.eval.reasoning_tasks import make_variant_factories
        from ab.eval.transfer import prepare_brain_for_new_world
        factory = make_variant_factories()["full"]
        env_a, actions_a, goal_a, _ = generate(0, "workshop")
        carried = factory(actions_a, {"compute", "environment"})
        _run_generic_env(env_a, actions_a, goal_a, factory,
                         brain=carried)
        prepare_brain_for_new_world(carried)
        env_b, actions_b, goal_b, _ = generate(1, "workshop")
        # Merge B's actions exactly as _run_generic_env does.
        present = {_action_identity(a) for a in carried.planner.actions}
        for a in actions_b:
            if _action_identity(a) not in present:
                carried.planner.add_action(a)
                present.add(_action_identity(a))
        carried.world = dict(env_b.observe())
        plans = carried.planner.plan_all(carried.world, dict(goal_b),
                                         limit=1)
        self.assertEqual(len(plans), 1)
        cuts = [a for a in plans[0].steps if a.name == "cut"]
        self.assertEqual(len(cuts), 2)
        self.assertEqual([a.args.get("material") for a in cuts],
                         ["copper", "tin"])


if __name__ == "__main__":
    unittest.main()
