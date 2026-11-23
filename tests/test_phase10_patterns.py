"""Phase 10 (P10-1): variant-aware pattern application + variant-aware
failure rate.

The P9-7 root cause: the x6.0 pattern multiplier in
``PatternLibrary.match`` applied to the action NAME, so a P-D pattern
``requires_has_mould:pressed`` learned in workshop world W1 fired on
the CORRECT press variant in world W2 (which needs ``has_cast``, not
``has_mould``), and the continued brain did worse than fresh (2 failed
/ 26 cycles vs 0 / 14 on the generated workshop leg).

These tests pin the fix at the library level (direct ``match`` calls)
and at the planner level (``pattern_multiplier`` / ``failure_rate_for``
through a real ``Brain``):

- P-D fires only when its key is in the current world's vocabulary
  (current-world actions' preconditions/effects, or the state);
- a variant that already requires the key is never penalised;
- P-A alternatives are compared by variant and must be satisfiable;
- P-B / P-C are already variant-safe (keyed by the evaluated
  variant's own tool entity) — measured here, no code change;
- ``failure_rate_for`` separates variants, but ``effective_cost``
  keeps the pooled rate (P10-1b: the A/B measurement showed the
  variant rate turns the generated-workshop leg from a 26-cycle stall
  into an 80-cycle budget exhaustion).

Stdlib only, deterministic.
"""
from __future__ import annotations

import unittest

from ab.brain import Brain
from ab.core.learning import Episode
from ab.planning.planner import Action, Planner, canonical_variant
from ab.world.model import WorldModel
from ab.world.patterns import PatternLibrary

EFFECTS = {
    "press": {"pressed": True},
    "fetch_mould": {"has_mould": True},
    "fetch_cast": {"has_cast": True},
    "open_alpha": {"gate_open": True},
    "open_beta": {"gate_open": True},
    "use_gamma_tool": {"gate_open": True},
    "use_delta_tool": {"gate_open": True},
    "use_alpha_tool": {"gate_open": True},
    "use_beta_tool": {"gate_open": True},
}


def _w1_actions() -> list[Action]:
    """Workshop W1 shape: press is one name with three variants —
    two cheap ones with no mould key, one expensive one carrying
    ``has_mould`` — plus the mould fetch."""
    return [
        Action("press", {"at": "m1"}, {"pressed": True}, cost=0.3,
               args={"_do": "press"}),
        Action("press", {"at": "m2"}, {"pressed": True}, cost=0.5,
               args={"_do": "press"}),
        Action("press", {"at": "f", "has_mould": True}, {"pressed": True},
               cost=2.0, args={"_do": "press"}),
        Action("take_mould", {"at": "f", "has_mould": False},
               {"has_mould": True}, args={"_do": "take_mould"}),
    ]


def _w2_actions() -> list[Action]:
    """Workshop W2 shape: same press shape, but the mould is a
    ``cast`` — ``has_mould`` appears in NO W2 action and NO W2 state."""
    return [
        Action("press", {"at": "m1"}, {"pressed": True}, cost=0.3,
               args={"_do": "press"}),
        Action("press", {"at": "m2"}, {"pressed": True}, cost=0.5,
               args={"_do": "press"}),
        Action("press", {"at": "f", "has_cast": True}, {"pressed": True},
               cost=2.0, args={"_do": "press"}),
        Action("take_cast", {"at": "f", "has_cast": False},
               {"has_cast": True}, args={"_do": "take_cast"}),
    ]


def _w1_state() -> dict:
    return {"at": "m1", "has_mould": False, "door_open": False,
            "pressed": False}


def _w2_state() -> dict:
    return {"at": "m1", "has_shears": False, "has_clippers": False,
            "has_cast": False, "door_open": False, "pressed": False,
            "has_amulet": False}


def _promote_pd_in_w1(brain: Brain):
    """World W1 evidence through the normal consolidation path: press
    fails twice without the mould (two distinct situations) and
    succeeds once with it. Returns the promoted P-D pattern."""
    brain.learner.record(Episode(1, "press", {"at": "m1"}, False))
    brain.learner.record(Episode(2, "press", {"at": "m2"}, False))
    brain.learner.record(
        Episode(3, "press", {"at": "f", "has_mould": True}, True))
    brain._consolidation_pass()
    found = [p for p in brain.patterns.patterns()
             if p.kind == "requires_precondition"]
    assert len(found) == 1, f"expected one P-D pattern, got {found!r}"
    return found[0]


def _pd_only_library() -> PatternLibrary:
    """A library holding ONLY the P-D pattern
    (``requires_has_mould:pressed``), constructed directly so no
    P-A pattern confounds the P-D assertions (learning the same
    episodes through ``learn_from_episodes`` also learns the P-A
    contrast, which is correct behaviour but the wrong instrument
    for isolating P-D)."""
    from ab.world.patterns import AbstractPattern

    lib = PatternLibrary()
    lib.add_pattern(AbstractPattern(
        name="requires_has_mould:pressed",
        kind="requires_precondition",
        effect="pressed",
        roles=[{"name": "action",
                "features": {"requires": "has_mould"}},
               {"name": "precondition",
                "features": {"key": "has_mould"}}],
        relations=[("action", "requires", "has_mould")],
        evidence=[{"world": "W1", "action": "press",
                   "outcome": "failed"},
                  {"world": "W1", "action": "press",
                   "outcome": "success"}],
        confidence=0.7))
    assert [p.kind for p in lib.patterns()] == ["requires_precondition"]
    return lib


def _pd_library() -> PatternLibrary:
    """A library holding one promoted P-D pattern
    (``requires_has_mould:pressed``), learned exactly as
    ``_promote_pd_in_w1`` learns it. NOTE: learning these episodes
    also learns the P-A contrast (same effect, tool-less failure vs
    tooled success), so this library holds BOTH patterns — the
    realistic case. Tests that must isolate P-D use
    ``_pd_only_library`` instead."""
    lib = PatternLibrary()
    episodes = [
        {"action": "press", "situation": {"at": "m1"},
         "outcome": "failed", "world": "W1"},
        {"action": "press", "situation": {"at": "m2"},
         "outcome": "failed", "world": "W1"},
        {"action": "press",
         "situation": {"at": "f", "has_mould": True},
         "outcome": "success", "world": "W1"},
    ]
    learned = lib.learn_from_episodes(episodes, EFFECTS)
    candidates = [p for p in learned if p.candidate]
    assert len(candidates) == 1, f"expected one candidate, got {learned!r}"
    lib.add_pattern(candidates[0])
    return lib


def _pa_library() -> PatternLibrary:
    """A library holding one P-A pattern (tool-less opener fails,
    tooled opener succeeds, same effect ``gate_open``)."""
    lib = PatternLibrary()
    lib.learn_from_episodes(
        [{"action": "open_alpha", "situation": {"at": "hall"},
          "outcome": "failed", "world": "W"},
         {"action": "open_beta",
          "situation": {"at": "hall", "has_beta_key": True},
          "outcome": "success", "world": "W"}],
        EFFECTS)
    assert [p.kind for p in lib.patterns()] == ["unreliable_toolless_opener"]
    return lib


class TestPdVariantAwareness(unittest.TestCase):
    def test_w2_correct_press_variant_not_penalised(self):
        # The P10-1 fix, end to end through a real Brain: the W1 P-D
        # pattern is promoted, the brain continues into W2 (W2 actions
        # ADDED, as the eval does — the old W1 actions stay in the
        # planner), and the CORRECT W2 press variant (with the cast)
        # is no longer penalised: has_mould is in no W2 action and
        # no W2 state, so the pattern is inapplicable there.
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_in_w1(brain)
        self.assertEqual(pattern.name, "requires_has_mould:pressed")
        for a in _w2_actions():
            brain.planner.add_action(a)
        brain.world = _w2_state()
        with_cast = next(a for a in brain.planner.actions
                         if a.name == "press" and "has_cast" in a.pre)
        self.assertEqual(brain.planner.pattern_multiplier(with_cast), 1.0)

    def test_w1_cheap_press_still_penalised(self):
        # W1 steering is preserved: the cheap press is still x6.0
        # when has_mould is absent, because the key IS in W1's
        # vocabulary (the expensive press declares it).
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        _promote_pd_in_w1(brain)
        brain.world = _w1_state()
        cheap = next(a for a in brain.planner.actions
                     if a.name == "press" and "has_mould" not in a.pre)
        self.assertEqual(brain.planner.pattern_multiplier(cheap), 6.0)
        # With the mould held, P-D itself stops firing — but the
        # multiplier stays 6.0 via P-A: the tooled alternative is
        # satisfiable now that the mould is held, so the tool-less
        # variant is still dispreferred. (The P-D-only case — key
        # present, no P-A pattern — is 1.0; see
        # test_pd_fires_when_key_in_vocabulary_and_absent.)
        brain.world = {**_w1_state(), "has_mould": True}
        self.assertEqual(brain.planner.pattern_multiplier(cheap), 6.0)

    def test_pd_fires_when_key_in_vocabulary_and_absent(self):
        lib = _pd_only_library()
        model = WorldModel()
        mult = lib.match("press", {"at": "m1"}, {"pressed": True}, model,
                         [(a.name, a.pre, a.eff) for a in _w1_actions()],
                         _w1_state())
        self.assertEqual(mult, 6.0)
        # ...and P-D alone stops firing once the key is present.
        self.assertEqual(
            lib.match("press", {"at": "m1"}, {"pressed": True}, model,
                      [(a.name, a.pre, a.eff) for a in _w1_actions()],
                      {**_w1_state(), "has_mould": True}),
            1.0)

    def test_pd_no_penalty_when_variant_already_requires_key(self):
        # (c): the expensive press variant already requires has_mould
        # truthy, so penalising it for lacking the key is vacuous —
        # it cannot run without the key anyway.
        lib = _pd_only_library()
        model = WorldModel()
        expensive = next(a for a in _w1_actions()
                         if "has_mould" in a.pre)
        self.assertEqual(
            lib.match("press", expensive.pre, {"pressed": True}, model,
                      [(a.name, a.pre, a.eff) for a in _w1_actions()],
                      _w1_state(),
                      variant=canonical_variant(expensive.pre)),
            1.0)
        # The same check through the planner's own variant passing.
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        _promote_pd_in_w1(brain)
        brain.world = _w1_state()
        self.assertEqual(brain.planner.pattern_multiplier(expensive), 1.0)

    def test_pd_key_nowhere_in_world_never_fires(self):
        # (b): with W2 actions + W2 state, has_mould is mentioned
        # nowhere — the P-D pattern is inapplicable and never fires,
        # even on the cheap press variant. (P-D-only library: the
        # cheap press would ALSO be P-A penalised via the with-cast
        # variant — see test_pa_same_name_tooled_variant_counts —
        # so P-D must be isolated here.)
        lib = _pd_only_library()
        model = WorldModel()
        cheap = next(a for a in _w2_actions()
                     if "has_cast" not in a.pre)
        self.assertEqual(
            lib.match("press", cheap.pre, {"pressed": True}, model,
                      [(a.name, a.pre, a.eff) for a in _w2_actions()],
                      _w2_state()),
            1.0)

    def test_pd_key_in_vocabulary_via_effect_only(self):
        # The retirement-world shape: has_mould appears in NO action's
        # preconditions, only as the EFFECT of fetch_mould — that
        # still counts as vocabulary, so the pattern fires there.
        lib = _pd_only_library()
        model = WorldModel()
        actions = [
            ("press", {"at": "bench"}, {"pressed": True}),
            ("fetch_mould", {"at": "bench"}, {"has_mould": True}),
        ]
        self.assertEqual(
            lib.match("press", {"at": "bench"}, {"pressed": True},
                      model, actions, {"at": "bench"}),
            6.0)

    def test_pd_key_in_vocabulary_via_state_only(self):
        # The Phase 6 shape: the key is in the current state, though
        # no action mentions it — the state itself is vocabulary.
        lib = PatternLibrary()
        episodes = [
            {"action": "use_alpha_tool",
             "situation": {"at": "hall", "has_alpha_tool": True},
             "outcome": "failed", "world": "W"},
            {"action": "use_alpha_tool",
             "situation": {"at": "lab", "has_alpha_tool": True},
             "outcome": "failed", "world": "W"},
            {"action": "use_alpha_tool",
             "situation": {"at": "hall", "has_alpha_tool": True,
                           "has_key": True},
             "outcome": "success", "world": "W"},
        ]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        lib.add_pattern([p for p in learned if p.candidate][0])
        model = WorldModel()
        pre = {"at": "hall", "has_alpha_tool": True}
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model, state={"at": "hall", "has_key": False}),
            6.0)
        self.assertEqual(
            lib.match("use_alpha_tool", pre, {"gate_open": True},
                      model, state={"at": "hall", "has_key": True}),
            1.0)

    def test_pd_no_state_never_fires(self):
        lib = _pd_only_library()
        model = WorldModel()
        self.assertEqual(
            lib.match("press", {"at": "m1"}, {"pressed": True}, model,
                      [(a.name, a.pre, a.eff) for a in _w1_actions()]),
            1.0)

    def test_match_without_variant_still_works(self):
        # Backward compatibility: direct callers that never pass a
        # variant get exactly the old behaviour (modulo the
        # vocabulary gate, which needs the action list).
        lib = _pd_only_library()
        model = WorldModel()
        self.assertEqual(
            lib.match("press", {"at": "m1"}, {"pressed": True}, model,
                      [(a.name, a.pre, a.eff) for a in _w1_actions()],
                      _w1_state()),
            6.0)


class TestPaVariantAwareness(unittest.TestCase):
    def test_pa_phantom_alternative_does_not_count(self):
        # The only tooled alternative demands a tool no action in
        # this world can supply — a phantom, not a real alternative.
        # The state reports the key (False), so the alternative
        # passes the current-world filter; it is the SATISFIABILITY
        # check that rejects it (not held, not producible).
        lib = _pa_library()
        model = WorldModel()
        actions = [
            ("open_alpha", {"at": "hall"}, {"gate_open": True}),
            ("open_phantom", {"at": "hall", "has_phantom_tool": True},
             {"gate_open": True}),
        ]
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model, actions,
                      {"at": "hall", "has_phantom_tool": False}),
            1.0)
        # With no state supplied the old presence check applies, so
        # direct callers that pass no state keep working.
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model, actions),
            6.0)

    def test_pa_obtainable_alternative_counts(self):
        # The tooled alternative's key is obtainable (take_beta
        # produces it), so the alternative is real even though the
        # key is not held yet. (The state reports the key as False —
        # as a real env's observe() does from the first cycle — so
        # the alternative's preconditions are in the current world's
        # vocabulary.)
        lib = _pa_library()
        model = WorldModel()
        actions = [
            ("open_alpha", {"at": "hall"}, {"gate_open": True}),
            ("open_beta", {"at": "hall", "has_beta_key": True},
             {"gate_open": True}),
            ("take_beta", {"at": "hall"}, {"has_beta_key": True}),
        ]
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model, actions, {"at": "hall", "has_beta_key": False}),
            6.0)

    def test_pa_same_name_tooled_variant_counts(self):
        # The learned P-A contrast is situational, so a tooled
        # variant of the SAME action name is an alternative: the
        # workshop cheap press is penalised (correctly — it fails
        # without the cast) while the tooled variant is not.
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        _promote_pd_in_w1(brain)
        for a in _w2_actions():
            brain.planner.add_action(a)
        brain.world = _w2_state()
        cheap = next(a for a in brain.planner.actions
                     if a.name == "press" and "has_cast" not in a.pre
                     and "has_mould" not in a.pre)
        # The cheap press in W2: P-D is blocked (has_mould not in
        # W2's vocabulary), but P-A fires via the with-cast variant.
        self.assertEqual(brain.planner.pattern_multiplier(cheap), 6.0)

    def test_pa_accumulated_old_world_alternative_does_not_count(self):
        # A continued brain's action list accumulates old-world
        # actions: an opener whose tool belongs to the OLD world
        # (its key is in no current state) is not an alternative
        # in the new world.
        lib = _pa_library()
        model = WorldModel()
        actions = [
            ("open_alpha", {"at": "hall"}, {"gate_open": True}),
            ("use_snips", {"at": "hall", "has_snips": True},
             {"gate_open": True}),
        ]
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model, actions, {"at": "hall"}),
            1.0)


class TestPbPcVariantSafety(unittest.TestCase):
    def test_pb_keyed_by_evaluated_variant_tool(self):
        # P-B is already variant-safe (measured, no code change):
        # the penalty follows the TOOL ENTITY named in the evaluated
        # variant's own precondition, so two variants of one action
        # name with different tools are priced differently.
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        lib.learn_from_episodes(
            [{"action": "use_gamma_tool", "situation": situation,
              "outcome": "success", "world": "W"},
             {"action": "use_gamma_tool", "situation": situation,
              "outcome": "failed", "world": "W"}],
            EFFECTS)
        model = WorldModel()
        model.mark_single_use_suspect("gamma_tool", cycle=2)
        self.assertEqual(
            lib.match("open", {"at": "hall", "has_gamma_tool": True},
                      {"gate_open": True}, model),
            6.0)
        self.assertEqual(
            lib.match("open", {"at": "hall", "has_delta_tool": True},
                      {"gate_open": True}, model),
            1.0)

    def test_pc_keyed_by_evaluated_variant_tool(self):
        # P-C likewise: the durable bonus follows the evaluated
        # variant's own tool entity and the world model's per-entity
        # property — a variant with an unmarked tool gets no bonus.
        lib = PatternLibrary()
        episodes = [
            {"action": "use_alpha_tool",
             "situation": {"at": "hall", "has_alpha_tool": True},
             "outcome": "success", "world": "W"},
            {"action": "use_alpha_tool",
             "situation": {"at": "lab", "has_alpha_tool": True},
             "outcome": "success", "world": "W"},
            {"action": "use_alpha_tool",
             "situation": {"at": "hall", "has_alpha_tool": True},
             "outcome": "success", "world": "W"},
        ]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        lib.add_pattern([p for p in learned if p.candidate][0])
        model = WorldModel()
        model.set_property("alpha_tool", "breaks_after_first_use", False,
                           status="inferred", confidence=0.7, cycle=1)
        self.assertEqual(
            lib.match("cut", {"at": "hall", "has_alpha_tool": True},
                      {"gate_open": True}, model),
            0.8)
        self.assertEqual(
            lib.match("cut", {"at": "hall", "has_beta_tool": True},
                      {"gate_open": True}, model),
            1.0)


class TestVariantAwareFailureRate(unittest.TestCase):
    def _press_brain(self) -> Brain:
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        # The cheap press fails twice (two variants); the expensive
        # press succeeds once. Pooled, press is 2 failed / 3.
        brain.learner.record(Episode(1, "press", {"at": "m1"}, False))
        brain.learner.record(Episode(2, "press", {"at": "m2"}, False))
        brain.learner.record(
            Episode(3, "press", {"at": "f", "has_mould": True}, True))
        return brain

    def test_failure_rate_for_separates_variants(self):
        brain = self._press_brain()
        cheap = next(a for a in brain.planner.actions
                     if a.name == "press" and a.pre == {"at": "m1"})
        expensive = next(a for a in brain.planner.actions
                         if "has_mould" in a.pre)
        self.assertEqual(brain.planner.failure_rate_for(cheap), 1.0)
        self.assertEqual(brain.planner.failure_rate_for(expensive), 0.0)
        # The pooled rate still mixes the variants.
        self.assertAlmostEqual(brain.planner.failure_rate("press"), 2 / 3)

    def test_effective_cost_uses_variant_rate(self):
        # Phase 10.5 (P10.5-5, W2) — §9-precedent update of the W1
        # pin ``test_effective_cost_keeps_pooled_rate``. Same intent
        # (pin which failure rate ``effective_cost`` uses), new
        # intended semantics: the adoption is DONE, so the cost now
        # uses the variant-aware ``failure_rate_for`` (exact variant,
        # then the P10.5-8 key-shape fallback), not the pooled rate.
        # The mould press keeps rate 0.0, so its cost is its bare
        # 2.0 (the pooled rate would charge 2.0 * (1 + 2 * 2/3) =
        # 4.67); the cheap presses carry rate 1.0, so theirs is
        # tripled (0.3 -> 0.9, 0.5 -> 1.5). The P10-1b regression
        # (80-cycle budget exhaustion) does not recur: the P10.5-5
        # stall fix keeps idempotent re-presses from counting as
        # progress (see tests/test_phase10_5_stall.py).
        brain = self._press_brain()
        cheap = next(a for a in brain.planner.actions
                     if a.name == "press" and a.pre == {"at": "m1"})
        expensive = next(a for a in brain.planner.actions
                         if "has_mould" in a.pre)
        self.assertAlmostEqual(
            brain.planner.effective_cost(expensive), 2.0)
        self.assertAlmostEqual(
            brain.planner.effective_cost(cheap), 0.9)
        # ...while the variant rates themselves still report 1.0/0.0.
        self.assertEqual(brain.planner.failure_rate_for(cheap), 1.0)
        self.assertEqual(brain.planner.failure_rate_for(expensive), 0.0)

    def test_failure_rate_for_falls_back_without_learner(self):
        planner = Planner([Action("press", {"at": "m1"},
                                  {"pressed": True}, cost=1.0)])
        planner.record("press", False)
        planner.record("press", True)
        action = planner.actions[0]
        self.assertEqual(planner.failure_rate_for(action),
                         planner.failure_rate("press"))
        self.assertEqual(planner.failure_rate("press"), 0.5)

    def test_failure_rate_for_untried_variant_is_zero(self):
        brain = Brain(_w1_actions(), granted={"compute", "environment"})
        expensive = next(a for a in brain.planner.actions
                         if "has_mould" in a.pre)
        self.assertEqual(brain.planner.failure_rate_for(expensive), 0.0)
        self.assertEqual(brain.planner.failure_rate("press"), 0.0)


class TestPatternMultiplierNoLibrary(unittest.TestCase):
    def test_multiplier_is_one_without_library(self):
        planner = Planner(_w1_actions())
        cheap = next(a for a in planner.actions if a.name == "press")
        self.assertEqual(planner.pattern_multiplier(cheap), 1.0)


if __name__ == "__main__":
    unittest.main()
