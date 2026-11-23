"""Abstract transfer patterns (Phase 5, Worker 1): P-A contrastive
learning, P-B one-use learning, structural matching, persistence.

All names here are invented for the tests (alpha/beta/gamma/delta
tools, gate_open effect, hall room) — no world specs are imported, and
no pattern may be learned from, or matched by, a name.
"""
import unittest
from types import SimpleNamespace

from ab.world.model import WorldModel
from ab.world.patterns import AbstractPattern, PatternLibrary

EFFECTS = {
    "open_alpha": {"gate_open": True},
    "open_beta": {"gate_open": True},
    "use_gamma_tool": {"gate_open": True},
    "use_delta_tool": {"gate_open": True},
    "force_entry": {"gate_open": True},
    "use_epsilon_tool": {"gate_open": True},
}

ALL_ACTIONS = [
    ("open_alpha", {"at": "hall"}, {"gate_open": True}),
    ("open_beta", {"at": "hall", "has_beta_key": True}, {"gate_open": True}),
    ("use_gamma_tool", {"at": "hall", "has_gamma_tool": True}, {"gate_open": True}),
    ("use_delta_tool", {"at": "hall", "has_delta_tool": True}, {"gate_open": True}),
]


def _ep(action, situation, ok, world="world_alpha"):
    return {"action": action, "situation": dict(situation),
            "outcome": "success" if ok else "failed", "world": world}


class TestLearnPA(unittest.TestCase):
    def test_pa_learned_only_with_contrastive_evidence(self):
        lib = PatternLibrary()
        episodes = [
            _ep("open_alpha", {"at": "hall"}, False),          # tool-less fails
            _ep("open_beta", {"at": "hall", "has_beta_key": True}, True),
        ]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        pattern = learned[0]
        self.assertEqual(pattern.kind, "unreliable_toolless_opener")
        self.assertEqual(pattern.effect, "gate_open")
        self.assertEqual(
            {r["name"]: r["features"] for r in pattern.roles},
            {"opener": {"requires_object": False},
             "tool_opener": {"requires_object": True}})
        self.assertIn(("tool_opener", "produces", "gate_open"),
                      pattern.relations)
        self.assertEqual(len(pattern.evidence), 2)
        self.assertAlmostEqual(pattern.confidence, 0.7)
        self.assertEqual(len(lib.patterns()), 1)

    def test_pa_single_failure_alone_learns_nothing(self):
        lib = PatternLibrary()
        learned = lib.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False)], EFFECTS)
        self.assertEqual(learned, [])
        self.assertEqual(lib.patterns(), [])

    def test_pa_single_success_alone_learns_nothing(self):
        lib = PatternLibrary()
        learned = lib.learn_from_episodes(
            [_ep("open_beta", {"at": "hall", "has_beta_key": True}, True)],
            EFFECTS)
        self.assertEqual(learned, [])

    def test_pa_wrong_direction_learns_nothing(self):
        # Tool-based fails while tool-less succeeds: the opposite of
        # the P-A contrast — no unreliable-tool-less-opener pattern.
        lib = PatternLibrary()
        episodes = [
            _ep("open_beta", {"at": "hall", "has_beta_key": True}, False),
            _ep("open_alpha", {"at": "hall"}, True),
        ]
        self.assertEqual(lib.learn_from_episodes(episodes, EFFECTS), [])

    def test_pa_failure_and_success_for_different_effects_learn_nothing(self):
        lib = PatternLibrary()
        effects = {"open_alpha": {"gate_open": True},
                   "open_beta": {"other_open": True}}
        episodes = [
            _ep("open_alpha", {"at": "hall"}, False),
            _ep("open_beta", {"at": "hall", "has_beta_key": True}, True),
        ]
        self.assertEqual(lib.learn_from_episodes(episodes, effects), [])

    def test_no_duplicate_relearn(self):
        lib = PatternLibrary()
        episodes = [
            _ep("open_alpha", {"at": "hall"}, False),
            _ep("open_beta", {"at": "hall", "has_beta_key": True}, True),
        ]
        first = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(first), 1)
        second = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(second, [])
        self.assertEqual(len(lib.patterns()), 1)

    def test_confidence_capped_at_0_9(self):
        lib = PatternLibrary()
        episodes = ([_ep("open_alpha", {"at": "hall"}, False)] * 5
                    + [_ep("open_beta", {"at": "hall", "has_beta_key": True},
                           True)] * 5)
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0].confidence, 0.9)


class TestLearnPB(unittest.TestCase):
    def test_pb_learned_on_success_then_failure_same_situation(self):
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        episodes = [_ep("use_gamma_tool", situation, True),
                    _ep("use_gamma_tool", situation, False)]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        pattern = learned[0]
        self.assertEqual(pattern.kind, "one_use_tool")
        self.assertEqual(pattern.effect, "gate_open")
        self.assertEqual(
            {r["name"]: r["features"] for r in pattern.roles},
            {"tool": {"breaks_after_first_use": True},
             "durable_tool": {"breaks_after_first_use": False}})
        self.assertIn(("tool", "produces", "gate_open"), pattern.relations)
        self.assertIn(("durable_tool", "produces", "gate_open"),
                      pattern.relations)
        self.assertAlmostEqual(pattern.confidence, 0.7)

    def test_pb_failure_before_success_learns_nothing(self):
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        episodes = [_ep("use_gamma_tool", situation, False),
                    _ep("use_gamma_tool", situation, True)]
        self.assertEqual(lib.learn_from_episodes(episodes, EFFECTS), [])

    def test_pb_success_only_learns_nothing(self):
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        episodes = [_ep("use_gamma_tool", situation, True)]
        self.assertEqual(lib.learn_from_episodes(episodes, EFFECTS), [])

    def test_pb_different_situations_learn_nothing(self):
        lib = PatternLibrary()
        episodes = [
            _ep("use_gamma_tool", {"at": "hall", "has_gamma_tool": True}, True),
            _ep("use_gamma_tool", {"at": "lab", "has_gamma_tool": True}, False),
        ]
        self.assertEqual(lib.learn_from_episodes(episodes, EFFECTS), [])

    def test_episodes_as_objects(self):
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        episodes = [
            SimpleNamespace(action="use_gamma_tool", situation=situation, ok=True),
            SimpleNamespace(action="use_gamma_tool", situation=situation, ok=False),
        ]
        learned = lib.learn_from_episodes(episodes, EFFECTS)
        self.assertEqual(len(learned), 1)
        self.assertEqual(learned[0].kind, "one_use_tool")


class TestPatternIds(unittest.TestCase):
    def test_ids_are_stable_hashes_with_no_world_names(self):
        lib_a = PatternLibrary()
        lib_a.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False, world="world_alpha"),
             _ep("open_beta", {"at": "hall", "has_beta_key": True}, True,
                 world="world_alpha")],
            EFFECTS)
        lib_b = PatternLibrary()
        lib_b.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False, world="world_beta"),
             _ep("open_beta", {"at": "hall", "has_beta_key": True}, True,
                 world="world_beta")],
            EFFECTS)
        id_a = lib_a.patterns()[0].id
        id_b = lib_b.patterns()[0].id
        # Same structure, different world label: the same id.
        self.assertEqual(id_a, id_b)
        self.assertEqual(len(id_a), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in id_a))
        for word in ("world_alpha", "world_beta", "open_alpha", "open_beta",
                     "alpha", "beta"):
            self.assertNotIn(word, id_a)
        # The id is derived from the structure, not assigned at random.
        pattern = lib_a.patterns()[0]
        rebuilt = AbstractPattern(
            name=pattern.name, kind=pattern.kind, effect=pattern.effect,
            roles=pattern.roles, relations=pattern.relations,
            evidence=pattern.evidence, confidence=pattern.confidence)
        self.assertEqual(rebuilt.id, pattern.id)


class TestMatch(unittest.TestCase):
    def _pa_library(self):
        lib = PatternLibrary()
        lib.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False),
             _ep("open_beta", {"at": "hall", "has_beta_key": True}, True)],
            EFFECTS)
        return lib

    def _pb_library(self):
        lib = PatternLibrary()
        situation = {"at": "hall", "has_gamma_tool": True}
        lib.learn_from_episodes(
            [_ep("use_gamma_tool", situation, True),
             _ep("use_gamma_tool", situation, False)],
            EFFECTS)
        return lib

    def test_pa_match_penalises_toolless_opener_when_alternative_exists(self):
        lib = self._pa_library()
        model = WorldModel()
        mult = lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                         model, all_actions=ALL_ACTIONS)
        self.assertEqual(mult, 6.0)

    def test_pa_match_no_penalty_without_alternative(self):
        lib = self._pa_library()
        model = WorldModel()
        only_toolless = [("open_alpha", {"at": "hall"}, {"gate_open": True})]
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model, all_actions=only_toolless),
            1.0)
        # No action list at all: the alternative cannot be confirmed.
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                      model),
            1.0)

    def test_pa_match_no_penalty_for_tooled_or_uncovered_actions(self):
        lib = self._pa_library()
        model = WorldModel()
        # The tool-based opener itself is never penalised by P-A.
        self.assertEqual(
            lib.match("open_beta", {"at": "hall", "has_beta_key": True},
                      {"gate_open": True}, model, all_actions=ALL_ACTIONS),
            1.0)
        # An action for an effect no pattern covers is never penalised.
        self.assertEqual(
            lib.match("open_alpha", {"at": "hall"}, {"other_open": True},
                      model, all_actions=ALL_ACTIONS),
            1.0)
        # An empty library penalises nothing.
        self.assertEqual(
            PatternLibrary().match("open_alpha", {"at": "hall"},
                                   {"gate_open": True}, model,
                                   all_actions=ALL_ACTIONS),
            1.0)

    def test_pb_match_penalises_single_use_suspect_tool(self):
        lib = self._pb_library()
        model = WorldModel()
        model.observe("has_gamma_tool", True, cycle=1)
        # Not yet suspect: no penalty.
        self.assertEqual(
            lib.match("use_gamma_tool", {"at": "hall", "has_gamma_tool": True},
                      {"gate_open": True}, model),
            1.0)
        model.mark_single_use_suspect("gamma_tool", cycle=2)
        self.assertEqual(
            lib.match("use_gamma_tool", {"at": "hall", "has_gamma_tool": True},
                      {"gate_open": True}, model),
            6.0)
        # A different, unsuspected tool for the same effect: no penalty.
        self.assertEqual(
            lib.match("use_delta_tool", {"at": "hall", "has_delta_tool": True},
                      {"gate_open": True}, model),
            1.0)

    def test_match_accepts_global_effect_map(self):
        lib = self._pa_library()
        model = WorldModel()
        mult = lib.match("open_alpha", {"at": "hall"}, EFFECTS, model,
                         all_actions=ALL_ACTIONS)
        self.assertEqual(mult, 6.0)


class TestTransferByStructure(unittest.TestCase):
    def test_pattern_applies_to_differently_named_actions(self):
        # Learned from alpha/beta-shaped episodes in one world...
        lib = PatternLibrary()
        lib.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False),
             _ep("open_beta", {"at": "hall", "has_beta_key": True}, True)],
            EFFECTS)
        # ...applies to differently-named actions with the same
        # structure in another world: a tool-less opener for the same
        # effect, with a tool-based alternative available.
        model = WorldModel()
        other_actions = [
            ("force_entry", {"at": "hall"}, {"gate_open": True}),
            ("use_epsilon_tool", {"at": "hall", "has_epsilon_tool": True},
             {"gate_open": True}),
        ]
        self.assertEqual(
            lib.match("force_entry", {"at": "hall"}, {"gate_open": True},
                      model, all_actions=other_actions),
            6.0)
        self.assertEqual(
            lib.match("use_epsilon_tool",
                      {"at": "hall", "has_epsilon_tool": True},
                      {"gate_open": True}, model, all_actions=other_actions),
            1.0)


class TestPatternPersistence(unittest.TestCase):
    def test_json_round_trip_preserves_patterns_and_matching(self):
        lib = PatternLibrary()
        lib.learn_from_episodes(
            [_ep("open_alpha", {"at": "hall"}, False),
             _ep("open_beta", {"at": "hall", "has_beta_key": True}, True),
             _ep("use_gamma_tool", {"at": "hall", "has_gamma_tool": True}, True),
             _ep("use_gamma_tool", {"at": "hall", "has_gamma_tool": True}, False)],
            EFFECTS)
        self.assertEqual(len(lib.patterns()), 2)
        blob = lib.to_json()
        import json as _json
        _json.dumps(blob)  # must be JSON-serialisable
        loaded = PatternLibrary.from_json(_json.loads(_json.dumps(blob)))
        self.assertEqual([p.id for p in loaded.patterns()],
                         [p.id for p in lib.patterns()])
        self.assertEqual([p.kind for p in loaded.patterns()],
                         ["unreliable_toolless_opener", "one_use_tool"])
        self.assertEqual([p.confidence for p in loaded.patterns()],
                         [p.confidence for p in lib.patterns()])
        # The loaded library still matches the same way.
        model = WorldModel()
        self.assertEqual(
            loaded.match("open_alpha", {"at": "hall"}, {"gate_open": True},
                         model, all_actions=ALL_ACTIONS),
            6.0)
        # A second round-trip is stable.
        self.assertEqual(PatternLibrary.from_json(loaded.to_json()).to_json(),
                         loaded.to_json())


if __name__ == "__main__":
    unittest.main()
