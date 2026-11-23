"""Phase 9 §4.2: wrong-promoted-pattern retirement (W2, carried since
Phase 6 — must FIRE).

A P-D pattern (``requires has_mould: pressed``) is promoted in world
X through the normal consolidation path, then contradicted >= 3 times
in world Y (press SUCCEEDS without has_mould, which the pattern's
prediction rules out). The third contradiction retires the pattern:
it leaves the library, the planner stops applying it, the retirement
is a ``pattern_retired`` event + a ``retired`` record, and the same
evidence can never resurrect it — not via re-consolidation, not via
re-learning, not across save/load.

The contradicting-evidence trigger is the count: observations are
recorded with ``Brain.record_pattern_contradiction`` (each grounded
in a real recorded success episode in world Y); at
``PATTERN_CONTRADICTION_THRESHOLD`` (3) the brain retires the pattern
itself. No automatic harvesting from every episode is wired in —
existing evaluation runs must not change behaviour (their pinned
counts depend on it); the trigger fires only on observations fed to
the API, which is what the execution eval's carried-item leg would
feed it.

Stdlib only, deterministic.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from ab.brain import Brain
from ab.core.learning import Episode
from ab.planning.planner import Action
from ab.world.patterns import PatternLibrary


def _press_actions() -> list[Action]:
    """World X / world Y share one action shape: press claims the
    effect ``pressed``; fetch_mould supplies the key has_mould."""
    return [
        Action("press", {"at": "bench"}, {"pressed": True}, cost=1.0),
        Action("fetch_mould", {"at": "bench"}, {"has_mould": True},
               cost=1.0),
    ]


def _promote_pd_pattern(brain: Brain):
    """World X evidence through the normal consolidation path: press
    fails twice without has_mould (two distinct situations) and
    succeeds once with it. Returns the promoted P-D pattern."""
    brain.learner.record(Episode(1, "press", {"at": "bench"}, False))
    brain.learner.record(
        Episode(2, "press", {"at": "bench", "has_mould": False}, False))
    brain.learner.record(
        Episode(3, "press", {"at": "bench", "has_mould": True}, True))
    brain._consolidation_pass()
    found = [p for p in brain.patterns.patterns()
             if p.kind == "requires_precondition"]
    assert len(found) == 1, (
        f"expected exactly one promoted P-D pattern, got {found!r}")
    return found[0]


def _pd_match(brain: Brain) -> float:
    """The planner's pattern multiplier for press without has_mould:
    6.0 while the P-D pattern applies, 1.0 once it is retired."""
    return brain.patterns.match(
        "press", {"at": "bench"}, {"pressed": True}, brain.world_model,
        [(a.name, a.pre, a.eff) for a in brain.planner.actions],
        {"at": "bench"})


class TestContradictingEvidenceRetirement(unittest.TestCase):
    def test_three_contradictions_retire_promoted_pattern(self):
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_pattern(brain)
        pid = pattern.id
        self.assertEqual(pattern.name, "requires_has_mould:pressed")
        self.assertEqual(pattern.effect, "pressed")
        # The pattern is applied in world X: press without the key
        # is penalised x6.0.
        self.assertEqual(_pd_match(brain), 6.0)

        # World Y: press SUCCEEDS three times without has_mould.
        # Each contradiction is grounded in a real recorded episode.
        for i in range(3):
            brain.learner.record(
                Episode(10 + i, "press", {"at": "bench"}, True))
            result = brain.record_pattern_contradiction(
                pid, f"press succeeded without has_mould in world Y "
                     f"(observation {i + 1})")
            self.assertEqual(result["pattern_id"], pid)
            self.assertEqual(result["contradictions"], i + 1)
            if i < 2:
                self.assertFalse(result["retired"])
                self.assertIn(pattern, brain.patterns.patterns())
                self.assertEqual(_pd_match(brain), 6.0)
            else:
                self.assertTrue(result["retired"])

        # FIRED: retired, no longer applied, recorded twice over.
        self.assertNotIn(pattern, brain.patterns.patterns())
        self.assertEqual([p.id for p in brain.patterns.patterns()
                          if p.kind == "requires_precondition"], [])
        self.assertEqual(_pd_match(brain), 1.0)
        retired = brain.retired_patterns()
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["id"], pid)
        self.assertEqual(retired[0]["kind"], "requires_precondition")
        self.assertIn("3 observations", retired[0]["reason"])
        self.assertEqual(
            sum(1 for e in brain.events if e["kind"] == "pattern_retired"),
            1)
        self.assertEqual(
            sum(1 for e in brain.events
                if e["kind"] == "pattern_contradicted"), 3)
        self.assertTrue(brain.patterns.is_retired(pid))

        # The same evidence can never resurrect it: re-consolidation
        # (which re-learns the P-D candidate from the same episodes)
        # leaves the pattern retired.
        brain._consolidation_pass()
        self.assertEqual([p.id for p in brain.patterns.patterns()
                          if p.kind == "requires_precondition"], [])
        self.assertEqual(len(brain.retired_patterns()), 1)
        self.assertEqual(_pd_match(brain), 1.0)

        # Retirement survives save/load (the retired record rides in
        # world.json with the library).
        folder = tempfile.mkdtemp(prefix="ab-phase9-retire-")
        brain.save(folder)
        loaded = Brain(_press_actions(), granted={"compute", "environment"})
        loaded.load(folder)
        self.assertEqual([p.id for p in loaded.patterns.patterns()
                          if p.kind == "requires_precondition"], [])
        self.assertEqual(len(loaded.retired_patterns()), 1)
        self.assertEqual(loaded.retired_patterns()[0]["id"], pid)
        self.assertTrue(loaded.patterns.is_retired(pid))

    def test_explicit_human_retire_pattern(self):
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        pattern = _promote_pd_pattern(brain)
        self.assertTrue(brain.retire_pattern(
            pattern.id, "human: the mould is not actually required"))
        self.assertEqual(_pd_match(brain), 1.0)
        self.assertEqual(len(brain.retired_patterns()), 1)
        self.assertEqual(
            brain.retired_patterns()[0]["reason"],
            "human: the mould is not actually required")
        self.assertEqual(
            sum(1 for e in brain.events if e["kind"] == "pattern_retired"),
            1)
        # Idempotent: a second retirement, and an unknown id, fail.
        self.assertFalse(brain.retire_pattern(pattern.id, "again"))
        self.assertFalse(brain.retire_pattern("no-such-pattern", "x"))
        self.assertEqual(len(brain.retired_patterns()), 1)

    def test_retired_pattern_never_relearned_at_library_level(self):
        lib = PatternLibrary()
        episodes = [
            {"action": "press", "situation": {"at": "bench"},
             "outcome": "failed", "world": "X"},
            {"action": "press",
             "situation": {"at": "bench", "has_mould": False},
             "outcome": "failed", "world": "X"},
            {"action": "press",
             "situation": {"at": "bench", "has_mould": True},
             "outcome": "success", "world": "X"},
        ]
        learned = lib.learn_from_episodes(
            episodes, {"press": {"pressed": True},
                       "fetch_mould": {"has_mould": True}})
        candidates = [p for p in learned if p.candidate]
        self.assertEqual(len(candidates), 1)
        self.assertIsNotNone(lib.add_pattern(candidates[0]))
        pid = candidates[0].id
        self.assertIsNotNone(lib.retire_pattern(pid, "test"))
        # Re-learning the same episodes still DETECTS the structure
        # (it returns the candidate) — but the retirement gate is at
        # admission, so it is never re-ADMITTED to the library.
        relearned = lib.learn_from_episodes(
            episodes, {"press": {"pressed": True},
                       "fetch_mould": {"has_mould": True}})
        self.assertEqual(
            [p.id for p in relearned if p.id == pid], [pid])
        self.assertTrue(all(p.candidate for p in relearned
                            if p.id == pid))
        self.assertIsNone(lib.add_pattern(candidates[0]))
        self.assertEqual(
            [p.id for p in lib.patterns() if p.id == pid], [])
        self.assertEqual(len(lib.retired_patterns()), 1)


class TestRetireRule(unittest.TestCase):
    def _failing_brain(self) -> Brain:
        brain = Brain(_press_actions(), granted={"compute", "environment"})
        for i in range(3):
            brain.learner.record(Episode(i + 1, "press", {"at": "bench"},
                                         False))
        return brain

    def test_retire_pending_rule(self):
        brain = self._failing_brain()
        pending = brain.pending_rules()
        self.assertEqual(len(pending), 1)
        rid = pending[0]["rule_id"]
        self.assertEqual(pending[0]["action"], "press")
        self.assertTrue(brain.retire_rule(rid, "human: press is fine"))
        self.assertEqual(brain.rules.get(rid).status, "retired")
        self.assertEqual(
            sum(1 for e in brain.events if e["kind"] == "rule_retired"), 1)
        self.assertFalse(brain.retire_rule(rid, "again"))
        self.assertFalse(brain.retire_rule("no-such-rule", "x"))

    def test_retire_approved_rule(self):
        brain = self._failing_brain()
        rid = brain.pending_rules()[0]["rule_id"]
        self.assertTrue(brain.approve_rule(rid))
        self.assertEqual(
            [d["rule_id"] for d in brain.learner.approved_rules()], [rid])
        self.assertTrue(brain.retire_rule(rid, "human: approval was wrong"))
        self.assertEqual(
            [d["rule_id"] for d in brain.learner.approved_rules()], [])
        self.assertEqual(brain.rules.get(rid).status, "retired")
        self.assertEqual(
            sum(1 for e in brain.events if e["kind"] == "rule_retired"), 1)
        self.assertFalse(brain.retire_rule(rid, "again"))


if __name__ == "__main__":
    unittest.main()
