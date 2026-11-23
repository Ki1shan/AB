"""Phase 12 soak (design §3, ADR P12-2): >=10,000 cycles in one process,
mixed workload, bounded-memory proof.

Run:  python -m ab.eval.soak [--cycles N] [--record PATH]

Workload, in one process:
  * a long-lived brain that runs many treasure-hunt goals (fresh
    RoomWorld per hunt), research goals, and an interruption leg --
    this is the brain whose structures are sampled, so long-lived
    growth is what is measured;
  * periodic fresh-brain hunts (a fresh brain per hunt alone would not
    test long-lived growth, so these are a minority leg);
  * one impossible-goal leg and one budget leg, so non-"done" stop
    reasons are observed as well as "done".

Sampling every 500 global cycles: VmRSS from /proc/self/status plus
the sizes of the bounded structures of the long-lived brain
(learner episodes cap 500, audit/event log cap 1,000, LTM record
count, WM size, world-model event count, pattern library size, rule
count, and — Phase 12 W7 — the three residual accumulations the
first re-soak left open: cached goal Thoughts, world-model decay
bookkeeping, simulation-revision history).

"Bounded" threshold, fixed BEFORE the run (follow-up brief):
  * capped stores never exceed their caps;
  * after a 500-cycle warm-up, the linear-fit slope of RSS over the
    last 5,000 cycles is < 1% of baseline RSS (RSS at the warm-up
    sample) per 1,000 cycles;
  * no store grows linearly with cycle count: each sampled store's
    linear-fit slope over the last 5,000 cycles is < 1% of its
    warm-up-end value per 1,000 cycles (stores whose warm-up value
    is 0: < 1 unit per 1,000 cycles), or <= 0.

This script only MEASURES. It never modifies production code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ab.body.environment import RoomWorld, room_world_actions  # noqa: E402
from ab.body.providers import MemoryKnowledgeProvider  # noqa: E402
from ab.body.research import Researcher  # noqa: E402
from ab.brain import Brain  # noqa: E402
from ab.core.audit import MAX_EVENTS  # noqa: E402
from ab.core.learning import MAX_EPISODES  # noqa: E402

TARGET_CYCLES = 10_000
SAMPLE_EVERY = 500
WARMUP = 500
SLOPE_WINDOW = 5_000

# Research subjects: (subject, relation, value, country-wording variants)
# Two domains, same claim worded differently (mirror rule: identical
# sentences from both domains do not corroborate).
CAPITALS = [
    ("paris", "France"), ("berlin", "Germany"), ("tokyo", "Japan"),
    ("madrid", "Spain"), ("rome", "Italy"), ("london", "United Kingdom"),
    ("ottawa", "Canada"), ("canberra", "Australia"), ("delhi", "India"),
    ("cairo", "Egypt"),
]


def _research_pages(subject: str, country: str) -> dict:
    # Per-subject URLs: the provider is keyed by URL, so every subject
    # needs its own pages -- sharing one URL across subjects would let
    # a later subject's text overwrite an earlier one's (found in
    # diagnosis: the research goal then fails, the fact never verifies).
    return {
        f"https://alpha.org/{subject}": f"<p>{subject.title()} is the capital of {country}.</p>",
        f"https://beta.org/{subject}": (
            f"<html><script>x()</script>"
            f"<p>{subject.title()} is the capital city of {country}.</p></html>"),
    }


def _rss_kib() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found in /proc/self/status")


def _wm_list(obj, name):
    v = getattr(obj, name, None)
    if callable(v):
        v = v()
    return list(v or [])


def _linfit_slope(xs, ys) -> float:
    """Least-squares slope of y on x (per unit x)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


class Soak:
    def __init__(self, target: int = TARGET_CYCLES):
        self.target = target
        self.global_cycles = 0
        self.samples: list[dict] = []
        self.next_sample = SAMPLE_EVERY
        self.stop_reasons: Counter = Counter()
        # Per-leg GOAL outcomes (the leg goal's own status after the
        # leg). This is ground truth for "did the goal complete":
        # Brain.run()'s return is poisoned once ANY goal in the
        # brain's history has status "failed" -- run() checks
        # any(g.status == "failed") before the all-done branch, so a
        # single failed goal makes every later run() return
        # "impossible" even when the leg's own goal completed
        # (observed in diagnosis). Both are recorded.
        self.goal_outcomes: Counter = Counter()
        self.legs: Counter = Counter()
        self.interrupts_observed = 0
        self.research_goals_done = 0
        self.t0 = time.time()
        pages = {}
        for subj, country in CAPITALS:
            pages.update(_research_pages(subj, country))
        self.researcher = Researcher(MemoryKnowledgeProvider(pages), None)
        # Long-lived brain: cycle budget far above the soak target so
        # step() never stops it on budget; legs impose their own limits.
        self.brain = Brain(
            room_world_actions(), researcher=self.researcher,
            granted={"compute", "environment", "network"},
            max_cycles=1_000_000)
        self._research_idx = 0
        self._research_done_for_thousand: set[int] = set()

    # ------------------------------------------------------------ sampling
    def _snapshot(self, cycle: int) -> dict:
        b = self.brain
        return {
            "cycle": cycle,
            "rss_kib": _rss_kib(),
            "episodes": len(b.learner.episodes),
            "audit_events": len(b.events),
            "ltm_facts": len(b.ltm.all_facts()),
            "wm_size": len(b.wm.contents()),
            "wm_events": len(_wm_list(b.world_model, "events")),
            "patterns": len(_wm_list(b.patterns, "patterns")),
            "rules": len(b.rules.rules()),
            # Phase 12 (P12-W7): the three residual accumulations the
            # first re-soak left open. goal_thoughts is O(1) per
            # active/suspended goal once _reset_after_goal drops the
            # cached Thought with the goal; decay_bookkeeping is O(1)
            # once decay() prunes dead cycles; revision_history is
            # capped at MAX_REVISION_HISTORY=1,000, oldest-first.
            "goal_thoughts": len(b._goal_thoughts),
            "decay_bookkeeping": (len(b.world_model._observed_in_cycle)
                                  + len(b.world_model._decayed_cycles)),
            "revision_history": len(b.simulation_revision.history),
        }

    def _maybe_sample(self) -> None:
        while self.global_cycles >= self.next_sample:
            self.samples.append(self._snapshot(self.next_sample))
            self.next_sample += SAMPLE_EVERY

    def _advance(self, brain: Brain, before: int) -> None:
        self.global_cycles += brain.cycle - before
        self._maybe_sample()

    # ------------------------------------------------------------ legs
    def _treasure_leg(self, brain: Brain, env=None) -> str:
        env = env if env is not None else RoomWorld()
        before = brain.cycle
        goal = brain.add_goal("get the treasure", {"has_treasure": True})
        # run()'s limit is an ABSOLUTE brain cycle, not a per-leg
        # budget: a long-lived brain past cycle N would return
        # "budget_exhausted" immediately on run(env, N) with zero new
        # cycles (found in the smoke run: 737 empty legs). Pass
        # cycle+N so each leg really runs up to N fresh cycles.
        reason = brain.run(env, brain.cycle + 80)
        self._advance(brain, before)
        self.stop_reasons[reason] += 1
        self.goal_outcomes[goal.status] += 1
        self.legs["treasure"] += 1
        return reason

    def _fresh_brain_leg(self) -> str:
        fresh = Brain(room_world_actions(), granted={"compute", "environment"},
                      max_cycles=200)
        return self._treasure_leg(fresh)

    def _interrupt_leg(self) -> str:
        """Mid-task interruption on the long-lived brain: a few steps,
        an urgent percept (suspends the goal), resume, finish."""
        b, env = self.brain, RoomWorld()
        before = b.cycle
        goal = b.add_goal("get the treasure", {"has_treasure": True})
        for _ in range(5):
            if b.goals.active() is None:
                break
            b.step(env)
        b.perceive_text("urgent: halt work at once", urgency=0.95)
        b.step(env)
        suspended = [g for g in b.goals.all() if g.status == "suspended"]
        if any(e["kind"] == "interrupt" for e in b.events):
            self.interrupts_observed += 1
        for g in suspended:
            b.resume_goal(g.id)
        reason = b.run(env, b.cycle + 80)
        self._advance(b, before)
        self.stop_reasons[reason] += 1
        self.goal_outcomes[goal.status] += 1
        self.legs["interrupt"] += 1
        return reason

    def _research_leg(self) -> str:
        subj, country = CAPITALS[self._research_idx % len(CAPITALS)]
        self._research_idx += 1
        b = self.brain
        before = b.cycle
        pages = _research_pages(subj, country)
        goal = b.add_research_goal(subj, "capital_of", list(pages))
        reason = b.run(None, b.cycle + 40)
        self._advance(b, before)
        self.stop_reasons[reason] += 1
        self.goal_outcomes[goal.status] += 1
        self.legs["research"] += 1
        if b.ask(subj, "capital_of").best:
            self.research_goals_done += 1
        return reason

    def _impossible_leg(self) -> str:
        fresh = Brain(room_world_actions(), granted={"compute", "environment"},
                      max_cycles=200)
        env = RoomWorld()
        before = fresh.cycle
        goal = fresh.add_goal("find the zorblax", {"has_zorblax": True})
        reason = fresh.run(env, fresh.cycle + 40)
        self._advance(fresh, before)
        self.stop_reasons[reason] += 1
        self.goal_outcomes[goal.status] += 1
        self.legs["impossible"] += 1
        return reason

    def _budget_leg(self) -> str:
        fresh = Brain(room_world_actions(), granted={"compute", "environment"},
                      max_cycles=200)
        env = RoomWorld()
        before = fresh.cycle
        goal = fresh.add_goal("get the treasure", {"has_treasure": True})
        reason = fresh.run(env, 3)
        self._advance(fresh, before)
        self.stop_reasons[reason] += 1
        self.goal_outcomes[goal.status] += 1
        self.legs["budget"] += 1
        return reason

    # ------------------------------------------------------------ run
    def run(self) -> dict:
        # Non-"done" stops first, so they are observed regardless of
        # how the main loop goes.
        self._impossible_leg()
        self._budget_leg()
        leg = 0
        while self.global_cycles < self.target:
            leg += 1
            thousand = self.global_cycles // 1000
            if thousand not in self._research_done_for_thousand and self.global_cycles >= 500:
                self._research_done_for_thousand.add(thousand)
                self._research_leg()
            elif leg % 10 == 0:
                self._interrupt_leg()
            elif leg % 25 == 0:
                self._fresh_brain_leg()
            else:
                self._treasure_leg(self.brain)
            if self.global_cycles >= self.target:
                break
        # Final sample at/after the target if the last boundary was crossed.
        self._maybe_sample()
        return self._evaluate()

    # ------------------------------------------------------------ verdict
    def _evaluate(self) -> dict:
        samples = self.samples
        warm = next((s for s in samples if s["cycle"] == WARMUP), samples[0])
        baseline_rss = warm["rss_kib"]
        window = [s for s in samples if s["cycle"] > self.global_cycles - SLOPE_WINDOW]
        if len(window) < 2:
            window = samples[-2:]
        xs = [s["cycle"] for s in window]
        rss_slope_per_1000 = _linfit_slope(xs, [s["rss_kib"] for s in window]) * 1000
        rss_threshold = 0.01 * baseline_rss
        stores = ["episodes", "audit_events", "ltm_facts", "wm_size",
                  "wm_events", "patterns", "rules",
                  "goal_thoughts", "decay_bookkeeping", "revision_history"]
        store_verdicts = {}
        for name in stores:
            slope = _linfit_slope(xs, [s[name] for s in window]) * 1000
            v0 = warm[name]
            threshold = 0.01 * v0 if v0 > 0 else 1.0
            store_verdicts[name] = {
                "slope_per_1000_cycles": slope,
                "threshold_per_1000_cycles": threshold,
                "warmup_value": v0,
                "end_value": samples[-1][name],
                "max_value": max(s[name] for s in samples),
                "bounded": slope < threshold or slope <= 0,
            }
        caps_ok = (max(s["episodes"] for s in samples) <= MAX_EPISODES
                   and max(s["audit_events"] for s in samples) <= MAX_EVENTS)
        rss_bounded = rss_slope_per_1000 < rss_threshold
        bounded = caps_ok and rss_bounded and all(
            v["bounded"] for v in store_verdicts.values())
        peak_rss = max(s["rss_kib"] for s in samples)
        mid = samples[len(samples) // 2]
        record = {
            "soak": "phase12",
            "target_cycles": self.target,
            "cycles_run": self.global_cycles,
            "wall_seconds": round(time.time() - self.t0, 1),
            "caps": {"learner_episodes": MAX_EPISODES, "audit_events": MAX_EVENTS},
            "thresholds": {
                "warmup_cycles": WARMUP,
                "slope_window_cycles": SLOPE_WINDOW,
                "rss_slope_limit_per_1000_cycles_kib": rss_threshold,
                "rss_slope_limit_basis": "1% of baseline RSS (cycle-500 sample)",
                "store_slope_limit_per_1000_cycles":
                    "1% of warm-up value (1 unit if warm-up value is 0)",
            },
            "baseline_rss_kib": baseline_rss,
            "peak_rss_kib": peak_rss,
            "rss_slope_per_1000_cycles_kib": rss_slope_per_1000,
            "rss_bounded": rss_bounded,
            "caps_respected": caps_ok,
            "stores": store_verdicts,
            "bounded": bounded,
            "stop_reasons": dict(self.stop_reasons),
            "goal_outcomes": dict(self.goal_outcomes),
            "notes": [
                "Phase 12 fix (ADR P12-10/P12-11): audit_events "
                "(brain.events) is capped at MAX_EVENTS in-cycle by "
                "_event() (brain.py), oldest-first, and the world-model "
                "log is capped the same way via _append_event() "
                "(ab/world/model.py); the save-time prune remains as "
                "belt-and-braces.",
                "Phase 12 W7 (ADR P12-12): the sampler also tracks "
                "goal_thoughts (cached goal Thoughts, dropped with "
                "the goal by _reset_after_goal), decay_bookkeeping "
                "(world-model _observed_in_cycle + _decayed_cycles, "
                "pruned at the end of each decay), and "
                "revision_history (SimulationRevision.history, capped "
                "at 1,000 oldest-first). These were the residual "
                "per-cycle accumulations the first re-soak left open.",
                "Brain.run() returns 'impossible' for every later leg "
                "once any goal in history has status 'failed' "
                "(any-failed check precedes the all-done branch); "
                "per-leg goal_outcomes is the ground truth for whether "
                "each leg's own goal completed.",
            ],
            "legs": dict(self.legs),
            "interrupts_observed": self.interrupts_observed,
            "research_goals_answered": self.research_goals_done,
            "curve": {
                "start": samples[0],
                "mid": mid,
                "end": samples[-1],
            },
            "samples": samples,
        }
        return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", type=int, default=TARGET_CYCLES)
    ap.add_argument("--record", default=None,
                    help="record path (default: docs/phase12-soak-record.json)")
    args = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    record_path = args.record or os.path.join(repo, "docs", "phase12-soak-record.json")
    soak = Soak(target=args.cycles)
    record = soak.run()
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"cycles run : {record['cycles_run']} (target {record['target_cycles']})")
    print(f"wall time  : {record['wall_seconds']}s")
    print(f"peak RSS   : {record['peak_rss_kib']} KiB "
          f"(baseline {record['baseline_rss_kib']} KiB)")
    print(f"RSS slope  : {record['rss_slope_per_1000_cycles_kib']:.1f} KiB/1000 cycles "
          f"(limit {record['thresholds']['rss_slope_limit_per_1000_cycles_kib']:.1f})")
    print(f"caps respected: {record['caps_respected']} | "
          f"RSS bounded: {record['rss_bounded']} | BOUNDED: {record['bounded']}")
    print(f"stop reasons: {record['stop_reasons']}")
    print(f"goal outcomes: {record['goal_outcomes']}")
    print(f"legs: {record['legs']} | interrupts: {record['interrupts_observed']} | "
          f"research answered: {record['research_goals_answered']}")
    for name, v in record["stores"].items():
        print(f"  {name:<13} warmup={v['warmup_value']:<5} end={v['end_value']:<5} "
              f"max={v['max_value']:<5} slope/1000={v['slope_per_1000_cycles']:+.2f} "
              f"bounded={v['bounded']}")
    print(f"record: {record_path}")
    return 0 if record["bounded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
