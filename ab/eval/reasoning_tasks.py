"""Phase 4 discriminating held-out tasks (Phase 3 blocker #5).

The Phase 3 disconnected experiment does NOT separate the three brain
variants (full / no-learning / non-integrated): on the treasure task
and on disconnected world A the numbers come out identical, so no
winner can be declared. These three tasks are designed to separate
them, each with a pass/fail criterion asserted on behaviour (events +
state), never on printed output:

(a) ``deduction_3hop`` -- a 3-hop mixed-relation deduction chain
    (``velkra is_a torvin``, ``torvin is_a sarnak``,
    ``sarnak located_in kelmor``; all taught as verified teacher
    facts with unfamiliar names the brain was never taught). The
    brain must derive ``velkra located_in kelmor`` with a 3-step
    proof chain and exact confidence ``min premise x 0.95`` per extra
    step (all premises 1.0, so 0.9025). A rule-match-only brain that
    only does single-relation closure cannot solve it: single-relation
    closure from ``velkra`` reaches ``torvin``/``sarnak`` via ``is_a``
    and nothing via ``located_in``.
(b) ``contradiction`` -- two conflicting verified facts on the
    functional relation ``boils_at`` (``lyssan boils_at 100c`` from
    teacher, then ``lyssan boils_at 90c`` from a second teacher
    source, via the LTM conflict path). The brain is asked to decide.
    Pass = contradiction detected in-cycle (a contradiction event),
    both claims marked disputed in LTM, a research goal created for
    (``lyssan``, ``boils_at``), and neither side silently picked
    (``ask`` returns both values, neither fact verified).
(c) ``hypothesis_abduction`` -- an unfamiliar environment
    (``SurpriseWorld``) where the action ``heat_vapour`` is predicted
    to make the vapour hot but the vapour stays cold (a surprise).
    The brain must generate a ranked hypothesis explaining the
    surprise, with a test, and the test must run with its outcome
    recorded (corroborated or falsified -- either is a pass if the
    evidence trail exists; generating no hypothesis at all is a fail)
    AND the goal must complete (the vapour ends up hot). The goal
    leg is the Phase 5 (design SS4.2) fix for the Phase 4 stall: the
    criterion change is the mandated fix, not a weakening -- the
    task's goal is completion, and completion was previously blocked
    by the bug being fixed (a violated prediction revising the model
    must clear the pair-bans keyed to the old model and reset the
    stall window, so the brain can recover by warming the coil).

Each task is run for all three variants, built the same way as the
harness baselines (monkeypatching/subclassing, ``brain.py`` untouched):

- ``full``: the brain as built;
- ``no_learning``: the learner replaced by a no-op (planner costs
  never adapt, no episodes are recorded);
- ``non_integrated``: a ``Brain`` subclass overriding
  ``_reason_from_wm`` with a no-op, so no in-cycle WM read /
  inference runs.

The numbers are recorded and NO winner is declared without data.
A failed task is a recorded result, not a crash: every runner returns
a result dict with ``passed`` plus per-leg evidence, and
:func:`run_discriminating_tasks` never raises for a brain failure.

The task worlds/facts live in this file only; nothing is hardcoded
into the brain. The names (``velkra``, ``torvin``, ``sarnak``,
``kelmor``, ``lyssan``, ``vapour``) appear nowhere else in the repo:
they are held-out by construction, not development examples (no
Paris/cat/water, no RoomWorld). No LLM, stdlib only.
"""
from __future__ import annotations

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action

# ------------------------------------------------------------------ variants
# Mirrors ab.eval.harness._run_baselines (same monkeypatching approach,
# same definitions): no_learning replaces brain.learner with a no-op of
# the same public API; non_integrated subclasses Brain and overrides
# _reason_from_wm with a no-op. Kept in this module so the tasks and
# the harness run the same variants.


class _NoOpLearner:
    """Learner-shaped no-op: ``record`` does nothing, ``propose_rules``
    returns ``[]``. Planner costs therefore never adapt, and no
    episodes are ever recorded."""

    def __init__(self, *args, **kwargs):
        self.episodes = []

    def record(self, ep):
        pass

    def propose_rules(self):
        return []

    def episodes_for_induction(self):
        # Phase 4: Brain._reason_from_wm reads the learner's episodes
        # for induction every cycle. The no-op learner records none,
        # so this is faithfully [].
        return []

    def failures_for(self, action, situation, args=None):
        # Phase 10.5 (P10.5-7): the no-op learner records no
        # failures, so no plan is ever a loop plan for it.
        return 0

    def to_json(self):
        return {"episodes": [], "counts": []}


def _full_factory(actions, granted):
    return Brain(actions, granted=granted)


def _no_learning_factory(actions, granted):
    brain = Brain(actions, granted=granted)
    brain.learner = _NoOpLearner()
    return brain


class _NonIntegratedBrain(Brain):
    def _reason_from_wm(self):          # no in-cycle WM read / inference
        return []


def _non_integrated_factory(actions, granted):
    return _NonIntegratedBrain(actions, granted=granted)


def make_variant_factories() -> dict:
    """The three brain variants, keyed by name."""
    return {"full": _full_factory,
            "no_learning": _no_learning_factory,
            "non_integrated": _non_integrated_factory}


# ------------------------------------------------------------------ helpers
def _proof_steps(proof) -> int | None:
    """Number of steps in a proof: a Proof object's ``steps`` list, or
    the ``" -> "``-joined explain string. None when unparseable."""
    if proof is None:
        return None
    steps = getattr(proof, "steps", None)
    if isinstance(steps, list):
        return len(steps)
    if isinstance(proof, str):
        return proof.count(" -> ") + 1 if proof else 0
    return None


def _proof_relations(proof) -> tuple | None:
    """Relation path of a proof as a tuple, e.g.
    ``("is_a", "is_a", "located_in")``. None when unparseable."""
    if proof is None:
        return None
    steps = getattr(proof, "steps", None)
    if isinstance(steps, list):
        return tuple(getattr(s, "relation", "?") for s in steps)
    if isinstance(proof, str):
        rels = []
        for seg in proof.split(" -> "):
            parts = seg.strip().split()
            if len(parts) == 3:
                rels.append(parts[1])
        return tuple(rels) if rels else None
    return None


def _norm_derivation(d) -> dict | None:
    """Normalise a derivation (dict from ``Brain.last_inferences`` or a
    ``ReasoningEngine.Derivation`` object) to a plain dict, or None."""
    if isinstance(d, dict):
        return {"subject": str(d.get("subject", "")),
                "relation": str(d.get("relation", "")),
                "value": str(d.get("value", "")),
                "proof": d.get("proof"),
                "confidence": d.get("confidence"),
                "status": str(d.get("status", "derived"))}
    subject = getattr(d, "subject", None)
    if subject is None:
        return None
    return {"subject": str(subject),
            "relation": str(getattr(d, "relation", "")),
            "value": str(getattr(d, "value", "")),
            "proof": getattr(d, "proof", None),
            "confidence": getattr(d, "confidence", None),
            "status": str(getattr(d, "status", "derived"))}


def _find_derivation(brain, subject, relation, value) -> dict | None:
    """Find a derivation for (subject, relation, value) in the brain's
    observable reasoning output, in order:

    1. ``brain.last_inferences`` (the in-cycle derivation record);
    2. the ``inferred:<subject>|<relation>`` state entry (value,
       confidence, and proof string in provenance);
    3. a stored reasoning result (``last_reasoning`` /
       ``last_reasoning_result``), when the wiring exposes one.

    Returns a normalised derivation dict, or None. The search reads
    brain-observable state only; it never re-derives anything itself,
    so a brain whose in-cycle reasoning is disabled (non-integrated)
    cannot produce a hit here.
    """
    target = (subject, relation, value)
    for raw in (getattr(brain, "last_inferences", None) or []):
        d = _norm_derivation(raw)
        if d and (d["subject"], d["relation"], d["value"]) == target:
            return d
    entry = brain.state.get(f"inferred:{subject}|{relation}")
    if entry is not None and str(entry.value) == value:
        return {"subject": subject, "relation": relation, "value": value,
                "proof": getattr(entry, "provenance", None),
                "confidence": getattr(entry, "confidence", None),
                "status": "derived"}
    for attr in ("last_reasoning", "last_reasoning_result"):
        result = getattr(brain, attr, None)
        if result is None:
            continue
        for raw in (getattr(result, "derivations", None) or []):
            d = _norm_derivation(raw)
            if d and (d["subject"], d["relation"], d["value"]) == target:
                return d
    return None


def _has_event(brain, *needles: str) -> dict | None:
    """First brain event whose KIND contains any needle
    (case-insensitive), or None. Kind-only on purpose: the routine
    ``reason`` summary mentions e.g. ``0 contradiction(s)`` in its
    detail, and matching detail text would report a contradiction
    that never happened."""
    for e in brain.events:
        kind = str(e.get("kind", "")).lower()
        if any(n.lower() in kind for n in needles):
            return {"kind": e.get("kind"), "detail": e.get("detail")}
    return None


def _research_goal_for(brain, subject, relation) -> dict | None:
    """A research goal for (subject, relation), if one exists: target
    key ``known:<subject>|<relation>`` or a ``learn <relation> of
    <subject>`` description (the shape ``add_research_goal`` makes)."""
    key = f"known:{subject}|{relation}"
    for g in brain.goals.all():
        if key in g.target:
            return {"id": g.id, "description": g.description,
                    "status": g.status, "target": dict(g.target)}
        if g.description == f"learn {relation} of {subject}":
            return {"id": g.id, "description": g.description,
                    "status": g.status, "target": dict(g.target)}
    return None


# ------------------------------------------------------------------ (a) 3-hop deduction
DEDUCTION_FACTS = (("velkra", "is_a", "torvin"),
                   ("torvin", "is_a", "sarnak"),
                   ("sarnak", "located_in", "kelmor"))
DEDUCTION_SUBJECT = "velkra"
DEDUCTION_RELATION = "located_in"
DEDUCTION_VALUE = "kelmor"
# All premises are teacher facts (confidence 1.0); a 3-step chain costs
# 5% per extra step: 1.0 * 0.95 ** 2 = 0.9025 (Inference.proof_confidence).
DEDUCTION_EXPECTED_CONFIDENCE = 0.9025
DEDUCTION_EXPECTED_PATH = ("is_a", "is_a", "located_in")


def run_deduction_task(brain_factory) -> dict:
    """Task (a): teach the 3-hop chain, seed WM with the subject, run
    one cycle, and look for the derived conclusion in the brain's
    in-cycle reasoning output."""
    brain = brain_factory([], {"compute"})
    taught = [brain.teach_fact(s, r, v, "teacher")
              for s, r, v in DEDUCTION_FACTS]
    cycle_before = brain.cycle
    brain.perceive_text(DEDUCTION_SUBJECT)
    brain.step()
    deriv = _find_derivation(brain, DEDUCTION_SUBJECT,
                             DEDUCTION_RELATION, DEDUCTION_VALUE)
    steps = _proof_steps(deriv["proof"]) if deriv else None
    path = _proof_relations(deriv["proof"]) if deriv else None
    conf = deriv["confidence"] if deriv else None
    conf_ok = (conf is not None
               and abs(float(conf) - DEDUCTION_EXPECTED_CONFIDENCE) < 1e-9)
    legs = {
        "derived": deriv is not None,
        "proof_has_3_steps": steps == 3,
        "proof_path_is_is_a_is_a_located_in": path == DEDUCTION_EXPECTED_PATH,
        "confidence_exact_0_9025": bool(conf_ok),
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": brain.cycle - cycle_before,
        "legs": legs,
        "evidence": {
            "derivation": deriv and {
                "subject": deriv["subject"], "relation": deriv["relation"],
                "value": deriv["value"], "confidence": deriv["confidence"],
                "proof": (deriv["proof"].explain()
                          if hasattr(deriv["proof"], "explain")
                          else deriv["proof"]),
                "proof_steps": steps, "proof_path": list(path) if path else None,
                "status": deriv["status"]},
            "expected_confidence": DEDUCTION_EXPECTED_CONFIDENCE,
            "facts_taught": [f"{s} {r} {v}" for s, r, v in DEDUCTION_FACTS],
            "teach_results": [r[0].value for r in taught],
            "last_inferences_count": len(getattr(brain, "last_inferences", []) or []),
        },
        "detail": (
            f"derived {DEDUCTION_SUBJECT} {DEDUCTION_RELATION} "
            f"{DEDUCTION_VALUE} with {steps}-step proof at confidence "
            f"{conf}" if deriv else
            "no derivation of velkra located_in kelmor in-cycle"),
    }


# ------------------------------------------------------------------ (b) contradiction
CONTRADICTION_SUBJECT = "lyssan"
CONTRADICTION_RELATION = "boils_at"
CONTRADICTION_VALUES = ("100c", "90c")


def run_contradiction_task(brain_factory) -> dict:
    """Task (b): teach two conflicting verified facts on the functional
    relation ``boils_at`` via the LTM conflict path, put the subject
    in WM, run one cycle, then ask the brain to decide."""
    brain = brain_factory([], {"compute"})
    first = brain.teach_fact(CONTRADICTION_SUBJECT, CONTRADICTION_RELATION,
                             CONTRADICTION_VALUES[0], "teacher")
    second = brain.teach_fact(CONTRADICTION_SUBJECT, CONTRADICTION_RELATION,
                              CONTRADICTION_VALUES[1], "teacher")
    cycle_before = brain.cycle
    brain.perceive_text(f"{CONTRADICTION_SUBJECT} {CONTRADICTION_RELATION}")
    brain.step()
    answer = brain.ask(CONTRADICTION_SUBJECT, CONTRADICTION_RELATION)
    facts = brain.ltm.query(CONTRADICTION_SUBJECT, CONTRADICTION_RELATION,
                            statuses=("verified", "unverified", "disputed",
                                      "retracted"))
    disputed = sorted(f.value for f in facts if f.status == "disputed")
    event = _has_event(brain, "contradiction")
    goal = _research_goal_for(brain, CONTRADICTION_SUBJECT,
                              CONTRADICTION_RELATION)
    asked_values = sorted(v for v, _, _, _ in answer.answers)
    neither_picked = (asked_values == sorted(CONTRADICTION_VALUES)
                      and all(f.status != "verified" for f in facts))
    legs = {
        "both_claims_disputed_in_ltm": disputed == sorted(CONTRADICTION_VALUES),
        "contradiction_event_in_cycle": event is not None,
        "research_goal_created": goal is not None,
        "neither_side_silently_picked": bool(neither_picked),
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": brain.cycle - cycle_before,
        "legs": legs,
        "evidence": {
            "teach_results": [first[0].value, second[0].value],
            "disputed_values": disputed,
            "fact_statuses": sorted(
                (f.value, f.status) for f in facts),
            "contradiction_event": event,
            "research_goal": goal,
            "ask_answers": [(v, c, s) for v, c, s, _ in answer.answers],
            "ask_message": answer.message,
        },
        "detail": (
            f"both values disputed, contradiction event "
            f"{'seen' if event else 'MISSING'}, research goal "
            f"{'created' if goal else 'MISSING'}, ask returns "
            f"{asked_values}" if disputed == sorted(CONTRADICTION_VALUES)
            else f"LTM did not mark both values disputed: {disputed}"),
    }


# ------------------------------------------------------------------ (c) hypothesis / abduction
class SurpriseWorld(Environment):
    """An unfamiliar one-room world with a hidden rule the brain was
    never taught: ``heat_vapour`` is believed to make the vapour hot,
    but with a cold coil the vapour stays cold -- the predicted effect
    fails (a surprise). Warming the coil first makes the same action
    succeed, so the surprise is diagnosable, not random."""

    def __init__(self):
        self.s = {"at": "lab", "coil": "cold", "vapour_hot": False}
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        if name == "heat_vapour" and self.s["at"] == "lab":
            if self.s["coil"] == "warm":
                self.s["vapour_hot"] = True
                info = "the vapour boils"
            else:
                info = "the coil hums but the vapour stays cold"
            self.log.append(f"{name}: {info}")
            return True, info
        if name == "warm_coil" and self.s["at"] == "lab":
            self.s["coil"] = "warm"
            info = "the coil is warm now"
            self.log.append(f"{name}: {info}")
            return True, info
        self.log.append(f"{name}: nothing happened")
        return False, "nothing happened"


def surprise_world_actions() -> list[Action]:
    """The brain's (initially naive) believed model: heating the vapour
    makes it hot. It does NOT know the cold coil blocks the effect."""
    return [
        Action("heat_vapour", {"at": "lab"}, {"vapour_hot": True},
               cost=1.0, args={"_do": "heat_vapour"}),
        Action("warm_coil", {"at": "lab", "coil": "cold"}, {"coil": "warm"},
               cost=1.0, args={"_do": "warm_coil"}),
    ]


def _norm_hypothesis(h) -> dict | None:
    """Normalise a hypothesis record (Hypothesis object, dict, or state
    entry value) to a plain dict, or None."""
    if h is None:
        return None
    if isinstance(h, dict):
        return {"id": h.get("id"), "subject": str(h.get("subject", "")),
                "relation": str(h.get("relation", "")),
                "value": str(h.get("value", "")),
                "confidence": h.get("confidence"),
                "source": str(h.get("source", "")),
                "status": str(h.get("status", "open")),
                "test": h.get("test"),
                "evidence_for": list(h.get("evidence_for", []) or []),
                "evidence_against": list(h.get("evidence_against", []) or [])}
    subject = getattr(h, "subject", None)
    if subject is None and not isinstance(h, str):
        return None
    if isinstance(h, str):
        return {"id": None, "subject": "", "relation": "", "value": h,
                "confidence": None, "source": "", "status": "open",
                "test": None, "evidence_for": [], "evidence_against": []}
    return {"id": getattr(h, "id", None), "subject": str(subject),
            "relation": str(getattr(h, "relation", "")),
            "value": str(getattr(h, "value", "")),
            "confidence": getattr(h, "confidence", None),
            "source": str(getattr(h, "source", "")),
            "status": str(getattr(h, "status", "open")),
            "test": getattr(h, "test", None),
            "evidence_for": list(getattr(h, "evidence_for", []) or []),
            "evidence_against": list(getattr(h, "evidence_against", []) or [])}


def _find_hypothesis(brain) -> dict | None:
    """Find a hypothesis the brain generated, in the observable places
    the Phase 4 design names: ``hypothesis:*`` state entries, a
    hypothesis event, or a stored hypothesis / reasoning result on the
    brain. Returns a normalised hypothesis dict, or None. Reads
    brain-observable state only; it never abduces anything itself."""
    for key in brain.state.keys():
        if not str(key).startswith("hypothesis:"):
            continue
        entry = brain.state.get(key)
        if entry is None:
            continue
        h = _norm_hypothesis(entry.value)
        if h is not None:
            h["state_key"] = key
            return h
    for e in brain.events:
        if "hypothesis" not in e.get("kind", ""):
            continue
        detail = e.get("detail", "")
        return {"id": None, "subject": "heat_vapour", "relation": "caused_by",
                "value": detail, "confidence": None, "source": "event",
                "status": ("corroborated" if "corroborated" in e["kind"]
                           else "falsified" if "falsified" in e["kind"]
                           else "open"),
                "test": detail,
                "evidence_for": [detail], "evidence_against": [],
                "event_kind": e["kind"]}
    for attr in ("hypotheses", "last_hypotheses"):
        hyps = getattr(brain, attr, None)
        if callable(hyps):                  # Brain.hypotheses() is a method
            hyps = hyps()
        if hyps:
            h = _norm_hypothesis(next(iter(hyps)))
            if h is not None:
                return h
    for attr in ("last_reasoning", "last_reasoning_result"):
        result = getattr(brain, attr, None)
        if result is None:
            continue
        for raw in (getattr(result, "hypotheses", None) or []):
            h = _norm_hypothesis(raw)
            if h is not None:
                return h
    return None


def run_hypothesis_task(brain_factory, max_cycles: int = 30) -> dict:
    """Task (c): run the brain in SurpriseWorld with the goal of
    heating the vapour. The first ``heat_vapour`` succeeds as an
    action but its predicted effect fails -- a surprise. Pass =
    a hypothesis is generated with a test, the test runs, and the
    outcome is recorded (corroborated or falsified), AND the goal
    completes (the vapour ends up hot). The goal leg is the Phase 5
    criterion change (design SS4.2, following the Phase 2 SS9 /
    Phase 3 SS8 / Phase 4 SS14.1 precedent): completion was
    previously blocked by the bug the phase fixes, so requiring it
    is the mandated fix, not a weakening."""
    env = SurpriseWorld()
    brain = brain_factory(surprise_world_actions(), {"compute", "environment"})
    brain.add_goal("heat the vapour", {"vapour_hot": True})
    cycle_before = brain.cycle
    stop = brain.run(env, max_cycles)
    goal = brain.goals.all()[-1]
    surprise_events = [e for e in brain.events
                       if e["kind"] in ("surprise", "prediction_checked")
                       and ("violated" in e["detail"]
                            or "surprise" in e["detail"].lower())]
    violated = [e for e in brain.events
                if e["kind"] == "prediction_checked" and "violated" in e["detail"]]
    hyp = _find_hypothesis(brain)
    outcome_event = _has_event(brain, "hypothesis_corroborated",
                               "hypothesis_falsified")
    outcome_recorded = bool(
        outcome_event is not None
        or (hyp is not None and hyp.get("status") in ("corroborated", "falsified")))
    legs = {
        "surprise_occurred": len(surprise_events) > 0,
        "hypothesis_generated": hyp is not None,
        "hypothesis_has_test": bool(hyp and hyp.get("test")),
        "test_outcome_recorded": outcome_recorded,
        # Phase 5 (design SS4.2): the task's goal is completion, and
        # the Phase 4 stall blocked it. The principled recovery path
        # (a model revision clears pair-bans keyed to the old model
        # and resets the stall window) must let the brain warm the
        # coil and finish -- so completion is now a pass leg.
        "goal_completed": goal.status == "done",
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": brain.cycle - cycle_before,
        "legs": legs,
        "evidence": {
            "stop_reason": stop,
            "goal_status": goal.status,
            "goal_completed": goal.status == "done",
            "surprise_events": len(surprise_events),
            "prediction_violated_events": len(violated),
            "learner_episodes": len(getattr(brain.learner, "episodes", []) or []),
            "hypothesis": hyp,
            "hypothesis_outcome_event": outcome_event,
            "env_log": list(env.log),
        },
        "detail": (
            f"surprise seen ({len(surprise_events)} event(s)); hypothesis "
            f"{'generated' if hyp else 'NOT generated'}"
            f"{' with test' if hyp and hyp.get('test') else ''}; outcome "
            f"{'recorded' if outcome_recorded else 'NOT recorded'}; goal "
            f"{goal.status}"),
    }


# ------------------------------------------------------------------ all three, all variants
TASKS = (
    ("deduction_3hop", run_deduction_task,
     "3-hop mixed-relation deduction (velkra is_a torvin is_a sarnak "
     "located_in kelmor): pass = derivation with 3-step proof and "
     "exact confidence 0.9025."),
    ("contradiction", run_contradiction_task,
     "Two conflicting verified facts on functional relation boils_at "
     "(lyssan 100c vs 90c): pass = contradiction event in-cycle, both "
     "claims disputed, research goal created, neither side picked."),
    ("hypothesis_abduction", run_hypothesis_task,
     "Surprise in SurpriseWorld (heat_vapour predicted hot, stays "
     "cold): pass = hypothesis generated with a test, test run, "
     "outcome recorded (corroborated or falsified), AND the goal "
     "completes. Phase 5 criterion change (design SS4.2): completion "
     "was previously blocked by the stall bug the phase fixes, so "
     "requiring it is the mandated fix, not a weakening."),
)


def run_discriminating_tasks(brain_factories=None) -> dict:
    """Run the three discriminating tasks for each variant and return
    the record section (per-variant numbers, no winner declared).
    A brain failure in one task/variant is recorded, never raised."""
    factories = brain_factories or make_variant_factories()
    tasks: dict = {}
    for task_id, runner, criterion in TASKS:
        variants: dict = {}
        for name, factory in factories.items():
            try:
                variants[name] = runner(factory)
            except Exception as exc:      # a brain failure is data, not a crash
                variants[name] = {
                    "passed": False, "cycles": None, "legs": {},
                    "evidence": {"error": f"{type(exc).__name__}: {exc}"},
                    "detail": f"task errored: {type(exc).__name__}: {exc}",
                }
        tasks[task_id] = {"criterion": criterion, "variants": variants}
    return {
        "mechanism": (
            "Held-out tasks defined in ab/eval/reasoning_tasks.py only; "
            "the brain was never taught these names or this world. "
            "Variants are built by monkeypatching/subclassing Brain "
            "(brain.py NOT edited), as in the harness baselines: "
            "no_learning replaces brain.learner with a no-op; "
            "non_integrated overrides _reason_from_wm with a no-op. "
            "No winner is declared -- the numbers below are the comparison."),
        "tasks": tasks,
        "note": ("A task that the full brain fails is a finding about "
                 "the reasoning wiring, not a harness crash; the "
                 "harness still exits 0 when this record is written."),
    }
