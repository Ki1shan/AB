"""Phase 9 held-out reasoning tasks, second set (design docs/phase9-design.md SS3.2).

Four tasks, one per reasoning type with no held-out task yet. Same
conventions as ``reasoning_tasks.py``: unfamiliar invented names,
defined ONLY in eval code, run for all variants (full / no_learning /
non_integrated, via ``make_variant_factories`` from
``reasoning_tasks.py``), per-leg pass criteria, no winner declared.
A failed task is a recorded result, never a crash: every runner
returns a result dict with ``passed`` plus per-leg evidence, and
:func:`run_reasoning_tasks2` never raises for a brain failure.

(a) ``causal_reasoning`` — an action (``forge``) that reliably
    produces its effect in one variant context (with the catalyst
    seated) and fails in another (without it). The brain runs the
    forge task three times on one brain (fresh world each time), so
    the reliable variant can reach the causal-claim bar (3 met, 0
    counterexamples). Pass = the brain records a causal claim only
    for the reliable variant, and does not claim causation for the
    failing one (nor for the pooled name, whose counterexample
    blocks it).
(b) ``analogy`` — a relation pattern learned in domain A (``quivern
    opens garnik``) must be proposed as a hypothesis (capped
    confidence <= 0.4, never verified alone, never driving an action)
    for the structurally similar domain B (``norbex``, which shares
    ``quivern``'s kind and place). Pass = a hypothesis exists with
    source ``analogy``, and the brain does NOT act on it unconfirmed
    (no LTM fact, no inferred entry, ``ask`` still answers "I don't
    know"). The mechanism is ``ReasoningEngine.analogize`` — see the
    evidence note: in-cycle reasoning (``_reason_from_wm``) does not
    call it, so the proposal is engine-level, recorded here, not
    claimed as a brain-cycle behaviour.
(c) ``abstraction`` — subjects sharing >= 3 properties (``trellin``
    and ``vossar``) must yield a proposed pending category via the
    normal approval path, never auto-approved. Pass = a pending
    category is proposed with the shared properties listed.
(d) ``counterfactual_reasoning`` — given a failed plan (the direct
    route to the treasure, which cannot pass the closed door), the
    brain must state what would have had to differ for success (a
    counterfactual over the world model, assumption named). Pass =
    the counterfactual result names the changed fact and the outcome
    difference, and the real world model is untouched.

The task worlds/facts live in this file only; nothing is hardcoded
into the brain. The names (``vortan``, ``drelm``, ``forge``,
``catalyst``, ``quivern``, ``norbex``, ``mellor``, ``forvik``,
``garnik``, ``trellin``, ``vossar``, ``krelm``) appear nowhere else
in the repo outside ``ab/eval/``: they are held-out by construction.
No LLM, stdlib only.
"""
from __future__ import annotations

from ..body.environment import Environment
from ..brain import Brain
from ..planning.planner import Action, canonical_variant
from .reasoning_tasks import make_variant_factories

# ------------------------------------------------------------------ (a) causal reasoning
FORGE_START = {"at": "vortan", "has_catalyst": False, "forged": False}
FORGE_GOOD_VARIANT = canonical_variant({"at": "vortan", "has_catalyst": True})
FORGE_BAD_VARIANT = canonical_variant({"at": "vortan"})
FORGE_REPETITIONS = 3


class ForgeWorld(Environment):
    """An unfamiliar one-room forge: the ``forge`` action is believed
    to stamp the part, but the truth (known only here) is that it
    works only with the catalyst seated. The cheap variant (no
    catalyst in its preconditions) is tried first by the naive
    planner and fails; the expensive variant (catalyst seated)
    succeeds. The catalyst lives in the next room."""

    def __init__(self):
        self.s = dict(FORGE_START)
        self.log: list[str] = []

    def observe(self) -> dict:
        return dict(self.s)

    def act(self, name: str, args: dict) -> tuple[bool, str]:
        s = self.s
        ok, info = False, "nothing happened"
        if name == "go_drelm" and s["at"] == "vortan":
            s["at"], ok, info = "drelm", True, "moved to drelm"
        elif name == "go_vortan" and s["at"] == "drelm":
            s["at"], ok, info = "vortan", True, "moved to vortan"
        elif name == "take_catalyst" and s["at"] == "drelm" \
                and not s["has_catalyst"]:
            s["has_catalyst"], ok, info = True, True, "took the catalyst"
        elif name == "forge":
            if s["has_catalyst"]:
                s["forged"], ok, info = True, True, "the forge stamps the part"
            else:
                info = "the forge has no catalyst seated"
        self.log.append(f"{name}: {info}")
        return ok, info


def forge_actions() -> list[Action]:
    """The brain's (initially naive) believed model: ``forge`` stamps
    the part. It does NOT know the catalyst is required. One action
    name, two variants: the cheap one with no catalyst key, the
    expensive one with it."""
    return [
        Action("go_drelm", {"at": "vortan"}, {"at": "drelm"},
               args={"_do": "go_drelm"}),
        Action("go_vortan", {"at": "drelm"}, {"at": "vortan"},
               args={"_do": "go_vortan"}),
        Action("take_catalyst", {"at": "drelm", "has_catalyst": False},
               {"has_catalyst": True}, args={"_do": "take_catalyst"}),
        Action("forge", {"at": "vortan"}, {"forged": True}, cost=0.3,
               args={"_do": "forge"}),
        Action("forge", {"at": "vortan", "has_catalyst": True},
               {"forged": True}, cost=2.0, args={"_do": "forge"}),
    ]


def run_causal_task(brain_factory, max_cycles: int = 60) -> dict:
    """Task (a): run the forge task ``FORGE_REPETITIONS`` times on one
    brain (fresh world each time). The reliable variant must
    accumulate 3 met / 0 counterexamples and be claimed; the failing
    variant must not be claimed; the pooled name must not be claimed
    either (its counterexample blocks it — without variant-aware
    records there would be no claim at all)."""
    brain = brain_factory(forge_actions(), {"compute", "environment"})
    runs = []
    for _ in range(FORGE_REPETITIONS):
        env = ForgeWorld()
        brain.add_goal("forge the part", {"forged": True})
        cycle_before = brain.cycle
        stop = brain.run(env, brain.cycle + max_cycles)
        goal = brain.goals.all()[-1]
        runs.append({
            "stop_reason": stop, "goal_status": goal.status,
            "cycles": brain.cycle - cycle_before,
            "forge_attempts": sum(1 for e in env.log if e.startswith("forge")),
            "forge_failures": sum(1 for e in env.log
                                  if e.startswith("forge")
                                  and "no catalyst" in e),
            "env_log": list(env.log),
        })
    good_counts = brain.causal.counts("forge", "forged",
                                      variant=FORGE_GOOD_VARIANT)
    bad_counts = brain.causal.counts("forge", "forged",
                                     variant=FORGE_BAD_VARIANT)
    good_claim = brain.causal.causes("forge", "forged",
                                     variant=FORGE_GOOD_VARIANT)
    bad_claim = brain.causal.causes("forge", "forged",
                                    variant=FORGE_BAD_VARIANT)
    pooled_claim = brain.causal.causes("forge", "forged")
    legs = {
        "all_runs_completed": all(r["goal_status"] == "done" for r in runs),
        "failing_variant_actually_tried": any(
            r["forge_failures"] >= 1 for r in runs),
        "reliable_variant_claimed": bool(
            good_claim is not None and good_claim["observations"] == 3
            and good_claim["counterexamples"] == 0
            and good_claim["confidence"] == 1.0),
        "failing_variant_not_claimed": bad_claim is None,
        "pooled_name_not_claimed": pooled_claim is None,
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": sum(r["cycles"] for r in runs),
        "legs": legs,
        "evidence": {
            "runs": runs,
            "reliable_variant": FORGE_GOOD_VARIANT,
            "failing_variant": FORGE_BAD_VARIANT,
            "reliable_variant_counts": good_counts,
            "failing_variant_counts": bad_counts,
            "pooled_counts": brain.causal.counts("forge", "forged"),
            "reliable_variant_claim": good_claim,
            "failing_variant_claim": bad_claim,
            "pooled_claim": pooled_claim,
            "variant_claims_for_forge": [
                c for c in brain.causal.claims() if c["action"] == "forge"],
        },
        "detail": (
            f"forge with catalyst claimed ({good_counts}) after "
            f"{len(runs)} runs; forge without catalyst "
            f"{bad_counts} not claimed; pooled claim "
            f"{'absent' if pooled_claim is None else 'PRESENT (bad)'}" if passed
            else f"legs failed: {[k for k, v in legs.items() if not v]}"),
    }


# ------------------------------------------------------------------ (b) analogy
ANALOGY_FACTS = (("quivern", "is_a", "mellor"),
                 ("quivern", "located_in", "vortan"),
                 ("quivern", "opens", "garnik"),
                 ("norbex", "is_a", "mellor"),
                 ("norbex", "located_in", "vortan"))
ANALOGY_SUBJECT = "norbex"
ANALOGY_RELATION = "opens"
ANALOGY_VALUE = "garnik"
ANALOGY_SIBLING = "quivern"


def run_analogy_task(brain_factory) -> dict:
    """Task (b): teach the domain-A pattern and domain-B's shared
    features, attend to the target, then propose by analogy. Pass =
    a hypothesis exists with source ``analogy`` and confidence
    <= 0.4, and the brain does NOT act on it unconfirmed: no LTM
    fact, no inferred state entry, ``ask`` still answers "I don't
    know", no plan or goal touches it."""
    brain = brain_factory([], {"compute"})
    taught = [brain.teach_fact(s, r, v, "teacher")
              for s, r, v in ANALOGY_FACTS]
    cycle_before = brain.cycle
    brain.perceive_text(ANALOGY_SUBJECT)
    brain.step()
    hyp = brain.reasoning.analogize(ANALOGY_SUBJECT, ANALOGY_RELATION,
                                    brain._facts_fn)
    answer = brain.ask(ANALOGY_SUBJECT, ANALOGY_RELATION)
    ltm_facts = brain.ltm.query(ANALOGY_SUBJECT, ANALOGY_RELATION)
    inferred = brain.state.get(f"inferred:{ANALOGY_SUBJECT}|{ANALOGY_RELATION}")
    hyp_d = None
    if hyp is not None:
        hyp_d = {"id": hyp.id, "subject": hyp.subject,
                 "relation": hyp.relation, "value": hyp.value,
                 "confidence": hyp.confidence, "source": hyp.source,
                 "status": hyp.status, "test": hyp.test,
                 "evidence_for": list(hyp.evidence_for)}
    legs = {
        "hypothesis_proposed_with_source_analogy": bool(
            hyp is not None and hyp.source == "analogy"),
        "hypothesis_value_is_garnik": bool(
            hyp is not None and hyp.value == ANALOGY_VALUE),
        "confidence_capped_at_0_4": bool(
            hyp is not None and hyp.confidence <= 0.4),
        "never_verified_alone": bool(
            hyp is not None and hyp.status != "verified"
            and not answer.known and not ltm_facts),
        "not_acted_on": bool(
            inferred is None and not brain.plan
            and not any(ANALOGY_SUBJECT in str(g.target)
                        for g in brain.goals.all())),
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": brain.cycle - cycle_before,
        "legs": legs,
        "evidence": {
            "hypothesis": hyp_d,
            "ask_known": answer.known,
            "ask_message": answer.message,
            "ltm_facts_for_target": [
                (f.subject, f.relation, f.value, f.status)
                for f in ltm_facts],
            "inferred_entry": inferred.value if inferred is not None else None,
            "teach_results": [r[0].value for r in taught],
            "mechanism_note": (
                "Analogy is proposed by ReasoningEngine.analogize; the "
                "in-cycle _reason_from_wm path does not call it, so the "
                "proposal is engine-level here, not a brain-cycle "
                "behaviour. The 'not acted on' legs are the guard that "
                "the proposal never becomes a fact or a plan on its own."),
        },
        "detail": (
            f"analogy hypothesis {ANALOGY_SUBJECT} {ANALOGY_RELATION} "
            f"{ANALOGY_VALUE} at confidence "
            f"{hyp.confidence if hyp else None} (source "
            f"{hyp.source if hyp else None}); ask still unknown: "
            f"{not answer.known}" if hyp is not None else
            "no analogy hypothesis proposed"),
    }


# ------------------------------------------------------------------ (c) abstraction
ABSTRACTION_SUBJECTS = ("trellin", "vossar")
ABSTRACTION_SHARED = (("is_a", "krelm"),
                      ("located_in", "forvik"),
                      ("made_of", "mellor"))


def run_abstraction_task(brain_factory) -> dict:
    """Task (c): teach two subjects three shared verified properties,
    attend to both, run one cycle. Pass = a pending category is
    proposed with the shared properties listed, via the normal
    approval path — never auto-approved (no approved category, no
    LTM fact for it)."""
    brain = brain_factory([], {"compute"})
    taught = []
    for subj in ABSTRACTION_SUBJECTS:
        for r, v in ABSTRACTION_SHARED:
            taught.append(brain.teach_fact(subj, r, v, "teacher"))
    cycle_before = brain.cycle
    for subj in ABSTRACTION_SUBJECTS:
        brain.perceive_text(subj)
    brain.step()
    proposed = [c for c in brain.proposed_categories
                if set(c.get("members", [])) == set(ABSTRACTION_SUBJECTS)]
    approved = [c for c in brain.proposed_categories
                if c.get("status") == "approved"]
    shared_ok = bool(
        proposed and len(proposed[0].get("shared", [])) >= 3
        and sorted(tuple(p) for p in proposed[0]["shared"])
        == sorted(ABSTRACTION_SHARED))
    legs = {
        "pending_category_proposed": len(proposed) >= 1,
        "proposed_for_both_subjects": bool(proposed),
        "shared_properties_listed_ge_3": shared_ok,
        "status_pending_not_approved": bool(
            proposed and all(c.get("status") == "pending" for c in proposed)
            and not approved),
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": brain.cycle - cycle_before,
        "legs": legs,
        "evidence": {
            "proposed_categories": brain.proposed_categories,
            "matching_proposal": proposed[0] if proposed else None,
            "approved_categories": approved,
            "teach_results": [r[0].value for r in taught],
            "shared_expected": [list(p) for p in ABSTRACTION_SHARED],
        },
        "detail": (
            f"pending category proposed for {ABSTRACTION_SUBJECTS} "
            f"with {len(proposed[0]['shared']) if proposed else 0} "
            f"shared properties" if proposed else
            "no pending category proposed"),
    }


# ------------------------------------------------------------------ (d) counterfactual reasoning
CF_START = {"at": "vortan", "has_krelm": False, "door_open": False,
            "has_garnik": False}
CF_TARGET = {"has_garnik": True}
CF_DIRECT_ROUTE = ["go_drelm_from_vortan", "go_forvik", "take_garnik"]
CF_CHANGED_FACT = "door_open"


def cf_actions() -> list[Action]:
    """A small invented treasure shape: the key (``krelm``) opens the
    door to the ``forvik`` room, where the ``garnik`` waits. The
    direct route skips the key and the unlock, so it fails at the
    closed door."""
    return [
        Action("go_drelm_from_vortan", {"at": "vortan"}, {"at": "drelm"}),
        Action("take_krelm", {"at": "drelm"}, {"has_krelm": True}),
        Action("unlock", {"at": "drelm", "has_krelm": True},
               {"door_open": True}),
        Action("go_forvik", {"at": "drelm", "door_open": True},
               {"at": "forvik"}),
        Action("take_garnik", {"at": "forvik"}, {"has_garnik": True}),
    ]


def run_counterfactual_task(brain_factory) -> dict:
    """Task (d): give the brain the failed direct plan, then ask what
    would have had to differ. Pass = the counterfactual names the
    changed fact (``door_open``) and the outcome difference, is
    labelled simulated, and the real world model (and world state)
    is untouched."""
    brain = brain_factory(cf_actions(), {"compute", "environment"})
    brain.world = dict(CF_START)
    brain.add_goal("get the garnik", dict(CF_TARGET))
    wm_before = brain.world_model.to_json()
    world_before = dict(brain.world)
    result = brain.counterfactual(edits={CF_CHANGED_FACT: True},
                                  goal=dict(CF_TARGET),
                                  steps=list(CF_DIRECT_ROUTE))
    wm_after = brain.world_model.to_json()
    difference = result.get("difference") or {}
    assumptions = result.get("assumptions") or []
    legs = {
        "baseline_plan_failed": bool(
            (result.get("baseline") or {}).get("met_goal") is False),
        "counterfactual_plan_succeeds": bool(
            (result.get("counterfactual") or {}).get("met_goal") is True),
        "changed_fact_named": any(CF_CHANGED_FACT in a for a in assumptions),
        "outcome_difference_recorded": bool(
            difference and "has_garnik" in difference
            and difference["has_garnik"] == (False, True)),
        "labelled_simulated": result.get("simulated") is True,
        "real_world_model_untouched": wm_before == wm_after,
        "real_world_state_untouched": dict(brain.world) == world_before,
    }
    passed = all(legs.values())
    return {
        "passed": passed,
        "cycles": 0,
        "legs": legs,
        "evidence": {
            "changed_fact": CF_CHANGED_FACT,
            "direct_route": list(CF_DIRECT_ROUTE),
            "baseline_met_goal": (result.get("baseline") or {}).get("met_goal"),
            "counterfactual_met_goal": (result.get("counterfactual") or {}).get("met_goal"),
            "difference": difference,
            "outcome_changed": result.get("outcome_changed"),
            "assumptions": assumptions,
            "answer": result.get("answer"),
            "real_world_model_untouched": wm_before == wm_after,
        },
        "detail": (
            f"counterfactual: if {CF_CHANGED_FACT}=True, the direct "
            f"route succeeds (was failing); difference {difference}"
            if passed else
            f"legs failed: {[k for k, v in legs.items() if not v]}"),
    }


# ------------------------------------------------------------------ all four, all variants
TASKS = (
    ("causal_reasoning", run_causal_task,
     "Forge task x3 on one brain (forge fails without the catalyst, "
     "succeeds with it): pass = causal claim only for the reliable "
     "(with-catalyst) variant, no claim for the failing variant, no "
     "claim for the pooled name."),
    ("analogy", run_analogy_task,
     "Relation pattern in domain A (quivern opens garnik) proposed "
     "for structurally similar domain B (norbex): pass = hypothesis "
     "exists with source analogy and confidence <= 0.4, never "
     "verified alone, never driving an action."),
    ("abstraction", run_abstraction_task,
     "Two subjects sharing >= 3 properties (trellin, vossar): pass = "
     "a pending category is proposed with the shared properties "
     "listed, via the normal approval path, never auto-approved."),
    ("counterfactual_reasoning", run_counterfactual_task,
     "Failed direct plan (closed door) counterfactual: pass = the "
     "result names the changed fact (door_open) and the outcome "
     "difference, is labelled simulated, and the real world model "
     "is untouched."),
)


def run_reasoning_tasks2(brain_factories=None) -> dict:
    """Run the four Phase 9 reasoning tasks for each variant and
    return the record section (per-variant numbers, no winner
    declared). A brain failure in one task/variant is recorded,
    never raised."""
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
            "Held-out tasks defined in ab/eval/reasoning_tasks2.py only; "
            "the brain was never taught these names or these worlds. "
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
