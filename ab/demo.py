"""Run: python -m ab.demo   -- shows each capability working end to end."""
import os, tempfile

from .brain import Brain
from .body.environment import RoomWorld, room_world_actions
from .body.ocr import TesseractOcr
from .body.research import Researcher


def header(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def _wm_list(obj, name):
    """World-model collections are methods (events(), revisions(),
    patterns()); call them, tolerating a plain list/attribute too."""
    v = getattr(obj, name, None)
    if callable(v):
        v = v()
    return list(v or [])


def main():
    header("1. Goal planning + environment + evaluation + learning from surprise")
    # Phase 2: the default grant set is compute-only, so the demo passes
    # the environment grant explicitly (it acts on the world).
    env, brain = RoomWorld(), Brain(room_world_actions(), granted={"compute", "environment"})
    brain.add_goal("get the treasure", {"has_treasure": True})
    brain.run(env, 60)
    print("\n".join(l for l in brain.explain() if " rule " not in l))
    print("outcome:", brain.goals.all()[0].status, "| cycles:", brain.cycle,
          "| pick_lock failure rate now:", brain.planner.failure_rate("pick_lock"))

    header("2. Teaching, inference, and refusing to guess")
    b = Brain()
    b.teach_fact("cat", "is_a", "mammal"); b.teach_fact("mammal", "is_a", "animal")
    print("ask cat is_a      :", b.ask("cat", "is_a").best)
    print("verify cat animal :", b.verify("cat", "is_a", "animal"))
    print("verify cat robot  :", b.verify("cat", "is_a", "robot"))
    print("ask zorblax       :", b.ask("zorblax", "capital_of").message)

    header("3. Research with provenance (simulated web, two independent domains)")
    from .body.providers import MemoryKnowledgeProvider
    pages = {"https://alpha.org/f": "<p>Paris is the capital of France.</p>",
             # Phase 11 §4.1 (W1): the second source must word the fact
             # differently — the identical sentence from both domains
             # is a mirror under the mirror rule and no longer
             # corroborates. Same claim, different sentence.
             "https://beta.org/f": "<html><script>x()</script><p>Paris is the capital city of France.</p></html>"}
    # Phase 11 §2.2 (W5): the research tool runs isolated in a forked
    # child, which materializes its provider from the spec — so the
    # provider must survive a spec round-trip. A WebFetcher with a
    # fake transport does not (the transport is a closure, never in
    # the spec); the memory ("book") provider carries the same pages
    # and the same two web domains, so the outcome is unchanged.
    rb = Brain(researcher=Researcher(MemoryKnowledgeProvider(pages), None),
               granted={"compute", "network"})
    rb.add_research_goal("paris", "capital_of", list(pages))
    rb.run(None, 30)
    print("\n".join(l for l in rb.explain() if " rule " not in l))
    print("answer:", rb.ask("paris", "capital_of").best)

    header("4. OCR sense (real Tesseract) -> unverified knowledge")
    ocr = TesseractOcr()
    if not ocr.available():
        print("tesseract not installed: skipped")
    else:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (900, 120), "white")
        font = None
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                font = ImageFont.truetype(p, 44)
        ImageDraw.Draw(img).text((20, 30), "Mars is a planet.", fill="black", font=font or ImageFont.load_default(size=44))
        path = os.path.join(tempfile.mkdtemp(), "sign.png"); img.save(path)
        ob = Brain(ocr=ocr)
        out = ob.perceive_image(path)
        print("read:", repr(out["text"]), "| confidence:", out["confidence"])
        print("stored claims:", [(c.subject, c.relation, c.value, st) for c, _, st in out["stored"]])

    header("5. World model (Phase 5) — structured view over the same observations")
    wm = getattr(brain, "world_model", None)
    if wm is None:
        print("world model not present in this build (Phase 5 not integrated yet)")
    else:
        ents = list(wm.entities()) if callable(getattr(wm, "entities", None)) else []
        print("entities:", len(ents), "|",
              [(getattr(e, "id", "?"), getattr(e, "kind", "?")) for e in ents])
        rels = getattr(wm, "relations", None) or []
        if callable(rels):
            rels = rels()
        print("located_in relations:",
              [(getattr(r, "subject", "?"), getattr(r, "object", "?"))
               for r in rels if getattr(r, "relation", None) == "located_in"])
        print("world events:", len(_wm_list(wm, "events")),
              "| revisions:", len(_wm_list(wm, "revisions")))
        lib = getattr(brain, "patterns", None)
        pats = _wm_list(lib, "patterns") if lib is not None else []
        print("patterns learned:", [getattr(p, "name", getattr(p, "id", "?")) for p in pats])
        from collections import Counter
        statuses = Counter()
        for e in ents:
            for prop in (getattr(e, "properties", None) or {}).values():
                statuses[str(getattr(prop, "status", "unknown"))] += 1
        print("property epistemic statuses:", dict(statuses))

    header("6. Learning and adaptation (Phase 6) — what the treasure run taught")
    # Phase 6 APIs (ab/learning/: Consolidator, ConceptStore,
    # CalibrationTracker, repeated-mistake events) may not have landed
    # yet — every access below is guarded, so this section reports
    # "not present" / "not measured" instead of crashing.
    lib = getattr(brain, "patterns", None)
    pats = _wm_list(lib, "patterns") if lib is not None else []
    from collections import Counter as _Counter
    kinds = _Counter(getattr(p, "kind", "?") for p in pats)
    print("patterns learned:", len(pats), "| kinds:", dict(kinds), "|",
          [getattr(p, "name", getattr(p, "id", "?")) for p in pats])
    # Concepts: ConceptStore on the brain (Phase 6 C-1/C-2), if present.
    concepts = None
    for _attr in ("concept_store", "concepts"):
        concepts = getattr(brain, _attr, None)
        if concepts is not None:
            break
    if concepts is None:
        print("concepts learned: not present in this build (Phase 6 not integrated yet)")
    else:
        _all = getattr(concepts, "all", None)
        _all = _all() if callable(_all) else (_all if _all is not None else [])
        print("concepts learned:", [(getattr(c, "name", "?"),
              round(float(getattr(c, "confidence", 0.0)), 2)) for c in list(_all or [])])
    # Consolidation: review queue on the brain (Phase 6), if present.
    consolidator = getattr(brain, "consolidator", None)
    if consolidator is None:
        print("consolidation: not present in this build (Phase 6 not integrated yet)")
    else:
        def _count(obj, name):
            v = getattr(obj, name, None)
            v = v() if callable(v) else v
            return len(list(v or [])) if v is not None else 0
        print("consolidation candidates:", _count(consolidator, "candidates"),
              "| promoted:", _count(consolidator, "promoted"),
              "| rejected:", _count(consolidator, "rejected"))
    # Calibration gap: mean |Laplace-predicted - observed| per
    # (action, effect) with a causal record. The Phase 6
    # CalibrationTracker (ECE/Brier) is the authoritative measure;
    # until it lands this is computed from the public causal
    # counts() API and labelled as such. No record -> not measured.
    causal = getattr(brain, "causal", None)
    planner = getattr(brain, "planner", None)
    gaps = []
    if causal is not None and planner is not None:
        for a in getattr(planner, "actions", []) or []:
            for k in dict(getattr(a, "eff", {}) or {}):
                counts = causal.counts(a.name, k) if callable(getattr(causal, "counts", None)) else None
                if counts:
                    met, total = counts
                    gaps.append(abs((met + 1) / (total + 2) - met / total))
    if gaps:
        print(f"calibration gap (mean |predicted - observed|, from causal records, n={len(gaps)}):",
              round(sum(gaps) / len(gaps), 4))
    else:
        print("calibration gap: not measured (no causal records / no CalibrationTracker yet)")
    # Repeated-mistake events in the treasure run's event log.
    events = getattr(brain, "events", None) or []
    rm = [e for e in events if isinstance(e, dict) and e.get("kind") == "repeated_mistake"]
    print("repeated-mistake events:", len(rm),
          "|", [e.get("detail", "") for e in rm][:3])
    print("pattern_applied events:",
          sum(1 for e in events if isinstance(e, dict) and e.get("kind") == "pattern_applied"))

    header("7. Autonomous execution and security (Phase 7) — session, ledger, kill")
    # Phase 7 APIs (ab/execution/: ExecutionSession, ExecutionMonitor,
    # ExecutionLedger; ab/body/sandbox.py: IsolatedToolRunner,
    # ScopedFilesystem; Brain.cancel_execution; persistence v5 +
    # execution.json) may not have landed yet — every access below is
    # guarded, so this section reports "not present" instead of
    # crashing, exactly like §6 did in Phase 6. When the APIs land,
    # this section prints the live session / ledger summary for the
    # treasure run above plus a demonstrated timeout-kill.
    monitor = getattr(brain, "execution_monitor", None)
    session = getattr(brain, "execution_session", None)
    if session is None and monitor is not None:
        session = getattr(monitor, "session", None) or getattr(monitor, "current_session", None)
    ledger = getattr(brain, "execution_ledger", None)
    if ledger is None and monitor is not None:
        ledger = getattr(monitor, "ledger", None)

    def _ledger_len(obj):
        if obj is None:
            return None
        recs = getattr(obj, "records", None)
        recs = recs() if callable(recs) else recs
        return len(list(recs or [])) if recs is not None else None

    if session is None:
        if monitor is None:
            print("execution session: not present in this build (Phase 7 not integrated yet)")
        else:
            print("execution session: monitor present, but no session was opened during this run")
    else:
        print("execution session:",
              getattr(session, "status", "?"), "| task:",
              getattr(session, "task_id", "?") or "?",
              "| steps executed:",
              getattr(session, "steps_executed", "?"), "/",
              getattr(session, "steps_total", "?"),
              "| tool calls:", getattr(session, "tool_calls", "?"),
              "| stop reason:", getattr(session, "stop_reason", None))
    _n = _ledger_len(ledger)
    print("execution ledger:",
          "not present in this build (Phase 7 not integrated yet)" if _n is None
          else f"{_n} records")
    print("cancel_execution:",
          "present" if callable(getattr(Brain, "cancel_execution", None))
          else "not present in this build (Phase 7 not integrated yet)")
    print("persistence version:",
          getattr(Brain, "PERSISTENCE_VERSION", "?"),
          "| execution.json:",
          "present" if "execution.json" in getattr(Brain, "FILE_SCHEMA_VERSIONS", {}) else "not present (Phase 7 not integrated yet)")
    # Demonstrated timeout-kill, only if Worker 1's isolated runner exists.
    try:
        from .body.sandbox import IsolatedToolRunner  # noqa: F401
    except Exception:
        IsolatedToolRunner = None
    if IsolatedToolRunner is None:
        print("isolated tool runner: not present in this build (Phase 7 not integrated yet)")
        print("timeout-kill demonstration: not run — no isolated runner to kill with")
    else:
        # Runner exists: try its run-shaped entry points with a tool
        # that sleeps far beyond its budget, and report whether the
        # process was observed dead. Method names are probed, never
        # assumed; anything unexpected is reported, not hidden.
        _runner = None
        try:
            _runner = IsolatedToolRunner()
        except Exception as exc:  # noqa: BLE001 - guarded probe
            print("isolated tool runner: present but could not construct:", repr(exc))
        _method = None
        if _runner is not None:
            for _name in ("run", "run_isolated", "call", "run_tool"):
                _method = getattr(_runner, _name, None)
                if callable(_method):
                    break
        if _runner is None or _method is None:
            print("isolated tool runner: present but no run-shaped method found — kill not demonstrated")
        else:
            import time as _time
            def _sleeper():
                _time.sleep(30)
                return "should never return"
            # Landed API (worker 1, ab/body/sandbox.py): run(fn, args,
            # timeout) -> IsolatedResult with ok / timed_out / killed /
            # observed_dead / limits_applied. Call it exactly that way;
            # anything unexpected is reported, not hidden.
            try:
                _res = _method(_sleeper, {}, timeout=1.0)
            except TypeError:
                try:
                    _res = _method(_sleeper, {})
                except Exception as exc:  # noqa: BLE001 - guarded probe
                    _res = None
                    print("timeout-kill demonstration: runner call failed:", repr(exc))
            except Exception as exc:  # noqa: BLE001 - guarded probe
                _res = None
                print("timeout-kill demonstration: runner call failed:", repr(exc))
            if _res is not None:
                _get = (lambda k, d=None: getattr(_res, k, _res.get(k, d) if isinstance(_res, dict) else d))
                print("timeout-kill demonstration: timed_out =", _get("timed_out"),
                      "| killed =", _get("killed"),
                      "| observed_dead =", _get("observed_dead"),
                      "| ok =", _get("ok"),
                      "| exitcode =", _get("exitcode"),
                      "| limits_applied =", _get("limits_applied"))
    try:
        from .body.sandbox import ScopedFilesystem  # noqa: F401
    except Exception:
        ScopedFilesystem = None
    print("filesystem scope:",
          "present" if ScopedFilesystem is not None else "not present in this build (Phase 7 not integrated yet)")


    header("8. Perception and imagination (Phase 8) — one bus, simulation over the world model")
    # Perception: one bus for every source. Document: one doc stores
    # a second independent doc corroborates to verified.
    _tmp = tempfile.mkdtemp()
    _doc1 = os.path.join(_tmp, "paris-notes.md")
    _doc2 = os.path.join(_tmp, "atlas.md")
    with open(_doc1, "w", encoding="utf-8") as _fh:
        _fh.write("# Paris notes\n\nParis is the capital of France. "
                  "Mars is a planet. The sky is very blue today.\n")
    with open(_doc2, "w", encoding="utf-8") as _fh:
        _fh.write("Paris is the capital of France.\n")
    _pb = Brain()
    print("specialist attached:", _pb.specialist,
          "(none — the brain works fully with the stub/built-in paths)")
    _rep1 = _pb.read_document(_doc1)
    print("doc1 report:", {k: _rep1[k] for k in
                           ("stored", "corroborated", "duplicate", "conflict")},
          "| ignored:", _rep1["ignored_sentences"])
    print("ask paris capital_of after one doc:", _pb.ask("paris", "capital_of").best)
    _rep2 = _pb.read_document(_doc2)
    print("doc2 report:", {k: _rep2[k] for k in
                           ("stored", "corroborated", "duplicate", "conflict")})
    print("ask paris capital_of after two docs:", _pb.ask("paris", "capital_of").best)
    print("ask zorblax capital_of:", _pb.ask("zorblax", "capital_of").message)
    # Image: a deterministic stub sensor describes a scene; the record
    # enters the world model as observed entities + spatial relations.
    from .perception.image import StubImageSensor
    _sensor = StubImageSensor({
        "desk.png": {"objects": [
            {"name": "cup", "properties": {"color": "red"},
             "position": {"x": 10, "y": 20}, "confidence": 0.9},
            {"name": "plate", "properties": {"color": "white"},
             "position": {"x": 40, "y": 20}, "confidence": 0.85}],
            "relations": [{"subject": "cup", "relation": "left_of",
                           "object": "plate", "confidence": 0.9}],
            "source": "stub", "image_id": "desk.png"}})
    _record = _sensor.describe_image("desk.png")
    _percept = _pb.perceive("image", _record, source="stub")
    print("image percept:", _percept.status,
          "| provenance:", _percept.provenance)
    _wm = _pb.world_model
    _cup = _wm.entity("cup")
    print("world-model cup:", None if _cup is None else
          {k: (v.value, v.status.value) for k, v in _cup.properties.items()})
    print("world-model relations:",
          [(r.subject, r.relation, r.object, r.status.value)
           for r in _wm.relations()])
    print("percepts on the bus:", len(_pb.perception_bus.percepts),
          "| kinds:", [p.kind for p in _pb.perception_bus.percepts])


    print("\n-- Imagination: simulate routes, compare, select, execute, compare imagined vs actual --")
    # Fresh RoomWorld brain: sense once (imagine() works from the
    # current world state), then imagine routes to the treasure.
    env8, brain8 = RoomWorld(), Brain(room_world_actions(), granted={"compute", "environment"})
    brain8.world = dict(env8.observe())
    imagined = brain8.imagine({"has_treasure": True})
    print(f"imagined {len(imagined['scenarios'])} route(s) to the treasure (simulated, not observed):")
    for entry in imagined["ranking"]:
        print(f"  {entry['name']}: {' > '.join(entry['steps'])} — {entry['reason']}")
    best8 = imagined["best"]
    print("best imagined route:", best8["name"],
          "| predicted success:", best8["predicted_success"],
          "| predicted cost:", best8["predicted_cost"])
    print("assumptions (first two):", best8["assumptions"][:2])
    # What-if and counterfactual, answered from simulation.
    goal8 = brain8.add_goal("get the treasure", {"has_treasure": True})
    brain8.goals.activate(goal8.id)
    print("what if:", brain8.what_if("what if I use pick_lock?")["answer"])
    lab8 = Brain(room_world_actions(), granted={"compute", "environment"})
    lab8.world = {"at": "lab", "has_key": False, "door_open": False, "has_treasure": False}
    cf8 = lab8.counterfactual(edits={"door_open": True},
                              steps=["go_vault", "take_treasure"],
                              goal={"has_treasure": True})
    print("counterfactual:", cf8["answer"])
    print("changed assumption:", cf8["assumptions"][1])
    print("real world model untouched by the counterfactual:",
          lab8.world_model.status_of("door", "open"),
          "| state still:", dict(lab8.world))
    # Execute for real, then compare the imagined outcome with reality.
    # The run re-plans with simulation ranking: the pick_lock route is
    # cheapest on paper, fails in reality, and the revision loop feeds
    # that failure back into predicted success.
    brain8.run(env8, 60)
    print("executed; outcome:", brain8.goals.all()[-1].status,
          "| cycles:", brain8.cycle)
    actual8 = dict(env8.observe())
    matched8 = [k for k in best8["predicted_state"]
                if best8["predicted_state"].get(k) == actual8.get(k)]
    mismatched8 = [k for k in best8["predicted_state"]
                   if best8["predicted_state"].get(k) != actual8.get(k)]
    print("imagined vs actual: matched keys:", matched8,
          "| mismatched keys:", mismatched8)
    print("prediction errors recorded:",
          len(brain8.world_model.prediction_errors()),
          "| revised predicted success:",
          {f"{a}/{e}": round(v, 2)
           for (a, e), v in brain8.simulation_revision.adjusted_success.items()})
    print("simulation_compared events:",
          sum(1 for e in brain8.events
              if isinstance(e, dict) and e.get("kind") == "simulation_compared"))

    _section10()
    _section11()


def _tier3_tasks(tasks):
    """The Tier-3 tasks in a ``run_benchmarks`` result (P10.5-6):
    a task is Tier 3 iff its recorded ``tier`` field is 3.
    ``run_benchmarks`` records ``tier`` (1|2|3) on every task, so
    selection reads the field and nothing else. A task without a
    tier, or with a tier outside 1|2|3, is a harness bug: this
    function raises ``ValueError`` instead of guessing -- no
    task-id fallback, no criterion-text match, no difficulty-vector
    recount. The caller (``_section10``) reports the failure
    loudly and shows nothing, rather than showing the wrong tasks."""
    out = {}
    for name, task in (tasks or {}).items():
        if not isinstance(task, dict):
            raise ValueError(
                f"benchmark task {name!r} is not a task record "
                f"(got {type(task).__name__}); refusing to select "
                "Tier-3 tasks from a malformed run_benchmarks result")
        tier = task.get("tier")
        if tier is None:
            raise ValueError(
                f"benchmark task {name!r} carries no tier field; "
                "refusing to guess its tier from its task id "
                "(see P10.5-6: every task must record tier 1|2|3)")
        if tier not in (1, 2, 3):
            raise ValueError(
                f"benchmark task {name!r} carries an invalid tier "
                f"{tier!r} (expected 1, 2 or 3)")
        if tier == 3:
            out[name] = task
    return out


def _first_present(mapping, keys):
    """First present, non-None value among ``keys`` in a dict, or None."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _section10():
    """10. Complex real-world environments (Phase 10) — a Tier-3
    benchmark task run end to end: goal, imagined route if any,
    execution, outcome. Runs Worker 2's ``run_benchmarks(seed=0)``
    and shows the Tier-3 tasks from it (the composite worlds that
    combine >= 3 difficulty dimensions). Defensive like §6/§7: if
    the benchmarks module has not landed yet, this section reports
    that and the demo still exits 0."""
    header("10. Complex real-world environments (Phase 10) — Tier-3 benchmark task end to end")
    try:
        import importlib

        _bm = importlib.import_module("ab.eval.benchmarks")
    except Exception as exc:  # noqa: BLE001 - guarded probe
        print("benchmarks not landed yet (Worker 2):",
              f"{type(exc).__name__}: {exc}")
        print("Tier-3 demonstration: not run -- no benchmark module to run")
        return
    fn = getattr(_bm, "run_benchmarks", None)
    if fn is None:
        print("benchmarks module present but no run_benchmarks entry point")
        print("Tier-3 demonstration: not run")
        return
    try:
        result = fn(seed=0)
    except TypeError:
        result = fn()
    tasks = (result or {}).get("tasks") or {}
    if not tasks:
        print("run_benchmarks returned no tasks")
        print("Tier-3 demonstration: not run -- nothing to show")
        return
    try:
        tier3 = _tier3_tasks(tasks)
    except ValueError as exc:  # noqa: BLE001 - loud, no silent fallback
        print(f"Tier-3 demonstration: FAILED -- {exc}")
        print("Tier-3 demonstration: not run -- no Tier-3 task shown "
              "rather than a guessed one")
        return
    if not tier3:
        print("benchmarks: "
              f"{len(tasks)} task(s), "
              f"{sum(1 for t in tasks.values() if isinstance(t, dict) and t.get('passed'))} "
              "passed by the full brain (seed 0)")
        print("no task recorded itself as Tier 3; nothing to show")
        return
    show = tier3
    print(f"benchmarks: {len(tasks)} task(s), "
          f"{sum(1 for t in tasks.values() if isinstance(t, dict) and t.get('passed'))} "
          f"passed by the full brain (seed 0)")
    for name, task in show.items():
        full = ((task.get("variants") or {}).get("full") or task
                if isinstance(task, dict) else {})
        evidence = (task.get("evidence") or {}) if isinstance(task, dict) else {}
        goal = _first_present(
            evidence, ("goal", "goal_target", "target", "criterion"))
        if goal is None:
            goal = task.get("criterion") or task.get("detail") or "goal not recorded"
        imagined = _first_present(
            evidence, ("imagined_route", "best_scenario", "imagined",
                       "plan", "route"))
        print(f"\n-- Tier-3 task: {name} --")
        print("goal:", goal)
        print("imagined route:",
              imagined if imagined is not None
              else "none recorded (no imagined route in the evidence)")
        print("execution: cycles =", task.get("cycles"),
              "| stop reason =",
              _first_present(evidence, ("stop_reason", "stop", "outcome")),
              "| failed attempts =",
              _first_present(evidence, ("failed_attempts", "failures")))
        stages = evidence.get("stages_in_order")
        if stages is not None:
            print("stages in order:", stages,
                  "| stage counts:", evidence.get("stage_counts"))
        print("outcome:",
              "PASS" if task.get("passed") else "FAIL",
              "|", task.get("detail", ""))
        if task.get("failure_cause"):
            print("failure cause:", task["failure_cause"])


def _section11():
    """11. Research workflow (Phase 11 §2.5) — one research-workflow
    trace: question, sources selected, admitted/held outcomes.
    Separate, clearly delimited section appended after §10; it does
    not touch the §3 pages dict or any other demo section. The names
    here (Dunmere / Caldoria) are demo-only and differ from the
    held-out research-task names, so the leakage audit stays clean.
    Fast: two in-memory pages, one research goal, one run."""
    header("11. Research workflow (Phase 11) — question, sources, admitted/held")
    from .body.providers import MemoryKnowledgeProvider
    pages = {"https://alpha.org/p": "Dunmere is the capital of Caldoria.",
             "https://beta.org/p": "Dunmere is the capital city of Caldoria."}
    rb = Brain(researcher=Researcher(
        MemoryKnowledgeProvider(pages), None),
        granted={"compute", "network"})
    goal = rb.add_research_goal("dunmere", "capital_of", list(pages))
    print("research question:", goal.meta.get("research_question"))
    print("sources selected:", goal.meta.get("research_sources"))
    rb.run(None, 30)
    print("goal:", goal.status, "| cycles:", rb.cycle)
    print("answer:", rb.ask("dunmere", "capital_of").best)
    for kind in ("research_admitted", "research_held"):
        for e in rb.events:
            if isinstance(e, dict) and e.get("kind") == kind:
                print(f"  {kind}: {e.get('detail')}")


if __name__ == "__main__":
    main()
