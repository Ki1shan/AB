"""Simulation revision (Phase 8, P8-5): close the imagine → execute loop.

After execution, ``SimulationRevision`` compares an imagined outcome
with what actually happened, records the prediction error in the world
model (via the existing ``WorldModel.record_prediction_error`` — never
a second, parallel store), and moves the simulator's per-(action,
effect) predicted success toward the observed rate. A second,
structurally similar task then simulates with the revised rate, so a
predicted success/cost that was wrong moves toward the observed value
— or the revision honestly reports that it did not move.

The adjustment is a single exponential step toward the observed
outcome (``alpha`` = 0.5 by default): with a Laplace-smoothed causal
rate as the prior when one exists, 1.0 (the planner's untested model)
otherwise. One observed failure therefore halves the predicted
success of an untested step; repeated observations converge on the
observed rate. No revision ever rewrites history: every adjustment is
appended to ``history`` and the world-model error record is the
durable trail (design P8-7: no persistence version bump — revisions
ride on existing persisted structures).

Phase 9 (P9-1): adjustments are variant-aware, in two tables mirroring
the causal model. ``adjusted_success`` keeps its original
``(action, effect)`` pooled shape — an observation of any variant
moves it, so the pre-Phase-9 pooled behaviour is preserved exactly.
``adjusted_success_by_variant`` holds ``(action, effect, variant)``
cells: a read for a named variant uses its own cell when one exists
and only falls back to the pooled cell (then the causal rate) when
it has none, so one variant's failure never drags down another
variant's prediction once variant evidence exists.
"""
from __future__ import annotations

from .simulator import laplace_rate

# Phase 12 (P12-W7): the append-only adjustment trail is a bounded
# rolling window — the newest 1,000 records, oldest-first eviction,
# the same bound as the audit log (MAX_EVENTS, ab/core/audit.py) and
# the execution ledger. The trail grew ~0.6 records/cycle in the
# soak (1,829 records, ~960 KiB at 3,000 cycles). Every reader needs
# only recent records: compare() filters by the CURRENT cycle (the
# just-appended records are newest and never pruned first), and
# persistence round-trips short trails unchanged. The durable
# evidence is NOT in the trail: the per-(action, effect[, variant])
# adjusted rates (keyed aggregates, bounded by distinct pairs) and
# the world-model prediction errors carry it.
MAX_REVISION_HISTORY = 1000


class SimulationRevision:
    """Per-(action, effect, variant) predicted-success adjustments
    from observed outcomes, plus imagined-vs-actual comparison."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        # (action, effect) -> adjusted predicted success in [0, 1]
        # (pooled over variants; the pre-Phase-9 table, unchanged).
        self.adjusted_success: dict[tuple, float] = {}
        # (action, effect, variant) -> adjusted predicted success
        # (Phase 9, P9-1; variant=None entries live here only when an
        # outcome was observed with no variant to name).
        self.adjusted_success_by_variant: dict[tuple, float] = {}
        # Append-only trail of adjustments.
        self.history: list[dict] = []

    # ---- persistence (Phase 9 §4.5, decision P9-2) ----------------------

    def to_json(self) -> dict:
        """JSON-safe form: pooled keys serialise as ``[action, effect,
        adjusted]`` triples (the pre-Phase-9 shape) and variant keys
        as ``[action, effect, variant, adjusted]`` quadruples (tuples
        are not JSON keys; a None variant serialises as null);
        history records are already plain dicts."""
        return {"version": 1, "alpha": float(self.alpha),
                "adjusted_success": [[str(a), str(e), float(v)]
                                     for (a, e), v in sorted(
                                         self.adjusted_success.items())],
                "adjusted_success_by_variant": [
                    [str(a), str(e), var, float(v)]
                    for (a, e, var), v in sorted(
                        self.adjusted_success_by_variant.items(),
                        key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2])))],
                "history": [dict(r) for r in self.history]}

    @classmethod
    def from_json(cls, data: dict) -> "SimulationRevision":
        """Restore a ``to_json`` record. Unknown / malformed entries
        are skipped, never fatal — a corrupt adjustment is dropped,
        the table is not. The pooled table accepts the pre-Phase-9
        triple form; the variant table accepts quadruples (and, for
        tolerance, triples in the pooled table with a variant slot
        are read as variant entries)."""
        if not isinstance(data, dict):
            raise ValueError("revision data must be a dict")
        rev = cls(alpha=float(data.get("alpha", 0.5)))
        for item in data.get("adjusted_success", []) or []:
            try:
                if len(item) == 4:
                    action, effect, variant, value = item
                    rev.adjusted_success_by_variant[
                        (str(action), str(effect),
                         None if variant is None else str(variant))] = float(value)
                elif len(item) == 3:
                    action, effect, value = item
                    rev.adjusted_success[(str(action), str(effect))] = float(value)
                else:
                    continue
            except (TypeError, ValueError):
                continue
        for item in data.get("adjusted_success_by_variant", []) or []:
            try:
                action, effect, variant, value = item
                rev.adjusted_success_by_variant[
                    (str(action), str(effect),
                     None if variant is None else str(variant))] = float(value)
            except (TypeError, ValueError):
                continue
        for record in data.get("history", []) or []:
            if isinstance(record, dict):
                rev.history.append(dict(record))
        # Phase 12 (P12-W7): an oversized pre-fix trail prunes to
        # the newest MAX_REVISION_HISTORY on load (same precedent as
        # the world-model event log, ADR P12-11).
        if len(rev.history) > MAX_REVISION_HISTORY:
            rev.history = rev.history[-MAX_REVISION_HISTORY:]
        return rev

    # ---- predicted success ------------------------------------------------

    def predicted_success(self, action: str, effect: str, causal=None,
                          variant=None):
        """The predicted success for (action, effect, variant): the
        variant's adjusted rate when one exists, else the pooled
        adjusted rate, else the causal Laplace rate (variant-aware,
        with a pooled fallback inside), else None (no evidence yet —
        the caller falls back to 1.0)."""
        action, effect = str(action), str(effect)
        if variant is not None:
            key3 = (action, effect, str(variant))
            if key3 in self.adjusted_success_by_variant:
                return float(self.adjusted_success_by_variant[key3])
        key2 = (action, effect)
        if key2 in self.adjusted_success:
            return float(self.adjusted_success[key2])
        rate = laplace_rate(causal, action, effect, variant=variant)
        return float(rate) if rate is not None else None

    # ---- observing an outcome ----------------------------------------------

    def observe(self, action: str, effect: str, expected, actual,
                cycle: int = 0, world_model=None, causal=None,
                variant=None) -> dict:
        """Feed one checked outcome back into the simulator's
        predictions. ``observed`` is 1.0 when the actual value equals
        the expected value, else 0.0. The adjusted rate moves
        ``alpha`` of the way from its prior (adjusted, causal
        Laplace, or 1.0) toward the observed value. When a world
        model is given, the prediction error is recorded there —
        deduplicated on (action, effect, cycle, expected, actual),
        so a caller that already recorded the same error (the brain's
        ``_check_predictions`` violation branch) does not double it.
        Returns the adjustment record. Phase 9 (P9-1): a named
        variant moves its own cell (prior: its adjusted rate, else
        its causal Laplace rate, else 1.0) AND the pooled cell (prior:
        the pooled adjusted rate, else the pooled causal Laplace
        rate, else 1.0), so the pooled table stays the aggregate of
        all variants while the variant cell keeps the separation.
        The record's prior/adjusted describe the variant cell when a
        variant was named, else the pooled cell."""
        action, effect = str(action), str(effect)
        variant = None if variant is None else str(variant)
        observed = 1.0 if expected == actual else 0.0
        key2 = (action, effect)

        def _move(key, table, prior):
            adjusted = prior + self.alpha * (observed - prior)
            adjusted = max(0.0, min(1.0, adjusted))
            table[key] = adjusted
            return adjusted

        # Pooled cell always moves: it is the aggregate evidence.
        if key2 in self.adjusted_success:
            pooled_prior = float(self.adjusted_success[key2])
        else:
            pooled_prior = laplace_rate(causal, action, effect)
            pooled_prior = float(pooled_prior) if pooled_prior is not None else 1.0
        pooled_adjusted = _move(key2, self.adjusted_success, pooled_prior)
        # Variant cell moves when a variant was named.
        if variant is not None:
            key3 = (action, effect, variant)
            if key3 in self.adjusted_success_by_variant:
                prior = float(self.adjusted_success_by_variant[key3])
            else:
                prior = laplace_rate(causal, action, effect, variant=variant)
                prior = float(prior) if prior is not None else 1.0
            adjusted = _move(key3, self.adjusted_success_by_variant, prior)
        else:
            prior, adjusted = pooled_prior, pooled_adjusted
        record = {"action": action, "effect": effect, "variant": variant,
                  "expected": expected,
                  "actual": actual, "observed": observed, "prior": round(prior, 4),
                  "adjusted": round(adjusted, 4),
                  "moved": round(adjusted - prior, 4), "cycle": int(cycle)}
        self.history.append(record)
        # Phase 12 (P12-W7): bounded rolling trail, oldest-first.
        if len(self.history) > MAX_REVISION_HISTORY:
            del self.history[:len(self.history) - MAX_REVISION_HISTORY]
        if world_model is not None:
            already = [r for r in world_model.prediction_errors()
                       if r.get("action") == action and r.get("effect") == effect
                       and r.get("cycle") == int(cycle)
                       and r.get("expected") == expected and r.get("actual") == actual]
            if not already:
                world_model.record_prediction_error(
                    action, effect, expected, actual, int(cycle))
        return dict(record)

    # ---- imagined vs actual --------------------------------------------------

    def compare(self, imagined_state: dict, actual_state: dict,
                per_step=None, cycle: int = 0, world_model=None,
                causal=None) -> dict:
        """Compare an imagined final state with the actual state.
        ``per_step`` (the simulator's per-step records) attributes
        each imagined effect to its action, so every mismatched
        effect is fed back through ``observe``. Returns
        ``{matched, mismatched, prediction_errors, outcome}`` where
        ``outcome`` is ``matched`` / ``partial`` / ``mismatched``.
        Recording goes through ``observe`` (deduplicated), so calling
        this after the brain already recorded the same errors never
        doubles them."""
        imagined_state = dict(imagined_state or {})
        actual_state = dict(actual_state or {})
        action_for_key: dict[str, str] = {}
        variant_for_key: dict[str, str | None] = {}
        for step in (per_step or []):
            for key in (step.get("effects") or {}):
                action_for_key[str(key)] = str(step.get("action", "simulation"))
                # Phase 9 (P9-1): the simulator records the variant
                # it resolved for the step; the revision inherits it.
                variant_for_key[str(key)] = step.get("variant")
        matched, mismatched = [], []
        for key in sorted(set(imagined_state) | set(actual_state)):
            expected, actual = imagined_state.get(key), actual_state.get(key)
            entry = {"key": key, "expected": expected, "actual": actual}
            if expected == actual:
                matched.append(entry)
            else:
                mismatched.append(entry)
                action = action_for_key.get(str(key), "simulation")
                self.observe(action, str(key), expected, actual, int(cycle),
                             world_model=world_model, causal=causal,
                             variant=variant_for_key.get(str(key)))
        if not mismatched:
            outcome = "matched"
        elif not matched:
            outcome = "mismatched"
        else:
            outcome = "partial"
        return {"matched": matched, "mismatched": mismatched,
                "prediction_errors": [dict(r) for r in self.history
                                      if r.get("cycle") == int(cycle)],
                "outcome": outcome}
