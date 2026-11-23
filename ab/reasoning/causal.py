"""Causal model (Phase 4): what reliably causes what, from experience.

Records (action, effect) outcomes and only claims causation when the
evidence is strong: the effect was met at least 3 times with zero
counterexamples. A single counterexample blocks a causal claim — the
brain must not assert "X causes Y" after seeing X fail to produce Y.
Claims are data for abduction, never proof on their own.

Phase 9 (P9-1): records are VARIANT-aware. A variant is the canonical
precondition set of the action variant that actually ran (see
``ab.planning.planner.canonical_variant``), so two variants of one
action name (e.g. press-with-mould vs press-without-mould) keep
separate rates instead of sharing one. ``record`` / ``success_rate`` /
``counts`` / ``causes`` gain an optional ``variant`` parameter.

Storage is two-level, so the pre-Phase-9 contract is preserved
exactly:

- ``_records``: the pooled ``(action, effect) -> [met, total]``
  cell, kept in its original 2-tuple shape — this is the cell
  ``WorldModel.predict`` reads directly, and what every
  ``variant=None`` call returns. A record with a named variant
  contributes to the pooled cell too (it IS an observation of the
  action/effect pair); a record with ``variant=None`` contributes
  only to the pooled cell.
- ``_variant_records``: the per-variant ``(action, effect, variant)
  -> [met, total]`` cells, populated only by records that name a
  variant. A named-variant read consults exactly one cell here, so
  a reliable variant can be claimed while a failing variant of the
  same action name is not — the pooled cell, which mixes them, can
  never make that distinction.
"""
from __future__ import annotations

from ..core.vocab import normalize_text

MIN_OBSERVATIONS = 3


class CausalModel:
    def __init__(self):
        # (action, effect_key) -> [met, total] (pooled over variants).
        self._records: dict[tuple, list] = {}
        # (action, effect_key, variant) -> [met, total] (per variant).
        self._variant_records: dict[tuple, list] = {}

    @staticmethod
    def _key(action, effect_key, variant=None):
        return (normalize_text(action), normalize_text(effect_key),
                None if variant is None else str(variant))

    def record(self, action: str, effect_key: str, met: bool,
               variant=None) -> None:
        pooled_key = (normalize_text(action), normalize_text(effect_key))
        cell = self._records.setdefault(pooled_key, [0, 0])
        cell[0] += int(bool(met))
        cell[1] += 1
        if variant is not None:
            vcell = self._variant_records.setdefault(
                self._key(action, effect_key, variant), [0, 0])
            vcell[0] += int(bool(met))
            vcell[1] += 1

    def success_rate(self, action, effect_key, variant=None) -> float | None:
        """Fraction of records where the effect was met, or None when
        nothing has been recorded for this (action, effect). With a
        named ``variant`` the rate is that variant's alone when the
        variant has any record; when it has none, the pooled rate is
        the prior (a legacy ``variant=None`` record stays visible to
        every variant as the pre-Phase-9 rate — the fallback, never
        an override of variant evidence). ``variant=None`` (the
        default, backwards compatible) is the pooled rate."""
        if variant is None:
            cell = self._records.get(
                (normalize_text(action), normalize_text(effect_key)))
        else:
            cell = self._variant_records.get(
                self._key(action, effect_key, variant))
            if cell is None:
                cell = self._records.get(
                    (normalize_text(action), normalize_text(effect_key)))
        if cell is None or cell[1] == 0:
            return None
        return cell[0] / cell[1]

    def counts(self, action, effect_key, variant=None) -> tuple[int, int] | None:
        """Phase 5: raw (met, total) counts for (action, effect_key),
        or None when nothing has been recorded for the pair. The
        brain's prediction confidence is the Laplace-smoothed rate
        (met + 1) / (total + 2) derived from these counts. With a
        named ``variant`` the counts are that variant's alone when
        the variant has any record, else the pooled counts (same
        prior fallback as ``success_rate``); with ``variant=None``
        they are pooled over all recorded variants."""
        if variant is None:
            cell = self._records.get(
                (normalize_text(action), normalize_text(effect_key)))
        else:
            cell = self._variant_records.get(
                self._key(action, effect_key, variant))
            if cell is None:
                cell = self._records.get(
                    (normalize_text(action), normalize_text(effect_key)))
        if cell is None or cell[1] == 0:
            return None
        return (int(cell[0]), int(cell[1]))

    @staticmethod
    def _claim(action, effect, variant, met, total):
        counterexamples = total - met
        if met < MIN_OBSERVATIONS or counterexamples != 0:
            return None
        return {"action": action, "effect": effect, "variant": variant,
                "confidence": met / total,
                "observations": total, "counterexamples": counterexamples}

    def causes(self, action, effect_key, variant=None) -> dict | None:
        """A causal claim, only when met >= 3 and counterexamples == 0.
        Returns {"action", "effect", "variant", "confidence",
        "observations", "counterexamples"} or None when the bar is not
        met. With a named ``variant`` the claim is about that variant
        alone (Phase 9: a reliable variant can be claimed while a
        failing variant of the same action name is not); with
        ``variant=None`` the pooled counts are used. Unlike the rate
        reads, a claim NEVER falls back to the pooled counts: a
        variant with no record of its own claims nothing, even when
        the pooled counts would meet the bar — a causal claim needs
        variant-specific evidence."""
        if variant is None:
            key = (normalize_text(action), normalize_text(effect_key))
            cell = self._records.get(key)
            if cell is None:
                return None
            return self._claim(key[0], key[1], None, cell[0], cell[1])
        key = self._key(action, effect_key, variant)
        cell = self._variant_records.get(key)
        if cell is None:
            return None
        return self._claim(key[0], key[1], key[2], cell[0], cell[1])

    def claims(self) -> list[dict]:
        """All causal claims currently supported, sorted deterministically
        by (action, effect, variant): one per pooled cell that meets
        the bar (``variant=None``) and one per variant cell that
        meets it. A reliable variant's claim is therefore listed even
        when the pooled cell is blocked by another variant's
        failures."""
        out = []
        for (action, effect), cell in sorted(self._records.items()):
            claim = self._claim(action, effect, None, cell[0], cell[1])
            if claim is not None:
                out.append(claim)
        for (action, effect, variant), cell in sorted(
                self._variant_records.items(),
                key=lambda k: (k[0][0], k[0][1], str(k[0][2]))):
            claim = self._claim(action, effect, variant, cell[0], cell[1])
            if claim is not None:
                out.append(claim)
        return out


class NoOpCausal(CausalModel):
    """Phase 10 (P10-5): a causal model that never learns. ``record``
    is a no-op, ``counts`` / ``success_rate`` / ``causes`` always
    report no evidence, and ``claims`` is always empty, so a brain
    built with this model can never make a causal claim — and every
    consumer that reads a rate through ``laplace_rate`` falls back
    to the uninformative prior (``None`` -> the simulator's 1.0
    default). It is a ``CausalModel`` subclass so the same interface
    is honoured everywhere the real model is used (planner
    confidence, engine abduction, simulator rates, revision);
    nothing in ``CausalModel`` itself changes. Used only as the
    ``no_causal`` ablation: it must be injected at ``Brain``
    construction (``Brain(causal_model=NoOpCausal())``), never
    swapped into a live brain — the causal model is threaded
    through planner / engine / simulator at construction, and a
    post-construction swap would leave stale references."""

    def record(self, action: str, effect_key: str, met: bool,
               variant=None) -> None:
        return None

    def success_rate(self, action, effect_key, variant=None):
        return None

    def counts(self, action, effect_key, variant=None):
        return None

    def causes(self, action, effect_key, variant=None):
        return None

    def claims(self) -> list[dict]:
        return []
