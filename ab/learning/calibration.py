"""Prediction calibration (Phase 6, design §2).

Records (predicted_confidence, outcome) pairs per (action, effect)
and measures how well predictions match reality. The calibrated
confidence is the Phase 5 Laplace formula, kept unchanged::

    calibrated = (met + 1) / (total + 2)

-- the success rate shrunk toward the 0.5 prior. Phase 6 adds the
measurement: ECE, Brier score, and the per-pair gap between mean
predicted confidence and observed success rate. With no record a
pair stays at 1.0 and is labelled uncalibrated.

Persistence stores per-pair summaries only (met/total and the
sums needed to recompute Brier exactly); raw pairs are in-memory.
After a load, ECE falls back to a per-pair aggregate (each pair's
mean predicted confidence vs its observed rate), which equals the
exact ECE whenever predictions within a pair are constant.

Stdlib only, deterministic.
"""
from __future__ import annotations


SCHEMA_VERSION = 1


def _key(action: str, effect: str) -> tuple:
    return (action, effect)


class CalibrationTracker:
    """Per-(action, effect) calibration of prediction confidence."""

    def __init__(self):
        # key -> {"met": int, "total": int, "pred_sum": float,
        #         "pred_sq_sum": float, "met_pred_sum": float}
        self._stats: dict[tuple, dict] = {}
        # Raw pairs, in-memory only: list of (pred, outcome_bool).
        self._pairs: dict[tuple, list] = {}

    # -- recording --------------------------------------------------
    def record(self, action: str, effect: str, predicted_conf: float,
               outcome: bool) -> None:
        """Record one prediction: ``predicted_conf`` in [0, 1] and
        whether the predicted effect actually happened."""
        key = _key(action, effect)
        stat = self._stats.setdefault(
            key, {"met": 0, "total": 0, "pred_sum": 0.0,
                  "pred_sq_sum": 0.0, "met_pred_sum": 0.0})
        pred = float(predicted_conf)
        outcome = bool(outcome)
        stat["total"] += 1
        stat["pred_sum"] += pred
        stat["pred_sq_sum"] += pred * pred
        if outcome:
            stat["met"] += 1
            stat["met_pred_sum"] += pred
        self._pairs.setdefault(key, []).append((pred, outcome))

    # -- per-pair queries -------------------------------------------
    def counts(self, action: str, effect: str):
        """(met, total) for a pair, or None when never recorded."""
        stat = self._stats.get(_key(action, effect))
        if stat is None:
            return None
        return (stat["met"], stat["total"])

    def calibrated_confidence(self, action: str, effect: str) -> float:
        """Laplace-smoothed success rate, or 1.0 (uncalibrated)
        when the pair has no record."""
        stat = self._stats.get(_key(action, effect))
        if stat is None or stat["total"] == 0:
            return 1.0
        return (stat["met"] + 1) / (stat["total"] + 2)

    def is_calibrated(self, action: str, effect: str) -> bool:
        """False when the pair has no record at all."""
        stat = self._stats.get(_key(action, effect))
        return stat is not None and stat["total"] > 0

    def mean_predicted(self, action: str, effect: str):
        stat = self._stats.get(_key(action, effect))
        if stat is None or stat["total"] == 0:
            return None
        return stat["pred_sum"] / stat["total"]

    def observed_rate(self, action: str, effect: str):
        stat = self._stats.get(_key(action, effect))
        if stat is None or stat["total"] == 0:
            return None
        return stat["met"] / stat["total"]

    def gap(self, action: str, effect: str):
        """|mean predicted confidence - observed success rate|,
        or None when the pair has no record."""
        mean_pred = self.mean_predicted(action, effect)
        rate = self.observed_rate(action, effect)
        if mean_pred is None or rate is None:
            return None
        return abs(mean_pred - rate)

    # -- global measures --------------------------------------------
    def total_records(self) -> int:
        return sum(s["total"] for s in self._stats.values())

    def ece(self, bins: int = 5) -> float:
        """Expected calibration error over all recorded pairs.

        Bins are equal-width over predicted confidence in [0, 1];
        ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|.
        Uses the raw in-memory pairs; after a load (summaries
        only) falls back to the per-pair aggregate, which is exact
        when predictions within a pair are constant. 0.0 when
        nothing is recorded.
        """
        total = self.total_records()
        if total == 0:
            return 0.0
        if self._pairs:
            flat = [(pred, outcome)
                    for pairs in self._pairs.values() for (pred, outcome) in pairs]
            return self._ece_from_pairs(flat, bins)
        # Summary-only fallback: one pseudo-bin per pair.
        error = 0.0
        for stat in self._stats.values():
            if stat["total"] == 0:
                continue
            mean_pred = stat["pred_sum"] / stat["total"]
            rate = stat["met"] / stat["total"]
            error += (stat["total"] / total) * abs(rate - mean_pred)
        return error

    @staticmethod
    def _ece_from_pairs(pairs: list, bins: int) -> float:
        n = len(pairs)
        if n == 0:
            return 0.0
        sums = [0.0] * bins
        outcomes = [0] * bins
        counts = [0] * bins
        for pred, outcome in pairs:
            idx = min(bins - 1, max(0, int(pred * bins)))
            sums[idx] += pred
            outcomes[idx] += 1 if outcome else 0
            counts[idx] += 1
        error = 0.0
        for idx in range(bins):
            if counts[idx] == 0:
                continue
            error += (counts[idx] / n) * abs(
                outcomes[idx] / counts[idx] - sums[idx] / counts[idx])
        return error

    def brier(self) -> float:
        """Mean squared error between predicted confidence and
        outcome (1.0 = happened, 0.0 = did not). 0.0 when nothing
        is recorded. Computable exactly from the persisted sums."""
        total = self.total_records()
        if total == 0:
            return 0.0
        sq = sum(s["pred_sq_sum"] for s in self._stats.values())
        met_pred = sum(s["met_pred_sum"] for s in self._stats.values())
        met = sum(s["met"] for s in self._stats.values())
        # sum (pred - outcome)^2 = sum pred^2 - 2*sum(pred*outcome)
        #   + sum outcome^2 ; outcome^2 == outcome, sum == met.
        return (sq - 2.0 * met_pred + met) / total

    def summary(self) -> dict:
        """Per-pair calibration plus global ECE / Brier."""
        pairs = {}
        for (action, effect) in sorted(self._stats):
            stat = self._stats[(action, effect)]
            pairs[f"{action}|{effect}"] = {
                "met": stat["met"], "total": stat["total"],
                "observed_rate": (stat["met"] / stat["total"]
                                  if stat["total"] else None),
                "mean_predicted": (stat["pred_sum"] / stat["total"]
                                   if stat["total"] else None),
                "calibrated": self.calibrated_confidence(action, effect),
                "gap": self.gap(action, effect),
                "is_calibrated": True,
            }
        return {"pairs": pairs,
                "total_records": self.total_records(),
                "ece": self.ece(),
                "brier": self.brier()}

    # -- persistence ------------------------------------------------
    def to_json(self) -> dict:
        return {"version": SCHEMA_VERSION,
                "pairs": [{"action": a, "effect": e,
                           "met": s["met"], "total": s["total"],
                           "pred_sum": s["pred_sum"],
                           "pred_sq_sum": s["pred_sq_sum"],
                           "met_pred_sum": s["met_pred_sum"]}
                          for (a, e), s in sorted(self._stats.items())]}

    @classmethod
    def from_json(cls, data: dict) -> "CalibrationTracker":
        if not isinstance(data, dict):
            raise ValueError("calibration data must be a dict")
        tracker = cls()
        for entry in data.get("pairs", []):
            key = _key(entry["action"], entry["effect"])
            tracker._stats[key] = {
                "met": int(entry.get("met", 0)),
                "total": int(entry.get("total", 0)),
                "pred_sum": float(entry.get("pred_sum", 0.0)),
                "pred_sq_sum": float(entry.get("pred_sq_sum", 0.0)),
                "met_pred_sum": float(entry.get("met_pred_sum", 0.0)),
            }
        return tracker

