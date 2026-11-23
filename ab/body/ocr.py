"""OCR sensor. The brain only ever sees its text output; the engine is swappable."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class OcrUnavailable(Exception):
    pass


@dataclass
class OcrResult:
    text: str
    confidence: float          # 0..1 mean word confidence
    engine: str
    path: str


class OcrEngine:
    name = "base"

    def available(self) -> bool:
        return False

    def read(self, path: str) -> OcrResult:
        raise NotImplementedError


class TesseractOcr(OcrEngine):
    name = "tesseract"

    def available(self) -> bool:
        return shutil.which("tesseract") is not None

    def read(self, path: str) -> OcrResult:
        if not self.available():
            raise OcrUnavailable("tesseract is not installed")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        out = subprocess.run(["tesseract", path, "stdout", "--psm", "6", "tsv"],
                             capture_output=True, text=True, timeout=60, check=False)
        lines: dict[tuple, list[str]] = {}
        confs: list[float] = []
        for row in out.stdout.splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) < 12 or not cols[11].strip():
                continue
            try:
                conf = float(cols[10])
            except ValueError:
                continue
            if conf < 0:
                continue
            lines.setdefault((cols[2], cols[3], cols[4]), []).append(cols[11])
            confs.append(conf / 100.0)
        text = "\n".join(" ".join(w) for w in lines.values())
        return OcrResult(text, round(sum(confs) / len(confs), 3) if confs else 0.0, self.name, path)

    def read_words(self, path: str) -> list[dict]:
        """Per-word dicts ``{text, confidence, bbox, image}`` from the same
        TSV output ``read()`` parses (cols 6-9 are left/top/width/height,
        col 10 is the word confidence). Words with conf < 0 are skipped,
        matching ``read()``; a conf == 0 word is kept (confidence 0.0) so
        the gate — not the mean — rejects it."""
        if not self.available():
            raise OcrUnavailable("tesseract is not installed")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        out = subprocess.run(["tesseract", path, "stdout", "--psm", "6", "tsv"],
                             capture_output=True, text=True, timeout=60, check=False)
        words: list[dict] = []
        for row in out.stdout.splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) < 12 or not cols[11].strip():
                continue
            try:
                conf = float(cols[10])
                bbox = {"left": int(cols[6]), "top": int(cols[7]),
                        "width": int(cols[8]), "height": int(cols[9])}
            except ValueError:
                continue
            if conf < 0:
                continue
            words.append({"text": cols[11], "confidence": conf / 100.0,
                          "bbox": bbox, "image": path})
        return words


class StubOcr(OcrEngine):
    """Deterministic stand-in for tests: maps a path to canned text."""
    name = "stub"

    def __init__(self, table: dict[str, str], confidence: float = 0.9):
        self.table, self.confidence = table, confidence

    def available(self) -> bool:
        return True

    def read(self, path: str) -> OcrResult:
        return OcrResult(self.table.get(path, ""), self.confidence, self.name, path)

    def read_words(self, path: str) -> list[dict]:
        """Per-word dicts; the stub has no layout data, so bbox is None and
        every word carries the stub's single confidence."""
        return [{"text": w, "confidence": self.confidence, "bbox": None, "image": path}
                for w in self.table.get(path, "").split()]


# ------------------------------------------------------------------ Phase 3 §3.3: per-word confidence gate
#
# The gate runs BEFORE working-memory admission: words below the threshold
# never become WM thoughts and never become claims. They are returned in
# the ``uncertain`` list with their provenance (image, bbox if available,
# confidence). An OCR claim enters LTM only as an ``ocr`` source (trust
# 0.5 in ab/core/memory.py TRUST), so one OCR source can never alone make
# a fact verified — that is a property of LTM, verified by test, not of
# this module.

def words_above_threshold(words: list[dict], threshold: float = 0.6) -> tuple[list[dict], list[dict]]:
    """Split per-word dicts into (admitted, uncertain) at ``threshold``.

    Admitted: confidence >= threshold. Uncertain: below it — each dict
    already carries its provenance (text, confidence, bbox, image).
    """
    admitted = [w for w in words if float(w.get("confidence", 0.0)) >= threshold]
    uncertain = [w for w in words if float(w.get("confidence", 0.0)) < threshold]
    return admitted, uncertain


class OcrSensor:
    """Word-level OCR sensor: wraps any engine and gates before admission.

    ``read_words(path)`` returns per-word dicts ``{text, confidence, bbox,
    image}`` (bbox is a ``{left, top, width, height}`` dict for Tesseract,
    None for engines without layout data). ``gate(path)`` applies
    ``words_above_threshold`` and returns admitted/uncertain lists plus the
    admitted text — callers must build Thoughts/claims from the admitted
    text only, and log the uncertain list as ``ocr_uncertain`` events.
    """

    def __init__(self, engine: OcrEngine, threshold: float = 0.6):
        self.engine = engine
        self.threshold = threshold

    def read_words(self, path: str) -> list[dict]:
        read_words = getattr(self.engine, "read_words", None)
        if callable(read_words):
            return read_words(path)
        # Engine without word-level output: fall back to splitting its
        # whole-text result, every word carrying the mean confidence.
        res = self.engine.read(path)
        return [{"text": w, "confidence": res.confidence, "bbox": None, "image": path}
                for w in res.text.split()]

    def gate(self, path: str, threshold: float | None = None) -> dict:
        """Gate an image read: (admitted words, uncertain words with provenance)."""
        thr = self.threshold if threshold is None else threshold
        words = self.read_words(path)
        admitted, uncertain = words_above_threshold(words, thr)
        return {"image": path,
                "engine": getattr(self.engine, "name", "?"),
                "threshold": thr,
                "admitted": admitted,
                "uncertain": uncertain,
                "admitted_text": " ".join(w["text"] for w in admitted)}


class FakeWordOcr(OcrEngine):
    """Fake word-level OCR source for tests (never real tesseract).

    """
    name = "fake-words"

    def __init__(self, table: dict[str, list]):
        self.table = dict(table)

    def available(self) -> bool:
        return True

    def read_words(self, path: str) -> list[dict]:
        out = []
        for entry in self.table.get(path, []):
            if isinstance(entry, dict):
                out.append({"text": entry["text"],
                            "confidence": float(entry.get("confidence", 0.9)),
                            "bbox": entry.get("bbox"),
                            "image": path})
            else:
                text, conf = entry[0], float(entry[1])
                bbox = entry[2] if len(entry) > 2 else None
                out.append({"text": text, "confidence": conf, "bbox": bbox, "image": path})
        return out

