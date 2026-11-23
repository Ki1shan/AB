"""Reading loop (v0.2 draft, Phase 8 P8-2): learn by reading a local doc.

A document is untrusted data until corroborated, like the web — it is
never instructions. File source trust is 0.45, the same as ``web``
(``TRUST["file"]`` in ``ab/core/memory.py``; Phase 8 design decision
P8-2, recorded as a Phase 11 decision to revisit — do NOT silently
raise it to 0.8 or 1.0). A single document therefore NEVER verifies a
fact: the first read stores it ``unverified``; the same fact from a
second independent document (a different ``file:<name>`` source) or a
teacher corroborates it to ``verified``. A conflicting value on a
functional relation becomes ``disputed``, never silently overwritten.
Ignored sentences are counted and listed in the report, never
silently turned into claims.

Reads are size-capped; a missing, oversize, undecodable or
unsupported file fails cleanly with no partial claims.
"""
from __future__ import annotations

import os
import re

from ..core.memory import AddResult
from .research import ClaimExtractor

# Phase 8 (P8-2): file trust 0.45, same as web. A single document
# never verifies a fact. Recorded as a Phase 11 decision to revisit.
DOCUMENT_TRUST = 0.45
MAX_DOCUMENT_BYTES = 64 * 1024  # same bound as the read_file tool (64 KiB)
SUPPORTED_EXTENSIONS = (".txt", ".md")


def read_text_file(path: str, max_bytes: int = MAX_DOCUMENT_BYTES) -> str:
    """Read a ``.txt`` / ``.md`` file as UTF-8 text, size-capped.

    Raises ``FileNotFoundError`` for a missing file, ``ValueError``
    for an unsupported extension or an oversize file, and
    ``UnicodeDecodeError`` for non-UTF-8 bytes. Nothing is returned
    on failure, so a caller can never store partial claims from a
    file that failed to read.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"unsupported document type {ext!r}: only {list(SUPPORTED_EXTENSIONS)} "
            f"are readable (plain text / Markdown)")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) > max_bytes:
        raise ValueError(
            f"document {path!r} exceeds the {max_bytes}-byte read cap; "
            f"refusing rather than reading it partially")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_markdown(text: str) -> str:
    """Strip Markdown headings / lists / emphasis / links to plain
    sentences before extraction. Plain text passes through unchanged
    (no markers to strip)."""
    out = []
    for line in str(text).splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            continue  # headings are structure, not sentences: drop, don't count as ignored
        line = re.sub(r"^\s{0,3}>\s?", "", line)                # blockquotes
        line = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line)      # list items
        line = re.sub(r"```.*$", "", line)                      # code fences
        line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)   # images
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)    # links
        line = re.sub(r"(`{1,3})([^`]+)\1", r"\2", line)        # inline code
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)         # bold
        line = re.sub(r"(\*|_)(.+?)\1", r"\2", line)            # italic
        line = re.sub(r"<[^>]+>", "", line)                     # HTML tags
        out.append(line)
    return "\n".join(out)


def split_sentences(text: str) -> list[str]:
    """The same splitter ``ClaimExtractor`` uses, so the ignored
    count and the extractor agree on what a sentence is."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


class DocumentReader:
    """Read text (or a file) into LTM via the existing two-source /
    teacher admission path: each extracted claim is stored with
    provenance ``file:<name>`` through ``LongTermMemory.add``.

    The report counts dispositions per distinct claim in this call
    (``stored`` / ``corroborated`` / ``duplicate`` / ``conflict``)
    and lists every sentence that yielded no claim in
    ``ignored_sentences``; every claim entry keeps its ``sentence``.
    """

    def __init__(self, ltm, extractor: ClaimExtractor | None = None, cycle: int = 0):
        self.ltm = ltm
        self.extractor = extractor or ClaimExtractor()
        self.cycle = cycle

    def read_text(self, text: str, source: str = "file:text") -> dict:
        """Extract claims from ``text`` and store them under ``source``
        (e.g. ``file:paris-notes.md``). Returns the reading report."""
        source_id = str(source)
        body = strip_markdown(text) if source_id.lower().endswith(".md") else str(text)
        report: dict = {"stored": 0, "corroborated": 0, "duplicate": 0,
                        "conflict": 0, "ignored_sentences": [],
                        "source": source_id, "claims": []}
        for sentence in split_sentences(body):
            claims = self.extractor.extract(sentence)
            if not claims:
                # Includes sentences over 300 chars, which the
                # extractor ignores by design: reported, never guessed.
                report["ignored_sentences"].append(sentence)
                continue
            for claim in claims:
                result, fact = self.ltm.add(claim.subject, claim.relation,
                                            claim.value, source_id=source_id,
                                            cycle=self.cycle)
                if result is AddResult.STORED:
                    report["stored"] += 1
                elif result is AddResult.CORROBORATED:
                    report["corroborated"] += 1
                elif result is AddResult.DUPLICATE:
                    report["duplicate"] += 1
                elif result is AddResult.CONFLICT:
                    report["conflict"] += 1
                report["claims"].append(
                    {"subject": claim.subject, "relation": claim.relation,
                     "value": claim.value, "sentence": claim.sentence,
                     "result": result.value, "status": fact.status})
        return report

