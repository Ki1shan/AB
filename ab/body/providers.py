"""Knowledge providers (Phase 11 §2.4, worker W2): pluggable fetch backends
for the Researcher, plus source-quality tracking.

A provider's only contract is ``fetch(url)`` -> a ``FetchedDoc``-like
object (fields ``url, domain, text, fetched_at, untrusted=True``) and
``spec()`` -> plain, picklable data that ``provider_from_spec`` can
rebuild the provider from. Backends:

- ``WebKnowledgeProvider`` — wraps ``WebFetcher`` (the old path).
- ``FileKnowledgeProvider`` — wraps ``read_text_file``; ``domain`` is
  ``file:<basename>`` and its registrable domain is the same string, so
  files never corroborate web claims and vice versa.
- ``MemoryKnowledgeProvider`` — in-memory ``{url: text}`` pages; the
  "book" provider, the test/demo backend, and the one an isolated child
  materializes from a spec.

Source quality (``SourceQualityTracker``) is per registrable domain
(``file:<basename>`` counts as a domain for files): fetch success rate,
claim corroboration rate (claims from this source later admitted /
claims seen) and contradiction rate (claims from this source that
became disputed / claims seen). The Researcher feeds it from its
disposition loop (hook marked ``Phase 11 (W2)`` in ``research.py``);
``Researcher.select_sources`` ranks on the resulting score.
"""
from __future__ import annotations

import os
import re
import time
from typing import Protocol
from urllib.parse import urlparse

from .web import FetchError, FetchedDoc, WebFetcher, html_to_text

# A page that carries HTML markup is converted to text, exactly as
# WebFetcher does for text/html — so the memory ("book") backend
# behaves like the web backend for the same page content. Plain text
# passes through unchanged.
_HTML_TAG = re.compile(r"</?(?:p|div|html|body|head|h[1-6]|li|ul|ol|span|a|b|i|em|strong|table|tr|td|article|section)[\s>/]", re.IGNORECASE)

# NOTE: ``read_text_file`` is imported lazily inside
# ``FileKnowledgeProvider.fetch`` — importing ``.reading`` at module
# level would cycle (``reading`` imports ``ClaimExtractor`` from
# ``research``, which imports this module).


class KnowledgeProvider(Protocol):
    """Fetch backend for the Researcher. ``fetch`` returns a
    ``FetchedDoc``; ``spec`` returns plain, picklable data."""

    name: str
    kind: str

    def fetch(self, url: str) -> FetchedDoc: ...
    def spec(self) -> dict: ...


class WebKnowledgeProvider:
    """Wraps a ``WebFetcher``. The spec carries ``allowed_domains``
    only; the reader builds a default ``WebFetcher`` from them."""

    kind = "web"

    def __init__(self, fetcher: WebFetcher | None = None,
                 allowed_domains=None, name: str = "web"):
        if fetcher is not None:
            self.fetcher = fetcher
        else:
            self.fetcher = WebFetcher(list(allowed_domains or []))
        self.name = name

    def fetch(self, url: str) -> FetchedDoc:
        return self.fetcher.fetch(url)

    def spec(self) -> dict:
        allowed = sorted(getattr(self.fetcher, "allowed", set()) or set())
        return {"kind": "web", "name": self.name,
                "allowed_domains": allowed}


class FileKnowledgeProvider:
    """Wraps ``read_text_file``. ``fetch`` accepts ``file:<name>``,
    ``file://<path>`` or a plain path (relative paths resolve under
    ``root``). ``domain`` is ``file:<basename>``."""

    kind = "file"

    def __init__(self, root: str = ".", name: str = "file"):
        self.root = str(root)
        self.name = name

    def _resolve(self, url: str) -> str:
        path = str(url)
        if path.startswith("file://"):
            path = path[len("file://"):]
        elif path.startswith("file:"):
            path = path[len("file:"):]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        return path

    def fetch(self, url: str) -> FetchedDoc:
        from .reading import read_text_file
        path = self._resolve(url)
        # read_text_file raises FileNotFoundError / ValueError /
        # UnicodeDecodeError on failure; the Researcher records those
        # as failed fetches rather than crashing.
        text = read_text_file(path)
        return FetchedDoc(url=str(url), domain=f"file:{os.path.basename(path)}",
                          text=text, fetched_at=time.time())

    def spec(self) -> dict:
        return {"kind": "file", "name": self.name, "root": self.root}


class MemoryKnowledgeProvider:
    """In-memory ``{url: text}`` pages — the "book" provider. Unknown
    URLs fail with ``FetchError`` (a failed fetch, not a crash)."""

    kind = "memory"

    def __init__(self, pages: dict | None = None, name: str = "memory"):
        self.pages = dict(pages or {})
        self.name = name

    def fetch(self, url: str) -> FetchedDoc:
        if url not in self.pages:
            raise FetchError(f"unknown page {url!r}")
        parsed = urlparse(url)
        # Web URLs carry their hostname as the domain; anything else
        # (e.g. book://x) is its own domain, the raw URL.
        host = parsed.hostname if parsed.scheme in ("http", "https") else str(url)
        text = self.pages[url]
        if _HTML_TAG.search(text):
            text = html_to_text(text)
        return FetchedDoc(url=str(url), domain=host,
                          text=text, fetched_at=time.time())

    def spec(self) -> dict:
        return {"kind": "memory", "name": self.name,
                "pages": dict(self.pages)}


def provider_from_spec(spec: dict):
    """Rebuild a provider from plain data (``provider.spec()`` output).

    A web spec carries ``allowed_domains`` only (a default
    ``WebFetcher`` is built); a memory spec carries its pages; a file
    spec carries its root (``"root"``, with ``"scope"`` accepted as an
    alias). Unknown kinds raise ``ValueError``.
    """
    if not isinstance(spec, dict):
        raise ValueError("provider spec must be a dict")
    kind = spec.get("kind")
    if kind == "web":
        return WebKnowledgeProvider(
            allowed_domains=spec.get("allowed_domains", []),
            name=spec.get("name", "web"))
    if kind == "file":
        root = spec.get("root", spec.get("scope", "."))
        return FileKnowledgeProvider(root=root, name=spec.get("name", "file"))
    if kind == "memory":
        return MemoryKnowledgeProvider(pages=spec.get("pages", {}),
                                       name=spec.get("name", "memory"))
    raise ValueError(f"unknown provider kind {kind!r}")


# ----------------------------------------------------------------------
# Source quality
# ----------------------------------------------------------------------

class SourceQualityTracker:
    """Per registrable-domain quality history over the run.

    Counts (per domain): fetch successes / failures, claims seen,
    claims admitted (corroborated), claims disputed. Rates are None
    when the denominator is zero; ``score`` substitutes a neutral 0.5
    for an unobserved component, so a source with no history scores
    0.5 and neither beats nor loses to another unobserved source.

    ``score = 0.4 * fetch_success + 0.4 * corroboration
             + 0.2 * (1 - contradiction)``

    Fetch reliability and corroboration dominate; contradiction is a
    penalty, not a veto. The weights are a Phase 11 choice, recorded
    here (not tuned against any evaluation task).
    """

    SCHEMA_VERSION = 1

    def __init__(self):
        self._stats: dict[str, dict] = {}

    # ---------------------------------------------------------- recording
    def _bucket(self, domain: str) -> dict:
        return self._stats.setdefault(str(domain), {
            "fetches_ok": 0, "fetches_failed": 0,
            "claims_seen": 0, "claims_admitted": 0, "claims_disputed": 0,
        })

    def record_fetch(self, domain: str, ok: bool) -> None:
        b = self._bucket(domain)
        b["fetches_ok" if ok else "fetches_failed"] += 1

    def record_claim(self, domain: str) -> None:
        self._bucket(domain)["claims_seen"] += 1

    def record_admitted(self, domain: str) -> None:
        self._bucket(domain)["claims_admitted"] += 1

    def record_disputed(self, domain: str) -> None:
        self._bucket(domain)["claims_disputed"] += 1

    # ---------------------------------------------------------- reading
    def domains(self) -> list[str]:
        return sorted(self._stats)

    def stats(self, domain: str) -> dict:
        """Counts plus rates (None when unobserved) for one domain."""
        b = dict(self._stats.get(str(domain), {
            "fetches_ok": 0, "fetches_failed": 0,
            "claims_seen": 0, "claims_admitted": 0, "claims_disputed": 0,
        }))
        fetches = b["fetches_ok"] + b["fetches_failed"]
        seen = b["claims_seen"]
        b["fetch_success_rate"] = (b["fetches_ok"] / fetches) if fetches else None
        b["corroboration_rate"] = (b["claims_admitted"] / seen) if seen else None
        b["contradiction_rate"] = (b["claims_disputed"] / seen) if seen else None
        return b

    def score(self, domain: str) -> float:
        """Quality score in [0, 1]; 0.5 for a domain with no history."""
        st = self.stats(domain)
        fetch = st["fetch_success_rate"] if st["fetch_success_rate"] is not None else 0.5
        corr = st["corroboration_rate"] if st["corroboration_rate"] is not None else 0.5
        contra = st["contradiction_rate"] if st["contradiction_rate"] is not None else 0.5
        return 0.4 * fetch + 0.4 * corr + 0.2 * (1.0 - contra)

    # ---------------------------------------------------------- persistence
    def to_json(self) -> dict:
        return {"version": self.SCHEMA_VERSION,
                "domains": {d: dict(b) for d, b in sorted(self._stats.items())}}

    @classmethod
    def from_json(cls, data: dict) -> "SourceQualityTracker":
        if not isinstance(data, dict):
            raise ValueError("source-quality data must be a dict")
        if data.get("version", 1) != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported source-quality schema version "
                f"{data.get('version')!r}")
        tracker = cls()
        domains = data.get("domains", {})
        if not isinstance(domains, dict):
            raise ValueError("source-quality 'domains' must be a dict")
        for domain, counts in domains.items():
            if not isinstance(counts, dict):
                raise ValueError(f"source-quality counts for {domain!r} "
                                 f"must be a dict")
            b = tracker._bucket(domain)
            for key in ("fetches_ok", "fetches_failed", "claims_seen",
                        "claims_admitted", "claims_disputed"):
                b[key] = int(counts.get(key, 0))
        return tracker
