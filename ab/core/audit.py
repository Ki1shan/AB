"""Audit-trail helpers for AB brain (Phase 3, design SS3.4-3.5).

Standalone module: stdlib only, no imports from ``ab`` (in particular it
must NOT import ``ab.brain`` -- ``Brain`` imports this module, so a
reverse import would be circular).

Contents
--------
- :func:`redact_secrets` -- redacts secret values before anything is
  written to an event or an audit record.
- Event schema v1: :data:`EVENT_SCHEMA_VERSION`, :data:`EVENT_FIELDS`,
  :func:`validate_event`, :data:`MAX_EVENTS`, :func:`prune_events`.

  Coordinator decision (Phase 3 integration, 2026-11-21): the v1 schema
  IS Brain's established event shape ``{"cycle", "kind", "detail"}`` —
  the shape every existing test asserts and ``Brain.explain()`` renders.
  An earlier draft of this module described a different shape
  (``{"version", "cycle", "kind", "reason", "details"}``); migrating
  Brain to that shape would churn every event assertion in the suite
  for no behavioural gain. The schema version is therefore tracked by
  :data:`EVENT_SCHEMA_VERSION` and the persistence manifest, not as a
  per-event field.

Redaction policy
----------------
A value is a secret when it is associated with a key matching one of
``api_key`` / ``apikey`` / ``token`` / ``password`` / ``passwd`` /
``secret`` / ``bearer`` / ``authorization`` (case-insensitive, so
``API_KEY``, ``my_token`` and ``clientSecret`` all match). The key name
is kept; the value is replaced with ``"[REDACTED]"``.

Two further heuristics apply to plain strings:

1. ``key=value`` / ``key: value`` / ``"key": "value"`` patterns for the
   keys above have their value replaced.
2. A bare contiguous token of length >= 20 over ``[A-Za-z0-9_-]`` that
   contains a digit and at least one of (dash, underscore, uppercase)
   is treated as a token-like secret and redacted. This catches values
   such as ``sk-test-SECRET-VALUE-12345`` when they appear outside a
   key/value pair (e.g. as a bare tool result). It can over-redact long
   identifiers that happen to look token-like (a UUID, for example);
   that is a deliberate safety-first trade-off for an audit trail, and
   it never touches ordinary prose (a long lowercase word has no digit,
   a hex hash has no dash/underscore/uppercase).

``redact_secrets`` never raises: on any unexpected input it returns the
input unchanged.
"""
from __future__ import annotations

import re

EVENT_SCHEMA_VERSION = 1
EVENT_FIELDS = frozenset({"cycle", "kind", "detail"})
MAX_EVENTS = 1000

REDACTED = "[REDACTED]"

# Normalised (lowercase, punctuation stripped) secret-key words.
_SECRET_WORDS = ("apikey", "token", "password", "passwd", "secret",
                 "bearer", "authorization")

# key = value / key: value / "key": "value", for a secret key.
# Group 1: the key as written. Group 2: the separator (with any quotes
# and whitespace). Group 3: the value (up to whitespace, a quote, a
# comma, or a closing brace).
_KV_RE = re.compile(
    r"(?i)(api[_-]?key|apikey|token|passw(?:or)?d|secret|bearer|authorization)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?)"
    r"([^\s\"'}\],]+)"
)

# Contiguous token characters; length is checked by _looks_like_token.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")


def _is_secret_key(key) -> bool:
    """True when ``key`` names a secret (case-insensitive, substring match
    on the punctuation-stripped lowercase form, so ``my_token`` and
    ``clientSecret`` both match)."""
    if not isinstance(key, str):
        return False
    norm = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(word in norm for word in _SECRET_WORDS)


def _looks_like_token(text: str) -> bool:
    """Heuristic for a bare token-like secret: long, and carrying both a
    digit and something a plain word does not have (a dash, an
    underscore, or an uppercase letter)."""
    return (
        len(text) >= 20
        and any(c.isdigit() for c in text)
        and any(c in "-_" or c.isupper() for c in text)
    )


def _redact_str(text: str) -> str:
    out = _KV_RE.sub(lambda m: m.group(1) + m.group(2) + REDACTED, text)
    out = _TOKEN_RE.sub(
        lambda m: REDACTED if _looks_like_token(m.group(0)) else m.group(0),
        out,
    )
    return out


def _redact(value):
    if isinstance(value, str):
        return _redact_str(value)
    if isinstance(value, dict):
        return {
            key: (REDACTED if _is_secret_key(key) else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    return value


def redact_secrets(value):
    """Return ``value`` with secret substrings redacted.

    Works on ``str``, ``dict`` and ``list`` (a ``tuple`` is returned as a
    tuple); every other type is returned unchanged. Key names are kept,
    values are replaced with ``"[REDACTED]"``. Never raises: on odd input
    the input is returned unchanged.
    """
    try:
        return _redact(value)
    except Exception:
        return value


def validate_event(event) -> bool:
    """True iff ``event`` is a dict with exactly the allowed fields
    (:data:`EVENT_FIELDS`). Unknown fields and missing fields fail.
    The schema version is module-level (:data:`EVENT_SCHEMA_VERSION`),
    not a per-event field — see the module docstring."""
    try:
        if not isinstance(event, dict):
            return False
        return set(event.keys()) == set(EVENT_FIELDS)
    except Exception:
        return False


def prune_events(events: list) -> list:
    """Return the last :data:`MAX_EVENTS` entries, oldest-first order
    preserved. Shorter inputs are returned as a copy, unchanged."""
    try:
        items = list(events)
    except Exception:
        return []
    if len(items) > MAX_EVENTS:
        return items[-MAX_EVENTS:]
    return items


def new_event(cycle, kind, detail) -> dict:
    """Build one schema-v1 event, redacting ``detail`` first.
    Convenience for ``Brain._event``; the result always passes
    :func:`validate_event`."""
    return {
        "cycle": int(cycle),
        "kind": str(kind),
        "detail": redact_secrets(detail),
    }
