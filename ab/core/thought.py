"""A Thought is the unit the brain attends to, holds, and reasons about."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

_ids = itertools.count(1)


def reserve_thought_id(n: int) -> None:
    """Advance the global thought-id counter past ``n`` (used after loading
    persisted working memory, so a fresh thought never reuses a restored id)."""
    while next(_ids) <= n:
        pass


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


@dataclass
class Thought:
    content: Any
    kind: str = "belief"          # percept | goal | belief | plan | result | question
    source: str = "internal"
    urgency: float = 0.5
    confidence: float = 1.0
    tags: frozenset = field(default_factory=frozenset)
    created_cycle: int = 0
    salience: float = 0.0
    id: int = field(default_factory=lambda: next(_ids))

    def __post_init__(self) -> None:
        self.tags = frozenset(str(t).lower() for t in self.tags)
        self.urgency = clamp(self.urgency)
        self.confidence = clamp(self.confidence)

    def signature(self) -> str:
        return f"{self.kind}:{self.content}"
