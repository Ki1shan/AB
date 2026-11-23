"""Working memory: a small, capacity-limited, decaying workspace."""
from __future__ import annotations

from .thought import Thought, reserve_thought_id

# Phase 1: one shared pair of constants (audit A I2). Working-memory
# capacity and the attention focus size used to be three unrelated magic
# numbers (7, 3, 4) with no stated relation.
WM_CAPACITY = 7
FOCUS_K = 4


def _thought_to_json(t: Thought) -> dict:
    content = t.content
    if not isinstance(content, (str, int, float, bool)) and content is not None:
        content = str(content)
    return {"id": t.id, "content": content, "kind": t.kind, "source": t.source,
            "urgency": t.urgency, "confidence": t.confidence,
            "tags": sorted(t.tags), "created_cycle": t.created_cycle,
            "salience": t.salience}


def _thought_from_json(d: dict) -> Thought:
    t = Thought(d["content"], d.get("kind", "belief"), d.get("source", "internal"),
                d.get("urgency", 0.5), d.get("confidence", 1.0),
                frozenset(d.get("tags", [])), d.get("created_cycle", 0),
                d.get("salience", 0.0), d["id"])
    return t


class WorkingMemory:
    def __init__(self, capacity: int = WM_CAPACITY, decay: float = 0.15, floor: float = 0.05):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity, self.decay, self.floor = capacity, decay, floor
        self._items: dict[int, Thought] = {}

    def admit(self, thought: Thought) -> list[Thought]:
        """Admit a thought. Returns thoughts that did not fit / were pushed out."""
        if thought.id in self._items:
            self._items[thought.id] = thought
            return []
        if len(self._items) >= self.capacity:
            weakest = min(self._items.values(), key=lambda t: t.salience)
            if thought.salience <= weakest.salience:
                return [thought]              # rejected: not salient enough
            del self._items[weakest.id]
            self._items[thought.id] = thought
            return [weakest]                  # displaced
        self._items[thought.id] = thought
        return []

    def tick(self) -> list[Thought]:
        """One cycle of decay. Returns thoughts that faded out of working memory."""
        faded = []
        for t in list(self._items.values()):
            t.salience *= (1.0 - self.decay)
            if t.salience < self.floor:
                faded.append(self._items.pop(t.id))
        return faded

    def contents(self) -> list[Thought]:
        return sorted(self._items.values(), key=lambda t: -t.salience)

    def __len__(self) -> int:
        return len(self._items)

    # ---- persistence (Phase 1) -------------------------------------------
    def to_json(self) -> dict:
        return {"capacity": self.capacity, "decay": self.decay, "floor": self.floor,
                "items": [_thought_to_json(t) for t in self.contents()]}

    @classmethod
    def from_json(cls, data: dict) -> "WorkingMemory":
        if not isinstance(data, dict) or "items" not in data:
            raise ValueError("working-memory file has no 'items' list")
        wm = cls(int(data.get("capacity", WM_CAPACITY)),
                 float(data.get("decay", 0.15)), float(data.get("floor", 0.05)))
        for d in data["items"]:
            t = _thought_from_json(d)
            wm._items[t.id] = t
        if wm._items:
            reserve_thought_id(max(wm._items))
        return wm
