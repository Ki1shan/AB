"""World model core (Phase 5, Worker 1): entities, relations, spatial
graph, temporal log, constraints, epistemic status, permanence, partial
observability, contradictions, prediction errors, revision history.
"""
from ab.world.model import (
    PERMANENCE_DECAY,
    SCHEMA_VERSION,
    Entity,
    EntityProperty,
    EpistemicStatus,
    Relation,
    WorldEvent,
    WorldModel,
)
from ab.world.patterns import (
    PATTERN_PENALTY,
    AbstractPattern,
    PatternLibrary,
)

__all__ = [
    "PERMANENCE_DECAY",
    "SCHEMA_VERSION",
    "Entity",
    "EntityProperty",
    "EpistemicStatus",
    "Relation",
    "WorldEvent",
    "WorldModel",
    "PATTERN_PENALTY",
    "AbstractPattern",
    "PatternLibrary",
]
