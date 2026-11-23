"""Imagination (Phase 8, P8-5/P8-6): symbolic simulation over the world
model, scenario generation and comparison, and simulation revision.

Simulation works on a *copy* of the world model and never executes
anything; every result is labelled ``simulated`` with explicit
assumptions. Deterministic, stdlib only, CPU-first, no LLM.
"""
from ab.imagination.revision import SimulationRevision
from ab.imagination.scenarios import (
    Scenario,
    ScenarioComparator,
    ScenarioGenerator,
)
from ab.imagination.simulator import (
    MAX_SIMULATED_STEPS,
    SimulationResult,
    Simulator,
    laplace_rate,
)

__all__ = [
    "MAX_SIMULATED_STEPS",
    "Scenario",
    "ScenarioComparator",
    "ScenarioGenerator",
    "SimulationResult",
    "SimulationRevision",
    "Simulator",
    "laplace_rate",
]
