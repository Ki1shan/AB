# Architecture overview

AB separates cognition, stored knowledge, and external interfaces. Its core is implemented in Python and does not require an external language model.

| Area | Responsibility |
|---|---|
| Cognitive cycle | Coordinates perception, attention, reasoning, decisions, planning, action, evaluation, and learning. |
| Knowledge and experience | Stores world state, observations, provenance, experience, and learned material. |
| External interfaces | Connects tools, files, research sources, perception providers, and simulated environments. |

The intended cycle is:

**Perceive → Attend → Control → Reason → Decide → Plan → Act → Evaluate → Learn**

The main implementation is in `ab/brain.py`; supporting components are grouped under `ab/core/`, `ab/reasoning/`, `ab/planning/`, `ab/world/`, `ab/learning/`, `ab/imagination/`, `ab/perception/`, `ab/execution/`, and `ab/body/`.

Important design constraints include provenance-aware knowledge, distinguishing observation from inference and assumption, bounded simulation, and keeping external content behind admission and trust controls. The architecture is experimental; these design goals should not be read as proof of unrestricted reliability.
