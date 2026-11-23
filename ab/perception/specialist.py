"""Specialist perception (Phase 8, P8-4): an explicit replaceable interface.

A specialist model may extend perception; it must NOT perform core
reasoning, and its output arrives as a percept with confidence and
provenance like any other source (untrusted until corroborated).
Absent by default: ``Brain(specialist=None)`` works fully with the
stub / built-in paths.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SpecialistPerception(Protocol):
    """A replaceable vision/specialist perception model.

    ``describe_image`` returns a structured image record in the same
    shape as :func:`ab.perception.image.validate_image_record`:
    ``{objects: [{name, properties, position, confidence}],
    relations: [...], source, image_id}`` — never a bare string.
    """

    name: str

    def available(self) -> bool:
        """True when this specialist can currently describe images."""
        ...

    def describe_image(self, path: str) -> dict:
        """Describe the image at ``path`` as a structured record."""
        ...
