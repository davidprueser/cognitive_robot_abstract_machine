"""
Reports of the semantic annotations that join a world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    IsPerceivable,
)
from semantic_digital_twin.world import World

# %% new semantic annotation reporter


@dataclass
class NewSemanticAnnotationReporter:
    """
    Prints the semantic annotations that join :attr:`world`.
    """

    world: World
    """
    The world whose new semantic annotations are reported.
    """

    reported_annotation_ids: set[UUID] = field(init=False, default_factory=set)
    """
    The semantic annotations already reported, or in :attr:`world` before the reporter
    was created.
    """

    def __post_init__(self) -> None:
        self.reported_annotation_ids = {
            annotation.id for annotation in self.world.semantic_annotations
        }

    def report(self) -> None:
        """
        Print the semantic annotations that joined :attr:`world` since the last report,
        with the class label a perceived one was classified as and the current position
        of the body one is rooted at.
        """
        for annotation in self.world.semantic_annotations:
            if annotation.id in self.reported_annotation_ids:
                continue
            self.reported_annotation_ids.add(annotation.id)
            description = f"New {type(annotation).__name__} ({annotation.id})"
            if isinstance(annotation, IsPerceivable):
                description += f" labelled {annotation.class_label!r}"
            if isinstance(annotation, HasRootBody):
                x, y, z = annotation.root.global_transform.to_np()[:3, 3]
                description += (
                    f" on {annotation.root.name.name} at x={x:.2f} y={y:.2f} z={z:.2f}"
                )
            print(description)
