"""
Reports of the bodies that join a world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from semantic_digital_twin.world import World

# %% new body reporter


@dataclass
class NewBodyReporter:
    """
    Prints the bodies that join :attr:`world`.
    """

    world: World
    """
    The world whose new bodies are reported.
    """

    reported_body_ids: set[UUID] = field(init=False, default_factory=set)
    """
    The bodies already reported, or in :attr:`world` before the reporter was created.
    """

    def __post_init__(self) -> None:
        self.reported_body_ids = {body.id for body in self.world.bodies}

    def report(self) -> None:
        """
        Print the bodies that joined :attr:`world` since the last report, at their
        current position.
        """
        for body in self.world.bodies:
            if body.id in self.reported_body_ids:
                continue
            self.reported_body_ids.add(body.id)
            x, y, z = body.global_transform.to_np()[:3, 3]
            print(
                f"New body {body.name.name} ({body.id}) at x={x:.2f} y={y:.2f} z={z:.2f}"
            )
