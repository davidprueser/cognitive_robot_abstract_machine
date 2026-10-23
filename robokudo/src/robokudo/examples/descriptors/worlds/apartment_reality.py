"""
The apartment as it really is, which a simulated camera renders.
"""

from __future__ import annotations

from robokudo.examples.apartment_scene import ApartmentScene
from robokudo.world_descriptor import BaseWorldDescriptor

# %% world descriptor


class WorldDescriptor(BaseWorldDescriptor):
    """
    The apartment of :class:`ApartmentScene` with every object at its place, rooted at
    ``map`` like the world of the robot.
    """

    def __init__(self) -> None:
        super().__init__(root_name="map", root_prefix=None)
        ApartmentScene().build_reality(self.world)
