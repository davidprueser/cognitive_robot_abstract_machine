from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Elevator,
        Room,
    )
    from semantic_digital_twin.world import World

# %% buildings with an elevator


@dataclass
class ElevatorBuilding(ABC):
    """
    World-independent description of a building whose storeys one elevator connects.

    A spawned building is read from its world's annotations: an
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Elevator`,
    a :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Level`
    per storey and a
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Room` per
    room.
    """

    @abstractmethod
    def spawn(self, world: World, name: str) -> None:
        """
        Add the building's storeys, with a level annotation each, and its elevator to
        ``world``.

        :param world: The world the building is added to.
        :param name: Prefix for the names of every entity the building creates.
        """

    @abstractmethod
    def navigation_targets(self, world: World, name: str) -> list[Room]:
        """
        :param world: The world the building was spawned into.
        :param name: The prefix the building was spawned with.
        :return: The rooms of the building a robot may be sent into.
        """

    @abstractmethod
    def simulated_drive_joint_name(self, elevator: Elevator) -> str:
        """
        :param elevator: The building's elevator in the world it was spawned into.
        :return: The name the building's simulator reports ``elevator``'s drive joint
            under.
        """
