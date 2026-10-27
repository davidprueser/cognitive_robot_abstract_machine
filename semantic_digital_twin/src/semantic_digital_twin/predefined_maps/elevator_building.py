from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

# %% buildings with an elevator


@dataclass
class ElevatorBuilding(ABC):
    """
    World-independent description of a building whose ground floor and first floor an
    elevator connects.
    """

    @property
    @abstractmethod
    def world_T_robot_start(self) -> HomogeneousTransformationMatrix:
        """
        Where a simulated robot starts on the ground floor, in the world root's frame.
        """

    @abstractmethod
    def spawn(self, world: World) -> Elevator:
        """
        Materialize this building in ``world``, with a level annotation for each floor
        the elevator serves.

        :param world: The world the building is added to.
        :return: The elevator, serving the ground floor.
        """

    def create_world(self, root_name: str = "map") -> World:
        """
        :param root_name: Name of the world's root body, the frame the building is in.
        :return: A world holding nothing but this building.
        """
        world = World.create_with_root_body(root_name)
        self.spawn(world)
        return world
