"""
The apartment a robot perceives in: its furniture, the objects on its shelf and where
the robot stands.

The world of the robot describes the apartment as it believes it to be; the scene a
simulated camera renders shows it as it really is. The two differ in the objects on the
shelf: the world misplaces the ones it knows, and the others only reality holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cereal,
    Milk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% shelf items


@dataclass
class ShelfItem:
    """
    A box-shaped object standing on a layer of the apartment's shelf.
    """

    name: str
    """
    The name of the body of the object.
    """

    scale: Scale
    """
    The size of the object.
    """

    color: Color
    """
    The color the object is rendered with.
    """

    x: float
    """
    Where the center of the object stands along the world x axis.
    """

    y: float
    """
    Where the center of the object stands along the world y axis.
    """

    semantic_class: type[IsPerceivable] | None = None
    """
    The class the world of the robot describes the object as; None for an object only
    reality holds.
    """

    def body(self) -> Body:
        """
        :return: A new body of the object, shown and colliding as the box it is.
        """
        shape = Box(scale=self.scale, color=self.color)
        visual = ShapeCollection(shapes=[shape])
        return Body(
            name=PrefixedName(self.name),
            visual=visual,
            collision=visual.copy_without_reference_frame(),
        )


# %% apartment


@dataclass
class ApartmentScene:
    """
    The apartment with objects on the second layer of its shelf and a robot standing in
    front of the shelf, facing it.
    """

    items: list[ShelfItem] = field(
        default_factory=lambda: [
            ShelfItem(
                name="milk",
                scale=Scale(0.07, 0.07, 0.20),
                color=Color(0.95, 0.95, 0.95, 1.0),
                x=0.65,
                y=-0.15,
                semantic_class=Milk,
            ),
            ShelfItem(
                name="cereal",
                scale=Scale(0.19, 0.07, 0.26),
                color=Color(0.85, 0.65, 0.15, 1.0),
                x=0.92,
                y=-0.15,
                semantic_class=Cereal,
            ),
            ShelfItem(
                name="unknown_box",
                scale=Scale(0.08, 0.08, 0.10),
                color=Color(0.20, 0.45, 0.85, 1.0),
                x=1.15,
                y=-0.12,
            ),
        ]
    )
    """
    The objects on the shelf.
    """

    shelf_layer_name: str = "shelf_layer2"
    """
    The name of the body of the shelf layer the objects stand on.
    """

    description_error: float = 0.05
    """
    How far, along the world x axis, the world of the robot misplaces each object it
    knows.
    """

    robot_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.88, y=1.0, yaw=-np.pi / 2
        )
    )
    """
    Where the odometry frame of the robot is in the apartment, which puts a robot that
    has not moved from it in front of the shelf, facing it.
    """

    def build_reality(self, world: World) -> None:
        """
        Build the apartment in ``world`` as it really is, with every object at its
        place.

        :param world: The world to build the apartment in, holding nothing but its root.
        """
        ApartmentEnvironment().populate(world)
        for item in self.items:
            body = item.body()
            with world.modify_world():
                world.add_connection(
                    FixedConnection.create_with_dofs(
                        world=world,
                        parent=world.root,
                        child=body,
                        parent_T_connection_expression=self._pose_on_shelf(
                            world, item, offset=0.0
                        ),
                    )
                )

    def describe(self, world: World, odometry_frame: str) -> None:
        """
        Describe the apartment in ``world`` as the robot believes it to be: with the
        robot localized at :attr:`robot_pose` and the objects it knows, each misplaced
        by :attr:`description_error`.

        :param world: The world of the robot.
        :param odometry_frame: The name of the odometry frame of the robot, which hangs
            below the root of ``world`` on the connection localization sets.
        """
        world.get_body_by_name(odometry_frame).parent_connection.origin = (
            HomogeneousTransformationMatrix(
                data=self.robot_pose.to_np(), reference_frame=world.root
            )
        )
        ApartmentEnvironment().populate(world)
        for item in self.items:
            if item.semantic_class is None:
                continue
            body = item.body()
            with world.modify_world():
                world.add_connection(
                    Connection6DoF.create_with_dofs(
                        parent=world.root, child=body, world=world
                    )
                )
                world.add_semantic_annotation(item.semantic_class(root=body))
            body.parent_connection.origin = self._pose_on_shelf(
                world, item, offset=self.description_error
            )

    def _pose_on_shelf(
        self, world: World, item: ShelfItem, offset: float
    ) -> HomogeneousTransformationMatrix:
        """
        :param world: A world holding the apartment.
        :param item: The object.
        :param offset: How far to shift the object along the world x axis.
        :return: The pose of ``item`` standing on the shelf layer, shifted by
            ``offset``.
        """
        layer = world.get_body_by_name(self.shelf_layer_name)
        layer_top = (
            layer.collision.as_bounding_box_collection_in_frame(world.root)
            .bounding_box()
            .max_z
        )
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            x=item.x + offset,
            y=item.y,
            z=layer_top + item.scale.z / 2,
            reference_frame=world.root,
        )
