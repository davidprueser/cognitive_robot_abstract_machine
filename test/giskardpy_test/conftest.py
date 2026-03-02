from copy import deepcopy

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Cylinder, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


def robot_factory(fucking_huge_link_length: float, vel_limit: float) -> World:
    fucking_huge_cylinder = ShapeCollection(
        shapes=[
            Cylinder(
                width=fucking_huge_link_length / 10,
                height=fucking_huge_link_length,
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-(fucking_huge_link_length / 2)
                ),
            )
        ]
    )
    fucking_huge_sphere = ShapeCollection(
        shapes=[
            Sphere(
                radius=fucking_huge_link_length / 18,
            )
        ]
    )
    dof_limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(None, -vel_limit, None, None),
        upper=DerivativeMap(None, vel_limit, None, None),
    )
    world = World()
    with world.modify_world():
        # %% joint1
        root = Body(name=PrefixedName("map"))
        link1 = Body(
            name=PrefixedName("link1"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=root,
                child=link1,
                axis=Vector3.Z(),
                dof_limits=dof_limits,
            )
        )

        # %% joint2
        link2 = Body(
            name=PrefixedName("link2"),
            collision=deepcopy(fucking_huge_cylinder),
            visual=deepcopy(fucking_huge_cylinder),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link1,
                child=link2,
                axis=Vector3.X(),
                dof_limits=dof_limits,
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=fucking_huge_link_length,
                ),
            )
        )

        # %% joint3
        link3 = Body(
            name=PrefixedName("link3"),
            collision=deepcopy(fucking_huge_cylinder),
            visual=deepcopy(fucking_huge_cylinder),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link2,
                child=link3,
                axis=Vector3.X(),
                dof_limits=dof_limits,
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=fucking_huge_link_length,
                ),
            )
        )

        # %% joint4
        link4 = Body(
            name=PrefixedName("link4"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link3,
                child=link4,
                dof_limits=dof_limits,
                axis=Vector3.Z(),
            )
        )

        # %% joint5
        link5 = Body(
            name=PrefixedName("link5"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link4,
                child=link5,
                axis=Vector3.X(),
                dof_limits=dof_limits,
            )
        )

        # %% joint6
        eef = Body(
            name=PrefixedName("eef"),
            collision=deepcopy(fucking_huge_sphere),
            visual=deepcopy(fucking_huge_sphere),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=link5,
                child=eef,
                axis=Vector3.Y(),
                dof_limits=dof_limits,
            )
        )
    return world
