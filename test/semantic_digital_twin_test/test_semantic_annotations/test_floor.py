import numpy as np
import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Room,
    Wall,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% building a floor


def slab(
    min_x: float, min_y: float, max_x: float, max_y: float, thickness: float = 0.05
) -> Box:
    """
    :return: A floor slab with the given footprint, whose top lies at height zero.
    """
    return Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            (min_x + max_x) / 2, (min_y + max_y) / 2, -thickness / 2
        ),
        scale=Scale(max_x - min_x, max_y - min_y, thickness),
    )


def spawn_floor(
    world: World, name: str, slabs: list[Box], elevation: float = 0.0
) -> Floor:
    """
    :return: A floor made of ``slabs``, whose frame lies at ``elevation`` above the
        world's root.
    """
    return Floor.get_annotation_specification(
        name, BodySpecification(name=name, shapes=ShapeCollection(slabs))
    ).spawn(
        world,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=elevation),
    )


def spawn_wall_on(floor: Floor, name: str, height: float) -> Wall:
    """
    :return: A wall of the given height attached to ``floor``, beyond its slabs.
    """
    return Wall.get_annotation_specification(
        name,
        Wall.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(0.1, 1.0, height)
        ),
    ).spawn(
        floor._world,
        parent=floor.root,
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=100.0, z=height / 2
        ),
    )


def body_at(world: World, x: float, y: float, z: float = 0.0) -> Body:
    """
    :return: A body without geometry whose origin lies at the given position.
    """
    body = Body(name=PrefixedName(f"body_at_{x}_{y}_{z}"))
    with world.modify_world():
        world.add_body(body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x, y, z
                ),
            )
        )
    return body


@pytest.fixture
def world() -> World:
    return World.create_with_root_body("map")


@pytest.fixture
def storey_height() -> float:
    return 3.0


# %% what belongs to a floor


def test_extent_leaves_out_what_is_attached_to_the_floor(world, storey_height):
    floor = spawn_floor(world, "floor", [slab(0, 0, 4, 4)], elevation=storey_height)
    spawn_wall_on(floor, "wall", storey_height)

    extent = floor.extent

    assert extent.max_z == pytest.approx(storey_height)
    assert extent.max_x == pytest.approx(4)


def test_floor_centre_lies_on_the_floor_a_wall_is_attached_to(world, storey_height):
    floor = spawn_floor(world, "floor", [slab(0, 0, 4, 4)])
    spawn_wall_on(floor, "wall", storey_height)
    room = Room(name=PrefixedName("room"), floor=floor)
    with world.modify_world():
        world.add_semantic_annotation(room)

    assert room.floor_centre.position.to_np()[:3] == pytest.approx([2, 2, 0])


# %% the floor under an entity


def test_under_finds_a_floor_carrying_walls(world, storey_height):
    floor = spawn_floor(world, "floor", [slab(0, 0, 4, 4)])
    spawn_wall_on(floor, "wall", storey_height)

    assert Floor.under(body_at(world, 2, 2)) is floor


def test_under_does_not_mistake_the_walls_below_for_the_floor(world, storey_height):
    ground = spawn_floor(world, "ground", [slab(0, 0, 4, 4)])
    spawn_wall_on(ground, "ground_wall", storey_height)
    upper = spawn_floor(world, "upper", [slab(0, 0, 4, 4)], elevation=storey_height)
    spawn_wall_on(upper, "upper_wall", storey_height)

    assert Floor.under(body_at(world, 2, 2, storey_height)) is upper


def test_under_tells_a_floor_from_the_bounding_box_of_its_neighbour(world):
    """
    The L-shaped floor spawns first, so a test of the bounding boxes would find it at
    the same height as the floor standing in its inner corner.
    """
    spawn_floor(world, "l_shaped", [slab(0, 0, 4, 1), slab(0, 1, 1, 4)])
    inner_corner = spawn_floor(world, "inner_corner", [slab(1, 1, 4, 4)])

    assert Floor.under(body_at(world, 3, 3)) is inner_corner


@pytest.fixture
def turned_floor(world) -> Floor:
    """
    A floor of one long slab turned by 45 degrees about its middle at the origin.
    """
    turned_slab = Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(yaw=np.pi / 4, z=-0.025),
        scale=Scale(10.0, 1.0, 0.05),
    )
    return spawn_floor(world, "turned", [turned_slab])


def test_under_finds_a_floor_along_its_turned_slab(world, turned_floor):
    assert Floor.under(body_at(world, 3, 3)) is turned_floor


def test_under_leaves_out_the_corners_of_a_turned_slabs_bounding_box(
    world, turned_floor
):
    assert Floor.under(body_at(world, 3, -3)) is None
