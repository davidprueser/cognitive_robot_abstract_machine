"""
The apartment map tells whether a world already holds its furniture.
"""

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
    ApartmentFurniture,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Mesh
from semantic_digital_twin.world_description.world_entity import Body


# %% whether a world holds the apartment
def test_an_empty_world_does_not_hold_the_apartment() -> None:
    assert not ApartmentEnvironment.is_populated(World.create_with_root_body("root"))


def test_a_world_the_apartment_was_spawned_into_holds_it(
    apartment_environment_world: World,
) -> None:
    assert ApartmentEnvironment.is_populated(apartment_environment_world)


def test_a_world_holding_only_some_furniture_does_not_count() -> None:
    world = World.create_with_root_body("root")
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=Body(name=PrefixedName(ApartmentFurniture.SHELF)),
            )
        )

    assert not ApartmentEnvironment.is_populated(world)


def test_every_furniture_piece_is_spawned_under_its_listed_name(
    apartment_environment_world: World,
) -> None:
    world = apartment_environment_world
    for name in ApartmentFurniture:
        assert world.is_kinematic_structure_entity_in_world_by_name(name), name


# %% the walls' collision geometry


def test_the_walls_collide_as_boxes(apartment_environment_world: World) -> None:
    """
    The walls are one mesh enclosing every room, so they collide as the boxes it is made
    of rather than as that mesh, whose bounding box would cover the whole floor.
    """
    walls = apartment_environment_world.get_body_by_name(ApartmentFurniture.WALLS)

    assert walls.collision.shapes
    assert all(isinstance(shape, Box) for shape in walls.collision.shapes)


def test_the_walls_still_look_like_their_mesh(
    apartment_environment_world: World,
) -> None:
    walls = apartment_environment_world.get_body_by_name(ApartmentFurniture.WALLS)

    assert [type(shape) for shape in walls.visual.shapes] == [Mesh]
