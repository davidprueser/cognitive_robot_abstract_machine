import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    RectilinearRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey_layout import (
    PlacedRoom,
    StoreyLayout,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    BathroomCabin,
    Floor,
    Hallway,
    Wall,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture
def room_with_doorway() -> PlacedRoom:
    return PlacedRoom(
        name="room",
        layout=RectilinearRoom(
            outline=[
                Point2(0.0, 0.0),
                Point2(4.0, 0.0),
                Point2(4.0, 3.0),
                Point2(0.0, 3.0),
            ],
            doorways=[Doorway(start=Point2(1.0, 0.0), end=Point2(2.0, 0.0))],
        ),
        storey_T_outline=HomogeneousTransformationMatrix(),
    )


@pytest.fixture
def storey_layout(room_with_doorway) -> StoreyLayout:
    return StoreyLayout(rooms=[room_with_doorway], hallway=[])


@pytest.fixture
def storey_world(storey_layout) -> World:
    world = World.create_with_root_body("map")
    storey_layout.spawn(world, "storey")
    return world


# %% colors


def test_every_wall_takes_its_rooms_wall_color(room_with_doorway, storey_world):
    wall_colors = {
        shape.color
        for wall in storey_world.get_semantic_annotations_by_type(Wall)
        for shape in wall.root.collision.shapes + wall.root.visual.shapes
    }
    assert wall_colors == {room_with_doorway.layout.wall_color}


def test_every_floor_takes_the_layouts_floor_color(storey_layout, storey_world):
    floor_colors = {
        shape.color
        for floor in storey_world.get_semantic_annotations_by_type(Floor)
        for shape in floor.root.collision.shapes + floor.root.visual.shapes
    }
    assert floor_colors == {storey_layout.floor_color}


# %% kinds of rooms


@pytest.fixture
def bathroom_with_a_cabin() -> StoreyLayout:
    """
    A bathroom with one toilet cabin entered from it, side by side.
    """
    square = [Point2(0.0, 0.0), Point2(2.0, 0.0), Point2(2.0, 2.0), Point2(0.0, 2.0)]
    return StoreyLayout(
        rooms=[
            PlacedRoom(
                name="bathroom",
                layout=RectilinearRoom(outline=square),
                storey_T_outline=HomogeneousTransformationMatrix(),
                room_type=Bathroom,
                cabin_names=["cabin"],
            ),
            PlacedRoom(
                name="cabin",
                layout=RectilinearRoom(outline=square),
                storey_T_outline=HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0),
                room_type=BathroomCabin,
            ),
        ],
        hallway=[],
    )


def test_a_room_takes_its_room_type(bathroom_with_a_cabin):
    world = World.create_with_root_body("map")

    bathroom_with_a_cabin.spawn(world, "storey")

    assert [
        type(world.get_semantic_annotation_by_name(f"storey_{room.name}"))
        for room in bathroom_with_a_cabin.rooms
    ] == [room.room_type for room in bathroom_with_a_cabin.rooms]


def test_a_bathroom_lists_its_cabins(bathroom_with_a_cabin):
    world = World.create_with_root_body("map")

    bathroom_with_a_cabin.spawn(world, "storey")

    bathroom = world.get_semantic_annotation_by_name("storey_bathroom")
    assert bathroom.cabins == [world.get_semantic_annotation_by_name("storey_cabin")]


def test_the_hallway_is_a_hallway(storey_layout):
    world = World.create_with_root_body("map")

    hallway = storey_layout.spawn(world, "storey")

    assert type(hallway) is Hallway
