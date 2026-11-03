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
    Floor,
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
