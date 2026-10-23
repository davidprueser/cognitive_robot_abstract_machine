import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.exceptions import (
    DoorwayOffOutlineError,
    DoorwayOnOpenEdgeError,
    NonRectilinearOutlineError,
    OpenEdgeOffOutlineError,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    OpenEdge,
    RectilinearRoom,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Room,
    Wall,
)
from semantic_digital_twin.spatial_types import Point2, Point3
from semantic_digital_twin.world import World

# %% helpers


def l_shaped_outline() -> list[Point2]:
    """
    :return: A counter-clockwise L-shaped outline: a 3 x 1 strip with a 1 x 1 block on
        top of its left end.
    """
    return [
        Point2(0, 0),
        Point2(3, 0),
        Point2(3, 1),
        Point2(1, 1),
        Point2(1, 2),
        Point2(0, 2),
    ]


def is_inside_any_wall(world: World, point: Point3) -> bool:
    """
    :return: Whether ``point`` lies inside the collision geometry of any wall.
    """
    return any(
        wall.root.collision.as_bounding_box_collection_in_frame(world.root).contains(
            point
        )
        for wall in world.get_semantic_annotations_by_type(Wall)
    )


def is_on_floor(world: World, room: Room, point: Point3) -> bool:
    """
    :return: Whether ``point`` lies inside the collision geometry of the room's floor.
    """
    return room.floor.root.collision.as_bounding_box_collection_in_frame(
        world.root
    ).contains(point)


@pytest.fixture()
def l_shaped_room() -> RectilinearRoom:
    return RectilinearRoom(outline=l_shaped_outline())


@pytest.fixture()
def l_shaped_room_world(l_shaped_room) -> tuple[World, Room]:
    world = World.create_with_root_body("root")
    room = l_shaped_room.spawn(world, "l_room")
    return world, room


# %% walls


def test_spawn_creates_one_wall_per_outline_edge(l_shaped_room, l_shaped_room_world):
    world, _ = l_shaped_room_world
    assert len(world.get_semantic_annotations_by_type(Wall)) == len(
        l_shaped_room.outline
    )


def test_walls_stand_outside_the_outline(l_shaped_room, l_shaped_room_world):
    world, _ = l_shaped_room_world
    half_thickness = l_shaped_room.wall_thickness / 2
    height = l_shaped_room.wall_height / 2
    root = world.root
    assert not is_inside_any_wall(world, Point3(0.5, 0.5, height, reference_frame=root))
    assert is_inside_any_wall(
        world, Point3(0.5, -half_thickness, height, reference_frame=root)
    )
    assert is_inside_any_wall(
        world, Point3(2.0, 1 + half_thickness, height, reference_frame=root)
    )


def test_convex_corner_is_closed(l_shaped_room, l_shaped_room_world):
    world, _ = l_shaped_room_world
    half_thickness = l_shaped_room.wall_thickness / 2
    assert is_inside_any_wall(
        world,
        Point3(
            3 + half_thickness,
            -half_thickness,
            l_shaped_room.wall_height / 2,
            reference_frame=world.root,
        ),
    )


def test_reflex_corner_keeps_room_free(l_shaped_room, l_shaped_room_world):
    world, _ = l_shaped_room_world
    half_thickness = l_shaped_room.wall_thickness / 2
    assert not is_inside_any_wall(
        world,
        Point3(
            1 - half_thickness,
            1 - half_thickness,
            l_shaped_room.wall_height / 2,
            reference_frame=world.root,
        ),
    )


def test_clockwise_outline_puts_walls_outside():
    world = World.create_with_root_body("root")
    room = RectilinearRoom(outline=list(reversed(l_shaped_outline())))
    room.spawn(world, "clockwise_room")
    height = room.wall_height / 2
    assert not is_inside_any_wall(
        world, Point3(0.5, 0.5, height, reference_frame=world.root)
    )
    assert is_inside_any_wall(
        world,
        Point3(0.5, -room.wall_thickness / 2, height, reference_frame=world.root),
    )


def test_non_rectilinear_outline_is_rejected():
    with pytest.raises(NonRectilinearOutlineError):
        RectilinearRoom(outline=[Point2(0, 0), Point2(2, 0), Point2(1, 1)])


# %% floor


def test_floor_covers_the_outline_only(l_shaped_room_world):
    world, room = l_shaped_room_world
    root = world.root
    assert is_on_floor(world, room, Point3(2.5, 0.5, -0.01, reference_frame=root))
    assert is_on_floor(world, room, Point3(0.5, 1.5, -0.01, reference_frame=root))
    assert not is_on_floor(world, room, Point3(2.0, 1.5, -0.01, reference_frame=root))


def test_floor_rectangles_cover_the_outline_area(l_shaped_room):
    assert sum(
        rectangle.area for rectangle in l_shaped_room.floor_rectangles
    ) == pytest.approx(4.0)


def test_footprint_is_the_rectangle_around_the_outline(l_shaped_room):
    footprint = l_shaped_room.footprint
    assert (
        footprint.min_x,
        footprint.min_y,
        footprint.max_x,
        footprint.max_y,
    ) == (0, 0, 3, 2)


def test_spawn_registers_the_room_with_its_floor(l_shaped_room_world):
    world, room = l_shaped_room_world
    assert world.get_semantic_annotations_by_type(Room) == [room]


# %% doorways


@pytest.fixture()
def room_with_doorway() -> RectilinearRoom:
    return RectilinearRoom(
        outline=l_shaped_outline(),
        doorways=[Doorway(start=Point2(1.5, 0), end=Point2(2.5, 0), height=2.0)],
    )


def test_doorway_cuts_an_opening_below_its_height(room_with_doorway):
    world = World.create_with_root_body("root")
    room_with_doorway.spawn(world, "room")
    doorway = room_with_doorway.doorways[0]
    wall_middle = -room_with_doorway.wall_thickness / 2
    assert not is_inside_any_wall(
        world, Point3(2.0, wall_middle, doorway.height / 2, reference_frame=world.root)
    )
    assert is_inside_any_wall(
        world,
        Point3(
            2.0,
            wall_middle,
            (doorway.height + room_with_doorway.wall_height) / 2,
            reference_frame=world.root,
        ),
    )


def test_doorway_is_an_aperture_of_its_wall(room_with_doorway):
    world = World.create_with_root_body("root")
    room_with_doorway.spawn(world, "room")
    walls_with_apertures = [
        wall for wall in world.get_semantic_annotations_by_type(Wall) if wall.apertures
    ]
    assert len(walls_with_apertures) == 1
    assert walls_with_apertures[0].apertures == world.get_semantic_annotations_by_type(
        Aperture
    )


def test_doorway_off_the_outline_is_rejected():
    with pytest.raises(DoorwayOffOutlineError):
        RectilinearRoom(
            outline=l_shaped_outline(),
            doorways=[Doorway(start=Point2(1.5, 0.5), end=Point2(2.5, 0.5))],
        )


# %% open edges


@pytest.mark.parametrize(
    "open_edge",
    [
        OpenEdge(start=Point2(1, 2), end=Point2(0, 2)),
        OpenEdge(start=Point2(0, 2), end=Point2(1, 2)),
    ],
    ids=["along_the_winding", "against_the_winding"],
)
def test_open_edge_gets_no_wall(open_edge):
    world = World.create_with_root_body("root")
    room = RectilinearRoom(outline=l_shaped_outline(), open_edges=[open_edge])
    room.spawn(world, "room")
    assert len(world.get_semantic_annotations_by_type(Wall)) == len(room.outline) - 1
    assert not is_inside_any_wall(
        world,
        Point3(
            0.5,
            2 + room.wall_thickness / 2,
            room.wall_height / 2,
            reference_frame=world.root,
        ),
    )


def test_open_edge_that_is_not_a_whole_outline_edge_is_rejected():
    with pytest.raises(OpenEdgeOffOutlineError):
        RectilinearRoom(
            outline=l_shaped_outline(),
            open_edges=[OpenEdge(start=Point2(0, 0), end=Point2(1, 0))],
        )


def test_doorway_on_an_open_edge_is_rejected():
    with pytest.raises(DoorwayOnOpenEdgeError):
        RectilinearRoom(
            outline=l_shaped_outline(),
            doorways=[Doorway(start=Point2(0.25, 2), end=Point2(0.75, 2))],
            open_edges=[OpenEdge(start=Point2(1, 2), end=Point2(0, 2))],
        )


# %% colors


def test_walls_keep_their_color_after_a_doorway_is_cut(room_with_doorway):
    world = World.create_with_root_body("root")
    room_with_doorway.spawn(world, "room")
    wall_colors = {
        shape.color
        for wall in world.get_semantic_annotations_by_type(Wall)
        for shape in wall.root.collision.shapes + wall.root.visual.shapes
    }
    assert wall_colors == {room_with_doorway.wall_color}


def test_floor_takes_its_color(l_shaped_room, l_shaped_room_world):
    _, room = l_shaped_room_world
    floor_colors = {
        shape.color
        for shape in room.floor.root.collision.shapes + room.floor.root.visual.shapes
    }
    assert floor_colors == {l_shaped_room.floor_color}
