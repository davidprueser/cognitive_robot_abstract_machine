from __future__ import annotations

import numpy as np
import pytest
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    Room,
    Wall,
)
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
        FloorPlan,
        OfficeRoom,
    )
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Level,
    )
    from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


@pytest.fixture(scope="module")
def office() -> IAIOffice:
    return IAIOffice()


@pytest.fixture(scope="module")
def office_world(office) -> tuple[World, list[Level]]:
    world = World.create_with_root_body("map")
    levels = office.spawn(world, "lab")
    return world, levels


@pytest.fixture(scope="module")
def elevator(office_world) -> Elevator:
    world, _ = office_world
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    return elevator


def floor_plan_of(office: IAIOffice, storey: Storey) -> FloorPlan:
    """
    :return: The floor plan of ``storey``.
    """
    (floor_plan,) = [plan for plan in office.floor_plans if plan.storey is storey]
    return floor_plan


# %% points against bodies


def inside_any(world: World, bodies: list[Body], points: np.ndarray) -> np.ndarray:
    """
    :param points: Rows of positions in the world's root frame.
    :return: For each point, whether it lies in a collision box of any of ``bodies``.
    """
    homogeneous = np.c_[points, np.ones(len(points))]
    inside = np.zeros(len(points), dtype=bool)
    for body in bodies:
        body_T_world = np.linalg.inv(
            world.compute_forward_kinematics_np(world.root, body)
        )
        local = (homogeneous @ body_T_world.T)[:, :3]
        for box in body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_boxes:
            minimum = np.array([box.min_x, box.min_y, box.min_z])
            maximum = np.array([box.max_x, box.max_y, box.max_z])
            inside |= np.all((local >= minimum) & (local <= maximum), axis=1)
    return inside


def walls(world: World) -> list[Body]:
    return [wall.root for wall in world.get_semantic_annotations_by_type(Wall)]


def floors(world: World) -> list[Body]:
    return [room.floor.root for room in world.get_semantic_annotations_by_type(Room)]


def map_points(
    office: IAIOffice, office_room: OfficeRoom, points: np.ndarray, height: float
) -> np.ndarray:
    """
    :param points: Rows of positions in the room's outline frame.
    :param height: Height of every point above the ground floor.
    :return: The points in the map frame.
    """
    map_T_frame = office.map_T_outline_frame(office_room.outline_frame).to_np()
    homogeneous = np.c_[points, np.full(len(points), height), np.ones(len(points))]
    return (homogeneous @ map_T_frame.T)[:, :3]


# %% storeys


def test_each_storey_is_one_level(office_world):
    _, levels = office_world
    assert [type(level) for level in levels] == [storey.level_type for storey in Storey]


@pytest.mark.parametrize("storey", list(Storey))
def test_level_starts_at_the_floor_of_its_storey(office_world, storey):
    _, levels = office_world
    floor_plane = levels[storey].floor_plane[0]
    assert float(floor_plane.z) == pytest.approx(storey.elevation, abs=1e-6)


def test_room_numbers_count_the_storeys_below(office):
    for floor_plan in office.floor_plans:
        for office_room in floor_plan.rooms:
            assert office_room.number.split(".")[0] == str(floor_plan.storey.value)


def test_rooms_stand_on_the_floor_of_their_storey(office, office_world):
    world, _ = office_world
    for floor_plan in office.floor_plans:
        rooms = [
            room
            for room in world.get_semantic_annotations_by_type(Room)
            for office_room in floor_plan.rooms
            if str(room.name) == f"lab_room_{office_room.number}"
        ]
        assert len(rooms) == len(floor_plan.rooms)
        for room in rooms:
            assert world.compute_forward_kinematics_np(world.root, room.floor.root)[
                2, 3
            ] == pytest.approx(floor_plan.storey.elevation)


# %% doorways


def test_every_doorway_opens_through_both_walls_onto_a_floor(office, office_world):
    world, _ = office_world
    through_walls, beyond_walls = [], []
    for floor_plan in office.floor_plans:
        elevation = floor_plan.storey.elevation
        for office_room in floor_plan.rooms:
            layout = office_room.layout
            for doorway in layout.doorways:
                edge = layout.edge_of(doorway)
                middle = (doorway.start.to_np() + doorway.end.to_np()) / 2
                offsets = np.arange(0.05, 0.6, 0.05)
                through = middle + np.outer(offsets, edge.outward_normal)
                beyond = middle + 0.6 * edge.outward_normal
                through_walls.append(
                    map_points(office, office_room, through, elevation + 1.0)
                )
                beyond_walls.append(
                    map_points(office, office_room, beyond[None], elevation - 0.01)
                )
    assert not inside_any(world, walls(world), np.concatenate(through_walls)).any()
    assert inside_any(world, floors(world), np.concatenate(beyond_walls)).all()


# %% elevator


@pytest.mark.parametrize("storey", list(Storey))
def test_elevator_driven_to_a_storey_opens_onto_its_floor(
    office, office_world, elevator, storey
):
    world, levels = office_world
    elevator.drive_to_floor(levels[storey])
    world_T_cabin = world.compute_forward_kinematics_np(world.root, elevator.root)
    scale = office.elevator_room.elevator.scale
    in_front = np.array(
        [
            [-scale.x / 2 - offset, 0, -scale.z / 2 + height, 1]
            for offset in (0.05, 0.3, 0.6)
            for height in (-0.01, 1.0)
        ]
    )
    points = (in_front @ world_T_cabin.T)[:, :3]
    on_floor, in_air = points[0::2], points[1::2]
    assert world_T_cabin[2, 3] - scale.z / 2 == pytest.approx(
        storey.elevation, abs=1e-6
    )
    assert inside_any(world, floors(world), on_floor).all()
    assert not inside_any(world, walls(world), in_air).any()


def landing(office: IAIOffice, storey: Storey) -> OfficeRoom:
    """
    :return: The room of ``storey`` the elevator opens onto.
    """
    elevator_center = office.elevator_room.elevator_center.to_np()
    in_front = elevator_center - np.array(
        [0, office.elevator_room.elevator.scale.x / 2 + 0.3]
    )
    room_T_map = office.elevator_room.map_T_room.inverse().to_np()
    (office_room,) = [
        office_room
        for office_room in floor_plan_of(office, storey).rooms
        if inside_outline(
            office_room,
            (
                room_T_map
                @ office.map_T_outline_frame(office_room.outline_frame).to_np()
            ),
            in_front,
        )
    ]
    return office_room


def inside_outline(
    office_room: OfficeRoom, room_T_frame: np.ndarray, point: np.ndarray
) -> bool:
    """
    :param room_T_frame: Placement of the room's outline frame in the elevator room's.
    :param point: A position in the elevator room's outline frame.
    :return: Whether ``point`` lies on the room's floor.
    """
    local = np.linalg.inv(room_T_frame) @ np.array([*point, 0, 1])
    return any(
        rectangle.min_x < local[0] < rectangle.max_x
        and rectangle.min_y < local[1] < rectangle.max_y
        for rectangle in office_room.layout.floor_rectangles
    )


def test_first_floor_landing_looks_down_into_the_elevator_room(office, office_world):
    world, _ = office_world
    first_floor_landing = landing(office, Storey.FIRST_FLOOR)
    layout = first_floor_landing.layout
    lowest_y = min(vertex.to_np()[1] for vertex in layout.outline)
    (railing_side,) = [
        edge
        for edge in layout.edges
        if edge.start[1] == lowest_y and edge.end[1] == lowest_y
    ]
    assert layout.is_open(railing_side)
    below_the_railing = (railing_side.start + railing_side.end) / 2 + np.array(
        [0, -0.5]
    )
    assert not inside_any(
        world,
        floors(world),
        map_points(
            office,
            first_floor_landing,
            below_the_railing[None],
            Storey.FIRST_FLOOR.elevation - 0.01,
        ),
    ).any()


def test_second_floor_landing_is_walled_off_from_the_south_wing(office):
    layout = landing(office, Storey.SECOND_FLOOR).layout
    lowest_x = min(vertex.to_np()[0] for vertex in layout.outline)
    (south_side,) = [
        edge
        for edge in layout.edges
        if edge.start[0] == lowest_x and edge.end[0] == lowest_x
    ]
    assert not layout.is_open(south_side)
    assert not any(south_side.holds(doorway) for doorway in layout.doorways)
