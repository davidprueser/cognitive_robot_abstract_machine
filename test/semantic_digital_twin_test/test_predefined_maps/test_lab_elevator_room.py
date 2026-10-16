import pytest

from semantic_digital_twin.predefined_maps.lab_elevator_room import LabElevatorRoom
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    Room,
    Wall,
)
from semantic_digital_twin.spatial_types import Point2, Point3
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture(scope="module")
def lab_elevator_room() -> LabElevatorRoom:
    return LabElevatorRoom()


@pytest.fixture(scope="module")
def lab_elevator_room_world(lab_elevator_room) -> tuple[World, Room]:
    world = World.create_with_root_body("map")
    room = lab_elevator_room.spawn(world, "lab")
    return world, room


@pytest.fixture(scope="module")
def elevator(lab_elevator_room_world) -> Elevator:
    world, _ = lab_elevator_room_world
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    return elevator


# %% elevator


def test_room_has_one_elevator_with_two_sliding_doors(elevator):
    assert len(elevator.doors) == 2
    assert all(isinstance(door, Door) for door in elevator.doors)


def test_elevator_is_placed_at_its_configured_pose(
    lab_elevator_room, lab_elevator_room_world, elevator
):
    world, room = lab_elevator_room_world
    expected_pose = (
        lab_elevator_room.map_T_room @ lab_elevator_room.room_T_elevator
    ).to_np()
    assert world.compute_forward_kinematics_np(
        world.root, elevator.root
    ) == pytest.approx(expected_pose)


def test_elevator_footprint_lies_on_the_floor_clear_of_walls(
    lab_elevator_room, lab_elevator_room_world, elevator
):
    world, room = lab_elevator_room_world
    margin = 0.01
    half_depth = lab_elevator_room.elevator.scale.x / 2 - margin
    half_width = lab_elevator_room.elevator.scale.y / 2 - margin
    floor_boxes = room.floor.root.collision.as_bounding_box_collection_in_frame(
        room.floor.root
    )
    wall_boxes = [
        wall.root.collision.as_bounding_box_collection_in_frame(room.floor.root)
        for wall in world.get_semantic_annotations_by_type(Wall)
    ]
    for x in (-half_depth, half_depth):
        for y in (-half_width, half_width):
            on_floor = Point3(
                x,
                y,
                -lab_elevator_room.elevator.scale.z / 2 - margin,
                reference_frame=elevator.root,
            )
            beside_walls = Point3(x, y, 0, reference_frame=elevator.root)
            assert floor_boxes.contains(on_floor)
            assert not any(boxes.contains(beside_walls) for boxes in wall_boxes)


def test_elevator_doors_open_into_the_room(
    lab_elevator_room, lab_elevator_room_world, elevator
):
    world, room = lab_elevator_room_world
    in_front_of_doors = Point3(
        -lab_elevator_room.elevator.scale.x / 2 - 0.2,
        0,
        -lab_elevator_room.elevator.scale.z / 2 - 0.01,
        reference_frame=elevator.root,
    )
    assert room.floor.root.collision.as_bounding_box_collection_in_frame(
        room.floor.root
    ).contains(in_front_of_doors)


# %% walls


def test_staircase_beside_the_elevator_has_no_wall(
    lab_elevator_room, lab_elevator_room_world
):
    world, room = lab_elevator_room_world
    room_description = lab_elevator_room.room
    elevator_left_side = (
        lab_elevator_room.elevator_center.to_np()[0]
        - lab_elevator_room.elevator.scale.y / 2
    )
    wall_boxes = [
        wall.root.collision.as_bounding_box_collection_in_frame(room.floor.root)
        for wall in world.get_semantic_annotations_by_type(Wall)
    ]
    assert room_description.open_edges
    for open_edge in room_description.open_edges:
        (edge,) = [
            edge
            for edge in room_description.edges
            if edge.has_vertices(open_edge.start, open_edge.end)
        ]
        outside_middle = (
            edge.start + edge.end
        ) / 2 + edge.outward_normal * room_description.wall_thickness / 2
        assert outside_middle[0] < elevator_left_side
        point = Point3(
            *outside_middle,
            room_description.wall_height / 2,
            reference_frame=room.floor.root,
        )
        assert not any(boxes.contains(point) for boxes in wall_boxes)


def test_left_wall_doorway_faces_the_corridor_doorway(lab_elevator_room_world):
    world, room = lab_elevator_room_world
    left_doorway, corridor_doorway = sorted(
        (
            world.compute_forward_kinematics_np(room.floor.root, aperture.root)[:2, 3]
            for wall in world.get_semantic_annotations_by_type(Wall)
            for aperture in wall.apertures
        ),
        key=lambda position: position[0],
    )
    assert left_doorway[1] == pytest.approx(corridor_doorway[1])


def test_wall_beside_the_elevator_shaft_stands_up_to_the_next_corner(
    lab_elevator_room,
):
    room_description = lab_elevator_room.room
    elevator_center = lab_elevator_room.elevator_center.to_np()
    shaft_left_side = Point2(
        elevator_center[0] - lab_elevator_room.elevator.scale.y / 2,
        elevator_center[1],
    )
    edges = room_description.edges
    (shaft_left_index,) = [
        index for index, edge in enumerate(edges) if edge.contains(shaft_left_side)
    ]
    index = (shaft_left_index + 1) % len(edges)
    while not edges[index].is_convex_at_start:
        assert not room_description.is_open(edges[index])
        index = (index + 1) % len(edges)
