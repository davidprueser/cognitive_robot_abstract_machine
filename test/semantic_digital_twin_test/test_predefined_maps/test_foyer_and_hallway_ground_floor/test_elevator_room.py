import numpy as np
import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.elevator_room import (
    LabElevatorRoom,
    lab_corridor_doorway,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    FirstFloor,
    GroundFloor,
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


@pytest.fixture()
def corridor_door_world(lab_elevator_room) -> tuple[World, Room, Door]:
    world = World.create_with_root_body("map")
    room = lab_elevator_room.spawn(world, "lab")
    return world, room, corridor_door(world)


def corridor_door(world: World) -> Door:
    """
    :return: The only door that does not belong to the elevator.
    """
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    (door,) = [
        door
        for door in world.get_semantic_annotations_by_type(Door)
        if door not in elevator.doors
    ]
    return door


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
            if wall.root.parent_kinematic_structure_entity is room.floor.root
            for aperture in wall.apertures
        ),
        key=lambda position: position[0],
    )
    assert left_doorway[1] == pytest.approx(corridor_doorway[1])


# %% corridor door


def test_corridor_door_stands_in_the_corridor_doorway(lab_elevator_room_world):
    world, room = lab_elevator_room_world
    (corridor_aperture,) = [
        aperture
        for wall in world.get_semantic_annotations_by_type(Wall)
        if wall.root.parent_kinematic_structure_entity is room.floor.root
        for aperture in wall.apertures
        if world.compute_forward_kinematics_np(room.floor.root, aperture.root)[0, 3]
        > lab_corridor_doorway().start.to_np()[0]
    ]
    assert world.compute_forward_kinematics_np(
        room.floor.root, corridor_door(world).root
    ) == pytest.approx(
        world.compute_forward_kinematics_np(room.floor.root, corridor_aperture.root)
    )


def test_corridor_door_is_as_large_as_the_corridor_doorway(lab_elevator_room):
    doorway = lab_corridor_doorway()
    door_scale = lab_elevator_room.corridor_door.scale
    assert (door_scale.y, door_scale.z) == pytest.approx(
        (doorway.width, doorway.height)
    )


def test_corridor_door_swings_into_the_room(corridor_door_world):
    world, room, door = corridor_door_world
    door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    door_center = world.compute_forward_kinematics_np(room.floor.root, door.root)
    below_door_center = Point3(
        *door_center[:2, 3], -0.01, reference_frame=room.floor.root
    )
    assert room.floor.root.collision.as_bounding_box_collection_in_frame(
        room.floor.root
    ).contains(below_door_center)


def test_opened_corridor_door_stands_at_the_jamb_nearer_the_elevator(
    lab_elevator_room, corridor_door_world
):
    world, room, door = corridor_door_world
    door.movable_joint.position = door.movable_joint.dof.limits.upper.position
    door_center = world.compute_forward_kinematics_np(room.floor.root, door.root)
    jambs = lab_corridor_doorway().start.to_np(), lab_corridor_doorway().end.to_np()
    nearer_jamb = min(
        jambs,
        key=lambda jamb: np.linalg.norm(
            jamb - lab_elevator_room.elevator_center.to_np()
        ),
    )
    assert door_center[1, 3] == pytest.approx(nearer_jamb[1])


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


# %% first floor


@pytest.fixture()
def lab_world_with_elevator_upstairs(lab_elevator_room) -> tuple[World, Room, Elevator]:
    world = World.create_with_root_body("map")
    room = lab_elevator_room.spawn(world, "lab")
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    (first_floor_level,) = world.get_semantic_annotations_by_type(FirstFloor)
    elevator.drive_to_floor(first_floor_level)
    return world, room, elevator


def first_floor_room(world: World, ground_floor_room: Room) -> Room:
    """
    :return: The room spawned for the first floor, which stands on the ground floor.
    """
    (room,) = [
        room
        for room in world.get_semantic_annotations_by_type(Room)
        if room.floor.root.parent_kinematic_structure_entity
        is ground_floor_room.floor.root
    ]
    return room


def test_first_floor_rests_on_the_ground_floor_walls(
    lab_elevator_room, lab_elevator_room_world
):
    world, room = lab_elevator_room_world
    upstairs = first_floor_room(world, room)
    assert world.compute_forward_kinematics_np(room.floor.root, upstairs.floor.root)[
        2, 3
    ] == pytest.approx(lab_elevator_room.room.wall_height)


def test_ground_and_first_floor_are_levels(lab_elevator_room_world):
    world, _ = lab_elevator_room_world
    assert len(world.get_semantic_annotations_by_type(GroundFloor)) == 1
    assert len(world.get_semantic_annotations_by_type(FirstFloor)) == 1


def test_elevator_driven_upstairs_stands_on_the_first_floor(
    lab_elevator_room, lab_world_with_elevator_upstairs
):
    world, room, elevator = lab_world_with_elevator_upstairs
    cabin_bottom = (
        world.compute_forward_kinematics_np(room.floor.root, elevator.root)[2, 3]
        - lab_elevator_room.elevator.scale.z / 2
    )
    assert cabin_bottom == pytest.approx(lab_elevator_room.room.wall_height)


def test_elevator_driven_upstairs_opens_onto_the_first_floor(
    lab_elevator_room, lab_world_with_elevator_upstairs
):
    world, room, elevator = lab_world_with_elevator_upstairs
    upstairs = first_floor_room(world, room)
    in_front_of_doors = Point3(
        -lab_elevator_room.elevator.scale.x / 2 - 0.2,
        0,
        -lab_elevator_room.elevator.scale.z / 2 - 0.01,
        reference_frame=elevator.root,
    )
    assert upstairs.floor.root.collision.as_bounding_box_collection_in_frame(
        upstairs.floor.root
    ).contains(in_front_of_doors)


def test_first_floor_has_no_wall_across_the_elevator_entrance(lab_elevator_room):
    elevator_center = lab_elevator_room.elevator_center.to_np()
    entrance_middle = Point2(
        elevator_center[0],
        elevator_center[1] - lab_elevator_room.elevator.scale.x / 2,
    )
    first_floor = lab_elevator_room.first_floor
    (entrance,) = [edge for edge in first_floor.edges if edge.contains(entrance_middle)]
    assert first_floor.is_open(entrance)


def test_first_floor_has_no_wall_on_the_railing_side(lab_elevator_room):
    first_floor = lab_elevator_room.first_floor
    lowest_y = min(vertex.to_np()[1] for vertex in first_floor.outline)
    (railing_side,) = [
        edge
        for edge in first_floor.edges
        if edge.start[1] == lowest_y and edge.end[1] == lowest_y
    ]
    assert first_floor.is_open(railing_side)


def test_first_floor_has_a_door_right_above_each_ground_floor_doorway(
    lab_elevator_room_world,
):
    world, room = lab_elevator_room_world
    upstairs = first_floor_room(world, room)

    def aperture_positions(storey: Room) -> list:
        return [
            world.compute_forward_kinematics_np(room.floor.root, aperture.root)[:3, 3]
            for wall in world.get_semantic_annotations_by_type(Wall)
            if wall.root.parent_kinematic_structure_entity is storey.floor.root
            for aperture in wall.apertures
        ]

    first_floor_doors = aperture_positions(upstairs)
    for archway in aperture_positions(room):
        assert any(
            door[:2] == pytest.approx(archway[:2]) and door[2] > archway[2]
            for door in first_floor_doors
        )


def test_no_first_floor_wall_hangs_over_the_ground_floor_room(lab_elevator_room):
    first_floor = lab_elevator_room.first_floor
    ground_floor_rectangles = lab_elevator_room.room.floor_rectangles
    for edge in first_floor.edges:
        if first_floor.is_open(edge):
            continue
        x, y = (
            edge.start + edge.end
        ) / 2 + edge.outward_normal * first_floor.wall_thickness / 2
        assert not any(
            rectangle.min_x < x < rectangle.max_x
            and rectangle.min_y < y < rectangle.max_y
            for rectangle in ground_floor_rectangles
        )


# %% hallway


def is_on_any_floor(world: World, point: Point3) -> bool:
    """
    :return: Whether ``point`` lies inside the floor of any room.
    """
    return any(
        room.floor.root.collision.as_bounding_box_collection_in_frame(
            room.floor.root
        ).contains(point)
        for room in world.get_semantic_annotations_by_type(Room)
    )


def is_inside_any_wall(world: World, point: Point3) -> bool:
    """
    :return: Whether ``point`` lies inside any wall.
    """
    return any(
        wall.root.collision.as_bounding_box_collection_in_frame(wall.root).contains(
            point
        )
        for wall in world.get_semantic_annotations_by_type(Wall)
    )


def test_hallway_continues_beyond_the_corridor_doorway(
    lab_elevator_room, lab_elevator_room_world
):
    world, room = lab_elevator_room_world
    doorway = lab_corridor_doorway()
    beyond_doorway = (doorway.start.to_np() + doorway.end.to_np()) / 2 + np.array(
        [lab_elevator_room.room.wall_thickness + 0.3, 0]
    )
    assert is_on_any_floor(
        world, Point3(*beyond_doorway, -0.01, reference_frame=room.floor.root)
    )
    assert not is_inside_any_wall(
        world, Point3(*beyond_doorway, 1.0, reference_frame=room.floor.root)
    )


def test_hallway_stretches_meet_without_a_wall(
    lab_elevator_room, lab_elevator_room_world
):
    world, _ = lab_elevator_room_world
    hallway = lab_elevator_room.hallway
    south_stretch = hallway.south_stretch
    highest_y = max(vertex.to_np()[1] for vertex in south_stretch.outline)
    (joint,) = [
        edge
        for edge in south_stretch.edges
        if edge.start[1] == highest_y and edge.end[1] == highest_y
    ]
    assert south_stretch.is_open(joint)
    joint_middle = (joint.start + joint.end) / 2
    for offset in (-0.3, 0.3):
        x, y = joint_middle + joint.outward_normal * offset
        map_point = hallway.map_T_south_stretch.to_np() @ np.array([x, y, 0, 1])
        assert is_on_any_floor(
            world, Point3(*map_point[:2], -0.01, reference_frame=world.root)
        )
        assert not is_inside_any_wall(
            world, Point3(*map_point[:2], 1.0, reference_frame=world.root)
        )


def test_hallway_ends_in_a_doorway_as_wide_as_its_end(lab_elevator_room):
    south_stretch = lab_elevator_room.hallway.south_stretch
    lowest_y = min(vertex.to_np()[1] for vertex in south_stretch.outline)
    (end,) = [
        edge
        for edge in south_stretch.edges
        if edge.start[1] == lowest_y and edge.end[1] == lowest_y
    ]
    (doorway,) = [doorway for doorway in south_stretch.doorways if end.holds(doorway)]
    assert doorway.width == pytest.approx(end.length)
