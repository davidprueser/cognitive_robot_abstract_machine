import pytest

from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Human,
    Level,
    Room,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Vector3,
)

# %% elevator
from semantic_digital_twin.world_description.graph_of_convex_sets.exceptions import (
    PointOutsideSearchSpaceError,
)


def test_cabin_floor_is_level_with_the_ground_floor(arise_office_world, arise_office):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    cabin_bottom = float(elevator.root.global_transform.z) - elevator.scale.z / 2
    assert cabin_bottom == pytest.approx(arise_office.ground_floor_height)


def test_elevator_drives_to_the_first_floor_slab(arise_office_world, arise_office):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert elevator.drive_position_for_floor(first_floor) == pytest.approx(
        arise_office.first_floor_height
    )


def test_elevator_opens_towards_negative_y(arise_office_world):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    world_V_hole = arise_office_world.transform(
        elevator.hole_direction, arise_office_world.root
    )
    assert world_V_hole.to_np()[:3] == pytest.approx(Vector3(0, -1, 0).to_np()[:3])


def test_elevator_has_two_doors(arise_office_world):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    assert len(elevator.doors) == 2


# %% levels


def test_levels_start_at_the_slab_tops(arise_office_world, arise_office):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert float(ground_floor.floor_plane[0].z) == pytest.approx(
        arise_office.ground_floor_height
    )
    assert float(first_floor.floor_plane[0].z) == pytest.approx(
        arise_office.first_floor_height
    )


def test_level_under_a_person_is_the_storey_they_stand_on(
    arise_office_world, arise_office
):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    downstairs, upstairs = (
        Human.create_with_new_body_in_world(
            name=f"person_at_{height}",
            world=arise_office_world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=height),
        )
        for height in (
            arise_office.ground_floor_height,
            arise_office.first_floor_height,
        )
    )
    assert Level.under(downstairs.root) is ground_floor
    assert Level.under(upstairs.root) is first_floor


def test_level_under_an_entity_below_every_storey_is_unknown(arise_office_world):
    basement = Human.create_with_new_body_in_world(
        name="person_in_the_basement",
        world=arise_office_world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-3.0),
    )
    assert Level.under(basement.root) is None


def test_levels_are_spoken_of_by_the_storey_they_stand_for(arise_office_world):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert ground_floor.spoken_name == "ground floor"
    assert first_floor.spoken_name == "first floor"


def test_doorway_lies_in_the_middle_of_the_doors_on_the_cabin_floor(
    arise_office_world, arise_office
):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    world_T_doorway = arise_office_world.transform(
        elevator.doorway_pose, arise_office_world.root
    ).to_np()
    assert world_T_doorway[:3, 3] == pytest.approx(
        [
            float(arise_office.cabin_centre.x),
            float(arise_office.cabin_centre.y) - arise_office.cabin_scale.x / 2,
            arise_office.ground_floor_height,
        ]
    )
    # The office's doors face -y, so the doorway's x-axis points along +y into the cabin
    assert world_T_doorway[:3, 0] == pytest.approx([0.0, 1.0, 0.0])


def test_level_under_a_person_riding_the_elevator_is_the_storey_it_arrived_at(
    arise_office_world, arise_office
):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    rider = Human.create_with_new_body_in_world(
        name="rider",
        world=arise_office_world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            arise_office.cabin_centre.x,
            arise_office.cabin_centre.y,
            arise_office.ground_floor_height,
        ),
    )
    with arise_office_world.modify_world():
        arise_office_world.move_branch(rider.root, elevator.root)
    elevator.drive_to_floor(first_floor)
    assert Level.under(rider.root) is first_floor


# %% rooms


def rooms_on(level: Level) -> list[Room]:
    """
    :return: The rooms whose floor lies on ``level``.
    """
    return [
        room
        for room in level._world.get_semantic_annotations_by_type(Room)
        if Level.under(room.floor.root) is level
    ]


@pytest.mark.parametrize(
    "level_type, middle_room",
    [(GroundFloor, AriseOfficeRoom.LOBBY), (FirstFloor, AriseOfficeRoom.LANDING)],
)
def test_each_storey_has_an_office_on_either_side_and_a_room_between_them(
    arise_office_world, arise_office, level_type, middle_room
):
    level = arise_office_world.get_semantic_annotations_by_type(level_type)[0]
    assert set(rooms_on(level)) == {
        arise_office.get_room(arise_office_world, level_type, room)
        for room in (
            AriseOfficeRoom.WEST_OFFICE,
            middle_room,
            AriseOfficeRoom.EAST_OFFICE,
        )
    }


@pytest.mark.parametrize("level_type", [GroundFloor, FirstFloor])
def test_room_floors_lie_at_the_height_of_their_storey(arise_office_world, level_type):
    level = arise_office_world.get_semantic_annotations_by_type(level_type)[0]
    for room in rooms_on(level):
        assert float(room.floor.root.global_transform.z) == pytest.approx(
            float(level.floor_plane[0].z)
        )


def test_ground_floor_rooms_cover_the_whole_storey(arise_office_world, arise_office):
    ground_floor = arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    covered_area = sum(
        room.floor.extent.depth * room.floor.extent.width
        for room in rooms_on(ground_floor)
    )
    assert covered_area == pytest.approx(
        arise_office.storey_scale.x * arise_office.storey_scale.y
    )


def test_first_floor_landing_ends_at_the_shaft(arise_office_world, arise_office):
    landing = arise_office.get_room(
        arise_office_world, FirstFloor, AriseOfficeRoom.LANDING
    )
    assert landing.floor.extent.max_y == pytest.approx(arise_office.shaft_front)


def test_ground_floor_lobby_is_the_floor_above_the_world_origin(
    arise_office_world, arise_office
):
    lobby = arise_office.get_room(
        arise_office_world, GroundFloor, AriseOfficeRoom.LOBBY
    )
    assert Floor.under(arise_office_world.root) is lobby.floor


def test_first_floor_free_space_leads_from_the_landing_into_the_east_office(
    arise_office_world, arise_office
):
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    landing, east_office = (
        arise_office.get_room(arise_office_world, FirstFloor, room)
        for room in (AriseOfficeRoom.LANDING, AriseOfficeRoom.EAST_OFFICE)
    )
    start = Point2.from_pose(landing.floor_centre)
    goal = Point2.from_pose(east_office.floor_centre)

    path = first_floor.planar_free_space(max_height=1.0).path_from_to(start, goal)

    assert [float(path[-1].x), float(path[-1].y)] == pytest.approx(
        [float(goal.x), float(goal.y)]
    )


def test_first_floor_free_space_leaves_out_the_elevator_shaft(
    arise_office_world, arise_office
):
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    landing = arise_office.get_room(
        arise_office_world, FirstFloor, AriseOfficeRoom.LANDING
    )
    start = Point2.from_pose(landing.floor_centre)
    shaft = Point2(
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        reference_frame=arise_office_world.root,
    )
    free_space = first_floor.planar_free_space(max_height=1.0)

    with pytest.raises(PointOutsideSearchSpaceError):
        free_space.path_from_to(start, shaft)
