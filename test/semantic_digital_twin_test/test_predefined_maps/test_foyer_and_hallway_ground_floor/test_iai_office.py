import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    StoreyHighRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bathroom,
    BathroomCabin,
    Elevator,
    ElevatorLobby,
    FirstFloor,
    GroundFloor,
    Hallway,
    Level,
    SecondFloor,
)
from semantic_digital_twin.spatial_types import Point2

# %% floor plans


def test_office_has_a_floor_plan_for_every_storey(iai_office):
    floor_plans = iai_office.floor_plans
    assert [floor_plan.storey for floor_plan in floor_plans] == list(Storey)


def test_storey_high_room_walls_reach_the_next_storey():
    room = StoreyHighRoom(
        outline=[Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0), Point2(0.0, 1.0)]
    )
    assert room.wall_height == Storey.height


# %% elevator


def test_elevator_travels_no_higher_than_the_top_storey(iai_office_world):
    (elevator,) = iai_office_world.get_semantic_annotations_by_type(Elevator)
    assert elevator.movable_joint.dof.limits.upper.position == pytest.approx(
        elevator.drive_position_for_floor(elevator.floors[-1])
    )


def test_the_elevator_serves_the_storeys_from_the_lowest_up(iai_office_world):
    (elevator,) = iai_office_world.get_semantic_annotations_by_type(Elevator)

    assert [type(floor) for floor in elevator.floors] == [
        GroundFloor,
        FirstFloor,
        SecondFloor,
    ]


def test_the_simulator_reports_the_drive_under_its_connections_name(
    iai_office, iai_office_world
):
    (elevator,) = iai_office_world.get_semantic_annotations_by_type(Elevator)

    assert iai_office.simulated_drive_joint_name(elevator) == (
        elevator.movable_joint.name.name
    )


# %% kinds of rooms


def test_every_room_takes_the_type_its_floor_plan_shows(
    iai_office, iai_office_name, iai_office_world
):
    mistyped_rooms = [
        office_room.number
        for floor_plan in iai_office.floor_plans
        for office_room in floor_plan.rooms
        if type(
            iai_office.get_room(
                iai_office_world,
                iai_office_name,
                floor_plan.storey,
                office_room.number,
            )
        )
        is not office_room.room_type
    ]

    assert mistyped_rooms == []


def test_every_bathroom_lists_the_cabins_its_floor_plan_shows(
    iai_office, iai_office_name, iai_office_world
):
    for floor_plan in iai_office.floor_plans:
        for office_room in floor_plan.rooms:
            if not office_room.cabin_numbers:
                continue
            bathroom = iai_office.get_room(
                iai_office_world,
                iai_office_name,
                floor_plan.storey,
                office_room.number,
            )
            assert bathroom.cabins == [
                iai_office.get_room(
                    iai_office_world, iai_office_name, floor_plan.storey, cabin_number
                )
                for cabin_number in office_room.cabin_numbers
            ]


def test_every_toilet_cabin_belongs_to_one_bathroom(iai_office_world):
    cabins_of_bathrooms = [
        cabin
        for bathroom in iai_office_world.get_semantic_annotations_by_type(Bathroom)
        for cabin in bathroom.cabins
    ]

    assert sorted(cabin.name.name for cabin in cabins_of_bathrooms) == sorted(
        cabin.name.name
        for cabin in iai_office_world.get_semantic_annotations_by_type(BathroomCabin)
    )


def test_only_bathrooms_list_cabins(iai_office):
    rooms_listing_cabins = [
        office_room.number
        for floor_plan in iai_office.floor_plans
        for office_room in floor_plan.rooms
        if office_room.cabin_numbers and office_room.room_type is not Bathroom
    ]

    assert rooms_listing_cabins == []


def test_every_storey_has_one_hallway(iai_office_world):
    assert len(iai_office_world.get_semantic_annotations_by_type(Hallway)) == len(
        iai_office_world.get_semantic_annotations_by_type(Level)
    )


def test_the_elevator_room_is_the_only_elevator_lobby(
    iai_office, iai_office_name, iai_office_world
):
    assert [
        lobby.name.name
        for lobby in iai_office_world.get_semantic_annotations_by_type(ElevatorLobby)
    ] == [
        f"{iai_office.storey_name(iai_office_name, Storey.GROUND_FLOOR)}_elevator_room"
    ]


# %% rooms a robot may be sent into


def test_the_navigation_targets_are_the_rooms_the_floor_plans_mark(
    iai_office, iai_office_name, iai_office_world
):
    assert iai_office.navigation_targets(iai_office_world, iai_office_name) == [
        iai_office.get_room(
            iai_office_world, iai_office_name, floor_plan.storey, office_room.number
        )
        for floor_plan in iai_office.floor_plans
        for office_room in floor_plan.rooms
        if office_room.navigation_target
    ]
