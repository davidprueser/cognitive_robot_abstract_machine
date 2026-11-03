import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.iai_office import (
    IAIOffice,
)
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
    Hallway,
)
from semantic_digital_twin.spatial_types import Point2
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture(scope="module")
def iai_office_world() -> tuple[World, list]:
    world = World.create_with_root_body("map")
    levels = IAIOffice().spawn(world, "office")
    return world, levels


# %% floor plans


def test_office_has_a_floor_plan_for_every_storey():
    floor_plans = IAIOffice().floor_plans
    assert [floor_plan.storey for floor_plan in floor_plans] == list(Storey)


def test_storey_high_room_walls_reach_the_next_storey():
    room = StoreyHighRoom(
        outline=[Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(1.0, 1.0), Point2(0.0, 1.0)]
    )
    assert room.wall_height == Storey.height


# %% elevator


def test_elevator_travels_no_higher_than_the_top_storey(iai_office_world):
    world, levels = iai_office_world
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    assert elevator.movable_joint.dof.limits.upper.position == pytest.approx(
        elevator.drive_position_for_floor(levels[-1])
    )


# %% kinds of rooms


def test_every_room_takes_the_type_its_floor_plan_shows(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()

    mistyped_rooms = [
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if type(
            world.get_semantic_annotation_by_name(
                f"office_{floor_plan.storey.name.lower()}_room_{office_room.number}"
            )
        )
        is not office_room.room_type
    ]

    assert mistyped_rooms == []


def test_every_bathroom_lists_the_cabins_its_floor_plan_shows(iai_office_world):
    world, _ = iai_office_world
    office = IAIOffice()

    for floor_plan in office.floor_plans:
        storey_name = f"office_{floor_plan.storey.name.lower()}"
        for office_room in floor_plan.rooms:
            if not office_room.cabin_numbers:
                continue
            bathroom = world.get_semantic_annotation_by_name(
                f"{storey_name}_room_{office_room.number}"
            )
            assert [cabin.name.name for cabin in bathroom.cabins] == [
                f"{storey_name}_room_{cabin_number}"
                for cabin_number in office_room.cabin_numbers
            ]


def test_every_toilet_cabin_belongs_to_one_bathroom(iai_office_world):
    world, _ = iai_office_world
    cabins_of_bathrooms = [
        cabin
        for bathroom in world.get_semantic_annotations_by_type(Bathroom)
        for cabin in bathroom.cabins
    ]

    assert sorted(cabin.name.name for cabin in cabins_of_bathrooms) == sorted(
        cabin.name.name
        for cabin in world.get_semantic_annotations_by_type(BathroomCabin)
    )


def test_only_bathrooms_list_cabins():
    office = IAIOffice()

    rooms_listing_cabins = [
        office_room.number
        for floor_plan in office.floor_plans
        for office_room in floor_plan.rooms
        if office_room.cabin_numbers and office_room.room_type is not Bathroom
    ]

    assert rooms_listing_cabins == []


def test_every_storey_has_one_hallway(iai_office_world):
    world, levels = iai_office_world

    assert len(world.get_semantic_annotations_by_type(Hallway)) == len(levels)


def test_the_elevator_room_is_the_only_elevator_lobby(iai_office_world):
    world, _ = iai_office_world

    assert [
        lobby.name.name
        for lobby in world.get_semantic_annotations_by_type(ElevatorLobby)
    ] == ["office_ground_floor_elevator_room"]
