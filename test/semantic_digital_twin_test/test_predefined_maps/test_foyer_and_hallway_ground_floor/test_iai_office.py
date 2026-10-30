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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Elevator
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
