import numpy as np
import pytest

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.elevator_room import (
    LabElevatorRoom,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)

# %% elevator


def test_cabin_floor_is_level_with_the_ground_floor(lab_building_world):
    elevator = lab_building_world.get_semantic_annotations_by_type(Elevator)[0]
    ground_floor = lab_building_world.get_semantic_annotations_by_type(GroundFloor)[0]
    cabin_bottom = float(elevator.root.global_transform.z) - elevator.scale.z / 2
    assert cabin_bottom == pytest.approx(float(ground_floor.floor_plane[0].z), abs=1e-6)


def test_elevator_drives_to_the_first_floor_slab(lab_building_world):
    elevator = lab_building_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = lab_building_world.get_semantic_annotations_by_type(FirstFloor)[0]
    assert elevator.drive_position_for_floor(first_floor) == pytest.approx(
        LabElevatorRoom.first_floor_height
    )


def test_elevator_opens_east_into_the_foyer(lab_building_world, lab_building):
    elevator = lab_building_world.get_semantic_annotations_by_type(Elevator)[0]
    world_V_hole = lab_building_world.transform(
        elevator.hole_direction, lab_building_world.root
    )
    map_R_room = lab_building.elevator_room.map_T_room.to_np()[:3, :3]
    assert world_V_hole.to_np()[:3] == pytest.approx(
        map_R_room @ np.array([0.0, -1.0, 0.0])
    )


def test_elevator_has_two_doors(lab_building_world):
    elevator = lab_building_world.get_semantic_annotations_by_type(Elevator)[0]
    assert len(elevator.doors) == 2


# %% levels


def test_levels_start_at_the_slab_tops(lab_building_world, lab_building):
    ground_floor = lab_building_world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = lab_building_world.get_semantic_annotations_by_type(FirstFloor)[0]
    ground_floor_height = float(lab_building.elevator_room.map_T_room.to_np()[2, 3])
    assert float(ground_floor.floor_plane[0].z) == pytest.approx(
        ground_floor_height, abs=1e-6
    )
    assert float(first_floor.floor_plane[0].z) == pytest.approx(
        ground_floor_height + LabElevatorRoom.first_floor_height, abs=1e-6
    )
