import numpy as np
import pytest

from semantic_digital_twin.reasoning.elevator_areas import (
    ElevatorCabinArea,
    ElevatorWaitingArea,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.spatial_types import Point3

# %% fixtures


@pytest.fixture
def elevator(lab_building_world):
    return lab_building_world.get_semantic_annotations_by_type(Elevator)[0]


@pytest.fixture
def ground_floor(lab_building_world):
    return lab_building_world.get_semantic_annotations_by_type(GroundFloor)[0]


@pytest.fixture
def first_floor(lab_building_world):
    return lab_building_world.get_semantic_annotations_by_type(FirstFloor)[0]


def cabin_to_world(
    elevator: Elevator, cabin_x: float, cabin_y: float, cabin_z: float
) -> np.ndarray:
    """
    :return: The world coordinates of the point at ``(cabin_x, cabin_y, cabin_z)`` in
        the cabin's frame.
    """
    world = elevator._world
    return world.transform(
        Point3(cabin_x, cabin_y, cabin_z, reference_frame=elevator.root), world.root
    ).to_np()[:3]


def head_of_person_at(
    elevator: Elevator, cabin_x: float, cabin_y: float, floor_height: float
) -> Point3:
    """
    :return: The head of a person standing at ``(cabin_x, cabin_y)`` in the cabin's
        frame on the floor at ``floor_height``.
    """
    x, y, _ = cabin_to_world(elevator, cabin_x, cabin_y, 0.0)
    return Point3(x, y, floor_height + 1.65, reference_frame=elevator._world.root)


def corners_of(points: list[Point3]) -> np.ndarray:
    """
    :return: The coordinates of ``points``, sorted so that two outlines compare equal
        regardless of their winding.
    """
    return np.array(sorted((float(p.x), float(p.y), float(p.z)) for p in points))


# %% cabin


def test_person_inside_the_cabin_is_in_the_cabin_area(elevator, ground_floor):
    person = head_of_person_at(
        elevator, 0.25, 0.2, float(ground_floor.floor_plane[0].z)
    )
    assert ElevatorCabinArea(elevator).contains(person)


def test_person_in_front_of_the_doors_is_not_in_the_cabin_area(elevator, ground_floor):
    person = head_of_person_at(
        elevator,
        -elevator.scale.x / 2 - 0.625,
        0.0,
        float(ground_floor.floor_plane[0].z),
    )
    assert not ElevatorCabinArea(elevator).contains(person)


def test_cabin_area_moves_with_the_cabin(elevator, ground_floor, first_floor):
    upstairs = head_of_person_at(
        elevator, 0.0, 0.0, float(first_floor.floor_plane[0].z)
    )
    downstairs = head_of_person_at(
        elevator, 0.0, 0.0, float(ground_floor.floor_plane[0].z)
    )
    elevator.drive_to_floor(first_floor)
    assert ElevatorCabinArea(elevator).contains(upstairs)
    assert not ElevatorCabinArea(elevator).contains(downstairs)


def test_cabin_area_footprint_is_the_cabin_floor(elevator):
    scale = elevator.scale
    cabin_floor = [
        cabin_to_world(elevator, x, y, -scale.z / 2)
        for x in (-scale.x / 2, scale.x / 2)
        for y in (-scale.y / 2, scale.y / 2)
    ]
    assert corners_of(ElevatorCabinArea(elevator).footprint()) == pytest.approx(
        np.array(sorted(map(tuple, cabin_floor))), abs=1e-6
    )


# %% waiting area


def test_person_in_front_of_the_doors_waits_on_its_level(
    elevator, ground_floor, first_floor
):
    person = head_of_person_at(
        elevator,
        -elevator.scale.x / 2 - 0.925,
        0.45,
        float(ground_floor.floor_plane[0].z),
    )
    assert ElevatorWaitingArea(elevator, ground_floor).contains(person)
    assert not ElevatorWaitingArea(elevator, first_floor).contains(person)


def test_person_inside_the_cabin_is_not_waiting(elevator, ground_floor):
    person = head_of_person_at(elevator, 0.0, 0.0, float(ground_floor.floor_plane[0].z))
    assert not ElevatorWaitingArea(elevator, ground_floor).contains(person)


def test_person_beyond_the_waiting_depth_is_not_waiting(elevator, ground_floor):
    area = ElevatorWaitingArea(elevator, ground_floor)
    person = head_of_person_at(
        elevator,
        -elevator.scale.x / 2 - area.depth - 0.1,
        0.0,
        float(ground_floor.floor_plane[0].z),
    )
    assert not area.contains(person)


def test_waiting_area_footprint_lies_in_front_of_the_doors_on_the_level(
    elevator, ground_floor
):
    area = ElevatorWaitingArea(elevator, ground_floor)
    door_x = -elevator.scale.x / 2
    half_width = elevator.scale.y / 2 + area.side_margin
    in_front_of_the_doors = [
        cabin_to_world(elevator, x, y, -elevator.scale.z / 2)
        for x in (door_x - area.depth, door_x)
        for y in (-half_width, half_width)
    ]
    assert corners_of(area.footprint()) == pytest.approx(
        np.array(sorted(map(tuple, in_front_of_the_doors))), abs=1e-6
    )
