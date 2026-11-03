from copy import deepcopy

import pytest

from semantic_digital_twin.reasoning.predicates import InsideRegion
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Human,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Region

# %% fixtures


@pytest.fixture
def elevator(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(Elevator)[0]


@pytest.fixture
def ground_floor(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(GroundFloor)[0]


@pytest.fixture
def first_floor(arise_office_world):
    return arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]


def person_standing_at(world, x: float, y: float, floor_height: float) -> Human:
    """
    A person the size of an adult standing on ``floor_height`` at ``x``, ``y``.
    """
    scale = Scale(0.4, 0.4, 1.7)
    return Human.create_with_new_body_in_world(
        name=f"person_{x}_{y}_{floor_height}",
        world=world,
        world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            x, y, floor_height + scale.z / 2
        ),
        scale=scale,
    )


def box_scale(region: Region) -> Scale:
    """
    :return: The extents of the box that makes up ``region``.
    """
    return region.area[0].scale


def floor_outline(region: Region) -> list[float]:
    """
    :return: The region's x and y extents and the height of its bottom, in the world
        root frame.
    """
    bounds = region.area.as_bounding_box_collection_in_frame(
        region._world.root
    ).bounding_box()
    return [bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y, bounds.min_z]


# %% cabin


def test_person_inside_the_cabin_is_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x + 0.25,
        arise_office.cabin_centre.y + 0.2,
        arise_office.ground_floor_height,
    )
    assert InsideRegion(person.root, elevator.cabin_area)()


def test_person_in_front_of_the_doors_is_not_in_the_cabin_area(
    arise_office_world, elevator, arise_office
):
    person = person_standing_at(
        arise_office_world, 0.0, 1.5, arise_office.ground_floor_height
    )
    assert not InsideRegion(person.root, elevator.cabin_area)()


def test_cabin_area_moves_with_the_cabin(
    arise_office_world, elevator, first_floor, arise_office
):
    upstairs = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.first_floor_height,
    )
    downstairs = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    elevator.drive_to_floor(first_floor)
    assert InsideRegion(upstairs.root, elevator.cabin_area)()
    assert not InsideRegion(downstairs.root, elevator.cabin_area)()


def test_cabin_area_stands_on_the_cabin_floor(elevator, arise_office):
    assert floor_outline(elevator.cabin_area) == pytest.approx(
        [
            float(arise_office.cabin_centre.x) - elevator.scale.y / 2,
            float(arise_office.cabin_centre.x) + elevator.scale.y / 2,
            float(arise_office.cabin_centre.y) - elevator.scale.x / 2,
            float(arise_office.cabin_centre.y) + elevator.scale.x / 2,
            arise_office.ground_floor_height,
        ]
    )


def test_cabin_area_enters_the_world_once_on_first_access(arise_office_world, elevator):
    regions_before = list(arise_office_world.regions)
    cabin_area = elevator.cabin_area
    assert elevator.cabin_area is cabin_area
    assert list(arise_office_world.regions) == regions_before + [cabin_area]


# %% waiting area


def test_person_in_front_of_the_doors_waits_on_its_level(
    arise_office_world, elevator, ground_floor, first_floor, arise_office
):
    person = person_standing_at(
        arise_office_world, 0.45, 1.2, arise_office.ground_floor_height
    )
    assert InsideRegion(person.root, elevator.waiting_areas[ground_floor])()
    assert not InsideRegion(person.root, elevator.waiting_areas[first_floor])()


def test_person_inside_the_cabin_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    person = person_standing_at(
        arise_office_world,
        arise_office.cabin_centre.x,
        arise_office.cabin_centre.y,
        arise_office.ground_floor_height,
    )
    assert not InsideRegion(person.root, elevator.waiting_areas[ground_floor])()


def test_person_beyond_the_waiting_depth_is_not_waiting(
    arise_office_world, elevator, ground_floor, arise_office
):
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    waiting_area_depth = box_scale(elevator.waiting_areas[ground_floor]).x
    person = person_standing_at(
        arise_office_world,
        0.0,
        door_y - waiting_area_depth - 0.3,
        arise_office.ground_floor_height,
    )
    assert not InsideRegion(person.root, elevator.waiting_areas[ground_floor])()


def test_waiting_area_lies_in_front_of_the_doors_on_the_level(
    elevator, ground_floor, arise_office
):
    door_y = float(arise_office.cabin_centre.y) - elevator.scale.x / 2
    waiting_area_scale = box_scale(elevator.waiting_areas[ground_floor])
    half_width = waiting_area_scale.y / 2
    assert floor_outline(elevator.waiting_areas[ground_floor]) == pytest.approx(
        [
            -half_width,
            half_width,
            door_y - waiting_area_scale.x,
            door_y,
            arise_office.ground_floor_height,
        ]
    )


def test_waiting_area_stays_on_its_level_while_the_cabin_rides(
    elevator, ground_floor, first_floor
):
    before = floor_outline(elevator.waiting_areas[ground_floor])
    elevator.drive_to_floor(first_floor)
    assert floor_outline(elevator.waiting_areas[ground_floor]) == pytest.approx(before)


def test_waiting_area_has_the_elevators_waiting_area_scale(elevator, ground_floor):
    elevator.waiting_area_scale = Scale(1.0, 3.0, 2.0)
    assert (
        box_scale(elevator.waiting_areas[ground_floor]) == elevator.waiting_area_scale
    )


def test_waiting_area_is_fixed_to_the_elevator_shaft(elevator, ground_floor):
    waiting_area = elevator.waiting_areas[ground_floor]
    assert waiting_area.parent_connection.parent is elevator.movable_joint.parent


def test_elevator_has_a_waiting_area_per_level(elevator, ground_floor, first_floor):
    assert list(elevator.waiting_areas) == [ground_floor, first_floor]


def test_waiting_areas_enter_the_world_once_on_first_access(
    arise_office_world, elevator
):
    regions_before = list(arise_office_world.regions)
    waiting_areas = elevator.waiting_areas
    assert elevator.waiting_areas is waiting_areas
    assert list(arise_office_world.regions) == regions_before + list(
        waiting_areas.values()
    )


# %% elevator whose drive hangs off the map


def test_person_in_front_of_the_doors_of_an_elevator_hanging_off_the_map_waits(
    iai_office_world,
):
    elevator = iai_office_world.get_semantic_annotations_by_type(Elevator)[0]
    ground_floor = iai_office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    in_front_of_doors = iai_office_world.transform(
        elevator.pose_in_front_of_doors(0.75), iai_office_world.root
    )
    person = person_standing_at(
        iai_office_world,
        float(in_front_of_doors.x),
        float(in_front_of_doors.y),
        float(ground_floor.floor_plane[0].z),
    )

    assert InsideRegion(person.root, elevator.waiting_areas[ground_floor])()


# %% areas of a rebuilt world


def test_rebuilt_world_keeps_its_cabin_area(arise_office_world, elevator):
    elevator.cabin_area
    rebuilt_world = deepcopy(arise_office_world)
    regions_before = list(rebuilt_world.regions)

    rebuilt_cabin_area = rebuilt_world.get_semantic_annotations_by_type(Elevator)[
        0
    ].cabin_area

    assert rebuilt_cabin_area in regions_before
    assert list(rebuilt_world.regions) == regions_before


def test_rebuilt_world_keeps_its_waiting_areas(arise_office_world, elevator):
    elevator.waiting_areas
    rebuilt_world = deepcopy(arise_office_world)
    regions_before = list(rebuilt_world.regions)

    rebuilt_waiting_areas = rebuilt_world.get_semantic_annotations_by_type(Elevator)[
        0
    ].waiting_areas

    assert all(area in regions_before for area in rebuilt_waiting_areas.values())
    assert list(rebuilt_world.regions) == regions_before
