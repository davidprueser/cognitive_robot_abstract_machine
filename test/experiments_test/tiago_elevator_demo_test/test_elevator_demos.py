import json

import numpy as np
import pytest

from coraplex.datastructures.enums import ExecutionType
from experiments.ra_fetch_elevator_people.elevator_people import ElevatorPeople
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_elevator_waiting_demo.demo import (
    BoardingRequest,
    BoardingResult,
    BoardingStatus,
    ElevatorAreaName,
    ElevatorDescription,
    PersonNearElevator,
    TiagoElevatorWaitingDemonstration,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from ..ra_fetch_elevator_people_test.test_elevator_people import robot_base_at_height

# %% riding the elevator


def test_tiago_rides_the_elevator_to_the_first_floor(rclpy_node):
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(first_floor.floor_plane[0].z)
    )
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )


# %% handing the base over to the social navigation module


@pytest.fixture
def waiting_demonstration() -> TiagoElevatorWaitingDemonstration:
    return TiagoElevatorWaitingDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )


@pytest.fixture
def building_world(waiting_demonstration) -> World:
    world = waiting_demonstration.build_simulated_world()
    waiting_demonstration.populate_scene(world)
    return world


def test_tiago_waits_in_front_of_the_doors_facing_the_cabin(
    waiting_demonstration, building_world
):
    elevator = building_world.get_semantic_annotations_by_type(Elevator)[0]

    waiting_pose = building_world.transform(
        waiting_demonstration.waiting_pose(elevator), building_world.root
    ).to_np()

    description = ElevatorDescription.of(elevator)
    inward = np.array(
        [np.cos(description.inward_yaw), np.sin(description.inward_yaw), 0.0]
    )
    assert waiting_pose[:3, 3] == pytest.approx(
        np.array(description.door_center)
        - waiting_demonstration.distance_from_doors * inward
    )
    assert np.arctan2(waiting_pose[1, 0], waiting_pose[0, 0]) == pytest.approx(
        description.inward_yaw
    )


def test_elevator_description_gives_doors_and_cabin_on_the_cabin_floor(
    waiting_demonstration, building_world
):
    elevator_room = waiting_demonstration.building.elevator_room
    elevator = building_world.get_semantic_annotations_by_type(Elevator)[0]

    description = ElevatorDescription.of(elevator)

    map_T_cabin = (elevator_room.map_T_room @ elevator_room.room_T_elevator).to_np()
    cabin_scale = elevator_room.elevator.scale
    cabin_floor_center = map_T_cabin[:3, 3] - np.array([0.0, 0.0, cabin_scale.z / 2])
    inward = map_T_cabin[:3, 0]
    assert description.cabin_center == pytest.approx(
        cabin_floor_center.tolist(), abs=1e-6
    )
    assert description.door_center == pytest.approx(
        (cabin_floor_center - inward * cabin_scale.x / 2).tolist(), abs=1e-6
    )
    assert description.inward_yaw == pytest.approx(np.arctan2(inward[1], inward[0]))
    assert (description.cabin_width, description.cabin_depth) == pytest.approx(
        (cabin_scale.y, cabin_scale.x)
    )


def test_boarding_request_names_the_persons_in_the_cabin_and_the_waiting_area(
    rclpy_node, waiting_demonstration, building_world
):
    elevator = building_world.get_semantic_annotations_by_type(Elevator)[0]
    robot = building_world.get_semantic_annotations_by_type(Tiago)[0]
    ground_floor = building_world.get_semantic_annotations_by_type(GroundFloor)[0]
    elevator_people = ElevatorPeople.for_world(
        rclpy_node, building_world, robot.root.name.name, None
    )
    elevator_people.tf.tf_buffer.set_transform_static(
        robot_base_at_height(building_world, float(robot.root.global_transform.z)),
        "test",
    )
    head_height = float(ground_floor.floor_plane[0].z) + 1.65
    world_T_cabin = elevator.root.global_transform.to_np()
    inward, sideways = world_T_cabin[:3, 0], world_T_cabin[:3, 1]
    door_center = world_T_cabin[:3, 3] - inward * elevator.scale.x / 2
    waiting_position = door_center - inward * 1.0 + sideways * 0.3
    walker_position = door_center - inward * 5.0 + sideways * 2.5
    world_T_rider = HomogeneousTransformationMatrix.from_xyz_rpy(
        world_T_cabin[0, 3],
        world_T_cabin[1, 3],
        head_height,
        reference_frame=building_world.root,
    )
    elevator_people.mirror.place("rider", world_T_rider)
    elevator_people.mirror.place(
        "waiting",
        HomogeneousTransformationMatrix.from_xyz_rpy(
            waiting_position[0],
            waiting_position[1],
            head_height,
            reference_frame=building_world.root,
        ),
    )
    elevator_people.mirror.place(
        "walker",
        HomogeneousTransformationMatrix.from_xyz_rpy(
            walker_position[0],
            walker_position[1],
            head_height,
            reference_frame=building_world.root,
        ),
    )

    request = waiting_demonstration._boarding_request(elevator_people, elevator, robot)

    assert request.request_id == waiting_demonstration.request_id
    assert [(person.person_id, person.area) for person in request.persons] == [
        ("rider", ElevatorAreaName.CABIN),
        ("waiting", ElevatorAreaName.WAITING_AREA),
    ]
    assert request.persons[0].position == pytest.approx(
        world_T_rider.to_np()[:3, 3].tolist()
    )


def test_boarding_request_is_plain_json_for_the_social_navigation_module():
    request = BoardingRequest(
        request_id=3,
        frame_id="map",
        robot_position=[0.0, 0.625, 0.0],
        robot_yaw=1.57,
        elevator=ElevatorDescription(
            door_center=[0.0, 2.125, 0.0],
            cabin_center=[0.0, 2.925, 0.0],
            inward_yaw=1.57,
            cabin_width=1.6,
            cabin_depth=1.6,
        ),
        persons=[
            PersonNearElevator(
                person_id="in_cabin",
                area=ElevatorAreaName.CABIN,
                position=[0.35, 3.2, 1.75],
            )
        ],
    )

    assert json.loads(request.to_message().data) == {
        "request_id": 3,
        "frame_id": "map",
        "robot_position": [0.0, 0.625, 0.0],
        "robot_yaw": 1.57,
        "elevator": {
            "door_center": [0.0, 2.125, 0.0],
            "cabin_center": [0.0, 2.925, 0.0],
            "inward_yaw": 1.57,
            "cabin_width": 1.6,
            "cabin_depth": 1.6,
        },
        "persons": [
            {
                "person_id": "in_cabin",
                "area": ElevatorAreaName.CABIN.value,
                "position": [0.35, 3.2, 1.75],
            }
        ],
    }


def test_boarding_result_is_read_from_the_social_navigation_modules_json():
    result = BoardingResult.from_json(
        json.loads('{"request_id": 3, "status": "SUCCEEDED", "reason": "aboard"}')
    )

    assert result == BoardingResult(
        request_id=3, status=BoardingStatus.SUCCEEDED, reason="aboard"
    )
