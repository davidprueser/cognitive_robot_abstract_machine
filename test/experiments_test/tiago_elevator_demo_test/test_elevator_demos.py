import numpy as np
import pytest
from typing_extensions import Iterator

from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans.actions.core.navigation import ElevatorRideAnnouncement
from coraplex.robot_plans.actions.core.speech import SayAction
from experiments.tiago_elevator_demo.demo import (
    ElevatorController,
    TiagoElevatorRideDemonstration,
)
from experiments.tiago_elevator_waiting_demo.demo import (
    ElevatorBoardingAnnouncement,
    TiagoElevatorWaitingDemonstration,
)
from experiments.tiago_elevator_waiting_demo.exceptions import BoardingFailedError
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Region

from .scripted_social_navigation import BoardingAnswer

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


def test_tiago_says_where_it_is_and_where_it_is_headed_while_riding(tiago_speech):
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.run()

    ground_floor = world.get_semantic_annotations_by_type(GroundFloor)[0]
    first_floor = world.get_semantic_annotations_by_type(FirstFloor)[0]
    expected_sentences = [
        announcement.verbalize(current_floor=ground_floor, target_floor=first_floor)
        for announcement in (
            ElevatorRideAnnouncement.DEPARTING,
            ElevatorRideAnnouncement.WAITING_AT_DOORS,
            ElevatorRideAnnouncement.RIDING,
            ElevatorRideAnnouncement.ARRIVED,
        )
    ]
    assert (
        tiago_speech.wait_for_sentences(len(expected_sentences)) == expected_sentences
    )


def test_simulated_tiago_stands_on_the_ground_floor():
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )

    world = demonstration.build_simulated_world()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    lobby = demonstration.office.get_room(world, GroundFloor, AriseOfficeRoom.LOBBY)
    assert float(robot.root.global_transform.z) == pytest.approx(
        float(lobby.floor.root.global_transform.z)
    )


# %% serving the elevator's floors


def test_the_elevator_controller_opens_the_doors_at_the_floor_it_serves(
    arise_office_world,
):
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    controller = ElevatorController(_world=arise_office_world, elevator=elevator)

    controller.serve_floor(first_floor)

    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )
    assert [door.movable_joint.position for door in elevator.doors] == [
        door.movable_joint.dof.limits.upper.position for door in elevator.doors
    ]
    controller.stop()


def test_the_elevator_controller_keeps_a_cabin_a_motion_moves_off_its_floor_at_rest_there(
    arise_office_world,
):
    """
    A robot riding the cabin makes the cabin's drive part of the robot's motions, which
    move it in a kinematic simulation unless something holds the cabin.
    """
    elevator = arise_office_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = arise_office_world.get_semantic_annotations_by_type(FirstFloor)[0]
    controller = ElevatorController(_world=arise_office_world, elevator=elevator)
    controller.serve_floor(first_floor)
    velocities = np.zeros(len(arise_office_world.state))
    (cabin_column,) = arise_office_world.state.column_indices(
        [elevator.movable_joint.raw_dof]
    )
    velocities[cabin_column] = -1.0

    arise_office_world.apply_control_commands(
        velocities, dt=0.1, derivative=Derivatives.velocity
    )

    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(first_floor)
    )
    cabin_state = arise_office_world.state[elevator.movable_joint.raw_dof.id]
    assert [
        cabin_state[derivative]
        for derivative in Derivatives.range(Derivatives.velocity, Derivatives.jerk)
    ] == [0.0, 0.0, 0.0]
    controller.stop()


# %% calling the cabin


@pytest.fixture
def ride_demonstration() -> Iterator[TiagoElevatorRideDemonstration]:
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )
    yield demonstration
    demonstration.tear_down()


@pytest.fixture
def operated_world(ride_demonstration) -> World:
    """
    The office with Tiago in the lobby and the cabin sent up to the first floor, run by
    the demonstration's elevator operator.
    """
    world = ride_demonstration.build_simulated_world()
    ride_demonstration.build_context(world)
    ride_demonstration.elevator_operator.controller.serve_floor(
        world.get_semantic_annotations_by_type(FirstFloor)[0]
    )
    return world


def move_robot_into(robot: AbstractRobot, region: Region) -> None:
    """
    Put ``robot``'s base on the middle of ``region``, on the level the robot stands on.
    """
    world = robot._world
    odom = robot.root.parent_kinematic_structure_entity
    odom_T_region = world.transform(
        HomogeneousTransformationMatrix(reference_frame=region), odom
    )
    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        float(odom_T_region.x), float(odom_T_region.y), 0, reference_frame=odom
    )
    world.notify_state_change()


def test_the_cabin_comes_to_a_robot_waiting_on_another_level(operated_world):
    elevator = operated_world.get_semantic_annotations_by_type(Elevator)[0]
    ground_floor = operated_world.get_semantic_annotations_by_type(GroundFloor)[0]
    robot = operated_world.get_semantic_annotations_by_type(Tiago)[0]

    move_robot_into(robot, elevator.waiting_areas[ground_floor])

    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(ground_floor)
    )
    assert [door.movable_joint.position for door in elevator.doors] == [
        door.movable_joint.dof.limits.upper.position for door in elevator.doors
    ]


def test_the_cabin_stays_while_no_robot_waits_for_it(operated_world):
    elevator = operated_world.get_semantic_annotations_by_type(Elevator)[0]
    first_floor = operated_world.get_semantic_annotations_by_type(FirstFloor)[0]

    operated_world.notify_state_change()

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
def office_world(waiting_demonstration) -> World:
    return waiting_demonstration.build_simulated_world()


def test_tiago_waits_in_front_of_the_doors_facing_the_cabin(
    waiting_demonstration, office_world
):
    elevator = office_world.get_semantic_annotations_by_type(Elevator)[0]

    world_T_waiting = office_world.transform(
        elevator.pose_in_front_of_doors(waiting_demonstration.distance_from_doors),
        office_world.root,
    ).to_np()

    world_T_doorway = office_world.transform(
        elevator.doorway_pose, office_world.root
    ).to_np()
    inward = world_T_doorway[:3, 0]
    assert world_T_waiting[:3, 3] == pytest.approx(
        world_T_doorway[:3, 3] - waiting_demonstration.distance_from_doors * inward
    )
    assert world_T_waiting[:3, 0] == pytest.approx(inward)


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.ABORT], indirect=True)
def test_tiago_hands_its_base_over_from_where_it_waits(
    social_navigation, waiting_demonstration
):
    office = waiting_demonstration.office
    with pytest.raises(BoardingFailedError):
        waiting_demonstration.run()

    [goal] = social_navigation.goals
    doors_y = float(office.cabin_centre.y) - office.cabin_scale.x / 2
    position = goal.robot_pose.pose.position
    assert [position.x, position.y] == pytest.approx(
        [
            float(office.cabin_centre.x),
            doors_y - waiting_demonstration.distance_from_doors,
        ],
        abs=0.1,
    )


@pytest.mark.parametrize("social_navigation", [BoardingAnswer.ABORT], indirect=True)
def test_tiago_says_it_waits_to_board_in_front_of_the_elevator(
    social_navigation, tiago_speech, waiting_demonstration, office_world
):
    with pytest.raises(BoardingFailedError):
        waiting_demonstration.run()

    ground_floor = office_world.get_semantic_annotations_by_type(GroundFloor)[0]
    expected_sentence = ElevatorBoardingAnnouncement.WAITING_TO_BOARD.verbalize(
        current_floor=ground_floor
    )
    assert tiago_speech.wait_for_sentences(1) == [expected_sentence]


def test_tiago_says_it_is_inside_the_elevator_once_it_boarded(
    waiting_demonstration, office_world
):
    context = waiting_demonstration.build_context(office_world)

    plan = waiting_demonstration.build_plan(context)

    last_step = plan.children[-1].designator
    assert isinstance(last_step, SayAction)
    assert last_step.text == ElevatorBoardingAnnouncement.BOARDED
