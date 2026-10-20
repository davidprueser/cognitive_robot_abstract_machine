import pytest

from experiments.lab_elevator_ride_demo import ElevatorRide, LabElevatorRideDemo
from experiments.lab_elevator_room_demo import create_lab_elevator_room_world
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
)

# %% fixtures


@pytest.fixture()
def ride_to_first_floor() -> ElevatorRide:
    world = create_lab_elevator_room_world()
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    (first_floor,) = world.get_semantic_annotations_by_type(FirstFloor)
    return ElevatorRide(elevator=elevator, destination=first_floor)


def ride_until_finished(ride: ElevatorRide, step_seconds: float) -> list[tuple]:
    """
    Step ``ride`` until it is finished.

    :return: The cabin position and the door positions before the first step and after
        every step.
    """

    def snapshot() -> tuple:
        return ride.elevator.movable_joint.position, [
            door.movable_joint.position for door in ride.elevator.doors
        ]

    snapshots = [snapshot()]
    for _ in range(10_000):
        if ride.is_finished:
            return snapshots
        ride.step(step_seconds)
        snapshots.append(snapshot())
    raise AssertionError("The ride did not finish.")


# %% ride


def test_ride_ends_at_the_destination_with_the_doors_open(ride_to_first_floor):
    ride_until_finished(ride_to_first_floor, step_seconds=0.1)
    elevator = ride_to_first_floor.elevator
    assert elevator.movable_joint.position == pytest.approx(
        elevator.drive_position_for_floor(ride_to_first_floor.destination)
    )
    assert [door.movable_joint.position for door in elevator.doors] == pytest.approx(
        [door.movable_joint.dof.limits.upper.position for door in elevator.doors]
    )


def test_cabin_moves_only_while_the_doors_are_closed(ride_to_first_floor):
    snapshots = ride_until_finished(ride_to_first_floor, step_seconds=0.1)
    closed_positions = [
        door.movable_joint.dof.limits.lower.position
        for door in ride_to_first_floor.elevator.doors
    ]
    for (cabin_before, doors_before), (cabin_after, _) in zip(snapshots, snapshots[1:]):
        if cabin_after != pytest.approx(cabin_before):
            assert doors_before == pytest.approx(closed_positions)


def test_step_moves_the_doors_by_the_speed_times_the_step_duration(
    ride_to_first_floor,
):
    doors = ride_to_first_floor.elevator.doors
    before = [door.movable_joint.position for door in doors]
    step_seconds = 0.1
    ride_to_first_floor.step(step_seconds)
    assert [
        before_position - door.movable_joint.position
        for before_position, door in zip(before, doors)
    ] == pytest.approx([ride_to_first_floor.speed * step_seconds] * len(doors))


# %% demo


def test_demo_waits_for_the_start_delay_before_riding(rclpy_node):
    demo = LabElevatorRideDemo(node=rclpy_node)
    assert int(demo.start_delay_seconds * 1e9) in {
        timer.timer_period_ns for timer in rclpy_node.timers
    }
