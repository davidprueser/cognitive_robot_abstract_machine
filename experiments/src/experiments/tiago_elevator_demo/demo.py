"""
Tiago rides the elevator of the ARISE office from the ground floor to the first floor.

Running with :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` drives the robot
of the TIAGo ARISE Gazebo simulator, or the real TIAGo: the world comes from the running
controller, the base lidar from ``/scan``, and the cabin's height from the elevator's joint
states. The elevator itself is operated outside the plan, by a person on the real robot and
by the simulator's elevator operator in Gazebo.

In a kinematic simulation an :class:`ElevatorOperator` stands in for whoever operates
the elevator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import ElevatorNavigation
from experiments.tiago_arise_gazebo.elevator_drive import ElevatorDrive, SimulatorTopic
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.predefined_maps.arise_office import (
    AriseOfficeBuilding,
    AriseOfficeRoom,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    GroundFloor,
    Level,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World

# %% operating the elevator in a kinematic simulation


@dataclass(eq=False)
class ElevatorMotor(StateChangeCallback):
    """
    Drives an elevator's cabin to a floor and holds it there at rest, standing in for
    the motor of a real elevator.

    .. note:: A robot riding the cabin makes the cabin's drive part of the robot's
        motions, which move the drive in a kinematic simulation unless something holds
        the cabin.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator whose cabin this motor drives.
    """

    held_position: float = field(init=False)
    """
    The drive position the cabin is held at.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.held_position = self.elevator.movable_joint.position

    def drive_to_floor(self, floor: Level) -> None:
        """
        Drive the cabin to ``floor`` and hold it there.
        """
        self.held_position = self.elevator.drive_position_for_floor(floor)
        self.elevator.drive_to_floor(floor)

    def on_state_change(self, **kwargs) -> None:
        cabin_state = self._world.state[self.elevator.movable_joint.raw_dof.id]
        moving_derivatives = Derivatives.range(Derivatives.velocity, Derivatives.jerk)
        if np.isclose(
            self.elevator.movable_joint.position, self.held_position
        ) and not any(cabin_state[derivative] for derivative in moving_derivatives):
            return
        for derivative in moving_derivatives:
            cabin_state[derivative] = 0.0
        self.elevator.movable_joint.position = self.held_position


@dataclass(eq=False)
class ElevatorOperator(ModelChangeCallback):
    """
    Drives an elevator to a floor as soon as a robot boards it, standing in for whatever
    operates the elevator in the real world.
    """

    elevator: Elevator = field(kw_only=True)
    """
    The elevator this operator drives.
    """

    floor: Level = field(kw_only=True)
    """
    The floor the elevator is sent to once the robot is aboard.
    """

    robot: AbstractRobot = field(kw_only=True)
    """
    The robot whose boarding sets the elevator off.
    """

    motor: ElevatorMotor = field(init=False)
    """
    The motor that drives the cabin and holds it at its floor.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.motor = ElevatorMotor(_world=self._world, elevator=self.elevator)

    def on_model_change(self, **kwargs) -> None:
        if self.robot.root.parent_kinematic_structure_entity is not self.elevator.root:
            return
        self.elevator.close()
        self.motor.drive_to_floor(self.floor)
        self.elevator.open()

    def stop(self) -> None:
        self.motor.stop()
        super().stop()


# %% demonstration


@dataclass
class TiagoElevatorRideDemonstration(RobotDemonstration):
    """
    Tiago drives from the ground floor of the ARISE office into the elevator and leaves
    it on the first floor.
    """

    ros_node_name: str = "tiago_elevator_demo_node"

    office: AriseOfficeBuilding = field(default_factory=AriseOfficeBuilding)
    """
    The building the robot rides the elevator in.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where a simulated Tiago starts, as it is spawned in the Gazebo simulator.
    """

    elevator_joint_state_topic: str = SimulatorTopic.ELEVATOR_JOINT_STATES
    """
    Topic the elevator of a real run publishes its joint states on.
    """

    elevator_arrival_threshold: float = 0.03
    """
    How far the cabin may stand from a floor's height and still count as serving it; a
    loaded cabin sags by a few millimetres.
    """

    elevator_drive_update_period: float = 0.1
    """
    Seconds between two updates of the cabin's height from the elevator's joint states
    in a real run.
    """

    elevator_operator: ElevatorOperator | None = field(init=False, default=None)
    """
    The operator of the elevator in a kinematic simulation.
    """

    elevator_inputs: WorldStateInputs | None = field(init=False, default=None)
    """
    Writes the cabin's height from the elevator's joint states in a real run.
    """

    def build_simulated_world(self) -> World:
        """
        Populate a world holding nothing but a map body and put the Tiago on its drive
        on the office's ground floor.
        """
        world = World.create_with_root_body()
        self.populate_scene(world)
        RobotSpecification(
            semantic_annotation_type=self.used_robot,
            world_T_odom=world.transform(
                HomogeneousTransformationMatrix(
                    reference_frame=self.office.get_room(
                        world, GroundFloor, AriseOfficeRoom.LOBBY
                    ).floor.root
                ),
                world.root,
            ),
            odom_T_robot_start=self.odom_T_robot_start,
        ).spawn(world)
        return world

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(Elevator))

    def populate_scene(self, world: World) -> None:
        """
        Spawn the office.

        In a kinematic simulation the elevator waits with open doors.
        """
        elevator = self.office.spawn(world)
        if self.execution_type is not ExecutionType.REAL:
            elevator.open()

    def acquire_world(self) -> World:
        """
        On a real run, the base lidar is read from the robot instead of the world.
        """
        world = super().acquire_world()
        if self.execution_type is ExecutionType.REAL:
            lidar = world.get_semantic_annotations_by_type(self.used_robot)[
                0
            ].mobile_base.lidar
            lidar.use_source(lidar.real_source(self.ros_node))
        return world

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the Tiago in ``world`` and connect the elevator to
        whatever moves its cabin.

        .. note:: The ROS node has to be in the context for a real robot.
        """
        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]
        elevator = world.get_semantic_annotations_by_type(Elevator)[0]
        if self.execution_type is ExecutionType.REAL:
            if self.elevator_inputs is None:
                self._follow_elevator_drive(elevator)
        elif self.elevator_operator is None:
            self.elevator_operator = ElevatorOperator(
                _world=world,
                elevator=elevator,
                floor=world.get_semantic_annotations_by_type(FirstFloor)[0],
                robot=robot,
            )
        return Context(
            world=world,
            robot=robot,
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
            _debug=self.debug,
        )

    def _follow_elevator_drive(self, elevator: Elevator) -> None:
        """
        Keep the cabin's height in the world equal to the one the elevator reports.
        """
        self.elevator_inputs = WorldStateInputs(world=elevator._world)
        self.elevator_inputs.add_input(
            ElevatorDrive(
                world=elevator._world,
                node=self.ros_node,
                topic_name=self.elevator_joint_state_topic,
                elevator=elevator,
                position_offset=self.office.ground_floor_height,
            )
        )
        self.ros_node.create_timer(
            self.elevator_drive_update_period, self.elevator_inputs.synchronize
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Ride the elevator to the first floor.
        """
        elevator = context.world.get_semantic_annotations_by_type(Elevator)[0]
        first_floor = context.world.get_semantic_annotations_by_type(FirstFloor)[0]
        return sequential(
            [
                ElevatorNavigation(
                    elevator,
                    first_floor,
                    arrival_threshold=self.elevator_arrival_threshold,
                )
            ],
            context=context,
        )

    def tear_down(self) -> None:
        if self.elevator_operator is not None:
            self.elevator_operator.stop()
            self.elevator_operator = None
        super().tear_down()


def main(
    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED,
) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=execution_type
    ).run()


if __name__ == "__main__":
    main(execution_type=ExecutionType.REAL)
