"""
Tiago rides the elevator of a building, the lab by default, from the ground floor to the
first floor.

Running with :attr:`~coraplex.datastructures.enums.ExecutionType.REAL` drives the real
TIAGo, or the robot of the TIAGo ARISE Gazebo simulator: the world comes from the running
controller, the base lidar from ``/scan``, and the cabin's height from the elevator's joint
states. The elevator itself is operated
outside the plan, by a person on the real robot and by the simulator's elevator operator
in Gazebo.

In a kinematic simulation an :class:`ElevatorOperator` stands in for whoever operates
the elevator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import ElevatorNavigation
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from experiments.ra_fetch_elevator_people.elevator_people import (
    ElevatorDrive,
    SimulatorTopic,
)
from semantic_digital_twin.callbacks.callback import ModelChangeCallback
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.lab_building import (
    LabBuilding,
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
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

# %% operating the elevator in a kinematic simulation


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

    def on_model_change(self, **kwargs) -> None:
        if self.robot.root.parent_kinematic_structure_entity is not self.elevator.root:
            return
        self.elevator.close()
        self.elevator.drive_to_floor(self.floor)
        self.elevator.open()


# %% demonstration


@dataclass
class TiagoElevatorRideDemonstration(RobotDemonstration):
    """
    Tiago drives from the ground floor of a building into the elevator and leaves it on
    the first floor.
    """

    ros_node_name: str = "tiago_elevator_demo_node"

    building: ElevatorBuilding = field(default_factory=LabBuilding)
    """
    The building the robot rides the elevator in.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where a simulated Tiago starts, relative to the start pose of :attr:`building`.
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

    elevator_drive: ElevatorDrive | None = field(init=False, default=None)
    """
    Where the cabin's height comes from in a real run.
    """

    def build_simulated_world(self) -> World:
        """
        Put the Tiago on its drive in a world holding nothing but a map body, with its
        odom at the start pose of :attr:`building`.
        """
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    world_T_odom=self.building.world_T_robot_start,
                    odom_T_robot_start=self.odom_T_robot_start,
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        return bool(world.get_semantic_annotations_by_type(Elevator))

    def populate_scene(self, world: World) -> None:
        """
        Spawn the building.

        In a kinematic simulation the elevator waits with open doors.
        """
        elevator = self.building.spawn(world)
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
            if self.elevator_drive is None:
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
        self.elevator_drive = ElevatorDrive(
            node=self.ros_node,
            topic_name=self.elevator_joint_state_topic,
            position_offset=elevator.drive_position_for_floor(
                elevator._world.get_semantic_annotations_by_type(GroundFloor)[0]
            ),
        )
        self.ros_node.create_timer(
            self.elevator_drive_update_period,
            lambda: self.elevator_drive.apply_to(elevator),
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Park the arms and ride the elevator to the first floor.
        """
        elevator = context.world.get_semantic_annotations_by_type(Elevator)[0]
        first_floor = context.world.get_semantic_annotations_by_type(FirstFloor)[0]
        return sequential(
            [
                ParkArmsAction(context.robot.all_arms),
                ElevatorNavigation(
                    elevator,
                    first_floor,
                    arrival_threshold=self.elevator_arrival_threshold,
                ),
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
