from __future__ import division

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.exceptions import NodeInitializationError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianOrientation,
    CartesianPositionStraight,
    CartesianPose,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.world_description.connections import DiffDrive
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class DiffDriveBaseGoal(Sequence):
    diff_drive_connection: DiffDrive | None = field(kw_only=True, default=None)
    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    max_linear_velocity: float = 0.1
    max_angular_velocity: float = 0.5
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    pointing_axis = None
    always_forward: bool = False
    nodes: list[MotionStatechartNode] = field(default_factory=list, init=False)

    def expand(self, context: BuildContext) -> None:
        if self.diff_drive_connection is None:
            diff_drives = context.world.get_connections_by_type(DiffDrive)
            if len(diff_drives) == 0:
                raise NodeInitializationError(self, "No diff drives found in world.")
            if len(diff_drives) > 1:
                raise NodeInitializationError(
                    self, "More than one diff drive found in world."
                )
            self.diff_drive_connection = diff_drives[0]
        map = context.world.root
        tip = self.diff_drive_connection.child

        root_T_goal = context.world.transform(self.goal_pose, map)
        root_T_current = tip.global_pose
        root_V_current_to_goal = (
            root_T_goal.to_position() - root_T_current.to_position()
        )
        root_V_current_to_goal.scale(1)
        root_V_z = Vector3.Z(reference_frame=map)
        root_R_first_orientation = RotationMatrix.from_vectors(
            x=root_V_current_to_goal, z=root_V_z, reference_frame=map
        )

        root_T_goal2 = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_T_goal.to_position(), rotation_matrix=root_R_first_orientation
        )

        self.nodes = [
            CartesianOrientation(
                root_link=map,
                tip_link=tip,
                goal_orientation=root_R_first_orientation,
                name=self.name,
                weight=self.weight,
            ),
            CartesianPose(
                root_link=map,
                tip_link=tip,
                goal_pose=root_T_goal2,
                name=self.name,
                weight=self.weight,
            ),
            CartesianPose(
                root_link=map,
                tip_link=tip,
                goal_pose=root_T_goal,
                name=self.name,
                weight=self.weight,
            ),
        ]
        super().expand(context)


@dataclass(eq=False, repr=False)
class CartesianPoseStraight(Goal):
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)
    goal_pose: HomogeneousTransformationMatrix = field(kw_only=True)
    reference_linear_velocity: float = CartesianPosition.default_reference_velocity
    reference_angular_velocity: float = CartesianOrientation.default_reference_velocity
    weight: float = DefaultWeights.WEIGHT_ABOVE_CA
    absolute: bool = False

    def __post_init__(self):
        """
        See CartesianPose. In contrast to it, this goal will try to move tip_link in a straight line.
        """
        self.add_task(
            CartesianPositionStraight(
                root_link=self.root_link,
                tip_link=self.tip_link,
                name=self.name + "/pos",
                goal_point=self.goal_pose.to_position(),
                reference_velocity=self.reference_linear_velocity,
                weight=self.weight,
                absolute=self.absolute,
            )
        )
        self.add_task(
            CartesianOrientation(
                root_link=self.root_link,
                tip_link=self.tip_link,
                name=self.name + "/rot",
                goal_orientation=self.goal_pose.to_rotation_matrix(),
                reference_velocity=self.reference_angular_velocity,
                absolute=self.absolute,
                weight=self.weight,
                point_of_debug_matrix=self.goal_pose.to_position(),
            )
        )
        obs_expressions = []
        for task in self.tasks:
            obs_expressions.append(task.observation_expression)
        self.observation_expression = sm.logic_all(*obs_expressions)


@dataclass(eq=False, repr=False)
class RelativePositionSequence(Goal):
    goal1: HomogeneousTransformationMatrix = field(kw_only=True)
    goal2: HomogeneousTransformationMatrix = field(kw_only=True)
    root_link: Body = field(kw_only=True)
    tip_link: Body = field(kw_only=True)

    def __post_init__(self):
        """
        Only meant for testing.
        """
        name1 = f"{self.name}/goal1"
        name2 = f"{self.name}/goal2"
        task1 = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_pose=self.goal1,
            name=name1,
            absolute=True,
        )
        self.add_task(task1)
        task2 = CartesianPose(
            root_link=self.root_link,
            tip_link=self.tip_link,
            goal_pose=self.goal2,
            name=name2,
            absolute=True,
        )
        self.add_task(task2)
        task2.start_condition = task1
        task1.end_condition = task1
        self.observation_expression = task2.observation_expression
