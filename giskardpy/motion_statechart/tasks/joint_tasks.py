from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.monitors.joint_monitors import JointGoalReached
from giskardpy.motion_statechart.tasks.task import Task, WEIGHT_BELOW_CA
from semantic_world.connections import Has1DOFState, RevoluteConnection
from semantic_world.prefixed_name import PrefixedName
from semantic_world.spatial_types.derivatives import Derivatives


@dataclass
class JointPositionList(Task):
    name: str
    goal_state: Dict[Union[PrefixedName, str], float]
    threshold: float = 0.01
    weight: float = WEIGHT_BELOW_CA
    max_velocity: float = 1.0

    def __post_init__(self):
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.connections = []
        if len(self.goal_state) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')

        for joint_name, goal_position in self.goal_state.items():
            connection = god_map.world.get_connection_by_name(joint_name)
            self.connections.append(connection)
            if not isinstance(connection, Has1DOFState):
                raise GoalInitalizationException(f'Connection {joint_name} must be of type Has1DOFState')

            ul_pos = connection.dof.upper_limits.position
            ll_pos = connection.dof.lower_limits.position
            if ll_pos is not None:
                goal_position = cas.limit(goal_position, ll_pos, ul_pos)

            ul_vel = connection.dof.upper_limits.velocity
            ll_vel = connection.dof.lower_limits.velocity
            velocity_limit = cas.limit(self.max_velocity, ll_vel, ul_vel)

            self.current_positions.append(connection.dof.symbols.position)
            self.goal_positions.append(goal_position)
            self.velocity_limits.append(velocity_limit)

        for connection, current, goal, velocity_limit in zip(self.connections, self.current_positions,
                                                       self.goal_positions, self.velocity_limits):
            if (isinstance(connection, RevoluteConnection)
                    and not connection.dof.has_position_limits()):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(name=f'{self.name}/{connection.name}',
                                         reference_velocity=velocity_limit,
                                         equality_bound=error,
                                         weight=self.weight,
                                         task_expression=current)
        joint_monitor = JointGoalReached(goal_state=self.goal_state,
                                         threshold=self.threshold)
        self.observation_expression = joint_monitor.observation_expression

class MirrorJointPosition(Task):
    def __init__(self, *,
                 name: str,
                 mapping: Dict[str, str],
                 group_name: Optional[str] = None,
                 threshold: float = 0.01,
                 weight: Optional[float] = None,
                 max_velocity: Optional[float] = None,
                 plot: bool = True):
        super().__init__(name=name, plot=plot)
        if weight is None:
            weight = WEIGHT_BELOW_CA
        if max_velocity is None:
            max_velocity = 1.0
        self.current_positions = []
        self.goal_positions = []
        self.velocity_limits = []
        self.joint_names = []
        self.max_velocity = max_velocity
        self.weight = weight
        goal_state = {}
        for joint_name, target_joint_name in mapping.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            target_joint_name = god_map.world.search_for_joint_name(target_joint_name, group_name)
            self.joint_names.append(joint_name)

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = cas.limit(max_velocity, ll_vel, ul_vel)

            joint: OneDofJoint = god_map.world.joints[joint_name]
            target_joint: OneDofJoint = god_map.world.joints[target_joint_name]
            self.current_positions.append(joint.get_symbol(Derivatives.position))
            self.goal_positions.append(target_joint.get_symbol(Derivatives.position))
            self.velocity_limits.append(velocity_limit)
            goal_state[joint_name.short_name] = self.goal_positions[-1]

        for name, current, goal, velocity_limit in zip(self.joint_names, self.current_positions,
                                                       self.goal_positions, self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                error = cas.shortest_angular_distance(current, goal)
            else:
                error = goal - current

            self.add_equality_constraint(name=f'{self.name}/{name}',
                                         reference_velocity=velocity_limit,
                                         equality_bound=0,
                                         weight=self.weight,
                                         task_expression=error)
        joint_monitor = JointGoalReached(goal_state=goal_state,
                                         threshold=threshold)
        self.observation_expression = joint_monitor.observation_expression


class JointPositionLimitList(Task):
    def __init__(self,
                 lower_upper_limits: Dict[str, Tuple[float, float]],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None):
        """
        Calls JointPosition for a list of joints.
        :param goal_state: maps joint_name to goal position
        :param group_name: if joint_name is not unique, search in this group for matches.
        :param weight:
        :param max_velocity: will be applied to all joints, you should group joint types, e.g., prismatic joints
        :param hard: turns this into a hard constraint.
        """
        self.current_positions = []
        self.lower_limits = []
        self.upper_limits = []
        self.velocity_limits = []
        self.names = []
        self.joint_names = list(sorted(lower_upper_limits.keys()))
        if name is None:
            name = f'{self.__class__.__name__} {self.joint_names}'
        super().__init__(name=name)
        self.max_velocity = max_velocity
        self.weight = weight
        if len(lower_upper_limits) == 0:
            raise GoalInitalizationException(f'Can\'t initialize {self} with no joints.')
        for joint_name, (lower_limit, upper_limit) in lower_upper_limits.items():
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)

            ll_pos, ul_pos = god_map.world.compute_joint_limits(joint_name, Derivatives.position)
            if ll_pos is not None:
                lower_limit = min(ul_pos, max(ll_pos, lower_limit))
                upper_limit = min(ul_pos, max(ll_pos, upper_limit))

            ll_vel, ul_vel = god_map.world.compute_joint_limits(joint_name, Derivatives.velocity)
            velocity_limit = min(ul_vel, max(ll_vel, max_velocity))

            joint: OneDofJoint = god_map.world.joints[joint_name]
            self.names.append(str(joint_name))
            self.current_positions.append(joint.get_symbol(Derivatives.position))
            self.lower_limits.append(lower_limit)
            self.upper_limits.append(upper_limit)
            self.velocity_limits.append(velocity_limit)

        for name, current, lower_limit, upper_limit, velocity_limit in zip(self.names, self.current_positions,
                                                                           self.lower_limits, self.upper_limits,
                                                                           self.velocity_limits):
            if god_map.world.is_joint_continuous(name):
                lower_error = cas.shortest_angular_distance(current, lower_limit)
                upper_error = cas.shortest_angular_distance(current, upper_limit)
            else:
                lower_error = lower_limit - current
                upper_error = upper_limit - current

            self.add_inequality_constraint(name=name,
                                           reference_velocity=velocity_limit,
                                           lower_error=lower_error,
                                           upper_error=upper_error,
                                           weight=self.weight,
                                           task_expression=current)


class JustinTorsoLimit(Task):
    def __init__(self,
                 joint_name: PrefixedName,
                 lower_limit: Optional[float] = None,
                 upper_limit: Optional[float] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        joint: JustinTorso = god_map.world.joints[joint_name]
        self.max_velocity = max_velocity
        self.weight = weight

        current = joint.q3

        if god_map.world.is_joint_continuous(joint_name):
            lower_error = cas.shortest_angular_distance(current, lower_limit)
            upper_error = cas.shortest_angular_distance(current, upper_limit)
        else:
            lower_error = lower_limit - current
            upper_error = upper_limit - current

        god_map.debug_expression_manager.add_debug_expression('torso 4 joint', current)
        god_map.debug_expression_manager.add_debug_expression('torso 2 joint',
                                                              joint.q1.get_symbol(Derivatives.position))
        god_map.debug_expression_manager.add_debug_expression('torso 3 joint',
                                                              joint.q2.get_symbol(Derivatives.position))
        god_map.debug_expression_manager.add_debug_expression('lower_limit', lower_limit)
        god_map.debug_expression_manager.add_debug_expression('upper_limit', upper_limit)

        self.add_inequality_constraint(name=name,
                                       reference_velocity=1,
                                       lower_error=lower_error,
                                       upper_error=upper_error,
                                       weight=self.weight,
                                       task_expression=current)


class JointVelocityLimit(Task):
    def __init__(self,
                 joint_names: List[str],
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False,
                 name: Optional[str] = None):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.weight = weight
        self.max_velocity = max_velocity
        self.hard = hard
        self.joint_names = joint_names
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_names}'
        super().__init__(name=name)

        for joint_name in self.joint_names:
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            joint: OneDofJoint = god_map.world.joints[joint_name]
            current_joint = joint.get_symbol(Derivatives.position)
            try:
                limit_expr = joint.get_limit_expressions(Derivatives.velocity)[1]
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            if self.hard:
                self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity,
                                             lower_slack_limit=0,
                                             upper_slack_limit=0)
            else:
                self.add_velocity_constraint(lower_velocity_limit=-max_velocity,
                                             upper_velocity_limit=max_velocity,
                                             weight=self.weight,
                                             task_expression=current_joint,
                                             velocity_limit=max_velocity,
                                             name=joint_name)


class JointVelocity(Task):
    def __init__(self,
                 joint_names: List[str],
                 vel_goal: float,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 max_velocity: float = 1,
                 hard: bool = False,
                 name: Optional[str] = None):
        """
        Limits the joint velocity of a revolute joint.
        :param joint_name:
        :param group_name: if joint_name is not unique, will search in this group for matches.
        :param weight:
        :param max_velocity: rad/s
        :param hard: turn this into a hard constraint.
        """
        self.weight = weight
        self.vel_goal = vel_goal
        self.max_velocity = max_velocity
        self.hard = hard
        self.joint_names = joint_names
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_names}'
        super().__init__(name=name)

        for joint_name in self.joint_names:
            joint_name = god_map.world.search_for_joint_name(joint_name, group_name)
            joint: OneDofJoint = god_map.world.joints[joint_name]
            current_joint = joint.get_symbol(Derivatives.position)
            try:
                limit_expr = joint.get_limit_expressions(Derivatives.velocity)[1]
                max_velocity = cas.min(self.max_velocity, limit_expr)
            except IndexError:
                max_velocity = self.max_velocity
            self.add_velocity_eq_constraint(velocity_goal=self.vel_goal,
                                            weight=self.weight,
                                            task_expression=current_joint,
                                            velocity_limit=max_velocity,
                                            name=joint_name)


class UnlimitedJointGoal(Task):
    def __init__(self, name: str, joint_name: str, goal_position: float):
        super().__init__(name=name, )
        joint_name = god_map.world.search_for_joint_name(joint_name)
        joint = god_map.world.joints[joint_name]
        joint_symbol = joint.get_symbol(Derivatives.position)
        self.add_position_constraint(expr_current=joint_symbol,
                                     expr_goal=goal_position,
                                     reference_velocity=2,
                                     weight=WEIGHT_BELOW_CA)


class AvoidJointLimits(Task):
    def __init__(self,
                 percentage: float = 15,
                 joint_list: Optional[List[str]] = None,
                 group_name: Optional[str] = None,
                 weight: float = WEIGHT_BELOW_CA,
                 name: Optional[str] = None):
        """
        Calls AvoidSingleJointLimits for each joint in joint_list
        :param percentage:
        :param joint_list: list of joints for which AvoidSingleJointLimits will be called
        :param weight:
        """
        self.joint_list = joint_list
        if name is None:
            name = f'{self.__class__.__name__}/{self.joint_list}'
        super().__init__(name=name)
        self.weight = weight
        self.percentage = percentage
        if joint_list is not None:
            joint_list = [god_map.world.search_for_joint_name(joint_name, group_name) for joint_name in joint_list]
        else:
            if group_name is None:
                joint_list = god_map.world.controlled_joints
            else:
                joint_list = god_map.world.groups[group_name].controlled_joints
        for joint_name in joint_list:
            if god_map.world.is_joint_prismatic(joint_name) or god_map.world.is_joint_revolute(joint_name):
                weight = self.weight
                joint = god_map.world.joints[joint_name]
                joint_symbol = joint.get_symbol(Derivatives.position)
                percentage = self.percentage / 100.
                lower_limit, upper_limit = god_map.world.get_joint_position_limits(joint_name)
                max_velocity = 100
                max_velocity = cas.min(max_velocity,
                                       god_map.world.get_joint_velocity_limits(joint_name)[1])

                joint_range = upper_limit - lower_limit
                center = (upper_limit + lower_limit) / 2.

                max_error = joint_range / 2. * percentage

                upper_goal = center + joint_range / 2. * (1 - percentage)
                lower_goal = center - joint_range / 2. * (1 - percentage)

                upper_err = upper_goal - joint_symbol
                lower_err = lower_goal - joint_symbol

                error = cas.max(cas.abs(cas.min(upper_err, 0)), cas.abs(cas.max(lower_err, 0)))
                weight = weight * (error / max_error)

                self.add_inequality_constraint(reference_velocity=max_velocity,
                                               name=str(joint_name),
                                               lower_error=lower_err,
                                               upper_error=upper_err,
                                               weight=weight,
                                               task_expression=joint_symbol)

