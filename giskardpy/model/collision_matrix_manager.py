from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import product, combinations_with_replacement, chain
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict, Callable

import numpy as np
from line_profiler import profile

from giskardpy.data_types.key_default_dict import KeyDefaultDict
from giskardpy.god_map import god_map
from giskardpy.qp.free_variable import FreeVariable
from semantic_world.connections import ActiveConnection
from semantic_world.degree_of_freedom import DegreeOfFreedom
from semantic_world.robots import AbstractRobot
from semantic_world.world import World
from semantic_world.world_entity import Body


def is_body_controlled(body: Body, world: World) -> bool:
    root_part, tip_part = world.compute_split_chain_of_connections(world.root, body)
    connections = root_part + tip_part
    for c in connections:
        if (isinstance(c, ActiveConnection)
                and c.is_controlled
                and not c.frozen_for_collision_avoidance):
            return True
    return False


def sort_bodies(body_a: Body, body_b: Body) -> Tuple[Body, Body]:
    if body_a.name > body_b.name:
        body_a, body_b = body_b, body_a
    is_body_a_controlled = is_body_controlled(body_a, god_map.world)
    is_body_b_controlled = is_body_controlled(body_b, god_map.world)
    if (not is_body_a_controlled
            and is_body_b_controlled):
        body_a, body_b = body_b, body_a
    return body_a, body_b


@dataclass
class CollisionViewRequest:
    AVOID_COLLISION: int = field(default=0, init=False)
    ALLOW_COLLISION: int = field(default=1, init=False)

    type_: Optional[int] = None
    distance: Optional[float] = None
    view1: Optional[AbstractRobot] = None
    view2: Optional[AbstractRobot] = None

    def __post_init__(self):
        if self.type_ is None:
            self.type_ = self.AVOID_COLLISION
        if self.view1 is None and self.view2 is not None:
            self.view1, self.view2 = self.view2, self.view1
        if self.distance is not None and self.distance < 0:
            raise ValueError(f"Distance must be positive or None, got {self.distance}")
        if self.type_ not in [self.AVOID_COLLISION, self.ALLOW_COLLISION]:
            raise ValueError(f"Unknown type {self.type_}")

    def is_distance_set(self) -> bool:
        return self.distance is not None

    def any_view1(self):
        return self.view1 is None

    def any_view2(self):
        return self.view2 is None

    def is_avoid_collision(self) -> bool:
        return self.type_ == self.AVOID_COLLISION

    def is_allow_collision(self) -> bool:
        return self.type_ == self.ALLOW_COLLISION

    def is_avoid_all_self_collision(self) -> bool:
        return self.is_avoid_collision() and self.view1 == self.view2

    def is_allow_all_self_collision(self) -> bool:
        return self.is_allow_collision() and self.view1 == self.view2

    def is_avoid_all_collision(self) -> bool:
        return self.is_avoid_collision() and self.any_view1() and self.any_view2()

    def is_allow_all_collision(self) -> bool:
        return self.is_allow_collision() and self.any_view1() and self.any_view2()


@dataclass
class CollisionCheck:
    body_a: Body
    body_b: Body
    distance: float

    def __post_init__(self):
        self.body_a, self.body_b = sort_bodies(self.body_a, self.body_b)

    def __hash__(self):
        return hash((self.body_a, self.body_b))

    def __eq__(self, other: CollisionCheck):
        return self.body_a == other.body_a and self.body_b == other.body_b

    def bodies(self) -> Tuple[Body, Body]:
        return self.body_a, self.body_b

    def _validate(self, world: World) -> None:
        """Validates the collision check parameters."""
        if self.distance <= 0:
            raise ValueError(f'Distance must be positive, got {self.distance}')

        if self.body_a == self.body_b:
            raise ValueError(f'Cannot create collision check between the same body "{self.body_a.name}"')

        if not self.body_a.has_collision():
            raise ValueError(f'Body {self.body_a.name} has no collision geometry')

        if not self.body_b.has_collision():
            raise ValueError(f'Body {self.body_b.name} has no collision geometry')

        if self.body_a not in world.bodies_with_enabled_collision:
            raise ValueError(f'Body {self.body_a.name} is not in list of bodies with collisions')

        if self.body_b not in world.bodies_with_enabled_collision:
            raise ValueError(f'Body {self.body_b.name} is not in list of bodies with collisions')

        root_chain, tip_chain = world.compute_split_chain_of_connections(self.body_a, self.body_b)
        if all(not isinstance(c, ActiveConnection) for c in chain(root_chain, tip_chain)):
            raise ValueError(f'Relative pose between {self.body_a.name} and {self.body_b.name} is fixed')

    @classmethod
    def create_and_validate(cls, body_a: Body, body_b: Body, distance: float,
                            world: World) -> CollisionCheck:
        """
        Creates a collision check with additional world-context validation.
        Returns None if the check should be skipped (e.g., bodies are linked).
        """
        collision_check = cls(body_a=body_a, body_b=body_b, distance=distance)
        collision_check._validate(world)
        return collision_check


class DisableCollisionReason(Enum):
    Unknown = -1
    Never = 1
    Adjacent = 2
    Default = 3
    AlmostAlways = 4


@dataclass
class SelfCollisionMatrix:
    # collision_config: CollisionConfig

    @profile
    def compute_self_collision_matrix(self,
                                      body_combinations: Optional[Iterable[Tuple[Body, Body]]] = None,
                                      distance_threshold_zero: float = 0.0,
                                      distance_threshold_always: float = 0.005,
                                      distance_threshold_never_max: float = 0.05,
                                      distance_threshold_never_min: float = -0.02,
                                      distance_threshold_never_range: float = 0.05,
                                      distance_threshold_never_zero: float = 0.0,
                                      number_of_tries_always: int = 200,
                                      almost_percentage: float = 0.95,
                                      number_of_tries_never: int = 10000,
                                      progress_callback: Optional[Callable[[int, str], None]] = None) \
            -> Set[Tuple[Body, Body]]:
        """
        :param progress_callback: a function that is used to display the progress. it's called with a value of 0-100 and
                                    a string representing the current action
        """
        if progress_callback is None:
            progress_callback = lambda value, text: None
        np.random.seed(1337)
        remaining_pairs = set()
        disabled_pairs = copy.copy(self.collision_config.disabled_pairs)
        robot = self.collision_config._robot

        # 0. GENERATE ALL POSSIBLE LINK PAIRS
        if body_combinations is None:
            body_combinations = set(combinations_with_replacement(robot.di, 2))
        for body_a, body_b in list(body_combinations):
            remaining_pairs.add(sort_bodies(body_a, body_b))

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_adjacent(remaining_pairs, robot)
        disabled_pairs.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_default(remaining_pairs,
                                                                                     robot,
                                                                                     distance_threshold_zero)
        disabled_pairs.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_always(
            body_combinations=remaining_pairs,
            robot=robot,
            distance_threshold_always=distance_threshold_always,
            number_of_tries=number_of_tries_always,
            almost_percentage=almost_percentage)
        disabled_pairs.update(matrix_updates)

        # %%
        remaining_pairs, matrix_updates = self.compute_self_collision_matrix_never(
            body_combinations=remaining_pairs,
            robot=robot,
            distance_threshold_never_initial=distance_threshold_never_max,
            distance_threshold_never_min=distance_threshold_never_min,
            distance_threshold_never_range=distance_threshold_never_range,
            distance_threshold_never_zero=distance_threshold_never_zero,
            number_of_tries=number_of_tries_never,
            progress_callback=progress_callback)
        disabled_pairs.update(matrix_updates)

        return disabled_pairs

    def compute_self_collision_matrix_adjacent(self,
                                               body_combinations: Set[Tuple[Body, Body]],
                                               robot: AbstractRobot) \
            -> Tuple[Set[Tuple[Body, Body]], Dict[Tuple[Body, Body], DisableCollisionReason]]:
        """
        Find connecting links and disable all adjacent link collisions
        """
        disabled_pairs = self.collision_config.compute_uncontrolled_body_pairs()
        black_list = {pair: DisableCollisionReason.Adjacent for pair in disabled_pairs}
        remaining_pairs = body_combinations.difference(disabled_pairs)
        return remaining_pairs, black_list

    def compute_self_collision_matrix_default(self,
                                              body_combinations: Set[Tuple[Body, Body]],
                                              robot: AbstractRobot,
                                              distance_threshold_zero: float) \
            -> Tuple[Set[Tuple[Body, Body]], Dict[
                Tuple[Body, Body], DisableCollisionReason]]:
        """
        Disable link pairs that are in collision in default state
        """
        with god_map.world.reset_state_context():
            self.set_default_joint_state(robot)
            self_collision_matrix = {}
            remaining_pairs = copy.copy(body_combinations)
            for link_a, link_b, _ in god_map.collision_scene.collision_detector.find_colliding_combinations(
                    remaining_pairs, distance_threshold_zero, True):
                link_combination = god_map.world.sort_links(link_a, link_b)
                remaining_pairs.remove(link_combination)
                self_collision_matrix[link_combination] = DisableCollisionReason.Default
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_always(self,
                                             body_combinations: Set[Tuple[Body, Body]],
                                             robot: AbstractRobot,
                                             distance_threshold_always: float,
                                             number_of_tries: int = 200,
                                             almost_percentage: float = 0.95) \
            -> Tuple[Set[Tuple[Body, Body]], Dict[
                Tuple[Body, Body], DisableCollisionReason]]:
        """
        Disable link pairs that are (almost) always in collision.
        """
        if number_of_tries == 0:
            return body_combinations, {}
        with god_map.world.reset_state_context():
            self_collision_matrix = {}
            remaining_pairs = copy.copy(body_combinations)
            counts: DefaultDict[Tuple[Body, Body], int] = defaultdict(int)
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(robot)
                for link_a, link_b, _ in god_map.collision_scene.collision_detector.find_colliding_combinations(
                        remaining_pairs, distance_threshold_always,
                        True):
                    link_combination = sort_bodies(link_a, link_b)
                    counts[link_combination] += 1
            for link_combination, count in counts.items():
                if count > number_of_tries * almost_percentage:
                    remaining_pairs.remove(link_combination)
                    self_collision_matrix[link_combination] = DisableCollisionReason.AlmostAlways
        return remaining_pairs, self_collision_matrix

    def compute_self_collision_matrix_never(self,
                                            body_combinations: Set[Tuple[Body, Body]],
                                            robot: AbstractRobot,
                                            distance_threshold_never_initial: float,
                                            distance_threshold_never_min: float,
                                            distance_threshold_never_range: float,
                                            distance_threshold_never_zero: float,
                                            number_of_tries: int = 10000,
                                            progress_callback: Optional[Callable[[int, str], None]] = None) \
            -> Tuple[Set[Tuple[Body, Body]], Dict[
                Tuple[Body, Body], DisableCollisionReason]]:
        """
        Disable link pairs that are never in collision.
        """
        if number_of_tries == 0:
            return body_combinations, {}
        with god_map.world.reset_state_context():
            one_percent = number_of_tries // 100
            self_collision_matrix = {}
            remaining_pairs = copy.copy(body_combinations)
            update_query = True
            distance_ranges: Dict[Tuple[Body, Body], Tuple[float, float]] = {}
            once_without_contact = set()
            for try_id in range(int(number_of_tries)):
                self.set_rnd_joint_state(robot)
                contacts = god_map.collision_scene.collision_detector.find_colliding_combinations(remaining_pairs,
                                                                                                  distance_threshold_never_initial,
                                                                                                  update_query)
                update_query = False
                contact_keys = set()
                for link_a, link_b, distance in contacts:
                    key = god_map.world.sort_links(link_a, link_b)
                    contact_keys.add(key)
                    if key in distance_ranges:
                        old_min, old_max = distance_ranges[key]
                        distance_ranges[key] = (min(old_min, distance), max(old_max, distance))
                    else:
                        distance_ranges[key] = (distance, distance)
                    if distance < distance_threshold_never_min:
                        remaining_pairs.remove(key)
                        update_query = True
                        del distance_ranges[key]
                once_without_contact.update(remaining_pairs.difference(contact_keys))
                if try_id % one_percent == 0:
                    progress_callback(try_id // one_percent, 'checking collisions')
            never_in_contact = remaining_pairs
            for key in once_without_contact:
                if key in distance_ranges:
                    old_min, old_max = distance_ranges[key]
                    distance_ranges[key] = (old_min, np.inf)
            for key, (min_, max_) in list(distance_ranges.items()):
                if (max_ - min_) < distance_threshold_never_range or min_ > distance_threshold_never_zero:
                    never_in_contact.add(key)
                    del distance_ranges[key]

            for combi in never_in_contact:
                self_collision_matrix[combi] = DisableCollisionReason.Never
        return remaining_pairs, self_collision_matrix

    def set_joint_state_to_zero(self) -> None:
        for free_variable in god_map.world.degrees_of_freedoms:
            god_map.world.state[free_variable].position = 0
        god_map.world.notify_state_change()

    def set_default_joint_state(self, robot: AbstractRobot):
        for connection in robot.connections:
            if not isinstance(connection, ActiveConnection):
                continue
            dof: DegreeOfFreedom
            for dof in connection.active_dofs:
                if dof.has_position_limits():
                    lower_limit = dof.lower_limits.position
                    upper_limit = dof.upper_limits.position
                    god_map.world.state[dof.name].position = (upper_limit + lower_limit) / 2
                else:
                    god_map.world.state[dof.name].position = 0
        god_map.world.notify_state_change()

    @profile
    def set_rnd_joint_state(self, robot: AbstractRobot):
        for connection in robot.connections:
            if not isinstance(connection, ActiveConnection):
                continue
            dof: FreeVariable
            for dof in connection.active_dofs:
                if dof.has_position_limits():
                    lower_limit = dof.lower_limits.position
                    upper_limit = dof.upper_limits.position
                    rnd_position = (np.random.random() * (upper_limit - lower_limit)) + lower_limit
                else:
                    rnd_position = np.random.random() * np.pi * 2
                god_map.world.state[dof.name].position = rnd_position
        god_map.world.notify_state_change()


@dataclass
class CollisionMatrixManager:
    """
    Handles all matrix related operations for multiple robots.
    """
    world: World
    robots: Set[AbstractRobot]

    added_checks: Set[CollisionCheck] = field(default_factory=set)

    collision_requests: List[CollisionViewRequest] = field(default_factory=list)
    """
    Motion goal specific requests for collision avoidance checks.
    Can overwrite the thresholds and add additional disabled bodies/pairs.
    """

    # external_thresholds: Dict[Body, CollisionAvoidanceThreshold] = field(init=False)
    # self_thresholds: Dict[Tuple[Body, Body], CollisionAvoidanceThreshold] = field(init=False)
    # """
    # Thresholds used for collision avoidance checks.
    # They already include collision requests
    # """

    # disabled_bodies: Set[Body] = field(default_factory=set)
    # disabled_pairs: Set[Tuple[Body, Body]] = field(default_factory=set)
    # """
    # Bodies disables for collision avoidance checks.
    # Include collision configs for the robot and collision requests
    # """

    # def __post_init__(self):
    #     self.compute_thresholds()
    #     self.combine_collision_configs()

    # def compute_thresholds(self):
    #     def get_external_threshold(body: Body) -> CollisionAvoidanceThreshold:
    #         for robot in self.robots:
    #             # if body is part of the robot, use robot threshold
    #             if body in robot.bodies_with_collisions:
    #                 return robot.collision_config.external_avoidance_threshold[body]
    #         raise KeyError(f'Could not find collision avoidance threshold for body "{body.name}". '
    #                        f'Body is not part of any robot.')
    #
    #     self.external_thresholds = KeyDefaultDict(get_external_threshold)
    #
    #     def get_self_threshold(key: Tuple[Body, Body]) -> CollisionAvoidanceThreshold:
    #         body_a, body_b = key
    #         soft_a = body_a._collision_config.buffer_zone_distance
    #         soft_b = body_b._collision_config.buffer_zone_distance
    #         if soft_a is None and soft_b is None:
    #             raise ValueError(f'Could not find collision avoidance threshold for body pair '
    #                              f'"{body_a.name}", "{body_b.name}". '
    #                              f'Neither body has a soft threshold set.')
    #         if soft_a is None:
    #             soft = soft_b
    #         elif soft_b is None:
    #             soft = soft_a
    #         else:
    #             soft = max(soft_a, soft_b)
    #         return CollisionAvoidanceThreshold(
    #             soft_threshold=soft,
    #             hard_threshold=max(body_a._collision_config.violated_distance,
    #                                body_b._collision_config.violated_distance))
    #
    #     self.self_thresholds = KeyDefaultDict(get_self_threshold)

    # def combine_collision_configs(self):
    #     for robot in self.robots:
    #         self.disabled_pairs.update(robot.collision_config.disabled_pairs)
    #         self.disabled_bodies.update(robot.collision_config.disabled_bodies)

    def compute_collision_matrix(self) -> Set[CollisionCheck]:
        """
        Parses the collision requrests and (temporary) collision configs in the world
        to create a set of collision checks.
        """
        collision_matrix: Set[CollisionCheck] = set()
        for collision_request in self.collision_requests:
            if collision_request.any_view1():
                view_1_bodies = self.world.bodies_with_enabled_collision
            else:
                view_1_bodies = collision_request.view1.bodies_with_enabled_collision
            if collision_request.any_view2():
                view2_bodies = self.world.bodies_with_enabled_collision
            else:
                view2_bodies = collision_request.view2.bodies_with_enabled_collision
            disabled_pairs = self.world.disabled_collision_pairs
            for body1 in view_1_bodies:
                for body2 in view2_bodies:
                    (robot_body, env_body) = sort_bodies(body1, body2)
                    if (robot_body, env_body) in disabled_pairs:
                        continue
                    if collision_request.distance is None:
                        distance = max(robot_body.collision_config.buffer_zone_distance or 0.0,
                                       env_body.collision_config.buffer_zone_distance or 0.0)
                    else:
                        distance = collision_request.distance
                    if collision_request.is_allow_collision():
                        collision_check = CollisionCheck(body_a=robot_body,
                                                         body_b=env_body,
                                                         distance=0)
                    else:
                        collision_check = CollisionCheck.create_and_validate(body_a=robot_body,
                                                                             body_b=env_body,
                                                                             distance=distance,
                                                                             world=god_map.world)
                    if collision_request.is_allow_collision():
                        if collision_check in collision_matrix:
                            collision_matrix.remove(collision_check)
                    if collision_request.is_avoid_collision():
                        if collision_request.is_distance_set():
                            collision_matrix.add(collision_check)
                        else:
                            collision_matrix.add(collision_check)
        return collision_matrix

    # def create_default_thresholds(self):
    #     max_distances = {}
    #     for robot_name in self.robot_names:
    #         collision_avoidance_config = god_map.collision_scene.collision_avoidance_configs[robot_name]
    #         external_distances = collision_avoidance_config.external_collision_avoidance
    #         self_distances = collision_avoidance_config.self_collision_avoidance
    #
    #         # override max distances based on external distances dict
    #         for robot in god_map.collision_scene.robots:
    #             for body in robot.bodies_with_collisions:
    #                 try:
    #                     controlled_parent_joint = god_map.world.get_controlled_parent_joint_of_link(body)
    #                 except KeyError as e:
    #                     continue  # this happens when the root link of a robot has a collision model
    #                 distance = external_distances[controlled_parent_joint].buffer_zone_distance
    #                 for child_link_name in god_map.world.get_directly_controlled_child_links_with_collisions(
    #                         controlled_parent_joint):
    #                     max_distances[child_link_name] = distance
    #
    #         for link_name in self_distances:
    #             distance = self_distances[link_name].buffer_zone_distance
    #             if link_name in max_distances:
    #                 max_distances[link_name] = max(distance, max_distances[link_name])
    #             else:
    #                 max_distances[link_name] = distance
    #
    #     return max_distances

    def add_collision_check(self, body_a: Body, body_b: Body, distance: float):
        """
        Tell Giskard to check this collision, even if it got disabled through other means such as allow_all_collisions.
        """
        check = CollisionCheck.create_and_validate(body_a=body_a, body_b=body_b, distance=distance, world=god_map.world)
        if check in self.added_checks:
            raise ValueError(f"Collision check {check} already added")
        self.added_checks.add(check)

    def parse_collision_requests(self, collision_goals: List[CollisionViewRequest]) -> None:
        """
        Resolve an incoming list of collision goals into collision checks.
        1. remove redundancy
        2. remove entries where view1 or view2 are none
        :param collision_goals:
        :return:
        """
        for i, collision_goal in enumerate(reversed(collision_goals)):
            if collision_goal.is_avoid_all_collision():
                # remove everything before the avoid all
                collision_goals = collision_goals[len(collision_goals) - i - 1:]
                break
            if collision_goal.is_allow_all_collision():
                # remove everything before the allow all, including the allow all
                collision_goals = collision_goals[len(collision_goals) - i:]
                break
        else:
            # put an avoid all at the front
            collision_goal = CollisionViewRequest()
            collision_goal.type_ = CollisionViewRequest.AVOID_COLLISION
            collision_goal.distance = None
            collision_goals.insert(0, collision_goal)

        self.collision_requests = list(collision_goals)

    def apply_world_model_updates(self) -> None:
        # self.update_self_collision_matrices_for_attached_bodies()
        # self.disable_non_robot_collisions()
        # self.disable_env_with_unmovable_bodies()
        pass

    # def update_self_collision_matrices_for_attached_bodies(self):
    #     for robot in self.robots:
    #         attached_links = [link for link in robot.bodies_with_collisions
    #                           if (link, link) not in robot.collision_config.disabled_pairs]
    #         body_combinations = set(product(attached_links, robot.bodies_with_collisions))
    #         scm = SelfCollisionMatrix(collision_config=robot.collision_config)
    #         if body_combinations:
    #             disabled_pairs = scm.compute_self_collision_matrix(body_combinations=body_combinations)
    #             robot.collision_config.disabled_pairs.update(disabled_pairs)



    def get_non_robot_bodies(self) -> Set[Body]:
        return set(self.world.bodies_with_enabled_collision).difference(self.get_robot_bodies())

    def get_robot_bodies(self) -> Set[Body]:
        robot_bodies = set()
        for robot in self.robots:
            robot_bodies.update(robot.bodies_with_enabled_collision)
        return robot_bodies

    # def disable_env_with_unmovable_bodies(self) -> None:
    #     for robot in self.robots:
    #         for body_a in robot.unmovable_bodies_with_collision:
    #             for body_b in self.get_non_robot_bodies():
    #                 body_a, body_b = sort_bodies(body_a, body_b)
    #                 self.disabled_pairs.add((body_a, body_b))

    # def disable_non
    #     for group in god_map.world.groups.values():
    #         if group.name not in self.robot_names:
    #             for link_a, link_b in set(combinations_with_replacement(group.link_names_with_collisions, 2)):
    #                 key = god_map.world.sort_links(link_a, link_b)
    #                 self.self_collision_matrix[key] = DisableCollisionReason.Unknown
