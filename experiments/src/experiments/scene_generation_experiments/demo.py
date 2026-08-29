from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from coraplex.robot_plans.actions.base import ActionDescription
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.misc import MoveToReach
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from experiments.scene_generation_experiments.exceptions import (
    MissingShelfCabinetError,
    NoFittingObjectError,
    UnreachableShelfError,
)
from experiments.scene_generation_experiments.shelf_placement import (
    layer_named,
    mode_query,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _load_or_train_shelf_model,
    generate_shelf_with_arbitrary_objects,
    visualize_spawned_shelf,
    VisualizationBackend,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import an, entity, variable
from experiments.scene_generation_experiments.processed_database import (
    load_objects_of_types,
    _processed_database_session,
)
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    rclpy_node,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_theme_shelf_query,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeVisualization,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_publisher import (
    SpatialTypePublisher,
)
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    ObjectType,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Floor,
    Table,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.world_entity import Body


# %% the floor the robot drives on


def floor_point(world: World, point: Point3, floor: Floor) -> Point3:
    """
    Project *point* onto the height at which the floor's free space is described.

    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasSupportingSurface.calculate_free_space`
    decomposes its graph as a thin slab at the supporting surface's own height, so a
    point only falls inside a node of that graph once it sits at that height.

    :param world: The world both the point and the floor belong to.
    :param point: The point to project, in any frame.
    :param floor: The floor whose free space the point is meant for.
    :return: The point in the floor's supporting-surface frame, at that surface's
        height.
    """
    point_on_surface = world.transform(point, floor.supporting_surface)
    return Point3(
        float(point_on_surface.x),
        float(point_on_surface.y),
        0.0,
        reference_frame=floor.supporting_surface,
    )


def object_annotation(world: World, body: Body) -> HasRootBody:
    """
    Resolve the semantic annotation *body* was spawned with.

    :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGObject2D.spawn`
    registers a :class:`~semantic_digital_twin.semantic_annotations.natural_language.
    NaturalLanguageWithTypeDescription` for the body it creates, but hands the plain
    :class:`Body` back rather than that annotation. Actions such as
    :class:`~coraplex.robot_plans.actions.core.pick_up.PickUpAction` are typed against
    the annotation, not the body it wraps, so a caller holding only the body needs
    this to look the annotation back up.

    :param world: The world *body* was spawned into.
    :param body: The body to find the annotation of.
    :return: The body's own semantic annotation.
    """
    return an(
        entity(
            semantic_annotation := variable(
                HasRootBody, domain=world.semantic_annotations
            )
        ).where(semantic_annotation.root == body)
    ).first()


def robot_shelf_standing_point(
    spawned_shelf: EGShelf, placed_object: EGObject2D
) -> Point3:
    """
    Where the robot should stand, clear of *spawned_shelf*'s open face, to reach
    *placed_object*'s spot on it.

    The standoff sits outside the corpus, along the axis its face opens on;
    *placed_object*'s own y coordinate already reads directly onto the corpus's y axis
    (see :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
    object_local_pose`), so it carries over unchanged.

    :param spawned_shelf: The shelf to stand in front of.
    :param placed_object: The object whose on-shelf position decides where along the
        open face the robot stands.
    :return: The standing point, in the shelf corpus's own frame.
    """
    footprint = spawned_shelf.corpus_footprint
    standoff = Cabinet.hole_direction * (footprint.x / 2 + 0.5)
    return Point3(
        float(standoff.x),
        float(placed_object.pose.y),
        reference_frame=spawned_shelf.corpus,
    )


def path_to_shelf(
    world: World,
    floor: Floor,
    robot: AbstractRobot,
    standing_point: Point3,
    free_space: GraphOfBoundingBoxes,
) -> list[Pose]:
    """
    Find the navigation goals leading the robot from where it stands to the ground in
    front of the shelf.

    The route comes out of *free_space*, a graph of convex sets over the ground the
    floor's occupants leave, so it goes around what stands on the floor rather than
    through it. Each goal faces the one it leads to, so the robot drives forwards along
    the route. The last leg is left out: :func:`move_to_reach_shelf` drives it as part
    of reaching, so a route with nothing in the way yields no goals at all.

    :param world: The world the robot drives through.
    :param floor: The floor the route crosses.
    :param robot: The robot to route, from wherever it currently stands.
    :param standing_point: Where the route ends.
    :param free_space: The floor's free space, as computed by
        :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasSupportingSurface.calculate_free_space`.
        Passed in rather than computed here so a caller building several routes, or
        move-to-reach actions, for the same floor computes it only once.
    :raises PointOccupiedError: If the robot or *standing_point* is not on free floor.
    :raises UnreachableShelfError: If the floor's free space connects the two nowhere.
    :return: The navigation goals, in the world's root frame.
    """
    start = floor_point(world, robot.root.global_pose.to_position(), floor)
    goal = floor_point(world, standing_point, floor)
    waypoints = free_space.path_from_to(start, goal)
    if waypoints is None:
        raise UnreachableShelfError(
            walking_distance=float(start.euclidean_distance(goal)),
            floor_occupants=[str(occupant.root.name) for occupant in floor.objects],
        )

    goals = []
    for waypoint, next_waypoint in zip(waypoints[1:-1], waypoints[2:]):
        here = world.transform(waypoint, world.root)
        onwards = world.transform(next_waypoint, world.root)
        goals.append(
            Pose.from_xyz_rpy(
                x=float(here.x),
                y=float(here.y),
                yaw=math.atan2(
                    float(onwards.y) - float(here.y), float(onwards.x) - float(here.x)
                ),
                reference_frame=world.root,
            )
        )
    return goals


@dataclasses.dataclass
class ShelfTidyingAction(ActionDescription):
    floor: Floor = dataclasses.field(default=None)
    table: Table = dataclasses.field(default=None)
    obj: Body = dataclasses.field(default=None)
    obj_goal_pose: Pose = dataclasses.field(default=None)
    arm: Arms = dataclasses.field(default=None)
    grasp_description: GraspDescription = dataclasses.field(default=None)
    navigation_goals: List[Pose] = dataclasses.field(default=None)
    shelf: EGShelf = dataclasses.field(default=None)
    layer_name: str = dataclasses.field(default=None)
    placed_obj: EGObject2D = dataclasses.field(default=None)

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                self.move_to_reach_book(),
                PickUpAction(
                    object_designator=object_annotation(self.context.world, self.obj),
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
                *[NavigateAction(goal) for goal in self.navigation_goals],
                self.move_to_reach_shelf(),
                PlaceAction(
                    object_designator=self.obj,
                    target_location=self.obj_goal_pose,
                    arm=self.arm,
                ),
            ],
            self.context,
        )

    def move_to_reach_book(self) -> MoveToReach:
        """
        Build a move-to-reach action for a pose, clear of the table, from which the
        robot could pick up *book*.

        The standing pose is a fixed standoff outside the table's near edge -- the side
        the pre-grasp pose below already approaches *book* from -- and is checked
        against *free_space* before use.

        :return: A concrete move-to-reach action.
        :raises PointOccupiedError: If the computed standing point is not free.
        """
        world = self.context.world

        min_p = self.obj.collision.min_point
        max_p = self.obj.collision.max_point
        pre_grasp_pose = Pose.from_xyz_rpy(
            x=min_p.x - 0.05,
            y=(min_p.y + max_p.y) / 2,
            z=(min_p.z + max_p.z) / 2,
            reference_frame=self.obj,
        )

        table_min, table_max = self.table.min_max_points
        standing_clearance = 0.5
        standing_point_on_table = Point3(
            float(table_min.x) - standing_clearance,
            (float(table_min.y) + float(table_max.y)) / 2,
            0.0,
            reference_frame=self.table.root,
        )
        standing_point_on_floor = floor_point(
            world, standing_point_on_table, self.floor
        )

        standing_offset = world.transform(standing_point_on_floor, self.obj)

        return MoveToReach(
            target_pose_offset_robot=Pose2D(
                x=float(standing_offset.x),
                y=float(standing_offset.y),
                yaw=0.0,
                reference_frame=self.obj,
            ),
            hip_rotation=0.0,
            target_pose_end_effector=pre_grasp_pose,
            grasp_description=self.grasp_description,
        )

    def move_to_reach_shelf(self) -> MoveToReach:
        """
        Build a move-to-reach action that drives in front of the shelf's open face and
        reaches in to where *placed_object* goes.

        The reach pose keeps the corpus's own orientation instead of the placement's, so
        the arm goes in along the axis the shelf opens on and the robot ends up outside
        the open face rather than wherever the placement happens to be turned. What the
        object itself is turned to is left to the place action that follows.
        :return: A concrete move-to-reach action.
        """
        layer = layer_named(self.shelf, self.layer_name)
        slab_top_height = next(
            geometry.slab_top_height
            for shelf_layer, geometry in zip(
                self.shelf.layers, self.shelf.layer_geometries()
            )
            if shelf_layer is layer
        )
        placement_position = self.shelf.object_local_pose(
            self.placed_obj, slab_top_height, self.shelf.corpus
        ).to_position()
        standing_point = robot_shelf_standing_point(self.shelf, self.placed_obj)
        return MoveToReach(
            target_pose_offset_robot=Pose2D(
                x=float(standing_point.x) - float(placement_position.x),
                y=float(standing_point.y) - float(placement_position.y),
                yaw=0.0,
                reference_frame=self.shelf.corpus,
            ),
            hip_rotation=0.0,
            target_pose_end_effector=Pose.from_xyz_rpy(
                x=float(placement_position.x),
                y=float(placement_position.y),
                z=float(placement_position.z),
                reference_frame=self.shelf.corpus,
            ),
            grasp_description=self.grasp_description,
        )


if __name__ == "__main__":
    # DB Connection
    session = _processed_database_session()

    with rclpy_node() as node:
        world = World.create_with_root_body()

        # PREPARATION AND MODEL LOADING
        shelf_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, y=0.0, z=0.0)
        robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.0, y=0.0, z=0.0)
        floor_scale = Scale(x=8.0, y=8.0, z=0.02)
        table_scale = Scale(x=0.9, y=0.6, z=0.2)
        table_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=-1.5, z=table_scale.z / 2
        )
        model_path = Path(__file__).parent / "models" / "arbitrary_shelf_rspn.json"
        trained_model = _load_or_train_shelf_model(model_path, session)

        # create query
        query = build_theme_shelf_query(
            ObjectType.BOOK,
            [3, 3, 3],
        )

        # SHELF
        spawned_shelf = generate_shelf_with_arbitrary_objects(
            query, trained_model, session, world=world, parent_T_self=shelf_pose
        )
        viz_marker = visualize_spawned_shelf(
            node, world, visualization_backend=VisualizationBackend.FOXGLOVE
        )

        RobotSpecification(
            semantic_annotation_type=HSRB, odom_T_robot_start=robot_pose
        ).spawn(world)

        with world.modify_world():
            # Box is centered on its pose; drop it by half its thickness so the
            # top surface sits at z=0, level with the robot's and shelf's base.
            floor: Floor = Floor.create_with_new_body_in_world(
                name="floor",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-floor_scale.z / 2
                ),
                scale=floor_scale,
            )
            table = Table.create_with_new_body_in_world(
                name="table",
                world=world,
                world_root_T_self=table_pose,
                scale=table_scale,
            )
            floor.calculate_supporting_surface()
            floor.add_object(table)
            floor.add_object(spawned_shelf.annotation)

        book_candidates = _get_source_ids_for_objects(
            load_objects_of_types(session, {ObjectType.BOOK})
        )
        book_candidates_standing = [
            candidate
            for candidate in book_candidates
            if candidate.native_extents is not None
            and candidate.native_extents[2] == max(candidate.native_extents)
        ]
        # The book has to be one this shelf could take back: its layers are spaced
        # evenly across the drawn corpus, so a four-layer shelf leaves under 0.2 m
        # above each slab, while a standing book scan is 0.25 m tall on average.
        layer_geometries = spawned_shelf.layer_geometries()
        tallest_layer_room = max(
            geometry.maximum_object_extents.z for geometry in layer_geometries
        )
        book_candidates_fitting = [
            candidate
            for candidate in book_candidates_standing
            if candidate.native_extents[2] <= tallest_layer_room
        ]
        if not book_candidates_fitting:
            raise NoFittingObjectError(
                object_type=ObjectType.BOOK.value,
                shortest_height=min(
                    candidate.native_extents[2]
                    for candidate in book_candidates_standing
                ),
                layer_rooms=[
                    geometry.maximum_object_extents.z for geometry in layer_geometries
                ],
            )
        book_candidate = random.choice(book_candidates_fitting)
        book_extents = book_candidate.native_extents
        # EGShelf.object_local_pose maps an object's own scale.x/y straight onto the
        # shelf corpus's x/y axes at yaw 0, and Cabinet.hole_direction is along the
        # corpus's x axis -- so a book placed at yaw 0 shows its spine to the shelf's
        # open face only if its thicker (page-width) extent is on scale.x (matching
        # the corpus's depth) and its thinner (spine-width) extent is on scale.y
        # (matching the corpus's face), regardless of which native extent is which.
        book_thin_extent = min(book_extents[0], book_extents[1])
        book_thick_extent = max(book_extents[0], book_extents[1])
        table_edge_margin = 0.02
        book = EGObject2D(
            object_type=ObjectType.BOOK,
            scale=Scale(x=book_thick_extent, y=book_thin_extent, z=book_extents[2]),
            pose=Pose2D(),
            source_id=book_candidate.source_id,
            name="demo_book",
        )
        book_body = book.spawn(
            world,
            parent=table.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.31, y=0.15, z=table_scale.z / 2, reference_frame=table.root
            ),
            mesh_path=book_candidate.scene_dir,
        )

        context = Context.from_world(world, query_backend=ProbabilisticBackend())
        # The fitted circuit's own yaw preference is close to uniform (see
        # project_rspn_placement_constraints memory), so its mode search cannot be
        # trusted to pick a physically sensible orientation -- pin the book spine-out
        # (yaw 0, per the scale convention set above) instead of asking it.
        placed_object, layer_name = mode_query(
            spawned_shelf,
            trained_model.relational_probabilistic_circuit,
            book,
            held_object_yaw=0.0,
        )
        goal_layer = layer_named(spawned_shelf, layer_name)
        position_x = float(placed_object.pose.x)
        position_y = float(placed_object.pose.y)
        orientation_yaw = float(placed_object.pose.yaw)
        pose2d = Pose2D(
            position_x,
            position_y,
            orientation_yaw,
            reference_frame=goal_layer.annotation.root,
        )
        object_goal_pose_in_map = world.transform(pose2d.to_pose(), world.root)
        free_space = floor.calculate_free_space()
        standing_pose = robot_shelf_standing_point(spawned_shelf, placed_object)

        navigation_goals = path_to_shelf(
            world, floor, context.robot, standing_pose, free_space
        )

        arm = Arms.LEFT
        grasp_description = GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=context.robot.end_effector,
            rotate_gripper=False,
        )
        shelf_tidying = ShelfTidyingAction(
            floor=floor,
            table=table,
            obj=book_body,
            obj_goal_pose=object_goal_pose_in_map,
            arm=arm,
            grasp_description=grasp_description,
            navigation_goals=navigation_goals,
            shelf=spawned_shelf,
            layer_name=layer_name,
            placed_obj=placed_object,
        )

        try:
            with simulated_robot():
                execute_single(shelf_tidying, context).perform()
        finally:
            pass
