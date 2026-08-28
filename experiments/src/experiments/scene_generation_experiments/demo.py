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


def move_to_reach_book(
    context: Context, floor: Floor, table: Table, book: Body
) -> MoveToReach:
    """
    Build a move-to-reach action for a pose, clear of the table, from which the robot
    could pick up *book*.

    The standing pose is a fixed standoff outside the table's near edge -- the side
    the pre-grasp pose below already approaches *book* from -- and is checked
    against the floor's free space, which
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasSupportingSurface.calculate_free_space`
    computes excluding the table's own footprint, before use.

    :param context: The context whose world and robot the action is built for.
    :param floor: The floor the robot stands on. Its supporting surface and occupant
        list must already be populated (via ``calculate_supporting_surface()`` and
        ``add_object()``), since both feed the free-space check.
    :param table: The table the robot must stand clear of, and *book* rests on.
    :param book: The book to reach for.
    :return: A concrete move-to-reach action.
    :raises PointOccupiedError: If the computed standing point is not free.
    """
    world = context.world

    min_p = book.collision.min_point
    max_p = book.collision.max_point
    pre_grasp_pose = Pose.from_xyz_rpy(
        x=min_p.x - 0.05,
        y=(min_p.y + max_p.y) / 2,
        z=(min_p.z + max_p.z) / 2,
        reference_frame=book,
    )

    table_min, table_max = table.min_max_points
    standing_clearance = 0.5
    standing_point_on_table = Point3(
        float(table_min.x) - standing_clearance,
        (float(table_min.y) + float(table_max.y)) / 2,
        0.0,
        reference_frame=table.root,
    )
    standing_point_on_floor = floor_point(world, standing_point_on_table, floor)

    if floor.calculate_free_space().node_of_point(standing_point_on_floor) is None:
        raise PointOccupiedError(world.transform(standing_point_on_floor, world.root))

    standing_offset = world.transform(standing_point_on_floor, book)

    return MoveToReach(
        target_pose_offset_robot=Pose2D(
            x=float(standing_offset.x),
            y=float(standing_offset.y),
            yaw=0.0,
            reference_frame=book,
        ),
        hip_rotation=0.0,
        target_pose_end_effector=pre_grasp_pose,
        grasp_description=GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=context.robot.end_effector,
            rotate_gripper=False,
        ),
    )


# %% putting the book on the shelf
def shelf_cabinet(world: World, spawned_shelf: EGShelf) -> Cabinet:
    """
    Look up the cabinet annotation the shelf's corpus was spawned as.

    :param world: The world the shelf stands in.
    :param spawned_shelf: The shelf whose corpus is looked up.
    :raises MissingShelfCabinetError: If the world holds no cabinet on that corpus.
    :return: The annotation rooted at the corpus.
    """
    cabinets = [
        cabinet
        for cabinet in world.get_semantic_annotations_by_type(Cabinet)
        if cabinet.root is spawned_shelf.corpus
    ]
    if not cabinets:
        raise MissingShelfCabinetError(corpus_name=str(spawned_shelf.corpus.name))
    return cabinets[0]


def move_to_reach_shelf(
    spawned_shelf: EGShelf,
    placed_object: EGObject2D,
    layer_name: str,
) -> MoveToReach:
    """
    Build a move-to-reach action that drives in front of the shelf's open face and
    reaches in to where *placed_object* goes.

    The reach pose keeps the corpus's own orientation instead of the placement's, so the
    arm goes in along the axis the shelf opens on and the robot ends up outside the open
    face rather than wherever the placement happens to be turned. What the object itself
    is turned to is left to the place action that follows.

    :param spawned_shelf: The shelf to reach into.
    :param placed_object: The object :func:`~experiments.scene_generation_experiments.
        shelf_placement.mode_query` placed, with its pose filled in.
    :param layer_name: The layer :func:`~experiments.scene_generation_experiments.
        shelf_placement.mode_query` placed *placed_object* onto, resolved back to the
        real layer here -- the same resolution the place action's own target pose uses,
        so both agree on which layer the arm reaches for.
    :return: A concrete move-to-reach action.
    """
    layer = layer_named(spawned_shelf, layer_name)
    slab_top_height = next(
        geometry.slab_top_height
        for shelf_layer, geometry in zip(
            spawned_shelf.layers, spawned_shelf.layer_geometries()
        )
        if shelf_layer is layer
    )
    placement_position = spawned_shelf.object_local_pose(
        placed_object, slab_top_height, spawned_shelf.corpus
    ).to_position()
    footprint = spawned_shelf.corpus_footprint
    standoff = Cabinet.hole_direction * (footprint.x / 2 + 0.5)
    standing_pose = Point3(
        float(standoff.x),
        float(placement_position.y),
        reference_frame=spawned_shelf.corpus,
    )
    return MoveToReach(
        target_pose_offset_robot=Pose2D(
            x=float(standing_pose.x) - float(placement_position.x),
            y=float(standing_pose.y) - float(placement_position.y),
            yaw=0.0,
            reference_frame=spawned_shelf.corpus,
        ),
        hip_rotation=0.0,
        target_pose_end_effector=Pose.from_xyz_rpy(
            x=float(placement_position.x),
            y=float(placement_position.y),
            z=float(placement_position.z),
            reference_frame=spawned_shelf.corpus,
        ),
        grasp_description=GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            end_effector=context.robot.end_effector,
            rotate_gripper=False,
        ),
    )


def path_to_shelf(
    world: World, floor: Floor, robot: AbstractRobot, standing_point: Point3
) -> list[Pose]:
    """
    Find the navigation goals leading the robot from where it stands to the ground in
    front of the shelf.

    The route comes out of the floor's free space, a graph of convex sets over the
    ground its occupants leave, so it goes around what stands on the floor rather than
    through it. Each goal faces the one it leads to, so the robot drives forwards along
    the route. The last leg is left out: :func:`move_to_reach_shelf` drives it as part
    of reaching, so a route with nothing in the way yields no goals at all.

    :param world: The world the robot drives through.
    :param floor: The floor the route crosses. Its occupant list decides what the route
        avoids, so everything standing on it must have been added.
    :param robot: The robot to route, from wherever it currently stands.
    :param standing_point: Where the route ends.
    :raises PointOccupiedError: If the robot or *standing_point* is not on free floor.
    :raises UnreachableShelfError: If the floor's free space connects the two nowhere.
    :return: The navigation goals, in the world's root frame.
    """
    start = floor_point(world, robot.root.global_pose.to_position(), floor)
    goal = floor_point(world, standing_point, floor)
    waypoints = floor.calculate_free_space().path_from_to(start, goal)
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
                move_to_reach_book(self.context, self.floor, self.table, self.obj),
                PickUpAction(
                    object_designator=self.obj,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
                *[NavigateAction(goal) for goal in self.navigation_goals],
                move_to_reach_shelf(
                    self.shelf,
                    self.placed_obj,
                    self.layer_name,
                ),
                PlaceAction(
                    object_designator=self.obj,
                    target_location=self.obj_goal_pose,
                    arm=self.arm,
                ),
            ],
            self.context,
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
            [3, 3],
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
            load_objects_of_types(_processed_database_session(), {ObjectType.BOOK})
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
        # After the yaw below is applied, the book's footprint on the table is always
        # its thinner extent along x and its thicker extent along y (whichever native
        # extent that is), regardless of which one is book_extents[0] vs [1].
        book_footprint_x = min(book_extents[0], book_extents[1])
        book_footprint_y = max(book_extents[0], book_extents[1])
        table_edge_margin = 0.02
        book = EGObject2D(
            object_type=ObjectType.BOOK,
            scale=Scale(x=book_extents[1], y=book_extents[0], z=book_extents[2]),
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
        placed_object, layer_name = mode_query(
            spawned_shelf, trained_model.relational_probabilistic_circuit, book
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
        footprint = spawned_shelf.corpus_footprint
        standoff = Cabinet.hole_direction * (footprint.x / 2 + 0.5)
        standing_pose = Point3(
            float(standoff.x),
            float(position_y),
            reference_frame=spawned_shelf.corpus,
        )
        standing_point_in_map = world.transform(standing_pose, world.root)

        navigation_goals = path_to_shelf(world, floor, context.robot, standing_pose)

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
