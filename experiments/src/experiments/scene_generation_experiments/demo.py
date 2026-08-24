from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from visualization_msgs.msg import Marker, MarkerArray

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.misc import MoveToReach
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from experiments.scene_generation_experiments.exceptions import NoFittingObjectError
from experiments.scene_generation_experiments.shelf_placement import (
    most_likely_shelf_placement,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _load_or_train_shelf_model,
    _processed_database_session,
    generate_shelf_with_arbitrary_objects,
    visualize_spawned_shelf,
    VisualizationBackend,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    load_objects_of_types,
    rclpy_node,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_theme_shelf_query,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    ObjectType,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
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


def book_grasp_description(context: Context) -> GraspDescription:
    """
    Build the grasp the demo takes the book with, from the front and without turning the
    gripper.

    Shared by the reach and the pick-up so both approach the book the same way: a
    pick-up that grasped differently than the reach was aimed for would have to move
    the arm all over again from the pose the reach left it in.

    :param context: The context whose robot's end effector grasps.
    :return: The grasp description.
    """
    return GraspDescription(
        approach_direction=ApproachDirection.FRONT,
        vertical_alignment=VerticalAlignment.NoAlignment,
        end_effector=context.robot.end_effector,
        rotate_gripper=False,
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
    standing_point_on_surface = world.transform(
        standing_point_on_table, floor.supporting_surface
    )
    # calculate_free_space()'s graph is decomposed as a thin slab at the surface's
    # own height, so the query point is projected onto that height.
    standing_point_on_floor = Point3(
        float(standing_point_on_surface.x),
        float(standing_point_on_surface.y),
        0.0,
        reference_frame=floor.supporting_surface,
    )

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
        grasp_description=book_grasp_description(context),
    )


@dataclass(eq=False)
class _AnimationPacer(StateChangeCallback):
    """
    Sleeps briefly on every world state change, so a viewer such as Foxglove can render
    each intermediate pose of a simulated action.

    :class:`~coraplex.execution_environment.simulated_robot` ticks its motion state
    chart back to back with no delay between ticks, calling
    :meth:`~semantic_digital_twin.world.World.notify_state_change` -- and so this
    callback -- on every one; without this pause, a whole reach trajectory finishes
    faster than any viewer could render it, and only its final pose is ever seen.
    """

    seconds_per_tick: float = 0.03
    """
    Wall-clock delay added per world state change.
    """

    def on_state_change(self, **kwargs):
        time.sleep(self.seconds_per_tick)


if __name__ == "__main__":
    with rclpy_node() as node:
        world = World()

        hsrb_world = URDFParser.from_file(file_path=HSRB.get_ros_file_path()).parse()
        hsrb = HSRB.from_world(hsrb_world)
        shelf_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0, y=0.0, z=0.0)
        robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=0.0, y=0.0, z=0.0)
        floor_scale = Scale(x=8.0, y=8.0, z=0.02)
        # Matches the HSRB-reachable counter height already used in
        # kitchen_environment.py, rather than a human dining-table height the
        # robot's arm cannot comfortably reach.
        table_scale = Scale(x=0.9, y=0.6, z=0.6)
        # Off to the side of both the robot's start and the shelf, so nothing overlaps.
        table_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=-1.5, z=table_scale.z / 2
        )

        model_path = Path(__file__).parent / "models" / "arbitrary_shelf_rspn.json"
        trained_model = _load_or_train_shelf_model(model_path)

        # Two layers rather than four: slabs are spread evenly over the drawn corpus,
        # and four of them leave 0.07 m to 0.37 m above each slab, which the shortest
        # book scan in the dataset already does not always fit under.
        query = build_theme_shelf_query(
            trained_model.relational_probabilistic_circuit,
            ObjectType.JAR,
            [3, 3],
        )

        spawned_shelf, trained_model = generate_shelf_with_arbitrary_objects(
            query, model_path=model_path
        )

        # HSRB's URDF only describes the physical robot, with no joint attaching it
        # to a world frame, so a movable "odom_combined" frame and the robot's own
        # drive connection are inserted between it and whatever it is merged into --
        # mirroring the "map -> odom_combined -> urdf tree" that
        # test.conftest.world_with_urdf_factory builds for the same reason. Without
        # this, the merged-in root only gets the plain Connection6DoF world.merge_world
        # defaults to, and navigation actions -- which look up the robot's unique
        # OmniDrive-typed connection -- have none to find.
        with hsrb_world.modify_world():
            hsrb_root = hsrb_world.root
            odom_combined = Body(name=PrefixedName("odom_combined"))
            hsrb_world.add_body(odom_combined)
            drive_connection_type = HSRB.get_drive_connection_type()
            odom_C_root = drive_connection_type.create_with_dofs(
                parent=odom_combined, child=hsrb_root, world=hsrb_world
            )
            hsrb_world.add_connection(odom_C_root)
            odom_C_root.has_hardware_interface = True
        odom_C_root.origin = robot_pose.copy_with_new_reference_frames(
            new_reference_frame=odom_combined, new_child_frame=hsrb_root
        )

        with world.modify_world():
            world.add_body(Body(name=PrefixedName("map")))
            # Box is centered on its pose; drop it by half its thickness so the
            # top surface sits at z=0, level with the robot's and shelf's base.
            floor = Floor.create_with_new_body_in_world(
                name="floor",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-floor_scale.z / 2
                ),
                scale=floor_scale,
            )
            world.merge_world(hsrb_world)
            table = Table.create_with_new_body_in_world(
                name="table",
                world=world,
                world_root_T_self=table_pose,
                scale=table_scale,
            )
            floor.calculate_supporting_surface()
            floor.add_object(table)

        with spawned_shelf.world.modify_world():
            spawned_shelf.parent.name = PrefixedName(name="shelf_origin")
        world.merge_world_at_pose(spawned_shelf.world, shelf_pose)
        spawned_shelf.world = world

        book_candidates = _get_source_ids_for_objects(
            load_objects_of_types(_processed_database_session(), {ObjectType.BOOK})
        )
        # sage10k book scans come in two poses: lying flat, cover up, or standing
        # upright on a shelf, spine down. Only a candidate whose recorded height --
        # its extent along the scan's up axis -- is its largest side was captured
        # standing; that is the pose that leaves the book's two covers exposed on its
        # sides at roll=pitch=0, the faces a parallel gripper can grasp from above.
        book_candidates_standing = [
            candidate
            for candidate in book_candidates
            if candidate.native_extents is not None
            and candidate.native_extents[2] == max(candidate.native_extents)
        ]
        # The book has to be one this shelf could take back: its layers are spaced
        # evenly across the drawn corpus, so a four-layer shelf leaves under 0.2 m
        # above each slab, while a standing book scan is 0.25 m tall on average.
        layer_geometries = spawned_shelf.shelf.layer_geometries()
        tallest_layer_room = max(
            geometry.maximum_object_extents.height for geometry in layer_geometries
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
                    geometry.maximum_object_extents.height
                    for geometry in layer_geometries
                ],
            )
        book_candidate = random.choice(book_candidates_fitting)
        book_extents = book_candidate.native_extents
        book = EGObject2D(
            id="demo_book",
            room_id="demo_room",
            place_id="demo_table",
            object_type=ObjectType.BOOK,
            scale=EGScale(
                width=book_extents[0], length=book_extents[1], height=book_extents[2]
            ),
            position=EGPoint2D(x=0.0, y=0.0),
            # Roll and pitch left at 0 so the book stands upright on its spine, the
            # same way floor and shelf objects are always spawned upright.
            orientation=EGRotation(x=0.0, y=0.0, z=0.0),
            source_id=book_candidate.source_id,
            theme_dominant_type=ObjectType.BOOK,
        )
        book_body = book.create_in_world(
            world,
            book_candidate.scene_dir,
            parent=table.root,
            # The book's mesh origin sits at its own lowest point (see
            # _mesh_centered_on_footprint), so resting it on the table only takes
            # the table's own half-height -- the table's box is centered on its pose.
            world_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=table_scale.z / 2, reference_frame=table.root
            ),
        )

        # Published before the robot moves, and with a pause for the viewer to
        # connect: spawning the shelf's meshes in Foxglove is slow enough that,
        # published only after the action runs, a viewer only ever sees the robot
        # already at its final pose and never the move itself.
        viz_marker = visualize_spawned_shelf(
            node, spawned_shelf, visualization_backend=VisualizationBackend.FOXGLOVE
        )
        input(
            "Scene published. Connect a Foxglove viewer, then press Enter to run "
            "the reach action..."
        )

        context = Context.from_world(world, query_backend=ProbabilisticBackend())

        animation_pacer = _AnimationPacer(_world=world)
        try:
            with simulated_robot:
                sequential(
                    [
                        move_to_reach_book(context, floor, table, book_body),
                        PickUpAction(
                            object_designator=book_body,
                            arm=Arms.LEFT,
                            grasp_description=book_grasp_description(context),
                        ),
                    ],
                    context,
                ).perform()
        finally:
            animation_pacer.stop()

        placement = most_likely_shelf_placement(
            spawned_shelf, trained_model.relational_probabilistic_circuit, book
        )
        placement_in_map = world.transform(placement.pose.to_pose(), world.root)
        print(
            f"The book belongs on layer {placement.layer_index} of "
            f"{len(spawned_shelf.layers)}, at "
            f"x={placement.placed_object.position.x:.3f} "
            f"y={placement.placed_object.position.y:.3f} "
            f"yaw={placement.placed_object.orientation.z:.1f} deg in the shelf's "
            f"frame (log-likelihood {placement.log_likelihood:.2f})"
        )
        print(
            f"  in map coordinates: x={float(placement_in_map.x):.3f} "
            f"y={float(placement_in_map.y):.3f} z={float(placement_in_map.z):.3f}"
        )

        try:
            while True:
                viz_marker._tf_publisher.on_state_change()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            # Since the marker publisher is TRANSIENT_LOCAL, its last published
            # state lingers for any viewer connecting after this process exits.
            # Clearing it here means markers are only ever on screen while this
            # script is actively running.
            viz_marker.publisher.publish(
                MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
            )
