# """
# The geometry the shelf demo drives on: where the robot stands to reach a placement, and
# how a route across the floor becomes navigation goals.
# """
#
# from __future__ import annotations
#
# import math
# from dataclasses import dataclass
#
# import pytest
#
# import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
# from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
# from coraplex.datastructures.grasp import GraspDescription
# from experiments.scene_generation_experiments.demo import (
#     floor_point,
#     move_to_reach_shelf,
#     path_to_shelf,
# )
# from experiments.scene_generation_experiments.exceptions import UnreachableShelfError
# from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
# from semantic_digital_twin.scene_generation.scene_schema import (
#     EGObject2D,
#     EGPoint2D,
#     EGRotation,
#     EGShelf,
#     EGShelfLayer,
#     ObjectType,
# )
# from semantic_digital_twin.semantic_annotations.semantic_annotations import (
#     Cabinet,
#     Floor,
#     Table,
# )
# from semantic_digital_twin.spatial_types import (
#     HomogeneousTransformationMatrix,
#     Point3,
# )
# from semantic_digital_twin.world import World
# from semantic_digital_twin.world_description.connections import FixedConnection
# from semantic_digital_twin.world_description.geometry import Scale
# from semantic_digital_twin.world_description.world_entity import Body
#
# # %% the scene these tests place onto
#
# _SHELF_SCALE = EGScale(height=1.2, length=0.4, width=0.8)
# """
# Dimensions of the shelf every test here places onto.
# """
#
# _SHELF_ORIGIN_YAW_DEGREES = 35.0
# """
# How far the shelf is turned away from the world's own axes.
#
# A shelf square with the world would let a standing pose that ignores the shelf's own
# orientation pass every assertion below.
# """
#
# _FLOOR_SCALE = Scale(x=8.0, y=8.0, z=0.02)
# """
# Extent of the floor the robot drives across.
# """
#
#
# @pytest.fixture
# def spawned_shelf() -> SpawnedShelf:
#     """
#     An empty two-layer shelf, standing turned in a world of its own.
#     """
#     shelf = EGShelf(
#         scale=_SHELF_SCALE,
#         layers=[
#             EGShelfLayer(objects=[], theme_dominant_type=ObjectType.BOOK)
#             for _ in range(2)
#         ],
#         theme_dominant_type=ObjectType.BOOK,
#     )
#     world = World.create_with_root_body("map")
#     shelf_origin = Body(name=PrefixedName("shelf_origin"))
#     with world.modify_world():
#         world.add_body(shelf_origin)
#         world.add_connection(
#             FixedConnection(
#                 parent=world.root,
#                 child=shelf_origin,
#                 parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
#                     x=2.0, y=1.0, yaw=math.radians(_SHELF_ORIGIN_YAW_DEGREES)
#                 ),
#             )
#         )
#     return shelf.spawn_in_world(world=world, parent=shelf_origin)
#
#
# def _placement(spawned_shelf: SpawnedShelf, layer_index: int) -> ShelfPlacement:
#     """
#     A placement on one of the shelf's layers, off-centre across its face so a standing
#     pose that ignores where on the layer the object goes stands out.
#
#     :param spawned_shelf: The shelf the placement is on.
#     :param layer_index: Index of the layer to place on.
#     :return: The placement.
#     """
#     geometry = spawned_shelf.shelf.layer_geometries()[layer_index]
#     placed_object = EGObject2D(
#         id="placed_book",
#         room_id="room",
#         place_id="shelf",
#         object_type=ObjectType.BOOK,
#         scale=EGScale(width=0.12, length=0.04, height=0.2),
#         position=EGPoint2D(x=0.05, y=-0.25),
#         orientation=EGRotation(x=0.0, y=0.0, z=20.0),
#         source_id=None,
#         theme_dominant_type=ObjectType.BOOK,
#     )
#     return ShelfPlacement(
#         layer_index=layer_index,
#         placed_object=placed_object,
#         pose=spawned_shelf.shelf.object_local_pose(
#             placed_object, geometry.slab_top_height, spawned_shelf.corpus
#         ),
#         log_likelihood=0.0,
#     )
#
#
# # %% where the robot stands to reach a placement
#
#
# def test_standing_point_is_outside_the_shelfs_open_face(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     A shelf is only reachable through its open face, so the standing point must sit the
#     requested clearance beyond that face -- not beyond any other side of the corpus.
#     """
#     placement = _placement(spawned_shelf, layer_index=0)
#     clearance = 0.4
#
#     standing_point = shelf_standing_point(
#         spawned_shelf, placement, standing_clearance=clearance
#     )
#
#     footprint = spawned_shelf.shelf.corpus_footprint
#     assert float(standing_point.x) == pytest.approx(
#         float(Cabinet.hole_direction.x) * (footprint.length / 2 + clearance)
#     )
#
#
# def test_standing_point_clears_the_corpus_the_shelf_was_spawned_with(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     The clearance is taken from the footprint the shelf reserves, which is padded beyond
#     its own dimensions, so the robot has to end up outside the corpus that was really
#     built -- not merely outside the shelf's bare scale.
#     """
#     placement = _placement(spawned_shelf, layer_index=0)
#
#     standing_point = shelf_standing_point(spawned_shelf, placement)
#
#     corpus_bounds = spawned_shelf.corpus.collision.as_bounding_box_collection_in_frame(
#         spawned_shelf.corpus
#     ).bounding_box()
#     assert float(standing_point.x) < float(corpus_bounds.x_interval.lower)
#
#
# def test_standing_point_lines_up_with_the_placement_across_the_face(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     The robot has to stand in front of the spot it places at, so its position across the
#     shelf's face must be the placement's own.
#     """
#     placement = _placement(spawned_shelf, layer_index=1)
#
#     standing_point = shelf_standing_point(spawned_shelf, placement)
#
#     assert float(standing_point.y) == pytest.approx(
#         float(placement.pose.to_position().y)
#     )
#
#
# def test_standing_point_is_on_the_floor_the_shelf_stands_on(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     The robot drives on the ground, so the standing point must be level with the shelf's
#     base rather than with the corpus frame, which sits halfway up it.
#     """
#     placement = _placement(spawned_shelf, layer_index=0)
#
#     standing_point = shelf_standing_point(spawned_shelf, placement)
#
#     assert float(standing_point.z) == pytest.approx(
#         -spawned_shelf.shelf.corpus_footprint.height / 2
#     )
#
#
# # %% reaching into the shelf from there
#
# _GRASP_DESCRIPTION = GraspDescription(
#     approach_direction=ApproachDirection.FRONT,
#     vertical_alignment=VerticalAlignment.NoAlignment,
#     end_effector=None,
# )
# """
# The grasp the demo holds the book with, which the reach only carries through.
# """
#
#
# def test_reach_offsets_the_robot_from_the_placement_to_the_standing_point(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     A move-to-reach places the robot by offsetting it from the pose it reaches for, so
#     that offset has to land exactly on the standing point in front of the open face.
#     """
#     placement = _placement(spawned_shelf, layer_index=1)
#     standing_point = shelf_standing_point(spawned_shelf, placement)
#
#     reach = move_to_reach_shelf(spawned_shelf, placement, _GRASP_DESCRIPTION)
#
#     reached = reach.target_pose_end_effector.to_position()
#     offset = reach.target_pose_offset_robot
#     assert (
#         float(reached.x) + float(offset.x),
#         float(reached.y) + float(offset.y),
#     ) == pytest.approx((float(standing_point.x), float(standing_point.y)))
#
#
# def test_reach_aims_at_the_placement_turned_with_the_shelf(
#     spawned_shelf: SpawnedShelf,
# ) -> None:
#     """
#     The arm has to go in along the axis the shelf opens on, so the pose it reaches for
#     sits at the placement but keeps the corpus's own orientation, whatever the placement
#     itself is turned to.
#     """
#     placement = _placement(spawned_shelf, layer_index=1)
#
#     reach = move_to_reach_shelf(spawned_shelf, placement, _GRASP_DESCRIPTION)
#
#     assert reach.target_pose_end_effector.reference_frame is spawned_shelf.corpus
#     assert reach.target_pose_end_effector.to_position().to_np()[:3] == pytest.approx(
#         placement.pose.to_position().to_np()[:3]
#     )
#     assert float(reach.target_pose_end_effector.yaw) == pytest.approx(0.0)
#     assert float(placement.pose.to_pose().yaw) != pytest.approx(0.0)
#
#
# # %% turning a route into navigation goals
#
#
# def _route(world: World, *positions: tuple[float, float]) -> list[Point3]:
#     """
#     A route through the given positions, as :meth:`path_from_to` hands one back.
#
#     :param world: The world the route crosses.
#     :param positions: The waypoints' x/y positions, in the world's root frame.
#     :return: The waypoints.
#     """
#     return [Point3(x, y, 0.0, reference_frame=world.root) for x, y in positions]
#
#
# @pytest.fixture
# def empty_world() -> World:
#     """
#     A world holding nothing but its root, for routes that need only a frame.
#     """
#     return World.create_with_root_body("map")
#
#
# def test_navigation_goals_leave_out_the_ends_of_the_route(
#     empty_world: World,
# ) -> None:
#     """
#     A route starts where the robot already stands and ends where the reach action drives
#     to itself, so sending it to either would only make it turn on the spot.
#     """
#     route = _route(empty_world, (0.0, 0.0), (1.0, 1.0), (3.0, 0.0))
#
#     goals = _navigation_goals(empty_world, route)
#
#     assert [(float(goal.x), float(goal.y)) for goal in goals] == [(1.0, 1.0)]
#
#
# def test_navigation_goals_face_the_waypoint_they_lead_to(empty_world: World) -> None:
#     """
#     Every goal is only a corner of the route, so the robot must arrive there already
#     turned towards where it goes next.
#     """
#     route = _route(empty_world, (0.0, 0.0), (2.0, 0.0), (2.0, 2.0))
#
#     goals = _navigation_goals(empty_world, route)
#
#     assert float(goals[0].yaw) == pytest.approx(math.radians(90.0))
#
#
# def test_a_route_with_nothing_in_the_way_needs_no_goals(empty_world: World) -> None:
#     """
#     A clear line to the shelf leaves a route of its two ends alone, which the reach
#     action drives on its own.
#     """
#     route = _route(empty_world, (0.0, 0.0), (3.0, 0.0))
#
#     goals = _navigation_goals(empty_world, route)
#
#     assert goals == []
#
#
# # %% routing across the floor
#
#
# @dataclass
# class RobotStandingAt:
#     """
#     Stands in for the robot, which a route needs only for where it currently is.
#     """
#
#     root: Body
#     """
#     The body whose global pose the route starts from.
#     """
#
#
# @dataclass
# class DividedFloor:
#     """
#     A floor whose free space falls into two halves that no route joins.
#     """
#
#     world: World
#     """
#     The world the floor and everything on it belongs to.
#     """
#
#     floor: Floor
#     """
#     The floor itself, with its supporting surface and occupants already populated.
#     """
#
#     divider: Table
#     """
#     What stands across the floor and splits its free space.
#     """
#
#     robot: RobotStandingAt
#     """
#     The robot, standing on the near side of the divider.
#     """
#
#
# _ROBOT_DISTANCE_FROM_DIVIDER = 2.0
# """
# How far, in metres, either side of the divider the robot and its goal stand.
# """
#
#
# @pytest.fixture
# def divided_floor() -> DividedFloor:
#     """
#     A floor cut in two by something spanning it, with the robot on the near side.
#     """
#     world = World.create_with_root_body("map")
#     robot_root = Body(name=PrefixedName("robot_root"))
#     with world.modify_world():
#         world.add_body(robot_root)
#         world.add_connection(
#             FixedConnection(
#                 parent=world.root,
#                 child=robot_root,
#                 parent_T_connection_expression=(
#                     HomogeneousTransformationMatrix.from_xyz_rpy(
#                         x=-_ROBOT_DISTANCE_FROM_DIVIDER
#                     )
#                 ),
#             )
#         )
#     with world.modify_world():
#         floor = Floor.create_with_new_body_in_world(
#             name="floor",
#             world=world,
#             world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
#                 z=-_FLOOR_SCALE.z / 2
#             ),
#             scale=_FLOOR_SCALE,
#         )
#         # Wider than the floor, so the split leaves nothing joined along the edges.
#         divider = Table.create_with_new_body_in_world(
#             name="divider",
#             world=world,
#             world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.5),
#             scale=Scale(x=0.2, y=_FLOOR_SCALE.y + 1.0, z=1.0),
#         )
#         floor.calculate_supporting_surface()
#         floor.add_object(divider)
#     return DividedFloor(
#         world=world, floor=floor, divider=divider, robot=RobotStandingAt(robot_root)
#     )
#
#
# def test_floor_point_sits_at_the_height_the_free_space_is_described_at(
#     divided_floor: DividedFloor,
# ) -> None:
#     """
#     The floor's free space is a thin slab at its supporting surface, so a point only
#     lands inside it once it has been brought to that height.
#     """
#     projected = floor_point(
#         divided_floor.world,
#         Point3(
#             -_ROBOT_DISTANCE_FROM_DIVIDER,
#             0.0,
#             1.4,
#             reference_frame=divided_floor.world.root,
#         ),
#         divided_floor.floor,
#     )
#
#     assert projected.reference_frame is divided_floor.floor.supporting_surface
#     assert float(projected.z) == 0.0
#     assert (
#         divided_floor.floor.calculate_free_space().node_of_point(projected) is not None
#     )
#
#
# def test_a_floor_the_free_space_cannot_cross_is_reported(
#     divided_floor: DividedFloor,
# ) -> None:
#     """
#     Both ends standing on free ground says nothing about the ground between them, so a
#     floor no route crosses has to be reported rather than driven across.
#     """
#     standing_point = Point3(
#         _ROBOT_DISTANCE_FROM_DIVIDER,
#         0.0,
#         0.0,
#         reference_frame=divided_floor.world.root,
#     )
#
#     with pytest.raises(UnreachableShelfError) as raised:
#         path_to_shelf(
#             divided_floor.world,
#             divided_floor.floor,
#             divided_floor.robot,
#             standing_point,
#         )
#
#     assert raised.value.walking_distance == pytest.approx(
#         2 * _ROBOT_DISTANCE_FROM_DIVIDER
#     )
#     assert raised.value.floor_occupants == [str(divided_floor.divider.root.name)]
