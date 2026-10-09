from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field
from functools import cached_property

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from experiments.shelf_generation_experiments.placement.shelf_placement import (
    Placement,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.tidying_demo.exceptions import (
    UnreachableShelfError,
)
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import (
    Point2,
    Point3,
    Pose,
    RotationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)


# %% the object to tidy
@dataclass(eq=False)
class GraspableShelfObject(HasGraspCandidates):
    """
    An object of the shelf dataset, grasped around the centre of its mesh.

    The origin of such an object lies at the bottom of its mesh, where the default
    grasps would reach into the surface it stands on.
    """

    def grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: Grasps at the centre of the object's collision geometry, from evenly
            spaced directions around its z-axis.
        """
        bounding_box = self.root.collision.as_bounding_box_collection_in_frame(
            self.root
        ).bounding_box()
        centre = Point3(
            (bounding_box.min_x + bounding_box.max_x) / 2,
            (bounding_box.min_y + bounding_box.max_y) / 2,
            (bounding_box.min_z + bounding_box.max_z) / 2,
            reference_frame=self.root,
        )
        return [
            GraspCandidate(
                self,
                Pose(
                    position=centre,
                    orientation=RotationMatrix.from_rpy(yaw=yaw).quaternion,
                    reference_frame=self.root,
                ),
            )
            for yaw in np.linspace(
                0, 2 * np.pi, self.grasp_candidate_count, endpoint=False
            )
        ]


# %% the floor the robot drives on
@dataclass
class FloorNavigation:
    """
    Routes a robot across the free space of a floor.
    """

    floor: Floor
    """
    The floor the robot drives on.
    """

    robot_height: float = 2.0
    """
    Height, in metres, of the space above the floor the robot needs free.
    """

    @cached_property
    def free_space(self) -> PlanarGraphOfBoundingBoxes:
        """
        The free space above :attr:`floor`, up to :attr:`robot_height`.
        """
        return self.floor.planar_free_space(max_height=self.robot_height)

    @property
    def world(self) -> World:
        """
        The world :attr:`floor` belongs to.
        """
        return self.floor._world

    def point_on_floor(self, point: Point3) -> Point2:
        """
        :param point: A point in any frame.
        :return: *point* projected onto the plane of :attr:`floor`, in the frame of the
            floor's body, which :attr:`free_space` is expressed in.
        """
        point_on_floor = self.world.transform(point, self.floor.root)
        return Point2(
            float(point_on_floor.x),
            float(point_on_floor.y),
            reference_frame=self.floor.root,
        )

    def route(self, start: Point3, goal: Point3) -> list[Pose]:
        """
        The navigation goals leading from *start* around everything on the floor towards
        *goal*, each facing the next one.

        The last leg is left out, since reaching for something drives it anyway; a
        straight way to *goal* yields no goals at all.

        :param start: Where the route starts.
        :param goal: Where the route ends.
        :raises UnreachableShelfError: If the free space connects *start* and *goal*
            nowhere.
        :return: The navigation goals, in the frame of the world root.
        """
        start_on_floor = self.point_on_floor(start)
        goal_on_floor = self.point_on_floor(goal)
        waypoints = self.free_space.path_from_to(start_on_floor, goal_on_floor)
        if waypoints is None:
            raise UnreachableShelfError(
                walking_distance=float(
                    start_on_floor.to_point3().euclidean_distance(
                        goal_on_floor.to_point3()
                    )
                ),
                floor_occupant_names=[
                    str(occupant.root.name) for occupant in self.floor.objects
                ],
            )
        root = self.world.root
        goals = []
        for waypoint, next_waypoint in zip(waypoints[1:-1], waypoints[2:]):
            here = self.world.transform(waypoint.to_point3(), root)
            onwards = self.world.transform(next_waypoint.to_point3(), root)
            goals.append(
                Pose.from_xyz_rpy(
                    x=float(here.x),
                    y=float(here.y),
                    yaw=math.atan2(
                        float(onwards.y) - float(here.y),
                        float(onwards.x) - float(here.x),
                    ),
                    reference_frame=root,
                )
            )
        return goals


# %% where the robot stands
@dataclass
class ShelfFront:
    """
    The open face of a spawned shelf, in front of which the robot stands to reach in.
    """

    shelf: RelationalCircuitExperimentShelf
    """
    The spawned shelf.
    """

    corpus_wall_thickness: float
    """
    Thickness, in metres, of the corpus walls the shelf was spawned with.
    """

    standoff: float = 0.5
    """
    Distance, in metres, between the open face and where the robot stands.
    """

    def standing_point(self, placement: Placement) -> Point3:
        """
        :param placement: The placement the robot reaches for.
        :return: Where the robot stands, in the corpus frame: in front of the open face,
            level with the placed object.
        """
        depth = self.shelf.corpus_footprint(self.corpus_wall_thickness).x
        standoff = self.shelf.annotation.hole_direction * (depth / 2 + self.standoff)
        return Point3(
            float(standoff.x),
            float(placement.placed_object.pose.y),
            reference_frame=self.shelf.corpus,
        )


# %% the action
@dataclass
class ShelfTidyingAction(ActionDescription):
    """
    Pick an object up, drive along a route across the floor and place the object on the
    shelf.
    """

    transport: TransportAction
    """
    Picks the object up and places it where it belongs on the shelf.
    """

    route: list[NavigateAction] = field(default_factory=list)
    """
    The drive from where the object is picked up towards the shelf, around whatever
    stands on the floor.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(self.robot.all_arms),
                self.transport.pick_up,
                ParkArmsAction(self.robot.all_arms),
                *self.route,
                self.transport.place,
                ParkArmsAction(self.robot.all_arms),
            ]
        )
