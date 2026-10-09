from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import coraplex.orm.ormatic_interface  # noqa: F401  registers the action's DAOs
import semantic_digital_twin.orm.ormatic_interface  # noqa: F401  registers world DAOs
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from experiments.shelf_generation_experiments.dataset_environment import (
    DatasetEnvironmentVariable,
)
from experiments.shelf_generation_experiments.generation.shelf_generator import (
    ShelfGenerator,
)
from experiments.shelf_generation_experiments.placement.shelf_placement import (
    Placement,
    ShelfPlacement,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.tidying_demo.exceptions import (
    NoFittingObjectError,
    NoStandingMeshError,
)
from experiments.shelf_generation_experiments.tidying_demo.shelf_tidying import (
    FloorNavigation,
    GraspableShelfObject,
    ShelfFront,
    ShelfTidyingAction,
)
from experiments.shelf_generation_experiments.tidying_demo.visualization import (
    VisualizationBackend,
    spinning_ros_node,
)
from experiments.shelf_generation_experiments.training.processed_database import (
    ProcessedShelfDatabase,
)
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType
from rclpy.node import Node
from semantic_digital_twin.api import RobotSpecification
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


# %% demo
@dataclass
class TidiedObject:
    """
    What a run of the demo tidied, and where to.
    """

    shelf: RelationalCircuitExperimentShelf
    """
    The generated shelf.
    """

    placement: Placement
    """
    Where on :attr:`shelf` the object belongs.
    """

    body: Body
    """
    The body of the tidied object.
    """


@dataclass
class ShelfTidyingDemo:
    """
    A robot picks an object up from a table and places it where a generated shelf's
    model says it belongs.
    """

    database: ProcessedShelfDatabase
    """
    The processed database shelves are fitted on and meshes are looked up in.
    """

    scenes_root: Path
    """
    Directory holding the downloaded scenes whose meshes can be spawned.
    """

    model_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "models" / "shelf_model.json"
    )
    """
    Where the fitted shelf model is stored, and loaded from on later runs.
    """

    objects_per_layer: list[int] = field(default_factory=lambda: [3, 3, 3])
    """
    How many objects each layer of the generated shelf holds.
    """

    shelf_pose: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0)
    )
    """
    Where the shelf stands, relative to the world root.
    """

    robot_pose: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Where the robot starts, relative to its odometry frame.
    """

    floor_scale: Scale = field(default_factory=lambda: Scale(x=8.0, y=8.0, z=0.02))
    """
    The size of the floor.
    """

    table_scale: Scale = field(default_factory=lambda: Scale(x=0.9, y=0.6, z=0.2))
    """
    The size of the table the object lies on.
    """

    table_position: Pose2D = field(default_factory=lambda: Pose2D(x=1.0, y=-1.5))
    """
    Where the table stands on the floor.
    """

    held_object_position: Pose2D = field(
        default_factory=lambda: Pose2D(x=-0.31, y=0.15)
    )
    """
    Where the object lies on the table, relative to the table's centre.
    """

    visualization_backend: VisualizationBackend = VisualizationBackend.FOXGLOVE
    """
    The viewer the world is published for.
    """

    def run(self, node: Node) -> TidiedObject:
        """
        Generate the shelf, set the scene up around it and let the robot tidy the
        object.

        :param node: The ROS node the world is visualized with.
        :return: What was tidied, and where to.
        """
        world = World.create_with_root_body()
        model = self._shelf_model()
        generator = ShelfGenerator(
            model=model, database=self.database, scenes_root=self.scenes_root
        )
        candidates_by_type = self._standing_candidates_by_type(model)
        object_type = random.choice(sorted(candidates_by_type))

        generated = generator.generate(
            RelationalCircuitExperimentShelf.underspecified_query(
                ..., self.objects_per_layer
            ),
            world,
            parent_T_self=self.shelf_pose,
        )
        shelf = generated.shelf
        logging.getLogger(__name__).info(
            "Generated a %s shelf with %d layers, %d objects standing and %d dropped.",
            shelf.theme_dominant_type.value,
            len(shelf.layers),
            generated.standing_object_count,
            generated.dropped_object_count,
        )
        publisher = self.visualization_backend.publish(world, node)

        RobotSpecification(
            semantic_annotation_type=HSRB, odom_T_robot_start=self.robot_pose
        ).spawn(world)
        floor, table = self._floor_and_table(world, shelf)

        shelf_front = ShelfFront(
            shelf=shelf, corpus_wall_thickness=generator.corpus_wall_thickness
        )
        held_candidate = self._fitting_candidate(
            shelf, generator, object_type, candidates_by_type[object_type]
        )
        held_object = RelationalCircuitExperimentObject2D(
            object_type=object_type,
            scale=Scale(
                x=max(held_candidate.scale.x, held_candidate.scale.y),
                y=min(held_candidate.scale.x, held_candidate.scale.y),
                z=held_candidate.scale.z,
            ),
            pose=Pose2D(),
            source_id=held_candidate.source_id,
            name="held_object",
        )
        held_body = held_object.spawn(
            world,
            mesh_path=held_candidate.scene_directory,
            parent=table.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(self.held_object_position.x),
                y=float(self.held_object_position.y),
                z=self.table_scale.z / 2,
                reference_frame=table.root,
            ),
        )

        # The circuit's yaw distribution is close to uniform, so the yaw is pinned to
        # the one turning the thin side of the object, its y extent, towards the open
        # face.
        placement = ShelfPlacement(shelf=shelf, model=model).most_likely_placement(
            held_object, yaw=0.0
        )
        goal_pose = world.transform(
            Pose2D(
                float(placement.placed_object.pose.x),
                float(placement.placed_object.pose.y),
                float(placement.placed_object.pose.yaw),
                reference_frame=placement.layer.annotation.root,
            ).pose,
            world.root,
        )

        graspable = GraspableShelfObject(root=held_body)
        with world.modify_world():
            world.add_semantic_annotation(graspable)
        context = Context.from_world(world)
        route = FloorNavigation(floor).route(
            context.robot.root.global_pose.position,
            shelf_front.standing_point(placement),
        )
        tidying = ShelfTidyingAction(
            transport=TransportAction.from_graspable_by_closest_grasps(
                graspable, goal_pose, context.robot.all_arms[0], context
            ),
            route=[NavigateAction(goal) for goal in route],
        )
        with simulated_robot():
            execute_single(tidying, context).perform()
        publisher.stop()
        return TidiedObject(shelf=shelf, placement=placement, body=held_body)

    def _shelf_model(self) -> ShelfModel:
        """
        :return: The model stored at :attr:`model_path`, fitted on the processed
            database and stored there first if there is none yet.
        """
        if self.model_path.exists():
            return ShelfModel.load(self.model_path)
        model = ShelfModel.fit(self.database.shelves())
        model.save(self.model_path)
        return model

    def _standing_candidates_by_type(
        self, model: ShelfModel
    ) -> dict[ObjectType, list[MeshCandidate]]:
        """
        :param model: The shelf model the held object has to be placeable by.
        :raises NoStandingMeshError: If no placeable type has a standing mesh.
        :return: Per placeable object type, its meshes that are taller than they are
            wide or deep.
        """
        candidates_by_type: dict[ObjectType, list[MeshCandidate]] = defaultdict(list)
        for candidate in self.database.mesh_candidates(
            model.coarsening.placeable_object_types, self.scenes_root
        ):
            scale = candidate.scale
            if scale is not None and scale.z >= max(scale.x, scale.y):
                candidates_by_type[candidate.object_type].append(candidate)
        if not candidates_by_type:
            raise NoStandingMeshError()
        return candidates_by_type

    def _floor_and_table(
        self, world: World, shelf: RelationalCircuitExperimentShelf
    ) -> tuple[Floor, Table]:
        """
        Add the floor, with the shelf standing on it, and the table to *world*.

        :param world: The world to add them to.
        :param shelf: The spawned shelf.
        :return: The floor and the table.
        """
        with world.modify_world():
            floor = Floor.create_with_new_body_in_world(
                name="floor",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-self.floor_scale.z / 2
                ),
                scale=self.floor_scale,
            )
            table = Table.create_with_new_body_in_world(
                name="table",
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=float(self.table_position.x),
                    y=float(self.table_position.y),
                    z=self.table_scale.z / 2,
                ),
                scale=self.table_scale,
            )
            floor.calculate_supporting_surface()
            floor.add_object(table)
            floor.add_object(shelf.annotation)
        return floor, table

    @staticmethod
    def _fitting_candidate(
        shelf: RelationalCircuitExperimentShelf,
        generator: ShelfGenerator,
        object_type: ObjectType,
        candidates: list[MeshCandidate],
    ) -> MeshCandidate:
        """
        :param shelf: The spawned shelf the object has to fit on.
        :param generator: The generator that spawned *shelf*.
        :param object_type: The type of the object.
        :param candidates: The standing meshes of *object_type*.
        :raises NoFittingObjectError: If no candidate fits any layer of *shelf*.
        :return: A random candidate low enough for some layer of *shelf*.
        """
        layer_heights = [
            layer.maximum_object_extents.z
            for layer in shelf.layers_with_geometry(
                generator.slab_thickness, generator.corpus_wall_thickness
            )
        ]
        fitting_candidates = [
            candidate
            for candidate in candidates
            if candidate.scale.z <= max(layer_heights)
        ]
        if not fitting_candidates:
            raise NoFittingObjectError(
                object_type=object_type,
                shortest_height=min(candidate.scale.z for candidate in candidates),
                layer_heights=layer_heights,
            )
        return random.choice(fitting_candidates)


# %% command-line entry point
def main() -> None:
    """
    Run the shelf tidying demo against the processed database and scenes the environment
    names.
    """
    logging.basicConfig(level=logging.INFO)
    demo = ShelfTidyingDemo(
        database=ProcessedShelfDatabase.from_environment(),
        scenes_root=Path(DatasetEnvironmentVariable.SCENES_ROOT.read()),
    )
    with spinning_ros_node() as node:
        demo.run(node)


if __name__ == "__main__":
    main()
