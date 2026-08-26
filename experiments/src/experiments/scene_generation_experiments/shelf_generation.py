from __future__ import annotations

import dataclasses
from collections import Counter
from enum import StrEnum
from pathlib import Path

import trimesh
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sqlalchemy.orm import Session
from visualization_msgs.msg import Marker

from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.base import ActionDescription
from experiments.scene_generation_experiments.rspn_model_storage import (
    TrainedArbitraryShelfModel,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import (
    RelationalCircuitRegistry,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from experiments.scene_generation_experiments.processed_database import (
    load_objects_of_types,
    load_shelves,
)
from experiments.scene_generation_experiments.utils import (
    MINIMUM_SAMPLES_PER_QUANTILE,
    _get_source_ids_for_objects,
    min_samples_per_leaf_for,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.scene_generation.scene_schema import (
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
    EGObject2D,
    SpawnedShelf,
)


def _frequent_object_types(
    shelf_layers: list[EGShelfLayer],
    keep_count: int,
) -> set[ObjectType]:
    """
    Return the *keep_count* most frequent object types across all objects in
    *shelf_layers*.

    :param shelf_layers: Layers whose objects' types are counted.
    :param keep_count: Number of distinct, most frequent object types to return.
    :return: The most frequent object types.
    """
    type_counts = Counter(
        object_2d.object_type for layer in shelf_layers for object_2d in layer.objects
    )
    return {object_type for object_type, _ in type_counts.most_common(keep_count)}


def _coarsen_rare_object_types(
    shelf_layers: list[EGShelfLayer],
    keep_count: int = 20,
) -> list[EGShelfLayer]:
    """
    Return new shelf layers where every object's type outside the *keep_count* most
    frequent types (across all objects in *shelf_layers*) is replaced with
    ``ObjectType.OTHER``.

    :param shelf_layers: Layers whose objects' types should be coarsened.
    :param keep_count: Number of distinct, most frequent object types to leave
        unchanged.
    :return: New EGShelfLayer instances with coarsened object types; all other fields
        (position, scale, orientation, source_id, ...) are unchanged.
    """
    frequent_types = _frequent_object_types(shelf_layers, keep_count)
    return [
        dataclasses.replace(
            layer,
            objects=[
                (
                    object_2d
                    if object_2d.object_type in frequent_types
                    else dataclasses.replace(object_2d, object_type=ObjectType.OTHER)
                )
                for object_2d in layer.objects
            ],
        )
        for layer in shelf_layers
    ]


def _coarsen_mesh_candidate_types(
    candidates: list[MeshCandidate],
    frequent_types: set[ObjectType],
) -> list[MeshCandidate]:
    """
    Return new mesh candidates where every candidate whose type falls outside
    *frequent_types* is relabeled as ``ObjectType.OTHER``.

    Mirrors :func:`_coarsen_rare_object_types` so the mesh pool's type labels
    line up with the coarsened types the RSPN actually samples -- without
    this, a sampled ``ObjectType.OTHER`` object would never find a same-type
    mesh candidate, since every candidate still carries its original,
    uncoarsened type.

    :param candidates: Mesh candidates whose types should be coarsened.
    :param frequent_types: Object types to leave unchanged; every other
        type is replaced with ``ObjectType.OTHER``.
    :return: New candidates with coarsened types.
    """
    return [
        (
            candidate
            if candidate.object_type in frequent_types
            else dataclasses.replace(candidate, object_type=ObjectType.OTHER)
        )
        for candidate in candidates
    ]


def _mesh_candidate_types_for_shelf(
    shelf: EGShelf, frequent_types: set[ObjectType]
) -> set[ObjectType]:
    """
    Return the object types a mesh-candidate query needs to cover *shelf*.

    A grounded shelf only ever needs meshes for the types it actually sampled, not
    every type the database holds. ``ObjectType.OTHER`` is the exception: it is a
    coarsening sentinel, not a real database value (see :func:`_coarsen_rare_object_types`),
    so it resolves to every type outside *frequent_types* -- the same types
    :func:`_coarsen_mesh_candidate_types` would relabel as ``OTHER`` again afterwards.

    :param shelf: The grounded shelf whose layers' objects name the needed types.
    :param frequent_types: The types the shelf's RSPN was trained to sample directly;
        used to resolve ``ObjectType.OTHER`` back to the types it can stand in for.
    :return: The set of object types a mesh-candidate query must include.
    """
    sampled_types = {
        object_2d.object_type for layer in shelf.layers for object_2d in layer.objects
    }
    if ObjectType.OTHER not in sampled_types:
        return sampled_types
    return (sampled_types - {ObjectType.OTHER}) | (set(ObjectType) - frequent_types)


def _coarsen_rare_object_types_of_shelves(
    shelves: list[EGShelf],
    keep_count: int = 20,
) -> list[EGShelf]:
    """
    Return new shelves whose layers' object types have been coarsened together.

    Every layer of every shelf is coarsened in one pass, so one shared set of frequent
    types survives. Coarsening each shelf on its own would let a type be kept on one
    shelf and collapsed on another, and the mesh pool -- coarsened once against the same
    set -- would then find no mesh for it.

    :param shelves: Shelves whose objects' types should be coarsened.
    :param keep_count: Number of distinct, most frequent object types to leave
        unchanged.
    :return: New shelves carrying coarsened layers.
    """
    coarsened_layers = iter(
        _coarsen_rare_object_types(
            [layer for shelf in shelves for layer in shelf.layers], keep_count
        )
    )
    return [
        dataclasses.replace(
            shelf, layers=[next(coarsened_layers) for _ in shelf.layers]
        )
        for shelf in shelves
    ]


def _frequent_shelf_themes(
    shelves: list[EGShelf],
    keep_count: int,
) -> set[ObjectType]:
    """
    Return the *keep_count* most frequent dominant types across *shelves*.

    A shelf's theme (the object type its own objects have the most of) is a
    per-shelf statistic, distinct from the per-object frequency
    :func:`_frequent_object_types` counts -- a type common on individual objects is
    not necessarily common as a shelf's *mode*, so this needs its own frequency
    count rather than reusing that one.

    :param shelves: Shelves whose dominant types are counted.
    :param keep_count: Number of distinct, most frequent themes to return.
    :return: The most frequent themes.
    """
    theme_counts = Counter(shelf.theme_dominant_type for shelf in shelves)
    return {theme for theme, _ in theme_counts.most_common(keep_count)}


def _coarsen_rare_shelf_themes(
    shelves: list[EGShelf],
    keep_count: int = 20,
) -> list[EGShelf]:
    """
    Return new shelves whose theme, outside the *keep_count* most frequent themes
    across *shelves*, is replaced with ``ObjectType.OTHER`` on the shelf, every
    layer, and every object -- the three places it is denormalized onto.

    :param shelves: Shelves whose themes should be coarsened.
    :param keep_count: Number of distinct, most frequent themes to leave unchanged.
    :return: New shelves carrying a coarsened theme throughout.
    """
    frequent_themes = _frequent_shelf_themes(shelves, keep_count)
    return [
        (
            shelf
            if shelf.theme_dominant_type in frequent_themes
            else dataclasses.replace(
                shelf,
                theme_dominant_type=ObjectType.OTHER,
                layers=[
                    dataclasses.replace(
                        layer,
                        theme_dominant_type=ObjectType.OTHER,
                        objects=[
                            dataclasses.replace(
                                object_2d, theme_dominant_type=ObjectType.OTHER
                            )
                            for object_2d in layer.objects
                        ],
                    )
                    for layer in shelf.layers
                ],
            )
        )
        for shelf in shelves
    ]


class VisualizationBackend(StrEnum):
    """
    Viewer the generated shelf's markers are published for.
    """

    FOXGLOVE = "foxglove"
    """
    A browser-based Foxglove client connected through ``foxglove_bridge``.

    Browsers cannot load ``file://`` mesh resources, so mesh URIs are
    rewritten onto a ``package://`` resource :func:`_rewrite_mesh_uris_for_foxglove`
    serves over the same websocket.
    """

    RVIZ = "rviz"
    """
    A local RViz2 instance, which loads ``file://`` mesh resources directly and
    needs no ``foxglove_bridge`` package to serve them through.
    """


_MESH_RESOURCE_PACKAGE = "foxglove_bridge"
"""
ROS package whose share directory mesh files are copied into so a browser-based viewer
can load them via a ``package://`` marker resource.

Browsers refuse to ``fetch()`` ``file://`` URLs, which is what a MESH_RESOURCE
marker normally points at -- ``package://`` is instead resolved server-side by
foxglove_bridge and streamed back over the same websocket, so it works from an
arbitrary remote browser. ``foxglove_bridge`` is reused here only because it is
already installed and its share directory is guaranteed to exist; the meshes
have no other relation to that package.
"""

_MESH_RESOURCE_SHARE_SUBDIR = "shelf_generation_meshes"
"""
Subdirectory of :data:`_MESH_RESOURCE_PACKAGE`'s share directory that copied mesh files
are placed under.
"""

_ROS_PACKAGE_SHARE_SEGMENT = "/share/"
"""
Path segment separating a ROS install prefix from the ``<package>/...`` layout every
package's share directory follows, used to recover a mesh's owning package name from its
installed ``file://`` path.
"""

_FOXGLOVE_GLTF_UP_AXIS_CORRECTION = Rotation.from_euler("x", -90, degrees=True)
"""
Rotation applied to a mesh marker's local orientation to cancel out Foxglove's glTF up-
axis handling.

glTF meshes are conventionally authored Y-up, and Foxglove's 3D panel documents that its
"Mesh up axis" override does not apply to glTF/``.glb`` files because they are assumed
to already carry that convention -- unlike STL/OBJ, there is no user-facing toggle to
disable it. Our exported ``.glb`` files are written directly from Z-up (ROS/world)
vertex data with no axis conversion, so Foxglove's built-in Y-up-to-Z-up correction
misinterprets them and renders them tipped onto their side. RViz's mesh loader applies
no such correction, so this compensation is Foxglove-only.
"""


def _rewrite_mesh_uris_for_foxglove(viz_marker: VizMarkerPublisher) -> None:
    """
    Rewrite every marker's ``mesh_resource`` to a ``package://`` URI a browser-based
    Foxglove client can fetch over the websocket, instead of a ``file://`` path it
    cannot load directly.

    A generated shelf mesh (``file:///tmp/...``) is converted from OBJ to glTF and
    copied into a ROS package share directory by
    :func:`_convert_generated_mesh_to_package_uri`; a mesh already installed under a ROS
    package's share directory (e.g. a robot's URDF meshes) only needs its URI rewritten,
    by :func:`_rewrite_installed_package_mesh_uri`. A marker with no mesh is left alone.

    :param viz_marker: Publisher whose current markers are rewritten in place.
    """
    share_root = (
        Path(get_package_share_directory(_MESH_RESOURCE_PACKAGE))
        / _MESH_RESOURCE_SHARE_SUBDIR
    )
    for marker in viz_marker.markers.markers:
        if marker.mesh_resource.startswith("file:///tmp/"):
            _convert_generated_mesh_to_package_uri(marker, share_root)
        elif marker.mesh_resource.startswith("file://"):
            _rewrite_installed_package_mesh_uri(marker)


def _convert_generated_mesh_to_package_uri(marker: Marker, share_root: Path) -> None:
    """
    Convert a generated shelf mesh from OBJ to glTF (``.glb``) in *share_root* and
    rewrite *marker*'s ``mesh_resource`` to the matching ``package://`` URI.

    ``Mesh.from_ply_file`` exports OBJ with a separate ``.mtl`` sidecar for
    material/texture, which RViz2 and MuJoCo both read but Foxglove does not --
    Foxglove only loads material/texture from a self-contained glTF file. Converting
    here, rather than in ``Mesh.from_ply_file`` itself, keeps that OBJ default intact
    for every other consumer.

    Also pre-rotates the marker's local orientation to cancel Foxglove's glTF up-axis
    correction; see :data:`_FOXGLOVE_GLTF_UP_AXIS_CORRECTION`.

    :param marker: Marker whose generated mesh is converted and rewritten in place.
    :param share_root: Directory converted meshes are copied into.
    """
    source_path = Path(marker.mesh_resource[len("file://") :])
    dest_dir = share_root / source_path.parent.name
    dest_path = dest_dir / f"{source_path.stem}.glb"
    if not dest_path.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        trimesh.load(source_path, force="mesh").export(dest_path, file_type="glb")
    marker.mesh_resource = (
        f"package://{_MESH_RESOURCE_PACKAGE}/{_MESH_RESOURCE_SHARE_SUBDIR}/"
        f"{source_path.parent.name}/{dest_path.name}"
    )
    orientation = marker.pose.orientation
    corrected = (
        Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        * _FOXGLOVE_GLTF_UP_AXIS_CORRECTION
    )
    (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ) = corrected.as_quat()


def _rewrite_installed_package_mesh_uri(marker: Marker) -> None:
    """
    Rewrite a mesh already installed under a ROS package's share directory (e.g. a
    robot's URDF meshes) from its local ``file://`` path to the equivalent
    ``package://`` URI, which ``foxglove_bridge`` resolves and streams over the
    websocket instead of a host-local path only this machine can read.

    Every ROS package share directory follows the ``<install prefix>/share/<package
    name>/...`` layout, so the package name is recovered from the path segment right
    after ``share/`` rather than by querying the ament index for every mesh.

    :param marker: Marker whose installed mesh is rewritten in place. Left untouched if
        its path does not follow the ``.../share/<package>/...`` layout.
    """
    path = marker.mesh_resource[len("file://") :]
    _, separator, relative_to_share = path.partition(_ROS_PACKAGE_SHARE_SEGMENT)
    if not separator:
        return
    package_name, _, package_relative_path = relative_to_share.partition("/")
    marker.mesh_resource = f"package://{package_name}/{package_relative_path}"


@dataclasses.dataclass(eq=False)
class FoxgloveVizMarkerPublisher(VizMarkerPublisher):
    """
    A marker publisher whose meshes a browser-based Foxglove client can load.

    Rewriting on the way out rather than once is what makes it hold: markers are rebuilt
    from the world on every model change, and picking an object up or putting it down
    re-parents it, which is one. A single rewrite is undone by the first such change,
    putting ``file://`` paths no browser can fetch back on the topic.
    """

    def publish_markers(self) -> None:
        _rewrite_mesh_uris_for_foxglove(self)
        super().publish_markers()


def _publish_with_deleteall(viz_marker: VizMarkerPublisher) -> None:
    """
    Prepend a DELETEALL marker and publish, so a fresh run replaces the previous one
    instead of piling on top of it.

    Every run gives its bodies fresh UUIDs: RViz/Foxglove key a marker's
    identity on ``(ns, id)``, so a run's markers never share an id with an
    earlier run's and an ADD alone would leave the previous run's shelf on
    screen alongside the new one.

    :param viz_marker: Publisher whose current markers are republished on its
        own topic, DELETEALL first.
    """
    viz_marker.markers.markers.insert(0, Marker(action=Marker.DELETEALL))
    viz_marker.publish_markers()


def _load_or_train_shelf_model(
    model_path: Path, session: Session
) -> TrainedArbitraryShelfModel:
    """
    Load the model cached at *model_path*, training and caching one from the
    processed database if it doesn't exist yet.

    Every object type found on shelves takes part in training, so the circuit learns
    the joint spatial distribution across all of them. Training data comes from the
    processed database built by :func:`preprocess_sage10k_for_training`, so what is
    read here is already centred, filtered and grouped.

    :param model_path: Where the fitted model is exported to and, on a later run,
        loaded from instead of being refit. Training data is only queried and the
        RSPN only fit when no model exists at this path yet.
    :return: The cached or freshly fitted model.
    """
    if model_path.exists():
        return TrainedArbitraryShelfModel.load(model_path)

    shelves = load_shelves(session)
    shelf_layers = [layer for shelf in shelves for layer in shelf.layers]
    frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
    frequent_themes = _frequent_shelf_themes(shelves, keep_count=20)
    shelves = _coarsen_rare_object_types_of_shelves(shelves)
    shelves = _coarsen_rare_shelf_themes(shelves)

    rspn = RelationalProbabilisticCircuit(
        EGShelf,
        min_samples_per_leaf=min_samples_per_leaf_for,
        min_samples_per_quantile=MINIMUM_SAMPLES_PER_QUANTILE,
    ).fit([to_dao(shelf) for shelf in shelves])

    trained_model = TrainedArbitraryShelfModel(
        relational_probabilistic_circuit=rspn,
        frequent_object_types=frequent_types,
        frequent_theme_types=frequent_themes,
    )
    trained_model.save(model_path)
    return trained_model


def generate_shelf_with_arbitrary_objects(
    query, model: TrainedArbitraryShelfModel, session: Session
) -> SpawnedShelf:
    """
    Evaluate an EGShelf query against a trained RSPN and return a collision-free,
    spawned shelf, together with the model it was sampled from.

    The circuit is rooted at the shelf rather than at a loose layer, so a shelf's
    dimensions, its layers and their contents are drawn together, conditioned on
    whatever evidence *query* carries: a book-dominant shelf comes out with the
    proportions and contents book-dominant shelves were observed to have, a bottle-
    dominant one with a bottle-dominant shelf's.

    Mesh assets are drawn at random from the pool of shelf-object PLY files sharing
    the sampled object's generalized type; an object whose type has no mesh small
    enough for its layer is left out.

    .. note::
        Meshes keep their native size rather than being rescaled to the sampled
        scale, since the dataset's meshes are already modelled at real-world size.

    :param query: An EGShelf query to sample, e.g. built with
        :func:`~experiments.scene_generation_experiments.rspn_sampling.build_shelf_query`
        or :func:`~experiments.scene_generation_experiments.rspn_sampling.build_theme_shelf_query`.
    :raises NoSolutionFound: If the model gives *query* no probability.
    :return: The sampled, repaired, spawned shelf and the model it came from.
    """
    circuit = model.relational_probabilistic_circuit
    frequent_types = model.frequent_object_types

    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=circuit)
    backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    sample = next(iter(backend.evaluate(query)))

    needed_types = _mesh_candidate_types_for_shelf(sample, frequent_types)
    source_ids = _get_source_ids_for_objects(
        load_objects_of_types(session, needed_types), object_type=None
    )
    sample.source_ids = _coarsen_mesh_candidate_types(source_ids, frequent_types)

    layer_template = circuit.exchangeable_distribution_templates["layers"]
    resolver = InWorldLayoutResolver.for_shelf(
        sample,
        layer_template.template_distribution,
    )
    spawned_shelf = resolver.resolve()
    placed = sum(len(layer.object_bodies) for layer in spawned_shelf.layers)
    print(
        f"{sample.theme_dominant_type.value}: {len(spawned_shelf.layers)} "
        f"layers, {placed} objects standing "
        f"{resolver.dropped_body_count} dropped in repair)"
    )
    return spawned_shelf


def visualize_spawned_shelf(
    node,
    spawned_shelf: SpawnedShelf,
    visualization_backend: VisualizationBackend = VisualizationBackend.FOXGLOVE,
) -> VizMarkerPublisher:
    """
    Publish a spawned shelf's world as visualisation markers for
    :attr:`visualization_backend`.

    :param node: An active rclpy node used to publish visualisation markers.
    :param spawned_shelf: The shelf to publish, e.g. as returned by
        :func:`generate_shelf_with_arbitrary_objects`.
    :param visualization_backend: Viewer the markers are published for --
        Foxglove needs its mesh URIs rewritten onto a ``package://`` resource
        it can serve over its websocket, RViz loads ``file://`` resources
        directly.
    :return: The publisher, so a caller can keep it alive and re-trigger its TF
        publish for viewers that connect after the initial, one-shot publish --
        TF, unlike the marker publisher, is not transient-local.
    """
    publisher_type = (
        FoxgloveVizMarkerPublisher
        if visualization_backend is VisualizationBackend.FOXGLOVE
        else VizMarkerPublisher
    )
    viz_marker = publisher_type(
        _world=spawned_shelf.world,
        node=node,
        qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    viz_marker.with_tf_publisher()
    _publish_with_deleteall(viz_marker)
    return viz_marker


@dataclasses.dataclass
class ShelfTidyingAction(ActionDescription):
    shelf: EGShelf

    shelf_annotation: SpawnedShelf

    obj: EGObject2D

    model_registry: RelationalCircuitRegistry

    arm: EndEffector

    grasp_description: GraspDescription

    def perform(self):
        pass
        # navigation_map_obj = navigation_map_at_target(self.obj.body)
        # navigation_map_shelf = navigation_map_at_target(self.shelf_annotation.corpus)
        #
        # min_p = self.obj.body.collision.min_point
        # max_p = self.obj.body.collision.max_point
        #
        # x = min_p.x - 0.05
        # y = (min_p.y + max_p.y) / 2
        # z = (min_p.z + max_p.z) / 2
        #
        # pre_grasp_pose = Pose.from_xyz_rpy(x=x, y=y, z=z, reference_frame=self.obj.body)
        #
        # reach_query = a(MoveToReach)(
        #     target_pose_offset_robot=a(Pose2D)(
        #         x=..., y=..., yaw=..., reference_frame=None
        #     ),
        #     hip_rotation=0.0,
        #     target_pose_end_effector=pre_grasp_pose,
        #     grasp_description=a(GraspDescription)(
        #         approach_direction=ApproachDirection.FRONT,
        #         vertical_alignment=VerticalAlignment.NoAlignment,
        #         end_effector=variable(EndEffector, self.world.semantic_annotations),
        #         rotate_gripper=False,
        #     ),
        # )
        #
        # where_condition = translate_free_space_to_where_condition(
        #     navigation_map_obj.free_space_event,
        #     reach_query.expression,
        #     x_variable_name="MoveToReach.target_pose_offset_robot.x",
        #     y_variable_name="MoveToReach.target_pose_offset_robot.y",
        # )
        #
        # reach_action = reach_query.where(where_condition)
        #
        # pick_up = PickUpAction(object_designator=self.obj.body, arm=self.arm)
        #
        # # calculate where in the shelf a free pose would be for the obj.
        # # for every layer, new query, condition on everything in the shelf + all the sampled objects, new object unspecified except object_type of obj
        # # calculate log_mode of every layer, get free variables (position.x, position.y, rotation.z, with their corresponding probabilities
        # # take the pose that has the highest probability
        #
        # # navigateaction to this pose
        # # placeaction at the pose (maybe use Moveandplace for navigate + place, if easier)
        #
        # pass
