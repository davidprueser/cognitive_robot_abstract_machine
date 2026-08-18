from __future__ import annotations

import argparse
import dataclasses
import os
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Optional

import trimesh
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from sqlalchemy.orm import Session
from visualization_msgs.msg import Marker, MarkerArray

from experiments.scene_generation_experiments.rspn_model_storage import (
    TrainedArbitraryShelfModel,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.utils import (
    MINIMUM_SAMPLES_PER_QUANTILE,
    _get_source_ids_for_objects,
    load_all_objects,
    load_shelves,
    rclpy_node,
    min_samples_per_leaf_for,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    draw_shelf,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
    ShelfType,
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
    Convert every mesh a marker references from OBJ to glTF (``.glb``) in a ROS
    package share directory and rewrite that marker's ``mesh_resource`` to the
    matching ``package://`` URI, so a browser-based Foxglove client can load it
    over the websocket instead of a ``file://`` path it cannot fetch.

    ``Mesh.from_ply_file`` exports OBJ with a separate ``.mtl`` sidecar for
    material/texture, which RViz2 and MuJoCo both read but Foxglove does not --
    Foxglove only loads material/texture from a self-contained glTF file. Converting
    here, rather than in ``Mesh.from_ply_file`` itself, keeps that OBJ default intact
    for every other consumer.

    Also pre-rotates each mesh marker's local orientation to cancel Foxglove's glTF
    up-axis correction; see :data:`_FOXGLOVE_GLTF_UP_AXIS_CORRECTION`.

    :param viz_marker: Publisher whose current markers are rewritten in place.
    """
    share_root = (
        Path(get_package_share_directory(_MESH_RESOURCE_PACKAGE))
        / _MESH_RESOURCE_SHARE_SUBDIR
    )
    for marker in viz_marker.markers.markers:
        if not marker.mesh_resource.startswith("file:///tmp/"):
            continue
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
            Rotation.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            )
            * _FOXGLOVE_GLTF_UP_AXIS_CORRECTION
        )
        (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ) = corrected.as_quat()


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
    viz_marker.publisher.publish(viz_marker.markers)


def generate_shelf_with_arbitrary_objects(
    node,
    shelf_type: ShelfType = ShelfType.BOOKCASE,
    layer_count: Optional[int] = None,
    placeholders_for_missing_meshes: bool = True,
    model_path: Path = Path.home()
    / "Documents"
    / "sage-10k-models"
    / "arbitrary_shelf_rspn.json",
    visualization_backend: VisualizationBackend = VisualizationBackend.FOXGLOVE,
) -> VizMarkerPublisher:
    """
    Train an RSPN on whole shelves and visualise a sampled, collision-free shelf of the
    requested kind via :attr:`visualization_backend`.

    The circuit is rooted at the shelf rather than at a loose layer, so a shelf's
    dimensions, its layers and their contents are drawn together and conditioned
    on the kind of shelf asked for: a bookcase comes out with the proportions and
    contents bookcases were observed to have, a cabinet with a cabinet's.

    Every object type found on shelves takes part, so the circuit learns the joint
    spatial distribution across all of them. Mesh assets are drawn at random from
    the pool of shelf-object PLY files sharing the sampled object's generalized
    type; an object whose type has no mesh small enough for its layer is left out.

    Training data comes from the processed database built by
    :func:`preprocess_sage10k_for_training`, so what is read here is already
    centred, filtered and grouped.

    .. note::
        Meshes keep their native size rather than being rescaled to the sampled
        scale, since the dataset's meshes are already modelled at real-world size.

    :param node: An active rclpy node used to publish visualisation markers.
    :param shelf_type: Kind of shelf to generate.
    :param layer_count: Number of layers to draw. Drawn from the kind of shelf
        when omitted, which is what makes a bookcase deeper-stacked than a cabinet.
    :param placeholders_for_missing_meshes: Stand a plain box in for objects whose
        type has no cached mesh, so a sparse render can be told apart from a sparse
        draw while the mesh library is incomplete.
    :param model_path: Where the fitted model is exported to and, on a later
        run, loaded from instead of being refit. Training data is only
        queried and the RSPN only fit when no model exists at this path yet.
    :param visualization_backend: Viewer the markers are published for --
        Foxglove needs its mesh URIs rewritten onto a ``package://`` resource
        it can serve over its websocket, RViz loads ``file://`` resources
        directly.
    :return: The publisher, so a caller can keep it alive and re-trigger its TF
        publish for viewers that connect after the initial, one-shot publish --
        TF, unlike the marker publisher, is not transient-local.
    """
    start = time.time()
    uri = os.environ.get("SAGE10K_PROCESSED_DATABASE_URI")
    assert (
        uri is not None
    ), "Please set the SAGE10K_PROCESSED_DATABASE_URI environment variable."
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    if model_path.exists():
        trained_model = TrainedArbitraryShelfModel.load(model_path)
    else:
        shelves = load_shelves(session)
        shelf_layers = [layer for shelf in shelves for layer in shelf.layers]
        frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
        shelves = _coarsen_rare_object_types_of_shelves(shelves)

        rspn = RelationalProbabilisticCircuit(
            EGShelf,
            min_samples_per_leaf=min_samples_per_leaf_for,
            min_samples_per_quantile=MINIMUM_SAMPLES_PER_QUANTILE,
        ).fit([to_dao(shelf) for shelf in shelves])

        trained_model = TrainedArbitraryShelfModel(
            relational_probabilistic_circuit=rspn,
            frequent_object_types=frequent_types,
        )
        trained_model.save(model_path)

    rspn = trained_model.relational_probabilistic_circuit
    frequent_types = trained_model.frequent_object_types

    shelf_sample = draw_shelf(rspn, shelf_type, layer_count)

    source_ids = _get_source_ids_for_objects(
        load_all_objects(session), object_type=None
    )
    shelf_sample.source_ids = _coarsen_mesh_candidate_types(source_ids, frequent_types)

    layer_template = rspn.exchangeable_distribution_templates["layers"]
    resolver = InWorldLayoutResolver.for_shelf(
        shelf_sample,
        layer_template.template_distribution,
        placeholders_for_missing_meshes=placeholders_for_missing_meshes,
    )
    spawned_shelf = resolver.resolve()
    # Counted after repair, which drops what it cannot separate, while the
    # placeholders were counted when the shelf was spawned. Reporting them as a
    # share of what survived would read as more stand-ins than there are objects.
    placed = sum(len(layer.object_bodies) for layer in spawned_shelf.layers)
    print(
        f"{shelf_type.value}: {len(spawned_shelf.layers)} layers, "
        f"{placed} objects standing "
        f"(spawned with {spawned_shelf.placeholder_count} placeholders, "
        f"{resolver.dropped_body_count} dropped in repair)"
    )
    world = spawned_shelf.world
    viz_marker = VizMarkerPublisher(
        _world=world,
        node=node,
        # depth=1 so a fresh subscriber's transient-local history only ever
        # holds the final, published-below markers, not the file:// URIs this
        # publisher writes on construction -- a Foxglove viewer cannot fetch
        # those, so that first sample must not linger.
        qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    viz_marker.with_tf_publisher()
    if visualization_backend is VisualizationBackend.FOXGLOVE:
        _rewrite_mesh_uris_for_foxglove(viz_marker)
    _publish_with_deleteall(viz_marker)
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")
    return viz_marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualization",
        type=VisualizationBackend,
        choices=list(VisualizationBackend),
        default=VisualizationBackend.FOXGLOVE,
        help="Viewer to publish markers for.",
    )
    parser.add_argument(
        "--shelf-type",
        type=ShelfType,
        choices=list(ShelfType),
        default=ShelfType.BOOKCASE,
        help="Kind of shelf to sample.",
    )
    args = parser.parse_args()

    with rclpy_node() as node:
        viz_marker = generate_shelf_with_arbitrary_objects(
            node,
            shelf_type=args.shelf_type,
            visualization_backend=args.visualization,
        )
        # The MarkerArray publisher is TRANSIENT_LOCAL, so it can still serve a
        # viewer that connects after this point, but TF is not: a viewer that
        # connects (or reconnects, e.g. on a page refresh) after the one-shot
        # publish in with_tf_publisher() would see no transforms at all, so TF
        # is re-published here on every tick to cover that.
        print(
            "Publishing until interrupted (Ctrl+C); keep this running while "
            "a viewer is connected."
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
