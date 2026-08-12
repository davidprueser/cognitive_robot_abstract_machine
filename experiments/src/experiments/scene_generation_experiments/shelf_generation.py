from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

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
    _get_source_ids_for_objects,
    load_all_objects,
    rclpy_node,
    min_samples_per_leaf_for,
    objects_for_rooms,
    sampled_room_ids,
)
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    MeshCandidate,
    ObjectType,
    EGObject2D,
    wrap_angle_degrees,
)


def _extract_shelf_layers_from_place_id(
    session: Session,
    edge_margin_fraction: float = 0.10,
    object_type: ObjectType | None = ObjectType.BOOK,
) -> tuple[list[EGShelfLayer], list[EGObjectDAO]]:
    """
    Load a random sample of rooms and group their objects by the shelf
    declared in their ``place_id``.

    An object is considered a shelf occupant when ``"shelf"`` appears in its
    ``place_id`` (e.g. ``room_b12d7278_shelf_51fd4e1e``).  Shelf membership
    is determined purely from the dataset metadata rather than spatial
    bounding-box containment.

    After grouping, objects whose centre falls outside the shelf's XY footprint
    (inset by *edge_margin_fraction*) are discarded so that the learned RSPN
    does not place objects at positions where they would protrude from the shelf.

    Rooms are sampled first, then loaded in full, so a shelf's contents are
    never truncated by a row-count limit on the underlying object query.

    :param edge_margin_fraction: Fraction of each shelf dimension to use as
        an inset margin on X and Y when filtering out-of-bounds objects.
    :param object_type: Only objects whose type equals this value are
        included. Defaults to :attr:`ObjectType.BOOK` to reproduce the
        original book-only behaviour; pass ``None`` to include every type.
    :return: Extracted shelf layers and all loaded object DAOs.
    """
    objects = objects_for_rooms(session, sampled_room_ids(session))
    return (
        shelf_layers_from_objects(objects, edge_margin_fraction, object_type),
        objects,
    )


def shelf_layers_from_objects(
    objects: list[EGObjectDAO],
    edge_margin_fraction: float = 0.10,
    object_type: Optional[ObjectType] = ObjectType.BOOK,
) -> list[EGShelfLayer]:
    """
    Group already-loaded *objects* into shelf layers, flattened across shelves.

    :param objects: Object DAOs of the rooms to extract shelves from.
    :param edge_margin_fraction: Fraction of each shelf dimension to use as an
        inset margin on X and Y when filtering out-of-bounds objects.
    :param object_type: Only objects whose type equals this value are included;
        pass ``None`` to include every type.
    :return: The extracted shelf layers.
    """
    return [
        layer
        for shelf_layers in shelf_layers_by_shelf(
            objects, edge_margin_fraction, object_type
        )
        for layer in shelf_layers
    ]


def shelf_layers_by_shelf(
    objects: list[EGObjectDAO],
    edge_margin_fraction: float = 0.10,
    object_type: ObjectType | None = ObjectType.BOOK,
) -> list[list[EGShelfLayer]]:
    """
    Group already-loaded *objects* into shelf layers, one list per shelf, so a
    caller that has loaded a room sample once can fit several circuits from it
    instead of re-querying the database per circuit.

    :param objects: Object DAOs of the rooms to extract shelves from.
    :param edge_margin_fraction: Fraction of each shelf dimension to use as an
        inset margin on X and Y when filtering out-of-bounds objects.
    :param object_type: Only objects whose type equals this value are included;
        pass ``None`` to include every type.
    :return: The extracted layers, grouped per shelf, so a caller can draw how
        many layers a generated shelf should have from the real distribution
        rather than fixing it.
    """
    shelves: list[EGObjectDAO] = [
        obj for obj in objects if obj.object_type == ObjectType.SHELF
    ]

    objects_by_place_id: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        objects_by_place_id[obj.place_id].append(obj)

    shelf_layers: list[list[EGShelfLayer]] = []
    for shelf in shelves:
        members = objects_by_place_id[shelf.id]
        if not members:
            continue
        max_relative_x = shelf.scale.width / 2 * (1 - edge_margin_fraction)
        max_relative_y = shelf.scale.length / 2 * (1 - edge_margin_fraction)

        within_bounds = [
            obj
            for obj in members
            if (object_type is None or obj.object_type == object_type)
            and abs(obj.position.x - shelf.position.x) <= max_relative_x
            and abs(obj.position.y - shelf.position.y) <= max_relative_y
        ]
        if not within_bounds:
            continue

        z_positions = np.array([obj.position.z for obj in within_bounds]).reshape(-1, 1)
        labels = DBSCAN(eps=0.05, min_samples=1).fit_predict(z_positions)

        objects_per_layer: defaultdict[int, list[EGObject2D]] = defaultdict(list)
        for obj, label in zip(within_bounds, labels):
            # Store the pose in the shelf's content frame (the shelf yaw plus
            # EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES), the same frame
            # :meth:`EGShelf.spawn_in_world` builds the corpus in. Rotating the
            # world offset into that frame puts the contents' wide face spread on
            # the corpus's wide axis; without it the spread lands on the shelf's
            # shallow depth axis and contents overflow front and back.
            content_frame_yaw = (
                shelf.orientation.z + EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES
            )
            shelf_local_offset = EGPoint2D(
                x=obj.position.x - shelf.position.x,
                y=obj.position.y - shelf.position.y,
            ).rotated_into_frame(content_frame_yaw)
            relative_object = EGObject2D(
                id=obj.id,
                room_id=obj.room_id,
                place_id=obj.place_id,
                object_type=obj.object_type,
                scale=EGScale(
                    width=obj.scale.width,
                    length=obj.scale.length,
                    height=obj.scale.height,
                ),
                position=EGPoint2D(x=shelf_local_offset.x, y=shelf_local_offset.y),
                orientation=EGRotation(
                    x=obj.orientation.x,
                    y=obj.orientation.y,
                    z=wrap_angle_degrees(obj.orientation.z - content_frame_yaw),
                ),
                source_id=obj.source_id,
            )

            objects_per_layer[label].append(relative_object)

        shelf_layers.append(
            [
                EGShelfLayer(
                    scale=EGScale(
                        width=shelf.scale.width, length=shelf.scale.length, height=0.02
                    ),
                    objects=layer_objects,
                )
                for layer_objects in objects_per_layer.values()
            ]
        )

    return shelf_layers


def _frequent_object_types(
    shelf_layers: list[EGShelfLayer],
    keep_count: int,
) -> set[ObjectType]:
    """
    Return the *keep_count* most frequent object types across all objects in
    *shelf_layers*.

    :param shelf_layers: Layers whose objects' types are counted.
    :param keep_count: Number of distinct, most frequent object types to
        return.
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
    Return new shelf layers where every object's type outside the *keep_count*
    most frequent types (across all objects in *shelf_layers*) is replaced with
    ``ObjectType.OTHER``.

    :param shelf_layers: Layers whose objects' types should be
        coarsened.
    :param keep_count: Number of distinct, most frequent object types to
        leave unchanged.
    :return: New EGShelfLayer instances with coarsened object types; all
        other fields (position, scale, orientation, source_id, ...) are
        unchanged.
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


def generate_shelf_with_arbitrary_objects(
    node,
    model_path: Path = Path.home()
    / "Documents"
    / "sage-10k-models"
    / "arbitrary_shelf_rspn.json",
) -> None:
    """
    Train an RSPN on all object types found on shelves in the dataset and
    visualise a sampled, collision-free arrangement via RViz.

    Unlike :func:`book_shelf_generation.generate_book_shelf`, this demo
    includes every object type found on shelves in the training data — books,
    cups, plants, containers, and more — so the RSPN learns the joint
    spatial distribution across all of them. Mesh assets are drawn at random
    from the pool of available shelf-object PLY files that share the same
    (generalized) object type as the object sampled by the RSPN; if no mesh
    of that type is available, a mesh is drawn from the full pool instead.

    .. note::
        Meshes are rescaled so their bounding box matches the RSPN-sampled
        scale, and collisions are resolved against those real meshes. A mesh
        whose native aspect ratio differs from the sampled scale is stretched
        to fit, which can look unnatural for high-variance types (e.g. plants,
        containers).

    :param node: An active rclpy node used to publish visualisation markers.
    :param model_path: Where the fitted model is exported to and, on a later
        run, loaded from instead of being refit. Training data is only
        queried and the RSPN only fit when no model exists at this path yet.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    if model_path.exists():
        trained_model = TrainedArbitraryShelfModel.load(model_path)
    else:
        shelf_layers, _ = _extract_shelf_layers_from_place_id(session, object_type=None)
        frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
        shelf_layers = _coarsen_rare_object_types(shelf_layers)
        shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

        rspn = RelationalProbabilisticCircuit(
            EGShelfLayer,
            min_samples_per_leaf=min_samples_per_leaf_for(
                sum(len(layer.objects) for layer in shelf_layers)
            ),
        ).fit(shelf_layer_data_access_objects)

        trained_model = TrainedArbitraryShelfModel(
            relational_probabilistic_circuit=rspn,
            frequent_object_types=frequent_types,
        )
        trained_model.save(model_path)

    rspn = trained_model.relational_probabilistic_circuit
    frequent_types = trained_model.frequent_object_types

    probability_backend = probabilistic_backend(rspn)

    objects_per_layer = 3
    layer_count = 4
    reference_layer = next(
        iter(probability_backend.evaluate(build_free_layer_query(objects_per_layer)))
    )
    target_scale = reference_layer.scale
    remaining_layers = [
        next(
            iter(
                probability_backend.evaluate(
                    build_layer_query_with_fixed_scale(objects_per_layer, target_scale)
                )
            )
        )
        for _ in range(layer_count - 1)
    ]
    sampled_layers = [reference_layer] + remaining_layers

    source_ids = _get_source_ids_for_objects(
        load_all_objects(session), object_type=None
    )
    source_ids = _coarsen_mesh_candidate_types(source_ids, frequent_types)
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=target_scale.length, width=target_scale.width),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        source_ids=source_ids,
    )

    spawned_shelf = InWorldLayoutResolver.for_shelf(shelf_sample, rspn).resolve()
    world = spawned_shelf.world
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_shelf_with_arbitrary_objects(node)
