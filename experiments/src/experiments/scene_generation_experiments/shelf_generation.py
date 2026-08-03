from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter

from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.book_shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    load_all_objects,
    rclpy_node,
)
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.scene_generation.scene_schema import (
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    MeshCandidate,
    ObjectType,
)

_MIN_SAMPLES_PER_LEAF_FRACTION = 0.05
"""
Fraction of the training set required to create another split node when fitting
the shelf-layer RSPN, passed as ``min_samples_per_leaf`` to
:class:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`.

Each shelf object carries near-unique identifiers (``id``, ``source_id``), so
with the library default of one sample per leaf the object-level circuit grows
one leaf per training object; grounding then deep-copies that circuit once per
sampled object, which makes sampling run for minutes. A fraction bounds the
circuit's size instead.
"""


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
                object_2d
                if object_2d.object_type in frequent_types
                else dataclasses.replace(object_2d, object_type=ObjectType.OTHER)
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
        candidate
        if candidate.object_type in frequent_types
        else dataclasses.replace(candidate, object_type=ObjectType.OTHER)
        for candidate in candidates
    ]


def generate_shelf_with_arbitrary_objects(node) -> None:
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
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    shelf_layers, _ = _extract_shelf_layers_from_place_id(session, object_type=None)
    frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
    shelf_layers = _coarsen_rare_object_types(shelf_layers)
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(
        EGShelfLayer, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
    )
    rspn = rspn.fit(shelf_layer_data_access_objects)

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

    source_ids = _get_source_ids_for_objects(load_all_objects(session), object_type=None)
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
