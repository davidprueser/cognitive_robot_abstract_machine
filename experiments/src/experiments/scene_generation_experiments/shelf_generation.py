from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter

from sqlalchemy.orm import Session

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.book_shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.utils import rclpy_node, _get_source_ids_for_objects
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
    resolve_shelf_collisions,
)
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
        The RSPN learns object *scale* from training data, but PLY meshes are
        rendered at their native size. Object types with high size variance
        (e.g. plants, containers) may produce visual overlaps even after
        collision resolution.

    :param node: An active rclpy node used to publish visualisation markers.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    shelf_layers, training_objects = _extract_shelf_layers_from_place_id(
        session, object_type=None
    )
    frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
    shelf_layers = _coarsen_rare_object_types(shelf_layers)
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(EGShelfLayer)
    rspn = rspn.fit(shelf_layer_data_access_objects)

    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rspn)
    probability_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)

    reference_layer = next(iter(probability_backend.evaluate(build_free_layer_query(3))))
    target_scale = reference_layer.scale
    remaining_layers = [
        next(iter(probability_backend.evaluate(build_layer_query_with_fixed_scale(3, target_scale))))
        for _ in range(3)
    ]
    sampled_layers = resolve_shelf_collisions([reference_layer] + remaining_layers, rspn)

    source_ids = _get_source_ids_for_objects(training_objects, object_type=None)
    source_ids = _coarsen_mesh_candidate_types(source_ids, frequent_types)
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=target_scale.length, width=target_scale.width),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        source_ids=source_ids,
    )

    world = shelf_sample.create_in_world()
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_shelf_with_arbitrary_objects(node)
