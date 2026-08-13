from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter
from pathlib import Path

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
    load_shelf_layers,
    rclpy_node,
    min_samples_per_leaf_for,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_layer_query,
    probabilistic_backend,
)
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


def generate_shelf_with_arbitrary_objects(
    node,
    model_path: Path = Path.home()
    / "Documents"
    / "sage-10k-models"
    / "arbitrary_shelf_rspn.json",
) -> None:
    """
    Train an RSPN on all object types found on shelves in the dataset and visualise a
    sampled, collision-free arrangement via RViz.

    Unlike :func:`book_shelf_generation.generate_book_shelf`, this demo
    includes every object type found on shelves in the training data — books,
    cups, plants, containers, and more — so the RSPN learns the joint
    spatial distribution across all of them. Mesh assets are drawn at random
    from the pool of available shelf-object PLY files that share the same
    (generalized) object type as the object sampled by the RSPN; if no mesh
    of that type is available, a mesh is drawn from the full pool instead.

    Training data comes from the processed database built by
    :func:`preprocess_sage10k_for_training`, so the layers read here are
    already centred, filtered and grouped.

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
        shelf_layers = load_shelf_layers(session)
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
        iter(
            probability_backend.evaluate(
                build_layer_query(free_count=objects_per_layer)
            )
        )
    )
    target_scale = reference_layer.scale
    remaining_layers = [
        next(
            iter(
                probability_backend.evaluate(
                    build_layer_query(free_count=objects_per_layer, scale=target_scale)
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
