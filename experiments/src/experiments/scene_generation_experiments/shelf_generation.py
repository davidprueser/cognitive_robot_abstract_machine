from __future__ import annotations

import dataclasses
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

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

    Every layer of every shelf is coarsened in one pass, so one shared set of
    frequent types survives. Coarsening each shelf on its own would let a type be
    kept on one shelf and collapsed on another, and the mesh pool -- coarsened
    once against the same set -- would then find no mesh for it.

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


def generate_shelf_with_arbitrary_objects(
    node,
    shelf_type: ShelfType = ShelfType.BOOKCASE,
    layer_count: Optional[int] = None,
    objects_per_layer: int = 3,
    placeholders_for_missing_meshes: bool = True,
    model_path: Path = Path.home()
    / "Documents"
    / "sage-10k-models"
    / "arbitrary_shelf_rspn.json",
) -> None:
    """
    Train an RSPN on whole shelves and visualise a sampled, collision-free shelf
    of the requested kind via RViz.

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
    :param objects_per_layer: Number of objects to draw onto each layer.
    :param placeholders_for_missing_meshes: Stand a plain box in for objects whose
        type has no cached mesh, so a sparse render can be told apart from a sparse
        draw while the mesh library is incomplete.
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
        shelves = load_shelves(session)
        shelf_layers = [layer for shelf in shelves for layer in shelf.layers]
        frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
        shelves = _coarsen_rare_object_types_of_shelves(shelves)

        rspn = RelationalProbabilisticCircuit(
            EGShelf,
            min_samples_per_leaf=min_samples_per_leaf_for(
                sum(len(layer.objects) for layer in shelf_layers)
            ),
        ).fit([to_dao(shelf) for shelf in shelves])

        trained_model = TrainedArbitraryShelfModel(
            relational_probabilistic_circuit=rspn,
            frequent_object_types=frequent_types,
        )
        trained_model.save(model_path)

    rspn = trained_model.relational_probabilistic_circuit
    frequent_types = trained_model.frequent_object_types

    shelf_sample = draw_shelf(rspn, shelf_type, objects_per_layer, layer_count)

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
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_shelf_with_arbitrary_objects(node)
