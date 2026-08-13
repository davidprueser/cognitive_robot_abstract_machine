from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from experiments.scene_generation_experiments.utils import (
    min_samples_per_leaf_for,
    _get_source_ids_for_objects,
    load_all_objects,
    load_shelf_layers,
    rclpy_node,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
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
    ObjectType,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )

SHELF_HEIGHT = 2.0
"""
Height, in metres, of the shelf the sampled layers are spawned into.
"""


def generate_book_shelf(node, downloader: Sage10kSceneDownloader | None = None) -> None:
    """
    Train an RSPN on the stored shelf layers' books, spawn a sampled arrangement into a
    world, repair collisions and off-surface placements directly in that world, and
    visualise the result via RViz markers.

    Training data comes from the processed database built by
    :func:`preprocess_sage10k_for_training`, reduced to the books on each
    layer.

    :param node: An active rclpy node used to publish visualisation
        markers.
    :param downloader: When given, book meshes are downloaded on demand until
        the candidate pool is filled. Left as ``None`` the pool is whatever is
        already cached, which keeps the demo fast for iterative testing; pass a
        downloader for a final demo that needs a broad mesh pool.
    """
    start = time.time()
    uri = os.environ.get("SAGE10K_PROCESSED_DATABASE_URI")
    assert (
        uri is not None
    ), "Please set the SAGE10K_PROCESSED_DATABASE_URI environment variable."
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    shelf_layers = load_shelf_layers(session, object_type=ObjectType.BOOK)
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(
        EGShelfLayer,
        min_samples_per_leaf=min_samples_per_leaf_for(
            sum(len(layer.objects) for layer in shelf_layers)
        ),
    )
    rspn = rspn.fit(shelf_layer_data_access_objects)

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

    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        load_all_objects(session), downloader=downloader
    )
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(
            height=SHELF_HEIGHT,
            length=target_scale.length,
            width=target_scale.width,
        ),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        source_ids=source_ids_for_sampled_objects,
    )

    spawned_shelf = InWorldLayoutResolver.for_shelf(shelf_sample, rspn).resolve()
    world = spawned_shelf.world
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_book_shelf(node)
