from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy.orm import Session

from experiments.scene_generation_experiments.utils import (
    DEFAULT_TRAINING_ROOM_COUNT,
    min_samples_per_leaf_for,
    _get_source_ids_for_objects,
    load_all_objects,
    objects_for_rooms,
    rclpy_node,
    sampled_room_ids,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
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
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    ObjectType,
    wrap_angle_degrees,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )


def generate_book_shelf(node, downloader: Sage10kSceneDownloader | None = None) -> None:
    """
    Train an RSPN on shelf-layer data from the database, spawn a sampled
    arrangement into a world, repair collisions and off-surface placements
    directly in that world, and visualise the result via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    :param downloader: When given, book meshes are downloaded on demand until
        the candidate pool is filled. Left as ``None`` the pool is whatever is
        already cached, which keeps the demo fast for iterative testing; pass a
        downloader for a final demo that needs a broad mesh pool.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    shelf_layers, _ = _extract_shelf_layers_from_place_id(session)
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

    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        load_all_objects(session), downloader=downloader
    )
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(
            height=_SHELF_HEIGHT,
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
