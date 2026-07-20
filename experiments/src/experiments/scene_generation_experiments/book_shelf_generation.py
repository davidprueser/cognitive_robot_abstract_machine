from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.scene_generation_experiments.data_preprocessing import (
    Sage10kSceneDownloader,
)
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    rclpy_node,
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
)


def _extract_shelf_layers_from_place_id(
    session: Session,
    edge_margin_fraction: float = 0.10,
    object_type: ObjectType | None = ObjectType.BOOK,
) -> tuple[list[EGShelfLayer], list[EGObjectDAO]]:
    """
    Load all scenes and group objects by the shelf declared in their
    ``place_id``.

    An object is considered a shelf occupant when ``"shelf"`` appears in its
    ``place_id`` (e.g. ``room_b12d7278_shelf_51fd4e1e``).  Shelf membership
    is determined purely from the dataset metadata rather than spatial
    bounding-box containment.

    After grouping, objects whose centre falls outside the shelf's XY footprint
    (inset by *edge_margin_fraction*) are discarded so that the learned RSPN
    does not place objects at positions where they would protrude from the shelf.

    :param edge_margin_fraction: Fraction of each shelf dimension to use as
        an inset margin on X and Y when filtering out-of-bounds objects.
    :param object_type: Only objects whose type equals this value are
        included. Defaults to :attr:`ObjectType.BOOK` to reproduce the
        original book-only behaviour; pass ``None`` to include every type.
    :return: Extracted shelf layers and all loaded object DAOs.
    """
    objects = session.scalars(
        select(EGObjectDAO)
        .options(
            joinedload(EGObjectDAO.scale),
            joinedload(EGObjectDAO.position),
            joinedload(EGObjectDAO.orientation),
        )
        .distinct()
        .limit(50000)
    ).all()

    shelves: list[EGObjectDAO] = [
        obj for obj in objects if obj.object_type == ObjectType.SHELF
    ]

    objects_by_place_id: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        objects_by_place_id[obj.place_id].append(obj)

    shelf_layers = []
    for shelf in shelves:
        members = objects_by_place_id[shelf.id]
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
                position=EGPoint2D(
                    x=obj.position.x - shelf.position.x,
                    y=obj.position.y - shelf.position.y,
                ),
                orientation=EGRotation(
                    x=obj.orientation.x, y=obj.orientation.y, z=obj.orientation.z
                ),
                source_id=obj.source_id,
            )
            objects_per_layer[label].append(relative_object)

        for _, layer_objects in objects_per_layer.items():
            shelf_layers.append(
                EGShelfLayer(
                    scale=EGScale(
                        width=shelf.scale.width, length=shelf.scale.length, height=0.02
                    ),
                    objects=layer_objects,
                )
            )

    return shelf_layers, objects


def generate_book_shelf(node) -> None:
    """
    Train an RSPN on shelf-layer data from the database, spawn a sampled
    arrangement into a world, repair collisions and off-surface placements
    directly in that world, and visualise the result via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    shelf_layers, training_objects = _extract_shelf_layers_from_place_id(session)
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(EGShelfLayer)
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

    sage10k_session = Session(create_engine(os.environ.get("SAGE10k_DATABASE_URI")))
    downloader = Sage10kSceneDownloader(session=sage10k_session)
    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        training_objects, downloader=downloader
    )
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=target_scale.length, width=target_scale.width),
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
