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


_SHELF_HEIGHT = 2.0
"""
Height, in metres, of the spawned shelf corpus.

The corpus height is what :meth:`EGShelf.spawn_in_world` spreads the layers
across, so it must be the shelf's own height, not a layer's slab thickness.
The extracted layers only carry a fixed slab thickness, so a real corpus
height is supplied here until the shelf's own height is carried through the
training data.
"""


def _extract_shelf_layers_from_place_id(
    session: Session,
    edge_margin_fraction: float = 0.10,
    object_type: ObjectType | None = ObjectType.BOOK,
    room_count: int = DEFAULT_TRAINING_ROOM_COUNT,
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
    :param room_count: Maximum number of distinct rooms to sample.
    :return: Extracted shelf layers and all loaded object DAOs.
    """
    objects = objects_for_rooms(session, sampled_room_ids(session, room_count))
    return shelf_layers_from_objects(objects, edge_margin_fraction, object_type), objects


def shelf_layers_from_objects(
    objects: list[EGObjectDAO],
    edge_margin_fraction: float = 0.10,
    object_type: ObjectType | None = ObjectType.BOOK,
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
        EGShelfLayer, min_samples_per_leaf=min_samples_per_leaf_for(sum(len(layer.objects) for layer in shelf_layers))
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
