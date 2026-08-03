from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from experiments.scene_generation_experiments.utils import (
    DEFAULT_TRAINING_ROOM_COUNT,
    _get_source_ids_for_objects,
    load_all_objects,
    objects_for_rooms,
    rclpy_node,
    sampled_room_ids,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_free_table_query,
    sample_chair_count,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGTableWithChairs,
    ObjectType,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )

_MIN_SAMPLES_PER_LEAF_FRACTION = 0.05
"""
Fraction of the training set required to create another split node when fitting
the table-with-chairs RSPN, passed as ``min_samples_per_leaf`` to
:class:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`.

Each chair carries near-unique identifiers (``id``, ``source_id``), so with the
library default of one sample per leaf the chair-level circuit grows one leaf
per training chair; grounding then deep-copies that circuit once per sampled
chair, which makes sampling run for minutes. A fraction bounds the circuit's
size instead.
"""


def _distance_between(first: EGObjectDAO, second: EGObjectDAO) -> float:
    """
    Euclidean distance between the XY positions of two objects.
    """
    return math.dist(
        (first.position.x, first.position.y), (second.position.x, second.position.y)
    )


def _build_chair(chair: EGObjectDAO, table: EGObjectDAO) -> EGChair:
    """
    Build an :class:`EGChair` for *chair*, with its pose expressed relative to
    *table*.
    """
    return EGChair(
        id=chair.id,
        room_id=chair.room_id,
        object_type=chair.object_type,
        scale=EGScale(
            width=chair.scale.width,
            length=chair.scale.length,
            height=chair.scale.height,
        ),
        relative_pose=EGRelativePolarPose.from_absolute_poses(
            chair.position.x,
            chair.position.y,
            chair.orientation.z,
            table.position.x,
            table.position.y,
            table.orientation.z,
        ),
        source_id=chair.source_id,
    )


def _extract_table_chair_groups_from_spatial_proximity(
    session: Session,
    max_distance_from_table: float = 1.5,
    object_type: ObjectType = ObjectType.CHAIR,
    room_count: int = DEFAULT_TRAINING_ROOM_COUNT,
) -> tuple[list[EGTableWithChairs], list[EGObjectDAO]]:
    """
    Load a random sample of rooms and group each one's chairs with the
    nearest table in the same room, since chairs do not carry a ``place_id``
    link to their table in the source data the way shelf contents link to
    their shelf.

    Tables that end up with no assigned chairs are dropped: the RSPN's
    feature extractor decides whether a relation gets an exchangeable
    template and aggregation features at all by inspecting only the first
    training instance's collection, so a bare table as the first instance
    would silently suppress chair modelling entirely.

    Rooms are sampled first, then loaded in full, so a room's tables and
    chairs are never truncated by a row-count limit on the underlying object
    query.

    :param session: Database session to query objects from.
    :param max_distance_from_table: Maximum Euclidean distance, in metres,
        between a chair and a table for the chair to be assigned to it.
    :param object_type: Only objects whose type equals this value are
        considered chairs. Defaults to :attr:`ObjectType.CHAIR`.
    :param room_count: Maximum number of distinct rooms to sample.
    :return: Extracted table-with-chairs groups and all loaded object DAOs.
    """
    objects = objects_for_rooms(session, sampled_room_ids(session, room_count))

    tables_by_room: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        if obj.object_type == ObjectType.TABLE:
            tables_by_room[obj.room_id].append(obj)

    chairs_by_table_id: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        if obj.object_type != object_type:
            continue
        candidate_tables = tables_by_room.get(obj.room_id, [])
        if not candidate_tables:
            continue
        nearest_table = min(
            candidate_tables, key=lambda table: _distance_between(obj, table)
        )
        if _distance_between(obj, nearest_table) > max_distance_from_table:
            continue
        chairs_by_table_id[nearest_table.id].append(obj)

    table_chair_groups = []
    for tables in tables_by_room.values():
        for table in tables:
            chairs = chairs_by_table_id.get(table.id, [])
            if not chairs:
                continue
            table_chair_groups.append(
                EGTableWithChairs(
                    position=EGPoint2D(x=table.position.x, y=table.position.y),
                    scale=EGScale(
                        width=table.scale.width,
                        length=table.scale.length,
                        height=table.scale.height,
                    ),
                    orientation=EGRotation(
                        x=table.orientation.x,
                        y=table.orientation.y,
                        z=table.orientation.z,
                    ),
                    chairs=[_build_chair(chair, table) for chair in chairs],
                )
            )

    return table_chair_groups, objects


def generate_table_with_chairs(
    node, downloader: Sage10kSceneDownloader | None = None
) -> None:
    """
    Train an RSPN on table-with-chairs data from the database, spawn a sampled
    arrangement into a world, repair chair collisions directly in that world,
    and visualise the result via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    :param downloader: When given, chair meshes are downloaded on demand until
        the candidate pool is filled. Left as ``None`` the pool is whatever is
        already cached, which keeps the demo fast for iterative testing; pass a
        downloader for a final demo that needs a broad mesh pool.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    table_chair_groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)
    data_access_objects = [to_dao(group) for group in table_chair_groups]

    rspn = RelationalProbabilisticCircuit(
        EGTableWithChairs, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
    )
    rspn = rspn.fit(data_access_objects)

    probability_backend = probabilistic_backend(rspn)

    chair_count = sample_chair_count([len(group.chairs) for group in table_chair_groups])
    sample = next(iter(probability_backend.evaluate(build_free_table_query(chair_count))))

    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        load_all_objects(session), object_type=ObjectType.CHAIR, downloader=downloader
    )
    sample.position = EGPoint2D(x=0.0, y=0.0)
    sample.orientation = EGRotation(x=0.0, y=0.0, z=0.0)
    sample.source_ids = source_ids_for_sampled_objects

    spawned_group = InWorldLayoutResolver.for_table_with_chairs(sample, rspn).resolve()
    world = spawned_group.world
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating table-with-chairs sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_table_with_chairs(node)
