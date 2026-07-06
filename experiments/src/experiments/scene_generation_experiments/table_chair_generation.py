from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.scene_generation_experiments.utils import rclpy_node, _get_source_ids_for_objects
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.model_registries import RelationalCircuitRegistry
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
    EGSize,
    EGTableWithChairs,
    ObjectType,
)


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
        scale=EGSize(
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
    type_predicate: Callable[[ObjectType], bool] = lambda object_type: object_type
    == ObjectType.CHAIR,
) -> tuple[list[EGTableWithChairs], list[EGObjectDAO]]:
    """
    Load all scenes and group chairs with the nearest table in the same room,
    since chairs do not carry a ``place_id`` link to their table in the source
    data the way shelf contents link to their shelf.

    Tables that end up with no assigned chairs are dropped: the RSPN's
    feature extractor decides whether a relation gets an exchangeable
    template and aggregation features at all by inspecting only the first
    training instance's collection, so a bare table as the first instance
    would silently suppress chair modelling entirely.

    :param session: Database session to query objects from.
    :param max_distance_from_table: Maximum Euclidean distance, in metres,
        between a chair and a table for the chair to be assigned to it.
    :param type_predicate: Called with each object's :class:`ObjectType`;
        only objects for which this returns ``True`` are considered
        chairs. Defaults to matching :attr:`ObjectType.CHAIR`.
    :return: Extracted table-with-chairs groups and all loaded object DAOs.
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

    tables_by_room: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        if obj.object_type == ObjectType.TABLE:
            tables_by_room[obj.room_id].append(obj)

    chairs_by_table_id: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        if not type_predicate(obj.object_type):
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
                    scale=EGSize(
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


def generate_table_with_chairs(node) -> None:
    """
    Train an RSPN on table-with-chairs data from the database, sample a
    collision-free arrangement, and visualise it via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    table_chair_groups, training_objects = (
        _extract_table_chair_groups_from_spatial_proximity(session)
    )
    data_access_objects = [to_dao(group) for group in table_chair_groups]

    rspn = RelationalProbabilisticCircuit(EGTableWithChairs)
    rspn = rspn.fit(data_access_objects)

    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rspn)
    probability_backend = ProbabilisticBackend(
        model_registry=registry, number_of_samples=1
    )

    from experiments.scene_generation_experiments.table_chair_collision_resolution import (
        build_free_table_query,
        resolve_table_chair_collisions,
        sample_chair_count,
    )

    chair_count = sample_chair_count([len(group.chairs) for group in table_chair_groups])
    sample = next(iter(probability_backend.evaluate(build_free_table_query(chair_count))))
    sample = resolve_table_chair_collisions(sample, rspn)

    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        training_objects, type_predicate=lambda object_type: object_type == ObjectType.CHAIR
    )
    sample.position = EGPoint2D(x=0.0, y=0.0)
    sample.orientation = EGRotation(x=0.0, y=0.0, z=0.0)
    sample.source_ids = source_ids_for_sampled_objects

    world = sample.create_in_world()
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating table-with-chairs sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_table_with_chairs(node)
