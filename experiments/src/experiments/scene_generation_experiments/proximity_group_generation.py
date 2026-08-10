from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sqlalchemy.orm import Session

from experiments.scene_generation_experiments.utils import (
    DEFAULT_MINIMUM_CANDIDATES_PER_TYPE,
    DEFAULT_TRAINING_ROOM_COUNT,
    _get_source_ids_for_objects,
    objects_for_rooms,
    objects_of_type,
    rclpy_node,
    sampled_room_ids,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.proximity_group_collision_resolution import (
    build_free_group_query,
    sample_member_count,
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
    EGGroupMember,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGProximityGroup,
    ObjectType,
    PlaceId,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )



def _distance_between(first: EGObjectDAO, second: EGObjectDAO) -> float:
    """
    Euclidean distance between the XY positions of two objects.
    """
    return math.dist(
        (first.position.x, first.position.y), (second.position.x, second.position.y)
    )


def _build_member(member: EGObjectDAO, anchor: EGObjectDAO) -> EGGroupMember:
    """
    Build an :class:`EGGroupMember` for *member*, with its pose expressed relative to
    *anchor*.
    """
    return EGGroupMember(
        id=member.id,
        room_id=member.room_id,
        object_type=member.object_type,
        scale=EGScale(
            width=member.scale.width,
            length=member.scale.length,
            height=member.scale.height,
        ),
        relative_pose=EGRelativePolarPose.from_absolute_poses(
            member.position.x,
            member.position.y,
            member.orientation.z,
            anchor.position.x,
            anchor.position.y,
            anchor.orientation.z,
        ),
        source_id=member.source_id,
    )


DEFAULT_GROUP_DIAMETER = 2.0
"""
Widest, in metres, that a proximity group may span.

Chosen by sweeping the threshold over 9439 floor objects and reading off how the
clusters come out:

===========  ========  ==========  =========  ======  ==================
 diameter    clusters  singletons  median      p90     chairs per table
===========  ========  ==========  =========  ======  ==================
 1.0 m          5320      43.2 %          2       3               0.25
 1.5 m          3662      15.6 %          2       4               0.48
 2.0 m          2779       7.0 %          3       5               0.65
 3.0 m          1804       2.8 %          5       9               0.98
 4.0 m          1293       1.5 %          7      12               1.17
===========  ========  ==========  =========  ======  ==================

Both ends fail differently. Too tight and a table loses the chairs around it --
they sit 0.8 m out on either side, so the set already spans 1.6 m. Too loose and
a furnished room stops separating at all: at 4 m barely one cluster in seventy
is a lone object and the largest holds 47 pieces, which is the whole room.

Two metres is the smallest value that still admits a table with chairs on
opposite sides, and it keeps nine groups in ten at five pieces or fewer.
"""


def _extract_proximity_groups(
    session: Session,
    group_diameter: float = DEFAULT_GROUP_DIAMETER,
    room_count: int = DEFAULT_TRAINING_ROOM_COUNT,
) -> tuple[list[EGProximityGroup], list[EGObjectDAO]]:
    """
    Load a random sample of rooms and cluster each one's floor objects into
    proximity groups.

    Rooms are sampled first, then loaded in full, so a room's clusters are never
    truncated by a row-count limit on the underlying object query.

    :param session: Database session to query objects from.
    :param group_diameter: Widest a group may span.
    :param room_count: Maximum number of distinct rooms to sample.
    :return: Extracted proximity groups and all loaded object DAOs.
    """
    objects = objects_for_rooms(session, sampled_room_ids(session, room_count))
    return proximity_groups_from_objects(objects, group_diameter), objects


def _footprint_area(obj: EGObjectDAO) -> float:
    """
    Floor area an object covers, used to pick a cluster's anchor.
    """
    return obj.scale.width * obj.scale.length


def _build_group(cluster: list[EGObjectDAO]) -> EGProximityGroup:
    """
    Turn one cluster of co-located objects into a group anchored on its largest
    member.

    :param cluster: The co-located objects, at least one.
    :return: The group, with every non-anchor object posed relative to the
        anchor.
    """
    anchor = max(cluster, key=_footprint_area)
    return EGProximityGroup(
        position=EGPoint2D(x=anchor.position.x, y=anchor.position.y),
        scale=EGScale(
            width=anchor.scale.width,
            length=anchor.scale.length,
            height=anchor.scale.height,
        ),
        orientation=EGRotation(
            x=anchor.orientation.x,
            y=anchor.orientation.y,
            z=anchor.orientation.z,
        ),
        object_type=anchor.object_type,
        members=[
            _build_member(member, anchor) for member in cluster if member is not anchor
        ],
    )


def proximity_groups_from_objects(
    objects: list[EGObjectDAO],
    group_diameter: float = DEFAULT_GROUP_DIAMETER,
) -> list[EGProximityGroup]:
    """
    Cluster already-loaded *objects* into proximity groups, one per spatially
    connected set of floor objects, so a caller that has loaded a room sample
    once can fit several circuits from it instead of re-querying the database.

    This is what makes object-to-object arrangement learnable at all. A room
    layout circuit models its pieces as exchangeable, so given the room's
    aggregations the pieces are independent and "the counter belongs beside the
    sink" cannot be expressed -- no aggregation fixes that, being a room-level
    scalar broadcast identically to every piece. Clustering lifts the recurring
    arrangements out of the data so the circuit models each one as a unit.

    Clusters are found per room, since stored positions are room-local and two
    objects in different rooms would otherwise look adjacent. Only floor-resting
    objects take part: two thirds of the dataset's :attr:`ObjectType.TABLE` rows
    are decimetre-sized items lying *on* a table -- a ``"tablecloth"`` is
    generalized to :attr:`ObjectType.TABLE` by keyword -- and they sit at
    practically the same position as the table they rest on, so they would drag
    the learned anchor scale down to a small box.

    Singleton clusters are returned too: most floor objects stand alone, and a
    group of one is the honest description of that. See
    :func:`groups_for_circuit_training` for why they are nonetheless kept out of
    the fitted circuit.

    Clusters are bounded by their own width rather than grown outwards from
    each object's neighbourhood. Density-based clustering links two objects
    whenever some third lies within reach of both, so in a furnished room every
    piece is transitively connected and the whole room collapses into one group
    -- measured at roughly 12 pieces per cluster, yielding sampled groups of 26
    members around a single chair. Complete linkage instead keeps *every* pair
    in a group within *group_diameter*, which is the property "these things
    stand together" actually means.

    :param objects: Object DAOs of the rooms to extract groups from.
    :param group_diameter: Widest a group may span, measured between its two
        furthest-apart objects.
    :return: The extracted proximity groups.
    """
    return [
        _build_group(cluster)
        for clusters in clusters_by_room(objects, group_diameter).values()
        for cluster in clusters
    ]


def clusters_by_room(
    objects: list[EGObjectDAO], group_diameter: float = DEFAULT_GROUP_DIAMETER
) -> dict[str, list[list[EGObjectDAO]]]:
    """
    Cluster each room's floor objects, keeping the room they belong to.

    Clustering runs per room because stored positions are room-local, so two
    objects in different rooms would otherwise look adjacent.

    :param objects: Object DAOs of the rooms to cluster.
    :param group_diameter: Widest a cluster may span.
    :return: The clusters of each room, keyed by room id.
    """
    objects_by_room: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in objects:
        if obj.place_id == PlaceId.FLOOR:
            objects_by_room[obj.room_id].append(obj)

    clustered: dict[str, list[list[EGObjectDAO]]] = {}
    for room_id, room_objects in objects_by_room.items():
        positions = np.array(
            [(obj.position.x, obj.position.y) for obj in room_objects]
        )
        labels = _cluster_labels(positions, group_diameter)
        clusters: defaultdict[int, list[EGObjectDAO]] = defaultdict(list)
        for obj, label in zip(room_objects, labels):
            clusters[label].append(obj)
        clustered[room_id] = list(clusters.values())
    return clustered


def anchors_by_room(
    objects: list[EGObjectDAO], group_diameter: float = DEFAULT_GROUP_DIAMETER
) -> dict[str, list[EGObjectDAO]]:
    """
    Return each room's group anchors -- the largest object of every cluster.

    These, not the room's every object, are what a room floor layout is learned
    from. Each sampled piece anchors a group that draws its own members, so a
    layout holding every object multiplies the room: 29 sampled pieces became 71
    spawned objects in a 5-metre room, which no repair pass can pack. Anchors
    place the arrangements and the group circuit fills them in, so the two
    together come back to the room's real object count.

    :param objects: Object DAOs of the rooms to extract anchors from.
    :param group_diameter: Widest a cluster may span.
    :return: The anchor objects of each room, keyed by room id.
    """
    return {
        room_id: [max(cluster, key=_footprint_area) for cluster in clusters]
        for room_id, clusters in clusters_by_room(objects, group_diameter).items()
    }


def _cluster_labels(positions: np.ndarray, group_diameter: float) -> np.ndarray:
    """
    Label each position with the group it belongs to, so that no two members of
    a group stand further apart than *group_diameter*.

    :param positions: One ``(x, y)`` row per object, at least one.
    :param group_diameter: Widest a group may span.
    :return: A group label per row.
    """
    if len(positions) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=group_diameter,
        linkage="complete",
    ).fit_predict(positions)


def member_counts_by_anchor_type(
    groups: list[EGProximityGroup],
) -> dict[ObjectType, list[int]]:
    """
    Collect how many members each group holds, keyed by what it is anchored on.

    A collection's length is a structural property of the sampling query, so it
    is drawn before the query is built -- and it has to be drawn per anchor
    type, since a dining table gathers chairs while a refrigerator stands alone.
    Pooling the counts would surround every fridge with chairs.

    Memberless groups contribute their zero, which is what keeps solitary
    furniture solitary.

    :param groups: The extracted proximity groups.
    :return: Observed member counts per anchor object type.
    """
    counts: defaultdict[ObjectType, list[int]] = defaultdict(list)
    for group in groups:
        counts[group.object_type].append(len(group.members))
    return dict(counts)


def groups_for_circuit_training(
    groups: list[EGProximityGroup],
) -> list[EGProximityGroup]:
    """
    Keep only the groups that actually hold members, for fitting the group
    circuit on.

    The feature extractor decides whether the member relation gets an
    exchangeable template at all by inspecting only the *first* training
    instance's collection. Since most clusters are singletons, a memberless
    group arriving first would silently suppress member modelling for the whole
    circuit -- no error, and no member ever sampled.

    Dropping them costs nothing: how many members a group holds is drawn
    separately by :func:`member_counts_by_anchor_type`, which still sees every
    zero, so the circuit only ever needs to answer *where* members go given that
    there are some.

    :param groups: The extracted proximity groups.
    :return: Those holding at least one member.
    """
    return [group for group in groups if group.members]


def generate_proximity_group(
    node, downloader: Sage10kSceneDownloader | None = None
) -> None:
    """
    Train an RSPN on proximity-group data from the database, spawn a sampled
    arrangement into a world, repair member collisions directly in that world,
    and visualise the result via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    :param downloader: When given, member meshes are downloaded on demand until
        :data:`DEFAULT_MINIMUM_CANDIDATES_PER_TYPE` of them are cached. Left as
        ``None`` the pool is whatever is already cached, which keeps the demo
        fast for iterative testing; pass a downloader for a final demo that
        needs a broad mesh pool.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    proximity_groups, _ = _extract_proximity_groups(session)
    training_groups = groups_for_circuit_training(proximity_groups)
    data_access_objects = [to_dao(group) for group in training_groups]

    rspn = RelationalProbabilisticCircuit(
        EGProximityGroup,
        min_samples_per_leaf=min_samples_per_leaf_for(
            sum(len(group.members) for group in training_groups)
        ),
    )
    rspn = rspn.fit(data_access_objects)

    probability_backend = probabilistic_backend(rspn)

    member_count = sample_member_count(
        member_counts_by_anchor_type(proximity_groups), ObjectType.TABLE
    )
    sample = next(iter(probability_backend.evaluate(build_free_group_query(member_count))))

    source_ids_for_sampled_objects = _get_source_ids_for_objects(
        objects_of_type(session, ObjectType.CHAIR),
        object_type=ObjectType.CHAIR,
        downloader=downloader,
        minimum_candidates=DEFAULT_MINIMUM_CANDIDATES_PER_TYPE,
    )
    sample.position = EGPoint2D(x=0.0, y=0.0)
    sample.orientation = EGRotation(x=0.0, y=0.0, z=0.0)
    sample.source_ids = source_ids_for_sampled_objects

    spawned_group = InWorldLayoutResolver.for_proximity_group(sample, rspn).resolve()
    world = spawned_group.world
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating proximity-group sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_proximity_group(node)
