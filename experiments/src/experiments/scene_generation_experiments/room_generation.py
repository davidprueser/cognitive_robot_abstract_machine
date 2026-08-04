from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import *  # type: ignore  # noqa: F401,F403  registers ORM mappers
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_room_floor_query,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.room_floor_sampling import (
    build_room_from_floor_layout,
    sample_room_shape,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.table_chair_generation import (
    _extract_table_chair_groups_from_spatial_proximity,
)
from experiments.scene_generation_experiments.utils import (
    DEFAULT_TRAINING_ROOM_COUNT,
    _get_source_ids_for_objects,
    build_cached_mesh_pool,
    objects_for_rooms,
    rclpy_node,
    sampled_room_ids,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    EGShelfLayer,
    EGTableWithChairs,
    ObjectType,
    PlaceId,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )

_ROOM_MARGIN = 1.0
"""
Extra floor extent, in metres, added around the training pieces' bounding box so
the learned room is not sized flush to its furniture.
"""

_ROOM_HEIGHT = 2.7
"""
Ceiling height, in metres, used for every trained and sampled room.

Room height is a fixed generation parameter rather than something to learn:
the dataset's authoritative room records (``EGRoomDAO``) carry this exact
height on every row. Sampling it as a free variable let the RSPN draw
implausible values (walls shorter than the furniture standing in them), and
nothing downstream ever checked a piece's height against its room -- fixing
this constant is paired with clamping each sampled piece's height to it in
:func:`~experiments.scene_generation_experiments.room_floor_sampling.build_room_from_floor_layout`.
"""

_MIN_SAMPLES_PER_LEAF_FRACTION = 0.05
"""
Fraction of the training set required to create another split node when fitting
an RSPN, passed as ``min_samples_per_leaf`` to
:class:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`.

Each floor piece carries near-unique identifiers (``id``, ``source_id``), so with
the library default of one sample per leaf, the piece-level circuit grows one
leaf per training piece -- tens of thousands of nodes for this dataset's floor
pieces. Grounding deep-copies that circuit once per sampled piece, so an
unbounded circuit exhausts memory well before a room can be sampled.

Re-measured at ``0.05`` (up from ``0.01``) after fixing the extraction query
(see :func:`_extract_room_floor_layouts`): completing the training rooms
instead of truncating them raised both the piece count per room (median
2 -> 23) and the number of distinct training pieces by roughly an order of
magnitude, so the previous fraction no longer bounded the piece-level
circuit enough -- grounding a realistic ~20-piece room reliably exceeded a
10GB cap. ``0.05`` roughly halves peak grounding memory at the same piece
count; re-measure again if the training data's scale changes materially.
"""


def _extract_room_floor_layouts(
    session: Session,
    floor_place_id: str = "floor",
    room_count: int = DEFAULT_TRAINING_ROOM_COUNT,
) -> tuple[list[EGRoomFloorLayout], list[EGObjectDAO]]:  # noqa: F405
    """
    Load a random sample of rooms and group each one's floor-resting pieces
    into one :class:`EGRoomFloorLayout`, so the RSPN can learn which pieces a
    room holds and where they sit relative to the room centre.

    An object rests on the floor when its ``place_id`` equals *floor_place_id*.
    Pieces that instead reference another piece -- e.g. a table placed on
    another table -- carry that piece's id as their ``place_id`` and are skipped
    here.

    Rooms are sampled first, then loaded in full, so each returned layout
    reflects a room's true piece count rather than a row-count-limited
    fragment of it.

    :param session: Database session to query objects from.
    :param floor_place_id: The ``place_id`` value marking an object as resting
        directly on the room floor.
    :param room_count: Maximum number of distinct rooms to sample.
    :return: Extracted room floor layouts and all loaded object DAOs.
    """
    objects = objects_for_rooms(session, sampled_room_ids(session, room_count))

    floor_pieces_by_room: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)  # noqa: F405
    for obj in objects:
        if obj.place_id == floor_place_id:
            floor_pieces_by_room[obj.room_id].append(obj)

    floor_layouts = [
        _room_floor_layout(room_pieces)
        for room_pieces in floor_pieces_by_room.values()
        if room_pieces
    ]
    return floor_layouts, objects


def _room_floor_layout(room_pieces: list[EGObjectDAO]) -> EGRoomFloorLayout:  # noqa: F405
    """
    Build one :class:`EGRoomFloorLayout` from a room's floor-piece DAOs, sizing
    the floor to the pieces' bounding box (widened by :data:`_ROOM_MARGIN`) and
    re-expressing their positions relative to that box's centre, so the layout
    is learnable independent of where the room sits in world coordinates.

    :param room_pieces: The floor-resting object DAOs of a single room.
    :return: The room's floor layout.
    """
    x_values = [piece.position.x for piece in room_pieces]
    y_values = [piece.position.y for piece in room_pieces]
    center_x = (min(x_values) + max(x_values)) / 2
    center_y = (min(y_values) + max(y_values)) / 2

    return EGRoomFloorLayout(
        scale=EGScale(
            width=(max(x_values) - min(x_values)) + _ROOM_MARGIN,
            length=(max(y_values) - min(y_values)) + _ROOM_MARGIN,
            height=_ROOM_HEIGHT,
        ),
        pieces=[
            EGObject2D(
                id=piece.id,
                room_id=piece.room_id,
                place_id=piece.place_id,
                object_type=piece.object_type,
                scale=EGScale(
                    width=piece.scale.width,
                    length=piece.scale.length,
                    height=piece.scale.height,
                ),
                position=EGPoint2D(
                    x=piece.position.x - center_x,
                    y=piece.position.y - center_y,
                ),
                orientation=EGRotation(
                    x=piece.orientation.x,
                    y=piece.orientation.y,
                    z=piece.orientation.z,
                ),
                source_id=piece.source_id,
            )
            for piece in room_pieces
        ],
    )


def generate_room(node, downloader: Sage10kSceneDownloader | None = None) -> None:
    """
    Train an RSPN on room floor layouts from the database, sample a room, let
    each shelf and table sample its own contents, spawn the whole room into a
    world, repair floor placement and per-furniture contents directly in that
    world, and visualise the result via RViz markers.

    :param node: An active rclpy node used to publish visualisation markers.
    :param downloader: When given, floor-object meshes are downloaded on demand
        to broaden the mesh pool. Left as ``None`` the pool is whatever is
        already cached, which keeps the demo fast for iterative testing; pass a
        downloader for a final demo that needs a broad mesh pool.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)  # noqa: F405
    session = Session(engine)

    floor_layouts, _ = _extract_room_floor_layouts(session)
    room_rspn = RelationalProbabilisticCircuit(
        EGRoomFloorLayout, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
    ).fit([to_dao(layout) for layout in floor_layouts])

    shelf_layers, _ = _extract_shelf_layers_from_place_id(session)
    shelf_rspn = RelationalProbabilisticCircuit(
        EGShelfLayer, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
    ).fit([to_dao(layer) for layer in shelf_layers])

    table_chair_groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)
    table_rspn = RelationalProbabilisticCircuit(
        EGTableWithChairs, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
    ).fit([to_dao(group) for group in table_chair_groups])

    room_shape = sample_room_shape(floor_layouts)
    sampled_layout = next(
        iter(
            probabilistic_backend(room_rspn).evaluate(
                build_free_room_floor_query(room_shape)
            )
        )
    )

    mesh_pool_objects = build_cached_mesh_pool(session, downloader)
    all_object_source_ids = _get_source_ids_for_objects(mesh_pool_objects, object_type=None)
    room, object_id_to_mesh_path = build_room_from_floor_layout(
        sampled_layout,
        probabilistic_backend(shelf_rspn),
        probabilistic_backend(table_rspn),
        [len(group.chairs) for group in table_chair_groups],
        all_object_source_ids,
        _get_source_ids_for_objects(
            mesh_pool_objects, object_type=ObjectType.CHAIR, place_id=PlaceId.FLOOR
        ),
        all_object_source_ids,
    )

    spawned_room = InWorldLayoutResolver.for_scene(
        room,
        shelf_rspn,
        table_rspn,
        object_id_to_mesh_path=object_id_to_mesh_path,
    ).resolve()

    viz_marker = VizMarkerPublisher(_world=spawned_room.world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating room sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_room(node)
