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
    sample_room_composition,
)
from experiments.scene_generation_experiments.book_shelf_generation import (
    shelf_layers_by_shelf,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_mesh_candidate_types,
    _coarsen_rare_object_types,
    _frequent_object_types,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.proximity_group_generation import (
    anchors_by_room,
    groups_for_circuit_training,
    member_counts_by_anchor_type,
    proximity_groups_from_objects,
)
from experiments.scene_generation_experiments.utils import (
    DEFAULT_TRAINING_ROOM_COUNT,
    min_samples_per_leaf_for,
    _get_source_ids_for_objects,
    build_cached_mesh_pool,
    objects_for_rooms,
    rclpy_node,
    sampled_rooms_of_type,
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
    EGFloorPiece,
    EGRoom,
    EGRoomFloorLayout,
    EGScale,
    EGShelfLayer,
    EGProximityGroup,
    EGWallRelativePose,
    ObjectType,
    PlaceId,
    RoomType,
)

if TYPE_CHECKING:
    from experiments.scene_generation_experiments.data_preprocessing import (
        Sage10kSceneDownloader,
    )


def _extract_room_floor_layouts(
    session: Session,
    room_type: RoomType,
    room_count: int = DEFAULT_TRAINING_ROOM_COUNT,
) -> tuple[list[EGRoomFloorLayout], list[EGObjectDAO]]:  # noqa: F405
    """
    Load a random sample of stored rooms of *room_type* and group each one's
    floor-resting pieces into one :class:`EGRoomFloorLayout`, so the circuit can
    learn which pieces that kind of room holds and where they sit relative to
    its centre.

    An object rests on the floor when its ``place_id`` is
    :attr:`PlaceId.FLOOR`. Pieces that instead reference another piece -- e.g. a
    lamp standing on an anchor -- carry that piece's id as their ``place_id`` and
    are skipped here.

    Every layout takes its footprint from the room's own stored dimensions.
    Deriving it from the pieces instead, as this used to, made the room a
    function of whatever furniture happened to be loaded, so a truncated or
    tightly clustered room collapsed to a couple of metres across.

    :param session: Database session to query rooms and objects from.
    :param room_type: The generalized category of room to train on.
    :param room_count: Maximum number of distinct rooms to sample.
    :return: Extracted room floor layouts and all loaded object DAOs.
    """
    rooms = sampled_rooms_of_type(session, room_type, room_count)
    rooms_by_id = {room.from_dao().id: room for room in rooms}
    objects = objects_for_rooms(session, list(rooms_by_id))

    floor_layouts = [
        _room_floor_layout(rooms_by_id[room_id].from_dao(), room_anchors)
        for room_id, room_anchors in anchors_by_room(objects).items()
        if room_anchors and room_id in rooms_by_id
    ]
    return floor_layouts, objects


def _room_floor_layout(
    room: EGRoom, room_pieces: list[EGObjectDAO]  # noqa: F405
) -> EGRoomFloorLayout:
    """
    Build one :class:`EGRoomFloorLayout` from a room and its floor-piece DAOs,
    taking the footprint from the room itself and re-expressing each piece's
    position relative to the footprint centre, so the layout is learnable
    independent of where the room sits in world coordinates.

    Stored piece positions are already room-local, with the room's lower-left
    corner at the origin, so the centre is simply half its extent. Each
    re-centred pose is then expressed relative to the wall the piece stands
    nearest, which is what makes "shelves stand against a wall" learnable at all
    -- see :class:`EGWallRelativePose`.

    :param room: The room the pieces stand in.
    :param room_pieces: The floor-resting object DAOs of that room.
    :return: The room's floor layout.
    """
    room_scale = EGScale(
        width=room.scale.width,
        length=room.scale.length,
        height=room.scale.height,
    )
    center_x = room_scale.width / 2
    center_y = room_scale.length / 2

    return EGRoomFloorLayout(
        scale=room_scale,
        pieces=[
            EGFloorPiece(
                object_type=piece.object_type,
                scale=EGScale(
                    width=piece.scale.width,
                    length=piece.scale.length,
                    height=piece.scale.height,
                ),
                pose=EGWallRelativePose.from_absolute_pose(
                    piece.position.x - center_x,
                    piece.position.y - center_y,
                    piece.orientation.z,
                    room_scale,
                ),
            )
            for piece in room_pieces
        ],
    )


def generate_room(
    node,
    room_type: RoomType = RoomType.LIVING_ROOM,
    downloader: Sage10kSceneDownloader | None = None,
) -> None:
    """
    Train circuits on rooms of *room_type* from the database, sample a room, let
    each shelf and anchor sample its own contents, spawn the whole room into a
    world, repair floor placement and per-furniture contents directly in that
    world, and visualise the result via RViz markers.

    All three circuits are fitted from a single loaded room sample. Loading
    once per circuit, as this used to, trained each of them on a different
    random population of rooms and scanned the object anchor three times.

    :param node: An active rclpy node used to publish visualisation markers.
    :param room_type: The kind of room to generate. Circuits are fitted on this
        category alone, so the sampled room describes one real setting rather
        than an average over every setting in the dataset.
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

    floor_layouts, training_objects = _extract_room_floor_layouts(session, room_type)
    room_rspn = RelationalProbabilisticCircuit(
        EGRoomFloorLayout,
        min_samples_per_leaf=min_samples_per_leaf_for(
            sum(len(layout.pieces) for layout in floor_layouts)
        ),
    ).fit([to_dao(layout) for layout in floor_layouts])

    layers_by_shelf = shelf_layers_by_shelf(training_objects, object_type=None)
    shelf_layers = [layer for layers in layers_by_shelf for layer in layers]
    frequent_object_types = _frequent_object_types(shelf_layers, keep_count=20)
    shelf_layers = _coarsen_rare_object_types(shelf_layers)
    shelf_rspn = RelationalProbabilisticCircuit(
        EGShelfLayer,
        min_samples_per_leaf=min_samples_per_leaf_for(
            sum(len(layer.objects) for layer in shelf_layers)
        ),
    ).fit([to_dao(layer) for layer in shelf_layers])

    # Clustered over every floor object, so the arrangements a room type holds
    # are discovered rather than authored. Singleton clusters still carry their
    # zero into the member counts, but are kept out of the fitted circuit -- see
    # groups_for_circuit_training.
    proximity_groups = proximity_groups_from_objects(training_objects)
    grouped_member_counts = member_counts_by_anchor_type(proximity_groups)
    training_groups = groups_for_circuit_training(proximity_groups)
    group_rspn = RelationalProbabilisticCircuit(
        EGProximityGroup,
        min_samples_per_leaf=min_samples_per_leaf_for(
            sum(len(group.members) for group in training_groups)
        ),
    ).fit([to_dao(group) for group in training_groups])

    room_composition = sample_room_composition(floor_layouts)
    sampled_layout = next(
        iter(
            probabilistic_backend(room_rspn).evaluate(
                build_free_room_floor_query(room_composition)
            )
        )
    )

    mesh_pool_objects = build_cached_mesh_pool(session, downloader)
    all_object_source_ids = _get_source_ids_for_objects(
        mesh_pool_objects, object_type=None
    )
    # The shelf circuit is fitted on coarsened types, so its contents are matched
    # against a pool relabelled the same way. Floor pieces keep their real types,
    # so their pool must not be coarsened -- relabelling it would leave every
    # piece of a rare type unable to find its own mesh.
    shelf_content_source_ids = _coarsen_mesh_candidate_types(
        all_object_source_ids, frequent_object_types
    )
    built = build_room_from_floor_layout(
        sampled_layout,
        probabilistic_backend(shelf_rspn),
        probabilistic_backend(group_rspn),
        grouped_member_counts,
        shelf_content_source_ids,
        # Members can be any kind of floor object now that groups are
        # discovered rather than authored as table-and-chairs, so they draw
        # from the same pool as loose pieces and are matched by their own type.
        all_object_source_ids,
        all_object_source_ids,
        room_type=room_type,
        training_layer_counts=[len(layers) for layers in layers_by_shelf],
        training_objects_per_layer=[len(layer.objects) for layer in shelf_layers],
    )

    resolver = InWorldLayoutResolver.for_scene(
        built.room,
        shelf_rspn,
        group_rspn,
        object_id_to_mesh_path=built.object_id_to_mesh_path,
    )
    spawned_room = resolver.resolve()
    print(built.report.summary())
    print(f"resolver dropped {resolver.dropped_body_count} bodies it could not place")

    viz_marker = VizMarkerPublisher(_world=spawned_room.world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating room sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_room(node)
