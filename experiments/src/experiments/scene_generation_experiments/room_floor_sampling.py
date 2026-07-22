from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_free_table_query,
    sample_chair_count,
)
from krrood.entity_query_language.backends import ProbabilisticBackend

from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGObject2D,
    EGPoint2D,
    EGPosition,
    EGRoom,
    EGRoomFloorLayout,
    EGScale,
    EGShelf,
    EGTableWithChairs,
    EGWall,
    MeshCandidate,
    ObjectType,
    _MeshTypeMatcher,
)

if TYPE_CHECKING:
    from pathlib import Path

_LAYERS_PER_SHELF = 4
"""
Number of layers sampled for each shelf piece, matching the fixed-layer shelf
generator.
"""

_OBJECTS_PER_LAYER = 3
"""
Number of objects sampled per shelf layer, matching the fixed shelf generator.
"""

_WALL_THICKNESS = 0.1
"""
Thickness, in metres, of the walls enclosing a generated room.
"""


def sample_piece_count(training_piece_counts: list[int]) -> int:
    """
    Draw the number of floor pieces to place in a room from the empirical
    distribution of piece counts observed in the training rooms.

    Mirrors :func:`sample_chair_count`: an exchangeable relation's list length
    is a structural property of the sampling query, so the count is drawn first
    and the query is then built for exactly that many free slots.

    :param training_piece_counts: Number of floor pieces observed per training
        room.
    :return: A piece count drawn from the training distribution.
    """
    return random.choice(training_piece_counts)


def _rectangular_walls(scale: EGScale) -> list[EGWall]:
    """
    Build four walls enclosing a *scale*-sized rectangle centred on the room
    origin, so the room has a floor for its pieces to rest on.

    :param scale: The room floor footprint.
    :return: The four enclosing walls, each running start → end with
        non-decreasing coordinates so its length stays positive.
    """
    half_width = scale.width / 2
    half_length = scale.length / 2
    edges = [
        ((-half_width, -half_length), (half_width, -half_length)),
        ((half_width, -half_length), (half_width, half_length)),
        ((-half_width, half_length), (half_width, half_length)),
        ((-half_width, -half_length), (-half_width, half_length)),
    ]
    return [
        EGWall(
            id=f"wall_{index}",
            start_point=EGPoint2D(x=start[0], y=start[1]),
            end_point=EGPoint2D(x=end[0], y=end[1]),
            height=scale.height,
            thickness=_WALL_THICKNESS,
        )
        for index, (start, end) in enumerate(edges)
    ]


def _sampled_shelf(
    piece: EGObject2D,
    shelf_backend: ProbabilisticBackend,
    source_ids: list[MeshCandidate],
) -> EGShelf:
    """
    Build an :class:`EGShelf` for a sampled shelf *piece*, filling it with layers
    drawn from the shelf circuit so the furniture samples its own contents.

    :param piece: The sampled floor piece standing for a shelf.
    :param shelf_backend: The single-sample backend over the shelf circuit.
    :param source_ids: Mesh candidates for the shelf's sampled contents.
    :return: The populated shelf, placed at the piece's floor pose.
    """
    reference_layer = next(
        iter(shelf_backend.evaluate(build_free_layer_query(_OBJECTS_PER_LAYER)))
    )
    target_scale = reference_layer.scale
    remaining_layers = [
        next(
            iter(
                shelf_backend.evaluate(
                    build_layer_query_with_fixed_scale(_OBJECTS_PER_LAYER, target_scale)
                )
            )
        )
        for _ in range(_LAYERS_PER_SHELF - 1)
    ]
    return EGShelf(
        position=EGPoint2D(x=piece.position.x, y=piece.position.y),
        scale=EGScale(
            height=piece.scale.height,
            length=target_scale.length,
            width=target_scale.width,
        ),
        orientation=piece.orientation,
        layers=[reference_layer] + remaining_layers,
        source_ids=source_ids,
    )


def _sampled_table(
    piece: EGObject2D,
    table_backend: ProbabilisticBackend,
    chair_count: int,
    source_ids: list[MeshCandidate],
) -> EGTableWithChairs:
    """
    Build an :class:`EGTableWithChairs` for a sampled table *piece*, surrounding
    it with chairs drawn from the table circuit.

    :param piece: The sampled floor piece standing for a table.
    :param table_backend: The single-sample backend over the table circuit.
    :param chair_count: Number of chairs to sample around the table.
    :param source_ids: Mesh candidates for the table's sampled chairs.
    :return: The populated table-with-chairs group, placed at the piece's pose.
    """
    sampled = next(iter(table_backend.evaluate(build_free_table_query(chair_count))))
    return EGTableWithChairs(
        position=EGPoint2D(x=piece.position.x, y=piece.position.y),
        scale=EGScale(
            width=piece.scale.width,
            length=piece.scale.length,
            height=piece.scale.height,
        ),
        orientation=piece.orientation,
        chairs=sampled.chairs,
        source_ids=source_ids,
    )


def _height_clamped(piece: EGObject2D, max_height: float) -> EGObject2D:
    """
    Return *piece* with its height clamped to *max_height*.

    Nothing downstream checks a piece's height against its room at collision
    time, so a piece the RSPN sampled taller than the room's own ceiling (a
    lamp taller than the walls) would otherwise spawn poking through it.

    :param piece: The sampled floor piece.
    :param max_height: The room's ceiling height.
    :return: *piece* unchanged if already within bounds, otherwise a copy
        with a clamped scale.
    """
    if piece.scale.height <= max_height:
        return piece
    return dataclasses.replace(
        piece, scale=dataclasses.replace(piece.scale, height=max_height)
    )


def _free_object(
    piece: EGObject2D, object_index: int, candidate: MeshCandidate
) -> EGObject:
    """
    Build a free-standing floor :class:`EGObject` for a sampled *piece* that is
    neither a shelf nor a table, resolving its mesh from *candidate*.

    The RSPN never samples a usable ``id``/``source_id`` for a piece -- both
    are fixed to ``None`` in the free-object query, since the circuit only
    models the spatial fields -- so both are drawn fresh here instead, the
    same way shelf and table contents get their mesh from a candidate pool
    rather than from the piece itself.

    :param piece: The sampled floor piece.
    :param object_index: Index used to build a unique id for the object.
    :param candidate: The mesh candidate matched to the piece's object type.
    :return: The free floor object, placed at the piece's pose.
    """
    return EGObject(
        id=f"free_object_{object_index}",
        room_id=piece.room_id,
        place_id="floor",
        object_type=piece.object_type,
        scale=piece.scale,
        position=EGPosition(x=piece.position.x, y=piece.position.y, z=0.0),
        orientation=piece.orientation,
        source_id=candidate.source_id,
    )


def build_room_from_floor_layout(
    layout: EGRoomFloorLayout,
    shelf_backend: ProbabilisticBackend,
    table_backend: ProbabilisticBackend,
    training_chair_counts: list[int],
    shelf_source_ids: list[MeshCandidate],
    chair_source_ids: list[MeshCandidate],
    free_object_source_ids: list[MeshCandidate],
) -> tuple[EGRoom, dict[str, Path]]:
    """
    Turn a sampled floor *layout* into a spawnable :class:`EGRoom`: each shelf
    and table piece samples its own contents, and every other piece becomes a
    free floor object.

    :param layout: The sampled room floor layout.
    :param shelf_backend: Backend over the shelf circuit, for shelf contents.
    :param table_backend: Backend over the table circuit, for table chairs.
    :param training_chair_counts: Observed chair counts, for sampling how many
        chairs each table gets.
    :param shelf_source_ids: Mesh candidates for shelf contents.
    :param chair_source_ids: Mesh candidates for chairs.
    :param free_object_source_ids: Mesh candidates for free floor objects,
        matched to each piece by its sampled object type. A piece is dropped
        when this pool is empty, since it could otherwise never be spawned.
    :return: The assembled room and, for its free objects, a mapping from each
        object's id to its mesh directory. Several objects may map to the same
        directory, since one scene directory commonly holds many objects.
    """
    mesh_matcher = _MeshTypeMatcher(candidates=free_object_source_ids)
    shelves: list[EGShelf] = []
    tables: list[EGTableWithChairs] = []
    free_objects: list[EGObject] = []
    object_id_to_mesh_path: dict[str, Path] = {}
    for piece in layout.pieces:
        piece = _height_clamped(piece, layout.scale.height)
        if piece.object_type == ObjectType.SHELF:
            shelves.append(_sampled_shelf(piece, shelf_backend, shelf_source_ids))
        elif piece.object_type == ObjectType.TABLE:
            tables.append(
                _sampled_table(
                    piece,
                    table_backend,
                    sample_chair_count(training_chair_counts),
                    chair_source_ids,
                )
            )
        elif free_object_source_ids:
            candidate = mesh_matcher.random_match(piece.object_type)
            free_object = _free_object(piece, len(free_objects), candidate)
            free_objects.append(free_object)
            object_id_to_mesh_path[free_object.id] = candidate.scene_dir

    room = EGRoom(
        id="room_1",
        room_type="living_room",
        scale=EGScale(
            width=layout.scale.width,
            length=layout.scale.length,
            height=layout.scale.height,
        ),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        objects=free_objects,
        walls=_rectangular_walls(layout.scale),
        shelves=shelves,
        tables=tables,
    )
    return room, object_id_to_mesh_path
