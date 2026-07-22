from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_room_floor_query,
)
from experiments.scene_generation_experiments.room_floor_sampling import (
    _rectangular_walls,
    build_room_from_floor_layout,
    sample_piece_count,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGObject2D,
    EGPoint2D,
    EGRelativePolarPose,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    EGShelfLayer,
    EGTableWithChairs,
    MeshCandidate,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


def _piece(object_type: ObjectType, x: float, y: float, source_id: str) -> EGObject2D:
    return EGObject2D(
        id=f"{object_type.value}_{source_id}",
        room_id="room_1",
        place_id="floor",
        object_type=object_type,
        scale=EGScale(width=0.8, length=0.8, height=1.0),
        position=EGPoint2D(x=x, y=y),
        orientation=EGRotation(x=0.0, y=0.0, z=15.0),
        source_id=source_id,
    )


def _shelf_backend() -> MagicMock:
    layer = EGShelfLayer(
        scale=EGScale(width=1.0, length=1.0, height=0.02),
        objects=[],
    )
    backend = MagicMock()
    backend.evaluate.return_value = [layer]
    return backend


def _table_backend() -> MagicMock:
    sampled = EGTableWithChairs(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(width=1.0, length=1.0, height=0.75),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        chairs=[
            EGChair(
                id="chair_0",
                room_id="room_1",
                object_type=ObjectType.CHAIR,
                scale=EGScale(width=0.5, length=0.5, height=0.9),
                relative_pose=EGRelativePolarPose(
                    distance_from_table_center=1.0,
                    angle_from_table_center=0.0,
                    facing_angle_relative_to_table=180.0,
                ),
                source_id="chair_src",
            )
        ],
    )
    backend = MagicMock()
    backend.evaluate.return_value = [sampled]
    return backend


def test_sample_piece_count_draws_from_the_training_counts() -> None:
    """
    The sampled piece count must come from the observed training counts, so a
    room holds a plausible number of floor pieces.
    """
    assert sample_piece_count([4, 4, 4]) == 4


def test_build_free_room_floor_query_builds_for_any_piece_count() -> None:
    """
    The room-floor sampling query must build for a range of piece counts without
    error, so the count drawn from the training distribution can always be
    turned into a query.
    """
    for piece_count in (1, 3, 7):
        build_free_room_floor_query(piece_count, height=2.7)


def test_room_floor_layout_round_trips_through_json() -> None:
    """
    An :class:`EGRoomFloorLayout` must survive a JSON round-trip unchanged, so it
    can be persisted and reloaded as training data.
    """
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=6.0, height=2.5),
        pieces=[_piece(ObjectType.SHELF, 1.0, -1.0, "shelf_src")],
    )
    restored = EGRoomFloorLayout._from_json(layout.to_json())

    assert restored.scale.width == 5.0
    assert restored.pieces[0].object_type == ObjectType.SHELF
    assert restored.pieces[0].position.x == 1.0
    assert restored.pieces[0].orientation.z == 15.0


def test_rectangular_walls_enclose_the_footprint_with_positive_lengths() -> None:
    """
    The generated room must be enclosed by four walls whose signed lengths are
    positive, so a floor spanning the footprint can be built from them.
    """
    walls = _rectangular_walls(EGScale(width=4.0, length=6.0, height=2.5))

    assert len(walls) == 4
    for wall in walls:
        assert wall.wall_length_and_yaw[0] > 0


def test_build_room_maps_pieces_to_furniture_and_free_objects() -> None:
    """
    Each shelf and table piece must become furniture that has sampled its own
    contents, and every other piece a free floor object with a resolved mesh, so
    the assembled room is ready for in-world placement and content resolution.
    """
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=5.0, height=2.5),
        pieces=[
            _piece(ObjectType.SHELF, 1.0, 1.0, "shelf_src"),
            _piece(ObjectType.TABLE, -1.0, -1.0, "table_src"),
            _piece(ObjectType.VASE, 0.0, 2.0, "vase_src"),
        ],
    )
    vase_candidate = MeshCandidate(
        scene_dir=Path("/scenes/vase"), source_id="vase_mesh", object_type=ObjectType.VASE
    )

    room, mesh_to_object_mapping = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[vase_candidate],
    )

    assert len(room.shelves) == 1
    assert len(room.shelves[0].layers) == 4
    assert len(room.tables) == 1
    assert len(room.tables[0].chairs) == 1
    assert len(room.objects) == 1
    assert room.objects[0].object_type == ObjectType.VASE
    assert room.objects[0].source_id == vase_candidate.source_id
    assert len(room.walls) == 4
    assert mesh_to_object_mapping == {room.objects[0].id: vase_candidate.scene_dir}


def test_build_room_keeps_a_mesh_path_per_object_when_objects_share_a_scene_dir() -> None:
    """
    Free objects are matched to mesh candidates by object type, so two
    sampled free objects of the same type resolve to the same candidate's
    scene directory. Each object must still keep its own mesh path, so a
    shared directory doesn't silently drop one object's mesh.
    """
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=5.0, height=2.5),
        pieces=[
            _piece(ObjectType.VASE, 0.0, 1.0, "vase_src"),
            _piece(ObjectType.VASE, 2.0, 2.0, "lamp_src"),
        ],
    )
    shared_candidate = MeshCandidate(
        scene_dir=Path("/scenes/shared"), source_id="vase_mesh", object_type=ObjectType.VASE
    )

    room, mesh_to_object_mapping = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[shared_candidate],
    )

    assert len(room.objects) == 2
    assert mesh_to_object_mapping == {
        room.objects[0].id: shared_candidate.scene_dir,
        room.objects[1].id: shared_candidate.scene_dir,
    }


def test_build_room_drops_free_pieces_when_no_mesh_candidate_is_available() -> None:
    """
    A free piece can never be spawned without a mesh, so it must be dropped
    from the room rather than included with an unresolvable mesh, when the
    free-object candidate pool is empty.
    """
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=5.0, height=2.5),
        pieces=[_piece(ObjectType.VASE, 0.0, 1.0, "vase_src")],
    )

    room, mesh_to_object_mapping = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[],
    )

    assert room.objects == []
    assert mesh_to_object_mapping == {}


def test_build_room_clamps_a_piece_taller_than_the_room_to_the_ceiling_height() -> None:
    """
    A piece the RSPN sampled taller than the room's own ceiling (e.g. a lamp
    taller than the walls) must be clamped to the room height, since nothing
    downstream checks a piece's height against its room at collision time.
    """
    tall_piece = _piece(ObjectType.VASE, 0.0, 1.0, "lamp_src")
    tall_piece.scale = EGScale(width=0.3, length=0.3, height=4.0)
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=5.0, height=2.7),
        pieces=[tall_piece],
    )
    candidate = MeshCandidate(
        scene_dir=Path("/scenes/lamp"), source_id="lamp_mesh", object_type=ObjectType.VASE
    )

    room, _ = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[candidate],
    )

    assert room.objects[0].scale.height == 2.7


def test_build_room_places_furniture_at_the_pieces_floor_pose() -> None:
    """
    A furniture piece must be placed at its sampled floor pose, so a shelf or
    table lands where the room layout put it rather than at the origin.
    """
    layout = EGRoomFloorLayout(
        scale=EGScale(width=5.0, length=5.0, height=2.5),
        pieces=[_piece(ObjectType.SHELF, 1.5, -2.0, "shelf_src")],
    )

    room, _ = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[],
    )

    assert room.shelves[0].position == EGPoint2D(x=1.5, y=-2.0)
    assert room.shelves[0].orientation.z == 15.0
