from __future__ import annotations

import shutil

import trimesh
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_room_floor_query,
)
from experiments.scene_generation_experiments.room_floor_sampling import (
    SampledRoomComposition,
    _rectangular_walls,
    build_room_from_floor_layout,
    sample_room_composition,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGFloorPiece,
    EGObject,
    EGPosition,
    EGPoint2D,
    EGRelativePolarPose,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    EGShelfLayer,
    EGTableWithChairs,
    EGWallRelativePose,
    MeshCandidate,
    RoomWall,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


_PIECE_ROOM = EGScale(width=5.0, length=6.0, height=2.5)
"""
Footprint the helper pieces are posed against, so a wall-relative pose can be
built from the absolute coordinates a test cares about.
"""


def _piece(
    object_type: ObjectType,
    x: float,
    y: float,
    source_id: str,
    room_scale: EGScale = _PIECE_ROOM,
) -> EGFloorPiece:
    return EGFloorPiece(
        object_type=object_type,
        scale=EGScale(width=0.8, length=0.8, height=1.0),
        pose=EGWallRelativePose.from_absolute_pose(x, y, 15.0, room_scale),
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


def test_sample_room_composition_keeps_composition_and_footprint_from_the_same_layout() -> None:
    """
    Piece count and floor footprint are correlated in the training data, so they
    must be drawn together from one layout. Drawing them independently would
    readily pair a large room's piece count with a small room's footprint.
    """
    small = EGRoomFloorLayout(
        scale=EGScale(width=2.0, length=2.0, height=2.7),
        pieces=[_piece(ObjectType.VASE, 0.0, 0.0, "small_src")],
    )
    large = EGRoomFloorLayout(
        scale=EGScale(width=12.0, length=9.0, height=2.7),
        pieces=[_piece(ObjectType.VASE, float(index), 0.0, f"large_{index}") for index in range(7)],
    )
    valid_pairings = {(1, 2.0), (7, 12.0)}

    for _ in range(30):
        shape = sample_room_composition([small, large])
        assert (shape.piece_count, shape.scale.width) in valid_pairings


def test_build_free_room_floor_query_fixes_the_sampled_footprint() -> None:
    """
    The footprint must enter the query as fixed evidence, so the
    ``floor_area``/``aspect_ratio`` aggregations stay determinable at grounding
    rather than being integrated out via Monte-Carlo.
    """
    scale = EGScale(width=6.0, length=3.0, height=2.7)
    query = build_free_room_floor_query(
        SampledRoomComposition(
            scale=scale, object_types=[ObjectType.SHELF] * 3
        )
    )
    query.resolve()

    assert query.construct_instance().scale == scale


def test_build_free_room_floor_query_builds_for_any_piece_count() -> None:
    """
    The room-floor sampling query must build for a range of piece counts without
    error, so the count drawn from the training distribution can always be
    turned into a query.
    """
    for piece_count in (1, 3, 7):
        build_free_room_floor_query(
            SampledRoomComposition(
                object_types=[ObjectType.SHELF] * piece_count,
                scale=EGScale(width=5.0, length=5.0, height=2.7),
            )
        )


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
    recovered_x, _, recovered_yaw = restored.pieces[0].pose.to_absolute_pose(
        restored.scale
    )
    assert recovered_x == pytest.approx(1.0)
    assert recovered_yaw == pytest.approx(15.0)


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

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[vase_candidate],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

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

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[shared_candidate],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

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

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

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

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[candidate],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

    assert room.objects[0].scale.height == 2.7


def test_build_room_places_furniture_at_the_pieces_floor_pose() -> None:
    """
    A furniture piece must be placed at its sampled floor pose, so a shelf or
    table lands where the room layout put it rather than at the origin.
    """
    # Posed against the same footprint the helper uses, and clear of the
    # diagonal where two walls are equidistant.
    layout = EGRoomFloorLayout(
        scale=_PIECE_ROOM,
        pieces=[_piece(ObjectType.SHELF, 1.5, -2.2, "shelf_src")],
    )

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

    assert room.shelves[0].position.x == pytest.approx(1.5)
    assert room.shelves[0].position.y == pytest.approx(-2.2)
    assert room.shelves[0].orientation.z == pytest.approx(15.0)


def test_free_floor_object_spawns_at_the_meshs_own_real_size(tmp_path: Path) -> None:
    """
    A sage10k mesh already carries its real-world size, so it spawns at identity
    scale and is never rescaled -- the same contract the shelf pipeline relies
    on.

    Replaces an earlier test that asserted the mesh was rescaled to the sampled
    size. That behaviour was reversed deliberately in 31a259f98, and applying
    the sampled scale on top of an already correctly-sized mesh would stretch it
    by its own dimensions again. The size the circuit samples is now honoured by
    *selecting* a mesh close to it -- see
    :meth:`_MeshTypeMatcher.random_match` -- rather than by deforming whichever
    mesh was drawn.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    native_extents = trimesh.load(
        objects_dir / "test_object.ply", force="mesh"
    ).extents

    floor_object = EGObject(
        id="free_object_0",
        room_id="room_1",
        place_id="floor",
        object_type=ObjectType.CHAIR,
        scale=EGScale(width=0.2, length=0.3, height=0.4),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = floor_object.create_in_world(world, tmp_path, parent=root)

    rendered_extents = body.collision.shapes[0].mesh.extents
    assert rendered_extents == pytest.approx(native_extents, rel=1e-6)


def test_a_free_object_adopts_the_real_size_of_the_mesh_chosen_for_it() -> None:
    """
    Collision resolution, height clamping and containment all reason about the
    piece's scale, so it has to be the size that actually spawns rather than the
    size the circuit sampled.
    """
    layout = EGRoomFloorLayout(
        scale=_PIECE_ROOM,
        pieces=[_piece(ObjectType.CHAIR, 1.0, -2.2, "chair_src")],
    )
    candidate = MeshCandidate(
        scene_dir=Path("/scenes/chairs"),
        source_id="real_chair",
        object_type=ObjectType.CHAIR,
        native_extents=(0.62, 0.58, 0.94),
    )

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[candidate],
    )
    room = built.room
    mesh_to_object_mapping = built.object_id_to_mesh_path

    assert room.objects[0].scale.width == pytest.approx(0.62)
    assert room.objects[0].scale.length == pytest.approx(0.58)
    assert room.objects[0].scale.height == pytest.approx(0.94)


def test_a_wide_piece_is_pushed_clear_of_the_wall_it_stands_against() -> None:
    """
    A piece adopts the real extents of whichever mesh is chosen for it, and that
    happens after its pose was sampled. A mesh wider than the piece the circuit
    drew therefore reaches past the wall its centre was placed against, so the
    centre has to be pushed in by whatever the rotated footprint overhangs.
    """
    room_scale = EGScale(width=6.0, length=6.0, height=2.7)
    layout = EGRoomFloorLayout(
        scale=room_scale,
        pieces=[
            EGFloorPiece(
                object_type=ObjectType.CABINET,
                scale=EGScale(width=0.4, length=0.4, height=1.0),
                pose=EGWallRelativePose(
                    wall=RoomWall.SOUTH,
                    distance_from_wall=0.05,
                    position_along_wall=0.5,
                    yaw_relative_to_wall=0.0,
                ),
            )
        ],
    )
    wide_mesh = MeshCandidate(
        scene_dir=Path("/scenes/cabinets"),
        source_id="wide_cabinet",
        object_type=ObjectType.CABINET,
        native_extents=(0.6, 0.6, 1.0),
    )

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[wide_mesh],
    )

    placed = built.room.objects[0]
    assert placed.scale.length == pytest.approx(0.6)
    # Half the 0.6 m depth is 0.3, so the centre cannot sit closer than 0.3 m
    # to the wall at y = -3.0.
    assert placed.position.y >= -room_scale.length / 2 + 0.3 - 1e-9


def test_a_shelf_takes_its_footprint_from_the_piece_not_the_layer_circuit() -> None:
    """
    A shelf must stand at the size the room layout drew for it. Taking the
    corpus width and depth from the shelf-layer circuit instead made a shelf's
    footprint unrelated to the piece the room circuit placed -- a sampled
    0.36 x 0.21 m shelf spawning as 0.66 x 0.35 m -- so the room was arranged
    around one size and rendered with another.
    """
    room_scale = EGScale(width=6.0, length=6.0, height=2.7)
    piece_scale = EGScale(width=1.1, length=0.35, height=1.8)
    layout = EGRoomFloorLayout(
        scale=room_scale,
        pieces=[
            EGFloorPiece(
                object_type=ObjectType.SHELF,
                scale=piece_scale,
                pose=EGWallRelativePose(
                    wall=RoomWall.SOUTH,
                    distance_from_wall=0.3,
                    position_along_wall=0.5,
                    yaw_relative_to_wall=0.0,
                ),
            )
        ],
    )

    built = build_room_from_floor_layout(
        layout,
        shelf_backend=_shelf_backend(),
        table_backend=_table_backend(),
        training_chair_counts=[1],
        shelf_source_ids=[],
        chair_source_ids=[],
        free_object_source_ids=[],
    )

    shelf = built.room.shelves[0]
    assert shelf.scale.width == pytest.approx(piece_scale.width)
    assert shelf.scale.length == pytest.approx(piece_scale.length)
    assert shelf.scale.height == pytest.approx(piece_scale.height)
