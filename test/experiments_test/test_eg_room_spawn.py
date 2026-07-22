from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGObject,
    EGPoint2D,
    EGPosition,
    EGRelativePolarPose,
    EGRotation,
    EGRoom,
    EGScale,
    EGShelf,
    EGShelfLayer,
    EGObject2D,
    EGTableWithChairs,
    EGWall,
    MeshCandidate,
    ObjectType,
    SpawnedRoom,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


def _scene_dir_with_mesh(root: Path) -> Path:
    """
    Populate *root* with an ``objects/`` folder holding the bundled chair PLY,
    so a piece pointed at it spawns with real geometry.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = root / "objects"
    objects_dir.mkdir(parents=True)
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    return root


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    """
    A scene directory holding the bundled chair PLY under ``objects/``, so every
    piece in the room spawns with real geometry.
    """
    return _scene_dir_with_mesh(tmp_path)


def _free_object(object_id: str, x: float, y: float) -> EGObject:
    return EGObject(
        id=object_id,
        room_id="room_1",
        place_id="floor",
        object_type=ObjectType.VASE,
        scale=EGScale(width=0.3, length=0.3, height=0.5),
        position=EGPosition(x=x, y=y, z=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )


def _rectangular_walls() -> list[EGWall]:
    # Each wall runs start -> end with non-decreasing coordinates, so the
    # signed wall length stays positive (a Wall's thickness must be its
    # smallest extent).
    edges = [
        ((0.0, 0.0), (5.0, 0.0)),
        ((5.0, 0.0), (5.0, 5.0)),
        ((0.0, 5.0), (5.0, 5.0)),
        ((0.0, 0.0), (0.0, 5.0)),
    ]
    return [
        EGWall(
            id=f"room_1_wall_{index}",
            start_point=EGPoint2D(x=start[0], y=start[1]),
            end_point=EGPoint2D(x=end[0], y=end[1]),
            height=2.5,
            thickness=0.1,
        )
        for index, (start, end) in enumerate(edges)
    ]


def _shelf(candidate: MeshCandidate) -> EGShelf:
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=1.0, width=1.0),
        objects=[
            EGObject2D(
                id="book_0",
                room_id="room_1",
                place_id="shelf_1",
                object_type=ObjectType.BOOK,
                scale=EGScale(height=0.4, length=0.3, width=0.3),
                position=EGPoint2D(x=0.0, y=0.0),
                orientation=EGRotation(x=0.0, y=0.0, z=0.0),
                source_id="test_object",
            )
        ],
    )
    return EGShelf(
        position=EGPoint2D(x=1.0, y=1.0),
        scale=EGScale(height=2.0, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[layer],
        source_ids=[candidate],
    )


def _table(candidate: MeshCandidate) -> EGTableWithChairs:
    chair = EGChair(
        id="chair_0",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(width=0.5, length=0.5, height=0.9),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=1.0,
            angle_from_table_center=0.0,
            facing_angle_relative_to_table=180.0,
        ),
        source_id="test_object",
    )
    return EGTableWithChairs(
        position=EGPoint2D(x=3.0, y=3.0),
        scale=EGScale(width=1.0, length=1.0, height=0.75),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        chairs=[chair],
        source_ids=[candidate],
    )


def _world_with_root() -> tuple[World, Body]:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def _room(objects: list[EGObject]) -> EGRoom:
    return EGRoom(
        id="room_1",
        room_type="living_room",
        scale=EGScale(height=2.5, length=5.0, width=5.0),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        objects=objects,
        walls=_rectangular_walls(),
    )


def test_spawn_in_world_returns_handles_for_floor_walls_and_free_objects(
    tmp_path: Path,
) -> None:
    """
    Spawning a room must hand back its floor annotation, a body per wall, and a
    body per free floor object, so a scene resolver can move the pieces and
    check them against the walls without rebuilding the world.
    """
    objects = [_free_object("object_0", 1.0, 1.0), _free_object("object_1", 4.0, 4.0)]
    room = _room(objects)
    world, root = _world_with_root()
    mesh_to_object_mapping = {
        obj.id: _scene_dir_with_mesh(tmp_path / obj.id) for obj in objects
    }

    spawned = room.spawn_in_world(world, mesh_to_object_mapping, parent=root)

    assert isinstance(spawned, SpawnedRoom)
    assert isinstance(spawned.floor, Floor)
    assert len(spawned.wall_bodies) == 4
    assert set(spawned.object_bodies) == {0, 1}
    assert all(isinstance(body, Body) for body in spawned.object_bodies.values())


def test_spawn_in_world_resolves_meshes_for_objects_sharing_one_scene_dir(
    scene_dir: Path,
) -> None:
    """
    Two free objects can resolve to the same scene directory (it commonly
    holds several objects with distinct source ids). Each must still get its
    own body, so a shared directory doesn't silently drop one object's mesh.
    """
    objects = [_free_object("object_0", 1.0, 1.0), _free_object("object_1", 4.0, 4.0)]
    room = _room(objects)
    world, root = _world_with_root()
    mesh_to_object_mapping = {obj.id: scene_dir for obj in objects}

    spawned = room.spawn_in_world(world, mesh_to_object_mapping, parent=root)

    assert set(spawned.object_bodies) == {0, 1}
    assert all(isinstance(body, Body) for body in spawned.object_bodies.values())


def test_spawn_in_world_wires_through_nested_shelves_and_tables(
    scene_dir: Path,
) -> None:
    """
    A room's shelves and tables must be spawned and their handles collected, so
    each furniture piece can later sample its own contents in place.
    """
    candidate = MeshCandidate(
        scene_dir=scene_dir, source_id="test_object", object_type=ObjectType.CHAIR
    )
    room = _room([])
    room.shelves = [_shelf(candidate)]
    room.tables = [_table(candidate)]
    world, root = _world_with_root()

    spawned = room.spawn_in_world(world, None, parent=root)

    assert len(spawned.spawned_shelves) == 1
    assert len(spawned.spawned_tables) == 1
    assert spawned.spawned_shelves[0].corpus is not None
    assert spawned.spawned_tables[0].table is not None


def test_create_in_world_still_returns_the_world_root(scene_dir: Path) -> None:
    """
    The spawn refactor must keep :meth:`EGRoom.create_in_world` returning the
    world root, so existing callers stay unaffected.
    """
    objects = [_free_object("object_0", 1.0, 1.0)]
    room = _room(objects)
    world, root = _world_with_root()
    mesh_to_object_mapping = {obj.id: scene_dir for obj in objects}

    result = room.create_in_world(world, mesh_to_object_mapping, parent=root)

    assert result is world.root
