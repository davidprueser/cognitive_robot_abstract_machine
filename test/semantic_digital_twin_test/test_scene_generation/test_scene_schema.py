from __future__ import annotations

import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGGroupMember,
    EGDoor,
    EGObject2D,
    EGPoint2D,
    EGPosition,
    EGRelativePolarPose,
    EGRoom,
    RoomType,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    EGProximityGroup,
    EGWall,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Wall
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def chair_mesh_directory(tmp_path: Path) -> Path:
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )
    return tmp_path


def _make_shelf(orientation_z: float = 0.0) -> EGShelf:
    return EGShelf(
        position=EGPoint2D(x=1.0, y=2.0),
        scale=EGScale(height=2.0, length=0.4, width=0.8),
        orientation=EGRotation(x=0.0, y=0.0, z=orientation_z),
        layers=[
            EGShelfLayer(
                scale=EGScale(width=0.8, length=0.4, height=0.02),
                objects=[
                    EGObject2D(
                        id="book_1",
                        room_id="room_1",
                        place_id="shelf_1",
                        object_type=ObjectType.BOOK,
                        scale=EGScale(width=0.1, length=0.05, height=0.2),
                        position=EGPoint2D(x=0.0, y=0.0),
                        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
                        source_id="chair_src",
                    )
                ],
            )
        ],
        source_ids=[],
    )


def _make_table_with_chairs() -> EGProximityGroup:
    return EGProximityGroup(
        position=EGPoint2D(x=3.0, y=3.0),
        scale=EGScale(height=0.75, length=1.2, width=0.8),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        members=[
            EGGroupMember(
                id="chair_1",
                room_id="room_1",
                object_type=ObjectType.CHAIR,
                scale=EGScale(height=0.9, length=0.5, width=0.5),
                relative_pose=EGRelativePolarPose(
                    distance_from_anchor=1.0,
                    angle_from_anchor=-90.0,
                    facing_angle_relative_to_anchor=0.0,
                ),
                source_id="chair_src",
            )
        ],
        source_ids=[],
    )


def _make_room(shelf_orientation_z: float = 0.0) -> EGRoom:
    return EGRoom(
        id="room_1",
        room_type=RoomType.LIVING_ROOM,
        scale=EGScale(height=2.7, length=5.0, width=5.5),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        shelves=[_make_shelf(shelf_orientation_z)],
        groups=[_make_table_with_chairs()],
        walls=[
            EGWall(
                id="wall_1",
                start_point=EGPoint2D(0.0, 5.0),
                end_point=EGPoint2D(5.5, 5.0),
                height=2.7,
                thickness=0.1,
            ),
            EGWall(
                id="wall_2",
                start_point=EGPoint2D(0.0, 0.0),
                end_point=EGPoint2D(5.5, 0.0),
                height=2.7,
                thickness=0.1,
            ),
            EGWall(
                id="wall_3",
                start_point=EGPoint2D(5.5, 0.0),
                end_point=EGPoint2D(5.5, 5.0),
                height=2.7,
                thickness=0.1,
            ),
            EGWall(
                id="wall_4",
                start_point=EGPoint2D(0.0, 0.0),
                end_point=EGPoint2D(0.0, 5.0),
                height=2.7,
                thickness=0.1,
            ),
        ],
        doors=[],
    )


def test_egroom_round_trips_shelves_and_tables_through_json() -> None:
    room = _make_room()

    reconstructed = EGRoom._from_json(room.to_json())

    assert len(reconstructed.shelves) == 1
    assert reconstructed.shelves[0].layers[0].objects[0].id == "book_1"
    assert reconstructed.shelves[0].position.x == pytest.approx(1.0)
    assert len(reconstructed.groups) == 1
    assert reconstructed.groups[0].members[0].id == "chair_1"


def test_room_create_in_world_mounts_shelf_and_table_under_given_parent(
    chair_mesh_directory: Path,
) -> None:
    """
    EGRoom.create_in_world must instantiate its shelves' and groups' bodies
    under the room's own parent entity, not a separate throwaway World, since a
    room-level scene needs every furniture group to share one kinematic tree
    with the room's floor/walls.
    """
    room = _make_room()
    room.shelves[0].source_ids = [
        MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.BOOK)
    ]
    room.groups[0].source_ids = [
        MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.CHAIR)
    ]

    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)

    room.create_in_world(world, object_id_to_mesh_path=None, parent=root)

    shelf_corpus_bodies = [
        body for body in world.bodies if body.name.name == "shelf_corpus"
    ]
    table_bodies = [body for body in world.bodies if body.name.name == "anchor"]
    assert len(shelf_corpus_bodies) == 1
    assert len(table_bodies) == 1
    # A separate, throwaway World would leave these bodies parented to a
    # different root body than the one passed in; assert they are reachable
    # from the given root's own connection graph instead.
    assert shelf_corpus_bodies[0] in world.bodies
    assert table_bodies[0] in world.bodies
    assert world.root is root


def test_rotated_shelf_mounts_at_correct_absolute_pose_under_given_parent(
    chair_mesh_directory: Path,
) -> None:
    """
    A shelf sampled with a non-zero orientation must actually render rotated
    -- EGShelf.create_in_world previously ignored self.orientation entirely,
    so every shelf rendered axis-aligned regardless of its sampled yaw.
    """
    shelf = _make_shelf(orientation_z=90.0)
    shelf.source_ids = [MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.BOOK)]

    world = World()
    parent = Body(name=PrefixedName(name="room_parent"))
    with world.modify_world():
        world.add_body(parent)

    shelf.create_in_world(world, parent=parent)

    [corpus_body] = [body for body in world.bodies if body.name.name == "shelf_corpus"]
    assert corpus_body.parent_connection.parent is parent

    translation = corpus_body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(shelf.position.x, abs=1e-6)
    assert translation[1] == pytest.approx(shelf.position.y, abs=1e-6)

    yaw = corpus_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(math.pi / 2, abs=1e-6)


def test_room_create_in_world_adds_door_aperture_to_its_wall() -> None:
    """
    EGDoor.create_in_world must register its entry way on the door's own wall
    via the wall annotation's generic ``add(part, field_name=...)`` -- it
    previously called a nonexistent ``add_aperture`` method (``HasApertures``
    only ever declared the ``apertures`` field, never a dedicated setter),
    raising an AttributeError for any room with a door.
    """
    room = _make_room()
    room.doors = [
        EGDoor(
            id="door_1",
            wall_id="wall_1",
            position_on_wall=0.5,
            width=0.95,
            height=2.05,
            opens_inward=False,
        )
    ]

    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)

    room.create_in_world(world, object_id_to_mesh_path=None, parent=root)

    [wall_annotation] = [
        annotation
        for annotation in world.get_semantic_annotations_by_type(Wall)
        if annotation.name.name == "wall_1"
    ]
    assert len(wall_annotation.apertures) == 1
