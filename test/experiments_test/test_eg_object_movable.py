from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGPosition,
    EGRotation,
    EGScale,
    ObjectType,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def mesh_path(tmp_path: Path) -> Path:
    """
    A scene directory holding the bundled chair PLY, so an object spawns with
    real geometry.
    """
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    return tmp_path


def _object(place_id: str) -> EGObject:
    return EGObject(
        id=f"object_{place_id}",
        room_id="room_1",
        place_id=place_id,
        object_type=ObjectType.VASE,
        scale=EGScale(width=0.3, length=0.3, height=0.5),
        position=EGPosition(x=1.0, y=2.0, z=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )


def _world_with_root() -> tuple[World, Body]:
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def test_floor_object_is_movable(mesh_path: Path) -> None:
    """
    A floor object must be attached with a movable 6-DoF connection whose
    ``origin`` setter repositions it, so a scene resolver can move it to free
    floor space.
    """
    world, root = _world_with_root()
    body = _object("floor").create_in_world(world, mesh_path, parent=root)

    assert isinstance(body.parent_connection, Connection6DoF)

    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5.0, 5.0, 0.0
    )
    moved = body.global_pose.to_position().to_np()
    assert moved[0] == pytest.approx(5.0)
    assert moved[1] == pytest.approx(5.0)


def test_wall_object_stays_fixed(mesh_path: Path) -> None:
    """
    A wall object must keep a fixed connection: walls are static obstacles, not
    things the resolver repositions.
    """
    world, root = _world_with_root()
    body = _object("wall").create_in_world(world, mesh_path, parent=root)

    assert isinstance(body.parent_connection, FixedConnection)


def test_object_placed_at_its_declared_pose(mesh_path: Path) -> None:
    """
    A floor object with no explicit *world_pose* must spawn at the pose built
    from its declared position and orientation.
    """
    world, root = _world_with_root()
    body = _object("floor").create_in_world(world, mesh_path, parent=root)

    position = body.global_pose.to_position().to_np()
    assert position[0] == pytest.approx(1.0)
    assert position[1] == pytest.approx(2.0)
