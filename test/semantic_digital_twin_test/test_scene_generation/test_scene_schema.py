from __future__ import annotations

import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def chair_mesh_directory(tmp_path: Path) -> Path:
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
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


def test_rotated_shelf_mounts_at_correct_absolute_pose_under_given_parent(
    chair_mesh_directory: Path,
) -> None:
    """
    A shelf sampled with a non-zero orientation must actually render rotated
    -- EGShelf.create_in_world previously ignored self.orientation entirely,
    so every shelf rendered axis-aligned regardless of its sampled yaw.

    The corpus is built in the shelf's content frame, so its yaw is the shelf's
    own plus :attr:`EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES`. Were the
    orientation ignored again, the yaw would collapse to that offset alone.
    """
    shelf = _make_shelf(orientation_z=90.0)
    shelf.source_ids = [
        MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.BOOK)
    ]

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

    expected_yaw = math.radians(
        shelf.orientation.z + EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES
    )
    yaw = corpus_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(expected_yaw, abs=1e-6)
