from __future__ import annotations

import shutil
from importlib.resources import files
from itertools import combinations
from math import radians
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.scene_generation_experiments.exceptions import LayoutResolutionError
from experiments.scene_generation_experiments.in_world_resolver import (
    FloorObjectGroup,
    InWorldLayoutResolver,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGPosition,
    EGRotation,
    EGScale,
    ObjectType,
    SpawnedLayout,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def scene_dir(tmp_path: Path) -> Path:
    """
    A scene directory holding the bundled chair PLY, so floor pieces spawn with
    real geometry the FCL detector can act on.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    return tmp_path


def _world_with_floor() -> tuple[World, Body, Floor]:
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
        floor = Floor.create_with_new_body_in_world(
            scale=Scale(x=6.0, y=6.0, z=0.01),
            world=world,
            name=PrefixedName(name="floor"),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=root
            ),
        )
    return world, root, floor


def _floor_object(
    object_id: str, x: float, y: float, yaw_degrees: float, z: float = 0.0
) -> EGObject:
    return EGObject(
        id=object_id,
        room_id="room_1",
        place_id="floor",
        object_type=ObjectType.VASE,
        scale=EGScale(width=0.4, length=0.4, height=0.6),
        position=EGPosition(x=x, y=y, z=z),
        orientation=EGRotation(x=0.0, y=0.0, z=yaw_degrees),
        source_id="test_object",
    )


def _colliding(bodies: list[Body], world: World) -> bool:
    detector = FCLCollisionDetector(_world=world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(bodies, 2)
        }
    )
    return detector.check_collisions(matrix).any()


def test_resample_moves_piece_to_free_point_keeping_height_and_yaw(
    scene_dir: Path,
) -> None:
    """
    Redrawing an offending floor piece must place its footprint over the sampled
    free point while keeping its resting height and yaw, so a shelf stays upright
    and a table keeps standing after being moved.
    """
    world, root, floor = _world_with_floor()
    body = _floor_object("piece_0", 0.0, 0.0, 30.0, z=0.3).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body}, supporting_body=None, floor=floor
    )

    free_point = Point3(2.0, 1.0, 0.0, reference_frame=root)
    with patch.object(
        Floor, "sample_points_from_surface", return_value=[free_point]
    ):
        group.resample_and_move({0})

    origin = body.parent_connection.origin.to_np()
    assert origin[0, 3] == pytest.approx(2.0)
    assert origin[1, 3] == pytest.approx(1.0)
    assert origin[2, 3] == pytest.approx(0.3)
    expected_yaw = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.0, 0.0, 0.0, 0.0, 0.0, radians(30.0)
    ).to_np()
    assert origin[:3, :3] == pytest.approx(expected_yaw[:3, :3])


def test_resolver_moves_colliding_floor_piece_until_group_is_collision_free(
    scene_dir: Path,
) -> None:
    """
    Two overlapping floor pieces must be resolved by moving one onto a redrawn,
    separated free point, leaving the group collision-free under a real-mesh
    check.
    """
    world, root, floor = _world_with_floor()
    body_0 = _floor_object("piece_0", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    body_1 = _floor_object("piece_1", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body_0, 1: body_1}, supporting_body=None, floor=floor
    )

    separated_points = [
        Point3(2.5, 2.5, 0.0, reference_frame=root),
        Point3(-2.5, -2.5, 0.0, reference_frame=root),
    ]
    with patch.object(
        Floor, "sample_points_from_surface", return_value=separated_points
    ):
        resolver = InWorldLayoutResolver(
            spawned=SpawnedLayout(world=world), groups=[group]
        )
        resolver.resolve()

    assert not _colliding([body_0, body_1], world)


def test_resolver_raises_when_floor_pieces_cannot_be_separated(
    scene_dir: Path,
) -> None:
    """
    When every redraw lands the pieces back on top of each other, the resolver
    must give up and raise rather than spin forever.
    """
    world, root, floor = _world_with_floor()
    body_0 = _floor_object("piece_0", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    body_1 = _floor_object("piece_1", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body_0, 1: body_1}, supporting_body=None, floor=floor
    )

    overlapping_point = [Point3(0.0, 0.0, 0.0, reference_frame=root)]
    with patch.object(
        Floor, "sample_points_from_surface", return_value=overlapping_point
    ):
        resolver = InWorldLayoutResolver(
            spawned=SpawnedLayout(world=world), groups=[group], max_passes=3
        )
        with pytest.raises(LayoutResolutionError):
            resolver.resolve()
