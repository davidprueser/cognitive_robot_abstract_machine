from __future__ import annotations

from itertools import combinations
from unittest.mock import MagicMock, patch

import pytest

from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGPoint2D,
    EGPosition,
    EGRotation,
    EGRoom,
    RoomType,
    EGScale,
    EGShelf,
    EGShelfLayer,
    EGWall,
    SpawnedRoom,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World


def _rectangular_walls() -> list[EGWall]:
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


def _empty_shelf(x: float, y: float) -> EGShelf:
    """
    A shelf with a single empty layer and no mesh candidates, so it spawns just
    its corpus -- enough to make two shelves' corpuses overlap on the floor.
    """
    return EGShelf(
        position=EGPoint2D(x=x, y=y),
        scale=EGScale(height=2.0, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[EGShelfLayer(scale=EGScale(height=0.02, length=1.0, width=1.0), objects=[])],
    )


def _room(shelves: list[EGShelf]) -> EGRoom:
    return EGRoom(
        id="room_1",
        room_type=RoomType.LIVING_ROOM,
        scale=EGScale(height=2.5, length=5.0, width=5.0),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        walls=_rectangular_walls(),
        shelves=shelves,
    )


def _corpuses_collide(spawned: SpawnedRoom) -> bool:
    corpuses = [spawned_shelf.corpus for spawned_shelf in spawned.spawned_shelves]
    detector = FCLCollisionDetector(_world=spawned.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(corpuses, 2)
        }
    )
    return detector.check_collisions(matrix).any()


def _separated_floor_points(self, amount: int = 1, **kwargs) -> list[Point3]:
    return [
        Point3(1.5, 1.5, 0.0, reference_frame=self.root),
        Point3(-1.5, -1.5, 0.0, reference_frame=self.root),
    ]


def test_for_scene_separates_overlapping_furniture_pieces() -> None:
    """
    Two shelves spawned at the same spot must have their corpuses driven apart by
    the single floor-placement group, proving cross-furniture collisions are
    resolved without a dedicated cross-furniture check.
    """
    room = _room([_empty_shelf(2.5, 2.5), _empty_shelf(2.5, 2.5)])

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend",
        return_value=MagicMock(),
    ), patch(
        "semantic_digital_twin.semantic_annotations.semantic_annotations.Floor.sample_points_from_surface",
        _separated_floor_points,
    ):
        resolver = InWorldLayoutResolver.for_scene(
            room, shelf_rspn=MagicMock(), table_rspn=MagicMock()
        )
        assert _corpuses_collide(resolver.spawned)
        spawned = resolver.resolve()

    assert isinstance(spawned, SpawnedRoom)
    assert not _corpuses_collide(spawned)


def test_for_scene_returns_a_resolver_over_the_spawned_room() -> None:
    """
    ``for_scene`` must spawn the room once and hand back a resolver whose layout
    is that :class:`SpawnedRoom`, so a caller can inspect its floor and pieces.
    """
    room = _room([_empty_shelf(1.0, 1.0)])

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend",
        return_value=MagicMock(),
    ):
        resolver = InWorldLayoutResolver.for_scene(
            room, shelf_rspn=MagicMock(), table_rspn=MagicMock()
        )

    assert isinstance(resolver.spawned, SpawnedRoom)
    assert isinstance(resolver.spawned.world, World)
    assert len(resolver.spawned.spawned_shelves) == 1
