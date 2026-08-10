from __future__ import annotations

import shutil
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
from semantic_digital_twin.scene_generation.scene_schema import (
    EGGroupMember,
    EGPoint2D,
    EGProximityGroup,
    EGRelativePolarPose,
    ObjectType,
    EGPosition,
    EGRotation,
    EGRoom,
    RoomType,
    EGScale,
    EGShelf,
    EGShelfLayer,
    EGWall,
    MeshCandidate,
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
            room, shelf_rspn=MagicMock(), group_rspn=MagicMock()
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
            room, shelf_rspn=MagicMock(), group_rspn=MagicMock()
        )

    assert isinstance(resolver.spawned, SpawnedRoom)
    assert isinstance(resolver.spawned.world, World)
    assert len(resolver.spawned.spawned_shelves) == 1


@pytest.fixture
def member_mesh_pool(tmp_path: Path) -> list[MeshCandidate]:
    """
    A one-mesh candidate pool, so a group's members spawn with real geometry.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "member_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "member_src_texture.png"
    )
    return [MeshCandidate(tmp_path, "member_src", ObjectType.CHAIR)]


def _group_with_member(
    x: float, y: float, distance: float, source_ids: list[MeshCandidate]
) -> EGProximityGroup:
    """
    A group whose single member stands *distance* metres from its anchor along
    the anchor's local x-axis.
    """
    return EGProximityGroup(
        position=EGPoint2D(x=x, y=y),
        scale=EGScale(height=0.75, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        object_type=ObjectType.TABLE,
        members=[
            EGGroupMember(
                id="member_0",
                room_id="room_1",
                object_type=ObjectType.CHAIR,
                scale=EGScale(height=0.9, length=0.5, width=0.5),
                relative_pose=EGRelativePolarPose(
                    distance_from_anchor=distance,
                    angle_from_anchor=0.0,
                    facing_angle_relative_to_anchor=0.0,
                ),
                source_id="member_src",
            )
        ],
        source_ids=source_ids,
    )


def test_group_members_are_checked_against_the_rooms_walls(
    member_mesh_pool: list[MeshCandidate],
) -> None:
    """
    A group's members must be collision-checked against the room's walls.

    Members are posed relative to their anchor and nothing bounds that offset by
    the room, so a member routinely lands in a wall or outside the room. It used
    to be harmless because only tables formed groups, so members were a handful
    of chairs; now that groups are discovered rather than authored, most of a
    room's objects are members and none of them was being checked at all.
    """
    room = _room([])
    room.groups = [_group_with_member(2.5, 2.5, 0.0, member_mesh_pool)]

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend",
        return_value=MagicMock(),
    ):
        resolver = InWorldLayoutResolver.for_scene(
            room, shelf_rspn=MagicMock(), group_rspn=MagicMock()
        )

    wall_bodies = set(resolver.spawned.wall_bodies)
    member_groups = [
        group
        for group in resolver.groups
        if not isinstance(group, FloorObjectGroup) and group.bodies
    ]
    assert member_groups, "expected a collision group for the spawned members"
    for group in member_groups:
        assert wall_bodies <= set(group.static_obstacles)


def test_group_members_are_checked_against_the_other_floor_pieces(
    member_mesh_pool: list[MeshCandidate],
) -> None:
    """
    A member must also be checked against everything else standing on the floor
    -- free objects, shelves, other groups' anchors and their members -- since
    the resolver only ever checks within a group plus its static obstacles, and
    two members of different groups would otherwise be free to overlap.
    """
    room = _room([])
    room.groups = [
        _group_with_member(1.5, 1.5, 0.5, member_mesh_pool),
        _group_with_member(3.5, 3.5, 0.5, member_mesh_pool),
    ]

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend",
        return_value=MagicMock(),
    ):
        resolver = InWorldLayoutResolver.for_scene(
            room, shelf_rspn=MagicMock(), group_rspn=MagicMock()
        )

    [floor_group] = [g for g in resolver.groups if isinstance(g, FloorObjectGroup)]
    member_groups = [
        g for g in resolver.groups if not isinstance(g, FloorObjectGroup) and g.bodies
    ]
    all_members = {body for group in member_groups for body in group.bodies.values()}

    # Each member sees every floor piece except the anchor it belongs to --
    # those two are meant to be adjacent and the pair could never be cleared --
    # and every member of every other group.
    for group in member_groups:
        own = set(group.bodies.values())
        obstacles = set(group.static_obstacles)
        assert set(floor_group.bodies.values()) - {group.anchor} <= obstacles
        assert group.anchor not in obstacles
        assert (all_members - own) <= obstacles


def test_a_redrawn_member_is_kept_inside_the_room(
    member_mesh_pool: list[MeshCandidate],
) -> None:
    """
    Repairing a member redraws its polar pose from the circuit, and nothing in
    that pose is bounded by the room -- so a repair undoes the containment the
    member was built with and puts it back through a wall. Containing it only at
    build time left bodies outside the room after the resolver had run.
    """
    room = _room([])
    room.groups = [_group_with_member(0.0, 0.0, 0.5, member_mesh_pool)]
    far_outside = EGProximityGroup(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=0.75, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        object_type=ObjectType.TABLE,
        members=[
            EGGroupMember(
                id="member_0",
                room_id="room_1",
                object_type=ObjectType.CHAIR,
                scale=EGScale(height=0.9, length=0.5, width=0.5),
                relative_pose=EGRelativePolarPose(
                    distance_from_anchor=20.0,
                    angle_from_anchor=0.0,
                    facing_angle_relative_to_anchor=0.0,
                ),
                source_id="member_src",
            )
        ],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [far_outside]
        resolver = InWorldLayoutResolver.for_scene(
            room, shelf_rspn=MagicMock(), group_rspn=MagicMock()
        )
        [member_group] = [
            g
            for g in resolver.groups
            if not isinstance(g, FloorObjectGroup) and g.bodies
        ]
        member_group.resample_and_move({0})

    resolver.spawned.world.update_forward_kinematics()
    body = member_group.bodies[0]
    pose = resolver.spawned.world.compute_forward_kinematics_np(
        resolver.spawned.world.root, body
    )
    assert abs(float(pose[0, 3])) <= room.scale.width / 2
    assert abs(float(pose[1, 3])) <= room.scale.length / 2
