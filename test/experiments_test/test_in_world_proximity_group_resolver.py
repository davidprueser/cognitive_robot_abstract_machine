from __future__ import annotations

import shutil
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from krrood.entity_query_language.exceptions import NoSolutionFound
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
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGProximityGroup,
    MeshCandidate,
    ObjectType,
    SpawnedProximityGroup,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF


@pytest.fixture
def mesh_candidate(tmp_path: Path) -> MeshCandidate:
    """
    A mesh candidate backed by the bundled member PLY, so members spawn with real
    geometry the FCL detector can act on.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    return MeshCandidate(
        scene_dir=tmp_path, source_id="test_object", object_type=ObjectType.CHAIR
    )


def _chair(chair_id: str, distance: float, angle: float) -> EGGroupMember:
    return EGGroupMember(
        id=chair_id,
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(width=0.5, length=0.5, height=0.9),
        relative_pose=EGRelativePolarPose(
            distance_from_anchor=distance,
            angle_from_anchor=angle,
            facing_angle_relative_to_anchor=180.0,
        ),
        source_id="test_object",
    )


def _group(members: list[EGGroupMember], candidate: MeshCandidate) -> EGProximityGroup:
    return EGProximityGroup(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(width=1.0, length=1.0, height=0.75),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        members=members,
        source_ids=[candidate],
    )


def _colliding_bodies(spawned: SpawnedProximityGroup) -> bool:
    """
    True if any two spawned member bodies collide.
    """
    bodies = list(spawned.member_bodies.values())
    detector = FCLCollisionDetector(_world=spawned.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(bodies, 2)
        }
    )
    return detector.check_collisions(matrix).any()


def test_create_in_world_still_returns_a_world(mesh_candidate: MeshCandidate) -> None:
    """
    The spawn refactor must keep :meth:`EGProximityGroup.create_in_world`
    returning a plain :class:`World`, so existing callers stay unaffected.
    """
    group = _group([_chair("chair_0", 1.0, 0.0)], mesh_candidate)
    assert isinstance(group.create_in_world(), World)


def test_spawn_in_world_returns_a_body_per_chair_and_spawns_the_table(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    Spawning must hand back a body for every member and create the anchor in the
    world, so the resolver can validate and move members without rebuilding it.
    """
    group = _group(
        [_chair("chair_0", 1.0, 0.0), _chair("chair_1", 1.0, 90.0)], mesh_candidate
    )
    spawned = group.spawn_in_world()

    assert set(spawned.member_bodies) == {0, 1}
    assert spawned.world.get_semantic_annotations_by_type(Table)


def test_spawned_chair_body_pose_matches_chair_local_pose(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    A freshly spawned member body must sit, in the anchor frame, exactly where
    :meth:`EGProximityGroup.member_local_pose` says it should -- pinning the
    single pose formula that both spawning and later moving rely on.
    """
    group = _group([_chair("chair_0", 1.0, 45.0)], mesh_candidate)
    spawned = group.spawn_in_world()
    body = spawned.member_bodies[0]

    expected = group.member_local_pose(group.members[0], spawned.anchor)
    assert body.parent_connection.origin.to_np() == pytest.approx(expected.to_np())


def test_spawned_table_is_movable_as_a_unit(mesh_candidate: MeshCandidate) -> None:
    """
    The anchor must hang off its parent by a movable 6-DoF connection, so a
    room-level resolver can reposition the whole group -- anchor and members -- in
    place by setting the anchor origin, and its members follow.
    """
    group = _group([_chair("chair_0", 1.0, 0.0)], mesh_candidate)
    spawned = group.spawn_in_world()
    anchor = spawned.anchor
    member_body = spawned.member_bodies[0]

    assert isinstance(anchor.parent_connection, Connection6DoF)

    before = member_body.global_pose.to_position().to_np()
    table_origin = anchor.parent_connection.origin
    shifted = HomogeneousTransformationMatrix.from_xyz_rpy(
        table_origin.to_position().to_np()[0] + 2.0,
        table_origin.to_position().to_np()[1],
        table_origin.to_position().to_np()[2],
        reference_frame=table_origin.reference_frame,
    )
    anchor.parent_connection.origin = shifted

    after = member_body.global_pose.to_position().to_np()
    assert after[0] == pytest.approx(before[0] + 2.0)
    assert after[1] == pytest.approx(before[1])


def test_resolver_moves_colliding_chair_until_group_is_collision_free(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    Two overlapping members must be resolved by moving one body to a redrawn,
    separated pose, leaving the group collision-free under a real-mesh check.
    """
    group = _group(
        [_chair("chair_0", 1.0, 0.0), _chair("chair_1", 1.0, 0.0)], mesh_candidate
    )
    separated = EGProximityGroup(
        position=group.position,
        scale=group.scale,
        orientation=group.orientation,
        members=[_chair("fixed", 1.0, 0.0), _chair("moved", 1.0, 180.0)],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [separated]
        resolver = InWorldLayoutResolver.for_proximity_group(group, rspn=MagicMock())
        spawned = resolver.resolve()

    assert group.members[1].relative_pose.angle_from_anchor == 180.0
    assert not _colliding_bodies(spawned)


def test_resolver_falls_back_to_relaxed_query_when_neighbour_evidence_has_no_solution(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    When the neighbour-conditioned resample query has no support in the
    fitted circuit -- a real failure mode once a member's pose has drifted
    through several repair passes -- the resolver must retry without the
    fixed neighbour's evidence instead of letting NoSolutionFound abort the
    whole repair.
    """
    group = _group(
        [_chair("chair_0", 1.0, 0.0), _chair("chair_1", 1.0, 0.0)], mesh_candidate
    )
    relaxed = EGProximityGroup(
        position=group.position,
        scale=group.scale,
        orientation=group.orientation,
        members=[_chair("moved", 1.0, 180.0)],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.side_effect = [
            NoSolutionFound(expression=MagicMock(), found_number=0),
            [relaxed],
        ]
        resolver = InWorldLayoutResolver.for_proximity_group(group, rspn=MagicMock())
        spawned = resolver.resolve()

    assert backend_factory.return_value.evaluate.call_count == 2
    assert group.members[1].relative_pose.angle_from_anchor == 180.0
    assert not _colliding_bodies(spawned)


def test_resolver_drops_chairs_it_cannot_separate(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    When resampling never separates the members, the resolver must give up moving
    them and drop the offenders, returning a collision-free layout rather than
    spinning forever or failing the whole sample.
    """
    group = _group(
        [_chair("chair_0", 1.0, 0.0), _chair("chair_1", 1.0, 0.0)], mesh_candidate
    )
    still_overlapping = EGProximityGroup(
        position=group.position,
        scale=group.scale,
        orientation=group.orientation,
        members=[_chair("fixed", 1.0, 0.0), _chair("moved", 1.0, 0.0)],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [still_overlapping]
        resolver = InWorldLayoutResolver.for_proximity_group(
            group, rspn=MagicMock(), max_passes=3
        )
        spawned = resolver.resolve()

    assert not _colliding_bodies(spawned)
    assert len(spawned.member_bodies) < 2
