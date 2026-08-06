from __future__ import annotations

import shutil
from importlib.resources import files
from itertools import combinations
from math import radians
from pathlib import Path
from unittest.mock import patch

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
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGPoint2D,
    EGPosition,
    EGRotation,
    EGScale,
    EGWall,
    ObjectType,
    RoomInterior,
    SpawnedLayout,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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


_INTERIOR = RoomInterior(
    scale=EGScale(width=6.0, length=6.0, height=2.7), wall_thickness=0.1
)
"""
Footprint matching the floor the helper world builds, so wall-relative repairs
resolve against the right room.
"""


def _south_wall(world: World, root: Body) -> Body:
    """
    The south wall of the helper room, centred on ``y = -3.0`` as
    :func:`~experiments.scene_generation_experiments.room_floor_sampling._rectangular_walls`
    builds it, so it reaches half its thickness back into the room.
    """
    wall = EGWall(
        id="wall_0",
        start_point=EGPoint2D(x=-3.0, y=-3.0),
        end_point=EGPoint2D(x=3.0, y=-3.0),
        height=2.7,
        thickness=0.1,
    )
    return wall.create_in_world(world, parent=root).root


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


def test_resample_slides_the_piece_along_its_wall_keeping_height_and_yaw(
    scene_dir: Path,
) -> None:
    """
    Redrawing an offending floor piece must slide it along the wall it already
    stands against, keeping which wall it uses, how far from it, and its resting
    height and yaw.

    Drawing a fresh point from the floor surface instead discarded the sampled
    pose entirely, which is how wall-hugging furniture ended up standing in open
    floor.
    """
    world, root, floor = _world_with_floor()
    # 0.6 m from the south wall of a 6 m room, i.e. y = -2.4, which is far
    # enough out that the 0.74 m deep mesh clears the wall's inner face and the
    # repair has no reason to push the piece off its wall.
    body = _floor_object("piece_0", 0.0, -2.4, 30.0, z=0.3).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body}, supporting_body=None, floor=floor, interior=_INTERIOR
    )

    group.resample_and_move({0})

    origin = body.parent_connection.origin.to_np()
    assert origin[2, 3] == pytest.approx(0.3)
    expected_yaw = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.0, 0.0, 0.0, 0.0, 0.0, radians(30.0)
    ).to_np()
    assert origin[:3, :3] == pytest.approx(expected_yaw[:3, :3])
    # Still against the same wall, at the same distance from it.
    assert origin[1, 3] == pytest.approx(-2.4)
    assert -3.0 <= origin[0, 3] <= 3.0


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
        bodies={0: body_0, 1: body_1},
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )

    resolver = InWorldLayoutResolver(spawned=SpawnedLayout(world=world), groups=[group])
    resolver.resolve()

    assert not _colliding([body_0, body_1], world)


def test_resolver_drops_floor_pieces_it_cannot_separate(
    scene_dir: Path,
) -> None:
    """
    When every redraw lands the pieces back on top of each other, the resolver
    must give up rather than spin forever, and drop what it could not place so
    the room renders without those pieces instead of not at all.

    Previously asserted that it raises :class:`LayoutResolutionError`. It does
    not, and never did: ``resolve`` calls ``_drop_objects`` after its passes,
    which clears the violation, so the raise is unreachable whenever dropping
    succeeds -- the raise is reserved for offenders that survive even that.
    """
    world, root, floor = _world_with_floor()
    body_0 = _floor_object("piece_0", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    body_1 = _floor_object("piece_1", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body_0, 1: body_1},
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )

    # Every slide lands on the same position along the wall, so the two pieces
    # keep coming back on top of each other.
    with patch("random.random", return_value=0.5):
        resolver = InWorldLayoutResolver(
            spawned=SpawnedLayout(world=world), groups=[group], max_passes=3
        )
        resolver.resolve()

    assert resolver.dropped_body_count > 0
    assert not _colliding(list(group.bodies.values()), world)


def test_slide_prefers_a_position_clear_of_the_other_pieces(scene_dir: Path) -> None:
    """
    The wall-relative slide is not occupancy-aware by itself, unlike the floor
    sampler it replaced, whose sample space excluded already-placed objects.
    Without trying several positions two pieces on the same wall land on each
    other repeatedly and the repair loop degenerates into a random search.
    """
    world, root, floor = _world_with_floor()
    blocker = _floor_object("blocker", 0.0, -2.4, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    offender = _floor_object("offender", 0.0, -2.4, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: blocker, 1: offender},
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )

    group.resample_and_move({1})

    moved = offender.parent_connection.origin.to_np()
    # Still on the same wall at the same distance from it ...
    assert moved[1, 3] == pytest.approx(-2.4)
    # ... but clear of the piece it was sitting on top of.
    assert abs(moved[0, 3] - 0.0) > 0.4


def test_resolver_clears_a_piece_that_cuts_into_the_wall_it_stands_against(
    scene_dir: Path,
) -> None:
    """
    A wall is centred on the room's boundary, so a piece standing the measured
    0.25 m from one still cuts into it. The slide holds ``distance_from_wall``
    fixed, so no position along that wall clears the overlap -- yet the group
    checks against its walls every pass, so the piece offends until it is
    dropped, taking fifty fruitless passes to get there.
    """
    world, root, floor = _world_with_floor()
    wall = _south_wall(world, root)
    # 0.2 m from the south wall, which the 0.74 m deep mesh reaches through.
    body = _floor_object("piece_0", 0.0, -2.8, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body},
        supporting_body=None,
        static_obstacles=[wall],
        floor=floor,
        interior=_INTERIOR,
    )

    resolver = InWorldLayoutResolver(
        spawned=SpawnedLayout(world=world), groups=[group], max_passes=5
    )
    resolver.resolve()

    assert resolver.dropped_body_count == 0
    assert not _colliding([body, wall], world)


def test_resolver_leaves_no_collision_detector_callbacks_on_the_world(
    scene_dir: Path,
) -> None:
    """
    A collision detector registers a model- and a state-change callback that
    the world keeps alive, so building one per repair pass never releases the
    previous one: fifty passes leave fifty detectors, each holding a compiled
    forward-kinematics function and one collision model per body. Measured on a
    29-piece room that was about 230 MB per pass, and the process was killed
    out of memory on the thirteenth.
    """
    world, root, floor = _world_with_floor()
    body_0 = _floor_object("piece_0", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    body_1 = _floor_object("piece_1", 0.0, 0.0, 0.0).create_in_world(
        world, scene_dir, parent=root
    )
    group = FloorObjectGroup(
        bodies={0: body_0, 1: body_1},
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )
    callbacks_before = len(world.state.state_change_callbacks)

    resolver = InWorldLayoutResolver(
        spawned=SpawnedLayout(world=world), groups=[group], max_passes=5
    )
    resolver.resolve()

    assert len(world.state.state_change_callbacks) == callbacks_before


def test_dropping_a_furniture_piece_takes_its_contents_with_it(
    scene_dir: Path,
) -> None:
    """
    The floor group holds shelf corpuses and table bodies, whose layers, chairs
    and contents hang beneath them. Removing only the piece's own body leaves
    every child parentless, so the world ends up with as many roots as the
    dropped piece had descendants and the next model change fails its
    single-root assertion -- observed on a 41-piece room after fifty passes.

    Both pieces carry contents, so the invariant holds whichever of them the
    greedy cover picks to drop.
    """
    world, root, floor = _world_with_floor()
    pieces = [
        _floor_object(f"piece_{index}", 0.0, 0.0, 0.0).create_in_world(
            world, scene_dir, parent=root
        )
        for index in range(2)
    ]
    contents = [
        _floor_object(f"content_{index}", 0.0, 0.0, 0.0, z=1.0).create_in_world(
            world, scene_dir, parent=root
        )
        for index in range(2)
    ]
    with world.modify_world():
        for piece, content in zip(pieces, contents):
            world.move_branch(content, piece)
    group = FloorObjectGroup(
        bodies=dict(enumerate(pieces)),
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )

    # Every slide lands on the same position, so the two pieces stay on top of
    # each other and the resolver has to drop one.
    with patch("random.random", return_value=0.5):
        resolver = InWorldLayoutResolver(
            spawned=SpawnedLayout(world=world), groups=[group], max_passes=3
        )
        resolver.resolve()

    # One offending piece, counted once -- the content that left with it was
    # placeable and is not charged to the resolver.
    assert resolver.dropped_body_count == 1
    assert sum(content in world.bodies for content in contents) == 1
    for body in world.bodies:
        if body is not root:
            assert body.parent_connection.parent in world.bodies


def test_dropping_a_furniture_piece_forgets_its_contents_group_members(
    scene_dir: Path,
) -> None:
    """
    A dropped floor piece takes its contents out of the world with it, but those
    contents are members of their own group -- a table's chairs, a shelf layer's
    objects. Left in that group, the next pass asks the detector about bodies
    the world no longer holds, and it raises rather than reporting collisions.
    """
    world, root, floor = _world_with_floor()
    pieces = [
        _floor_object(f"piece_{index}", 0.0, 0.0, 0.0).create_in_world(
            world, scene_dir, parent=root
        )
        for index in range(2)
    ]
    # Well apart, so the contents group has no violations of its own and is
    # never visited by the drop -- the case that leaves it holding a dead body.
    contents = [
        _floor_object(f"content_{index}", -2.0 + 4.0 * index, 0.0, 0.0, z=1.0)
        .create_in_world(world, scene_dir, parent=root)
        for index in range(2)
    ]
    with world.modify_world():
        for piece, content in zip(pieces, contents):
            world.move_branch(content, piece)
    floor_group = FloorObjectGroup(
        bodies=dict(enumerate(pieces)),
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )
    contents_group = FloorObjectGroup(
        bodies=dict(enumerate(contents)),
        supporting_body=None,
        floor=floor,
        interior=_INTERIOR,
    )

    with patch("random.random", return_value=0.5):
        resolver = InWorldLayoutResolver(
            spawned=SpawnedLayout(world=world),
            groups=[floor_group, contents_group],
            max_passes=3,
        )
        resolver.resolve()

    assert all(body in world.bodies for body in contents_group.bodies.values())
