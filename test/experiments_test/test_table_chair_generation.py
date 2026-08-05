from __future__ import annotations

import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    EGObjectDAO,
    EGRotationDAO,
    EGPositionDAO,
    EGScaleDAO,
)
from experiments.scene_generation_experiments.table_chair_generation import (
    _extract_table_chair_groups_from_spatial_proximity,
)
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_chair_pose_resample_query,
)
from krrood.ormatic.utils import create_engine
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGChair,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGTableWithChairs,
    MeshCandidate,
    ObjectType,
    PlaceId,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    database_session = Session(engine)
    yield database_session
    database_session.close()


def _make_object(
    object_id: str,
    room_id: str,
    object_type: ObjectType,
    x: float,
    y: float,
    yaw: float = 0.0,
    width: float = 0.8,
    length: float = 1.2,
    height: float = 0.75,
    place_id: str = PlaceId.FLOOR,
) -> EGObjectDAO:
    return EGObjectDAO(
        id=object_id,
        room_id=room_id,
        place_id=place_id,
        source_id=f"{object_id}_src",
        object_type=object_type,
        scale=EGScaleDAO(height=height, length=length, width=width),
        position=EGPositionDAO(x=x, y=y, z=height / 2),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=yaw),
    )


# ---------------------------------------------------------------------------
# Nearest-table assignment, distance threshold, room scoping
# ---------------------------------------------------------------------------


def test_chair_is_assigned_to_the_nearest_table(session: Session) -> None:
    """
    A chair closer to one of two tables in the same room must be grouped with
    that nearer table, not the farther one.
    """
    near_table = _make_object("table_near", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    far_table = _make_object("table_far", "room_1", ObjectType.TABLE, x=10.0, y=10.0)
    chair = _make_object("chair_1", "room_1", ObjectType.CHAIR, x=0.5, y=0.0)
    session.add_all([near_table, far_table, chair])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert len(groups) == 1
    assert len(groups[0].chairs) == 1
    assert groups[0].position == EGPoint2D(x=0.0, y=0.0)


def test_chair_beyond_max_distance_is_not_assigned(session: Session) -> None:
    """
    A chair farther than max_distance_from_table from every table in its room
    must be excluded entirely, rather than force-assigned to the nearest
    (implausibly far) table.
    """
    table = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    far_chair = _make_object("chair_far", "room_1", ObjectType.CHAIR, x=5.0, y=0.0)
    session.add_all([table, far_chair])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(
        session, max_distance_from_table=1.5
    )

    assert groups == []


def test_chair_in_different_room_is_not_assigned_despite_proximity(
    session: Session,
) -> None:
    """
    Cross-room distance comparisons are meaningless -- a chair must never be
    assigned to a table in a different room, even if geometrically close.
    """
    table = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    chair_in_other_room = _make_object(
        "chair_1", "room_2", ObjectType.CHAIR, x=0.1, y=0.0
    )
    session.add_all([table, chair_in_other_room])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert groups == []


def test_table_with_no_assigned_chairs_is_dropped(session: Session) -> None:
    """
    A bare table with zero surrounding chairs must not appear in the extraction
    result.

    This is correctness-critical, not just a quality filter: the RSPN's
    feature extractor decides whether a relation gets an exchangeable
    template and aggregation features at all by inspecting only the
    first training instance's collection -- an empty ``chairs`` list on
    the first instance would silently suppress chair modelling for every
    table.
    """
    lonely_table = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    session.add(lonely_table)
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert groups == []


def test_object_resting_on_a_table_is_not_itself_a_table_candidate(
    session: Session,
) -> None:
    """
    A small item lying on a table -- e.g. a tablecloth, which the object-type
    classifier generalizes to :attr:`ObjectType.TABLE` -- must never compete as
    a table candidate.

    It sits at almost the same position as the table it rests on, so the
    nearest-table assignment would hand it the real table's chairs and fit the
    RSPN's table scale to a decimetre-sized box.
    """
    table = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, width=1.2, length=0.8
    )
    tablecloth = _make_object(
        "tablecloth_1",
        "room_1",
        ObjectType.TABLE,
        x=0.05,
        y=0.0,
        width=0.2,
        length=0.2,
        height=0.02,
        place_id="table_1",
    )
    chair = _make_object("chair_1", "room_1", ObjectType.CHAIR, x=0.9, y=0.0)
    session.add_all([table, tablecloth, chair])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert len(groups) == 1
    assert groups[0].scale == EGScale(width=1.2, length=0.8, height=0.75)
    assert len(groups[0].chairs) == 1


def test_object_resting_on_a_table_is_not_grouped_as_a_chair(
    session: Session,
) -> None:
    """
    A small item lying on a table whose raw name makes the classifier call it a
    :attr:`ObjectType.CHAIR` -- e.g. ``"bookchair..."`` -- must not be grouped
    as one of the table's chairs, or the group learns book-sized chairs.
    """
    table = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    book_on_table = _make_object(
        "bookchair_1",
        "room_1",
        ObjectType.CHAIR,
        x=0.1,
        y=0.0,
        width=0.2,
        length=0.15,
        height=0.05,
        place_id="table_1",
    )
    real_chair = _make_object("chair_1", "room_1", ObjectType.CHAIR, x=0.9, y=0.0)
    session.add_all([table, book_on_table, real_chair])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert len(groups) == 1
    assert [chair.id for chair in groups[0].chairs] == ["chair_1"]


# ---------------------------------------------------------------------------
# Local-frame conversion
# ---------------------------------------------------------------------------


def test_relative_pose_is_invariant_to_the_tables_absolute_pose(
    session: Session,
) -> None:
    """
    Two tables with identical chair layouts relative to themselves, but
    different absolute positions and orientations in the room, must produce
    identical EGRelativePolarPose values for their chairs.

    This proves the extraction subtracts the table's own yaw before
    computing distance/angle, which is what lets "roughly evenly spaced,
    facing the table" be learnable independent of where a table happens
    to sit in a given room.
    """
    table_a = _make_object(
        "table_a", "room_1", ObjectType.TABLE, x=0.0, y=0.0, yaw=0.0
    )
    chair_a = _make_object(
        "chair_a", "room_1", ObjectType.CHAIR, x=1.0, y=0.0, yaw=180.0
    )

    table_b = _make_object(
        "table_b", "room_2", ObjectType.TABLE, x=5.0, y=-3.0, yaw=40.0
    )
    yaw_b_radians = math.radians(40.0)
    chair_b_x = 5.0 + 1.0 * math.cos(yaw_b_radians)
    chair_b_y = -3.0 + 1.0 * math.sin(yaw_b_radians)
    chair_b = _make_object(
        "chair_b", "room_2", ObjectType.CHAIR, x=chair_b_x, y=chair_b_y, yaw=180.0 + 40.0
    )

    session.add_all([table_a, chair_a, table_b, chair_b])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    groups_by_room = {group.chairs[0].room_id: group for group in groups}
    pose_a = groups_by_room["room_1"].chairs[0].relative_pose
    pose_b = groups_by_room["room_2"].chairs[0].relative_pose

    assert pose_a.distance_from_table_center == pytest.approx(
        pose_b.distance_from_table_center
    )
    assert pose_a.angle_from_table_center == pytest.approx(
        pose_b.angle_from_table_center
    )
    assert pose_a.facing_angle_relative_to_table == pytest.approx(
        pose_b.facing_angle_relative_to_table
    )


def test_chair_facing_the_table_head_on_has_zero_facing_angle(
    session: Session,
) -> None:
    """
    A chair placed directly "south" of an axis-aligned table and yawed to face
    the table centre must compute a facing_angle_relative_to_table of
    (approximately) zero, per the zero-means-facing-dead-on convention.
    """
    table = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, yaw=0.0)
    chair = _make_object(
        "chair_1", "room_1", ObjectType.CHAIR, x=0.0, y=-1.0, yaw=90.0
    )
    session.add_all([table, chair])
    session.commit()

    groups, _ = _extract_table_chair_groups_from_spatial_proximity(session)

    assert len(groups) == 1
    relative_pose = groups[0].chairs[0].relative_pose
    assert relative_pose.facing_angle_relative_to_table == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# create_in_world round trip
# ---------------------------------------------------------------------------


def test_create_in_world_places_chair_at_the_expected_absolute_pose(tmp_path) -> None:
    """
    EGChair.create_in_world must convert the table-relative polar pose back
    into an absolute pose using the table's own position and orientation -- the
    exact inverse of the extraction conversion.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    table_position = EGPoint2D(x=2.0, y=1.0)
    table_orientation = EGRotation(x=0.0, y=0.0, z=90.0)
    chair = EGChair(
        id="chair_1",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=1.0,
            angle_from_table_center=0.0,
            facing_angle_relative_to_table=0.0,
        ),
        source_id="chair_src",
    )

    table_yaw_radians = math.radians(table_orientation.z)
    expected_x = table_position.x + 1.0 * math.cos(table_yaw_radians)
    expected_y = table_position.y + 1.0 * math.sin(table_yaw_radians)

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = chair.create_in_world(
        world,
        tmp_path,
        parent=root,
        table_position=table_position,
        table_orientation=table_orientation,
    )

    translation = body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(expected_x, abs=1e-6)
    assert translation[1] == pytest.approx(expected_y, abs=1e-6)


def test_create_in_world_rests_the_chair_on_the_floor(tmp_path) -> None:
    """
    EGChair.create_in_world must place the body at z=0, not lift it by half its
    own height.

    The chair PLY assets are modelled with their origin at the base
    (their lowest vertex sits at z~=0, not centred on the object), the
    same convention EGObject2D.create_in_world relies on for shelf
    contents. Half-height had been added here by mistake, which floated
    every chair in the air above the floor.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    chair = EGChair(
        id="chair_1",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=1.0,
            angle_from_table_center=0.0,
            facing_angle_relative_to_table=0.0,
        ),
        source_id="chair_src",
    )

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = chair.create_in_world(
        world,
        tmp_path,
        parent=root,
        table_position=EGPoint2D(x=0.0, y=0.0),
        table_orientation=EGRotation(x=0.0, y=0.0, z=0.0),
    )

    translation = body.parent_connection.origin.to_position().to_np()
    assert translation[2] == pytest.approx(0.0, abs=1e-6)


def test_table_with_chairs_create_in_world_places_every_chair_facing_the_table(
    tmp_path,
) -> None:
    """
    EGTableWithChairs.create_in_world must actually place each chair (not just
    the table corpus), and a chair whose facing_angle_relative_to_table is zero
    must end up yawed to face the table centre in the room frame -- exercising
    the group-level mesh-matching and placement loop end to end, not just the
    per-chair pose conversion in isolation.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    table_position = EGPoint2D(x=0.0, y=0.0)
    table_orientation = EGRotation(x=0.0, y=0.0, z=0.0)
    chair = EGChair(
        id="chair_1",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=1.0,
            angle_from_table_center=-90.0,
            facing_angle_relative_to_table=0.0,
        ),
        source_id="chair_src",
    )
    table_with_chairs = EGTableWithChairs(
        position=table_position,
        scale=EGScale(height=0.75, length=1.2, width=0.8),
        orientation=table_orientation,
        chairs=[chair],
        source_ids=[MeshCandidate(tmp_path, "chair_src", ObjectType.CHAIR)],
    )

    world = table_with_chairs.create_in_world()

    chair_bodies = [body for body in world.bodies if body.name.prefix == "chair_1"]
    assert len(chair_bodies) == 1

    chair_body = chair_bodies[0]
    translation = chair_body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(0.0, abs=1e-6)
    assert translation[1] == pytest.approx(-1.0, abs=1e-6)

    yaw = chair_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(math.pi / 2, abs=1e-6)


def _chair(chair_id: str, distance: float, angle: float) -> EGChair:
    return EGChair(
        id=chair_id,
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_table_center=distance,
            angle_from_table_center=angle,
            facing_angle_relative_to_table=0.0,
        ),
        source_id="chair_src",
    )


def test_build_chair_pose_resample_query_frees_resampled_scale_and_pose() -> None:
    """
    build_chair_pose_resample_query must condition only the fixed chairs'
    scale and relative pose, leaving the resampled chair's scale and relative
    pose both free to be redrawn.

    Conditioning a resampled slot on its own scale pins the query to the
    single training example that combination of evidence (its own scale plus
    every fixed neighbour's exact pose) came from, collapsing the RSPN's
    posterior for that slot's relative pose back to its original,
    still-colliding value -- observed as a repair pass that redraws the exact
    same pose every time and so can never actually resolve a collision.
    Regression test for that collapse.
    """
    query = build_chair_pose_resample_query(
        [_chair("fixed", distance=1.0, angle=0.0)],
        [_chair("resampled", distance=1.0, angle=0.0)],
        EGScale(height=0.75, length=1.2, width=0.8),
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    conditioned_distances = [
        name for name in conditioned_names if "distance_from_table_center" in name
    ]
    # "chairs[" scopes to per-chair scale, excluding the table's own
    # (always-fixed) EGTableWithChairs.scale.width.
    conditioned_scales = [
        name for name in conditioned_names if "chairs[" in name and "scale.width" in name
    ]
    # Only the one fixed chair's relative pose and scale are conditioned; the
    # resampled chair's are left entirely free.
    assert len(conditioned_distances) == 1
    assert len(conditioned_scales) == 1


def test_build_chair_pose_resample_query_does_not_pin_the_table_pose() -> None:
    """
    Chair poses are polar and relative to their table, with the table's yaw
    already subtracted, so neither the table's absolute position nor its
    orientation carries information about them -- conditioning on either only
    shrinks the circuit's support.

    Repair passes hand this query the table's room-centred position, while the
    circuit is fitted on raw room coordinates. Pinning the position therefore
    conditioned on a coordinate the circuit assigns zero mass (a negative x for
    any table left of the room centre), so both the primary and the relaxed
    query raised NoSolutionFound and the whole layout aborted.
    """
    query = build_chair_pose_resample_query(
        [_chair("fixed", distance=1.0, angle=0.0)],
        [_chair("resampled", distance=1.0, angle=0.0)],
        EGScale(height=0.75, length=1.2, width=0.8),
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }

    assert not [name for name in conditioned_names if "position" in name]
    assert not [name for name in conditioned_names if "orientation" in name]
    # The table's own scale stays as evidence: a wider table seats its chairs
    # further out, so it genuinely informs the redrawn poses.
    assert [
        name
        for name in conditioned_names
        if "scale" in name and "chairs[" not in name
    ]
