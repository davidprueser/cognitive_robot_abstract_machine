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
from experiments.scene_generation_experiments.proximity_group_generation import (
    _extract_proximity_groups,
    groups_for_circuit_training,
    member_counts_by_anchor_type,
)
from experiments.scene_generation_experiments.proximity_group_collision_resolution import (
    build_member_pose_resample_query,
)
from krrood.ormatic.utils import create_engine
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGGroupMember,
    EGPoint2D,
    EGRelativePolarPose,
    EGRotation,
    EGScale,
    EGProximityGroup,
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
# Nearest-anchor assignment, distance threshold, room scoping
# ---------------------------------------------------------------------------


def test_nearby_objects_are_clustered_into_one_group(session: Session) -> None:
    """
    A room's floor objects must be grouped by spatial proximity, so the
    arrangements a room type actually holds are discovered rather than authored:
    a dining cluster here, and a refrigerator standing on its own.
    """
    anchor = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, width=1.4, length=0.9
    )
    near = [
        _make_object(f"chair_{index}", "room_1", ObjectType.CHAIR, x=x, y=y,
                     width=0.5, length=0.5)
        for index, (x, y) in enumerate([(0.8, 0.0), (-0.8, 0.0), (0.0, 0.8)])
    ]
    far = _make_object(
        "fridge_1", "room_1", ObjectType.REFRIGERATOR, x=6.0, y=6.0,
        width=0.7, length=0.7,
    )
    session.add_all([anchor, *near, far])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    by_anchor_type = {group.object_type: group for group in groups}
    assert set(by_anchor_type) == {ObjectType.TABLE, ObjectType.REFRIGERATOR}
    assert len(by_anchor_type[ObjectType.TABLE].members) == 3
    assert by_anchor_type[ObjectType.REFRIGERATOR].members == []


def test_the_largest_object_in_a_cluster_becomes_its_anchor(
    session: Session,
) -> None:
    """
    A cluster is described from the piece the arrangement is built around, so
    the largest footprint anchors it and everything else is posed relative to
    that -- not to whichever object the query happened to return first.
    """
    small = _make_object(
        "stool_1", "room_1", ObjectType.CHAIR, x=0.0, y=0.0, width=0.4, length=0.4
    )
    large = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.6, y=0.0, width=1.6, length=1.0
    )
    session.add_all([small, large])
    session.commit()

    [group] = _extract_proximity_groups(session)[0]

    assert group.object_type == ObjectType.TABLE
    assert group.position == EGPoint2D(x=0.6, y=0.0)
    assert [member.object_type for member in group.members] == [ObjectType.CHAIR]


def test_distant_objects_form_their_own_group_rather_than_becoming_members(
    session: Session,
) -> None:
    """
    Proximity is what defines membership, so an object beyond the neighbourhood
    radius must anchor its own group instead of being attached to a far-away
    piece it has no relationship with.
    """
    anchor = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
    far_chair = _make_object("chair_far", "room_1", ObjectType.CHAIR, x=5.0, y=0.0)
    session.add_all([anchor, far_chair])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) == 2
    assert all(group.members == [] for group in groups)


def test_objects_in_different_rooms_are_never_grouped_together(
    session: Session,
) -> None:
    """
    Cross-room distance comparisons are meaningless -- room-local coordinates
    make objects in two different rooms look adjacent -- so clustering must run
    per room.
    """
    anchor = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, width=1.4, length=0.9
    )
    chair_in_other_room = _make_object(
        "chair_1", "room_2", ObjectType.CHAIR, x=0.1, y=0.0
    )
    session.add_all([anchor, chair_in_other_room])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) == 2
    assert all(group.members == [] for group in groups)


def test_member_counts_are_reported_per_anchor_type(session: Session) -> None:
    """
    How many members a group holds depends on what it is built around -- a
    dining table gathers chairs, a refrigerator stands alone -- and that count
    is a structural property of the sampling query, drawn before it is built.
    Pooling the counts across anchor types would surround every fridge with
    chairs.
    """
    anchor = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, width=1.4, length=0.9
    )
    chairs = [
        _make_object(f"chair_{index}", "room_1", ObjectType.CHAIR, x=x, y=y,
                     width=0.5, length=0.5)
        for index, (x, y) in enumerate([(0.8, 0.0), (-0.8, 0.0)])
    ]
    fridge = _make_object(
        "fridge_1", "room_1", ObjectType.REFRIGERATOR, x=6.0, y=6.0,
        width=0.7, length=0.7,
    )
    session.add_all([anchor, *chairs, fridge])
    session.commit()

    groups, _ = _extract_proximity_groups(session)
    counts = member_counts_by_anchor_type(groups)

    assert counts[ObjectType.TABLE] == [2]
    assert counts[ObjectType.REFRIGERATOR] == [0]


def test_groups_without_members_are_kept_out_of_the_circuit_training_set(
    session: Session,
) -> None:
    """
    Most clusters are singletons, and the feature extractor decides whether the
    member relation gets an exchangeable template at all by inspecting only the
    *first* training instance's collection. A memberless group arriving first
    silently suppresses member modelling for the whole circuit, with no error
    and no members ever sampled -- so they are excluded, while still counting
    towards :func:`member_counts_by_anchor_type` so a fridge keeps drawing zero.
    """
    anchor = _make_object(
        "table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, width=1.4, length=0.9
    )
    chair = _make_object(
        "chair_1", "room_1", ObjectType.CHAIR, x=0.8, y=0.0, width=0.5, length=0.5
    )
    lone = _make_object(
        "fridge_1", "room_1", ObjectType.REFRIGERATOR, x=6.0, y=6.0,
        width=0.7, length=0.7,
    )
    session.add_all([anchor, chair, lone])
    session.commit()

    groups, _ = _extract_proximity_groups(session)
    training_groups = groups_for_circuit_training(groups)

    assert len(groups) == 2
    assert [group.object_type for group in training_groups] == [ObjectType.TABLE]


def test_object_resting_on_a_table_is_not_itself_a_table_candidate(
    session: Session,
) -> None:
    """
    A small item lying on an anchor -- e.g. a tablecloth, which the object-type
    classifier generalizes to :attr:`ObjectType.TABLE` -- must never compete as
    an anchor candidate.

    It sits at almost the same position as the anchor it rests on, so the
    nearest-anchor assignment would hand it the real anchor's members and fit the
    RSPN's anchor scale to a decimetre-sized box.
    """
    anchor = _make_object(
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
    member = _make_object("chair_1", "room_1", ObjectType.CHAIR, x=0.9, y=0.0)
    session.add_all([anchor, tablecloth, member])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) == 1
    assert groups[0].scale == EGScale(width=1.2, length=0.8, height=0.75)
    assert len(groups[0].members) == 1


def test_object_resting_on_a_table_is_not_grouped_as_a_chair(
    session: Session,
) -> None:
    """
    A small item lying on an anchor whose raw name makes the classifier call it a
    :attr:`ObjectType.CHAIR` -- e.g. ``"bookchair..."`` -- must not be grouped
    as one of the anchor's members, or the group learns book-sized members.
    """
    anchor = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0)
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
    session.add_all([anchor, book_on_table, real_chair])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) == 1
    assert [member.id for member in groups[0].members] == ["chair_1"]


# ---------------------------------------------------------------------------
# Local-frame conversion
# ---------------------------------------------------------------------------


def test_relative_pose_is_invariant_to_the_tables_absolute_pose(
    session: Session,
) -> None:
    """
    Two groups with identical member layouts relative to themselves, but
    different absolute positions and orientations in the room, must produce
    identical EGRelativePolarPose values for their members.

    This proves the extraction subtracts the anchor's own yaw before
    computing distance/angle, which is what lets "roughly evenly spaced,
    facing the anchor" be learnable independent of where an anchor happens
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

    groups, _ = _extract_proximity_groups(session)

    groups_by_room = {group.members[0].room_id: group for group in groups}
    pose_a = groups_by_room["room_1"].members[0].relative_pose
    pose_b = groups_by_room["room_2"].members[0].relative_pose

    assert pose_a.distance_from_anchor == pytest.approx(
        pose_b.distance_from_anchor
    )
    assert pose_a.angle_from_anchor == pytest.approx(
        pose_b.angle_from_anchor
    )
    assert pose_a.facing_angle_relative_to_anchor == pytest.approx(
        pose_b.facing_angle_relative_to_anchor
    )


def test_chair_facing_the_table_head_on_has_zero_facing_angle(
    session: Session,
) -> None:
    """
    A member placed directly "south" of an axis-aligned anchor and yawed to face
    the anchor centre must compute a facing_angle_relative_to_anchor of
    (approximately) zero, per the zero-means-facing-dead-on convention.
    """
    anchor = _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0, yaw=0.0)
    member = _make_object(
        "chair_1", "room_1", ObjectType.CHAIR, x=0.0, y=-1.0, yaw=90.0
    )
    session.add_all([anchor, member])
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) == 1
    relative_pose = groups[0].members[0].relative_pose
    assert relative_pose.facing_angle_relative_to_anchor == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# create_in_world round trip
# ---------------------------------------------------------------------------


def test_create_in_world_places_chair_at_the_expected_absolute_pose(tmp_path) -> None:
    """
    EGGroupMember.create_in_world must convert the anchor-relative polar pose back
    into an absolute pose using the anchor's own position and orientation -- the
    exact inverse of the extraction conversion.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    anchor_position = EGPoint2D(x=2.0, y=1.0)
    anchor_orientation = EGRotation(x=0.0, y=0.0, z=90.0)
    member = EGGroupMember(
        id="chair_1",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_anchor=1.0,
            angle_from_anchor=0.0,
            facing_angle_relative_to_anchor=0.0,
        ),
        source_id="chair_src",
    )

    table_yaw_radians = math.radians(anchor_orientation.z)
    expected_x = anchor_position.x + 1.0 * math.cos(table_yaw_radians)
    expected_y = anchor_position.y + 1.0 * math.sin(table_yaw_radians)

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = member.create_in_world(
        world,
        tmp_path,
        parent=root,
        anchor_position=anchor_position,
        anchor_orientation=anchor_orientation,
    )

    translation = body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(expected_x, abs=1e-6)
    assert translation[1] == pytest.approx(expected_y, abs=1e-6)


def test_create_in_world_rests_the_chair_on_the_floor(tmp_path) -> None:
    """
    EGGroupMember.create_in_world must place the body at z=0, not lift it by half its
    own height.

    The member PLY assets are modelled with their origin at the base
    (their lowest vertex sits at z~=0, not centred on the object), the
    same convention EGObject2D.create_in_world relies on for shelf
    contents. Half-height had been added here by mistake, which floated
    every member in the air above the floor.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    member = EGGroupMember(
        id="chair_1",
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_anchor=1.0,
            angle_from_anchor=0.0,
            facing_angle_relative_to_anchor=0.0,
        ),
        source_id="chair_src",
    )

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = member.create_in_world(
        world,
        tmp_path,
        parent=root,
        anchor_position=EGPoint2D(x=0.0, y=0.0),
        anchor_orientation=EGRotation(x=0.0, y=0.0, z=0.0),
    )

    translation = body.parent_connection.origin.to_position().to_np()
    assert translation[2] == pytest.approx(0.0, abs=1e-6)


def test_table_with_chairs_create_in_world_places_every_chair_facing_the_table(
    tmp_path,
) -> None:
    """
    EGProximityGroup.create_in_world must actually place each member (not just
    the anchor corpus), and a member whose facing_angle_relative_to_anchor is zero
    must end up yawed to face the anchor centre in the room frame -- exercising
    the group-level mesh-matching and placement loop end to end, not just the
    per-member pose conversion in isolation.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )

    anchor_position = EGPoint2D(x=0.0, y=0.0)
    anchor_orientation = EGRotation(x=0.0, y=0.0, z=0.0)
    member = EGGroupMember(
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
    table_with_chairs = EGProximityGroup(
        position=anchor_position,
        scale=EGScale(height=0.75, length=1.2, width=0.8),
        orientation=anchor_orientation,
        members=[member],
        source_ids=[MeshCandidate(tmp_path, "chair_src", ObjectType.CHAIR)],
    )

    world = table_with_chairs.create_in_world()

    member_bodies = [body for body in world.bodies if body.name.prefix == "chair_1"]
    assert len(member_bodies) == 1

    member_body = member_bodies[0]
    translation = member_body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(0.0, abs=1e-6)
    assert translation[1] == pytest.approx(-1.0, abs=1e-6)

    yaw = member_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(math.pi / 2, abs=1e-6)


def _chair(chair_id: str, distance: float, angle: float) -> EGGroupMember:
    return EGGroupMember(
        id=chair_id,
        room_id="room_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(height=0.9, length=0.5, width=0.5),
        relative_pose=EGRelativePolarPose(
            distance_from_anchor=distance,
            angle_from_anchor=angle,
            facing_angle_relative_to_anchor=0.0,
        ),
        source_id="chair_src",
    )


def test_build_chair_pose_resample_query_frees_resampled_scale_and_pose() -> None:
    """
    build_member_pose_resample_query must condition only the fixed members'
    scale and relative pose, leaving the resampled member's scale and relative
    pose both free to be redrawn.

    Conditioning a resampled slot on its own scale pins the query to the
    single training example that combination of evidence (its own scale plus
    every fixed neighbour's exact pose) came from, collapsing the RSPN's
    posterior for that slot's relative pose back to its original,
    still-colliding value -- observed as a repair pass that redraws the exact
    same pose every time and so can never actually resolve a collision.
    Regression test for that collapse.
    """
    query = build_member_pose_resample_query(
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
        name for name in conditioned_names if "distance_from_anchor" in name
    ]
    # "members[" scopes to per-member scale, excluding the anchor's own
    # (always-fixed) EGProximityGroup.scale.width.
    conditioned_scales = [
        name for name in conditioned_names if "members[" in name and "scale.width" in name
    ]
    # Only the one fixed member's relative pose and scale are conditioned; the
    # resampled member's are left entirely free.
    assert len(conditioned_distances) == 1
    assert len(conditioned_scales) == 1


def test_build_chair_pose_resample_query_does_not_pin_the_table_pose() -> None:
    """
    Chair poses are polar and relative to their anchor, with the anchor's yaw
    already subtracted, so neither the anchor's absolute position nor its
    orientation carries information about them -- conditioning on either only
    shrinks the circuit's support.

    Repair passes hand this query the anchor's room-centred position, while the
    circuit is fitted on raw room coordinates. Pinning the position therefore
    conditioned on a coordinate the circuit assigns zero mass (a negative x for
    any anchor left of the room centre), so both the primary and the relaxed
    query raised NoSolutionFound and the whole layout aborted.
    """
    query = build_member_pose_resample_query(
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
    # The anchor's own scale stays as evidence: a wider anchor seats its members
    # further out, so it genuinely informs the redrawn poses.
    assert [
        name
        for name in conditioned_names
        if "scale" in name and "members[" not in name
    ]


def test_a_chain_of_objects_does_not_collapse_into_one_group(
    session: Session,
) -> None:
    """
    Membership must be bounded by how far apart the group's own objects stand,
    not by whether a path of short hops connects them.

    Density-based clustering links A to C whenever some B lies within reach of
    both, so in a furnished room every piece is transitively connected and the
    whole room becomes a single group -- measured at ~12 pieces per cluster over
    the training set, with sampled groups of 26 and 28 members around a single
    chair.
    """
    spaced = [
        _make_object(f"piece_{index}", "room_1", ObjectType.CHAIR, x=1.4 * index, y=0.0,
                     width=0.5, length=0.5)
        for index in range(6)
    ]
    session.add_all(spaced)
    session.commit()

    groups, _ = _extract_proximity_groups(session)

    assert len(groups) > 1
    assert max(len(group.members) for group in groups) < 5


def test_an_anchor_rendered_from_a_mesh_rests_on_the_floor(tmp_path) -> None:
    """
    A group anchor spawned from a mesh must stand on the floor.

    The anchor pose lifts by half the anchor's height, which is right for the
    box primitive an anchor used to be -- a box is centred on its own origin.
    A sage10k mesh instead carries its base at z = 0, so the same lift leaves it
    hovering by half its height, and since nearly every floor piece anchors a
    group that is most of the room floating.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "anchor_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "anchor_src_texture.png"
    )

    group = EGProximityGroup(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(width=0.69, length=0.74, height=0.88),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        object_type=ObjectType.CHAIR,
        members=[],
        anchor_mesh=MeshCandidate(tmp_path, "anchor_src", ObjectType.CHAIR),
    )

    spawned = group.spawn_in_world()

    anchor_z = spawned.anchor.parent_connection.origin.to_np()[2, 3]
    lowest_point = anchor_z + float(spawned.anchor.collision.combined_mesh.bounds[0][2])
    assert lowest_point == pytest.approx(0.0, abs=1e-2)
