from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.scene_schema import (
    EGScale,
    EGWallRelativePose,
    RoomWall,
    wrap_angle_degrees,
)

_ROOM = EGScale(height=2.7, length=8.0, width=6.0)
"""
A 6 m x 8 m room, so the two axes differ and an axis mix-up cannot pass.
"""


@pytest.mark.parametrize(
    "x, y, expected_wall",
    [
        (0.0, -3.5, RoomWall.SOUTH),
        (2.5, 0.0, RoomWall.EAST),
        (0.0, 3.5, RoomWall.NORTH),
        (-2.5, 0.0, RoomWall.WEST),
    ],
)
def test_nearest_wall_is_the_one_the_piece_actually_stands_against(
    x: float, y: float, expected_wall: RoomWall
) -> None:
    """
    Wall adjacency is the whole point of this parametrisation, so the chosen
    wall must be the nearest one on both axes of a non-square room.
    """
    pose = EGWallRelativePose.from_absolute_pose(x, y, 0.0, _ROOM)

    assert pose.wall == expected_wall


def test_distance_from_wall_is_the_perpendicular_gap_in_metres() -> None:
    """
    Distance stays in absolute metres because a shelf stands the same 0.25 m
    from a wall whatever the room's size -- that invariance is what makes it
    learnable as a single marginal.
    """
    pose = EGWallRelativePose.from_absolute_pose(0.0, -3.75, 0.0, _ROOM)

    assert pose.wall == RoomWall.SOUTH
    assert pose.distance_from_wall == pytest.approx(0.25)


def test_position_along_wall_is_a_fraction_so_it_transfers_across_room_sizes() -> None:
    """
    Expressed as a fraction, a position two thirds along a wall stays two
    thirds along it in a room of any size.
    """
    pose = EGWallRelativePose.from_absolute_pose(1.0, -3.5, 0.0, _ROOM)

    assert pose.wall == RoomWall.SOUTH
    assert pose.position_along_wall == pytest.approx((1.0 + 3.0) / 6.0)


def test_zero_yaw_relative_to_wall_means_facing_into_the_room() -> None:
    """
    A shelf against the south wall faces north, so measuring yaw against the
    wall's inward normal puts the common case at zero.
    """
    against_south = EGWallRelativePose.from_absolute_pose(0.0, -3.5, 90.0, _ROOM)
    against_west = EGWallRelativePose.from_absolute_pose(-2.5, 0.0, 0.0, _ROOM)

    assert against_south.yaw_relative_to_wall == pytest.approx(0.0)
    assert against_west.yaw_relative_to_wall == pytest.approx(0.0)


@pytest.mark.parametrize(
    "x, y, yaw",
    [
        (0.0, -3.5, 90.0),
        (2.5, 0.0, 180.0),
        (0.0, 3.5, -90.0),
        (-2.5, 0.0, 0.0),
        (1.7, -2.9, 37.0),
        (-2.2, 2.4, -140.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_round_trip_reproduces_the_original_pose(x: float, y: float, yaw: float) -> None:
    """
    ``from_absolute_pose`` and ``to_absolute_pose`` must be exact inverses for
    any pose inside its own wall's region, the same contract
    :class:`EGRelativePolarPose` holds for chairs.
    """
    pose = EGWallRelativePose.from_absolute_pose(x, y, yaw, _ROOM)

    recovered_x, recovered_y, recovered_yaw = pose.to_absolute_pose(_ROOM)

    assert recovered_x == pytest.approx(x, abs=1e-9)
    assert recovered_y == pytest.approx(y, abs=1e-9)
    # Compared as an angle: wrap_angle_degrees canonicalises 180 to -180, which
    # is the same bearing.
    assert wrap_angle_degrees(recovered_yaw - yaw) == pytest.approx(0.0, abs=1e-9)


def test_to_absolute_pose_clamps_a_distance_larger_than_the_room() -> None:
    """
    Distance is absolute metres and nothing in a fitted circuit bounds it by the
    room, so a distance drawn from a large room's marginal and applied to a
    small one would otherwise land outside -- reintroducing exactly the
    out-of-room placement this parametrisation exists to prevent.
    """
    pose = EGWallRelativePose(
        wall=RoomWall.SOUTH,
        distance_from_wall=50.0,
        position_along_wall=0.5,
        yaw_relative_to_wall=0.0,
    )

    x, y, _ = pose.to_absolute_pose(_ROOM)

    # Clamped to min(half_width, half_length) = 3.0, measured from the south
    # wall at y = -4.0.
    assert y == pytest.approx(-1.0)
    # The clamp must also preserve the invariant that `wall` is the nearest one.
    assert EGWallRelativePose.from_absolute_pose(x, y, 0.0, _ROOM).wall is RoomWall.SOUTH


@pytest.mark.parametrize("wall", list(RoomWall))
@pytest.mark.parametrize("distance", [0.0, 0.25, 1.5, 99.0])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_every_sampled_pose_lands_inside_the_room(
    wall: RoomWall, distance: float, fraction: float
) -> None:
    """
    Containment must be structural rather than checked afterwards: no
    combination of wall, distance and fraction may place a piece centre outside
    the room.
    """
    pose = EGWallRelativePose(
        wall=wall,
        distance_from_wall=distance,
        position_along_wall=fraction,
        yaw_relative_to_wall=0.0,
    )

    x, y, _ = pose.to_absolute_pose(_ROOM)

    assert -_ROOM.width / 2 - 1e-9 <= x <= _ROOM.width / 2 + 1e-9
    assert -_ROOM.length / 2 - 1e-9 <= y <= _ROOM.length / 2 + 1e-9


@pytest.mark.parametrize(
    "sampled_value, expected_wall",
    [
        (0.0, RoomWall.SOUTH),
        (0.4, RoomWall.SOUTH),
        (1.4, RoomWall.EAST),
        (1.6, RoomWall.NORTH),
        (2.0, RoomWall.NORTH),
        (3.4, RoomWall.WEST),
        (-0.7, RoomWall.SOUTH),
        (9.0, RoomWall.WEST),
    ],
)
def test_a_wall_sampled_as_a_float_is_coerced_back_onto_a_wall(
    sampled_value: float, expected_wall: RoomWall
) -> None:
    """
    A fitted circuit samples the wall index as a continuous value, so a pose
    read back from one carries a float rather than a RoomWall. Rounding it onto
    the nearest wall -- and clamping values outside the range -- is what keeps
    ``to_absolute_pose`` usable on sampled poses at all.
    """
    assert RoomWall.nearest(sampled_value) is expected_wall


def test_to_absolute_pose_accepts_a_wall_sampled_as_a_float() -> None:
    """
    Regression test: a sampled layout raised ``AttributeError: 'float' object
    has no attribute 'runs_along_x'`` the moment it was converted back to
    absolute coordinates.
    """
    pose = EGWallRelativePose(
        wall=1.2,
        distance_from_wall=0.25,
        position_along_wall=0.5,
        yaw_relative_to_wall=0.0,
    )

    x, y, _ = pose.to_absolute_pose(_ROOM)

    assert x == pytest.approx(_ROOM.width / 2 - 0.25)
    assert y == pytest.approx(0.0)
