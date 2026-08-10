from __future__ import annotations

from pathlib import Path

import pytest

from semantic_digital_twin.scene_generation.scene_schema import (
    EGScale,
    MeshCandidate,
    ObjectType,
    _MeshTypeMatcher,
)


def _candidate(
    object_type: ObjectType, width: float, length: float, height: float, name: str = "c"
) -> MeshCandidate:
    return MeshCandidate(
        scene_dir=Path("/scenes") / name,
        source_id=name,
        object_type=object_type,
        native_extents=(width, length, height),
    )


def test_a_missing_type_yields_nothing_rather_than_a_wrong_mesh() -> None:
    """
    Falling back to the whole pool is what strewed generated rooms with
    arbitrary objects: with only 143 floor-capable meshes across 28 types, a
    sampled BED or SOFA silently became whichever mesh was drawn -- a book, a
    piece of wall art. Dropping the piece is honest; substituting is not.
    """
    matcher = _MeshTypeMatcher(
        candidates=[_candidate(ObjectType.BOOK, 0.2, 0.05, 0.25, "book")]
    )

    assert matcher.random_match(ObjectType.SOFA, EGScale(0.8, 0.9, 2.0)) is None


def test_the_closest_size_match_of_the_right_type_is_preferred() -> None:
    """
    The circuit samples a size for each piece, so honouring it is what keeps a
    sampled 0.4 m stool from spawning as a 1.2 m armchair.
    """
    matcher = _MeshTypeMatcher(
        candidates=[
            _candidate(ObjectType.CHAIR, 1.2, 1.2, 1.3, "big"),
            _candidate(ObjectType.CHAIR, 0.45, 0.45, 0.9, "right"),
            _candidate(ObjectType.CHAIR, 0.15, 0.15, 0.2, "tiny"),
        ]
    )

    match = matcher.random_match(
        ObjectType.CHAIR, target_extents=EGScale(height=0.9, length=0.5, width=0.5)
    )

    assert match is not None
    assert match.source_id == "right"


def test_a_candidate_wildly_off_the_sampled_size_is_rejected() -> None:
    """
    A mesh whose real size bears no relation to what was sampled makes the room
    look wrong even though its category is right, so tolerance has a limit.
    """
    matcher = _MeshTypeMatcher(
        candidates=[_candidate(ObjectType.CHAIR, 0.05, 0.05, 0.06, "doll_house")]
    )

    assert (
        matcher.random_match(
            ObjectType.CHAIR, target_extents=EGScale(height=0.9, length=0.5, width=0.5)
        )
        is None
    )


def test_size_is_ignored_when_no_target_is_given() -> None:
    """
    Shelf contents are matched by category alone, so omitting the target size
    must keep the old behaviour for that caller.
    """
    matcher = _MeshTypeMatcher(
        candidates=[_candidate(ObjectType.BOOK, 0.2, 0.05, 0.25, "book")]
    )

    assert matcher.random_match(ObjectType.BOOK) is not None


def test_a_candidate_of_unknown_size_stays_eligible() -> None:
    """
    A pool entry with no recorded extents cannot be judged on size, and
    dropping it would shrink an already-thin pool for no reason.
    """
    unknown = MeshCandidate(
        scene_dir=Path("/scenes/x"),
        source_id="unknown",
        object_type=ObjectType.CHAIR,
        native_extents=None,
    )
    matcher = _MeshTypeMatcher(candidates=[unknown])

    assert (
        matcher.random_match(
            ObjectType.CHAIR, target_extents=EGScale(height=0.9, length=0.5, width=0.5)
        )
        is unknown
    )
