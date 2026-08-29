from __future__ import annotations

from pathlib import Path

import pytest

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
import experiments.scene_generation_experiments.utils as utils_module
from experiments.scene_generation_experiments.utils import _get_source_ids_for_objects
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGPosition,
    EGRotation,
    ObjectType,
)
from semantic_digital_twin.world_description.geometry import Scale


def _book_dao(scene_dir: Path):
    """
    A real EGObjectDAO -- not a mock -- so a stale attribute name on the generated
    ScaleDAO surfaces the same way it would against the live database.
    """
    book = EGObject(
        id="book_1",
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        # x is length, y is width, z is height -- see EGShelf's
        # CONTENT_FRAME_YAW_OFFSET_DEGREES for the axis convention.
        scale=Scale(x=0.2, y=0.1, z=0.3),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="book_src",
    )
    return to_dao(book)


def test_native_extents_are_read_from_the_scales_x_y_z_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    EGObject2D/EGObject's scale is a single Scale(x, y, z) field, not the old
    width/length/height fields it replaced, so building a candidate's native_extents has
    to read the fields that actually exist on the generated ScaleDAO.
    """
    monkeypatch.setattr(
        utils_module, "build_source_id_to_path", lambda: {"book_src": tmp_path}
    )

    [candidate] = _get_source_ids_for_objects([_book_dao(tmp_path)])

    # native_extents is documented as (width, length, height) == (scale.y, scale.x, scale.z).
    assert candidate.native_extents == pytest.approx((0.1, 0.2, 0.3))


def _scaleless_book_dao(scene_dir: Path):
    """
    A real EGObjectDAO whose ``scale`` column is unset, reproducing rows the live
    database actually holds for some objects.
    """
    book = EGObject(
        id="book_2",
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=None,
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="scaleless_book_src",
    )
    return to_dao(book)


def test_objects_without_a_scale_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A row with no recorded scale carries no native_extents, so it cannot become a
    MeshCandidate and must be left out of the pool rather than crashing the lookup.
    """
    monkeypatch.setattr(
        utils_module,
        "build_source_id_to_path",
        lambda: {"book_src": tmp_path, "scaleless_book_src": tmp_path},
    )

    candidates = _get_source_ids_for_objects(
        [_book_dao(tmp_path), _scaleless_book_dao(tmp_path)]
    )

    assert [candidate.source_id for candidate in candidates] == ["book_src"]
