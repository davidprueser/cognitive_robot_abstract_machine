from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import trimesh
from sqlalchemy import event
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    EGObjectDAO,
    EGRotationDAO,
    EGPositionDAO,
    EGScaleDAO,
)
from experiments.scene_generation_experiments.book_shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.utils import _get_source_ids_for_objects
from experiments.scene_generation_experiments.collision_resolution import (
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_rare_object_types,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    ObjectType,
    _MeshSizeMatcher,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import ShelfLayer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    pass

_FAKE_PATH = Path("/fake/scene")
_SHELF_ID = "room_1_shelf_1"


@dataclass
class _MockShelfObject:
    """
    Duck-type substitute for EGObjectDAO in source-ID filtering tests.
    """

    object_type: ObjectType
    source_id: str


@pytest.fixture
def source_path_map() -> dict[str, Path]:
    return {"book_src": _FAKE_PATH, "cup_src": _FAKE_PATH}


@pytest.fixture
def mixed_mock_objects() -> list[_MockShelfObject]:
    return [
        _MockShelfObject(object_type=ObjectType.BOOK, source_id="book_src"),
        _MockShelfObject(object_type=ObjectType.CUP, source_id="cup_src"),
        _MockShelfObject(object_type=ObjectType.SHELF, source_id="shelf_src"),
    ]


# ---------------------------------------------------------------------------
# Group A – _get_source_ids_for_objects (no DB required)
# ---------------------------------------------------------------------------


def test_default_object_type_includes_only_books(
    mixed_mock_objects: list[_MockShelfObject], source_path_map: dict[str, Path]
) -> None:
    """
    The default (``ObjectType.BOOK``) filter must include books and exclude
    cups and shelf-furniture objects.
    """
    with patch(
        "experiments.scene_generation_experiments.book_shelf_generation.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(mixed_mock_objects)
    source_ids = {source_id for _, source_id in result}
    assert "book_src" in source_ids
    assert "cup_src" not in source_ids
    assert "shelf_src" not in source_ids


def test_no_object_type_filter_includes_every_type(
    mixed_mock_objects: list[_MockShelfObject], source_path_map: dict[str, Path]
) -> None:
    """
    Passing ``object_type=None`` must include every type present in the input,
    subject only to source_id availability.
    """
    with patch(
        "experiments.scene_generation_experiments.book_shelf_generation.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(mixed_mock_objects, object_type=None)
    source_ids = {source_id for _, source_id in result}
    assert "book_src" in source_ids
    assert "cup_src" in source_ids
    assert "shelf_src" not in source_ids


def test_missing_source_id_is_excluded(source_path_map: dict[str, Path]) -> None:
    """
    Objects whose source_id has no corresponding PLY path must be silently
    dropped regardless of the object-type filter.
    """
    objects_without_path = [
        _MockShelfObject(object_type=ObjectType.BOOK, source_id="nonexistent_src"),
    ]
    with patch(
        "experiments.scene_generation_experiments.book_shelf_generation.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(objects_without_path, object_type=None)
    assert result == []


# ---------------------------------------------------------------------------
# Group B – _extract_shelf_layers_from_place_id (in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    database_session = Session(engine)
    yield database_session
    database_session.close()


@pytest.fixture
def shelf_with_book_and_cup(session: Session) -> Session:
    """
    Populate the session with one shelf, one book, and one cup all sharing the
    same shelf place_id so every object_type filter variant can be exercised.
    """
    shelf = EGObjectDAO(
        id=_SHELF_ID,
        room_id="room_1",
        place_id="floor",
        source_id="shelf_src",
        object_type=ObjectType.SHELF,
        scale=EGScaleDAO(height=2.0, length=1.0, width=0.5),
        position=EGPositionDAO(x=0.0, y=0.0, z=1.0),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    book = EGObjectDAO(
        id="book_1",
        room_id="room_1",
        place_id=_SHELF_ID,
        source_id="book_src",
        object_type=ObjectType.BOOK,
        scale=EGScaleDAO(height=0.3, length=0.1, width=0.05),
        position=EGPositionDAO(x=0.0, y=0.0, z=0.5),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    cup = EGObjectDAO(
        id="cup_1",
        room_id="room_1",
        place_id=_SHELF_ID,
        source_id="cup_src",
        object_type=ObjectType.CUP,
        scale=EGScaleDAO(height=0.1, length=0.1, width=0.1),
        position=EGPositionDAO(x=0.05, y=0.05, z=0.5),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    session.add_all([shelf, book, cup])
    session.commit()
    return session


def test_extract_shelf_layers_does_not_scale_query_count_with_object_count(
    session: Session,
) -> None:
    """
    _extract_shelf_layers_from_place_id must not issue a separate SQL
    statement per object for each of its scale/position/orientation
    relationships -- the number of executed statements must stay bounded
    regardless of how many objects are on the shelf.

    Before eager loading was added, each relationship access on each of the
    20 books below triggered its own lazy-load query once the session
    expired the loaded instances on commit, so statement count grew
    linearly with object count.
    """
    shelf = EGObjectDAO(
        id=_SHELF_ID,
        room_id="room_1",
        place_id="floor",
        source_id="shelf_src",
        object_type=ObjectType.SHELF,
        scale=EGScaleDAO(height=2.0, length=1.0, width=0.5),
        position=EGPositionDAO(x=0.0, y=0.0, z=1.0),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    books = [
        EGObjectDAO(
            id=f"book_{i}",
            room_id="room_1",
            place_id=_SHELF_ID,
            source_id=f"book_src_{i}",
            object_type=ObjectType.BOOK,
            scale=EGScaleDAO(height=0.3, length=0.1, width=0.05),
            position=EGPositionDAO(x=0.01 * i, y=0.0, z=0.5),
            orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
        )
        for i in range(20)
    ]
    session.add_all([shelf, *books])
    session.commit()

    statement_count = 0

    def _count_statement(*args, **kwargs) -> None:
        nonlocal statement_count
        statement_count += 1

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", _count_statement)
    try:
        _extract_shelf_layers_from_place_id(session)
    finally:
        event.remove(engine, "before_cursor_execute", _count_statement)

    assert statement_count <= 5


def test_default_object_type_excludes_cups_from_layers(
    shelf_with_book_and_cup: Session,
) -> None:
    """
    With the default ``ObjectType.BOOK`` filter, only the book must appear in
    the extracted shelf layers — the cup must be absent.
    """
    layers, _ = _extract_shelf_layers_from_place_id(shelf_with_book_and_cup)
    all_source_ids = {obj.source_id for layer in layers for obj in layer.objects}
    assert "book_src" in all_source_ids
    assert "cup_src" not in all_source_ids


def test_no_object_type_filter_includes_cup_and_book_in_layers(
    shelf_with_book_and_cup: Session,
) -> None:
    """
    Passing ``object_type=None`` must include both the book and the cup in the
    extracted shelf layers.
    """
    layers, _ = _extract_shelf_layers_from_place_id(
        shelf_with_book_and_cup, object_type=None
    )
    all_source_ids = {obj.source_id for layer in layers for obj in layer.objects}
    assert "book_src" in all_source_ids
    assert "cup_src" in all_source_ids


# ---------------------------------------------------------------------------
# Object-type coarsening – keep RSPN training's categorical domain small
# ---------------------------------------------------------------------------


def _typed_object(object_type: ObjectType, object_id: str) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=object_type,
        scale=EGScale(height=0.1, length=0.1, width=0.1),
        position=EGPoint2D(x=0.0, y=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id=object_id,
    )


def test_coarsen_rare_object_types_keeps_only_the_most_frequent_types() -> None:
    """
    Object types outside the keep_count most frequent ones must be replaced
    with ObjectType.OTHER; every other field must be preserved unchanged.

    The sage10k dataset's object_type labels are close to per-instance
    identifiers (128 distinct values observed across ~8k objects, most
    seen only a handful of times). Training the RSPN on that raw label
    space made grounding a single query take upwards of ten seconds,
    since grounding deep-copies every leaf of the categorical domain.
    Collapsing rare types into ObjectType.OTHER keeps the signal for
    common categories while cutting that domain -- and therefore
    grounding cost -- down sharply.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            _typed_object(ObjectType.CUP, "cup_1"),
            _typed_object(ObjectType.CUP, "cup_2"),
            _typed_object(ObjectType.PLANT, "plant_1"),
            _typed_object(ObjectType.CHAIR, "chair_1"),
        ],
    )

    result = _coarsen_rare_object_types([layer], keep_count=1)

    resulting_types = [obj.object_type for obj in result[0].objects]
    assert resulting_types == [
        ObjectType.CUP,
        ObjectType.CUP,
        ObjectType.OTHER,
        ObjectType.OTHER,
    ]
    assert [obj.id for obj in result[0].objects] == ["cup_1", "cup_2", "plant_1", "chair_1"]


def test_coarsen_rare_object_types_leaves_layer_within_keep_count_unchanged() -> None:
    """
    When every observed type already fits within keep_count, no object's type
    must be touched -- coarsening must not fall back to ObjectType.OTHER for
    types that were never actually rare.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[_typed_object(ObjectType.CUP, "cup_1"), _typed_object(ObjectType.PLANT, "plant_1")],
    )

    result = _coarsen_rare_object_types([layer], keep_count=2)

    assert [obj.object_type for obj in result[0].objects] == [ObjectType.CUP, ObjectType.PLANT]


# ---------------------------------------------------------------------------
# Layer scale fix – EGShelf.create_in_world must use per-layer scale
# ---------------------------------------------------------------------------


def test_each_layer_slab_uses_its_own_scale() -> None:
    """
    Each ShelfLayer slab must be created with the dimensions of its own
    EGShelfLayer.scale, not the maximum scale across all layers.

    Before the fix, create_in_world computed layer_scale = max(...) once outside
    the loop and applied it to every slab, causing narrower-scale layers to be
    rendered wider than the RSPN's spatial context for them.
    """
    narrow = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[],
    )
    wide = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.6, width=0.8),
        objects=[],
    )
    shelf = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=0.6, width=0.8),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[narrow, wide],
        book_source_ids=None,
    )
    world = shelf.create_in_world()
    slab_annotations = world.get_semantic_annotations_by_type(ShelfLayer)
    slab_face_widths = sorted(
        ann.root.collision.shapes[0].scale.y for ann in slab_annotations
    )
    assert slab_face_widths[0] == pytest.approx(0.4)
    assert slab_face_widths[1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Fixed-scale layer query – conditioning on EGSize during RSPN sampling
# ---------------------------------------------------------------------------


def test_build_layer_query_with_fixed_scale_conditions_scale() -> None:
    """
    build_layer_query_with_fixed_scale must register the target scale's width
    and length as conditioning assignments so the RSPN draws positions that are
    appropriate for that specific scale.
    """
    target_scale = EGScale(width=0.5, length=0.3, height=0.02)
    query = build_layer_query_with_fixed_scale(2, target_scale)
    params = UnderspecifiedParameters(query)
    conditioned_names = {variable.name for variable in params.conditioning_assignments_from_literal_values}
    assert any("scale.width" in name for name in conditioned_names)
    assert any("scale.length" in name for name in conditioned_names)


def test_build_free_layer_query_does_not_condition_scale() -> None:
    """
    build_free_layer_query must leave scale as a free variable so the RSPN
    samples scale from its marginal — the reference layer for the fixed-scale
    workflow is obtained this way.
    """
    from experiments.scene_generation_experiments.collision_resolution import (
        build_free_layer_query,
    )

    query = build_free_layer_query(2)
    params = UnderspecifiedParameters(query)
    conditioned_names = {variable.name for variable in params.conditioning_assignments_from_literal_values}
    assert not any("scale.width" in name for name in conditioned_names)
    assert not any("scale.length" in name for name in conditioned_names)


# ---------------------------------------------------------------------------
# Mesh rescaling – rendered geometry must match the declared EGSize
# ---------------------------------------------------------------------------


def test_object_mesh_is_rescaled_to_match_declared_egsize(tmp_path: Path) -> None:
    """
    EGObject2D.create_in_world must rescale the loaded mesh so its rendered
    bounding box matches the object's declared EGSize -- the same scale used by
    the collision-resolution box proxy -- instead of rendering the mesh at its
    native PLY size.

    Before the fix, meshes were loaded at native size regardless of
    EGSize, so a sampled scale had no effect on the actual
    rendered/collision geometry, letting mismatched mesh assets overlap
    despite collision resolution having judged the layout collision-
    free.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )

    target_scale = EGScale(width=0.2, length=0.3, height=0.4)
    obj = EGObject2D(
        id="obj_1",
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.CHAIR,
        scale=target_scale,
        position=EGPoint2D(x=0.0, y=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )

    world = World()
    root = Body(name=PrefixedName(name="root"))
    with world.modify_world():
        world.add_body(root)

    body = obj.create_in_world(world, tmp_path, parent=root)

    rendered_extents = body.collision.shapes[0].mesh.extents
    assert rendered_extents[0] == pytest.approx(target_scale.width, rel=1e-3)
    assert rendered_extents[1] == pytest.approx(target_scale.length, rel=1e-3)
    assert rendered_extents[2] == pytest.approx(target_scale.height, rel=1e-3)


# ---------------------------------------------------------------------------
# Mesh selection – pick the candidate closest in size to the sampled scale
# ---------------------------------------------------------------------------


def _write_box_ply(directory: Path, name: str, extents: tuple[float, float, float]) -> None:
    objects_dir = directory / "objects"
    objects_dir.mkdir(exist_ok=True)
    trimesh.creation.box(extents=extents).export(objects_dir / f"{name}.ply")


def test_mesh_size_matcher_selects_closest_native_bounding_box(tmp_path: Path) -> None:
    """
    _MeshSizeMatcher.closest_match must pick the candidate whose native mesh
    bounding box is nearest to the target EGSize.

    ObjectType labels in the source dataset are effectively per-instance
    identifiers (tens of thousands of distinct values), so matching by
    declared size -- rather than by label or picking uniformly at random
    -- is what keeps an assigned mesh visually plausible for the scale
    an object was sampled at.
    """
    _write_box_ply(tmp_path, "small", (0.1, 0.1, 0.1))
    _write_box_ply(tmp_path, "large", (1.0, 1.0, 1.0))
    matcher = _MeshSizeMatcher(candidates=[(tmp_path, "small"), (tmp_path, "large")])

    _, source_id = matcher.closest_match(EGScale(width=0.12, length=0.11, height=0.09))
    assert source_id == "small"

    _, source_id = matcher.closest_match(EGScale(width=0.9, length=1.1, height=1.05))
    assert source_id == "large"


def test_mesh_size_matcher_skips_mesh_loading_with_a_single_candidate(
    tmp_path: Path,
) -> None:
    """
    With only one candidate there is nothing to compare it against, so
    closest_match must return it directly without reading its mesh file.

    This also keeps callers that pass a placeholder/non-existent path as
    their sole candidate (e.g. tests exercising unrelated behaviour)
    working without needing a real mesh on disk.
    """
    matcher = _MeshSizeMatcher(candidates=[(Path("/nonexistent"), "missing")])
    scene_dir, source_id = matcher.closest_match(EGScale(width=0.2, length=0.2, height=0.2))
    assert (scene_dir, source_id) == (Path("/nonexistent"), "missing")
