from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

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
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    load_objects_with_cached_meshes,
    objects_of_type,
)
from experiments.scene_generation_experiments.collision_resolution import (
    build_layer_query_with_fixed_scale,
    build_pose_resample_query,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_mesh_candidate_types,
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
    MeshCandidate,
    ObjectType,
    _MeshTypeMatcher,
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
    scale: EGScale = field(default_factory=lambda: EGScale(width=0.1, length=0.1, height=0.1))


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
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(mixed_mock_objects)
    source_ids = {candidate.source_id for candidate in result}
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
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(mixed_mock_objects, object_type=None)
    source_ids = {candidate.source_id for candidate in result}
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
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(objects_without_path, object_type=None)
    assert result == []


def test_downloader_fills_pool_up_to_minimum_candidates() -> None:
    """
    With no book meshes cached locally, a downloader must be used to fetch
    scenes for distinct book source_ids until minimum_candidates is reached.
    """
    books = [
        _MockShelfObject(object_type=ObjectType.BOOK, source_id=f"book_{i}")
        for i in range(5)
    ]
    downloader = MagicMock()
    downloader.download_scene_for_source_id.side_effect = lambda source_id: (
        _FAKE_PATH / source_id
    )

    with patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value={},
    ):
        result = _get_source_ids_for_objects(
            books, downloader=downloader, minimum_candidates=3
        )

    assert len(result) == 3
    assert downloader.download_scene_for_source_id.call_count == 3


def test_downloader_is_not_used_once_the_pool_already_meets_the_minimum(
    source_path_map: dict[str, Path],
) -> None:
    """
    A downloader must not be consulted at all when enough matching meshes are
    already cached locally.
    """
    books = [_MockShelfObject(object_type=ObjectType.BOOK, source_id="book_src")]
    downloader = MagicMock()

    with patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ):
        _get_source_ids_for_objects(books, downloader=downloader, minimum_candidates=1)

    downloader.download_scene_for_source_id.assert_not_called()


def test_downloader_skips_source_ids_the_sage10k_database_does_not_know() -> None:
    """
    A source_id the Sage-10k database has no record of must be skipped rather
    than aborting the whole pool -- objects can come from a different data
    source than the one the downloader looks scenes up in.
    """
    from experiments.scene_generation_experiments.data_preprocessing import (
        SourceIdNotFoundError,
    )

    books = [
        _MockShelfObject(object_type=ObjectType.BOOK, source_id="unknown_book"),
        _MockShelfObject(object_type=ObjectType.BOOK, source_id="known_book"),
    ]
    downloader = MagicMock()

    def _download(source_id: str) -> Path:
        if source_id == "unknown_book":
            raise SourceIdNotFoundError(source_id)
        return _FAKE_PATH / source_id

    downloader.download_scene_for_source_id.side_effect = _download

    with patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value={},
    ):
        result = _get_source_ids_for_objects(
            books, downloader=downloader, minimum_candidates=5
        )

    assert [candidate.source_id for candidate in result] == ["known_book"]


def test_no_downloader_never_attempts_a_mesh_pool_download(
    source_path_map: dict[str, Path],
) -> None:
    """
    Without a downloader the candidate pool must be built from the local cache
    alone, never entering the download path -- this is what keeps the demos fast
    for iterative testing.
    """
    books = [_MockShelfObject(object_type=ObjectType.BOOK, source_id="book_src")]

    with patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ), patch(
        "experiments.scene_generation_experiments.utils._ensure_minimum_mesh_pool"
    ) as ensure_minimum_mesh_pool:
        _get_source_ids_for_objects(books)

    ensure_minimum_mesh_pool.assert_not_called()


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
    assert [obj.id for obj in result[0].objects] == [
        "cup_1",
        "cup_2",
        "plant_1",
        "chair_1",
    ]


def test_coarsen_rare_object_types_leaves_layer_within_keep_count_unchanged() -> None:
    """
    When every observed type already fits within keep_count, no object's type
    must be touched -- coarsening must not fall back to ObjectType.OTHER for
    types that were never actually rare.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            _typed_object(ObjectType.CUP, "cup_1"),
            _typed_object(ObjectType.PLANT, "plant_1"),
        ],
    )

    result = _coarsen_rare_object_types([layer], keep_count=2)

    assert [obj.object_type for obj in result[0].objects] == [
        ObjectType.CUP,
        ObjectType.PLANT,
    ]


def test_coarsen_mesh_candidate_types_relabels_candidates_outside_frequent_types() -> (
    None
):
    """
    _coarsen_mesh_candidate_types must relabel every candidate whose type
    falls outside frequent_types as ObjectType.OTHER, mirroring
    _coarsen_rare_object_types.

    Without this, a sampled ObjectType.OTHER object could never find a
    same-type mesh candidate in _MeshTypeMatcher.random_match, since every
    candidate would still carry its original, uncoarsened type -- silently
    falling back to a random mesh from the whole pool for every object
    outside the most frequent types.
    """
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    plant_candidate = MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.PLANT)

    result = _coarsen_mesh_candidate_types(
        [cup_candidate, plant_candidate], frequent_types={ObjectType.CUP}
    )

    assert result[0] == cup_candidate
    assert result[1] == MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.OTHER)


def test_coarsen_mesh_candidate_types_leaves_frequent_types_unchanged() -> None:
    """
    Candidates whose type is already within frequent_types must not be
    touched.
    """
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    plant_candidate = MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.PLANT)

    result = _coarsen_mesh_candidate_types(
        [cup_candidate, plant_candidate],
        frequent_types={ObjectType.CUP, ObjectType.PLANT},
    )

    assert result == [cup_candidate, plant_candidate]


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
        source_ids=None,
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
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
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
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert not any("scale.width" in name for name in conditioned_names)
    assert not any("scale.length" in name for name in conditioned_names)


def test_build_pose_resample_query_frees_resampled_scale_and_pose() -> None:
    """
    build_pose_resample_query must condition only the fixed objects' scale and
    pose, leaving the resampled object's scale, position, and orientation all
    free to be redrawn.

    Conditioning a resampled slot on its own scale pins the query to the
    single training example that combination of evidence (its own scale plus
    every fixed neighbour's exact pose) came from, collapsing the RSPN's
    posterior for that slot's position back to its original, still-colliding
    value -- observed as a repair pass that redraws the exact same pose every
    time and so can never actually resolve a collision. Regression test for
    that collapse.
    """
    query = build_pose_resample_query(
        [_typed_object(ObjectType.BOOK, "fixed")],
        len([_typed_object(ObjectType.BOOK, "resampled")]),
        EGScale(width=0.5, length=0.3, height=0.02),
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    conditioned_positions = [name for name in conditioned_names if "position.x" in name]
    # "objects[" scopes to per-object scale, excluding the layer's own
    # (always-fixed) EGShelfLayer.scale.width.
    conditioned_scales = [
        name
        for name in conditioned_names
        if "objects[" in name and "scale.width" in name
    ]
    # Only the one fixed object's position and scale are conditioned; the
    # resampled one's are left entirely free.
    assert len(conditioned_positions) == 1
    assert len(conditioned_scales) == 1


# ---------------------------------------------------------------------------
# Mesh rescaling – rendered geometry must match the declared EGSize
# ---------------------------------------------------------------------------


def test_object_mesh_keeps_its_native_size(tmp_path: Path) -> None:
    """
    EGObject2D.create_in_world must render the loaded mesh at its own native PLY
    size, not stretch it to the object's declared EGSize.

    sage10k meshes already carry their real-world size, so rescaling a randomly
    matched mesh to an independently sampled scale distorts its proportions. The
    declared scale therefore must not drive the rendered geometry.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )

    native_extents = trimesh.load(
        str(objects_dir / "test_object.ply"), process=False
    ).extents
    obj = EGObject2D(
        id="obj_1",
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.CHAIR,
        scale=EGScale(width=0.2, length=0.3, height=0.4),
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
    assert rendered_extents == pytest.approx(native_extents, abs=1e-3)


# ---------------------------------------------------------------------------
# Mesh selection – pick a random mesh whose object shares the sampled type
# ---------------------------------------------------------------------------


def test_mesh_type_matcher_only_returns_candidates_of_the_requested_type() -> None:
    """
    _MeshTypeMatcher.random_match must only return candidates whose
    object_type equals the requested type when at least one such candidate
    exists in the pool.

    ObjectType labels in the source dataset are effectively per-instance
    identifiers (tens of thousands of distinct values), so picking a mesh at
    random from the same generalized ObjectType -- rather than matching by
    declared size -- is what keeps an assigned mesh semantically plausible
    for the category an object was sampled as.
    """
    book_candidate = MeshCandidate(_FAKE_PATH, "book_src", ObjectType.BOOK)
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    matcher = _MeshTypeMatcher(candidates=[book_candidate, cup_candidate])

    results = {matcher.random_match(ObjectType.BOOK) for _ in range(30)}
    assert results == {book_candidate}


def test_mesh_type_matcher_falls_back_to_full_pool_when_type_absent() -> None:
    """
    When the pool holds no candidate of the requested type, random_match must
    still return a candidate from the full pool instead of raising, so
    sampling can never fail outright.
    """
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    plant_candidate = MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.PLANT)
    matcher = _MeshTypeMatcher(candidates=[cup_candidate, plant_candidate])

    result = matcher.random_match(ObjectType.BOOK)
    assert result in {cup_candidate, plant_candidate}


def test_mesh_type_matcher_returns_the_only_candidate_regardless_of_type() -> None:
    """
    With only one candidate in the pool, random_match must return it
    regardless of whether its type matches the request.
    """
    only_candidate = MeshCandidate(_FAKE_PATH, "book_src", ObjectType.BOOK)
    matcher = _MeshTypeMatcher(candidates=[only_candidate])

    assert matcher.random_match(ObjectType.CUP) == only_candidate


def test_mesh_type_matcher_excludes_candidates_larger_than_the_budget() -> None:
    """
    With a size budget, only candidates whose own real-world size fits are
    eligible, so an oversized mesh is never chosen when a fitting one exists.
    """
    fitting = MeshCandidate(_FAKE_PATH, "small", ObjectType.BOOK, (0.1, 0.1, 0.1))
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (0.1, 0.1, 1.0))
    matcher = _MeshTypeMatcher(candidates=[fitting, oversized])
    budget = EGScale(width=0.5, length=0.5, height=0.5)

    results = {matcher.random_match(ObjectType.BOOK, max_extents=budget) for _ in range(30)}
    assert results == {fitting}


def test_mesh_type_matcher_drops_when_no_candidate_of_type_fits() -> None:
    """
    When every candidate of the requested type is too big for the budget,
    random_match returns None so the caller can leave the object out rather than
    force an overflowing mesh into the space.
    """
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (0.1, 0.1, 1.0))
    matcher = _MeshTypeMatcher(candidates=[oversized])
    budget = EGScale(width=0.5, length=0.5, height=0.5)

    assert matcher.random_match(ObjectType.BOOK, max_extents=budget) is None


def test_mesh_type_matcher_ignores_size_without_a_budget() -> None:
    """
    Without a budget, size is not considered, so callers that do not constrain
    space (chairs, floor objects) keep the original type-only behaviour.
    """
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (1.0, 1.0, 1.0))
    matcher = _MeshTypeMatcher(candidates=[oversized])

    assert matcher.random_match(ObjectType.BOOK) is oversized


def test_mesh_type_matcher_treats_unknown_size_as_fitting() -> None:
    """
    A candidate whose native size is unknown must be treated as fitting, so
    manually built pools without size information are not silently emptied.
    """
    unknown = MeshCandidate(_FAKE_PATH, "unknown", ObjectType.BOOK)
    matcher = _MeshTypeMatcher(candidates=[unknown])
    budget = EGScale(width=0.01, length=0.01, height=0.01)

    assert matcher.random_match(ObjectType.BOOK, max_extents=budget) is unknown


# ---------------------------------------------------------------------------
# Shelf-frame normalisation – contents must be stored in the shelf's own frame
# ---------------------------------------------------------------------------


def test_shelf_content_orientation_is_stored_relative_to_the_shelf(
    session: Session,
) -> None:
    """
    A shelf's contents are consumed at spawn time inside a corpus rotated into
    the shelf's content frame, so extraction must store each object's yaw
    relative to that frame rather than the absolute one -- otherwise the
    rotation is double-counted for the 88% of shelves in the dataset that are
    rotated.

    The content frame is the shelf yaw plus
    :attr:`EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES`, so a book at absolute yaw
    110 on a shelf at yaw 90 is stored at ``110 - (90 + 90) = -70``.

    Guards the orientation half of the frame change; the position half is
    guarded behaviourally by
    :func:`test_shelf_contents_round_trip_from_world_pose_through_extraction_and_spawning`,
    since the stored coordinate convention is an internal detail.
    """
    shelf = EGObjectDAO(
        id=_SHELF_ID,
        room_id="room_1",
        place_id="floor",
        source_id="shelf_src",
        object_type=ObjectType.SHELF,
        scale=EGScaleDAO(height=2.0, length=1.0, width=1.0),
        position=EGPositionDAO(x=10.0, y=5.0, z=1.0),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=90.0),
    )
    book = EGObjectDAO(
        id="book_1",
        room_id="room_1",
        place_id=_SHELF_ID,
        source_id="book_src",
        object_type=ObjectType.BOOK,
        scale=EGScaleDAO(height=0.3, length=0.1, width=0.05),
        position=EGPositionDAO(x=10.3, y=5.0, z=0.5),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=110.0),
    )
    session.add_all([shelf, book])
    session.commit()

    layers, _ = _extract_shelf_layers_from_place_id(session)

    [stored_book] = layers[0].objects
    assert stored_book.orientation.z == pytest.approx(-70.0)


def test_shelf_contents_round_trip_from_world_pose_through_extraction_and_spawning(
    session: Session, tmp_path: Path
) -> None:
    """
    Extraction and spawning must be inverses: an object at a known world pose on
    a rotated shelf has to come back to that same world pose once its extracted,
    shelf-local pose is spawned again.

    Storing the pose in the shelf's frame is only half the contract -- the spawn
    side re-applies the shelf's rotation and maps the stored axes onto the corpus
    frame, so only a full round-trip proves the two agree.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "book_src.ply")
    shutil.copy(resources_root / "chair_texture.png", objects_dir / "book_src_texture.png")

    shelf_world_x, shelf_world_y, shelf_yaw = 10.0, 5.0, 90.0
    book_world_x, book_world_y = 10.3, 5.2
    shelf = EGObjectDAO(
        id=_SHELF_ID,
        room_id="room_1",
        place_id="floor",
        source_id="shelf_src",
        object_type=ObjectType.SHELF,
        scale=EGScaleDAO(height=2.0, length=1.0, width=1.0),
        position=EGPositionDAO(x=shelf_world_x, y=shelf_world_y, z=1.0),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=shelf_yaw),
    )
    book = EGObjectDAO(
        id="book_1",
        room_id="room_1",
        place_id=_SHELF_ID,
        source_id="book_src",
        object_type=ObjectType.BOOK,
        scale=EGScaleDAO(height=0.2, length=0.1, width=0.05),
        position=EGPositionDAO(x=book_world_x, y=book_world_y, z=0.5),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=110.0),
    )
    session.add_all([shelf, book])
    session.commit()

    layers, _ = _extract_shelf_layers_from_place_id(session)
    spawned_shelf = EGShelf(
        position=EGPoint2D(x=shelf_world_x, y=shelf_world_y),
        scale=EGScale(height=2.0, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=shelf_yaw),
        layers=layers,
        source_ids=[
            MeshCandidate(
                scene_dir=tmp_path, source_id="book_src", object_type=ObjectType.BOOK
            )
        ],
    )

    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    spawned = spawned_shelf.spawn_in_world(world, root)

    [body] = spawned.layers[0].object_bodies.values()
    position = body.global_pose.to_position().to_np()
    assert position[0] == pytest.approx(book_world_x, abs=1e-6)
    assert position[1] == pytest.approx(book_world_y, abs=1e-6)


def test_shelf_contents_spawn_within_the_layer_footprint(
    session: Session, tmp_path: Path
) -> None:
    """
    A content object offset along the shelf's wide face must spawn inside the
    corpus footprint on both axes, rather than overflowing the shallow depth.

    The shelf here is much wider (face) than it is deep, and the book is offset
    well along that face. If the face offset were mapped onto the corpus depth
    axis -- the original defect -- the book would land far outside the shallow
    depth and protrude front and back.
    """
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    shutil.copy(resources_root / "chair.ply", objects_dir / "book_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "book_src_texture.png"
    )

    shelf_depth, shelf_face = 0.3, 1.0
    shelf = EGObjectDAO(
        id=_SHELF_ID,
        room_id="room_1",
        place_id="floor",
        source_id="shelf_src",
        object_type=ObjectType.SHELF,
        scale=EGScaleDAO(height=2.0, length=shelf_depth, width=shelf_face),
        position=EGPositionDAO(x=0.0, y=0.0, z=1.0),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    # At the shelf's zero yaw its wide face lies along world x, so a world-x
    # offset is a face offset -- well within the face but far outside the depth.
    book = EGObjectDAO(
        id="book_1",
        room_id="room_1",
        place_id=_SHELF_ID,
        source_id="book_src",
        object_type=ObjectType.BOOK,
        scale=EGScaleDAO(height=0.2, length=0.1, width=0.05),
        position=EGPositionDAO(x=0.4, y=0.0, z=0.5),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )
    session.add_all([shelf, book])
    session.commit()

    layers, _ = _extract_shelf_layers_from_place_id(session)
    spawned_shelf = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=shelf_depth, width=shelf_face),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=layers,
        source_ids=[
            MeshCandidate(
                scene_dir=tmp_path, source_id="book_src", object_type=ObjectType.BOOK
            )
        ],
    )
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    spawned = spawned_shelf.spawn_in_world(world, root)

    [body] = spawned.layers[0].object_bodies.values()
    corpus_x, corpus_y = body.parent_connection.origin.to_position().to_np()[:2]
    assert abs(corpus_x) <= shelf_depth / 2
    assert abs(corpus_y) <= shelf_face / 2


def test_mesh_pool_loads_every_object_whose_mesh_is_cached(session: Session) -> None:
    """
    The mesh-candidate pool must be selected by mesh availability, not by an
    arbitrary row cap.

    Capping an unordered query and only then intersecting with the cached
    meshes made the pool an accident of which rows the database happened to
    return -- a handful of candidates dominated by whichever types earlier demos
    had downloaded, so most sampled object types found no mesh of their own kind
    and silently fell back to the whole pool.
    """
    cached_source_ids = {f"cached_{index}" for index in range(30)}
    session.add_all(
        [
            EGObjectDAO(
                id=f"object_{index}",
                room_id="room_1",
                place_id="floor",
                source_id=f"cached_{index}",
                object_type=ObjectType.PLANT,
                scale=EGScaleDAO(height=1.0, length=0.5, width=0.5),
                position=EGPositionDAO(x=float(index), y=0.0, z=0.5),
                orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
            )
            for index in range(30)
        ]
        + [
            EGObjectDAO(
                id="uncached_object",
                room_id="room_1",
                place_id="floor",
                source_id="not_downloaded",
                object_type=ObjectType.PLANT,
                scale=EGScaleDAO(height=1.0, length=0.5, width=0.5),
                position=EGPositionDAO(x=99.0, y=0.0, z=0.5),
                orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
            )
        ]
    )
    session.commit()

    loaded = load_objects_with_cached_meshes(session, cached_source_ids)

    assert {obj.source_id for obj in loaded} == cached_source_ids


def test_objects_of_type_selects_only_floor_resting_objects_of_that_type(
    session: Session,
) -> None:
    """
    Mesh coverage is targeted per object type, so the download worklist must
    contain exactly the floor-resting objects of the requested type -- objects
    standing on furniture are dressed from their own pools.
    """
    session.add_all(
        [
            EGObjectDAO(
                id="floor_plant",
                room_id="room_1",
                place_id="floor",
                source_id="plant_src",
                object_type=ObjectType.PLANT,
                scale=EGScaleDAO(height=1.0, length=0.5, width=0.5),
                position=EGPositionDAO(x=0.0, y=0.0, z=0.5),
                orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
            ),
            EGObjectDAO(
                id="shelf_plant",
                room_id="room_1",
                place_id=_SHELF_ID,
                source_id="shelf_plant_src",
                object_type=ObjectType.PLANT,
                scale=EGScaleDAO(height=0.2, length=0.1, width=0.1),
                position=EGPositionDAO(x=0.0, y=0.0, z=1.2),
                orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
            ),
            EGObjectDAO(
                id="floor_chair",
                room_id="room_1",
                place_id="floor",
                source_id="chair_src",
                object_type=ObjectType.CHAIR,
                scale=EGScaleDAO(height=0.9, length=0.5, width=0.5),
                position=EGPositionDAO(x=1.0, y=0.0, z=0.45),
                orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
            ),
        ]
    )
    session.commit()

    plants = objects_of_type(session, ObjectType.PLANT)

    assert [obj.source_id for obj in plants] == ["plant_src"]
