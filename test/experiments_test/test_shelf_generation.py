from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
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
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    load_objects_with_cached_meshes,
    load_shelf_layers,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_layer_query,
    probabilistic_backend,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_mesh_candidate_types,
    _coarsen_rare_object_types,
)
from experiments.scene_generation_experiments.rspn_model_storage import (
    TrainedArbitraryShelfModel,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
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

_FAKE_PATH = Path("/fake/scene")
_SHELF_ID = "room_1_shelf_1"


@dataclass
class _MockShelfObject:
    """
    Duck-type substitute for EGObjectDAO in source-ID filtering tests.
    """

    object_type: ObjectType
    source_id: str
    scale: EGScale = field(
        default_factory=lambda: EGScale(width=0.1, length=0.1, height=0.1)
    )
    place_id: str = "floor"


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
    The default (``ObjectType.BOOK``) filter must include books and exclude cups and
    shelf-furniture objects.
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
    Passing ``object_type=None`` must include every type present in the input, subject
    only to source_id availability.
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


def test_objects_resting_on_furniture_are_kept(
    source_path_map: dict[str, Path],
) -> None:
    """
    The pool must contain objects resting on furniture, since shelf and table contents
    are exactly the objects a shelf demo needs meshes for.
    """
    objects = [
        _MockShelfObject(
            object_type=ObjectType.BOOK, source_id="book_src", place_id=_SHELF_ID
        )
    ]
    with patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value=source_path_map,
    ):
        result = _get_source_ids_for_objects(objects)
    assert [candidate.source_id for candidate in result] == ["book_src"]


def test_missing_source_id_is_excluded(source_path_map: dict[str, Path]) -> None:
    """
    Objects whose source_id has no corresponding PLY path must be silently dropped
    regardless of the object-type filter.
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
    With no book meshes cached locally, a downloader must be used to fetch scenes for
    distinct book source_ids until minimum_candidates is reached.
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
    A downloader must not be consulted at all when enough matching meshes are already
    cached locally.
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
    A source_id the Sage-10k database has no record of must be skipped rather than
    aborting the whole pool -- objects can come from a different data source than the
    one the downloader looks scenes up in.
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
    Without a downloader the candidate pool must be built from the local cache alone,
    never entering the download path -- this is what keeps the demos fast for iterative
    testing.
    """
    books = [_MockShelfObject(object_type=ObjectType.BOOK, source_id="book_src")]

    with (
        patch(
            "experiments.scene_generation_experiments.utils.build_source_id_to_path",
            return_value=source_path_map,
        ),
        patch(
            "experiments.scene_generation_experiments.utils._ensure_minimum_mesh_pool"
        ) as ensure_minimum_mesh_pool,
    ):
        _get_source_ids_for_objects(books)

    ensure_minimum_mesh_pool.assert_not_called()


# ---------------------------------------------------------------------------
# load_shelf_layers – the one query path every training run takes
# ---------------------------------------------------------------------------


def test_loading_layers_does_not_scale_query_count_with_object_count(
    session: Session,
) -> None:
    """
    Loading the stored layers must not issue a statement per object for each of its
    scale, position and orientation relationships.

    Preprocessing exists to take that cost out of every training run, so leaving those
    to lazy loading would put it straight back on the read path.
    """
    session.add_all(
        to_dao(
            EGShelfLayer(
                scale=EGScale(height=0.02, length=0.3, width=0.4),
                objects=[
                    _typed_object(ObjectType.BOOK, f"book_{layer_index}_{index}")
                    for index in range(10)
                ],
            )
        )
        for layer_index in range(5)
    )
    session.commit()
    session.expire_all()

    statement_count = 0

    def _count_statement(*args, **kwargs) -> None:
        nonlocal statement_count
        statement_count += 1

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", _count_statement)
    try:
        layers = load_shelf_layers(session)
    finally:
        event.remove(engine, "before_cursor_execute", _count_statement)

    assert sum(len(layer.objects) for layer in layers) == 50
    assert statement_count <= 5


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
    Object types outside the keep_count most frequent ones must be replaced with
    ObjectType.OTHER; every other field must be preserved unchanged.

    The sage10k dataset's object_type labels are close to per-instance identifiers (128
    distinct values observed across ~8k objects, most seen only a handful of times).
    Training the RSPN on that raw label space made grounding a single query take upwards
    of ten seconds, since grounding deep-copies every leaf of the categorical domain.
    Collapsing rare types into ObjectType.OTHER keeps the signal for common categories
    while cutting that domain -- and therefore grounding cost -- down sharply.
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
    When every observed type already fits within keep_count, no object's type must be
    touched -- coarsening must not fall back to ObjectType.OTHER for types that were
    never actually rare.
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
    _coarsen_mesh_candidate_types must relabel every candidate whose type falls outside
    frequent_types as ObjectType.OTHER, mirroring _coarsen_rare_object_types.

    Without this, a sampled ObjectType.OTHER object could never find a same-type mesh
    candidate in _MeshTypeMatcher.random_match, since every candidate would still carry
    its original, uncoarsened type -- silently falling back to a random mesh from the
    whole pool for every object outside the most frequent types.
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
    Candidates whose type is already within frequent_types must not be touched.
    """
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    plant_candidate = MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.PLANT)

    result = _coarsen_mesh_candidate_types(
        [cup_candidate, plant_candidate],
        frequent_types={ObjectType.CUP, ObjectType.PLANT},
    )

    assert result == [cup_candidate, plant_candidate]


# ---------------------------------------------------------------------------
# TrainedArbitraryShelfModel – exporting and reloading a fitted RSPN
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_arbitrary_shelf_model() -> TrainedArbitraryShelfModel:
    layers = [
        EGShelfLayer(
            scale=EGScale(height=0.02, length=0.3, width=0.4),
            objects=[
                _typed_object(ObjectType.CUP, f"cup_{index}"),
                _typed_object(ObjectType.PLANT, f"plant_{index}"),
            ],
        )
        for index in range(5)
    ]
    rspn = RelationalProbabilisticCircuit(EGShelfLayer, min_samples_per_leaf=0.5).fit(
        [to_dao(layer) for layer in layers]
    )
    return TrainedArbitraryShelfModel(
        relational_probabilistic_circuit=rspn,
        frequent_object_types={ObjectType.CUP, ObjectType.PLANT},
    )


def test_save_writes_a_loadable_file_and_creates_parent_directories(
    fitted_arbitrary_shelf_model: TrainedArbitraryShelfModel, tmp_path: Path
) -> None:
    export_path = tmp_path / "nested" / "arbitrary_shelf_rspn.json"

    fitted_arbitrary_shelf_model.save(export_path)

    assert export_path.is_file()


def test_load_restores_the_frequent_object_types(
    fitted_arbitrary_shelf_model: TrainedArbitraryShelfModel, tmp_path: Path
) -> None:
    export_path = tmp_path / "arbitrary_shelf_rspn.json"
    fitted_arbitrary_shelf_model.save(export_path)

    restored = TrainedArbitraryShelfModel.load(export_path)

    assert restored.frequent_object_types == {ObjectType.CUP, ObjectType.PLANT}


def test_load_restores_a_circuit_that_can_still_be_grounded_and_sampled(
    fitted_arbitrary_shelf_model: TrainedArbitraryShelfModel, tmp_path: Path
) -> None:
    """
    A restored circuit must still answer queries through the same ProbabilisticBackend
    path :func:`generate_shelf_with_arbitrary_objects` uses, not just round-trip its
    structure.
    """
    export_path = tmp_path / "arbitrary_shelf_rspn.json"
    fitted_arbitrary_shelf_model.save(export_path)

    restored = TrainedArbitraryShelfModel.load(export_path)
    backend = probabilistic_backend(restored.relational_probabilistic_circuit)

    sample = next(iter(backend.evaluate(build_layer_query(free_count=2))))

    assert len(sample.objects) == 2


_SAVE_SCRIPT = """
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import RelationalProbabilisticCircuit
from experiments.orm.ormatic_interface import *  # noqa: F401,F403  registers ORM mappers
from experiments.scene_generation_experiments.shelf_generation import TrainedArbitraryShelfModel
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D, EGPoint2D, EGRotation, EGScale, EGShelfLayer, ObjectType,
)
from pathlib import Path
import sys

def typed_object(object_type, object_id):
    return EGObject2D(
        id=object_id, room_id="room_1", place_id="shelf_1", object_type=object_type,
        scale=EGScale(height=0.1, length=0.1, width=0.1),
        position=EGPoint2D(x=0.0, y=0.0), orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id=object_id,
    )

types = [ObjectType.CUP, ObjectType.PLANT, ObjectType.BOOK, ObjectType.SHELF, ObjectType.CHAIR]
layers = [
    EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[typed_object(t, f"{t.value}_{i}") for t in types],
    )
    for i in range(10)
]
rspn = RelationalProbabilisticCircuit(EGShelfLayer, min_samples_per_leaf=0.5).fit(
    [to_dao(layer) for layer in layers]
)
TrainedArbitraryShelfModel(
    relational_probabilistic_circuit=rspn, frequent_object_types=set(types)
).save(Path(sys.argv[1]))
"""

_LOAD_SCRIPT = """
from experiments.orm.ormatic_interface import *  # noqa: F401,F403  registers ORM mappers
from experiments.scene_generation_experiments.shelf_generation import TrainedArbitraryShelfModel
from experiments.scene_generation_experiments.rspn_sampling import build_layer_query, probabilistic_backend
from pathlib import Path
import sys

model = TrainedArbitraryShelfModel.load(Path(sys.argv[1]))
backend = probabilistic_backend(model.relational_probabilistic_circuit)
sample = next(iter(backend.evaluate(build_layer_query(free_count=2))))
assert len(sample.objects) == 2
print("GROUNDED_OK")
"""


def test_load_survives_a_different_hash_seed_process(tmp_path: Path) -> None:
    """
    A model exported by one process must still ground and sample correctly when loaded
    by a different process with a different PYTHONHASHSEED.

    Python randomizes hash() for str-backed types -- including the StrEnum
    ObjectType -- independently per process, so fitting and loading in the
    same process (as the other tests in this module do) cannot expose a
    regression here: only two genuinely separate processes with different
    seeds can.
    """
    export_path = tmp_path / "arbitrary_shelf_rspn.json"

    subprocess.run(
        [sys.executable, "-c", _SAVE_SCRIPT, str(export_path)],
        env={**os.environ, "PYTHONHASHSEED": "1"},
        check=True,
    )
    result = subprocess.run(
        [sys.executable, "-c", _LOAD_SCRIPT, str(export_path)],
        env={**os.environ, "PYTHONHASHSEED": "2"},
        check=True,
        capture_output=True,
        text=True,
    )

    assert "GROUNDED_OK" in result.stdout


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
# Mesh rescaling – rendered geometry must match the declared EGSize
# ---------------------------------------------------------------------------


def test_object_mesh_keeps_its_native_size(tmp_path: Path) -> None:
    """
    EGObject2D.create_in_world must render the loaded mesh at its own native PLY size,
    not stretch it to the object's declared EGSize.

    sage10k meshes already carry their real-world size, so rescaling a randomly matched
    mesh to an independently sampled scale distorts its proportions. The declared scale
    therefore must not drive the rendered geometry.
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
    _MeshTypeMatcher.random_match must only return candidates whose object_type equals
    the requested type when at least one such candidate exists in the pool.

    ObjectType labels in the source dataset are effectively per-instance identifiers
    (tens of thousands of distinct values), so picking a mesh at random from the same
    generalized ObjectType -- rather than matching by declared size -- is what keeps an
    assigned mesh semantically plausible for the category an object was sampled as.
    """
    book_candidate = MeshCandidate(_FAKE_PATH, "book_src", ObjectType.BOOK)
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    matcher = _MeshTypeMatcher(candidates=[book_candidate, cup_candidate])

    results = {matcher.random_match(ObjectType.BOOK) for _ in range(30)}
    assert results == {book_candidate}


def test_mesh_type_matcher_returns_nothing_when_the_type_is_absent() -> None:
    """
    When the pool holds no candidate of the requested type, random_match must return
    ``None`` so the caller can drop the piece.

    Replaces two earlier tests that asserted the opposite -- that a candidate
    was returned from the full pool regardless of type, so sampling could never
    fail outright. That fallback is precisely what strewed generated rooms with
    arbitrary objects: the mesh cache covers only a few dozen of the hundred-odd
    object types, so a sampled sofa or bed routinely spawned as whatever was
    drawn, commonly a book or a piece of wall art. Dropping the piece is honest
    and is now counted in :class:`RoomGenerationReport`.
    """
    cup_candidate = MeshCandidate(_FAKE_PATH, "cup_src", ObjectType.CUP)
    plant_candidate = MeshCandidate(_FAKE_PATH, "plant_src", ObjectType.PLANT)
    matcher = _MeshTypeMatcher(candidates=[cup_candidate, plant_candidate])

    assert matcher.random_match(ObjectType.BOOK) is None


def test_mesh_type_matcher_excludes_candidates_larger_than_the_budget() -> None:
    """
    With a size budget, only candidates whose own real-world size fits are eligible, so
    an oversized mesh is never chosen when a fitting one exists.
    """
    fitting = MeshCandidate(_FAKE_PATH, "small", ObjectType.BOOK, (0.1, 0.1, 0.1))
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (0.1, 0.1, 1.0))
    matcher = _MeshTypeMatcher(candidates=[fitting, oversized])
    budget = EGScale(width=0.5, length=0.5, height=0.5)

    results = {
        matcher.random_match(ObjectType.BOOK, max_extents=budget) for _ in range(30)
    }
    assert results == {fitting}


def test_mesh_type_matcher_drops_when_no_candidate_of_type_fits() -> None:
    """
    When every candidate of the requested type is too big for the budget, random_match
    returns None so the caller can leave the object out rather than force an overflowing
    mesh into the space.
    """
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (0.1, 0.1, 1.0))
    matcher = _MeshTypeMatcher(candidates=[oversized])
    budget = EGScale(width=0.5, length=0.5, height=0.5)

    assert matcher.random_match(ObjectType.BOOK, max_extents=budget) is None


def test_mesh_type_matcher_ignores_size_without_a_budget() -> None:
    """
    Without a budget, size is not considered, so callers that do not constrain space
    (chairs, floor objects) keep the original type-only behaviour.
    """
    oversized = MeshCandidate(_FAKE_PATH, "big", ObjectType.BOOK, (1.0, 1.0, 1.0))
    matcher = _MeshTypeMatcher(candidates=[oversized])

    assert matcher.random_match(ObjectType.BOOK) is oversized


def test_mesh_type_matcher_treats_unknown_size_as_fitting() -> None:
    """
    A candidate whose native size is unknown must be treated as fitting, so manually
    built pools without size information are not silently emptied.
    """
    unknown = MeshCandidate(_FAKE_PATH, "unknown", ObjectType.BOOK)
    matcher = _MeshTypeMatcher(candidates=[unknown])
    budget = EGScale(width=0.01, length=0.01, height=0.01)

    assert matcher.random_match(ObjectType.BOOK, max_extents=budget) is unknown


def test_mesh_pool_loads_every_object_whose_mesh_is_cached(session: Session) -> None:
    """
    The mesh-candidate pool must be selected by mesh availability, not by an arbitrary
    row cap.

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
                position_is_mesh_corrected=True,
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
                position_is_mesh_corrected=True,
            )
        ]
    )
    session.commit()

    loaded = load_objects_with_cached_meshes(session, cached_source_ids)

    assert {obj.source_id for obj in loaded} == cached_source_ids
