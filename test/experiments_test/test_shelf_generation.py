from __future__ import annotations

import os
import shutil
import subprocess
import sys
import dataclasses
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from sqlalchemy import event
from sqlalchemy.orm import Session
from visualization_msgs.msg import Marker, MarkerArray

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
    min_samples_per_leaf_for,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_layer_query,
    LayerObjectCountSampler,
    ShelfDimensionSampler,
    draw_shelf,
    ShelfDimensions,
    build_shelf_query,
    probabilistic_backend,
)
from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_mesh_candidate_types,
    _coarsen_rare_object_types,
    _rewrite_mesh_uris_for_foxglove,
)
from experiments.scene_generation_experiments.exceptions import (
    OutdatedTrainedModelError,
    UndrawableShelfError,
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
    ShelfType,
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
                shelf_type=ShelfType.BOOKCASE,
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
        shelf_type=ShelfType.BOOKCASE,
    )


@dataclass
class _MockVizMarkerPublisher:
    """
    Duck-type substitute for VizMarkerPublisher exposing only the ``markers`` attribute
    :func:`_rewrite_mesh_uris_for_foxglove` reads and mutates.
    """

    markers: MarkerArray


def test_foxglove_mesh_uri_rewrite_cancels_gltf_up_axis_convention(
    tmp_path: Path,
) -> None:
    """
    Foxglove's 3D panel documents that its "Mesh up axis" override does not apply to
    glTF/``.glb`` files, since they are assumed to already be authored Y-up -- unlike
    STL/OBJ, there is no toggle to disable this.

    Meshes are authored with Z-up (ROS/world) vertex data unmodified, so Foxglove's
    built-in Y-up-to-Z-up correction misinterprets them, rendering objects tipped onto
    their side. Every mesh marker rewritten for Foxglove must therefore be pre-rotated
    -90 degrees about X to cancel that correction. The rewrite must also convert the
    source OBJ to a self-contained ``.glb``, since Foxglove does not read the separate
    ``.mtl`` sidecar OBJ relies on for material/texture.
    """
    mesh_dir = tmp_path / "source_mesh_dir"
    mesh_dir.mkdir()
    mesh_file = mesh_dir / "object.obj"
    mesh_file.write_text("v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n")

    marker = Marker()
    marker.mesh_resource = f"file://{mesh_file}"
    original_orientation = Rotation.from_euler("z", 30, degrees=True).as_quat()
    (
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ) = original_orientation
    viz_marker = _MockVizMarkerPublisher(markers=MarkerArray(markers=[marker]))

    with patch(
        "experiments.scene_generation_experiments.shelf_generation."
        "get_package_share_directory",
        return_value=str(tmp_path / "share"),
    ):
        _rewrite_mesh_uris_for_foxglove(viz_marker)

    assert marker.mesh_resource.endswith(".glb")
    expected_orientation = (
        Rotation.from_quat(original_orientation)
        * Rotation.from_euler("x", -90, degrees=True)
    ).as_quat()
    actual_orientation = [
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ]
    assert np.allclose(actual_orientation, expected_orientation)


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
        shelf_type=ShelfType.BOOKCASE,
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
        shelf_type=ShelfType.BOOKCASE,
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
                dataclasses.replace(
                    _typed_object(ObjectType.CUP, f"cup_{index}"),
                    shelf_type=ShelfType.BOOKCASE,
                ),
                dataclasses.replace(
                    _typed_object(ObjectType.PLANT, f"plant_{index}"),
                    shelf_type=ShelfType.BOOKCASE,
                ),
            ],
            shelf_type=ShelfType.BOOKCASE,
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

    sample = next(
        iter(backend.evaluate(build_layer_query(ShelfType.BOOKCASE, free_count=2)))
    )

    assert len(sample.objects) == 2


_SAVE_SCRIPT = """
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import RelationalProbabilisticCircuit
from experiments.orm.ormatic_interface import *  # noqa: F401,F403  registers ORM mappers
from experiments.scene_generation_experiments.shelf_generation import TrainedArbitraryShelfModel
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D, EGPoint2D, EGRotation, EGScale, EGShelfLayer, ObjectType, ShelfType,
)
from pathlib import Path
import sys

def typed_object(object_type, object_id):
    return EGObject2D(
        id=object_id, room_id="room_1", place_id="shelf_1", object_type=object_type,
        scale=EGScale(height=0.1, length=0.1, width=0.1),
        position=EGPoint2D(x=0.0, y=0.0), orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id=object_id, shelf_type=ShelfType.BOOKCASE,
    )

types = [ObjectType.CUP, ObjectType.PLANT, ObjectType.BOOK, ObjectType.SHELF, ObjectType.CHAIR]
layers = [
    EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[typed_object(t, f"{t.value}_{i}") for t in types],
        shelf_type=ShelfType.BOOKCASE,
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
from semantic_digital_twin.scene_generation.scene_schema import ShelfType
from pathlib import Path
import sys

model = TrainedArbitraryShelfModel.load(Path(sys.argv[1]))
backend = probabilistic_backend(model.relational_probabilistic_circuit)
sample = next(iter(backend.evaluate(build_layer_query(ShelfType.BOOKCASE, free_count=2))))
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


def test_slab_size_does_not_depend_on_which_layers_are_present() -> None:
    """
    A slab spans the shelf's own footprint, so adding a layer of some other size must
    not resize the others.

    Slab size was once taken from the widest layer, which made every slab an accident of
    the company it kept; taking it from each layer's own scale instead let independently
    drawn layers disagree and leave the narrow ones floating clear of the corpus walls.
    The shelf is the one thing all its layers share.
    """
    shelf_scale = EGScale(height=2.0, length=0.6, width=0.8)

    def face_widths(layer_widths: tuple[float, ...]) -> list[float]:
        shelf = EGShelf(
            scale=shelf_scale,
            layers=[
                EGShelfLayer(
                    scale=EGScale(height=0.02, length=0.3, width=width),
                    objects=[],
                    shelf_type=ShelfType.BOOKCASE,
                )
                for width in layer_widths
            ],
            source_ids=None,
            shelf_type=ShelfType.BOOKCASE,
        )
        world = shelf.create_in_world()
        return sorted(
            annotation.root.collision.shapes[0].scale.y
            for annotation in world.get_semantic_annotations_by_type(ShelfLayer)
        )

    assert face_widths((0.4,)) == [pytest.approx(shelf_scale.width)]
    assert face_widths((0.4, 0.8)) == [pytest.approx(shelf_scale.width)] * 2


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
        shelf_type=ShelfType.BOOKCASE,
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


# ---- Group F -- conditioning contents and structure on the kind of shelf ----


def _layer_of(
    shelf_type: ShelfType, object_type: ObjectType, index: int
) -> EGShelfLayer:
    return EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            dataclasses.replace(
                _typed_object(object_type, f"{object_type.value}_{index}"),
                shelf_type=shelf_type,
            )
        ],
        shelf_type=shelf_type,
        relative_height=0.2,
    )


_BOOKCASE_SCALE = EGScale(height=2.0, length=0.3, width=0.4)
"""
Dimensions of the synthetic bookcases, so a query can pin what was fitted.
"""

_CABINET_SCALE = EGScale(height=1.0, length=0.3, width=0.4)
"""
Dimensions of the synthetic cabinets.
"""

_LAYER_SCALE = EGScale(height=0.02, length=0.3, width=0.4)
"""
Footprint of the synthetic layers, which every shelf of either type shares.

A layer carries the slab's own thickness, not the shelf's height, so pinning a layer to
a shelf-shaped scale asks the circuit for something it never saw.
"""


def _two_type_shelves() -> list[EGShelf]:
    """
    Shelves of two kinds whose contents and layer counts do not overlap.

    Disjoint object types are what makes conditioning observable: a draw for one
    kind that produced the other kind's objects could not be explained away as
    the model's own uncertainty.
    """
    bookcases = [
        EGShelf(
            scale=_BOOKCASE_SCALE,
            layers=[_layer_of(ShelfType.BOOKCASE, ObjectType.BOOK, index)] * 3,
            shelf_type=ShelfType.BOOKCASE,
        )
        for index in range(8)
    ]
    cabinets = [
        EGShelf(
            scale=_CABINET_SCALE,
            layers=[_layer_of(ShelfType.CABINET, ObjectType.CUP, index)],
            shelf_type=ShelfType.CABINET,
        )
        for index in range(8)
    ]
    return bookcases + cabinets


@pytest.fixture
def two_type_shelf_model() -> RelationalProbabilisticCircuit:
    return RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.25).fit(
        [to_dao(shelf) for shelf in _two_type_shelves()]
    )


def test_a_shelf_drawn_for_one_type_holds_that_types_objects(
    two_type_shelf_model: RelationalProbabilisticCircuit,
) -> None:
    """
    The whole point of the shelf type: a bookcase must be filled with what bookcases
    hold, not with the global mixture of everything on any shelf.

    Conditioning that quietly failed would still return a shelf, so the objects
    themselves are checked rather than merely that a draw succeeded.
    """
    backend = probabilistic_backend(two_type_shelf_model)

    bookcase = next(
        iter(backend.evaluate(build_shelf_query(ShelfType.BOOKCASE, _LAYER_SCALE, [1, 1, 1])))
    )
    cabinet = next(
        iter(backend.evaluate(build_shelf_query(ShelfType.CABINET, _LAYER_SCALE, [1])))
    )

    assert {obj.object_type for layer in bookcase.layers for obj in layer.objects} == {
        ObjectType.BOOK
    }
    assert {obj.object_type for layer in cabinet.layers for obj in layer.objects} == {
        ObjectType.CUP
    }


def test_a_drawn_shelf_carries_the_type_it_was_asked_for(
    two_type_shelf_model: RelationalProbabilisticCircuit,
) -> None:
    """
    The type is denormalized onto the layers, so a shelf whose layers disagreed with it
    would resample its contents against the wrong kind during repair.
    """
    backend = probabilistic_backend(two_type_shelf_model)

    shelf = next(
        iter(backend.evaluate(build_shelf_query(ShelfType.CABINET, _LAYER_SCALE, [1])))
    )

    assert shelf.shelf_type is ShelfType.CABINET
    assert {layer.shelf_type for layer in shelf.layers} == {ShelfType.CABINET}


def test_a_shelf_model_learns_a_template_for_its_layers(
    two_type_shelf_model: RelationalProbabilisticCircuit,
) -> None:
    """
    Layers are only reachable by fitting when an aggregation statistic declares them;
    without one a shelf-rooted circuit models the shelf's own dimensions and nothing of
    what it holds.
    """
    templates = two_type_shelf_model.exchangeable_distribution_templates

    assert "layers" in templates
    assert (
        "objects"
        in templates["layers"].template_distribution.exchangeable_distribution_templates
    )


def test_loading_a_model_fitted_before_shelf_types_is_refused(tmp_path: Path) -> None:
    """
    A model predating the shelf type loads and samples perfectly well, so nothing would
    look wrong -- every kind of shelf would simply come out identical.

    It is refused rather than served, since a cached model outliving a schema change is
    the ordinary case, not an exotic one.
    """
    layers = [
        EGShelfLayer(
            scale=EGScale(height=0.02, length=0.3, width=0.4),
            objects=[_typed_object(ObjectType.CUP, f"cup_{index}")],
            shelf_type=ShelfType.BOOKCASE,
        )
        for index in range(5)
    ]
    without_shelf_type = [
        dataclasses.replace(
            layer,
            objects=[
                dataclasses.replace(obj, shelf_type=ShelfType.BOOKCASE)
                for obj in layer.objects
            ],
        )
        for layer in layers
    ]
    model = TrainedArbitraryShelfModel(
        relational_probabilistic_circuit=RelationalProbabilisticCircuit(
            EGObject2D, min_samples_per_leaf=0.5
        ).fit([to_dao(obj) for layer in without_shelf_type for obj in layer.objects]),
        frequent_object_types={ObjectType.CUP},
    )
    export_path = tmp_path / "outdated.json"
    model.save(export_path)
    stored = json.loads(export_path.read_text())
    _drop_shelf_type_variables(stored)
    export_path.write_text(json.dumps(stored))

    with pytest.raises(OutdatedTrainedModelError):
        TrainedArbitraryShelfModel.load(export_path)


def _drop_shelf_type_variables(node: object) -> None:
    """
    Rename every stored ``shelf_type`` variable, as a pre-shelf-type fit had none.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                key == "name"
                and isinstance(value, str)
                and value.endswith("shelf_type")
            ):
                node[key] = "legacy_field"
            else:
                _drop_shelf_type_variables(value)
    elif isinstance(node, list):
        for item in node:
            _drop_shelf_type_variables(item)


# ---- Group G -- drawing the shelf's own attributes before its layers ----


def _shelf_of(shelf_type: ShelfType, layer_count: int, width: float) -> EGShelf:
    return EGShelf(
        scale=EGScale(height=2.0, length=0.3, width=width),
        layers=[
            EGShelfLayer(
                scale=EGScale(height=0.02, length=0.3, width=width),
                objects=[
                    dataclasses.replace(
                        _typed_object(ObjectType.BOOK, f"book_{index}"),
                        shelf_type=shelf_type,
                    )
                ],
                shelf_type=shelf_type,
                relative_height=0.2,
            )
            for index in range(layer_count)
        ],
        shelf_type=shelf_type,
    )


@pytest.fixture
def differing_structure_model() -> RelationalProbabilisticCircuit:
    """
    A model where one type has one narrow layer and the other has five wide ones.
    """
    shelves = [_shelf_of(ShelfType.CABINET, 1, 1.4) for _ in range(8)] + [
        _shelf_of(ShelfType.BOOKCASE, 5, 0.6) for _ in range(8)
    ]
    return RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.25).fit(
        [to_dao(shelf) for shelf in shelves]
    )


def test_the_drawn_layer_count_follows_the_type_it_was_asked_for(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    The count is learned per type, so pinning it at a constant throws that away and
    every kind of shelf comes out with the same number of levels.
    """
    sampler = ShelfDimensionSampler(differing_structure_model)

    cabinet_counts = {sampler.sample(ShelfType.CABINET).layer_count for _ in range(6)}
    bookcase_counts = {sampler.sample(ShelfType.BOOKCASE).layer_count for _ in range(6)}

    assert max(cabinet_counts) < min(bookcase_counts)


def test_a_drawn_layer_count_is_never_zero(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    A shelf with no layers holds nothing and spawns an empty box.
    """
    sampler = ShelfDimensionSampler(differing_structure_model)

    counts = [sampler.sample(shelf_type).layer_count
              for shelf_type in (ShelfType.CABINET, ShelfType.BOOKCASE)
              for _ in range(6)]

    assert all(count >= 1 for count in counts)


def test_every_layer_of_a_drawn_shelf_shares_the_shelfs_footprint(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    Object positions are drawn conditioned on the layer's scale, so layers whose
    footprints disagree with the shelf get positions meant for a differently sized
    surface -- bunched in the middle of a wide slab, or hanging off a narrow one.

    Every layer of a real shelf shares its footprint, and the draw has to preserve that.
    """
    shelf = draw_shelf(differing_structure_model, ShelfType.BOOKCASE)

    footprints = {(layer.scale.width, layer.scale.length) for layer in shelf.layers}
    assert len(footprints) == 1
    assert footprints == {(shelf.scale.width, shelf.scale.length)}


def test_a_shelf_can_be_drawn_for_every_type_the_model_knows(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    Pinning the layers to the shelf's own drawn scale left some types with no solution
    at all, which surfaced only when the demo was run: with real data the scale is
    continuous, so a value drawn from the shelf's distribution has no counterpart in the
    layers'.

    Drawing each known type is what catches that.
    """
    for shelf_type in (ShelfType.BOOKCASE, ShelfType.CABINET):
        shelf = draw_shelf(differing_structure_model, shelf_type)

        assert shelf.shelf_type is shelf_type
        assert shelf.layers


def test_a_shelf_is_drawn_even_when_a_layer_count_has_no_support(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    A layer count can carry mass in the shelf's own distribution while the shelf it
    implies has none: the grounded query conditions on the count, the kind of shelf
    and the layer structure together, which is stricter than the count's marginal.
    Drawing repeatedly is what turns that into a sample from the feasible
    conditional rather than an outright failure.
    """
    for _ in range(8):
        shelf = draw_shelf(differing_structure_model, ShelfType.CABINET)

        assert shelf.layers
        assert shelf.shelf_type is ShelfType.CABINET


def test_a_layer_count_the_model_rejects_outright_is_reported(
    differing_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    A count the caller pins is never redrawn, so one the model gives no probability to
    has to be reported rather than retried into a different shelf than asked for.

    Five is the telling case: the fit has seen five-layer shelves, just never
    five-layer cabinets. A count it has never seen at all lies outside the modelled
    range and is integrated out instead of rejected, so it would not exercise this.
    """
    with pytest.raises(UndrawableShelfError):
        draw_shelf(differing_structure_model, ShelfType.CABINET, layer_count=5)


# ---- Group H -- calibration must not block type differentiation on realistically
# sparse, uneven data ----


@pytest.fixture
def sparse_realistic_structure_model() -> RelationalProbabilisticCircuit:
    """
    Mirrors the real processed database's per-type shelf counts and layer counts (5
    cabinets mostly single-layer with one four-layer outlier, 6 bookcases spread 1-3, 11
    open shelves that are never single-layer and spread 2-5), fitted with the production
    calibration function rather than a hand-picked lenient fraction.

    A fixed fraction like the one :func:`differing_structure_model` uses never exercises
    the sparse-data calibration path an imbalanced, 22-row dataset this size triggers in
    production, so it could not have caught the regression this fixture guards against.
    """
    shelves = (
        [_shelf_of(ShelfType.CABINET, count, 1.4) for count in (1, 1, 1, 1, 4)]
        + [_shelf_of(ShelfType.BOOKCASE, count, 0.7) for count in (1, 1, 2, 3, 3, 3)]
        + [
            _shelf_of(ShelfType.OPEN_SHELF, count, 1.1)
            for count in (2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 5)
        ]
    )
    return RelationalProbabilisticCircuit(
        EGShelf, min_samples_per_leaf=min_samples_per_leaf_for
    ).fit([to_dao(shelf) for shelf in shelves])


def test_layer_count_still_differentiates_by_type_on_realistically_sparse_data(
    sparse_realistic_structure_model: RelationalProbabilisticCircuit,
) -> None:
    """
    Regression test for a real bug: applying one ``min_samples_per_leaf`` fraction
    calibrated for the whole dataset's row count unchanged to every circuit level gave
    the shelf-level circuit a leaf floor above the row count of two of its three shelf
    types, so it could never split on shelf type at all -- every type's drawn layer
    count collapsed to the same, type-blind distribution.

    Open shelf is the type that catches it: its training data never has a single
    layer, so a ``1`` appearing here means shelf-type conditioning has failed and
    the draw fell back to the pooled marginal (dominated by cabinet's four
    single-layer rows).
    """
    sampler = ShelfDimensionSampler(sparse_realistic_structure_model)

    open_shelf_counts = {
        sampler.sample(ShelfType.OPEN_SHELF).layer_count for _ in range(20)
    }

    assert 1 not in open_shelf_counts


# ---- Group I -- a layer's object count is learned per shelf type, not pinned ----


def _layer_with_object_count(shelf_type: ShelfType, count: int, index: int) -> EGShelfLayer:
    object_type = ObjectType.BOOK if shelf_type is ShelfType.BOOKCASE else ObjectType.CUP
    return EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            dataclasses.replace(
                _typed_object(object_type, f"{object_type.value}_{index}_{i}"),
                shelf_type=shelf_type,
            )
            for i in range(count)
        ],
        shelf_type=shelf_type,
        relative_height=0.2,
    )


@pytest.fixture
def differing_object_count_model() -> RelationalProbabilisticCircuit:
    """
    A model where bookcase layers hold many objects and cabinet layers hold few.
    """
    shelves = [
        EGShelf(
            scale=_BOOKCASE_SCALE,
            layers=[_layer_with_object_count(ShelfType.BOOKCASE, 4, index)] * 3,
            shelf_type=ShelfType.BOOKCASE,
        )
        for index in range(8)
    ] + [
        EGShelf(
            scale=_CABINET_SCALE,
            layers=[_layer_with_object_count(ShelfType.CABINET, 1, index)],
            shelf_type=ShelfType.CABINET,
        )
        for index in range(8)
    ]
    return RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.25).fit(
        [to_dao(shelf) for shelf in shelves]
    )


def test_the_drawn_object_count_follows_the_type_it_was_asked_for(
    differing_object_count_model: RelationalProbabilisticCircuit,
) -> None:
    """
    Object counts are learned per shelf type via EGShelfLayerAggregations, so pinning a
    layer's object count at a caller-chosen constant throws that away and every kind of
    shelf's layers would come out equally full.
    """
    layer_template = differing_object_count_model.exchangeable_distribution_templates[
        "layers"
    ]
    sampler = LayerObjectCountSampler(layer_template.template_distribution)

    cabinet_counts = {sampler.sample(ShelfType.CABINET) for _ in range(6)}
    bookcase_counts = {sampler.sample(ShelfType.BOOKCASE) for _ in range(6)}

    assert max(cabinet_counts) < min(bookcase_counts)


def test_a_drawn_object_count_is_never_negative(
    differing_object_count_model: RelationalProbabilisticCircuit,
) -> None:
    layer_template = differing_object_count_model.exchangeable_distribution_templates[
        "layers"
    ]
    sampler = LayerObjectCountSampler(layer_template.template_distribution)

    counts = [
        sampler.sample(shelf_type)
        for shelf_type in (ShelfType.CABINET, ShelfType.BOOKCASE)
        for _ in range(6)
    ]

    assert all(count >= 0 for count in counts)


def test_a_drawn_shelfs_layers_reflect_the_types_object_count(
    differing_object_count_model: RelationalProbabilisticCircuit,
) -> None:
    """
    draw_shelf must use the aggregation-sampled object count for each layer rather than
    a caller-supplied constant, so a bookcase comes out with fuller layers than a
    cabinet without either being asked for by count.
    """
    bookcase = draw_shelf(differing_object_count_model, ShelfType.BOOKCASE, layer_count=3)
    cabinet = draw_shelf(differing_object_count_model, ShelfType.CABINET, layer_count=1)

    bookcase_counts = [len(layer.objects) for layer in bookcase.layers]
    cabinet_counts = [len(layer.objects) for layer in cabinet.layers]
    assert min(bookcase_counts) > max(cabinet_counts)
