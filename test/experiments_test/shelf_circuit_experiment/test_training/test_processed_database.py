from __future__ import annotations

from pathlib import Path

import pytest

from experiments.shelf_generation_experiments.dataset_environment import (
    DatasetEnvironmentVariable,
)
from experiments.shelf_generation_experiments.preprocessing.preprocess_sage10k import (
    PreprocessedObject,
)
from experiments.shelf_generation_experiments.training.processed_database import (
    ProcessedShelfDatabase,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import random_shelves, scene_with_chair_meshes


# %% helpers
def _stored_object(
    source_id: str, object_type: ObjectType, scale: Scale
) -> PreprocessedObject:
    return PreprocessedObject(
        id=f"{source_id}_instance",
        room_id="room",
        place_id="floor",
        object_type=object_type,
        scale=scale,
        pose=Pose.from_xyz_rpy(x=1.0, y=2.0),
        source_id=source_id,
    )


# %% shelves
def test_stored_shelves_are_read_back_with_every_object_pose(
    processed_database,
) -> None:
    shelves = random_shelves(shelves_per_theme=1)
    processed_database.session.add_all([to_dao(shelf) for shelf in shelves])
    processed_database.session.commit()

    read_shelves = processed_database.shelves()

    def poses(shelf_collection):
        return sorted(
            (float(object_.pose.x), float(object_.pose.y), float(object_.pose.yaw))
            for shelf in shelf_collection
            for layer in shelf.layers
            for object_ in layer.objects
        )

    assert poses(read_shelves) == pytest.approx(poses(shelves))
    assert sorted(len(shelf.layers) for shelf in read_shelves) == sorted(
        len(shelf.layers) for shelf in shelves
    )


# %% mesh candidates
def test_mesh_candidates_are_the_stored_objects_of_the_types_with_a_mesh(
    processed_database, scenes_root
) -> None:
    book_scale = Scale(x=0.2, y=0.1, z=0.3)
    processed_database.session.add_all(
        [
            to_dao(_stored_object("book_mesh", ObjectType.BOOK, book_scale)),
            to_dao(_stored_object("lamp_mesh", ObjectType.LAMP, book_scale)),
            to_dao(_stored_object("book_without_mesh", ObjectType.BOOK, book_scale)),
        ]
    )
    processed_database.session.commit()
    scene_directory = scene_with_chair_meshes(
        scenes_root / "scene", ["book_mesh", "lamp_mesh"]
    )

    [candidate] = processed_database.mesh_candidates({ObjectType.BOOK}, scenes_root)

    assert candidate.source_id == "book_mesh"
    assert candidate.object_type is ObjectType.BOOK
    assert candidate.scene_directory == scene_directory
    assert candidate.scale == book_scale


# %% opening the database
def test_the_database_the_environment_names_is_opened_with_its_schema(
    tmp_path: Path, monkeypatch
) -> None:
    database_path = tmp_path / "from_environment.db"
    monkeypatch.setenv(
        DatasetEnvironmentVariable.PROCESSED_DATABASE_URI.value,
        f"sqlite:///{database_path}",
    )

    database = ProcessedShelfDatabase.from_environment()

    assert database.shelves() == []
    assert database_path.exists()
