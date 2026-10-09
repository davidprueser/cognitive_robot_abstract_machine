from __future__ import annotations

from pathlib import Path

import pytest

from experiments.shelf_generation_experiments.preprocessing.preprocess_sage10k import (
    PreprocessedObject,
)
from experiments.shelf_generation_experiments.tidying_demo.demo import (
    ShelfTidyingDemo,
)
from experiments.shelf_generation_experiments.tidying_demo.exceptions import (
    NoStandingMeshError,
)
from experiments.shelf_generation_experiments.tidying_demo.visualization import (
    VisualizationBackend,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.spatial_types import Point3, Pose
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import scene_with_chair_meshes


# %% fixtures
@pytest.fixture
def standing_source_ids_by_type() -> dict[ObjectType, str]:
    """
    The source id of the standing object stored for every type the model samples.
    """
    return {
        ObjectType.BOOK: "demo_book",
        ObjectType.BOTTLE: "demo_bottle",
        ObjectType.BOX: "demo_box",
    }


@pytest.fixture
def graspable_size_factor() -> float:
    """
    Shrinks the chair mesh to a graspable object of about 0.12 by 0.13 by 0.15 metres.
    """
    return 0.17


@pytest.fixture
def demo(
    shelf_model,
    processed_database,
    scenes_root,
    tmp_path: Path,
    standing_source_ids_by_type,
    graspable_size_factor,
):
    """
    A demo over a stored shelf model and, per type the model samples, one standing
    object small enough to grasp.
    """
    processed_database.session.add_all(
        [
            to_dao(
                PreprocessedObject(
                    id=source_id,
                    room_id="room",
                    place_id="table",
                    object_type=object_type,
                    scale=Scale(x=0.118, y=0.126, z=0.150),
                    pose=Pose.from_xyz_rpy(),
                    source_id=source_id,
                )
            )
            for object_type, source_id in standing_source_ids_by_type.items()
        ]
    )
    processed_database.session.commit()
    scene_with_chair_meshes(
        scenes_root / "scene",
        list(standing_source_ids_by_type.values()),
        size_factor=graspable_size_factor,
    )
    model_path = tmp_path / "shelf_model.json"
    shelf_model.save(model_path)
    return ShelfTidyingDemo(
        database=processed_database,
        scenes_root=scenes_root,
        model_path=model_path,
        objects_per_layer=[1, 1],
        visualization_backend=VisualizationBackend.RVIZ,
    )


# %% tidying
def test_the_robot_puts_the_object_on_the_layer_it_belongs_on(demo, rclpy_node) -> None:
    tidied = demo.run(rclpy_node)

    world = tidied.body._world
    placed_object = tidied.placement.placed_object
    goal = world.transform(
        Point3(
            float(placed_object.pose.x),
            float(placed_object.pose.y),
            0.0,
            reference_frame=tidied.placement.layer.annotation.root,
        ),
        world.root,
    )
    position = tidied.body.global_transform.position
    assert [float(position.x), float(position.y), float(position.z)] == pytest.approx(
        [float(goal.x), float(goal.y), float(goal.z)], abs=0.05
    )


def test_without_standing_meshes_there_is_nothing_to_tidy(
    shelf_model, processed_database, tmp_path: Path, rclpy_node
) -> None:
    empty_scenes_root = tmp_path / "empty_scenes"
    empty_scenes_root.mkdir()
    model_path = tmp_path / "shelf_model.json"
    shelf_model.save(model_path)
    demo = ShelfTidyingDemo(
        database=processed_database,
        scenes_root=empty_scenes_root,
        model_path=model_path,
        visualization_backend=VisualizationBackend.RVIZ,
    )

    with pytest.raises(NoStandingMeshError):
        demo.run(rclpy_node)
