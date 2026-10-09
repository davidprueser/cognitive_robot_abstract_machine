from __future__ import annotations

import pytest

from experiments.shelf_generation_experiments.generation.shelf_generator import (
    ShelfGenerator,
)
from experiments.shelf_generation_experiments.preprocessing.preprocess_sage10k import (
    PreprocessedObject,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from krrood.entity_query_language.exceptions import NoSolutionFound
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import scene_with_chair_meshes


# %% fixtures
@pytest.fixture
def source_ids_by_type() -> dict[ObjectType, str]:
    """
    The source id of the stored object of every type the shelf model samples.
    """
    return {
        ObjectType.BOOK: "generator_book",
        ObjectType.BOTTLE: "generator_bottle",
        ObjectType.BOX: "generator_box",
    }


@pytest.fixture
def generator(
    shelf_model, processed_database, scenes_root, source_ids_by_type
) -> ShelfGenerator:
    """
    A generator whose database holds one small object with a chair mesh per type the
    shelf model samples.
    """
    processed_database.session.add_all(
        [
            to_dao(
                PreprocessedObject(
                    id=source_id,
                    room_id="room",
                    place_id="shelf",
                    object_type=object_type,
                    scale=Scale(x=0.15, y=0.15, z=0.15),
                    pose=Pose.from_xyz_rpy(),
                    source_id=source_id,
                )
            )
            for object_type, source_id in source_ids_by_type.items()
        ]
    )
    processed_database.session.commit()
    scene_with_chair_meshes(scenes_root / "scene", list(source_ids_by_type.values()))
    return ShelfGenerator(
        model=shelf_model, database=processed_database, scenes_root=scenes_root
    )


# %% generation
def test_a_generated_shelf_answers_its_query(generator) -> None:
    world = World.create_with_root_body()

    generated = generator.generate(
        RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOOK, [2, 2]),
        world,
    )

    assert generated.shelf.theme_dominant_type is ObjectType.BOOK
    assert [len(layer.objects) for layer in generated.shelf.layers] == [2, 2]
    assert generated.shelf.world is world


def test_every_object_is_either_standing_with_a_mesh_or_dropped(
    generator, source_ids_by_type
) -> None:
    world = World.create_with_root_body()

    generated = generator.generate(
        RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOOK, [2, 2]),
        world,
    )

    standing_objects = [
        object_
        for layer in generated.shelf.layers
        for object_ in layer.objects
        if object_.annotation is not None
    ]
    assert len(standing_objects) == generated.standing_object_count
    assert generated.standing_object_count + generated.dropped_object_count == 4
    assert all(object_.annotation in world.bodies for object_ in standing_objects)
    assert {object_.source_id for object_ in standing_objects} <= set(
        source_ids_by_type.values()
    )


def test_a_generated_shelf_stands_where_it_is_placed(generator) -> None:
    world = World.create_with_root_body()
    placement = HomogeneousTransformationMatrix.from_xyz_rpy(x=3.0, y=-2.0)

    generated = generator.generate(
        RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOOK, [1]),
        world,
        parent_T_self=placement,
    )

    corpus_position = generated.shelf.corpus.global_transform.position
    assert [float(corpus_position.x), float(corpus_position.y)] == pytest.approx(
        [3.0, -2.0]
    )


def test_a_query_the_model_does_not_support_has_no_solution(generator) -> None:
    with pytest.raises(NoSolutionFound):
        generator.generate(
            RelationalCircuitExperimentShelf.underspecified_query(ObjectType.LAMP, [1]),
            World.create_with_root_body(),
        )
