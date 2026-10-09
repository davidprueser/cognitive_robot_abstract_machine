from __future__ import annotations

import dataclasses
from pathlib import Path

from experiments.shelf_generation_experiments.generation.pre_spawn_resolver import (
    PreSpawnLayoutResolver,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import random_shelves


# %% helpers
def _book_footprint() -> Scale:
    """
    :return: The size of every book of these tests, and of the mesh matched for it.
    """
    return Scale(x=0.15, y=0.15, z=0.15)


def _book_at(x: float, y: float) -> RelationalCircuitExperimentObject2D:
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.BOOK,
        scale=_book_footprint(),
        pose=Pose2D(x=x, y=y, yaw=0.0),
        source_id="book",
    )


def _resolver(
    objects: list[RelationalCircuitExperimentObject2D], shelf_model, **settings
) -> PreSpawnLayoutResolver:
    """
    :return: A resolver for a spawned, still empty single-layer shelf of the training
        data holding *objects*, every one matched to a book mesh of :func:`_book_footprint`.
    """
    training_shelf = random_shelves()[0]
    shelf: RelationalCircuitExperimentShelf = dataclasses.replace(
        training_shelf,
        layers=[dataclasses.replace(training_shelf.layers[0], objects=objects)],
        source_ids=[
            MeshCandidate(Path("scene"), "book", ObjectType.BOOK, _book_footprint())
        ],
    )
    corpus, layers = shelf._spawn_corpus_and_slabs(
        World.create_with_root_body(), slab_thickness=0.02, corpus_wall_thickness=0.03
    )
    resolver = PreSpawnLayoutResolver.for_shelf(shelf, layers, shelf_model, corpus)
    return dataclasses.replace(resolver, **settings)


# %% one layer
def test_an_object_outside_its_layer_is_moved_back_in(shelf_model) -> None:
    resolver = _resolver([_book_at(5.0, -5.0)], shelf_model)
    [group] = resolver.groups

    group.clamp_to_bounds()

    half_x = group.shelf_scale.x / 2 - _book_footprint().x / 2
    half_y = group.shelf_scale.y / 2 - _book_footprint().y / 2
    pose = group.objects[0].pose
    assert [float(pose.x), float(pose.y)] == [half_x, -half_y]


def test_overlapping_footprints_name_one_object_to_move(shelf_model) -> None:
    resolver = _resolver([_book_at(0.0, 0.0), _book_at(0.05, 0.0)], shelf_model)

    assert resolver.groups[0].colliding_indices() == {1}


def test_separate_footprints_name_none(shelf_model) -> None:
    resolver = _resolver([_book_at(-0.3, 0.0), _book_at(0.3, 0.0)], shelf_model)

    assert resolver.groups[0].colliding_indices() == set()


# %% whole shelf
def test_resolving_redraws_overlapping_objects_apart(shelf_model) -> None:
    resolver = _resolver([_book_at(0.0, 0.0), _book_at(0.0, 0.0)], shelf_model)

    matches = resolver.resolve()

    assert resolver.groups[0].colliding_indices() == set()
    assert resolver.dropped_object_count == 0
    assert set(matches[0]) == {0, 1}


def test_what_still_overlaps_without_repair_passes_is_dropped(shelf_model) -> None:
    resolver = _resolver(
        [_book_at(0.0, 0.0), _book_at(0.0, 0.0)], shelf_model, maximum_pass_count=0
    )

    matches = resolver.resolve()

    assert resolver.dropped_object_count == 1
    assert set(matches[0]) == {0}
