from __future__ import annotations

from experiments.shelf_generation_experiments.generation.in_world_resolver import (
    InWorldLayoutResolver,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale

from .spawned_shelves import chair_at, chair_candidate, spawned_single_layer_shelf


# %% helpers
def _standing_indices(shelf) -> list[int]:
    return [
        index
        for index, object_ in enumerate(shelf.layers[0].objects)
        if object_.annotation is not None
    ]


# %% repair
def test_a_collision_free_shelf_keeps_every_object(scenes_root) -> None:
    shelf = spawned_single_layer_shelf(
        [chair_at(-1.0, -1.0), chair_at(1.0, 1.0)], chair_candidate(scenes_root)
    )
    resolver = InWorldLayoutResolver(shelf=shelf)

    resolver.resolve()

    assert _standing_indices(shelf) == [0, 1]
    assert resolver.dropped_body_count == 0


def test_one_of_two_colliding_objects_is_dropped(scenes_root) -> None:
    shelf = spawned_single_layer_shelf(
        [chair_at(0.0, 0.0), chair_at(0.05, 0.0)], chair_candidate(scenes_root)
    )
    dropped_body = shelf.layers[0].objects[1].annotation
    resolver = InWorldLayoutResolver(shelf=shelf)

    resolver.resolve()

    assert _standing_indices(shelf) == [0]
    assert resolver.dropped_body_count == 1
    assert dropped_body not in shelf.world.bodies


def test_an_object_reaching_into_the_corpus_walls_is_dropped(scenes_root) -> None:
    shelf = spawned_single_layer_shelf(
        [chair_at(0.0, 0.0)], chair_candidate(scenes_root), Scale(x=0.2, y=0.2, z=2.0)
    )
    resolver = InWorldLayoutResolver(shelf=shelf)

    resolver.resolve()

    assert _standing_indices(shelf) == []
    assert resolver.dropped_body_count == 1


def test_an_object_lifted_off_its_slab_is_unsupported_but_not_colliding(
    scenes_root,
) -> None:
    shelf = spawned_single_layer_shelf(
        [chair_at(0.0, 0.0)], chair_candidate(scenes_root), Scale(x=4.0, y=4.0, z=4.0)
    )
    body = shelf.layers[0].objects[0].annotation
    resting_height = float(body.parent_connection.origin.position.z)
    body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        z=resting_height + 0.1, reference_frame=shelf.corpus
    )
    [group] = InWorldLayoutResolver(shelf=shelf).groups
    detector = FCLCollisionDetector(_world=shelf.world)

    colliding_indices = group.colliding_indices(detector)
    detector.stop()

    assert colliding_indices == set()
    assert group.unsupported_indices() == {0}
