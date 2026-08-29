from __future__ import annotations

from experiments.scene_generation_experiments.rspn_sampling import (
    build_theme_shelf_query,
    probabilistic_backend,
)
from krrood.entity_query_language.factories import a
from semantic_digital_twin.scene_generation.scene_schema import EGShelf, ObjectType
from semantic_digital_twin.world_description.geometry import Scale

from .rspn_test_fixtures import (
    build_factorized_shelf_rspn,
    factorized_shelf_rspn,  # noqa: F401  fixture
    is_single_leaf,
)


def test_the_fitted_circuit_is_single_leaf_everywhere(factorized_shelf_rspn) -> None:
    """
    The whole point of "fully factorized" is that no level of the fit -- not the
    shelf, not its layers, not their objects -- learned any correlation between
    fields.
    """
    assert is_single_leaf(factorized_shelf_rspn)


def test_the_fitted_circuit_still_covers_every_theme_it_was_given() -> None:
    """
    A single-leaf fit must not collapse a categorical field like ``theme_dominant_type``
    down to one value -- both themes given to the builder must still ground.
    """
    circuit = build_factorized_shelf_rspn(
        themes=(ObjectType.BOOK, ObjectType.BOTTLE), shelves_per_theme=4
    )
    backend = probabilistic_backend(circuit)
    for theme in (ObjectType.BOOK, ObjectType.BOTTLE):
        query = a(EGShelf)(
            scale=a(Scale)(x=..., y=..., z=...),
            layers=[],
            theme_dominant_type=theme,
        )
        query.resolve()
        sample = next(iter(backend.evaluate(query)))
        assert sample.theme_dominant_type is theme


def test_repeated_samples_of_the_same_query_vary(factorized_shelf_rspn) -> None:
    """
    A fully factorized circuit is still a real distribution, not a constant: drawing
    the same underspecified shelf query twice should not always land on the exact same
    pose, or the fixture would be useless for exercising resample logic that expects a
    fresh draw each time.
    """
    backend = probabilistic_backend(factorized_shelf_rspn)

    positions = set()
    for _ in range(10):
        query = build_theme_shelf_query(ObjectType.BOOK, objects_per_layer=[1])
        sample = next(iter(backend.evaluate(query)))
        pose = sample.layers[0].objects[0].pose
        positions.add((float(pose.x), float(pose.y)))

    assert len(positions) > 1
