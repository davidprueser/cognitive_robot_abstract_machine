from __future__ import annotations

from pathlib import Path

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.training.shelf_model import (
    ShelfModel,
    ShelfModelFitSettings,
    ShelfPart,
)
from experiments.shelf_generation_experiments.utils import ObjectType

from ..shelf_dataset import random_shelves, single_leaf_fit_settings


# %% fitting
def test_every_level_of_the_shelf_is_fitted_with_the_leaf_share(shelf_model) -> None:
    settings = single_leaf_fit_settings()
    layer_circuit = shelf_model.layer_circuit
    object_circuit = layer_circuit.exchangeable_distribution_templates[
        ShelfPart.OBJECTS
    ].template_distribution

    assert [
        circuit.learning_method.min_samples_per_leaf
        for circuit in (shelf_model.circuit, layer_circuit, object_circuit)
    ] == [settings.min_samples_per_leaf] * 3


def test_every_level_is_fitted_with_the_histogram_granularity(shelf_model) -> None:
    layer_circuit = shelf_model.layer_circuit

    assert layer_circuit.min_samples_per_quantile == (
        single_leaf_fit_settings().min_samples_per_quantile
    )


def test_the_layer_circuit_is_the_template_of_the_layers(shelf_model) -> None:
    assert (
        shelf_model.layer_circuit
        is shelf_model.circuit.exchangeable_distribution_templates[
            ShelfPart.LAYERS
        ].template_distribution
    )


def test_fitting_coarsens_the_types_beyond_the_keep_count() -> None:
    shelves = random_shelves(
        themes=(ObjectType.BOOK, ObjectType.BOTTLE),
        object_types=(ObjectType.BOOK, ObjectType.BOTTLE, ObjectType.BOX),
    )

    model = ShelfModel.fit(
        shelves,
        ShelfModelFitSettings(min_samples_per_leaf=0.99, object_type_keep_count=1),
    )

    sampled_shelves = list(
        model.shelf_backend(number_of_samples=20).evaluate(
            RelationalCircuitExperimentShelf.underspecified_query(..., [1])
        )
    )
    assert {shelf.theme_dominant_type for shelf in sampled_shelves} <= (
        model.coarsening.frequent_theme_types | {ObjectType.OTHER}
    )
    assert ObjectType.OTHER in {shelf.theme_dominant_type for shelf in sampled_shelves}


# %% storage
def test_a_saved_model_loads_with_the_same_circuit_and_coarsening(
    shelf_model, tmp_path: Path
) -> None:
    path = tmp_path / "models" / "shelf_model.json"

    shelf_model.save(path)
    loaded = ShelfModel.load(path)

    assert loaded.coarsening == shelf_model.coarsening
    assert {
        variable.name
        for variable in loaded.circuit.class_probabilistic_circuit.variables
    } == {
        variable.name
        for variable in shelf_model.circuit.class_probabilistic_circuit.variables
    }
