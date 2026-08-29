from __future__ import annotations

from pathlib import Path

import pytest

from experiments.scene_generation_experiments import shelf_generation
from experiments.scene_generation_experiments.utils import MAXIMUM_LEAF_COUNT
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


def _shelf(theme: ObjectType) -> EGShelf:
    """
    A minimal shelf with one layer holding one object, all carrying *theme*.
    """
    object_2d = EGObject2D(
        object_type=theme,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id="object",
    )
    layer = EGShelfLayer(objects=[object_2d], theme_dominant_type=theme)
    return EGShelf(
        scale=Scale(x=1.0, y=1.0, z=1.0), layers=[layer], theme_dominant_type=theme
    )


# %% the fitted circuit uses the fixed leaf-budget fraction


def test_the_fitted_shelf_circuit_uses_the_fixed_leaf_budget_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    At every level of the real processed sage10k v2 database (18,437 shelves / 44,609
    layers / 124,800 objects, confirmed against ``sage_processed_data_v2``), the leaf-
    count budget ``1 / MAXIMUM_LEAF_COUNT`` is far tighter than any per-row overfitting
    floor could be, so the RSPN is fit with that fixed fraction directly rather than a
    per-level dynamic calculation.
    """
    shelves = [
        _shelf(ObjectType.BOOK),
        _shelf(ObjectType.BOOK),
        _shelf(ObjectType.BOTTLE),
    ]
    monkeypatch.setattr(shelf_generation, "load_shelves", lambda session: shelves)

    trained_model = shelf_generation._load_or_train_shelf_model(
        tmp_path / "model.json", session=None
    )

    assert trained_model.relational_probabilistic_circuit.min_samples_per_leaf == (
        pytest.approx(1 / MAXIMUM_LEAF_COUNT)
    )
