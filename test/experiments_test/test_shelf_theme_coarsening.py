from __future__ import annotations

from experiments.scene_generation_experiments.shelf_generation import (
    _coarsen_rare_shelf_themes,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


def _themed_shelf(theme: ObjectType) -> EGShelf:
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


# %% coarsening a shelf whose theme falls outside the kept set


def test_a_rare_themed_shelf_is_coarsened_without_error() -> None:
    """
    A shelf whose theme is not among the most frequent ones is replaced with
    ``ObjectType.OTHER`` on the shelf and its layer -- not on the layer's objects,
    since ``EGObject2D`` carries no theme of its own to coarsen.
    """
    shelves = [
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOTTLE),
    ]

    coarsened = _coarsen_rare_shelf_themes(shelves, keep_count=1)

    rare_shelf = coarsened[2]
    assert rare_shelf.theme_dominant_type is ObjectType.OTHER
    assert rare_shelf.layers[0].theme_dominant_type is ObjectType.OTHER


def test_a_rare_themed_shelfs_objects_are_left_untouched() -> None:
    """
    Coarsening replaces the shelf's and its layers' theme, but the objects on those
    layers carry no theme field and must pass through unchanged.
    """
    shelves = [
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOTTLE),
    ]
    original_object = shelves[2].layers[0].objects[0]

    coarsened = _coarsen_rare_shelf_themes(shelves, keep_count=1)

    assert coarsened[2].layers[0].objects[0] is original_object


def test_a_frequently_themed_shelf_is_returned_unchanged() -> None:
    """
    A shelf whose theme is among the most frequent ones is not coarsened at all -- the
    very shelf instance is returned, not a replaced copy.
    """
    shelves = [
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOOK),
        _themed_shelf(ObjectType.BOTTLE),
    ]

    coarsened = _coarsen_rare_shelf_themes(shelves, keep_count=1)

    assert coarsened[0] is shelves[0]
    assert coarsened[1] is shelves[1]
