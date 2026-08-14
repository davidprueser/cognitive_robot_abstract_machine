from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    EGShelf,
    EGShelfLayer,
    ObjectType,
    ShelfType,
)
from semantic_digital_twin.scene_generation.scene_schema_aggregations import (
    EGShelfAggregations,
    EGShelfLayerAggregations,
)


def _object(object_id: str) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=EGScale(width=0.1, length=0.05, height=0.2),
        position=EGPoint2D(x=0.0, y=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="book_src",
        shelf_type=ShelfType.BOOKCASE,
    )


def _layer(object_count: int) -> EGShelfLayer:
    return EGShelfLayer(
        scale=EGScale(width=0.6, length=0.3, height=EGShelfLayer.SLAB_THICKNESS),
        objects=[_object(f"book_{index}") for index in range(object_count)],
        shelf_type=ShelfType.BOOKCASE,
        relative_height=0.2,
    )


@pytest.mark.parametrize("object_count", [0, 1, 3, 7])
def test_a_layer_counts_the_objects_it_holds(object_count: int) -> None:
    """
    Aggregation statistics are the only thing a fitted circuit passes from a shelf
    to its layers, or from a layer to its objects. One that reports a constant
    leaves the two levels independent while still appearing in the model, so its
    value has to be pinned and not merely its presence.
    """
    layer = _layer(object_count)

    assert EGShelfLayerAggregations(layer).total_count() == object_count


@pytest.mark.parametrize("layer_count", [1, 2, 5])
def test_a_shelf_counts_the_layers_it_holds(layer_count: int) -> None:
    """
    How many levels a kind of shelf has is the one structural quantity the data
    supplies, and it is what decides how many slabs a sampled shelf gets.
    """
    shelf = EGShelf(
        scale=EGScale(width=0.6, length=0.3, height=2.0),
        layers=[_layer(1) for _ in range(layer_count)],
        shelf_type=ShelfType.BOOKCASE,
    )

    assert EGShelfAggregations(shelf).layer_count() == layer_count
