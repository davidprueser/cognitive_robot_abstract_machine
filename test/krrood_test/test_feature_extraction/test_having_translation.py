from __future__ import annotations

import math

import pytest
from random_events.interval import closed_open, open_closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Integer

from krrood.entity_query_language.factories import (
    a,
    and_,
    count,
    entity,
    flat_variable,
    variable,
)
from krrood.parametrization.exceptions import (
    HavingConditionHasNoRandomEventRepresentation,
)
from krrood.parametrization.random_events_translator import (
    HavingExpressionToRandomEventTranslator,
)

from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    SceneFloor,
    SceneFloorAggregations,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    SceneRoomAggregations,
)

# %% translating having conditions


def statistic_variable(aggregation_class: type, statistic) -> Integer:
    """
    :param aggregation_class: The class declaring the statistic.
    :param statistic: A statistic counting something.
    :return: The random event variable standing for the statistic.
    """
    return Integer(aggregation_class.symbolic_feature_of(statistic)._name_)


def test_a_counted_collection_in_having_constrains_its_statistic():
    floor = a(SceneFloor)(rooms=[a(SceneRoom)(), a(SceneRoom)()])
    query = floor.having(count(flat_variable(floor.rooms)) >= 2).grouped_by(floor)
    room_count = statistic_variable(
        SceneFloorAggregations, SceneFloorAggregations.room_count
    )

    event = HavingExpressionToRandomEventTranslator(
        and_(*query._having_conditions_), SceneFloorAggregations
    ).translate()

    assert (
        event
        == SimpleEvent.from_data(
            {room_count: closed_open(2, math.inf)}
        ).as_composite_set()
    )


def test_a_filtered_count_in_having_constrains_its_statistic():
    room = a(SceneRoom)(objects=[a(SceneObject)(type=...)])
    object_type = flat_variable(room.objects).type
    query = room.having(
        entity(count(object_type)).where(object_type == SceneObjectType.CHAIR) <= 3
    ).grouped_by(room)
    chair_count = statistic_variable(
        SceneRoomAggregations, SceneRoomAggregations.chair_count
    )

    event = HavingExpressionToRandomEventTranslator(
        and_(*query._having_conditions_), SceneRoomAggregations
    ).translate()

    assert (
        event
        == SimpleEvent.from_data(
            {chair_count: open_closed(-math.inf, 3)}
        ).as_composite_set()
    )


def test_an_aggregation_no_statistic_computes_is_refused():
    room = variable(SceneRoom)
    object_type = flat_variable(room.objects).type
    condition = (
        entity(count(object_type)).where(object_type != SceneObjectType.CHAIR) <= 3
    )

    with pytest.raises(HavingConditionHasNoRandomEventRepresentation):
        HavingExpressionToRandomEventTranslator(
            condition, SceneRoomAggregations
        ).translate()
