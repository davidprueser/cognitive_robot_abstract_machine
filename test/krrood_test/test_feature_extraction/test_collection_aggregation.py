from __future__ import annotations

import math

import pytest

from krrood.entity_query_language.factories import (
    a,
    count,
    entity,
    flat_variable,
    variable,
)
from krrood.entity_query_language.operators.aggregators import AggregationBounds
from krrood.parametrization.feature_extraction.collection_aggregation import (
    CollectionAggregation,
)
from krrood.parametrization.feature_extraction.exceptions import (
    AggregationIsNotOverACollection,
)

from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import SceneObject, SceneObjectType, SceneRoom

# %% aggregations


def chair_count_of(room: SceneRoom) -> CollectionAggregation:
    """
    :param room: The room, or a variable standing for it.
    :return: The count of chairs in the room.
    """
    object_type = flat_variable(room.objects).type
    return CollectionAggregation.from_query(
        entity(count(object_type)).where(object_type == SceneObjectType.CHAIR)
    )


def object_count_of(room: SceneRoom) -> CollectionAggregation:
    """
    :param room: The room, or a variable standing for it.
    :return: The count of objects in the room.
    """
    return CollectionAggregation.from_query(count(flat_variable(room.objects)))


# %% bounds stated by a query


def test_count_bounds_leave_room_for_elements_whose_condition_is_unstated():
    room = a(SceneRoom)(
        objects=[
            a(SceneObject)(type=SceneObjectType.CHAIR),
            a(SceneObject)(type=SceneObjectType.TABLE),
            a(SceneObject)(type=...),
            a(SceneObject)(),
        ]
    )

    bounds = chair_count_of(variable(SceneRoom, [])).bounds_stated_by(room)

    assert bounds == AggregationBounds(lower=1, upper=3)


def test_count_of_stated_elements_is_exact():
    room = a(SceneRoom)(objects=[a(SceneObject)(type=...) for _ in range(3)])

    bounds = object_count_of(variable(SceneRoom, [])).bounds_stated_by(room)

    assert bounds == AggregationBounds(lower=3, upper=3)


def test_count_over_an_unstated_collection_is_open_above():
    room = a(SceneRoom)(objects=...)

    bounds = object_count_of(variable(SceneRoom, [])).bounds_stated_by(room)

    assert bounds == AggregationBounds(lower=0, upper=math.inf)


def test_count_over_a_concrete_instance_is_exact():
    room = SceneRoom(
        position=None,
        orientation=None,
        objects=[
            SceneObject(SceneObjectType.CHAIR),
            SceneObject(SceneObjectType.TABLE),
        ],
    )

    bounds = chair_count_of(variable(SceneRoom, [])).bounds_stated_by(room)

    assert bounds == AggregationBounds(lower=1, upper=1)


# %% the same aggregation written twice


def test_aggregations_written_over_different_subjects_are_equal():
    assert chair_count_of(variable(SceneRoom, [])) == chair_count_of(
        variable(SceneRoom, [])
    )


def test_aggregations_with_different_conditions_differ():
    room = variable(SceneRoom, [])
    object_type = flat_variable(room.objects).type
    table_count = CollectionAggregation.from_query(
        entity(count(object_type)).where(object_type == SceneObjectType.TABLE)
    )

    assert table_count != chair_count_of(room)


def test_counted_and_filtered_counts_differ():
    room = variable(SceneRoom, [])

    assert object_count_of(room) != chair_count_of(room)


# %% queries that aggregate no collection


def test_an_aggregation_over_a_variable_is_not_over_a_collection():
    with pytest.raises(AggregationIsNotOverACollection):
        CollectionAggregation.from_query(count(variable(SceneObject, [])))
