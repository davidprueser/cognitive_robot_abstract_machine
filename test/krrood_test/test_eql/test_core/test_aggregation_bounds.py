import math

import pytest

import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.exceptions import AggregatorHasNoBounds
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.operators.aggregators import (
    AggregationBounds,
    PartiallyKnownValues,
)

# %% counting


def test_count_bounds_range_from_certain_to_possible_values():
    values = PartiallyKnownValues(
        included_values=[1, 2],
        possibly_included_values=[3],
        unstated_included_count=1,
        unstated_possibly_included_count=2,
    )

    bounds = eql.count(variable(int, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=3, upper=6)


def test_count_bounds_are_open_above_for_a_collection_of_unstated_size():
    values = PartiallyKnownValues(included_values=[1, 2], is_size_unstated=True)

    bounds = eql.count(variable(int, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=2, upper=math.inf)


def test_count_bounds_of_stated_values_are_exact():
    values = PartiallyKnownValues(included_values=[1, 2, 3])

    bounds = eql.count(variable(int, [])).bounds_of(values)

    assert bounds.is_exact


# %% summing and averaging


def test_sum_bounds_add_the_possibly_included_values_of_each_sign():
    values = PartiallyKnownValues(
        included_values=[1.0, 2.0], possibly_included_values=[-4.0, 5.0]
    )

    bounds = eql.sum(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=-1.0, upper=8.0)


def test_sum_bounds_are_unbounded_when_an_unstated_value_is_included():
    values = PartiallyKnownValues(included_values=[1.0], unstated_included_count=1)

    bounds = eql.sum(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=-math.inf, upper=math.inf)


def test_average_bounds_are_the_extreme_averages_over_the_possible_selections():
    values = PartiallyKnownValues(
        included_values=[2.0], possibly_included_values=[0.0, 1.0, 6.0]
    )

    bounds = eql.average(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=1.0, upper=4.0)


def test_average_bounds_of_stated_values_are_their_average():
    values = PartiallyKnownValues(included_values=[1.0, 2.0, 6.0])

    bounds = eql.average(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=3.0, upper=3.0)


def test_average_bounds_are_unbounded_when_an_unstated_value_may_be_included():
    values = PartiallyKnownValues(
        included_values=[1.0], unstated_possibly_included_count=1
    )

    bounds = eql.average(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=-math.inf, upper=math.inf)


# %% extremes


def test_max_bounds_range_from_the_certain_to_the_largest_possible_value():
    values = PartiallyKnownValues(
        included_values=[1.0, 3.0], possibly_included_values=[2.0, 7.0]
    )

    bounds = eql.max(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=3.0, upper=7.0)


def test_max_bounds_are_open_above_when_an_unstated_value_may_be_included():
    values = PartiallyKnownValues(included_values=[1.0], is_size_unstated=True)

    bounds = eql.max(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=1.0, upper=math.inf)


def test_min_bounds_range_from_the_smallest_possible_to_the_certain_value():
    values = PartiallyKnownValues(
        included_values=[4.0, 3.0], possibly_included_values=[2.0, 7.0]
    )

    bounds = eql.min(variable(float, [])).bounds_of(values)

    assert bounds == AggregationBounds(lower=2.0, upper=3.0)


# %% aggregations without bounds


def test_mode_has_no_bounds():
    with pytest.raises(AggregatorHasNoBounds):
        eql.mode(variable(int, [])).bounds_of(
            PartiallyKnownValues(included_values=[1, 2])
        )


def test_distinct_count_has_no_bounds():
    with pytest.raises(AggregatorHasNoBounds):
        eql.count(variable(int, []), distinct=True).bounds_of(PartiallyKnownValues())
