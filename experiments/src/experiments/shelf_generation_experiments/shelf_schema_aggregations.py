from __future__ import annotations

from dataclasses import dataclass

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from krrood.entity_query_language.factories import (
    count,
    entity,
    flat_variable,
    variable,
)
from krrood.entity_query_language.query.query import Query
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)


@dataclass(eq=False)
class RelationalCircuitExperimentShelfLayerAggregations(
    AggregationStatistic[RelationalCircuitExperimentShelfLayer]
):
    """
    Aggregation statistics over the objects on an RelationalCircuitExperimentShelfLayer.
    """

    @aggregation_statistic(variable(RelationalCircuitExperimentShelfLayer).objects)
    def total_count(self) -> Query[int]:
        """
        Number of objects placed on the shelf layer.
        """
        return entity(count(flat_variable(self.objects)))


@dataclass(eq=False)
class RelationalCircuitExperimentShelfAggregations(
    AggregationStatistic[RelationalCircuitExperimentShelf]
):
    """
    Aggregation statistics over the layers of an RelationalCircuitExperimentShelf.

    Declaring these is what makes a shelf's layers visible to fitting at all: a
    collection with no aggregation statistic is skipped, leaving a shelf-rooted circuit
    that models the shelf's own dimensions but nothing about what it holds.
    """

    @aggregation_statistic(variable(RelationalCircuitExperimentShelf).layers)
    def layer_count(self) -> Query[int]:
        """
        Number of layers the shelf has.
        """
        return entity(count(flat_variable(self.layers)))
