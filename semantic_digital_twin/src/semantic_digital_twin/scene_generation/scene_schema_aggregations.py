from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from krrood.entity_query_language.factories import count, variable
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
)


@dataclass
class EGShelfLayerAggregations(AggregationStatistic[EGShelfLayer]):
    """
    Aggregation statistics over the objects on an EGShelfLayer.
    """

    @aggregation_statistic("objects")
    def total_count(self) -> int:
        """
        Number of objects placed on the shelf layer.

        ..note:: The variable is declared over the *element* type. Naming the
            owning type instead matches nothing and counts zero, which is silent:
            the statistic still appears in the fitted model, so the layer and its
            objects merely end up independent.
        """
        [object_count] = count(variable(EGObject2D, self.instance.objects)).tolist()
        return object_count


@dataclass
class EGShelfAggregations(AggregationStatistic[EGShelf]):
    """
    Aggregation statistics over the layers of an EGShelf.

    Declaring these is what makes a shelf's layers visible to fitting at all: a
    collection with no aggregation statistic is skipped, leaving a shelf-rooted
    circuit that models the shelf's own dimensions but nothing about what it
    holds.
    """

    @aggregation_statistic("layers")
    def layer_count(self) -> int:
        """
        Number of layers the shelf has.
        """
        [count_of_layers] = count(variable(EGShelfLayer, self.instance.layers)).tolist()
        return count_of_layers
