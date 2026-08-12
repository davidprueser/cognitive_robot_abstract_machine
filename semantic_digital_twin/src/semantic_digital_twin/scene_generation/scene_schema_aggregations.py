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
        """
        [object_count] = count(variable(EGShelfLayer, self.instance.objects)).tolist()
        return object_count
