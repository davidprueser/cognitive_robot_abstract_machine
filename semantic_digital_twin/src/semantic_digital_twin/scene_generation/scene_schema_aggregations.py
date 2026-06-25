from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from krrood.entity_query_language.factories import count, variable
from krrood.parametrization.feature_extraction.aggregations import AggregationStatistic, aggregation_statistic
from semantic_digital_twin.scene_generation.scene_schema import EGShelf, EGRoom, EGShelfLayer


@dataclass
class EGShelfAggregations(AggregationStatistic[EGShelf]):
    """
    Aggregation statistics over the layers of a shelf.
    """

    @aggregation_statistic("layers")
    def total_count(self) -> int:
        """
        Number of layers in the shelf.
        """
        [layer_count] = count(variable(EGShelf, self.instance.layers)).tolist()
        return layer_count


@dataclass
class RoomAggregations(AggregationStatistic[EGRoom]):
    """
    Aggregation statistics over the objects, walls, and doors in a room.
    """

    @aggregation_statistic("objects")
    def total_count(self) -> int:
        """
        Total number of objects.
        """
        [object_count] = count(variable(EGRoom, self.instance.objects)).tolist()
        return object_count

    @aggregation_statistic("walls")
    def wall_count(self) -> int:
        """
        Number of walls enclosing the room.
        """
        [wall_count] = count(variable(EGRoom, self.instance.walls)).tolist()
        return wall_count

    @aggregation_statistic("walls")
    def total_perimeter(self) -> float:
        """
        Sum of all wall lengths — equals the room's floor perimeter.
        """
        return float(
            sum(
                math.sqrt(
                    (w.end_point.x - w.start_point.x) ** 2
                    + (w.end_point.y - w.start_point.y) ** 2
                )
                for w in self.instance.walls
            )
        )

    @aggregation_statistic("doors")
    def door_count(self) -> int:
        """
        Number of doors in the room.
        """
        [door_count] = count(variable(EGRoom, self.instance.doors)).tolist()
        return door_count

    @aggregation_statistic("doors")
    def mean_width(self) -> float:
        """
        Mean door width across all doors in the room.
        """
        return float(np.mean([d.width for d in self.instance.doors]))


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
