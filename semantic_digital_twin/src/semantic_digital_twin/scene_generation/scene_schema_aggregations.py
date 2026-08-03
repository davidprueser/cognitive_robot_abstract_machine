from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from krrood.entity_query_language.factories import count, variable
from krrood.parametrization.feature_extraction.aggregations import AggregationStatistic, aggregation_statistic
from semantic_digital_twin.scene_generation.scene_schema import (
    EGShelf,
    EGRoom,
    EGRoomFloorLayout,
    EGShelfLayer,
    EGTableWithChairs,
)


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

    @aggregation_statistic("shelves")
    def shelf_count(self) -> int:
        """
        Number of shelves in the room.
        """
        [shelf_count] = count(variable(EGRoom, self.instance.shelves)).tolist()
        return shelf_count

    @aggregation_statistic("tables")
    def table_count(self) -> int:
        """
        Number of table-with-chairs groups in the room.
        """
        [table_count] = count(variable(EGRoom, self.instance.tables)).tolist()
        return table_count


@dataclass
class EGRoomFloorLayoutAggregations(AggregationStatistic[EGRoomFloorLayout]):
    """
    Aggregation statistics over the floor pieces of a room floor layout.

    Without this class, ``pieces`` has no registered
    :class:`AggregationStatistic`, so :class:`RelationalProbabilisticCircuit`
    silently treats it as non-exchangeable: it never fits a distribution
    template for it, and every piece's own attributes are then left
    unresolved (still the query's placeholder value) after sampling.
    """

    @aggregation_statistic("pieces")
    def total_count(self) -> int:
        """
        Number of floor pieces in the layout.
        """
        [piece_count] = count(
            variable(EGRoomFloorLayout, self.instance.pieces)
        ).tolist()
        return piece_count

    @aggregation_statistic("pieces")
    def floor_area(self) -> float:
        """
        Area of the room floor the pieces are arranged on.

        An aggregation is the only channel carrying room-level context into the
        per-piece distribution, so a piece conditioned solely on
        :meth:`total_count` cannot tell a cramped room from a hall. Reads the
        layout's own scale rather than the pieces, so it stays determinable from
        a query whose pieces are still free.
        """
        return float(self.instance.scale.width * self.instance.scale.length)

    @aggregation_statistic("pieces")
    def aspect_ratio(self) -> float:
        """
        Ratio of the floor's width to its length.

        Complements :meth:`floor_area`: together they are an invertible
        reparametrisation of the footprint, and expressing piece positions as
        fractions of each room axis removes shape from the coordinates
        themselves, leaving this statistic as the only carrier of it.
        """
        return float(self.instance.scale.width / self.instance.scale.length)


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


@dataclass
class EGTableWithChairsAggregations(AggregationStatistic[EGTableWithChairs]):
    """
    Aggregation statistics over the chairs surrounding a table.
    """

    @aggregation_statistic("chairs")
    def total_count(self) -> int:
        """
        Number of chairs surrounding the table.
        """
        [chair_count] = count(
            variable(EGTableWithChairs, self.instance.chairs)
        ).tolist()
        return chair_count
