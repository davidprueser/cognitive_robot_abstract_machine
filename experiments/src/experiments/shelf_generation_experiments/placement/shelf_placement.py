from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType

from experiments.shelf_generation_experiments.placement.exceptions import (
    NoShelfPlacementError,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel


# %% placement
@dataclass
class Placement:
    """
    Where on a shelf an object belongs.
    """

    placed_object: RelationalCircuitExperimentObject2D
    """
    The object at the pose it belongs at, relative to the shelf corpus.
    """

    layer: RelationalCircuitExperimentShelfLayer
    """
    The layer the object belongs on.
    """

    log_density: float
    """
    The log-density of the placement under the layer circuit, comparable across layers.
    """


@dataclass
class ShelfPlacement:
    """
    Answers where on a spawned shelf an object most likely belongs, by asking the layer
    circuit of a shelf model about every layer separately and keeping the most likely
    answer.

    .. note::
        What the object is decides where on a layer it goes, not which layer: the
        circuit passes only aggregation statistics from a layer to its objects, so the
        layers are told apart by how typical their own attributes and free space are.
    """

    shelf: RelationalCircuitExperimentShelf
    """
    The spawned shelf to place onto.
    """

    model: ShelfModel
    """
    The shelf model the layers are asked about.
    """

    def most_likely_placement(
        self,
        held_object: RelationalCircuitExperimentObject2D,
        yaw: float | EllipsisType = ...,
    ) -> Placement:
        """
        :param held_object: The object to place; its type and scale are held as
            evidence.
        :param yaw: The yaw to place the object at, or ``...`` to leave it to the
            circuit, whose yaw distribution is close to uniform.
        :raises NoShelfPlacementError: If no layer has room for the object.
        :return: The most likely placement over all layers.
        """
        placements = [
            placement
            for layer in self.shelf.layers
            if (placement := self._placement_on(layer, held_object, yaw)) is not None
        ]
        if not placements:
            raise NoShelfPlacementError(
                shelf_name=str(self.shelf.corpus.name),
                object_type=held_object.object_type,
            )
        return max(placements, key=lambda placement: placement.log_density)

    def _placement_on(
        self,
        layer: RelationalCircuitExperimentShelfLayer,
        held_object: RelationalCircuitExperimentObject2D,
        yaw: float | EllipsisType,
    ) -> Placement | None:
        """
        :param layer: The layer to place onto.
        :param held_object: The object to place.
        :param yaw: See :meth:`most_likely_placement`.
        :return: The most likely placement on *layer*, or ``None`` if it has no room for
            the object.
        """
        free_space = layer.annotation.planar_free_space(
            max_height=held_object.scale.z,
            bloat_obstacles=max(held_object.scale.x, held_object.scale.y) / 2,
        )
        if not free_space.graph.nodes():
            return None
        standing_objects = [
            object_ for object_ in layer.objects if object_.annotation is not None
        ]
        mode = self.model.layer_backend().evaluate_mode(
            layer.placement_query(
                standing_objects, held_object.placement_query(yaw), free_space
            )
        )
        if mode is None:
            return None
        return Placement(
            placed_object=mode.instance.objects[-1],
            layer=layer,
            log_density=mode.log_density,
        )
