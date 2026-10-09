from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType


# %% object type coarsening
@dataclass
class ObjectTypeCoarsening:
    """
    The object types and shelf themes a shelf model tells apart; every other type is
    coarsened to :attr:`ObjectType.OTHER`.

    Objects and themes are counted separately: a type common among objects is not
    necessarily common as the dominant type of a shelf.
    """

    frequent_object_types: set[ObjectType]
    """
    The object types left unchanged on objects.
    """

    frequent_theme_types: set[ObjectType]
    """
    The object types left unchanged as the theme of a shelf and its layers.
    """

    @classmethod
    def from_shelves(
        cls, shelves: Iterable[RelationalCircuitExperimentShelf], keep_count: int = 20
    ) -> ObjectTypeCoarsening:
        """
        :param shelves: The shelves whose objects and themes are counted.
        :param keep_count: How many of the most frequent object types, and of the most
            frequent themes, are kept.
        :return: The coarsening keeping the most frequent types of *shelves*.
        """
        shelves = list(shelves)
        object_type_counts = Counter(
            object_.object_type
            for shelf in shelves
            for layer in shelf.layers
            for object_ in layer.objects
        )
        theme_counts = Counter(shelf.theme_dominant_type for shelf in shelves)
        return cls(
            frequent_object_types={
                object_type
                for object_type, _ in object_type_counts.most_common(keep_count)
            },
            frequent_theme_types={
                theme for theme, _ in theme_counts.most_common(keep_count)
            },
        )

    @property
    def placeable_object_types(self) -> set[ObjectType]:
        """
        The types kept both as object types and as themes, excluding
        :attr:`ObjectType.OTHER`, which stands for whatever was coarsened rather than
        for a category of its own.
        """
        return (self.frequent_object_types & self.frequent_theme_types) - {
            ObjectType.OTHER
        }

    def coarsen_shelves(
        self, shelves: Iterable[RelationalCircuitExperimentShelf]
    ) -> list[RelationalCircuitExperimentShelf]:
        """
        :param shelves: The shelves to coarsen.
        :return: Copies of *shelves* whose themes, on the shelf and on every layer, and
            whose object types are coarsened.
        """
        return [self._coarsened_shelf(shelf) for shelf in shelves]

    def coarsen_mesh_candidates(
        self, candidates: Iterable[MeshCandidate]
    ) -> list[MeshCandidate]:
        """
        :param candidates: The mesh candidates to coarsen.
        :return: Copies of *candidates* whose object types are coarsened, so their labels
            match the types the shelf model samples.
        """
        return [
            dataclasses.replace(
                candidate,
                object_type=self._coarsened_object_type(candidate.object_type),
            )
            for candidate in candidates
        ]

    def stored_object_types_of(
        self, shelf: RelationalCircuitExperimentShelf
    ) -> set[ObjectType]:
        """
        :param shelf: A shelf sampled from the shelf model.
        :return: The stored object types whose meshes can dress *shelf*; a sampled
            :attr:`ObjectType.OTHER` stands for every type this coarsening does not keep.
        """
        sampled_types = {
            object_.object_type for layer in shelf.layers for object_ in layer.objects
        }
        if ObjectType.OTHER not in sampled_types:
            return sampled_types
        return (sampled_types - {ObjectType.OTHER}) | (
            set(ObjectType) - self.frequent_object_types
        )

    def _coarsened_object_type(self, object_type: ObjectType) -> ObjectType:
        """
        :param object_type: The type of an object.
        :return: *object_type* if it is kept, :attr:`ObjectType.OTHER` otherwise.
        """
        if object_type in self.frequent_object_types:
            return object_type
        return ObjectType.OTHER

    def _coarsened_theme(self, theme: ObjectType) -> ObjectType:
        """
        :param theme: The theme of a shelf.
        :return: *theme* if it is kept, :attr:`ObjectType.OTHER` otherwise.
        """
        if theme in self.frequent_theme_types:
            return theme
        return ObjectType.OTHER

    def _coarsened_shelf(
        self, shelf: RelationalCircuitExperimentShelf
    ) -> RelationalCircuitExperimentShelf:
        """
        :param shelf: The shelf to coarsen.
        :return: A copy of *shelf* with its theme and its layers coarsened.
        """
        theme = self._coarsened_theme(shelf.theme_dominant_type)
        return dataclasses.replace(
            shelf,
            theme_dominant_type=theme,
            layers=[self._coarsened_layer(layer, theme) for layer in shelf.layers],
        )

    def _coarsened_layer(
        self, layer: RelationalCircuitExperimentShelfLayer, theme: ObjectType
    ) -> RelationalCircuitExperimentShelfLayer:
        """
        :param layer: The layer to coarsen.
        :param theme: The coarsened theme of the layer's shelf.
        :return: A copy of *layer* carrying *theme* and coarsened objects.
        """
        return dataclasses.replace(
            layer,
            theme_dominant_type=theme,
            objects=[self._coarsened_object(object_) for object_ in layer.objects],
        )

    def _coarsened_object(
        self, object_: RelationalCircuitExperimentObject2D
    ) -> RelationalCircuitExperimentObject2D:
        """
        :param object_: The object to coarsen.
        :return: A copy of *object_* with its type coarsened.
        """
        return dataclasses.replace(
            object_, object_type=self._coarsened_object_type(object_.object_type)
        )
