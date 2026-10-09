from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from experiments.shelf_generation_experiments.generation.colliding_pairs import (
    CollidingPairs,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel
from krrood.entity_query_language.backends import ProbabilisticBackend
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.world_description.geometry import (
    PlanarBoundingBox,
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    PlanarGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


# %% one layer
@dataclass
class PreSpawnLayerGroup:
    """
    The objects sampled onto one shelf layer, kept free of overlap with each other and
    inside the layer before any of their meshes is loaded.

    Overlap is judged on axis-aligned footprints built from each matched mesh's real
    size, falling back to the sampled scale where that size is unknown.
    """

    objects: dict[int, RelationalCircuitExperimentObject2D]
    """
    The objects a mesh was matched for, keyed by their index in the layer's objects.
    Their poses are changed in place as they are resampled.
    """

    footprints: dict[int, Scale]
    """
    The real size of each object of :attr:`objects`, keyed the same way.
    """

    layer: RelationalCircuitExperimentShelfLayer
    """
    The layer the objects stand on.
    """

    shelf_scale: Scale
    """
    The scale of the shelf, whose width and depth every layer spans.
    """

    corpus: KinematicStructureEntity
    """
    The spawned shelf corpus every footprint is expressed in.
    """

    backend: ProbabilisticBackend = field(kw_only=True)
    """
    The backend over the layer circuit from which resampled poses are drawn.
    """

    def clamp_to_bounds(self) -> None:
        """
        Move every object whose footprint leaves the layer back to the nearest position
        where it fits.
        """
        half_x = self.shelf_scale.x / 2
        half_y = self.shelf_scale.y / 2
        for index, object_ in self.objects.items():
            footprint = self.footprints[index]
            maximum_x = max(half_x - footprint.x / 2, 0.0)
            maximum_y = max(half_y - footprint.y / 2, 0.0)
            x = float(object_.pose.x)
            y = float(object_.pose.y)
            clamped_x = min(max(x, -maximum_x), maximum_x)
            clamped_y = min(max(y, -maximum_y), maximum_y)
            if (clamped_x, clamped_y) == (x, y):
                continue
            object_.pose = Pose2D(x=clamped_x, y=clamped_y, yaw=object_.pose.yaw)

    def colliding_indices(self) -> set[int]:
        """
        :return: A small set of object indices whose resampling clears every overlap of
            two footprints.
        """
        footprint_events = {
            index: object_.local_bounding_box(
                self.footprints[index], self.corpus
            ).simple_event.as_composite_set()
            for index, object_ in self.objects.items()
        }
        colliding_pairs = CollidingPairs()
        for (first_index, first_event), (second_index, second_event) in combinations(
            footprint_events.items(), 2
        ):
            if not (first_event & second_event).is_empty():
                colliding_pairs.add(first_index, second_index)
        return colliding_pairs.minimal_resample_set()

    def resample(self, indices: set[int]) -> None:
        """
        Redraw the pose of every object of *indices* inside the free space the others
        leave, one at a time, so every redrawn object is an obstacle to the next.

        An object without any free space left, or whose redraw the layer circuit gives
        no support, keeps its pose.

        :param indices: The indices of the objects to redraw.
        """
        settled_indices = [index for index in self.objects if index not in indices]
        for index in sorted(indices):
            free_space = self._free_space_for(index, settled_indices)
            settled_indices.append(index)
            if not free_space.graph.nodes():
                continue
            object_ = self.objects[index]
            redrawn_layer = self.backend.generate_one(
                self.layer.placement_query(
                    [self.objects[other] for other in settled_indices[:-1]],
                    object_.placement_query(),
                    free_space,
                )
            )
            if redrawn_layer is None:
                continue
            object_.pose = redrawn_layer.objects[-1].pose

    def _free_space_for(
        self, index: int, obstacle_indices: list[int]
    ) -> PlanarGraphOfBoundingBoxes:
        """
        :param index: The index of the object to place.
        :param obstacle_indices: The indices of the objects it must not overlap.
        :return: Where on the layer the centre of the object can go without its
            footprint overlapping those of the obstacles.
        """
        object_radius = max(self.footprints[index].x, self.footprints[index].y) / 2
        obstacles = BoundingBoxCollection(
            [
                self._enlarged_footprint(obstacle_index, object_radius)
                for obstacle_index in obstacle_indices
            ],
            self.corpus,
        )
        free_space_event = PlanarGraphOfBoundingBoxes.free_space_from_bounding_boxes(
            obstacles, self._layer_area().simple_event.as_composite_set()
        )
        free_space = PlanarGraphOfBoundingBoxes(world=self.corpus._world)
        for box in BoundingBoxCollection.from_event(
            PlanarBoundingBox, self.corpus, free_space_event
        ):
            free_space.add_node(box)
        return free_space

    def _layer_area(self) -> VolumetricBoundingBox:
        """
        :return: The area of the layer in the corpus frame, extended to all heights.
        """
        half_x = self.shelf_scale.x / 2
        half_y = self.shelf_scale.y / 2
        return VolumetricBoundingBox(
            -half_x,
            -half_y,
            -math.inf,
            half_x,
            half_y,
            math.inf,
            HomogeneousTransformationMatrix(reference_frame=self.corpus),
        )

    def _enlarged_footprint(self, index: int, margin: float) -> VolumetricBoundingBox:
        """
        :param index: The index of the object whose footprint is enlarged.
        :param margin: How far to enlarge it in every direction.
        :return: The footprint of the object, enlarged by *margin*.
        """
        footprint = self.objects[index].local_bounding_box(
            self.footprints[index], self.corpus
        )
        footprint.enlarge_all(margin)
        return footprint


# %% whole shelf
@dataclass
class PreSpawnLayoutResolver:
    """
    Repairs the layout of a sampled shelf before its objects are spawned, by redrawing
    the poses of overlapping objects from the fitted layer circuit, and drops whatever
    still overlaps after :attr:`maximum_pass_count` repair passes.
    """

    groups: list[PreSpawnLayerGroup]
    """
    One group per layer of the shelf.
    """

    matches: dict[int, dict[int, MeshCandidate]]
    """
    Per layer index, per object index, the mesh to spawn; narrowed as objects are
    dropped.
    """

    maximum_pass_count: int = 10
    """
    The most repair passes before the objects still overlapping are dropped.
    """

    maximum_resample_count: int = 3
    """
    The most passes in a row an object is redrawn before it is left for dropping.
    """

    dropped_object_count: int = field(default=0, init=False)
    """
    How many objects :meth:`resolve` dropped.
    """

    @classmethod
    def for_shelf(
        cls,
        shelf: RelationalCircuitExperimentShelf,
        layers: list[RelationalCircuitExperimentShelfLayer],
        model: ShelfModel,
        corpus: KinematicStructureEntity,
    ) -> PreSpawnLayoutResolver:
        """
        Match meshes for the objects of *shelf* and build one group per layer.

        :param shelf: The shelf whose corpus and slabs are spawned, but not its objects.
        :param layers: The layers of *shelf* with their geometry.
        :param model: The shelf model *shelf* was sampled from.
        :param corpus: The spawned corpus of *shelf*.
        :return: The resolver.
        """
        matches = shelf.match_meshes(layers)
        backend = model.layer_backend()
        groups = [
            PreSpawnLayerGroup(
                objects={
                    index: layer.objects[index]
                    for index in matches.get(layer_index, {})
                },
                footprints={
                    index: candidate.scale or layer.objects[index].scale
                    for index, candidate in matches.get(layer_index, {}).items()
                },
                layer=layer,
                shelf_scale=shelf.scale,
                corpus=corpus,
                backend=backend,
            )
            for layer_index, layer in enumerate(shelf.layers)
        ]
        return cls(groups=groups, matches=matches)

    def resolve(self) -> dict[int, dict[int, MeshCandidate]]:
        """
        Repair every layer until no two footprints overlap, dropping the objects that
        still overlap once the repair passes are used up.

        :return: The narrowed :attr:`matches`.
        """
        resample_counts: dict[tuple[int, int], int] = {}
        for _ in range(self.maximum_pass_count):
            remaining = self._clamped_violations()
            if not remaining:
                return self.matches
            resample_counts = self._counted_resamples(remaining, resample_counts)
            resamplable = {
                group_index: {
                    index
                    for index in violations
                    if resample_counts[(group_index, index)]
                    <= self.maximum_resample_count
                }
                for group_index, violations in remaining.items()
            }
            if not any(resamplable.values()):
                break
            for group_index, indices in resamplable.items():
                self.groups[group_index].resample(indices)

        self._drop(self._clamped_violations())
        return self.matches

    def _clamped_violations(self) -> dict[int, set[int]]:
        """
        Clamp every group to its layer and collect what still overlaps.

        :return: Per group index, the indices of the objects to move.
        """
        for group in self.groups:
            group.clamp_to_bounds()
        return {
            group_index: violations
            for group_index, group in enumerate(self.groups)
            if (violations := group.colliding_indices())
        }

    @staticmethod
    def _counted_resamples(
        remaining: dict[int, set[int]], resample_counts: dict[tuple[int, int], int]
    ) -> dict[tuple[int, int], int]:
        """
        :param remaining: Per group index, the indices of the objects still overlapping.
        :param resample_counts: How many passes in a row each object overlapped so far.
        :return: The counts updated for this pass; an object no longer overlapping
            starts over.
        """
        return {
            (group_index, index): resample_counts.get((group_index, index), 0) + 1
            for group_index, violations in remaining.items()
            for index in violations
        }

    def _drop(self, offenders: dict[int, set[int]]) -> None:
        """
        Remove *offenders* from their groups and from :attr:`matches`.

        :param offenders: Per group index, the indices of the objects to drop.
        """
        for group_index, indices in offenders.items():
            group = self.groups[group_index]
            layer_matches = self.matches.get(group_index, {})
            for index in indices:
                group.objects.pop(index)
                group.footprints.pop(index)
                layer_matches.pop(index)
                self.dropped_object_count += 1
