from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import NoSolutionFound
from random_events.product_algebra import Event

from experiments.scene_generation_experiments.in_world_resolver import (
    minimal_resample_set,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    build_free_space_conditioned_layer_query,
    evaluate_first_supported,
    probabilistic_backend,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    MeshCandidate,
    ShelfLayerGeometry,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.world_description.geometry import (
    Scale,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    VolumetricGraphOfBoundingBoxes,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass
class PreSpawnLayerGroup:
    """
    The objects sampled onto one shelf layer that must not collide with each other,
    resampled against that layer's learned distribution and kept in bounds -- all before
    any mesh is loaded or any body is spawned.

    Mirrors :class:`~experiments.scene_generation_experiments.in_world_resolver.
    ShelfLayerGroup`, but checks and repairs collisions on bounding boxes built from
    each matched mesh candidate's own real extents (falling back to the RSPN-sampled
    scale when a candidate's size is unknown), rather than on real spawned meshes
    through an :class:`~semantic_digital_twin.collision_checking.
    trimesh_collision_detector.FCLCollisionDetector`.
    """

    objects: dict[int, EGObject2D]
    """
    The sampled objects to keep collision-free, keyed by their index in the owning
    layer's objects -- only those a mesh was matched for.
    """

    footprints: dict[int, Scale]
    """
    Each member's real-world footprint to bound and collide on, keyed the same as
    :attr:`objects` -- the matched candidate's own extents, or the sampled ``scale``
    when the candidate's size is unknown.
    """

    shelf: EGShelf
    """
    The shelf whose layer this group belongs to; its objects are mutated in place as
    they are resampled.
    """

    layer_index: int
    """
    Index of this group's layer in :attr:`EGShelf.layers`.
    """

    corpus: KinematicStructureEntity
    """
    The already-spawned shelf corpus every member's bounding box is expressed
    relative to.
    """

    backend: ProbabilisticBackend = field(kw_only=True)
    """
    The single-sample backend over this layer's fitted circuit, from which offending
    object poses are redrawn.
    """

    def clamp_to_bounds(self) -> None:
        """
        Move any member positioned outside this layer's own footprint back to its
        nearest in-bounds position.

        Pure pose arithmetic, mirroring :meth:`ShelfLayerGroup.clamp_to_bounds` -- run
        before every collision check for the same reason: an RSPN sample landing
        outside the layer is a plain, cheap geometric fix, not a collision-style
        violation worth an expensive resample.
        """
        half_x = self.shelf.scale.x / 2
        half_y = self.shelf.scale.y / 2
        for index, object_2d in self.objects.items():
            footprint = self.footprints[index]
            max_x = max(half_x - footprint.x / 2, 0.0)
            max_y = max(half_y - footprint.y / 2, 0.0)
            clamped_x = min(max(float(object_2d.pose.x), -max_x), max_x)
            clamped_y = min(max(float(object_2d.pose.y), -max_y), max_y)
            if clamped_x == float(object_2d.pose.x) and clamped_y == float(
                object_2d.pose.y
            ):
                continue
            object_2d.pose = Pose2D(x=clamped_x, y=clamped_y, yaw=object_2d.pose.yaw)

    def colliding_indices(self) -> set[int]:
        """
        Return a minimal set of member indices whose resampling clears every pairwise
        bounding-box overlap among this group's members.

        Compared via :attr:`~semantic_digital_twin.world_description.geometry.
        AxisAlignedBox.simple_event` rather than :meth:`~semantic_digital_twin.
        world_description.geometry.AxisAlignedBox.intersection_with`: every member
        already shares :attr:`corpus` as its box's own reference frame, so no cross-
        frame transform is needed, and ``intersection_with``'s corner-enumeration
        transform corrupts a box extended to an infinite z-column (``inf`` corners
        multiplied through a transform matrix produce ``nan``) -- ``simple_event``'s
        plain per-axis interval stays exact for an unbounded axis.

        :return: Indices whose bodies must be resampled to remove all box overlaps.
        """
        events = {
            index: obj.local_bounding_box(
                self.footprints[index], self.corpus
            ).simple_event.as_composite_set()
            for index, obj in self.objects.items()
        }
        colliding_pairs: set[tuple[int, int]] = set()
        for (index_a, event_a), (index_b, event_b) in combinations(events.items(), 2):
            if not (event_a & event_b).is_empty():
                colliding_pairs.add(tuple(sorted((index_a, index_b))))
        return minimal_resample_set(colliding_pairs)

    def _search_space_event(self) -> Event:
        """
        This group's layer footprint, as an event centred on the corpus frame's own
        origin -- the same convention :meth:`~semantic_digital_twin.scene_generation.
        scene_schema.EGShelf.object_local_pose` places every member's pose in.
        """
        half_x = self.shelf.scale.x / 2
        half_y = self.shelf.scale.y / 2
        origin = HomogeneousTransformationMatrix(reference_frame=self.corpus)
        search_space = VolumetricBoundingBox(
            -half_x, -half_y, -math.inf, half_x, half_y, math.inf, origin
        )
        return search_space.simple_event.as_composite_set()

    def _bloated_bounding_box(
        self, index: int, object_bloat: float
    ) -> VolumetricBoundingBox:
        """
        *index*'s own bounding box, enlarged by *object_bloat* -- the standard trick
        that lets the redrawn member be treated as a single point: bloating every
        obstacle by the mover's own half-size keeps the mover's whole footprint clear
        wherever that point lands.
        """
        box = self.objects[index].local_bounding_box(
            self.footprints[index], self.corpus
        )
        box.enlarge_all(object_bloat)
        return box

    def resample_and_move(self, indices: set[int]) -> None:
        """
        Redraw each of *indices*' poses, truncated to the free space its layer's
        bounding boxes actually have left, one at a time so each newly placed member is
        real evidence the next member's free-space calculation already sees.

        Mirrors :meth:`ShelfLayerGroup.resample_and_move`, but the free space comes from
        :attr:`footprints`-derived bounding boxes rather than
        :meth:`~semantic_digital_twin.semantic_annotations.mixins.
        HasSupportingSurface.calculate_free_space`'s real-mesh version.

        :param indices: Indices of this group's members to redraw.
        """
        layer = self.shelf.layers[self.layer_index]
        fixed_indices = [index for index in self.objects if index not in indices]
        search_space_event = self._search_space_event()

        for index in sorted(indices):
            object_2d = self.objects[index]
            object_bloat = max(self.footprints[index].x, self.footprints[index].y) / 2
            obstacle_boxes = BoundingBoxCollection(
                [
                    self._bloated_bounding_box(other_index, object_bloat)
                    for other_index in fixed_indices
                ],
                self.corpus,
            )
            free_space_event = (
                VolumetricGraphOfBoundingBoxes.free_space_from_bounding_boxes(
                    obstacle_boxes, search_space_event
                )
            )
            fixed_objects = [self.objects[other] for other in fixed_indices]

            try:
                redrawn_layer = evaluate_first_supported(
                    self.backend,
                    build_free_space_conditioned_layer_query(
                        layer.theme_dominant_type,
                        fixed_objects,
                        object_2d,
                        free_space_event,
                    ),
                    build_free_space_conditioned_layer_query(
                        layer.theme_dominant_type, [], object_2d, free_space_event
                    ),
                )
            except NoSolutionFound:
                fixed_indices.append(index)
                continue
            redrawn = redrawn_layer.objects[-1]

            object_2d.pose = redrawn.pose
            fixed_indices.append(index)


@dataclass
class PreSpawnLayoutResolver:
    """
    Repairs a sampled shelf's layout before anything is spawned, by moving offending
    members' poses, until every layer's matched objects are free of pairwise bounding-
    box overlap.

    Mirrors :class:`~experiments.scene_generation_experiments.in_world_resolver.
    InWorldLayoutResolver`'s repair loop, but drives it from cheap, in-memory bounding
    boxes -- built from each matched mesh candidate's own real extents -- instead of a
    real, spawned :class:`~semantic_digital_twin.world_description.world_entity.Body`
    and an :class:`~semantic_digital_twin.collision_checking.trimesh_collision_detector.
    FCLCollisionDetector`. A caller spawns the surviving :attr:`matches` afterwards
    (:meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.spawn_objects`);
    a lightweight, real-mesh safety net still belongs after that, since a matched
    candidate's own extents are only an approximation of the mesh's true geometry.
    """

    shelf: EGShelf
    """
    The shelf whose corpus and slabs are already spawned, and whose layout is being
    resolved before its objects are.
    """

    groups: list[PreSpawnLayerGroup]
    """
    One collision group per shelf layer.
    """

    matches: dict[int, dict[int, MeshCandidate]]
    """
    Per layer index, per object index, the mesh candidate to spawn -- narrowed by
    :meth:`resolve` as members are dropped.
    """

    dropped_object_count: int = field(default=0, init=False)
    """
    Matched objects removed by :meth:`resolve` because no repair pass could place them.
    """

    max_passes: int = 10
    """
    Upper bound on repair passes before giving up on an unsatisfiable layout.
    """

    stuck_after_passes: int = 3
    """
    Consecutive passes a member may remain in violation, unresolved, before it stops
    being resampled and is left for the final drop instead.
    """

    @classmethod
    def for_shelf(
        cls,
        shelf: EGShelf,
        rspn: RelationalProbabilisticCircuit,
        corpus_body: KinematicStructureEntity,
        layer_geometries: list[ShelfLayerGeometry],
        max_passes: int = 10,
        stuck_after_passes: int = 3,
    ) -> PreSpawnLayoutResolver:
        """
        Match meshes for *shelf*'s layers and build one collision group per layer.

        :param shelf: The shelf whose corpus and slabs are already spawned
            (:meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
            _spawn_corpus_and_slabs`), but whose objects are not.
        :param rspn: The fitted circuit used to redraw offending object poses.
        :param corpus_body: The already-spawned shelf corpus.
        :param layer_geometries: *shelf*'s own
            :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
            layer_geometries`.
        :param max_passes: Upper bound on repair passes.
        :param stuck_after_passes: Consecutive passes a member may remain in violation
            before it stops being resampled.
        :return: A resolver ready to repair the sampled layout.
        """
        matches = shelf.match_meshes(layer_geometries)
        backend = probabilistic_backend(rspn)
        groups = [
            cls._layer_group(
                shelf, layer_index, matches.get(layer_index, {}), corpus_body, backend
            )
            for layer_index in range(len(shelf.layers))
        ]
        return cls(
            shelf=shelf,
            groups=groups,
            matches=matches,
            max_passes=max_passes,
            stuck_after_passes=stuck_after_passes,
        )

    @staticmethod
    def _layer_group(
        shelf: EGShelf,
        layer_index: int,
        layer_matches: dict[int, MeshCandidate],
        corpus_body: KinematicStructureEntity,
        backend: ProbabilisticBackend,
    ) -> PreSpawnLayerGroup:
        layer = shelf.layers[layer_index]
        return PreSpawnLayerGroup(
            objects={index: layer.objects[index] for index in layer_matches},
            footprints={
                index: layer_matches[index].footprint_scale
                or layer.objects[index].scale
                for index in layer_matches
            },
            shelf=shelf,
            layer_index=layer_index,
            corpus=corpus_body,
            backend=backend,
        )

    def resolve(self) -> dict[int, dict[int, MeshCandidate]]:
        """
        Repair every group until all are free of bounding-box overlap, moving offending
        members' poses in place, dropping stragglers from :attr:`matches` after
        :attr:`max_passes`.

        :return: The narrowed :attr:`matches`, ready for
            :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.
            spawn_objects`.
        """
        stuck_counts: dict[tuple[int, int], int] = {}
        for _ in range(self.max_passes):
            self._clamp_groups_to_bounds()
            remaining = self._remaining_violations()
            if not remaining:
                return self.matches
            to_resample, stuck_counts = self._resamplable(remaining, stuck_counts)
            if not to_resample:
                break
            for group_index, violations in to_resample.items():
                self.groups[group_index].resample_and_move(violations)

        self._clamp_groups_to_bounds()
        remaining = self._remaining_violations()
        if remaining:
            self._drop(remaining)
        return self.matches

    def _clamp_groups_to_bounds(self) -> None:
        for group in self.groups:
            group.clamp_to_bounds()

    def _remaining_violations(self) -> dict[int, set[int]]:
        return {
            group_index: violations
            for group_index, group in enumerate(self.groups)
            if (violations := group.colliding_indices())
        }

    def _resamplable(
        self,
        remaining: dict[int, set[int]],
        stuck_counts: dict[tuple[int, int], int],
    ) -> tuple[dict[int, set[int]], dict[tuple[int, int], int]]:
        """
        Split *remaining* into members still worth resampling and an updated stuck-pass
        count for each, mirroring
        :meth:`~experiments.scene_generation_experiments.in_world_resolver.

        InWorldLayoutResolver._resamplable`.
        """
        updated_counts: dict[tuple[int, int], int] = {}
        to_resample: dict[int, set[int]] = {}
        for group_index, violations in remaining.items():
            resamplable = set()
            for member_index in violations:
                key = (group_index, member_index)
                count = stuck_counts.get(key, 0) + 1
                updated_counts[key] = count
                if count <= self.stuck_after_passes:
                    resamplable.add(member_index)
            if resamplable:
                to_resample[group_index] = resamplable
        return to_resample, updated_counts

    def _drop(self, offenders: dict[int, set[int]]) -> None:
        """
        Remove *offenders* from their groups and from :attr:`matches`, so a layout that
        cannot be packed is spawned without the objects that do not fit rather than not
        at all.

        :param offenders: Offending member indices per group index.
        """
        for group_index, indices in offenders.items():
            group = self.groups[group_index]
            layer_matches = self.matches.get(group.layer_index, {})
            for index in indices:
                group.objects.pop(index, None)
                group.footprints.pop(index, None)
                layer_matches.pop(index, None)
                self.dropped_object_count += 1
