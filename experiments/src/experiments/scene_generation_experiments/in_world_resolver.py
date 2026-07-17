from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from experiments.scene_generation_experiments.collision_resolution import (
    build_pose_resample_query,
    in_world_colliding_indices,
)
from experiments.scene_generation_experiments.exceptions import LayoutResolutionError
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_chair_pose_resample_query,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import NoSolutionFound
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.scene_generation.scene_schema import (
    EGShelf,
    EGTableWithChairs,
    SpawnedLayout,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


def _evaluate_with_relaxed_fallback(
    backend: ProbabilisticBackend, query, relaxed_query
):
    """
    Evaluate *query*, falling back to *relaxed_query* if the RSPN has no
    support for it.

    Conditioning a resample on every already-placed neighbour's exact pose can
    pin the query to a region the fitted circuit assigns zero probability
    mass -- the neighbours' poses drift further from the training
    distribution with each repair pass, so this becomes more likely deeper
    into a repair. Retrying once without that neighbour evidence keeps the
    repair loop making progress instead of aborting the whole layout.

    :param backend: The backend to evaluate both queries against.
    :param query: The primary, neighbour-conditioned query.
    :param relaxed_query: The same query with neighbour evidence dropped.
    :return: The first sample from whichever query found a solution.
    """
    try:
        return next(iter(backend.evaluate(query)))
    except NoSolutionFound:
        return next(iter(backend.evaluate(relaxed_query)))


@dataclass
class SpawnedCollisionGroup(ABC):
    """
    A set of spawned bodies that must not collide with each other, optionally
    resting on a shared supporting body.

    Each generator produces its own concrete group that knows how to redraw its
    offending members' poses from the RSPN and move the corresponding bodies in
    place; the resolver treats every generator uniformly through this interface.
    """

    bodies: dict[int, Body]
    """
    The movable bodies to keep collision-free, keyed by their index in the
    owning collection.
    """

    supporting_body: Body | None
    """
    The body the members must rest on, or ``None`` when the members are not
    checked for support (e.g. chairs standing on the floor).
    """

    static_obstacles: tuple[Body, ...] = field(default=(), kw_only=True)
    """
    Fixed bodies the members must not collide with, beyond each other (e.g. a
    shelf's corpus walls). Empty when the group has none.
    """

    def unsupported_indices(self) -> set[int]:
        """
        Return the indices of members that their supporting body does not
        support; empty when the group has no supporting body.
        """
        if self.supporting_body is None:
            return set()
        return {
            index
            for index, body in self.bodies.items()
            if not is_supported_by(body, self.supporting_body)
        }

    @abstractmethod
    def resample_and_move(
        self, indices: set[int], backend: ProbabilisticBackend
    ) -> None:
        """
        Redraw the poses of the members at *indices* from the RSPN, holding every
        other member fixed, then move the corresponding bodies to those poses.
        """


@dataclass
class ShelfLayerGroup(SpawnedCollisionGroup):
    """
    The objects on one shelf layer, resampled against that layer's learned
    distribution and re-seated at their existing resting height.
    """

    shelf: EGShelf
    """
    The shelf whose layer this group belongs to; its objects are mutated in
    place as they are resampled.
    """

    layer_index: int
    """
    Index of this group's layer in :attr:`EGShelf.layers`.
    """

    parent: KinematicStructureEntity
    """
    The frame the object bodies' poses are expressed relative to.
    """

    def resample_and_move(
        self, indices: set[int], backend: ProbabilisticBackend
    ) -> None:
        layer = self.shelf.layers[self.layer_index]
        fixed_objects = [
            object_2d
            for object_index, object_2d in enumerate(layer.objects)
            if object_index not in indices and object_index in self.bodies
        ]
        resampled_indices = sorted(indices)
        resampled_objects = [layer.objects[index] for index in resampled_indices]

        new_layer = _evaluate_with_relaxed_fallback(
            backend,
            build_pose_resample_query(
                fixed_objects, len(resampled_objects), layer.scale
            ),
            build_pose_resample_query([], len(resampled_objects), layer.scale),
        )
        redrawn_objects = new_layer.objects[-len(resampled_objects) :]

        for object_index, redrawn in zip(resampled_indices, redrawn_objects):
            object_2d = layer.objects[object_index]
            object_2d.position = redrawn.position
            object_2d.orientation = redrawn.orientation
            body = self.bodies[object_index]
            resting_z = body.global_pose.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_world_pose(
                resting_z, object_2d, self.parent
            )


@dataclass
class ChairGroup(SpawnedCollisionGroup):
    """
    The chairs around one table, resampled against that table's learned
    distribution; chairs stand on the floor, so the group has no supporting body.
    """

    group: EGTableWithChairs
    """
    The table-with-chairs group; its chairs are mutated in place as they are
    resampled.
    """

    parent: KinematicStructureEntity
    """
    The frame the chair bodies' poses are expressed relative to.
    """

    def resample_and_move(
        self, indices: set[int], backend: ProbabilisticBackend
    ) -> None:
        chairs = self.group.chairs
        fixed_chairs = [
            chair
            for chair_index, chair in enumerate(chairs)
            if chair_index not in indices and chair_index in self.bodies
        ]
        resampled_indices = sorted(indices)
        resampled_chairs = [chairs[index] for index in resampled_indices]

        new_sample = _evaluate_with_relaxed_fallback(
            backend,
            build_chair_pose_resample_query(
                fixed_chairs,
                resampled_chairs,
                self.group.position,
                self.group.scale,
                self.group.orientation,
            ),
            build_chair_pose_resample_query(
                [],
                resampled_chairs,
                self.group.position,
                self.group.scale,
                self.group.orientation,
            ),
        )
        redrawn_chairs = new_sample.chairs[-len(resampled_chairs) :]

        for chair_index, redrawn in zip(resampled_indices, redrawn_chairs):
            chair = chairs[chair_index]
            chair.relative_pose = redrawn.relative_pose
            body = self.bodies[chair_index]
            body.parent_connection.origin = self.group.chair_world_pose(
                chair, self.parent
            )


@dataclass
class InWorldLayoutResolver:
    """
    Repairs a spawned layout by validating it directly in its :class:`World` and
    moving offending bodies in place, until every collision group is
    collision-free and supported.

    The layout is spawned once; each repair pass redraws only the pose of
    offending members from the RSPN -- holding their scale, and therefore their
    mesh, fixed -- and moves the corresponding bodies, so meshes are never
    reloaded. Every generator plugs in through :class:`SpawnedCollisionGroup`, so the
    resolver is agnostic to what kind of objects it is arranging.
    """

    spawned: SpawnedLayout
    """
    The already-spawned layout to repair and return.
    """

    groups: list[SpawnedCollisionGroup]
    """
    The collision groups to keep collision-free and supported.
    """

    rspn: RelationalProbabilisticCircuit
    """
    The fitted circuit used to redraw offending members' poses.
    """

    max_passes: int = 50
    """
    Upper bound on repair passes before giving up on an unsatisfiable layout.
    """

    @classmethod
    def for_shelf(
        cls,
        shelf: EGShelf,
        rspn: RelationalProbabilisticCircuit,
        max_passes: int = 50,
    ) -> InWorldLayoutResolver:
        """
        Spawn *shelf* and build one collision group per layer, each supported by
        its own slab and checked against the shelf's own corpus walls.

        :param shelf: The sampled shelf to spawn and repair.
        :param rspn: The fitted circuit used to redraw offending object poses.
        :param max_passes: Upper bound on repair passes.
        :return: A resolver ready to repair the spawned shelf.
        """
        spawned = shelf.spawn_in_world()
        groups: list[SpawnedCollisionGroup] = [
            ShelfLayerGroup(
                bodies=spawned_layer.object_bodies,
                supporting_body=spawned_layer.surface.root,
                static_obstacles=(spawned.corpus,),
                shelf=shelf,
                layer_index=layer_index,
                parent=spawned.parent,
            )
            for layer_index, spawned_layer in enumerate(spawned.layers)
        ]
        return cls(spawned=spawned, groups=groups, rspn=rspn, max_passes=max_passes)

    @classmethod
    def for_table_with_chairs(
        cls,
        group: EGTableWithChairs,
        rspn: RelationalProbabilisticCircuit,
        max_passes: int = 50,
    ) -> InWorldLayoutResolver:
        """
        Spawn *group* and build a single collision group of its chairs, which
        stand on the floor and so are not support-checked.

        :param group: The sampled table-with-chairs group to spawn and repair.
        :param rspn: The fitted circuit used to redraw offending chair poses.
        :param max_passes: Upper bound on repair passes.
        :return: A resolver ready to repair the spawned group.
        """
        spawned = group.spawn_in_world()
        groups: list[SpawnedCollisionGroup] = [
            ChairGroup(
                bodies=spawned.chair_bodies,
                supporting_body=None,
                group=group,
                parent=spawned.parent,
            )
        ]
        return cls(spawned=spawned, groups=groups, rspn=rspn, max_passes=max_passes)

    def resolve(self) -> SpawnedLayout:
        """
        Repair every group until all are collision-free and supported, moving
        offending bodies in place.

        :raises LayoutResolutionError: If no valid layout is reached within
            :attr:`max_passes` passes.
        :return: The spawned, repaired layout.
        """
        detector = FCLCollisionDetector(_world=self.spawned.world)
        backend = probabilistic_backend(self.rspn)
        remaining: dict[int, set[int]] = {}
        for _ in range(self.max_passes):
            remaining = {
                group_index: violations
                for group_index, group in enumerate(self.groups)
                if (
                    violations := in_world_colliding_indices(
                        detector, group.bodies, group.static_obstacles
                    )
                    | group.unsupported_indices()
                )
            }
            if not remaining:
                return self.spawned
            for group_index, violations in remaining.items():
                self.groups[group_index].resample_and_move(violations, backend)
        raise LayoutResolutionError(
            remaining_groups=frozenset(remaining),
            passes_attempted=self.max_passes,
        )
