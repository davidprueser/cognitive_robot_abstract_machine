from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_pose_resample_query,
    in_world_colliding_indices,
)
from experiments.scene_generation_experiments.exceptions import LayoutResolutionError
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
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
    EGPoint2D,
    EGShelf,
    SpawnedLayout,
    SpawnedShelf,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


def _evaluate_first_supported(backend: ProbabilisticBackend, *queries):
    """
    Evaluate *queries* in order, returning the first sample the RSPN has
    support for.

    Each query is expected to hold strictly less evidence than the one before
    it, so the search walks outwards from the most informative conditioning to
    the least. Two kinds of evidence go unsupported in practice, and both abort
    the whole layout if the search stops early:

    - **Neighbour poses.** Conditioning a resample on every already-placed
      neighbour's exact pose pins the query to a region of zero probability
      mass, and the neighbours drift further from the training distribution
      with each repair pass.
    - **The layer's own scale.** Every layer of a shelf is conditioned on the
      reference layer's drawn scale (see :func:`build_layer_query_with_fixed_scale`),
      so a later layer's objects are resampled against a scale the circuit only
      ever saw paired with a different layer's contents. Relaxing only the
      neighbours keeps that scale pinned and fails again.

    :param backend: The backend to evaluate the queries against.
    :param queries: Progressively less-conditioned forms of the same query.
    :raises NoSolutionFound: If the circuit supports none of them.
    :return: The first sample from whichever query found a solution.
    """
    for query in queries[:-1]:
        try:
            return next(iter(backend.evaluate(query)))
        except NoSolutionFound:
            continue
    return next(iter(backend.evaluate(queries[-1])))


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
    checked for support (e.g. members standing on the floor).
    """

    static_obstacles: list[Body] = field(default_factory=list, kw_only=True)
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

    def clamp_to_bounds(self) -> None:
        """
        Move any member positioned outside this group's own footprint back
        to its nearest in-bounds position; a no-op when the group does not
        bound its members' positions.
        """

    @abstractmethod
    def resample_and_move(self, indices: set[int]) -> None:
        """
        Redraw the poses of the members at *indices* from this group's circuit,
        holding every other member fixed, then move the corresponding bodies to
        those poses.
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

    corpus: KinematicStructureEntity
    """
    The shelf corpus the object bodies hang under; their poses are expressed
    relative to it, so they stay correct after the whole shelf is repositioned.
    """

    backend: ProbabilisticBackend = field(kw_only=True)
    """
    The single-sample backend over this layer's fitted circuit, from which
    offending object poses are redrawn.
    """

    def clamp_to_bounds(self) -> None:
        """
        Move any object positioned outside this layer's own footprint back
        to its nearest in-bounds position.

        A resampled object is always re-seated at its previous resting
        height (see :meth:`resample_and_move`), so nothing else re-checks
        whether its X/Y position stayed within the layer -- an RSPN sample
        landing outside the footprint would otherwise keep being treated as
        a collision-style violation and sent through an expensive resample
        every pass, which is not conditioned on staying in bounds and can
        land outside it again just as easily. Moving it back directly is a
        plain, cheap geometric fix that always succeeds.
        """
        layer = self.shelf.layers[self.layer_index]
        half_width = layer.scale.width / 2
        half_length = layer.scale.length / 2
        for index, object_2d in enumerate(layer.objects):
            if index not in self.bodies:
                continue
            max_x = max(half_width - object_2d.scale.width / 2, 0.0)
            max_y = max(half_length - object_2d.scale.length / 2, 0.0)
            clamped_x = min(max(object_2d.position.x, -max_x), max_x)
            clamped_y = min(max(object_2d.position.y, -max_y), max_y)
            if clamped_x == object_2d.position.x and clamped_y == object_2d.position.y:
                continue
            object_2d.position = EGPoint2D(x=clamped_x, y=clamped_y)
            body = self.bodies[index]
            resting_z = body.parent_connection.origin.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_local_pose(
                object_2d, resting_z, self.corpus
            )

    def resample_and_move(self, indices: set[int]) -> None:
        layer = self.shelf.layers[self.layer_index]
        fixed_objects = [
            object_2d
            for object_index, object_2d in enumerate(layer.objects)
            if object_index not in indices and object_index in self.bodies
        ]
        resampled_indices = sorted(indices)
        resampled_objects = [layer.objects[index] for index in resampled_indices]

        new_layer = _evaluate_first_supported(
            self.backend,
            build_pose_resample_query(
                fixed_objects, len(resampled_objects), layer.scale
            ),
            build_pose_resample_query([], len(resampled_objects), layer.scale),
            build_free_layer_query(len(resampled_objects)),
        )
        redrawn_objects = new_layer.objects[-len(resampled_objects) :]

        for object_index, redrawn in zip(resampled_indices, redrawn_objects):
            object_2d = layer.objects[object_index]
            object_2d.position = redrawn.position
            object_2d.orientation = redrawn.orientation
            body = self.bodies[object_index]
            resting_z = body.parent_connection.origin.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_local_pose(
                object_2d, resting_z, self.corpus
            )


@dataclass
class InWorldLayoutResolver:
    """
    Repairs a spawned layout by validating it directly in its :class:`World` and
    moving offending bodies in place, until every collision group is
    collision-free and supported.

    The layout is spawned once; each repair pass redraws only the pose of
    offending members -- holding their scale, and therefore their mesh, fixed --
    and moves the corresponding bodies, so meshes are never reloaded. Every
    generator plugs in through :class:`SpawnedCollisionGroup`, and each group
    owns how it redraws its members -- from a fitted circuit or from a supporting
    surface -- so the resolver is agnostic to what kind of objects it is
    arranging and can mix groups from different generators in one layout.
    """

    spawned: SpawnedLayout
    """
    The already-spawned layout to repair and return.
    """

    dropped_body_count: int = field(default=0, init=False)
    """
    Bodies removed because no repair pass could place them.

    A generated shelf comes out sparser than the layout it was built from, and
    without this count an empty-looking shelf cannot be told apart from a model
    that simply sampled few pieces.
    """

    groups: list[SpawnedCollisionGroup]
    """
    The collision groups to keep collision-free and supported; each knows how to
    redraw its own offending members, whether from a fitted circuit or a
    supporting surface.
    """

    max_passes: int = 10
    """
    Upper bound on repair passes before giving up on an unsatisfiable layout.
    """

    @classmethod
    def for_shelf(
        cls,
        shelf: EGShelf,
        rspn: RelationalProbabilisticCircuit,
        max_passes: int = 10,
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
        groups = cls._shelf_layer_groups(shelf, spawned, probabilistic_backend(rspn))
        return cls(spawned=spawned, groups=groups, max_passes=max_passes)

    @staticmethod
    def _shelf_layer_groups(
        shelf: EGShelf,
        spawned: SpawnedShelf,
        backend: ProbabilisticBackend,
    ) -> list[SpawnedCollisionGroup]:
        """
        Build one :class:`ShelfLayerGroup` per layer of *spawned*, each supported
        by its own slab, checked against the shelf's corpus walls, and resampled
        from *backend*.
        """
        return [
            ShelfLayerGroup(
                bodies=spawned_layer.object_bodies,
                supporting_body=spawned_layer.surface.root,
                static_obstacles=[spawned.corpus],
                backend=backend,
                shelf=shelf,
                layer_index=layer_index,
                corpus=spawned.corpus,
            )
            for layer_index, spawned_layer in enumerate(spawned.layers)
        ]

    def resolve(self) -> SpawnedLayout:
        """
        Repair every group until all are collision-free and supported, moving
        offending bodies in place.

        Some sampled arrangements cannot be separated by moving alone -- objects
        too big or too many for the space. After :attr:`max_passes`, the still
        offending objects are dropped from the layout, so a best-effort
        collision-free arrangement is returned rather than failing the whole
        sample.

        :raises LayoutResolutionError: If violations remain even after dropping
            the offending objects -- a state that should not occur.
        :return: The spawned, repaired layout.
        """
        detector = FCLCollisionDetector(_world=self.spawned.world)
        remaining = self._repaired(detector)
        detector.stop()

        if remaining:
            raise LayoutResolutionError(
                remaining_groups=frozenset(remaining),
                passes_attempted=self.max_passes,
            )
        return self.spawned

    def _repaired(self, detector: FCLCollisionDetector) -> dict[int, set[int]]:
        """
        Run the repair passes and report whatever still offends once they, and
        the final drop, are done.

        :param detector: The detector to check the world through.
        :return: Offending member indices per group index; empty when the layout
            came out clean.
        """
        for _ in range(self.max_passes):
            self._clamp_groups_to_bounds()
            remaining = self._remaining_violations(detector)
            if not remaining:
                return {}
            for group_index, violations in remaining.items():
                self.groups[group_index].resample_and_move(violations)

        self._clamp_groups_to_bounds()
        remaining = self._remaining_violations(detector)
        if not remaining:
            return {}
        self._drop_objects(remaining)
        return self._remaining_violations(detector)

    def _clamp_groups_to_bounds(self) -> None:
        """
        Move every group's out-of-bounds members back within its footprint.

        Run before each collision check so an RSPN sample that landed outside
        a group's footprint is fixed by a plain, cheap geometric move rather
        than being treated as a collision-style violation and sent through
        an expensive resample that is not conditioned on staying in bounds
        and so could land outside it again just as easily.
        """
        for group in self.groups:
            group.clamp_to_bounds()

    def _remaining_violations(
        self, detector: FCLCollisionDetector
    ) -> dict[int, set[int]]:
        """
        Map each group index to its members that collide or are unsupported.

        One detector serves every pass: it registers world callbacks that
        re-sync it on the model and state changes that moving or dropping a body
        emits, so it always reflects the current world. Building a fresh one per
        pass left every previous detector registered on the world and alive,
        which cost about 230 MB a pass on a 29-piece room and ran the process
        out of memory before the fiftieth.

        :param detector: The detector to check the world through.
        :return: Offending member indices per group; groups with none are
            omitted.
        """
        return {
            group_index: violations
            for group_index, group in enumerate(self.groups)
            if (
                violations := in_world_colliding_indices(
                    detector, group.bodies, group.static_obstacles
                )
                | group.unsupported_indices()
            )
        }

    def _drop_objects(self, offenders: dict[int, set[int]]) -> None:
        """
        Remove *offenders* from their groups and from the world, so an
        arrangement that cannot be packed is rendered without the objects that
        do not fit rather than not at all.

        A dropped piece takes its whole branch with it.

        :param offenders: Offending member indices per group index.
        """
        world = self.spawned.world
        with world.modify_world():
            for group_index, indices in offenders.items():
                bodies = self.groups[group_index].bodies
                for index in indices:
                    self._remove_branch(bodies.pop(index))
                    self.dropped_body_count += 1
            world.delete_orphaned_dofs()
        self._forget_bodies_no_longer_in_the_world()

    def _remove_branch(self, branch_root: KinematicStructureEntity) -> None:
        """
        Remove *branch_root* and everything hanging beneath it from the world.

        :param branch_root: The entity whose branch is dropped.
        """
        world = self.spawned.world
        for entity in world.compute_descendent_child_kinematic_structure_entities(
            branch_root
        ) + [branch_root]:
            world.remove_kinematic_structure_entity(entity)

    def _forget_bodies_no_longer_in_the_world(self) -> None:
        """
        Drop from every group the members whose bodies have left the world.

        A group that was not itself offending is never visited by the drop, so
        without this it keeps referring to bodies the world no longer holds and
        the next collision check raises instead of reporting collisions.

        Entries are popped rather than the dict rebuilt: a group's ``bodies`` is
        the same object as the spawned layer's own body map, so popping is what
        keeps the returned layout in step with the world.
        """
        remaining_bodies = set(self.spawned.world.bodies)
        for group in self.groups:
            departed = [
                index
                for index, body in group.bodies.items()
                if body not in remaining_bodies
            ]
            for index in departed:
                group.bodies.pop(index)
            # A dropped obstacle left in place makes the next check ask the
            # detector about a body the world no longer holds.
            group.static_obstacles[:] = [
                body for body in group.static_obstacles if body in remaining_bodies
            ]
