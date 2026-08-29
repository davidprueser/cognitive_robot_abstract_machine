from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from experiments.scene_generation_experiments.exceptions import LayoutResolutionError
from experiments.scene_generation_experiments.rspn_sampling import (
    build_free_space_conditioned_layer_query,
    evaluate_first_supported,
    probabilistic_backend,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import NoSolutionFound
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.scene_generation.scene_schema import EGShelf
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


def minimal_resample_set(colliding_pairs: set[tuple[int, int]]) -> set[int]:
    """
    Return a minimal set of indices whose removal breaks every colliding pair.

    Greedy minimum vertex cover: repeatedly discard the index involved in the
    most remaining colliding pairs, breaking ties by the higher index for
    reproducibility. The result depends only on which indices collide, not on
    the order the pairs are reported, so callers get the same, stable choice
    regardless of how the underlying collision detector orders its contacts.

    :param colliding_pairs: Pairs of indices that collide, each a sorted
        ``(low, high)`` tuple.
    :return: Indices to resample so that no colliding pair remains.
    """
    remaining_pairs = set(colliding_pairs)
    indices_to_resample: set[int] = set()
    while remaining_pairs:
        involvement_counts = Counter(
            index for pair in remaining_pairs for index in pair
        )
        most_colliding_index = min(
            involvement_counts,
            key=lambda index: (-involvement_counts[index], -index),
        )
        indices_to_resample.add(most_colliding_index)
        remaining_pairs = {
            pair for pair in remaining_pairs if most_colliding_index not in pair
        }
    return indices_to_resample


@dataclass
class ShelfLayerGroup:
    """
    The objects on one shelf layer that must not collide with each other or the shelf's
    corpus, resampled against that layer's learned distribution and re-seated at their
    existing resting height.
    """

    bodies: dict[int, Body]
    """
    The movable bodies to keep collision-free, keyed by their index in the owning
    layer's objects.
    """

    supporting_body: Body
    """
    The body the members must rest on.
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
    The shelf corpus the object bodies hang under; their poses are expressed relative to
    it, so they stay correct after the whole shelf is repositioned.
    """

    backend: ProbabilisticBackend = field(kw_only=True)
    """
    The single-sample backend over this layer's fitted circuit, from which offending
    object poses are redrawn.
    """

    static_obstacles: list[Body] = field(default_factory=list, kw_only=True)
    """
    Fixed bodies the members must not collide with, beyond each other (e.g. a shelf's
    corpus walls).

    Empty when the group has none.
    """

    def unsupported_indices(self) -> set[int]:
        """
        Return the indices of members that :attr:`supporting_body` does not support.
        """
        return {
            index
            for index, body in self.bodies.items()
            if not is_supported_by(body, self.supporting_body)
        }

    def colliding_indices(self, detector: FCLCollisionDetector) -> set[int]:
        """
        Return a minimal set of member indices whose resampling clears every real-mesh
        collision among this group's members, and against its :attr:`static_obstacles`,
        in the spawned world.

        A body that hits a static obstacle (e.g. a shelf's corpus wall) is
        always resampled directly: unlike an inter-body collision, there is
        no choice of *which* side to move.

        :param detector: A collision detector already synced to the world
            the members live in; it re-syncs on the state changes body moves
            emit.
        :return: Indices whose bodies must be resampled to remove all
            collisions.
        """
        body_to_index = {body: index for index, body in self.bodies.items()}
        collision_checks = {
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(body_to_index, 2)
        } | {
            CollisionCheck(body_a=body, body_b=obstacle, distance=0.0)
            for body in body_to_index
            for obstacle in self.static_obstacles
        }
        if not collision_checks:
            return set()
        result = detector.check_collisions(
            CollisionMatrix(collision_checks=collision_checks)
        )
        if not result.any():
            return set()
        obstacles = set(self.static_obstacles)
        colliding_pairs: set[tuple[int, int]] = set()
        obstacle_hit_indices: set[int] = set()
        for contact in result.contacts:
            if contact.body_a in obstacles:
                obstacle_hit_indices.add(body_to_index[contact.body_b])
            elif contact.body_b in obstacles:
                obstacle_hit_indices.add(body_to_index[contact.body_a])
            else:
                colliding_pairs.add(
                    tuple(
                        sorted(
                            (
                                body_to_index[contact.body_a],
                                body_to_index[contact.body_b],
                            )
                        )
                    )
                )
        return minimal_resample_set(colliding_pairs) | obstacle_hit_indices

    def clamp_to_bounds(self) -> None:
        """
        Move any object positioned outside this layer's own footprint back to its
        nearest in-bounds position.

        A resampled object is always re-seated at its previous resting height (see
        :meth:`resample_and_move`), so nothing else re-checks whether its X/Y position
        stayed within the layer -- an RSPN sample landing outside the footprint would
        otherwise keep being treated as a collision-style violation and sent through an
        expensive resample every pass, which is not conditioned on staying in bounds and
        can land outside it again just as easily. Moving it back directly is a plain,
        cheap geometric fix that always succeeds.
        """
        # The content frame's x-axis spans the shelf's length (its depth) and y spans
        # its width (its face) -- the same mapping EGShelfLayer.spawn's slab Scale and
        # object_local_pose use, not shelf.scale's own x/y order confused with a
        # width/length label.
        layer = self.shelf.layers[self.layer_index]
        half_x = self.shelf.scale.x / 2
        half_y = self.shelf.scale.y / 2
        for index, object_2d in enumerate(layer.objects):
            if index not in self.bodies:
                continue
            max_x = max(half_x - object_2d.scale.x / 2, 0.0)
            max_y = max(half_y - object_2d.scale.y / 2, 0.0)
            clamped_x = min(max(float(object_2d.pose.x), -max_x), max_x)
            clamped_y = min(max(float(object_2d.pose.y), -max_y), max_y)
            if clamped_x == float(object_2d.pose.x) and clamped_y == float(
                object_2d.pose.y
            ):
                continue
            object_2d.pose = Pose2D(x=clamped_x, y=clamped_y, yaw=object_2d.pose.yaw)
            body = self.bodies[index]
            resting_z = body.parent_connection.origin.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_local_pose(
                object_2d, resting_z, self.corpus
            )

    def resample_and_move(self, indices: set[int]) -> None:
        """
        Redraw each of *indices*' poses, truncated to the free space its layer's
        surface actually has left, and move the corresponding body to match.

        A layer's object slots are independent in the fitted circuit, so an
        unconditioned redraw has no way to avoid a neighbour on its own -- only
        clearing the space it is drawn from does that (see
        :func:`~experiments.scene_generation_experiments.rspn_sampling.build_free_space_conditioned_layer_query`).
        The free space is computed from the *world*, not the query, so members are
        redrawn one at a time rather than all at once: each newly placed body is
        real geometry the next member's free-space calculation already sees, which is
        what keeps two members of the same call from landing on each other. A joint,
        one-shot draw could not offer that guarantee between its own free slots.

        :param indices: Indices of this group's members to redraw.
        """
        layer = self.shelf.layers[self.layer_index]
        fixed_objects = [
            object_2d
            for object_index, object_2d in enumerate(layer.objects)
            if object_index not in indices and object_index in self.bodies
        ]

        for index in sorted(indices):
            object_2d = layer.objects[index]
            object_bloat = max(object_2d.scale.x, object_2d.scale.y) / 2
            free_space_event = layer.annotation.calculate_free_space(
                object_bloat=object_bloat
            ).free_space_event

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
            redrawn = redrawn_layer.objects[-1]

            object_2d.pose = redrawn.pose
            body = self.bodies[index]
            resting_z = body.parent_connection.origin.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_local_pose(
                object_2d, resting_z, self.corpus
            )
            fixed_objects.append(object_2d)


@dataclass
class InWorldLayoutResolver:
    """
    Repairs a spawned layout by validating it directly in its :class:`World` and moving
    offending bodies in place, until every collision group is collision-free and
    supported.

    The layout is spawned once; each repair pass redraws only the pose of offending
    members -- holding their scale, and therefore their mesh, fixed -- and moves the
    corresponding bodies, so meshes are never reloaded. Each :class:`ShelfLayerGroup`
    owns how it redraws its own members from its layer's fitted circuit, so the resolver
    just drives the repair loop across every layer of the shelf.
    """

    shelf: EGShelf
    """
    The already-spawned shelf to repair and return.
    """

    dropped_body_count: int = field(default=0, init=False)
    """
    Bodies removed because no repair pass could place them.

    A generated shelf comes out sparser than the layout it was built from, and without
    this count an empty-looking shelf cannot be told apart from a model that simply
    sampled few pieces.
    """

    groups: list[ShelfLayerGroup]
    """
    One collision group per shelf layer, to keep collision-free and supported; each
    knows how to redraw its own offending members from its layer's fitted circuit.
    """

    max_passes: int = 10
    """
    Upper bound on repair passes before giving up on an unsatisfiable layout.
    """

    stuck_after_passes: int = 3
    """
    Consecutive passes a member may remain in violation, unresolved, before it stops
    being resampled and is left for the final drop instead.

    A redraw is an independent sample from roughly the same conditional distribution
    each time, so a member whose redrawn pose keeps landing in the same collision pass
    after pass is not converging -- it is only burning passes' worth of grounding cost
    on an object that was never going to resolve, at the expense of the budget every
    other member in the layout shares.
    """

    @classmethod
    def for_shelf(
        cls,
        shelf: EGShelf,
        rspn: RelationalProbabilisticCircuit,
        max_passes: int = 10,
        stuck_after_passes: int = 3,
        placeholders_for_missing_meshes: bool = False,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> InWorldLayoutResolver:
        """
        Spawn *shelf* and build one collision group per layer, each supported by its own
        slab and checked against the shelf's own corpus walls.

        :param shelf: The sampled shelf to spawn and repair.
        :param rspn: The fitted circuit used to redraw offending object poses.
        :param placeholders_for_missing_meshes: Stand a plain box in for objects with no
            cached mesh, so an incomplete mesh library is visible in the render rather
            than mistaken for a sparse draw.
        :param max_passes: Upper bound on repair passes.
        :param stuck_after_passes: Consecutive passes a member may remain in violation
            before it stops being resampled.
        :param world: World to spawn *shelf* into. A fresh, isolated world is created
            when omitted, matching every existing caller. Collision checking during
            repair runs against this whole world, so pass one that holds nothing besides
            the shelf yet -- unrelated bodies already in it would be treated as
            obstacles.
        :param parent: The entity *shelf* is placed under. Defaults to ``world``'s root.
        :param parent_T_self: Where the shelf's own origin sits in *parent*'s frame. See
            :meth:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.spawn`.
        :return: A resolver ready to repair the spawned shelf.
        """
        shelf.spawn(world=world, parent=parent, parent_T_self=parent_T_self)
        groups = cls._shelf_layer_groups(shelf, probabilistic_backend(rspn))
        return cls(
            shelf=shelf,
            groups=groups,
            max_passes=max_passes,
            stuck_after_passes=stuck_after_passes,
        )

    @staticmethod
    def _shelf_layer_groups(
        shelf: EGShelf,
        backend: ProbabilisticBackend,
    ) -> list[ShelfLayerGroup]:
        """
        Build one :class:`ShelfLayerGroup` per layer of *shelf*, each supported by its
        own slab, checked against the shelf's corpus walls, and resampled from
        *backend*.
        """
        return [
            ShelfLayerGroup(
                bodies={
                    index: obj.annotation
                    for index, obj in enumerate(layer.objects)
                    if obj.annotation is not None
                },
                supporting_body=layer.annotation.root,
                static_obstacles=[shelf.corpus],
                backend=backend,
                shelf=shelf,
                layer_index=layer_index,
                corpus=shelf.corpus,
            )
            for layer_index, layer in enumerate(shelf.layers)
        ]

    def resolve(self) -> EGShelf:
        """
        Repair every group until all are collision-free and supported, moving offending
        bodies in place.

        Some sampled arrangements cannot be separated by moving alone -- objects too big
        or too many for the space. After :attr:`max_passes`, the still offending objects
        are dropped from the layout, so a best-effort collision-free arrangement is
        returned rather than failing the whole sample.

        :raises LayoutResolutionError: If violations remain even after dropping the
            offending objects -- a state that should not occur.
        :return: The spawned, repaired shelf.
        """
        detector = FCLCollisionDetector(_world=self.shelf.world)
        remaining = self._repaired(detector)
        detector.stop()

        if remaining:
            raise LayoutResolutionError(
                remaining_groups=frozenset(remaining),
                passes_attempted=self.max_passes,
            )
        return self.shelf

    def _repaired(self, detector: FCLCollisionDetector) -> dict[int, set[int]]:
        """
        Run the repair passes and report whatever still offends once they, and the final
        drop, are done.

        :param detector: The detector to check the world through.
        :return: Offending member indices per group index; empty when the layout came
            out clean.
        """
        stuck_counts: dict[tuple[int, int], int] = {}
        for _ in range(self.max_passes):
            self._clamp_groups_to_bounds()
            remaining = self._remaining_violations(detector)
            if not remaining:
                return {}
            to_resample, stuck_counts = self._resamplable(remaining, stuck_counts)
            if not to_resample:
                break
            for group_index, violations in to_resample.items():
                self.groups[group_index].resample_and_move(violations)

        self._clamp_groups_to_bounds()
        remaining = self._remaining_violations(detector)
        if not remaining:
            return {}
        self._drop_objects(remaining)
        return self._remaining_violations(detector)

    def _resamplable(
        self,
        remaining: dict[int, set[int]],
        stuck_counts: dict[tuple[int, int], int],
    ) -> tuple[dict[int, set[int]], dict[tuple[int, int], int]]:
        """
        Split *remaining* into members still worth resampling and an updated stuck-pass
        count for each, dropping the count for any member no longer in violation so it
        gets a fresh budget if it offends again later for an unrelated reason.

        :param remaining: Offending member indices per group index.
        :param stuck_counts: Consecutive violation-pass counts from the previous pass,
            keyed by ``(group_index, member_index)``.
        :return: Members to resample per group index (groups with none are omitted), and
            the updated stuck counts.
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

    def _clamp_groups_to_bounds(self) -> None:
        """
        Move every group's out-of-bounds members back within its footprint.

        Run before each collision check so an RSPN sample that landed outside a group's
        footprint is fixed by a plain, cheap geometric move rather than being treated as
        a collision-style violation and sent through an expensive resample that is not
        conditioned on staying in bounds and so could land outside it again just as
        easily.
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
                violations := group.colliding_indices(detector)
                | group.unsupported_indices()
            )
        }

    def _drop_objects(self, offenders: dict[int, set[int]]) -> None:
        """
        Remove *offenders* from their groups and from the world, so an arrangement that
        cannot be packed is rendered without the objects that do not fit rather than not
        at all.

        A dropped piece takes its whole branch with it.

        :param offenders: Offending member indices per group index.
        """
        world = self.shelf.world
        with world.modify_world():
            for group_index, indices in offenders.items():
                group = self.groups[group_index]
                layer = self.shelf.layers[group.layer_index]
                for index in indices:
                    self._remove_branch(group.bodies.pop(index))
                    layer.objects[index].annotation = None
                    self.dropped_body_count += 1
            world.delete_orphaned_dofs()
        self._forget_bodies_no_longer_in_the_world()

    def _remove_branch(self, branch_root: KinematicStructureEntity) -> None:
        """
        Remove *branch_root* and everything hanging beneath it from the world.

        :param branch_root: The entity whose branch is dropped.
        """
        world = self.shelf.world
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

        A group's ``bodies`` is this resolver's own cache of its layer's spawned
        objects, so a departed entry is also cleared from the corresponding
        :class:`~semantic_digital_twin.scene_generation.scene_schema.EGObject2D`'s
        own :attr:`~semantic_digital_twin.scene_generation.scene_schema.EGObject2D.
        annotation`, keeping the returned layout in step with the world.
        """
        remaining_bodies = set(self.shelf.world.bodies)
        for group in self.groups:
            layer = self.shelf.layers[group.layer_index]
            departed = [
                index
                for index, body in group.bodies.items()
                if body not in remaining_bodies
            ]
            for index in departed:
                group.bodies.pop(index)
                layer.objects[index].annotation = None
            # A dropped obstacle left in place makes the next check ask the
            # detector about a body the world no longer holds.
            group.static_obstacles[:] = [
                body for body in group.static_obstacles if body in remaining_bodies
            ]
