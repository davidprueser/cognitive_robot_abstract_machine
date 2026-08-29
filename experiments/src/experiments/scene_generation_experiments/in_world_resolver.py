from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.scene_generation.scene_schema import EGShelf
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
    corpus, and must rest on their layer's own slab.
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
    The shelf whose layer this group belongs to.
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


@dataclass
class InWorldLayoutResolver:
    """
    A single-pass safety net that drops whatever a spawned shelf's real mesh geometry
    turns out to collide with or leave unsupported.

    The heavy lifting -- resampling colliding members against the RSPN, conditioned on
    free space -- happens before anything is spawned (see
    :class:`~experiments.scene_generation_experiments.pre_spawn_resolver.
    PreSpawnLayoutResolver`), on bounding boxes built from each matched mesh candidate's
    own extents. Those extents only approximate a candidate's true mesh geometry, so
    this resolver exists to catch whatever that approximation missed once the real
    mesh is loaded -- by dropping the offender, not by redrawing its pose, since a
    redraw here would mean reasoning about free space against real geometry again,
    which is exactly the per-pass cost moving resampling before spawning was meant to
    remove.
    """

    shelf: EGShelf
    """
    The already-spawned shelf to repair and return.
    """

    dropped_body_count: int = field(default=0, init=False)
    """
    Bodies removed by :meth:`resolve` because their real mesh geometry collided or lost
    support.

    A generated shelf comes out sparser than the layout it was built from, and without
    this count an empty-looking shelf cannot be told apart from a model that simply
    sampled few pieces.
    """

    groups: list[ShelfLayerGroup]
    """
    One collision group per shelf layer, to check for a real-mesh collision or lost
    support.
    """

    @classmethod
    def for_shelf(cls, shelf: EGShelf) -> InWorldLayoutResolver:
        """
        Build one collision group per layer of an already-spawned *shelf*, each
        supported by its own slab and checked against the shelf's own corpus walls.

        :param shelf: The already-spawned shelf to check.
        :return: A resolver ready to check the spawned shelf.
        """
        return cls(shelf=shelf, groups=cls._shelf_layer_groups(shelf))

    @staticmethod
    def _shelf_layer_groups(shelf: EGShelf) -> list[ShelfLayerGroup]:
        """
        Build one :class:`ShelfLayerGroup` per layer of *shelf*, each supported by its
        own slab and checked against the shelf's corpus walls.
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
                shelf=shelf,
                layer_index=layer_index,
                corpus=shelf.corpus,
            )
            for layer_index, layer in enumerate(shelf.layers)
        ]

    def resolve(self) -> EGShelf:
        """
        Drop every member that collides or is unsupported, so a real-mesh mismatch the
        pre-spawn resolver's own bounding-box approximation could not foresee never
        reaches the returned shelf.

        :return: The spawned shelf, with any offending members dropped.
        """
        detector = FCLCollisionDetector(_world=self.shelf.world)
        offenders = {
            group_index: violations
            for group_index, group in enumerate(self.groups)
            if (
                violations := group.colliding_indices(detector)
                | group.unsupported_indices()
            )
        }
        if offenders:
            self._drop_objects(offenders)
        detector.stop()
        return self.shelf

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
