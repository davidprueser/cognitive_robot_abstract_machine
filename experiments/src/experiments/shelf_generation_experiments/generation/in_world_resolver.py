from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from experiments.shelf_generation_experiments.generation.colliding_pairs import (
    CollidingPairs,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.reasoning.predicates import SupportedBy
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


# %% one layer
@dataclass
class SpawnedLayerGroup:
    """
    The spawned objects of one shelf layer, which must neither collide with each other
    or the shelf corpus nor stop resting on their layer's slab.
    """

    layer: RelationalCircuitExperimentShelfLayer
    """
    The spawned layer whose objects are checked.
    """

    corpus: Body
    """
    The shelf corpus the objects must not collide with.
    """

    @property
    def bodies(self) -> dict[int, Body]:
        """
        The bodies of the objects still standing on the layer, keyed by their index in
        the layer's objects.
        """
        return {
            index: object_.annotation
            for index, object_ in enumerate(self.layer.objects)
            if object_.annotation is not None
        }

    def unsupported_indices(self) -> set[int]:
        """
        :return: The indices of the objects their layer's slab does not hold up.
        """
        slab = self.layer.annotation.root
        return {
            index
            for index, body in self.bodies.items()
            if not SupportedBy(supported=body, supporting=slab)()
        }

    def colliding_indices(self, detector: FCLCollisionDetector) -> set[int]:
        """
        :param detector: A collision detector of the world the objects stand in.
        :return: A small set of object indices whose removal clears every collision
            among the objects; an object hitting the corpus is always among them, since
            only it can be moved.
        """
        index_by_body = {body: index for index, body in self.bodies.items()}
        collision_checks = {
            CollisionCheck(body_a=first_body, body_b=second_body, distance=0.0)
            for first_body, second_body in combinations(index_by_body, 2)
        } | {
            CollisionCheck(body_a=body, body_b=self.corpus, distance=0.0)
            for body in index_by_body
        }
        if not collision_checks:
            return set()
        result = detector.check_collisions(
            CollisionMatrix(collision_checks=collision_checks)
        )
        colliding_pairs = CollidingPairs()
        corpus_hitting_indices: set[int] = set()
        for contact in result.contacts:
            if contact.body_a is self.corpus:
                corpus_hitting_indices.add(index_by_body[contact.body_b])
            elif contact.body_b is self.corpus:
                corpus_hitting_indices.add(index_by_body[contact.body_a])
            else:
                colliding_pairs.add(
                    index_by_body[contact.body_a], index_by_body[contact.body_b]
                )
        return colliding_pairs.minimal_resample_set() | corpus_hitting_indices


# %% whole shelf
@dataclass
class InWorldLayoutResolver:
    """
    Drops whatever the real meshes of a spawned shelf turn out to collide with or leave
    unsupported.

    Overlap is repaired before spawning, on footprints that only approximate the real
    meshes (see :class:`~experiments.shelf_generation_experiments.generation.
    pre_spawn_resolver.PreSpawnLayoutResolver`); this catches what that approximation
    missed.
    """

    shelf: RelationalCircuitExperimentShelf
    """
    The spawned shelf to repair.
    """

    dropped_body_count: int = field(default=0, init=False)
    """
    How many bodies :meth:`resolve` removed.
    """

    @property
    def groups(self) -> list[SpawnedLayerGroup]:
        """
        One group per layer of :attr:`shelf`.
        """
        return [
            SpawnedLayerGroup(layer=layer, corpus=self.shelf.corpus)
            for layer in self.shelf.layers
        ]

    def resolve(self) -> RelationalCircuitExperimentShelf:
        """
        Remove every object that collides or is unsupported from the world and from
        :attr:`shelf`.

        :return: The repaired shelf.
        """
        detector = FCLCollisionDetector(_world=self.shelf.world)
        offenders = [
            (group, group.colliding_indices(detector) | group.unsupported_indices())
            for group in self.groups
        ]
        detector.stop()
        world = self.shelf.world
        with world.modify_world():
            for group, indices in offenders:
                for index in indices:
                    self._remove_object(group.layer, index)
            world.delete_orphaned_dofs()
        return self.shelf

    def _remove_object(
        self, layer: RelationalCircuitExperimentShelfLayer, index: int
    ) -> None:
        """
        Remove the object at *index* of *layer*, and everything hanging beneath its
        body, from the world.

        :param layer: The layer the object stands on.
        :param index: The index of the object in the layer's objects.
        """
        world = self.shelf.world
        body = layer.objects[index].annotation
        branch: list[KinematicStructureEntity] = [
            *world.compute_descendent_child_kinematic_structure_entities(body),
            body,
        ]
        for entity in branch:
            world.remove_kinematic_structure_entity(entity)
        layer.objects[index].annotation = None
        self.dropped_body_count += 1
