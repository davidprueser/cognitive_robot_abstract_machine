from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

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
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGRoom,
    EGShelf,
    EGTableWithChairs,
    SpawnedLayout,
    SpawnedRoom,
    SpawnedShelf,
    SpawnedTableWithChairs,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
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

    def resample_and_move(self, indices: set[int]) -> None:
        layer = self.shelf.layers[self.layer_index]
        fixed_objects = [
            object_2d
            for object_index, object_2d in enumerate(layer.objects)
            if object_index not in indices and object_index in self.bodies
        ]
        resampled_indices = sorted(indices)
        resampled_objects = [layer.objects[index] for index in resampled_indices]

        new_layer = _evaluate_with_relaxed_fallback(
            self.backend,
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
            resting_z = body.parent_connection.origin.to_position().to_np()[2]
            body.parent_connection.origin = self.shelf.object_local_pose(
                object_2d, resting_z, self.corpus
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

    table: KinematicStructureEntity
    """
    The table body the chair bodies hang under; their poses are expressed
    relative to it, so they stay correct after the whole group is repositioned.
    """

    backend: ProbabilisticBackend = field(kw_only=True)
    """
    The single-sample backend over this table's fitted circuit, from which
    offending chair poses are redrawn.
    """

    def resample_and_move(self, indices: set[int]) -> None:
        chairs = self.group.chairs
        fixed_chairs = [
            chair
            for chair_index, chair in enumerate(chairs)
            if chair_index not in indices and chair_index in self.bodies
        ]
        resampled_indices = sorted(indices)
        resampled_chairs = [chairs[index] for index in resampled_indices]

        new_sample = _evaluate_with_relaxed_fallback(
            self.backend,
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
            body.parent_connection.origin = self.group.chair_local_pose(
                chair, self.table
            )


@dataclass
class FloorObjectGroup(SpawnedCollisionGroup):
    """
    A room's floor pieces -- free objects and furniture roots -- kept from
    colliding with each other and the walls, each resting on the floor and, when
    it offends, redrawn onto a free point of the floor's surface.

    A piece keeps its resting height and yaw across a move: only its footprint on
    the floor is redrawn, so a shelf stays upright at its corpus height and a
    table keeps standing on its legs after being repositioned.
    """

    floor: Floor
    """
    The room floor annotation whose supporting surface offending pieces are
    redrawn onto, and against which they are support-checked.
    """

    _FREE_POINT_POOL_FACTOR: ClassVar[int] = 8
    """
    How many candidate floor points to draw per offending piece, so several
    pieces redrawn in one pass land spread across the surface rather than
    clustered at its most likely centre.
    """

    def resample_and_move(self, indices: set[int]) -> None:
        ordered_indices = sorted(indices)
        free_points = self.floor.sample_points_from_surface(
            amount=len(ordered_indices) * self._FREE_POINT_POOL_FACTOR
        )
        if not free_points:
            return
        stride = max(len(free_points) // len(ordered_indices), 1)
        spread_points = free_points[::stride][: len(ordered_indices)]
        for index, free_point in zip(ordered_indices, spread_points):
            self._move_piece_to_floor_point(self.bodies[index], free_point)

    def _move_piece_to_floor_point(self, body: Body, free_point: Point3) -> None:
        """
        Move *body* so its footprint sits over *free_point*, keeping its current
        resting height and yaw, so the piece stays upright and on the floor.

        :param body: The floor piece's root body to reposition.
        :param free_point: A point on the floor surface, in the surface frame.
        """
        connection = body.parent_connection
        parent = connection.parent
        world = self.floor._world
        world.update_forward_kinematics()
        parent_T_surface = world.compute_forward_kinematics_np(
            parent, free_point.reference_frame
        )
        point_in_parent = parent_T_surface @ free_point.to_np()

        new_origin = connection.origin.to_np().copy()
        new_origin[0, 3] = point_in_parent[0]
        new_origin[1, 3] = point_in_parent[1]
        connection.origin = HomogeneousTransformationMatrix(
            new_origin, reference_frame=parent
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

    groups: list[SpawnedCollisionGroup]
    """
    The collision groups to keep collision-free and supported; each knows how to
    redraw its own offending members, whether from a fitted circuit or a
    supporting surface.
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
        groups = cls._shelf_layer_groups(shelf, spawned, probabilistic_backend(rspn))
        return cls(spawned=spawned, groups=groups, max_passes=max_passes)

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
            cls._chair_group(group, spawned, probabilistic_backend(rspn))
        ]
        return cls(spawned=spawned, groups=groups, max_passes=max_passes)

    @classmethod
    def for_scene(
        cls,
        room: EGRoom,
        shelf_rspn: RelationalProbabilisticCircuit,
        table_rspn: RelationalProbabilisticCircuit,
        object_id_to_mesh_path: dict[str, Path] | None = None,
        max_passes: int = 50,
    ) -> InWorldLayoutResolver:
        """
        Spawn a whole *room* and build one floor-placement group plus a content
        group per shelf layer and per table.

        The floor-placement group holds every floor piece -- free objects and the
        movable roots of shelves and tables -- so their mutual collisions and
        collisions with the room's walls are all resolved together, without a
        separate cross-furniture check. Each shelf layer and table then keeps its
        own contents collision-free from its own fitted circuit; because a piece's
        contents are children of its movable root, they follow it rigidly and stay
        arranged as the piece is repositioned on the floor.

        :param room: The sampled room to spawn and repair.
        :param shelf_rspn: The fitted circuit used to redraw offending objects on
            shelf layers.
        :param table_rspn: The fitted circuit used to redraw offending chairs
            around tables.
        :param object_id_to_mesh_path: Mapping from a free floor object's id to
            its mesh directory, used to resolve per-object mesh paths.
        :param max_passes: Upper bound on repair passes.
        :return: A resolver ready to repair the whole spawned room.
        """
        world = World()
        root = Body(name=PrefixedName(name="map"))
        with world.modify_world():
            world.add_body(root)
        spawned = room.spawn_in_world(world, object_id_to_mesh_path, root)

        groups: list[SpawnedCollisionGroup] = [
            FloorObjectGroup(
                bodies=cls._floor_pieces(spawned),
                supporting_body=None,
                static_obstacles=spawned.wall_bodies,
                floor=spawned.floor,
            )
        ]
        shelf_backend = probabilistic_backend(shelf_rspn)
        for shelf, spawned_shelf in zip(room.shelves, spawned.spawned_shelves):
            groups.extend(
                cls._shelf_layer_groups(shelf, spawned_shelf, shelf_backend)
            )
        table_backend = probabilistic_backend(table_rspn)
        for table, spawned_table in zip(room.tables, spawned.spawned_tables):
            groups.append(cls._chair_group(table, spawned_table, table_backend))

        return cls(spawned=spawned, groups=groups, max_passes=max_passes)

    @staticmethod
    def _floor_pieces(spawned: SpawnedRoom) -> dict[int, Body]:
        """
        Collect every movable floor piece of *spawned* -- free objects first,
        then shelf corpuses, then table bodies -- keyed by a running index, so
        they form a single mutually-collision-checked group.
        """
        pieces: list[Body] = [
            *spawned.object_bodies.values(),
            *(spawned_shelf.corpus for spawned_shelf in spawned.spawned_shelves),
            *(spawned_table.table for spawned_table in spawned.spawned_tables),
        ]
        return dict(enumerate(pieces))

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

    @staticmethod
    def _chair_group(
        group: EGTableWithChairs,
        spawned: SpawnedTableWithChairs,
        backend: ProbabilisticBackend,
    ) -> SpawnedCollisionGroup:
        """
        Build the :class:`ChairGroup` for *spawned*'s chairs, which stand on the
        floor and so are not support-checked, resampled from *backend*.
        """
        return ChairGroup(
            bodies=spawned.chair_bodies,
            supporting_body=None,
            backend=backend,
            group=group,
            table=spawned.table,
        )

    def resolve(self) -> SpawnedLayout:
        """
        Repair every group until all are collision-free and supported, moving
        offending bodies in place.

        :raises LayoutResolutionError: If no valid layout is reached within
            :attr:`max_passes` passes.
        :return: The spawned, repaired layout.
        """
        detector = FCLCollisionDetector(_world=self.spawned.world)
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
                self.groups[group_index].resample_and_move(violations)
        raise LayoutResolutionError(
            remaining_groups=frozenset(remaining),
            passes_attempted=self.max_passes,
        )
