from __future__ import annotations

import dataclasses
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_pose_resample_query,
    in_world_colliding_indices,
)
from experiments.scene_generation_experiments.exceptions import LayoutResolutionError
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from experiments.scene_generation_experiments.table_chair_collision_resolution import (
    build_chair_pose_resample_query,
    build_free_table_query,
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
    EGScale,
    EGWallRelativePose,
    EGShelf,
    RoomInterior,
    EGTableWithChairs,
    SpawnedLayout,
    SpawnedRoom,
    SpawnedShelf,
    SpawnedTableWithChairs,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
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
    - **The collection's own scale.** A shelf assembled from a room layout has
      its layers' scales overwritten with the sampled piece's footprint *after*
      they were drawn, so a layer routinely carries dimensions the circuit never
      saw. Relaxing only the neighbours keeps that scale pinned and fails again.

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

        new_sample = _evaluate_first_supported(
            self.backend,
            build_chair_pose_resample_query(
                fixed_chairs, resampled_chairs, self.group.scale
            ),
            build_chair_pose_resample_query([], resampled_chairs, self.group.scale),
            build_free_table_query(len(resampled_chairs)),
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
    it offends, slid along the wall it stands against.

    A piece keeps its resting height and yaw across a move: only its position
    along its wall is redrawn, so a shelf stays upright at its corpus height, a
    table keeps standing on its legs, and a cabinet stays against the wall the
    circuit put it on.
    """

    floor: Floor
    """
    The room floor annotation the pieces are support-checked against.
    """

    interior: RoomInterior = field(kw_only=True)
    """
    The region of the room a piece's centre may occupy, so an offending piece
    can be repaired in the same wall-relative frame it was sampled in without
    being slid into a wall.
    """

    _SLIDE_CANDIDATE_COUNT: ClassVar[int] = 24
    """
    How many positions along its wall a piece tries before settling.

    The wall-relative slide is not occupancy-aware by itself, unlike the floor
    sampler it replaced, whose sample space excluded the objects already placed.
    Without trying several candidates two pieces on the same wall land on each
    other over and over and the repair loop degenerates into a random search.
    """

    def resample_and_move(self, indices: set[int]) -> None:
        for index in sorted(indices):
            self._slide_along_its_wall(index)

    def _slide_along_its_wall(self, index: int) -> None:
        """
        Move the piece at *index* to a fresh position along the wall it already
        stands against, keeping which wall it uses, how far from it, and its yaw.

        Repairing in the wall-relative frame is what keeps a repaired room
        looking like the one the circuit sampled. Drawing a fresh point from the
        floor surface instead -- as this used to -- discards the sampled pose
        entirely, which is how fridges and sinks ended up standing in open
        floor: every piece caught in a collision was teleported somewhere
        uniformly random, undoing the placement the circuit had learned. Only
        ``position_along_wall`` is redrawn, being the degree of freedom that
        carries the least of what was learned.

        Several positions are tried and the one overlapping the other pieces
        least is kept, so the slide is occupancy-aware the way the floor sampler
        it replaced was.

        Each candidate is then contained within :attr:`interior`. Without that
        the repair could never clear a wall: a piece standing the measured
        0.25 m from a wall already cuts into it once it is deeper than that, and
        sliding *along* the same wall leaves the overlap untouched, so the piece
        offended every pass until the resolver gave up and dropped it. Sliding
        towards a corner walks into the perpendicular wall the same way.
        Containment moves the piece off its wall only by the amount its own
        footprint demands, so a piece that already fits keeps the distance the
        circuit drew for it.

        :param index: Key of the floor piece to reposition.
        """
        body = self.bodies[index]
        connection = body.parent_connection
        origin = connection.origin.to_np().copy()
        yaw_degrees = math.degrees(
            math.atan2(float(origin[1, 0]), float(origin[0, 0]))
        )
        pose = EGWallRelativePose.from_absolute_pose(
            float(origin[0, 3]),
            float(origin[1, 3]),
            yaw_degrees,
            self.interior.scale,
        )
        half_width, half_length = self._footprint_half_extents(body)
        # Already widened for the piece's yaw, so it is contained as-is.
        rotated_footprint = EGScale(
            width=2 * half_width, length=2 * half_length, height=0.0
        )
        obstacles = [
            self._footprint(other)
            for other_index, other in self.bodies.items()
            if other_index != index
        ]

        slid_x, slid_y = self.interior.contained_position(
            float(origin[0, 3]), float(origin[1, 3]), rotated_footprint, 0.0
        )
        least_overlap = None
        for _ in range(self._SLIDE_CANDIDATE_COUNT):
            candidate_x, candidate_y, _unused = dataclasses.replace(
                pose, position_along_wall=random.random()
            ).to_absolute_pose(self.interior.scale)
            candidate_x, candidate_y = self.interior.contained_position(
                candidate_x, candidate_y, rotated_footprint, 0.0
            )
            overlap = sum(
                self._overlap_area(
                    (candidate_x, candidate_y, half_width, half_length), obstacle
                )
                for obstacle in obstacles
            )
            if least_overlap is None or overlap < least_overlap:
                slid_x, slid_y, least_overlap = candidate_x, candidate_y, overlap
            if overlap == 0.0:
                break

        origin[0, 3] = slid_x
        origin[1, 3] = slid_y
        connection.origin = HomogeneousTransformationMatrix(
            origin, reference_frame=connection.parent
        )

    @staticmethod
    def _footprint_half_extents(body: Body) -> tuple[float, float]:
        """
        Return *body*'s footprint half-extents along the room axes, widened for
        its yaw so a rotated piece is bounded by the span it really occupies.

        :param body: The floor piece's root body.
        :return: Half-extent along x and along y, in metres.
        """
        mesh = body.collision.combined_mesh if body.collision else None
        if mesh is None:
            return 0.0, 0.0
        origin = body.parent_connection.origin.to_np()
        yaw = math.atan2(float(origin[1, 0]), float(origin[0, 0]))
        half_x, half_y = float(mesh.extents[0]) / 2, float(mesh.extents[1]) / 2
        return (
            abs(half_x * math.cos(yaw)) + abs(half_y * math.sin(yaw)),
            abs(half_x * math.sin(yaw)) + abs(half_y * math.cos(yaw)),
        )

    @classmethod
    def _footprint(cls, body: Body) -> tuple[float, float, float, float]:
        """
        Return *body*'s footprint as ``(x, y, half_x, half_y)`` in its parent
        frame.

        :param body: The floor piece's root body.
        :return: Centre and half-extents of the footprint.
        """
        origin = body.parent_connection.origin.to_np()
        half_x, half_y = cls._footprint_half_extents(body)
        return float(origin[0, 3]), float(origin[1, 3]), half_x, half_y

    @staticmethod
    def _overlap_area(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        """
        Return the overlapping area of two axis-aligned footprints.

        An axis-aligned box around each rotated footprint over-estimates the
        real overlap, which errs towards moving a piece that might just have
        fitted -- the safe direction.

        :param first: Centre and half-extents of one footprint.
        :param second: Centre and half-extents of the other.
        :return: Overlapping area in square metres, zero when they are clear.
        """
        first_x, first_y, first_half_x, first_half_y = first
        second_x, second_y, second_half_x, second_half_y = second
        shared_x = min(first_x + first_half_x, second_x + second_half_x) - max(
            first_x - first_half_x, second_x - second_half_x
        )
        shared_y = min(first_y + first_half_y, second_y + second_half_y) - max(
            first_y - first_half_y, second_y - second_half_y
        )
        return max(shared_x, 0.0) * max(shared_y, 0.0)

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

    A generated room comes out sparser than the layout it was built from, and
    without this count an empty-looking room cannot be told apart from a model
    that simply sampled few pieces. Counts offending pieces only: a dropped
    shelf or table takes its contents out of the world with it, and those were
    placeable, so counting them would charge the pipeline for chairs it had no
    trouble with.
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
                interior=RoomInterior.of_room(room),
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
            remaining = self._remaining_violations(detector)
            if not remaining:
                return {}
            for group_index, violations in remaining.items():
                self.groups[group_index].resample_and_move(violations)

        remaining = self._remaining_violations(detector)
        if not remaining:
            return {}
        self._drop_objects(remaining)
        return self._remaining_violations(detector)

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

        A dropped piece takes its whole branch with it. The floor group holds
        shelf corpuses and table bodies, whose layers, chairs and contents hang
        beneath them, so removing the piece's own body alone left every child
        parentless and the world with as many roots as the piece had
        descendants -- which the next model change refuses.

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

        A dropped floor piece takes its contents with it, and each of those
        contents is a member of its own group -- a table's chairs, a shelf
        layer's objects. A group that was not itself offending is never visited
        by the drop, so it keeps referring to bodies the world no longer holds
        and the next collision check raises instead of reporting collisions.

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
