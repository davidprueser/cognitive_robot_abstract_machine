from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from krrood.ormatic.utils import classproperty

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    OutlineEdge,
    RectilinearRoom,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Floor,
    Room,
    Wall,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
    RegionSpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% rooms on a storey


@dataclass(eq=False)
class PlacedRoom:
    """
    A room of a storey, together with the placement of its outline's frame on the
    storey.
    """

    name: str
    """
    The room's name, unique on its storey.
    """

    layout: RectilinearRoom
    """
    The room's outline, doorways and open edges.
    """

    storey_T_outline: HomogeneousTransformationMatrix
    """
    Placement of the outline's frame in the storey's frame: a turn about the vertical
    axis and a horizontal shift.
    """

    @cached_property
    def _planar_storey_T_outline(self) -> np.ndarray:
        """
        :return: :attr:`storey_T_outline` as a 3x3 matrix acting on homogeneous plan
            coordinates.
        """
        return self.storey_T_outline.to_np()[np.ix_([0, 1, 3], [0, 1, 3])]

    def point_in_storey(self, point: np.ndarray) -> np.ndarray:
        """
        :param point: Plan coordinates in the outline's frame.
        :return: The same point in the storey's frame.
        """
        return (self._planar_storey_T_outline @ np.array([*point, 1.0]))[:2]

    def direction_in_storey(self, direction: np.ndarray) -> np.ndarray:
        """
        :param direction: A plan direction in the outline's frame.
        :return: The same direction in the storey's frame.
        """
        return self._planar_storey_T_outline[:2, :2] @ direction

    @cached_property
    def edges(self) -> list[PlacedEdge]:
        """
        :return: The room's outline edges, in the order of :attr:`RectilinearRoom.edges`.
        """
        return [PlacedEdge(room=self, edge=edge) for edge in self.layout.edges]


@dataclass(eq=False)
class PlacedEdge:
    """
    An outline edge of a :class:`PlacedRoom`, seen from the storey.
    """

    room: PlacedRoom
    """
    The room the edge belongs to.
    """

    edge: OutlineEdge
    """
    The edge, in the room's outline frame.
    """

    @cached_property
    def start(self) -> np.ndarray:
        """
        :return: The vertex the edge leaves, in the storey's frame.
        """
        return self.room.point_in_storey(self.edge.start)

    @cached_property
    def end(self) -> np.ndarray:
        """
        :return: The vertex the edge reaches, in the storey's frame.
        """
        return self.room.point_in_storey(self.edge.end)

    @cached_property
    def direction(self) -> np.ndarray:
        """
        :return: The edge's unit direction, in the storey's frame.
        """
        return self.room.direction_in_storey(self.edge.direction)

    @cached_property
    def outward_normal(self) -> np.ndarray:
        """
        :return: The unit vector pointing away from the room, in the storey's frame.
        """
        return self.room.direction_in_storey(self.edge.outward_normal)

    @property
    def is_open(self) -> bool:
        """
        :return: Whether the room leaves this edge without a wall.
        """
        return self.room.layout.is_open(self.edge)

    @property
    def carries_shared_walls(self) -> bool:
        """
        :return: Whether this edge, rather than an edge facing it, carries the walls the
            two share: the edge whose outward normal points along the positive x- or
            y-axis of its outline frame.
        """
        return float(np.sum(self.edge.outward_normal)) > 0

    def distance_along(self, point: np.ndarray) -> float:
        """
        :param point: A point in the storey's frame.
        :return: How far ``point`` lies from :attr:`start` in the edge's direction.
        """
        return float(np.dot(point - self.start, self.direction))

    def facing(
        self, other: PlacedEdge, maximum_gap: float, maximum_overlap: float
    ) -> EdgeFacing | None:
        """
        :param other: An edge of another room.
        :param maximum_gap: The widest gap between the two edges that one wall fills.
        :param maximum_overlap: How far ``other`` may reach into this room and still
            face it, which absorbs small misalignments between outline frames.
        :return: The stretch of this edge that ``other`` faces across a gap one wall
            fills, or None if ``other`` does not face this edge.
        """
        if other.room is self.room:
            return None
        if np.dot(self.outward_normal, other.outward_normal) > -0.999:
            return None
        along, across = zip(
            *sorted(
                (
                    self.distance_along(point),
                    float(np.dot(point - self.start, self.outward_normal)),
                )
                for point in (other.start, other.end)
            )
        )
        start, end = max(min(along), 0.0), min(max(along), self.edge.length)
        if end - start < EdgeSegment.shortest_length:
            return None
        gaps = [float(np.interp(position, along, across)) for position in (start, end)]
        if min(gaps) < -maximum_overlap or max(gaps) > maximum_gap:
            return None
        return EdgeFacing(edge=other, start=start, end=end, gap=max(max(gaps), 0.0))

    def segments(self, facings: list[EdgeFacing]) -> list[EdgeSegment]:
        """
        :param facings: The stretches of this edge that other rooms' edges face.
        :return: The edge cut into consecutive pieces, each facing the nearest of
            ``facings`` that covers it, or nothing.
        """
        cuts = [0.0]
        for cut in sorted(
            {position for facing in facings for position in (facing.start, facing.end)}
            | {self.edge.length}
        ):
            if cut - cuts[-1] >= EdgeSegment.shortest_length:
                cuts.append(cut)
        cuts[-1] = self.edge.length
        segments: list[EdgeSegment] = []
        for start, end in zip(cuts, cuts[1:]):
            middle = (start + end) / 2
            covering = [
                facing for facing in facings if facing.start <= middle <= facing.end
            ]
            nearest = min(covering, key=lambda facing: facing.gap, default=None)
            if segments and segments[-1].facing is nearest:
                segments[-1].end = end
            else:
                segments.append(
                    EdgeSegment(edge=self, start=start, end=end, facing=nearest)
                )
        return segments


@dataclass(eq=False)
class EdgeFacing:
    """
    A stretch of an outline edge that an edge of another room faces across a gap narrow
    enough for one wall.
    """

    edge: PlacedEdge
    """
    The facing edge of the other room.
    """

    start: float
    """
    Where the stretch begins, as the distance along the faced edge from its start.
    """

    end: float
    """
    Where the stretch ends, as the distance along the faced edge from its start.
    """

    gap: float
    """
    The widest distance between the two edges along the stretch.
    """


@dataclass(eq=False)
class EdgeSegment:
    """
    A piece of an outline edge that faces either one edge of another room or no room at
    all.
    """

    edge: PlacedEdge
    """
    The edge the piece belongs to.
    """

    start: float
    """
    Where the piece begins, as the distance along :attr:`edge` from its start.
    """

    end: float
    """
    Where the piece ends, as the distance along :attr:`edge` from its start.
    """

    facing: EdgeFacing | None
    """
    The edge of another room across the gap, or None if no room lies there.
    """

    @classproperty
    def shortest_length(cls) -> float:
        """
        :return: The length below which a piece of an edge is merged into its neighbour,
            in metres.
        """
        return 0.001

    @classproperty
    def thinnest_shared_wall(cls) -> float:
        """
        :return: The gap below which two facing rooms count as touching, so that the
            wall between them takes a room's own wall thickness, in metres.
        """
        return 0.01

    @property
    def wall_thickness(self) -> float:
        """
        :return: The thickness of the wall outside this piece: the gap to the facing
            room, or the room's own wall thickness where no room faces it or the
            facing room touches it.
        """
        if self.facing is None or self.facing.gap < self.thinnest_shared_wall:
            return self.edge.room.layout.wall_thickness
        return self.facing.gap

    @property
    def has_wall(self) -> bool:
        """
        :return: Whether a wall stands outside this piece: neither its own edge nor the
            facing edge is left open.
        """
        return not self.edge.is_open and (
            self.facing is None or not self.facing.edge.is_open
        )

    def doorways(self) -> list[Doorway]:
        """
        :return: The doorways of both rooms that lie on this piece, merged where they
            overlap and clipped to the piece, with jambs on :attr:`edge` in its room's
            outline frame.
        """
        placed_edges = [self.edge] + ([] if self.facing is None else [self.facing.edge])
        spans = [
            (
                *(
                    self.edge.distance_along(
                        placed_edge.room.point_in_storey(jamb.to_np())
                    )
                    for jamb in (doorway.start, doorway.end)
                ),
                doorway.height,
            )
            for placed_edge in placed_edges
            for doorway in placed_edge.room.layout.doorways
            if placed_edge.edge.holds(doorway)
        ]
        clipped = sorted(
            (
                max(min(first, second), self.start),
                min(max(first, second), self.end),
                height,
            )
            for first, second, height in spans
        )
        merged: list[list[float]] = []
        for start, end, height in clipped:
            if end - start < self.shortest_length:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2] = max(merged[-1][2], height)
            else:
                merged.append([start, end, height])
        outline_edge = self.edge.edge
        return [
            Doorway(
                start=Point2(*(outline_edge.start + outline_edge.direction * start)),
                end=Point2(*(outline_edge.start + outline_edge.direction * end)),
                height=height,
            )
            for start, end, height in merged
        ]


# %% walls


@dataclass(eq=False)
class WallStretch:
    """
    A straight stretch of wall standing outside a piece of one room's outline edge, with
    the doorways cut into it.

    Where the stretch faces another room, it fills the gap between the two rooms.
    """

    segment: EdgeSegment
    """
    The piece of the outline edge the wall stands outside of.
    """

    start_extension: float = 0.0
    """
    How far the wall reaches past the start of :attr:`segment`, to close a corner.
    """

    end_extension: float = 0.0
    """
    How far the wall reaches past the end of :attr:`segment`, to close a corner.
    """

    @property
    def is_standing(self) -> bool:
        """
        :return: Whether a wall stands on this stretch: its piece of edge has a wall and
            the stretch is longer than the wall is thick. Shorter stretches are left to
            the walls closing the corners around them.
        """
        return self.segment.has_wall and self.length > self.segment.wall_thickness

    @property
    def length(self) -> float:
        """
        :return: The length of the wall along its edge.
        """
        return (
            self.segment.end
            - self.segment.start
            + self.start_extension
            + self.end_extension
        )

    @property
    def middle_along_edge(self) -> float:
        """
        :return: How far the wall's middle lies from the start of its edge, along the
            edge.
        """
        return (
            self.segment.start
            - self.start_extension
            + self.segment.end
            + self.end_extension
        ) / 2

    def _storey_T_strip(
        self, start: float, end: float, height: float = 0.0
    ) -> HomogeneousTransformationMatrix:
        """
        :param start: Where the strip begins, as the distance along the edge from its
            start.
        :param end: Where the strip ends, as the distance along the edge from its start.
        :param height: Height above the floor.
        :return: Placement of the middle of the strip as thick as the wall outside the
            edge between ``start`` and ``end``, raised by ``height``, in the storey's
            frame, with its x-axis pointing away from the room.
        """
        outline_edge = self.segment.edge.edge
        middle = (
            outline_edge.start
            + outline_edge.direction * (start + end) / 2
            + outline_edge.outward_normal * self.segment.wall_thickness / 2
        )
        return self.segment.edge.room.storey_T_outline @ (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                middle[0], middle[1], height, yaw=outline_edge.outward_yaw
            )
        )

    @property
    def touches(self) -> list[PlacedRoom]:
        """
        :return: The room whose edge the stretch stands outside of, and the room facing
            it, if any.
        """
        facing = self.segment.facing
        return [self.segment.edge.room] + ([] if facing is None else [facing.edge.room])

    def opening_slabs(self, thickness: float) -> list[Box]:
        """
        :param thickness: The thickness of the floor slab.
        :return: Floor slabs under the openings of this stretch, with their tops at
            height zero, in the storey's frame: one under each doorway, or one under
            the whole piece of edge where no wall stands.
        """
        outline_edge = self.segment.edge.edge
        spans = (
            [
                (
                    outline_edge.distance_along(doorway.start),
                    outline_edge.distance_along(doorway.end),
                )
                for doorway in self.segment.doorways()
            ]
            if self.segment.has_wall
            else [(self.segment.start, self.segment.end)]
        )
        return [
            Box(
                origin=self._storey_T_strip(start, end, -thickness / 2),
                scale=Scale(self.segment.wall_thickness, end - start, thickness),
            )
            for start, end in spans
        ]

    def spawn(self, world: World, name: str, floor: Floor) -> Wall:
        """
        Spawn the wall on ``floor`` and cut its doorways into it.

        The wall is dyed after the doorways are cut, since cutting replaces its shapes.

        :param world: The world the wall is added to.
        :param name: Name of the wall; its doorways are named after it.
        :param floor: The floor the wall stands on, whose frame is the storey's frame.
        :return: The wall annotation.
        """
        layout = self.segment.edge.room.layout
        wall = Wall.get_annotation_specification(
            name,
            Wall.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(
                    self.segment.wall_thickness, self.length, layout.wall_height
                )
            ),
        ).spawn(
            world,
            parent=floor.root,
            parent_T_self=self._storey_T_strip(
                self.segment.start - self.start_extension,
                self.segment.end + self.end_extension,
            ),
        )
        for index, doorway in enumerate(self.segment.doorways()):
            self._cut_doorway(world, f"{name}_doorway_{index}", wall, doorway)
        wall.root.dye(layout.wall_color)
        return wall

    def _cut_doorway(
        self, world: World, name: str, wall: Wall, doorway: Doorway
    ) -> Aperture:
        """
        Cut ``doorway`` into ``wall`` as an aperture.

        :return: The aperture annotation.
        """
        outline_edge = self.segment.edge.edge
        doorway_middle_along_edge = (
            outline_edge.distance_along(doorway.start)
            + outline_edge.distance_along(doorway.end)
        ) / 2
        aperture = Aperture.get_annotation_specification(
            name,
            Aperture.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(self.segment.wall_thickness, doorway.width, doorway.height)
            ),
        ).spawn(
            world,
            parent=wall.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=doorway_middle_along_edge - self.middle_along_edge,
                z=doorway.height / 2,
            ),
        )
        with world.modify_world():
            wall.add(aperture)
        return aperture


# %% storey


@dataclass
class StoreyLayout:
    """
    World-independent description of one storey of a building: rooms that each stand on
    a floor of their own, a hallway whose pieces stand on one floor together, one wall
    between rooms that face each other across a narrow gap, and a wall of each room's
    own thickness where no room faces it.

    Every floor lies at the same height. The floor under a doorway or an open side
    belongs to the hallway where the opening leads into it, and otherwise to the room
    owning the wall, so that neighbouring floors meet without a gap.
    """

    rooms: list[PlacedRoom]
    """
    The rooms of the storey, each standing on a floor of its own.
    """

    hallway: list[PlacedRoom]
    """
    The pieces of the storey's hallway, which stand on one floor together.
    """

    maximum_shared_wall_thickness: float = 0.65
    """
    The widest gap between two rooms that one wall fills.
    """

    maximum_overlap: float = 0.05
    """
    How far two rooms in differently turned outline frames may reach into each other and
    still share the wall between them.
    """

    floor_thickness: float = 0.05
    """
    Thickness of every floor slab, measured downward from the floor surface.
    """

    supporting_surface_thickness: float = 0.01
    """
    Thickness of every floor's supporting surface region, measured upward from the floor
    surface.
    """

    floor_color: Color = field(default_factory=Color.GRAY)
    """
    Color of every floor.
    """

    def wall_stretches(self) -> list[WallStretch]:
        """
        :return: The wall stretches of the storey, each shared wall once, including the
            open stretches between rooms that face each other without a wall, which
            carry floor but no wall.
        """
        placed_rooms = self.rooms + self.hallway
        edges = [edge for room in placed_rooms for edge in room.edges]
        segments = {
            edge: edge.segments(
                [
                    facing
                    for other in edges
                    if (
                        facing := edge.facing(
                            other,
                            self.maximum_shared_wall_thickness,
                            self.maximum_overlap,
                        )
                    )
                    is not None
                ]
            )
            for edge in edges
        }
        stretches = []
        for room in placed_rooms:
            for index, edge in enumerate(room.edges):
                previous_edge = room.edges[index - 1]
                next_edge = room.edges[(index + 1) % len(room.edges)]
                edge_segments = segments[edge]
                for position, segment in enumerate(edge_segments):
                    if segment.facing is not None and not edge.carries_shared_walls:
                        continue
                    if segment.facing is None and not segment.has_wall:
                        continue
                    is_first, is_last = (
                        position == 0,
                        position == len(edge_segments) - 1,
                    )
                    stretches.append(
                        WallStretch(
                            segment=segment,
                            start_extension=(
                                segments[previous_edge][-1].wall_thickness
                                if is_first and edge.edge.is_convex_at_start
                                else 0.0
                            ),
                            end_extension=(
                                segments[next_edge][0].wall_thickness
                                if is_last and edge.edge.is_convex_at_end
                                else 0.0
                            ),
                        )
                    )
        return stretches

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Room:
        """
        Materialize the storey in ``world``: a floor with its supporting surface and a
        room annotation for the hallway and for each room, and the walls with their
        doorways.

        :param world: The world the storey is added to.
        :param name: Prefix for the names of every entity the storey creates.
        :param parent: The entity every floor attaches to. If None, ``world.root`` is
            used.
        :param parent_T_self: Placement of the storey's frame in ``parent``'s frame.
            Identity if None.
        :return: The room annotation of the hallway.
        """
        stretches = self.wall_stretches()
        into_hallway = [
            stretch
            for stretch in stretches
            if any(piece in stretch.touches for piece in self.hallway)
        ]
        hallway_floor = self._spawn_floor(
            world,
            f"{name}_hallway_floor",
            parent,
            parent_T_self,
            self.hallway,
            into_hallway,
        )
        floor_of = {piece: hallway_floor for piece in self.hallway}
        for room in self.rooms:
            floor_of[room] = self._spawn_floor(
                world,
                f"{name}_{room.name}_floor",
                parent,
                parent_T_self,
                [room],
                [
                    stretch
                    for stretch in stretches
                    if stretch.segment.edge.room is room and stretch not in into_hallway
                ],
            )
        for index, stretch in enumerate(
            stretch for stretch in stretches if stretch.is_standing
        ):
            stretch.spawn(
                world, f"{name}_wall_{index}", floor_of[stretch.segment.edge.room]
            )
        hallway = Room(name=PrefixedName(f"{name}_hallway"), floor=hallway_floor)
        with world.modify_world():
            world.add_semantic_annotation(hallway)
            for room in self.rooms:
                world.add_semantic_annotation(
                    Room(name=PrefixedName(f"{name}_{room.name}"), floor=floor_of[room])
                )
        return hallway

    def _spawn_floor(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None,
        parent_T_self: HomogeneousTransformationMatrix | None,
        placed_rooms: list[PlacedRoom],
        opening_stretches: list[WallStretch],
    ) -> Floor:
        """
        Spawn a floor whose top surface covers ``placed_rooms`` and the openings of
        ``opening_stretches`` at height zero, with a supporting surface region over the
        same area.

        :return: The floor annotation.
        """
        slabs = [
            Box(
                origin=room.storey_T_outline
                @ HomogeneousTransformationMatrix.from_xyz_rpy(
                    (rectangle.min_x + rectangle.max_x) / 2,
                    (rectangle.min_y + rectangle.max_y) / 2,
                    -self.floor_thickness / 2,
                ),
                scale=Scale(rectangle.depth, rectangle.width, self.floor_thickness),
            )
            for room in placed_rooms
            for rectangle in room.layout.floor_rectangles
        ] + [
            slab
            for stretch in opening_stretches
            for slab in stretch.opening_slabs(self.floor_thickness)
        ]
        floor = Floor.get_annotation_specification(
            name, BodySpecification(name=name, shapes=ShapeCollection(slabs))
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)
        floor.root.dye(self.floor_color)
        surface_name = f"{name}_supporting_surface"
        surface = RegionSpecification(
            name=surface_name,
            shapes=ShapeCollection(
                [
                    Box(
                        origin=slab.origin
                        @ HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=(self.floor_thickness + self.supporting_surface_thickness)
                            / 2
                        ),
                        scale=Scale(
                            slab.scale.x,
                            slab.scale.y,
                            self.supporting_surface_thickness,
                        ),
                    )
                    for slab in slabs
                ]
            ),
        ).spawn(world, parent=floor.root)
        with world.modify_world():
            floor.add_supporting_surface(surface)
        return floor
