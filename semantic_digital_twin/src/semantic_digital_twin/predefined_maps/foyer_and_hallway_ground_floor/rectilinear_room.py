from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.exceptions import (
    DoorwayOffOutlineError,
    DoorwayOnOpenEdgeError,
    NonRectilinearOutlineError,
    OpenEdgeOffOutlineError,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Aperture,
    Floor,
    Room,
    Wall,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point2
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    PlanarBoundingBox,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% plan geometry

POSITION_TOLERANCE = 1e-6
"""
Distance below which two plan coordinates count as equal, in metres.
"""


@dataclass
class Doorway:
    """
    An opening from the floor up to a given height, cut into one wall of a
    :class:`RectilinearRoom` between two jambs on the same outline edge.
    """

    start: Point2
    """
    The first jamb, on the room outline.
    """

    end: Point2
    """
    The second jamb, on the same outline edge as :attr:`start`.
    """

    height: float = 2.0
    """
    Height of the opening above the floor.
    """

    @property
    def width(self) -> float:
        """
        :return: The distance between the jambs.
        """
        return float(np.linalg.norm(self.end.to_np() - self.start.to_np()))


@dataclass
class OpenEdge:
    """
    A whole edge of a :class:`RectilinearRoom` outline on which no wall stands, such as
    where a staircase leaves the room.
    """

    start: Point2
    """
    One of the two outline vertices the edge runs between.
    """

    end: Point2
    """
    The other outline vertex the edge runs between.
    """


@dataclass(eq=False)
class OutlineEdge:
    """
    One edge of a counter-clockwise room outline, along the inner face of the wall that
    stands on it.
    """

    start: np.ndarray
    """
    The plan coordinates of the vertex the edge leaves.
    """

    end: np.ndarray
    """
    The plan coordinates of the vertex the edge reaches.
    """

    is_convex_at_start: bool
    """
    Whether the room turns left at :attr:`start`, so the outline bulges there.
    """

    is_convex_at_end: bool
    """
    Whether the room turns left at :attr:`end`, so the outline bulges there.
    """

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def direction(self) -> np.ndarray:
        """
        :return: The unit vector from :attr:`start` to :attr:`end`.
        """
        return (self.end - self.start) / self.length

    @property
    def outward_normal(self) -> np.ndarray:
        """
        :return: The unit vector pointing away from the room, which lies on the left of
            a counter-clockwise edge.
        """
        return np.array([self.direction[1], -self.direction[0]])

    @property
    def outward_yaw(self) -> float:
        """
        :return: The angle from the x-axis to :attr:`outward_normal`, in radians.
        """
        return float(np.arctan2(self.outward_normal[1], self.outward_normal[0]))

    def distance_along(self, point: Point2) -> float:
        """
        :return: How far ``point`` lies from :attr:`start` in the edge's direction.
        """
        return float(np.dot(point.to_np() - self.start, self.direction))

    def contains(self, point: Point2) -> bool:
        """
        :return: Whether ``point`` lies on this edge.
        """
        offset = point.to_np() - self.start
        across = abs(float(np.dot(offset, self.outward_normal)))
        along = self.distance_along(point)
        return (
            across < POSITION_TOLERANCE
            and -POSITION_TOLERANCE <= along <= self.length + POSITION_TOLERANCE
        )

    def has_vertices(self, start: Point2, end: Point2) -> bool:
        """
        :return: Whether this edge runs between ``start`` and ``end``, in either
            direction.
        """
        vertices = (start.to_np(), end.to_np())
        return any(
            np.allclose(self.start, first, atol=POSITION_TOLERANCE)
            and np.allclose(self.end, second, atol=POSITION_TOLERANCE)
            for first, second in (vertices, vertices[::-1])
        )

    def holds(self, doorway: Doorway) -> bool:
        """
        :return: Whether both jambs of ``doorway`` lie on this edge.
        """
        return self.contains(doorway.start) and self.contains(doorway.end)


# %% room


@dataclass
class RectilinearRoom:
    """
    World-independent description of a room whose floor plan has only axis-parallel
    edges, such as one traced from an occupancy grid map: a floor covering the outline
    and a wall standing outside each outline edge that is not left open, with doorways
    cut into them.
    """

    outline: list[Point2]
    """
    The inner faces of the walls, as the vertices of the floor polygon in order.

    Either winding is accepted; the room stores it counter-clockwise.
    """

    doorways: list[Doorway] = field(default_factory=list)
    """
    Openings cut into the walls.
    """

    open_edges: list[OpenEdge] = field(default_factory=list)
    """
    Outline edges on which no wall stands.
    """

    wall_height: float = 2.5
    """
    Height of every wall above the floor.
    """

    wall_thickness: float = 0.1
    """
    Thickness of every wall, measured outward from the outline.
    """

    floor_thickness: float = 0.05
    """
    Thickness of the floor slab, measured downward from the floor surface.
    """

    wall_color: Color = field(default_factory=Color.BEIGE)
    """
    Color of every wall.
    """

    floor_color: Color = field(default_factory=Color.GRAY)
    """
    Color of the floor.
    """

    def __post_init__(self):
        """
        Reject outlines with oblique edges, open edges that are not outline edges and
        doorways that are off the outline or on an open edge, and store the outline
        counter-clockwise.

        :raises NonRectilinearOutlineError: If an edge is neither parallel to the x-axis
            nor to the y-axis.
        :raises OpenEdgeOffOutlineError: If an open edge is not a whole outline edge.
        :raises DoorwayOffOutlineError: If the jambs of a doorway do not lie on the same
            outline edge.
        :raises DoorwayOnOpenEdgeError: If a doorway lies on an open edge.
        """
        for start, end in zip(self.outline, self.outline[1:] + self.outline[:1]):
            difference = np.abs(end.to_np() - start.to_np())
            if min(difference) > POSITION_TOLERANCE:
                raise NonRectilinearOutlineError(start=start, end=end)
        if self._signed_area < 0:
            self.outline = list(reversed(self.outline))
        for open_edge in self.open_edges:
            if not any(
                edge.has_vertices(open_edge.start, open_edge.end) for edge in self.edges
            ):
                raise OpenEdgeOffOutlineError(start=open_edge.start, end=open_edge.end)
        for doorway in self.doorways:
            if self.is_open(self.edge_of(doorway)):
                raise DoorwayOnOpenEdgeError(start=doorway.start, end=doorway.end)

    @property
    def _vertices(self) -> np.ndarray:
        """
        :return: The outline vertices as rows of plan coordinates.
        """
        return np.array([vertex.to_np() for vertex in self.outline])

    @property
    def _signed_area(self) -> float:
        """
        :return: The area of the outline, negative if it winds clockwise.
        """
        x, y = self._vertices.T
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @property
    def edges(self) -> list[OutlineEdge]:
        """
        :return: The outline edges in counter-clockwise order.
        """
        vertices = self._vertices
        previous_vertices = np.roll(vertices, 1, axis=0)
        next_vertices = np.roll(vertices, -1, axis=0)
        incoming = vertices - previous_vertices
        outgoing = next_vertices - vertices
        is_convex = (
            incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        ) > 0
        return [
            OutlineEdge(
                start=vertices[index],
                end=next_vertices[index],
                is_convex_at_start=bool(is_convex[index]),
                is_convex_at_end=bool(is_convex[(index + 1) % len(vertices)]),
            )
            for index in range(len(vertices))
        ]

    def edge_of(self, doorway: Doorway) -> OutlineEdge:
        """
        :return: The outline edge both jambs of ``doorway`` lie on.
        :raises DoorwayOffOutlineError: If no single edge holds both jambs.
        """
        for edge in self.edges:
            if edge.holds(doorway):
                return edge
        raise DoorwayOffOutlineError(start=doorway.start, end=doorway.end)

    def is_open(self, edge: OutlineEdge) -> bool:
        """
        :return: Whether no wall stands on ``edge``.
        """
        return any(
            edge.has_vertices(open_edge.start, open_edge.end)
            for open_edge in self.open_edges
        )

    @property
    def footprint(self) -> PlanarBoundingBox:
        """
        :return: The smallest axis-aligned rectangle around the outline, in the
            outline's frame.
        """
        minimum_x, minimum_y = self._vertices.min(axis=0)
        maximum_x, maximum_y = self._vertices.max(axis=0)
        return PlanarBoundingBox(
            float(minimum_x),
            float(minimum_y),
            float(maximum_x),
            float(maximum_y),
            HomogeneousTransformationMatrix(),
        )

    @property
    def floor_rectangles(self) -> list[PlanarBoundingBox]:
        """
        Split the outline into non-overlapping axis-aligned rectangles that together
        cover exactly the floor.

        :return: The rectangles, in the outline's frame.
        """
        vertices = self._vertices
        band_edges = np.unique(vertices[:, 1])
        vertical_edges = [
            edge for edge in self.edges if abs(edge.direction[0]) < POSITION_TOLERANCE
        ]
        finished: list[PlanarBoundingBox] = []
        open_rectangles: dict[tuple[float, float], PlanarBoundingBox] = {}
        for band_bottom, band_top in zip(band_edges, band_edges[1:]):
            band_middle = (band_bottom + band_top) / 2
            crossings = sorted(
                float(edge.start[0])
                for edge in vertical_edges
                if min(edge.start[1], edge.end[1])
                < band_middle
                < max(edge.start[1], edge.end[1])
            )
            intervals = list(zip(crossings[::2], crossings[1::2]))
            for interval in list(open_rectangles):
                if interval not in intervals:
                    finished.append(open_rectangles.pop(interval))
            for min_x, max_x in intervals:
                rectangle = open_rectangles.get((min_x, max_x))
                open_rectangles[(min_x, max_x)] = PlanarBoundingBox(
                    min_x,
                    float(band_bottom) if rectangle is None else rectangle.min_y,
                    max_x,
                    float(band_top),
                    HomogeneousTransformationMatrix(),
                )
        return finished + list(open_rectangles.values())

    def spawn(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Room:
        """
        Materialize this room in ``world``: a floor covering the outline, a wall outside
        each outline edge that is not open, and an aperture for each doorway.

        :param world: The world the room is added to.
        :param name: Prefix for the names of every entity this room creates.
        :param parent: The entity the room's floor attaches to. If None, ``world.root``
            is used.
        :param parent_T_self: Placement of the outline's frame in ``parent``'s frame.
            Identity if None.
        :return: The room annotation.
        """
        floor = self._spawn_floor(world, name, parent, parent_T_self)
        for index, edge in enumerate(self.edges):
            if self.is_open(edge):
                continue
            self._spawn_wall(world, f"{name}_wall_{index}", floor, edge)
        room = Room(name=PrefixedName(name), floor=floor)
        with world.modify_world():
            world.add_semantic_annotation(room)
        return room

    def _spawn_floor(
        self,
        world: World,
        name: str,
        parent: KinematicStructureEntity | None,
        parent_T_self: HomogeneousTransformationMatrix | None,
    ) -> Floor:
        """
        Spawn a floor slab whose top surface covers the outline at height zero.

        :return: The floor annotation.
        """
        floor_name = f"{name}_floor"
        slabs = ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        (rectangle.min_x + rectangle.max_x) / 2,
                        (rectangle.min_y + rectangle.max_y) / 2,
                        -self.floor_thickness / 2,
                    ),
                    scale=Scale(rectangle.depth, rectangle.width, self.floor_thickness),
                )
                for rectangle in self.floor_rectangles
            ]
        )
        floor = Floor.get_annotation_specification(
            floor_name, BodySpecification(name=floor_name, shapes=slabs)
        ).spawn(world, parent=parent, parent_T_self=parent_T_self)
        floor.root.dye(self.floor_color)
        return floor

    def _spawn_wall(
        self, world: World, name: str, floor: Floor, edge: OutlineEdge
    ) -> Wall:
        """
        Spawn the wall standing outside ``edge`` and cut the doorways on that edge into
        it.

        At a convex corner the wall reaches past the vertex by its thickness, so that
        the corner is closed. The wall is dyed after the doorways are cut, since
        cutting replaces its shapes.

        :return: The wall annotation.
        """
        start_extension = self.wall_thickness if edge.is_convex_at_start else 0.0
        end_extension = self.wall_thickness if edge.is_convex_at_end else 0.0
        middle_along_edge = (edge.length + end_extension - start_extension) / 2
        middle = (
            edge.start
            + edge.direction * middle_along_edge
            + edge.outward_normal * self.wall_thickness / 2
        )
        wall = Wall.get_annotation_specification(
            name,
            Wall.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(
                    self.wall_thickness,
                    edge.length + start_extension + end_extension,
                    self.wall_height,
                )
            ),
        ).spawn(
            world,
            parent=floor.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                middle[0],
                middle[1],
                0,
                yaw=edge.outward_yaw,
            ),
        )
        for index, doorway in enumerate(
            doorway for doorway in self.doorways if edge.holds(doorway)
        ):
            self._cut_doorway(
                world, f"{name}_doorway_{index}", wall, edge, doorway, middle_along_edge
            )
        wall.root.dye(self.wall_color)
        return wall

    def _cut_doorway(
        self,
        world: World,
        name: str,
        wall: Wall,
        edge: OutlineEdge,
        doorway: Doorway,
        wall_middle_along_edge: float,
    ) -> Aperture:
        """
        Cut ``doorway`` into ``wall`` as an aperture.

        :param wall_middle_along_edge: How far the wall's origin lies from the start of
            ``edge``, along the edge.
        :return: The aperture annotation.
        """
        doorway_middle_along_edge = (
            edge.distance_along(doorway.start) + edge.distance_along(doorway.end)
        ) / 2
        aperture = Aperture.get_annotation_specification(
            name,
            Aperture.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(self.wall_thickness, doorway.width, doorway.height)
            ),
        ).spawn(
            world,
            parent=wall.root,
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                y=doorway_middle_along_edge - wall_middle_along_edge,
                z=doorway.height / 2,
            ),
        )
        with world.modify_world():
            wall.add(aperture)
        return aperture
