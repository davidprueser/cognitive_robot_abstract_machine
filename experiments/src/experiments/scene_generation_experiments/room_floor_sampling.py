from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING

from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.proximity_group_collision_resolution import (
    build_free_group_query,
    sample_member_count,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import NoSolutionFound

from semantic_digital_twin.scene_generation.scene_schema import (
    EGFloorPiece,
    EGObject,
    EGPoint2D,
    EGPosition,
    EGRelativePolarPose,
    EGRoom,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    EGShelf,
    EGShelfLayer,
    EGProximityGroup,
    EGWall,
    MeshCandidate,
    ObjectType,
    PlaceId,
    RoomInterior,
    RoomType,
    _MeshTypeMatcher,
)

from pathlib import Path

_LAYER_SLAB_HEIGHT = 0.02
"""
Thickness, in metres, of a shelf layer's own slab, matching the height shelf
layers are extracted with.
"""


def sample_shelf_layer_count(training_layer_counts: list[int]) -> int:
    """
    Draw how many layers a shelf has from the empirical distribution.

    Mirrors :func:`sample_member_count`: a collection's length is a structural
    property of the sampling query, so it is drawn before the query is built.
    Fixing it made every generated shelf identical.

    :param training_layer_counts: Layer counts observed in the training shelves.
    :return: The drawn layer count.
    """
    return random.choice(training_layer_counts)


def sample_objects_per_layer(training_objects_per_layer: list[int]) -> int:
    """
    Draw how many objects a shelf layer holds from the empirical distribution.

    :param training_objects_per_layer: Object counts observed per training layer.
    :return: The drawn object count.
    """
    return random.choice(training_objects_per_layer)

_WALL_THICKNESS = 0.1
"""
Thickness, in metres, of the walls enclosing a generated room.
"""


@dataclasses.dataclass(frozen=True)
class SampledRoomComposition:
    """
    The structural choices a room-floor query must fix before it can be built:
    how large the room's floor is and what kinds of pieces stand on it.

    Drawing the composition rather than only a piece count is what makes a
    generated room hold the furniture its kind of room actually holds. Left to
    the circuit alone, each piece's type is drawn independently from the pooled
    marginal, so how many shelves or groups a room ends up with swings widely
    between samples -- one draw yields four groups, the next none at all.
    """

    scale: EGScale
    """
    Footprint of the room floor the pieces are placed on.
    """

    object_types: list[ObjectType]
    """
    The kind of each piece to place, in query slot order.
    """

    @property
    def piece_count(self) -> int:
        """
        Number of floor pieces to place in the room.
        """
        return len(self.object_types)


def sample_room_composition(
    training_layouts: list[EGRoomFloorLayout],
) -> SampledRoomComposition:
    """
    Draw a room's footprint and piece composition together from the empirical
    distribution observed in the training rooms.

    Mirrors :func:`sample_member_count`: an exchangeable relation's list length is
    a structural property of the sampling query, so it is drawn before the query
    is built. The footprint and the composition are drawn from the *same*
    training layout because they are correlated in the data; drawing them
    independently would readily place thirty pieces in a two-metre room, or a
    dishwasher in a room with no counter.

    :param training_layouts: The room floor layouts used for training.
    :return: The drawn footprint and piece composition.
    """
    layout = random.choice(training_layouts)
    return SampledRoomComposition(
        scale=layout.scale,
        object_types=[piece.object_type for piece in layout.pieces],
    )


@dataclasses.dataclass(frozen=True)
class PlacedFloorPiece:
    """
    A sampled floor piece resolved back into the room-centred frame.

    The circuit samples an :class:`EGFloorPiece`, whose pose is relative to a
    wall. Everything downstream -- shelf and anchor assembly, mesh placement,
    collision repair -- works in absolute room coordinates, so the conversion
    happens once here rather than at each use.
    """

    object_type: ObjectType
    """
    The category of the piece.
    """

    scale: EGScale
    """
    Physical dimensions of the piece.
    """

    position: EGPoint2D
    """
    Position of the piece relative to the room centre.
    """

    orientation: EGRotation
    """
    Absolute orientation of the piece, in degrees.
    """

    @classmethod
    def from_floor_piece(cls, piece: EGFloorPiece, room_scale: EGScale) -> PlacedFloorPiece:
        """
        Resolve *piece*'s wall-relative pose against the room it stands in.

        :param piece: The sampled floor piece.
        :param room_scale: Footprint of the room the piece stands in.
        :return: The piece in room-centred coordinates.
        """
        x, y, yaw = piece.pose.to_absolute_pose(room_scale)
        return cls(
            object_type=piece.object_type,
            scale=piece.scale,
            position=EGPoint2D(x=x, y=y),
            orientation=EGRotation(x=0.0, y=0.0, z=yaw),
        )


def _rectangular_walls(scale: EGScale) -> list[EGWall]:
    """
    Build four walls enclosing a *scale*-sized rectangle centred on the room
    origin, so the room has a floor for its pieces to rest on.

    :param scale: The room floor footprint.
    :return: The four enclosing walls, each running start → end with
        non-decreasing coordinates so its length stays positive.
    """
    half_width = scale.width / 2
    half_length = scale.length / 2
    edges = [
        ((-half_width, -half_length), (half_width, -half_length)),
        ((half_width, -half_length), (half_width, half_length)),
        ((-half_width, half_length), (half_width, half_length)),
        ((-half_width, -half_length), (-half_width, half_length)),
    ]
    return [
        EGWall(
            id=f"wall_{index}",
            start_point=EGPoint2D(x=start[0], y=start[1]),
            end_point=EGPoint2D(x=end[0], y=end[1]),
            height=scale.height,
            thickness=_WALL_THICKNESS,
        )
        for index, (start, end) in enumerate(edges)
    ]


def _sampled_layer(
    shelf_backend: ProbabilisticBackend, object_count: int, layer_scale: EGScale
) -> EGShelfLayer:
    """
    Draw one shelf layer sized to *layer_scale*.

    The layer is conditioned on the footprint of the piece the room layout
    placed, so the shelf that spawns is the size the room was arranged around --
    :meth:`EGShelf.spawn_in_world` derives its corpus from the layers, not from
    the shelf's own scale. A footprint the shelf circuit never saw carries no
    probability mass, so the draw falls back to a free-scale layer whose scale is
    then overwritten, rather than failing the whole room.

    :param shelf_backend: The single-sample backend over the shelf circuit.
    :param object_count: Number of objects to draw for the layer.
    :param layer_scale: The footprint the layer must have.
    :return: The drawn layer, sized to *layer_scale*.
    """
    try:
        sampled = next(
            iter(
                shelf_backend.evaluate(
                    build_layer_query_with_fixed_scale(object_count, layer_scale)
                )
            )
        )
    except NoSolutionFound:
        sampled = next(
            iter(shelf_backend.evaluate(build_free_layer_query(object_count)))
        )
    return dataclasses.replace(sampled, scale=layer_scale)


def _sampled_shelf(
    piece: PlacedFloorPiece,
    shelf_backend: ProbabilisticBackend,
    source_ids: list[MeshCandidate],
    training_layer_counts: list[int],
    training_objects_per_layer: list[int],
) -> EGShelf:
    """
    Build an :class:`EGShelf` for a sampled shelf *piece*, filling it with layers
    drawn from the shelf circuit so the furniture samples its own contents.

    The corpus takes its footprint from the *piece*, and the layers are
    conditioned on that footprint. Taking it from the layer circuit instead made
    a shelf's size unrelated to the piece the room layout placed, so the room was
    arranged around one footprint and drawn with another.

    :param piece: The sampled floor piece standing for a shelf.
    :param shelf_backend: The single-sample backend over the shelf circuit.
    :param source_ids: Mesh candidates for the shelf's sampled contents.
    :param training_layer_counts: Observed layer counts, for drawing how many
        layers this shelf gets.
    :param training_objects_per_layer: Observed object counts per layer.
    :return: The populated shelf, placed at the piece's floor pose.
    """
    layer_scale = EGScale(
        width=piece.scale.width,
        length=piece.scale.length,
        height=_LAYER_SLAB_HEIGHT,
    )
    layer_count = sample_shelf_layer_count(training_layer_counts)
    layers = [
        _sampled_layer(
            shelf_backend,
            sample_objects_per_layer(training_objects_per_layer),
            layer_scale,
        )
        for _ in range(layer_count)
    ]
    return EGShelf(
        position=EGPoint2D(x=piece.position.x, y=piece.position.y),
        scale=EGScale(
            height=piece.scale.height,
            length=piece.scale.length,
            width=piece.scale.width,
        ),
        orientation=piece.orientation,
        layers=layers,
        source_ids=source_ids,
    )


def _sampled_group(
    piece: PlacedFloorPiece,
    group_backend: ProbabilisticBackend,
    member_count: int,
    source_ids: list[MeshCandidate],
    anchor_mesh: MeshCandidate,
) -> EGProximityGroup:
    """
    Build an :class:`EGProximityGroup` around a sampled anchor *piece*, drawing
    its members from the group circuit.

    This is what puts one object in relation to another: the members are posed
    relative to the anchor, so they land beside the thing they belong with
    rather than wherever the room's own piece marginal happens to put them.
    Their poses come from the circuit fitted over clusters found in the training
    rooms, so the arrangements are the ones that setting actually holds.

    :param piece: The sampled floor piece standing for the anchor.
    :param group_backend: The single-sample backend over the group circuit.
    :param member_count: Number of members to draw, at least one.
    :param source_ids: Mesh candidates for the sampled members.
    :param anchor_mesh: The mesh matched to the anchor itself.
    :return: The populated proximity group, placed at the piece's pose.
    """
    sampled = next(iter(group_backend.evaluate(build_free_group_query(member_count))))
    return EGProximityGroup(
        position=EGPoint2D(x=piece.position.x, y=piece.position.y),
        scale=EGScale(
            width=piece.scale.width,
            length=piece.scale.length,
            height=piece.scale.height,
        ),
        orientation=piece.orientation,
        object_type=piece.object_type,
        members=sampled.members,
        source_ids=source_ids,
        anchor_mesh=anchor_mesh,
    )


def _height_clamped(piece: PlacedFloorPiece, max_height: float) -> PlacedFloorPiece:
    """
    Return *piece* with its height clamped to *max_height*.

    Nothing downstream checks a piece's height against its room at collision
    time, so a piece the RSPN sampled taller than the room's own ceiling (a
    lamp taller than the walls) would otherwise spawn poking through it.

    :param piece: The sampled floor piece.
    :param max_height: The room's ceiling height.
    :return: *piece* unchanged if already within bounds, otherwise a copy
        with a clamped scale.
    """
    if piece.scale.height <= max_height:
        return piece
    return dataclasses.replace(
        piece, scale=dataclasses.replace(piece.scale, height=max_height)
    )


def _resized_to_mesh(
    piece: PlacedFloorPiece, candidate: MeshCandidate
) -> PlacedFloorPiece:
    """
    Return *piece* carrying the real size of the mesh chosen for it.

    Meshes spawn at identity scale because the sage10k meshes already carry
    their real-world size, so the size the circuit sampled is not what ends up
    in the world. Leaving the sampled size on the piece made collision
    resolution, height clamping and containment all reason about dimensions
    nothing in the world had.

    :param piece: The placed floor piece.
    :param candidate: The mesh selected for it.
    :return: *piece* with the candidate's real extents, or unchanged when the
        candidate's size is unknown.
    """
    if candidate.native_extents is None:
        return piece
    width, length, height = candidate.native_extents
    return dataclasses.replace(
        piece, scale=EGScale(width=width, length=length, height=height)
    )


def _pushed_inside_room(
    piece: PlacedFloorPiece, interior: RoomInterior
) -> PlacedFloorPiece:
    """
    Return *piece* moved just far enough that its yaw-rotated footprint clears
    the room's walls.

    A wall-relative pose bounds a piece's *centre*, not its extent, so a piece
    standing the measured 0.25 m from a wall still cuts into it once it is
    deeper than that -- and a free object adopts the real extents of the mesh
    chosen for it only after its pose was drawn, so a mesh wider than the
    sampled size reaches further still.

    :param piece: The placed floor piece.
    :param interior: The region of the room its centre may occupy.
    :return: *piece* unchanged when it already fits, otherwise a copy pushed in.
    """
    x, y = interior.contained_position(
        piece.position.x, piece.position.y, piece.scale, piece.orientation.z
    )
    return dataclasses.replace(piece, position=EGPoint2D(x=x, y=y))


def _shelf_pushed_inside_room(shelf: EGShelf, interior: RoomInterior) -> EGShelf:
    """
    Return *shelf* moved just far enough that its spawned corpus clears the
    room's walls.

    A shelf's corpus is padded beyond the piece footprint the room layout was
    contained against, so containing the piece alone still leaves the corpus
    reaching through the wall by that pad -- enough for the collision check to
    flag the shelf every repair pass.

    :param shelf: The assembled shelf.
    :param interior: The region of the room its centre may occupy.
    :return: *shelf*, repositioned in place when it did not already fit.
    """
    x, y = interior.contained_position(
        shelf.position.x, shelf.position.y, shelf.corpus_footprint, shelf.orientation.z
    )
    shelf.position = EGPoint2D(x=x, y=y)
    return shelf


def _group_members_pushed_inside_room(
    group: EGProximityGroup, interior: RoomInterior
) -> EGProximityGroup:
    """
    Return *group* with every member moved just far enough to stay inside the
    room.

    A member's pose is polar and relative to its anchor, so nothing about it is
    bounded by the room the anchor stands in -- an anchor near a wall throws its
    members straight through it. Each member is contained in absolute
    coordinates and its relative pose rebuilt from the result, so the group
    keeps describing itself in the frame the circuit learned.

    :param group: The assembled group, already placed at its anchor's pose.
    :param interior: The region of the room a piece's centre may occupy.
    :return: *group*, with its members' relative poses corrected in place.
    """
    for member in group.members:
        member_x, member_y, member_yaw = member.relative_pose.to_absolute_pose(
            group.position.x, group.position.y, group.orientation.z
        )
        contained_x, contained_y = interior.contained_position(
            member_x, member_y, member.scale, member_yaw
        )
        if (contained_x, contained_y) == (member_x, member_y):
            continue
        member.relative_pose = EGRelativePolarPose.from_absolute_poses(
            contained_x,
            contained_y,
            member_yaw,
            group.position.x,
            group.position.y,
            group.orientation.z,
        )
    return group


def _free_object(
    piece: PlacedFloorPiece,
    object_index: int,
    candidate: MeshCandidate,
    room_id: str,
) -> EGObject:
    """
    Build a free-standing floor :class:`EGObject` for a sampled *piece* that is
    neither a shelf nor a group anchor, resolving its mesh from *candidate*.

    The RSPN never samples a usable ``id``/``source_id`` for a piece -- both
    are fixed to ``None`` in the free-object query, since the circuit only
    models the spatial fields -- so both are drawn fresh here instead, the
    same way shelf and anchor contents get their mesh from a candidate pool
    rather than from the piece itself.

    :param piece: The sampled floor piece.
    :param object_index: Index used to build a unique id for the object.
    :param candidate: The mesh candidate matched to the piece's object type.
    :return: The free floor object, placed at the piece's pose.
    """
    return EGObject(
        id=f"free_object_{object_index}",
        room_id=room_id,
        place_id=PlaceId.FLOOR,
        object_type=piece.object_type,
        scale=piece.scale,
        position=EGPosition(x=piece.position.x, y=piece.position.y, z=0.0),
        orientation=piece.orientation,
        source_id=candidate.source_id,
    )


@dataclasses.dataclass(frozen=True)
class RoomGenerationReport:
    """
    What became of each piece the circuit sampled for a room.

    Generated rooms come out sparser than the layout the circuit drew, because
    a piece with no suitable mesh is silently skipped and the collision resolver
    drops whatever it cannot separate. Without a count of those, an empty-looking
    room cannot be attributed to the model or to the pipeline.
    """

    sampled_pieces: int
    """
    Pieces the circuit drew for the room.
    """

    dropped_without_matching_mesh: int
    """
    Pieces skipped because the cache held no mesh of their type close enough to
    their sampled size.
    """

    shelves: int
    """
    Shelves assembled from the sampled pieces.
    """

    groups: int
    """
    Table-with-members groups assembled from the sampled pieces.
    """

    free_objects: int
    """
    Free-standing floor objects assembled from the sampled pieces.
    """

    @property
    def built_pieces(self) -> int:
        """
        Pieces that made it into the room.
        """
        return self.shelves + self.groups + self.free_objects

    def summary(self) -> str:
        """
        A one-line, human-readable account of the sampled pieces' fate.
        """
        return (
            f"{self.sampled_pieces} pieces sampled -> {self.built_pieces} built "
            f"({self.shelves} shelves, {self.groups} groups, "
            f"{self.free_objects} free objects); "
            f"{self.dropped_without_matching_mesh} dropped for want of a mesh"
        )


@dataclasses.dataclass(frozen=True)
class BuiltRoom:
    """
    An assembled room together with what it took to build it.
    """

    room: EGRoom
    """
    The assembled room.
    """

    object_id_to_mesh_path: dict[str, Path]
    """
    For each free object, the scene directory its mesh is read from. Several
    objects commonly map to the same directory.
    """

    report: RoomGenerationReport
    """
    What became of each sampled piece.
    """


def build_room_from_floor_layout(
    layout: EGRoomFloorLayout,
    shelf_backend: ProbabilisticBackend,
    group_backend: ProbabilisticBackend,
    member_counts_by_anchor_type: dict[ObjectType, list[int]],
    shelf_source_ids: list[MeshCandidate],
    member_source_ids: list[MeshCandidate],
    free_object_source_ids: list[MeshCandidate],
    room_type: RoomType = RoomType.LIVING_ROOM,
    training_layer_counts: list[int] | None = None,
    training_objects_per_layer: list[int] | None = None,
) -> BuiltRoom:
    """
    Turn a sampled floor *layout* into a spawnable :class:`EGRoom`: each shelf
    and anchor piece samples its own contents, and every other piece becomes a
    free floor object.

    :param layout: The sampled room floor layout.
    :param shelf_backend: Backend over the shelf circuit, for shelf contents.
    :param group_backend: Backend over the anchor circuit, for anchor members.
    :param member_counts_by_anchor_type: Observed member counts per anchor type,
        for drawing how many members each anchor gathers. Conditioning on the
        type is what keeps a refrigerator standing alone while a dining table
        gathers chairs.
    :param shelf_source_ids: Mesh candidates for shelf contents.
    :param member_source_ids: Mesh candidates for members.
    :param free_object_source_ids: Mesh candidates for free floor objects,
        matched to each piece by its sampled object type. A piece is dropped
        when this pool is empty, since it could otherwise never be spawned.
    :param room_type: The category the assembled room is labelled with.
    :param training_layer_counts: Observed shelf layer counts, for drawing how
        many layers each generated shelf gets. Defaults to a single four-layer
        shelf when omitted.
    :param training_objects_per_layer: Observed object counts per shelf layer.
        Defaults to three when omitted.
    :return: The assembled room, its free objects' mesh directories, and a
        report of what became of each sampled piece.
    """
    training_layer_counts = training_layer_counts or [4]
    training_objects_per_layer = training_objects_per_layer or [3]
    mesh_matcher = _MeshTypeMatcher(candidates=free_object_source_ids)
    shelves: list[EGShelf] = []
    groups: list[EGProximityGroup] = []
    free_objects: list[EGObject] = []
    object_id_to_mesh_path: dict[str, Path] = {}
    room_id = "room_1"
    dropped_without_matching_mesh = 0
    interior = RoomInterior(scale=layout.scale, wall_thickness=_WALL_THICKNESS)
    for sampled_piece in layout.pieces:
        piece = _pushed_inside_room(
            _height_clamped(
                PlacedFloorPiece.from_floor_piece(sampled_piece, layout.scale),
                layout.scale.height,
            ),
            interior,
        )
        if piece.object_type == ObjectType.SHELF:
            shelves.append(
                _shelf_pushed_inside_room(
                    _sampled_shelf(
                        piece,
                        shelf_backend,
                        shelf_source_ids,
                        training_layer_counts,
                        training_objects_per_layer,
                    ),
                    interior,
                )
            )
            continue
        if not free_object_source_ids:
            continue
        candidate = mesh_matcher.random_match(
            piece.object_type, target_extents=piece.scale
        )
        if candidate is None:
            dropped_without_matching_mesh += 1
            continue
        piece = _pushed_inside_room(_resized_to_mesh(piece, candidate), interior)
        member_count = sample_member_count(
            member_counts_by_anchor_type, piece.object_type
        )
        if member_count == 0:
            free_object = _free_object(piece, len(free_objects), candidate, room_id)
            free_objects.append(free_object)
            object_id_to_mesh_path[free_object.id] = candidate.scene_dir
            continue
        groups.append(
            _group_members_pushed_inside_room(
                _sampled_group(
                    piece, group_backend, member_count, member_source_ids, candidate
                ),
                interior,
            )
        )

    room = EGRoom(
        id=room_id,
        room_type=room_type,
        scale=EGScale(
            width=layout.scale.width,
            length=layout.scale.length,
            height=layout.scale.height,
        ),
        position=EGPosition(x=0.0, y=0.0, z=0.0),
        objects=free_objects,
        walls=_rectangular_walls(layout.scale),
        shelves=shelves,
        groups=groups,
    )
    return BuiltRoom(
        room=room,
        object_id_to_mesh_path=object_id_to_mesh_path,
        report=RoomGenerationReport(
            sampled_pieces=len(layout.pieces),
            dropped_without_matching_mesh=dropped_without_matching_mesh,
            shelves=len(shelves),
            groups=len(groups),
            free_objects=len(free_objects),
        ),
    )
