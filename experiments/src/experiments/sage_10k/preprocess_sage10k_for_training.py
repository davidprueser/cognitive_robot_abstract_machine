from __future__ import annotations

import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.utils import build_source_id_to_path
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.scene_generation.object_type_classifier import (
    ObjectTypeClassifier,
)
from semantic_digital_twin.scene_generation.shelf_membership_classifier import (
    ShelfMembershipClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGObject2D,
    EGPoint2D,
    EGPosition,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    ObjectType,
    ObjectTypeAffinity,
    ObjectTypeHeightProfile,
    wrap_angle_degrees,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale

COMMIT_BATCH_SIZE = 500

STREAM_CHUNK_SIZE = 2000
"""
Rows fetched per round trip while reading the raw dataset.

The dataset does not fit in memory as a whole, so it is walked in chunks and each object
is written out and let go of before the next arrives.
"""

LAYER_CLUSTERING_TOLERANCE = 0.05
"""
Largest height difference, in metres, between two objects still considered to be
standing on the same shelf layer.
"""

DEFAULT_EDGE_MARGIN_FRACTION = 0.10
"""
Fraction of a shelf's width and length kept free at its edges when deciding whether an
object really stands on it, so a learned layout never places an object where it would
protrude.
"""

MAX_MESH_MEASUREMENT_WORKER_COUNT = 128
"""
Upper bound on parallel workers for mesh measurement, independent of the host's core
count.

Measurement is disk-I/O bound, so beyond a point more workers thrash the disk instead of
finishing faster.
"""

MAX_OBJECT_PASS_WORKER_COUNT = 32
"""
Upper bound on parallel workers for the read-convert-write object pass.

The constraint here is concurrent write throughput to the processed database, not CPU,
so this stays far below the host's core count.
"""

MESH_MEASUREMENT_CHUNK_SIZE = 1000
"""
Source ids handed to one worker process per round trip during parallel mesh measurement.

Too small and inter-process communication dominates; too large and work balances poorly
across workers.
"""


def _available_worker_count(cap: int) -> int:
    """
    How many worker processes to use for a parallel pass, bounded by *cap*.

    :param cap: The most workers ever worth using for this pass, regardless of how many
        cores the host has.
    :return: The worker count to use.
    """
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    return min(available, cap)


@dataclass(frozen=True)
class CorrectedPosition:
    """
    An object's horizontal position after the mesh-centring correction, with the
    provenance of that position.
    """

    position: EGPoint2D
    """
    The object's position in world coordinates.
    """

    is_mesh_corrected: bool
    """
    Whether :attr:`position` is the mesh's measured bounding-box centre rather than the
    dataset's unmodified recorded position.
    """


@dataclass(frozen=True)
class VerticalExtent:
    """
    How far a mesh reaches below and above its own origin.

    Lets a shelf's real base and top be located in the world, which the
    recorded position alone does not give: it is the mesh's origin, and the
    dataset does not say where in the mesh that origin sits.
    """

    bottom: float
    """
    Lowest point of the mesh, in its own frame.
    """

    top: float
    """
    Highest point of the mesh, in its own frame.
    """

    @property
    def height(self) -> float:
        """
        The mesh's total vertical size.
        """
        return self.top - self.bottom


@dataclass(frozen=True)
class MeshBounds:
    """
    The measurements taken from one mesh: where its horizontal centre sits relative to
    its origin, and how far it reaches vertically.
    """

    footprint_center: EGPoint2D
    """
    Horizontal centre of the mesh's bounding box, in the mesh's own frame.
    """

    vertical_extent: VerticalExtent
    """
    The mesh's vertical reach, in its own frame.
    """


@dataclass
class MeshMeasurements:
    """
    Measures cached meshes, so an object's recorded position can be corrected onto its
    mesh's true horizontal centre and a shelf's real base and top can be located.

    A sage10k object's recorded position is its mesh's local origin, which the dataset
    does not guarantee to be that mesh's centre -- much as a room's recorded position is
    its lower-left corner.
    """

    source_id_to_path: dict[str, Path]
    """
    Maps a mesh's source id to the cached scene directory holding it, as returned by
    :func:`build_source_id_to_path`.
    """

    _bounds_by_source_id: dict[str, Optional[MeshBounds]] = field(default_factory=dict)
    """
    Memoizes each measured mesh, since many objects share one asset.

    ``None`` records that a mesh was not available to measure.
    """

    @property
    def measured_mesh_count(self) -> int:
        """
        How many distinct meshes were actually loaded and measured.
        """
        return sum(bounds is not None for bounds in self._bounds_by_source_id.values())

    def corrected_position(
        self, source_id: str, position: EGPoint2D, yaw_degrees: float
    ) -> CorrectedPosition:
        """
        Correct *position* onto the true centre of *source_id*'s mesh.

        Falls back to *position* unchanged, flagged as uncorrected, when the mesh is not
        cached locally, so preprocessing still runs on a partial mesh cache without
        silently passing off uncorrected data as corrected.

        :param source_id: Identifies the object's mesh asset.
        :param position: The object's recorded world position.
        :param yaw_degrees: The object's own yaw, which the mesh-local offset is rotated
            by to reach world axes.
        :return: The corrected position and whether the mesh supplied it.
        """
        bounds = self.bounds(source_id)
        if bounds is None:
            return CorrectedPosition(position=position, is_mesh_corrected=False)
        # rotated_into_frame(theta) is R(-theta); negating the angle gives
        # R(+theta), the forward rotation into world axes.
        world_offset = bounds.footprint_center.rotated_into_frame(-yaw_degrees)
        return CorrectedPosition(
            position=EGPoint2D(
                x=position.x + world_offset.x, y=position.y + world_offset.y
            ),
            is_mesh_corrected=True,
        )

    def vertical_extent(self, source_id: str) -> Optional[VerticalExtent]:
        """
        How far *source_id*'s mesh reaches below and above its own origin, or ``None``
        when the mesh is not cached locally.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's vertical reach, or ``None``.
        """
        bounds = self.bounds(source_id)
        return None if bounds is None else bounds.vertical_extent

    def bounds(self, source_id: str) -> Optional[MeshBounds]:
        """
        The measurements of *source_id*'s mesh, loading it on first request.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        if source_id not in self._bounds_by_source_id:
            self._bounds_by_source_id[source_id] = self._measure(source_id)
        return self._bounds_by_source_id[source_id]

    def _measure(self, source_id: str) -> Optional[MeshBounds]:
        """
        Load *source_id*'s mesh and measure its bounding box.

        :param source_id: Identifies the mesh asset to measure.
        :return: The mesh's measurements, or ``None`` when it is not cached.
        """
        return _load_mesh_bounds(source_id, self.source_id_to_path.get(source_id))


def _load_mesh_bounds(
    source_id: str, scene_directory: Optional[Path]
) -> Optional[MeshBounds]:
    """
    Load and measure *source_id*'s mesh from *scene_directory*.

    A free function rather than a method, so a parallel measurement pass can
    call it in a worker process without pickling a whole
    :class:`MeshMeasurements`.

    :param source_id: Identifies the mesh asset to measure.
    :param scene_directory: The cached scene directory holding the mesh, or
        ``None`` when it is not cached locally.
    :return: The mesh's measurements, or ``None`` when it is not cached.
    """
    if scene_directory is None:
        return None
    mesh = trimesh.load(
        str(scene_directory / "objects" / f"{source_id}.ply"), process=False
    )
    minimum_bound, maximum_bound = mesh.bounds
    # The bounds are numpy scalars, which pass for floats until PostgreSQL is handed
    # their repr instead of a number, so they are converted here rather than at every
    # place a measurement ends up in a stored field.
    return MeshBounds(
        footprint_center=EGPoint2D(
            x=float((minimum_bound[0] + maximum_bound[0]) / 2),
            y=float((minimum_bound[1] + maximum_bound[1]) / 2),
        ),
        vertical_extent=VerticalExtent(
            bottom=float(minimum_bound[2]), top=float(maximum_bound[2])
        ),
    )


def measure_meshes_in_parallel(
    source_id_to_path: dict[str, Path],
    source_ids: Iterable[str],
    worker_count: int,
    chunk_size: int = MESH_MEASUREMENT_CHUNK_SIZE,
) -> dict[str, Optional[MeshBounds]]:
    """
    Measure every mesh in *source_ids* across *worker_count* processes.

    Loading and parsing hundreds of thousands of mesh files is disk-I/O and CPU bound
    but embarrassingly parallel, since each mesh is measured independently of every
    other. Running it as its own pass, ahead of the object read-convert-write pass, is
    what lets that pass share one already-measured map instead of every worker re-
    measuring meshes another worker also happens to need.

    :param source_id_to_path: Maps a mesh's source id to its cached scene directory, as
        returned by :func:`build_source_id_to_path`.
    :param source_ids: The distinct source ids to measure.
    :param worker_count: How many worker processes to measure with.
    :param chunk_size: Source ids handed to one worker per round trip.
    :return: The measurements, by source id; ``None`` for a source id whose mesh is not
        cached locally.
    """
    source_id_list = list(source_ids)
    scene_directories = [source_id_to_path.get(sid) for sid in source_id_list]
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        measurements = pool.map(
            _load_mesh_bounds, source_id_list, scene_directories, chunksize=chunk_size
        )
    return dict(zip(source_id_list, measurements))


def eg_object_from_sage10k_object(
    sage10k_object: Sage10kObjectDAO,
    classifier: ObjectTypeClassifier,
    measurements: MeshMeasurements,
) -> EGObject:
    """
    Build the processed :class:`EGObject` equivalent of *sage10k_object*: its free-form
    type mapped onto a unified :class:`ObjectType`, its horizontal position corrected
    onto its mesh's centre, and the dataset's free text carried through for later
    placement reasoning.

    :param sage10k_object: Raw object row, with its ``position``, ``rotation`` and
        ``dimensions`` relationships already loaded.
    :param classifier: Maps the raw type string onto an :class:`ObjectType`.
    :param measurements: Supplies the mesh-centring correction.
    :return: The processed object.
    """
    corrected = measurements.corrected_position(
        source_id=sage10k_object.source_id,
        position=EGPoint2D(x=sage10k_object.position.x, y=sage10k_object.position.y),
        yaw_degrees=sage10k_object.rotation.z,
    )
    return EGObject(
        id=sage10k_object.id,
        room_id=sage10k_object.room_id,
        place_id=sage10k_object.place_id,
        object_type=classifier.classify(sage10k_object.type),
        # x is length (depth), y is width (face), z is height -- see EGShelf's
        # CONTENT_FRAME_YAW_OFFSET_DEGREES for why the corpus frame needs this axis
        # convention.
        scale=Scale(
            x=sage10k_object.dimensions.length,
            y=sage10k_object.dimensions.width,
            z=sage10k_object.dimensions.height,
        ),
        position=EGPosition(
            x=corrected.position.x,
            y=corrected.position.y,
            z=sage10k_object.position.z,
        ),
        orientation=EGRotation(
            x=sage10k_object.rotation.x,
            y=sage10k_object.rotation.y,
            z=sage10k_object.rotation.z,
        ),
        source_id=sage10k_object.source_id,
        description=sage10k_object.description,
        place_guidance=sage10k_object.place_guidance,
        position_is_mesh_corrected=corrected.is_mesh_corrected,
    )


def _is_within_shelf_footprint(
    position: EGPoint2D,
    shelf: EGObject,
    maximum_relative_x: float,
    maximum_relative_y: float,
) -> bool:
    """
    Whether *position* lies within the shelf's own footprint, inset by the caller's edge
    margin.

    The offset is rotated into the shelf's own frame first, since the bounds are
    expressed along the shelf's width and length. Testing the raw world-frame offset
    instead reads the wrong axes for any shelf whose orientation is not a multiple of
    180 degrees.

    :param position: The candidate's world-frame position.
    :param shelf: The shelf the position is tested against.
    :param maximum_relative_x: Half the shelf's width, inset by the margin.
    :param maximum_relative_y: Half the shelf's length, inset by the margin.
    :return: Whether the position falls within the inset footprint.
    """
    local_offset = EGPoint2D(
        x=position.x - shelf.position.x,
        y=position.y - shelf.position.y,
    ).rotated_into_frame(shelf.orientation.z)
    return (
        abs(local_offset.x) <= maximum_relative_x
        and abs(local_offset.y) <= maximum_relative_y
    )


def _dominant_object_type(objects: Iterable[EGObject]) -> ObjectType:
    """
    The object type that occurs most often among *objects*.

    Ties break on the type's own value, ascending, so the result is deterministic
    regardless of iteration order.

    :param objects: The objects to find the mode of. Must be non-empty.
    :return: The most frequent :class:`ObjectType` among *objects*.
    """
    counts = Counter(obj.object_type for obj in objects)
    return min(
        counts, key=lambda object_type: (-counts[object_type], object_type.value)
    )


def _object_in_content_frame(shelf: EGObject, obj: EGObject) -> EGObject2D:
    """
    Express *obj*'s pose relative to *shelf* in the shelf's content frame.

    The content frame is the shelf's own yaw plus
    :attr:`EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES`, the frame :meth:`EGShelf.spawn`
    builds its corpus in. Storing the pose in any other frame makes the contents' spread
    land on the corpus's shallow depth axis and overflow front and back.

    :param shelf: The shelf the object stands on.
    :param obj: The object whose pose is converted.
    :return: The object with a shelf-relative, content-frame pose.
    """
    content_frame_yaw = shelf.orientation.z + EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES
    local_offset = EGPoint2D(
        x=obj.position.x - shelf.position.x,
        y=obj.position.y - shelf.position.y,
    ).rotated_into_frame(content_frame_yaw)
    yaw_degrees = wrap_angle_degrees(obj.orientation.z - content_frame_yaw)
    return EGObject2D(
        object_type=obj.object_type,
        scale=obj.scale,
        pose=Pose2D(x=local_offset.x, y=local_offset.y, yaw=math.radians(yaw_degrees)),
        source_id=obj.source_id,
    )


def _layers_of_shelf(
    shelf: EGObject,
    members: list[EGObject],
    shelf_vertical_extent: VerticalExtent,
    edge_margin_fraction: float,
    measurements: MeshMeasurements,
) -> list[EGShelfLayer]:
    """
    Group the objects standing on *shelf* into its horizontal layers, ordered from the
    bottom up and each carrying where it sits in the shelf.

    Objects are assigned to a layer by clustering their heights, so the layer structure
    comes from the arrangement itself rather than from a fixed assumption about how many
    layers a shelf has.

    Only mesh-centred positions take part. A layer records each object's offset from the
    shelf's own origin, so an uncorrected *shelf* position shifts every offset on it,
    and an uncorrected *object* position shifts that object's own. Unlike the object
    table -- which keeps uncorrected rows and marks them -- layers are training data
    whose whole content is those offsets, so admitting an uncorrected one would teach a
    circuit an arrangement nobody built.

    :param shelf: The shelf whose contents are grouped.
    :param members: Objects declaring *shelf* as the place they stand on.
    :param shelf_vertical_extent: The shelf mesh's own vertical reach, which locates its
        base and top and so gives the layers their heights.
    :param edge_margin_fraction: Fraction of the shelf's width and length kept free at
        its edges.
    :param measurements: Supplies each object mesh's reach, which locates the slab an
        object rests on.
    :return: The shelf's layers, lowest first; empty when nothing qualifies.
    """
    if not shelf.position_is_mesh_corrected:
        return []

    maximum_relative_x = shelf.scale.y / 2 * (1 - edge_margin_fraction)
    maximum_relative_y = shelf.scale.x / 2 * (1 - edge_margin_fraction)
    within_bounds = [
        obj
        for obj in members
        if obj.position_is_mesh_corrected
        and _is_within_shelf_footprint(
            EGPoint2D(x=obj.position.x, y=obj.position.y),
            shelf,
            maximum_relative_x,
            maximum_relative_y,
        )
    ]
    if not within_bounds:
        return []

    theme_dominant_type = _dominant_object_type(within_bounds)
    heights = np.array([obj.position.z for obj in within_bounds]).reshape(-1, 1)
    labels = DBSCAN(eps=LAYER_CLUSTERING_TOLERANCE, min_samples=1).fit_predict(heights)

    objects_by_label: defaultdict[int, list[EGObject]] = defaultdict(list)
    for obj, label in zip(within_bounds, labels):
        objects_by_label[label].append(obj)

    ordered_groups = sorted(
        objects_by_label.values(),
        key=lambda objects: sum(obj.position.z for obj in objects) / len(objects),
    )
    # The shelf's recorded position is its mesh's origin, so its real base and
    # top follow from where the mesh reaches around that origin.
    base_height = shelf.position.z + shelf_vertical_extent.bottom
    top_height = shelf.position.z + shelf_vertical_extent.top
    # A slab sits at the underside of what stands on it, not at those objects'
    # centres. Averaging the centres would put every slab roughly half an object
    # height too high, and since spawning places slabs at the height recorded
    # here, that error compounds on each extract-and-regenerate round trip.
    slab_heights = [
        sum(_object_bottom(obj, measurements) for obj in objects) / len(objects)
        for objects in ordered_groups
    ]

    return [
        EGShelfLayer(
            objects=[_object_in_content_frame(shelf, obj) for obj in layer_objects],
            theme_dominant_type=theme_dominant_type,
            height_above_shelf_base=slab_height - base_height,
            relative_height=_relative_height(
                slab_height, base_height, shelf_vertical_extent.height
            ),
            vertical_clearance=_vertical_clearance(index, slab_heights, top_height),
        )
        for index, (layer_objects, slab_height) in enumerate(
            zip(ordered_groups, slab_heights)
        )
    ]


def _object_bottom(obj: EGObject, measurements: MeshMeasurements) -> float:
    """
    Height at which *obj* rests, which is the height of the slab beneath it.

    Falls back to the object's own origin when its mesh is not cached, which reads as an
    object of no height rather than inventing a reach for it.

    :param obj: The object standing on a slab.
    :param measurements: Supplies the mesh's vertical reach.
    :return: The height of the object's underside.
    """
    extent = measurements.vertical_extent(obj.source_id)
    if extent is None:
        return obj.position.z
    return obj.position.z + extent.bottom


def _relative_height(
    slab_height: float, base_height: float, shelf_height: float
) -> float:
    """
    Where a slab sits between its shelf's base and top, as a fraction.

    A shelf mesh of no measurable height leaves the fraction undefined, so it reads as
    sitting at the base rather than dividing by zero.

    :param slab_height: The slab's height in world coordinates.
    :param base_height: The shelf's base in world coordinates.
    :param shelf_height: The shelf's total height.
    :return: The fraction, zero at the base and one at the top.
    """
    if shelf_height <= 0:
        return 0.0
    return (slab_height - base_height) / shelf_height


def _vertical_clearance(
    index: int, slab_heights: list[float], top_height: float
) -> float:
    """
    Space above the slab at *index*, up to the next slab or, for the topmost, the
    shelf's own top.

    :param index: Position of the slab in *slab_heights*.
    :param slab_heights: Every slab's height, lowest first.
    :param top_height: The shelf's top in world coordinates.
    :return: The clearance, never negative.
    """
    surface_above = (
        slab_heights[index + 1] if index + 1 < len(slab_heights) else top_height
    )
    return max(surface_above - slab_heights[index], 0.0)


def shelves_with_layers(
    objects: list[EGObject],
    vertical_extents: dict[str, VerticalExtent],
    shelf_ids: set[str],
    measurements: MeshMeasurements,
    edge_margin_fraction: float = DEFAULT_EDGE_MARGIN_FRACTION,
    object_type: Optional[ObjectType] = None,
) -> list[EGShelf]:
    """
    Build one :class:`EGShelf` per shelf that holds something, carrying its own pose and
    its layers in order from the bottom up.

    Shelf membership comes from an object's ``place_id`` naming the shelf it stands on,
    rather than from spatial containment. Keeping the shelf itself, rather than loose
    layers, is what preserves both that grouping and the layers' order -- and lets a
    caller draw how many layers a generated shelf should have from the real
    distribution.

    Objects whose position could not be centred on their mesh are left out; see
    :func:`_layers_of_shelf`. An object that is itself classified as a shelf-like parent
    is also left out of *another* shelf's contents -- the raw dataset records a smaller
    piece of shelf-like furniture standing on a bigger one this way, and counting it as
    ordinary content teaches the circuit that shelves commonly hold other shelves.

    :param objects: Processed objects.
    :param vertical_extents: Each shelf mesh's vertical reach, by source id. A shelf
        with no entry is skipped, since its layers' heights would be guesswork.
    :param shelf_ids: Ids of the raw objects classified as shelf-like; an object absent
        from it is not treated as a shelf, and one present in it is never treated as
        another shelf's content.
    :param measurements: Supplies each object mesh's reach, used to locate slabs.
    :param edge_margin_fraction: Fraction of each shelf's width and length kept free at
        its edges.
    :param object_type: When given, only objects of this type are extracted.
    :return: The shelves that hold at least one layer.
    """
    objects_by_place_id: defaultdict[str, list[EGObject]] = defaultdict(list)
    for obj in objects:
        if obj.id in shelf_ids:
            continue
        if object_type is None or obj.object_type == object_type:
            objects_by_place_id[obj.place_id].append(obj)

    shelves = []
    for shelf in objects:
        if shelf.id not in shelf_ids or not objects_by_place_id[shelf.id]:
            continue
        vertical_extent = vertical_extents.get(shelf.source_id)
        if vertical_extent is None:
            continue
        layers = _layers_of_shelf(
            shelf,
            objects_by_place_id[shelf.id],
            vertical_extent,
            edge_margin_fraction,
            measurements,
        )
        if not layers:
            continue
        shelves.append(
            EGShelf(
                scale=Scale(
                    x=shelf.scale.x,
                    y=shelf.scale.y,
                    z=vertical_extent.height,
                ),
                layers=layers,
                theme_dominant_type=layers[0].theme_dominant_type,
            )
        )
    return shelves


@dataclass
class ShelfContents:
    """
    Holds back the objects that layer extraction reads -- the shelves and whatever
    stands on them -- while the rest of the dataset streams past.

    Extraction only ever looks at a shelf and the objects naming it as their place, so
    every other object can be written to the processed database and released straight
    away. Keeping the whole dataset instead exhausts memory long before it is written.
    """

    shelf_ids: set[str]
    """
    Ids of the raw objects classified as shelf-like.
    """

    objects: list[EGObject] = field(default_factory=list)
    """
    The kept objects, in the order they were read.
    """

    @classmethod
    def from_raw_objects(
        cls, session: Session, classifier: ShelfMembershipClassifier
    ) -> ShelfContents:
        """
        Find the shelves among the raw objects, reading only the two columns that decide
        it so the whole dataset can be scanned cheaply.

        The raw type string has to be read here rather than recovered later, since the
        generalized object type merges bookcases and open shelves into one category.

        :param session: Session on the raw sage10k database.
        :param classifier: Decides whether a raw type string is shelf-like.
        :return: Collector primed with the shelves it has to keep.
        """
        rows = session.execute(
            select(Sage10kObjectDAO.id, Sage10kObjectDAO.type).execution_options(
                yield_per=STREAM_CHUNK_SIZE
            )
        )
        return cls(
            shelf_ids={row.id for row in rows if classifier.is_shelf_like(row.type)}
        )

    def collect(self, processed_object: EGObject) -> None:
        """
        Keep *processed_object* when it is a shelf or stands on one.

        :param processed_object: An object just read from the raw dataset.
        """
        if (
            processed_object.id in self.shelf_ids
            or processed_object.place_id in self.shelf_ids
        ):
            self.objects.append(processed_object)

    def vertical_extents(
        self, measurements: MeshMeasurements
    ) -> dict[str, VerticalExtent]:
        """
        Measure the kept shelves' meshes, which is what locates their base and top.

        Shelves whose mesh is not cached are absent, and are skipped by
        :func:`shelves_with_layers` in turn.

        :param measurements: Supplies each mesh's reach.
        :return: The measured reach, by mesh source id.
        """
        shelf_source_ids = {
            obj.source_id for obj in self.objects if obj.id in self.shelf_ids
        }
        return {
            source_id: extent
            for source_id in shelf_source_ids
            if (extent := measurements.vertical_extent(source_id)) is not None
        }


@dataclass
class _AffinityAccumulator:
    """
    Running co-occurrence count and offset sum for one pair of object types.
    """

    count: int = 0
    """
    Observed pairs of objects of the two types on a common layer.
    """

    offset_sum_x: float = 0.0
    """
    Sum of the pairs' relative offsets along x.
    """

    offset_sum_y: float = 0.0
    """
    Sum of the pairs' relative offsets along y.
    """

    def add(self, offset: EGPoint2D) -> None:
        """
        Record one observed pair separated by *offset*.

        :param offset: The second object's position minus the first's.
        """
        self.count += 1
        self.offset_sum_x += offset.x
        self.offset_sum_y += offset.y

    @property
    def mean_offset(self) -> EGPoint2D:
        """
        The mean relative offset over every recorded pair.
        """
        return EGPoint2D(
            x=self.offset_sum_x / self.count, y=self.offset_sum_y / self.count
        )


def object_type_affinities(
    shelf_layers: list[EGShelfLayer],
) -> list[ObjectTypeAffinity]:
    """
    Summarise which object types were observed sharing a shelf layer, and how far apart
    they typically sat.

    Precomputing this is what lets a robot holding a book ask what usually accompanies
    books on a shelf, and roughly where, without scanning every stored layer at the
    moment it needs an answer.

    :param shelf_layers: Every extracted layer to summarise.
    :return: One entry per observed type pair, most frequent first.
    """
    accumulators: defaultdict[tuple[ObjectType, ObjectType], _AffinityAccumulator] = (
        defaultdict(_AffinityAccumulator)
    )
    for layer in shelf_layers:
        for index, first in enumerate(layer.objects):
            for second in layer.objects[index + 1 :]:
                pair = ObjectTypeAffinity.canonical_pair(
                    first.object_type, second.object_type
                )
                from_object, to_object = (
                    (first, second) if pair[0] == first.object_type else (second, first)
                )
                accumulators[pair].add(
                    EGPoint2D(
                        x=float(to_object.pose.x) - float(from_object.pose.x),
                        y=float(to_object.pose.y) - float(from_object.pose.y),
                    )
                )

    return sorted(
        (
            ObjectTypeAffinity(
                object_type_a=object_type_a,
                object_type_b=object_type_b,
                co_occurrence_count=accumulator.count,
                mean_relative_offset=accumulator.mean_offset,
            )
            for (object_type_a, object_type_b), accumulator in accumulators.items()
        ),
        key=lambda affinity: affinity.co_occurrence_count,
        reverse=True,
    )


@dataclass
class _HeightAccumulator:
    """
    Running count and height sums for one object type.
    """

    count: int = 0
    """
    Observed objects of the type.
    """

    relative_height_sum: float = 0.0
    """
    Sum of the heights at which they were found, as shelf fractions.
    """

    height_above_shelf_base_sum: float = 0.0
    """
    Sum of the heights at which they were found, in metres.
    """

    def add(self, layer: EGShelfLayer) -> None:
        """
        Record one observed object standing on *layer*.

        :param layer: The layer the object was found on.
        """
        self.count += 1
        self.relative_height_sum += layer.relative_height
        self.height_above_shelf_base_sum += layer.height_above_shelf_base


def object_type_height_profiles(
    shelf_layers: list[EGShelfLayer],
) -> list[ObjectTypeHeightProfile]:
    """
    Summarise how high on a shelf each object type was typically found.

    An object's own stored position is two-dimensional, so the height it was kept at
    lives only in its layer. Rolling that up per type is what lets a robot holding a
    book ask which shelf books belong on, without walking every stored shelf at the
    moment it needs an answer.

    :param shelf_layers: Every extracted layer to summarise.
    :return: One entry per observed type, most frequently seen first.
    """
    accumulators: defaultdict[ObjectType, _HeightAccumulator] = defaultdict(
        _HeightAccumulator
    )
    for layer in shelf_layers:
        for object_2d in layer.objects:
            accumulators[object_2d.object_type].add(layer)

    return sorted(
        (
            ObjectTypeHeightProfile(
                object_type=object_type,
                observation_count=accumulator.count,
                mean_relative_height=accumulator.relative_height_sum
                / accumulator.count,
                mean_height_above_shelf_base=accumulator.height_above_shelf_base_sum
                / accumulator.count,
            )
            for object_type, accumulator in accumulators.items()
        ),
        key=lambda profile: profile.observation_count,
        reverse=True,
    )


@dataclass
class BatchedRecordWriter:
    """
    Stores records one at a time, committing and detaching them periodically.

    Detaching, rather than only expiring, is what lets a stored record be released: an
    expired instance stays registered with the session and so stays alive for the whole
    run.
    """

    session: Session
    """
    Session on the processed database.
    """

    label: str
    """
    Name used in the progress output.
    """

    stored_count: int = 0
    """
    Records stored so far.
    """

    def store(self, record: Any) -> None:
        """
        Convert *record* to its data access object and stage it for the next commit.

        :param record: The processed record to persist.
        """
        self.session.add(to_dao(record))
        self.stored_count += 1
        if self.stored_count % COMMIT_BATCH_SIZE == 0:
            self._commit()
            print(f"  committed {self.stored_count} {self.label}")

    def store_all(self, records: Iterable[Any]) -> None:
        """
        Store every record in *records* and commit what is left over.

        :param records: The processed records to persist.
        """
        for record in records:
            self.store(record)
        self.finish()

    def finish(self) -> None:
        """
        Commit whatever has not been committed yet and report the total.
        """
        self._commit()
        print(f"Stored {self.stored_count} {self.label}.")

    def _commit(self) -> None:
        self.session.commit()
        self.session.expunge_all()


def _streamed_raw_objects(
    session: Session, room_ids: Optional[list[str]] = None
) -> Iterator[Sage10kObjectDAO]:
    """
    Walk raw objects, in chunks, with the pose relationships already loaded.

    :param session: Session on the raw sage10k database.
    :param room_ids: When given, only objects in these rooms are walked -- how the
        object pass is split into independent shards.
    :return: The raw objects, one at a time.
    """
    statement = (
        select(Sage10kObjectDAO)
        .options(
            joinedload(Sage10kObjectDAO.position),
            joinedload(Sage10kObjectDAO.rotation),
            joinedload(Sage10kObjectDAO.dimensions),
        )
        .execution_options(yield_per=STREAM_CHUNK_SIZE)
    )
    if room_ids is not None:
        statement = statement.where(Sage10kObjectDAO.room_id.in_(room_ids))
    return iter(session.scalars(statement))


def _partition_round_robin(items: list[str], partition_count: int) -> list[list[str]]:
    """
    Split *items* into *partition_count* roughly equal, interleaved groups.

    Round-robin, rather than contiguous slices, so an ordering correlated with how busy
    a room is (if any) does not concentrate onto one shard.

    :param items: The items to split.
    :param partition_count: How many groups to split into.
    :return: The groups, each in *items*'s original relative order.
    """
    partitions: list[list[str]] = [[] for _ in range(partition_count)]
    for index, item in enumerate(items):
        partitions[index % partition_count].append(item)
    return partitions


@dataclass
class ObjectPassShardResult:
    """
    What one worker's slice of the read-convert-write object pass produced.
    """

    stored_count: int
    """
    Objects written to the processed database by this shard.
    """

    corrected_count: int
    """
    Of those, how many had their position mesh-corrected.
    """

    shelf_relevant_objects: list[EGObject]
    """
    The shelf-like objects and their contents this shard encountered, held back the same
    way :class:`ShelfContents` holds them back in a single-process run.
    """


def _process_room_shard(
    room_ids: list[str],
    sage10k_database_uri: str,
    processed_database_uri: str,
    source_id_to_path: dict[str, Path],
    bounds_by_source_id: dict[str, Optional[MeshBounds]],
    shelf_ids: set[str],
    shard_label: str,
) -> ObjectPassShardResult:
    """
    Read, convert and write every object in *room_ids*, in one worker process.

    Runs in its own process with its own database connections: a connection
    opened before forking is not safe to share across processes, so this
    builds everything it needs from scratch rather than inheriting anything
    from the parent.

    :param room_ids: The rooms this shard is responsible for.
    :param sage10k_database_uri: Connection string for the raw database.
    :param processed_database_uri: Connection string for the processed
        database. Its schema must already exist -- shards write concurrently
        and never create it themselves.
    :param source_id_to_path: Maps a mesh's source id to its cached scene
        directory.
    :param bounds_by_source_id: Every mesh's measurements, computed once by
        :func:`measure_meshes_in_parallel` ahead of the object pass, so no
        shard measures a mesh another shard also happens to need.
    :param shelf_ids: Ids of the raw objects classified as shelf-like.
    :param shard_label: Distinguishes this shard's progress output from the
        other shards running alongside it.
    :return: This shard's counts and the shelf-relevant objects it kept.
    """
    sage10k_session = Session(create_engine(sage10k_database_uri))
    processed_session = Session(create_engine(processed_database_uri))
    measurements = MeshMeasurements(
        source_id_to_path=source_id_to_path,
        _bounds_by_source_id=dict(bounds_by_source_id),
    )
    classifier = ObjectTypeClassifier()
    shelf_contents = ShelfContents(shelf_ids=shelf_ids)

    object_writer = BatchedRecordWriter(
        session=processed_session, label=f"objects[{shard_label}]"
    )
    corrected_count = 0
    for sage10k_object in _streamed_raw_objects(sage10k_session, room_ids):
        processed_object = eg_object_from_sage10k_object(
            sage10k_object, classifier, measurements
        )
        corrected_count += processed_object.position_is_mesh_corrected
        shelf_contents.collect(processed_object)
        object_writer.store(processed_object)
    object_writer.finish()

    return ObjectPassShardResult(
        stored_count=object_writer.stored_count,
        corrected_count=corrected_count,
        shelf_relevant_objects=shelf_contents.objects,
    )


def process_objects_in_parallel(
    room_ids: list[str],
    sage10k_database_uri: str,
    processed_database_uri: str,
    source_id_to_path: dict[str, Path],
    bounds_by_source_id: dict[str, Optional[MeshBounds]],
    shelf_ids: set[str],
    worker_count: int,
) -> list[ObjectPassShardResult]:
    """
    Read, convert and write every raw object across *worker_count* worker processes,
    each responsible for a disjoint slice of rooms.

    A shelf and everything standing on it always share a room, so splitting on room id
    is what lets each shard decide shelf membership on its own, with no coordination
    between shards needed.

    :param room_ids: Every room id to distribute across shards.
    :param sage10k_database_uri: Connection string for the raw database.
    :param processed_database_uri: Connection string for the processed database, whose
        schema must already exist.
    :param source_id_to_path: Maps a mesh's source id to its cached scene directory.
    :param bounds_by_source_id: Every mesh's measurements, computed ahead of this pass.
    :param shelf_ids: Ids of the raw objects classified as shelf-like.
    :param worker_count: How many worker processes to split the work across.
    :return: One result per shard.
    """
    shards = [
        shard for shard in _partition_round_robin(room_ids, worker_count) if shard
    ]
    with ProcessPoolExecutor(
        max_workers=worker_count, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        futures = [
            pool.submit(
                _process_room_shard,
                shard,
                sage10k_database_uri,
                processed_database_uri,
                source_id_to_path,
                bounds_by_source_id,
                shelf_ids,
                f"shard {index + 1}/{len(shards)}",
            )
            for index, shard in enumerate(shards)
        ]
        return [future.result() for future in futures]


def preprocess_sage10k_for_training() -> None:
    """
    Read the raw sage10k layouts and store a processed, fitting-ready copy of them in
    the processed database.

    Every correction that shelf-layout fitting used to repeat on each run --
    unifying object types, centring positions on their meshes, discarding
    objects that overhang their shelf, grouping shelf contents into layers and
    expressing their poses in the shelf's content frame -- is applied once here
    instead, together with the object-type affinity summary that supports
    placement reasoning.

    Mesh measurement and the read-convert-write object pass each run across
    several worker processes: the object pass splits by room, since a shelf
    and everything standing on it always share one, so each shard can decide
    shelf membership without coordinating with the others. Only shelf/layer
    extraction and the final summary writes stay single-process, since they
    run against the already-small set of shelf-relevant objects rather than
    the whole dataset.

    The processed database is dropped and rebuilt, so a re-run replaces the
    stored dataset rather than appending a second copy of it.

    .. note::
        The stored ``description`` and ``place_guidance`` text is deliberately
        left unscored. Running
        :class:`~semantic_digital_twin.semantic_annotations.description_matching.DescriptionCategoryScorer`
        over it is a natural enrichment of this data, but it is transformer
        inference per object over the whole dataset and belongs in its own
        opt-in stage.
    """
    sage10k_database_uri = os.environ.get("SAGE10k_DATABASE_URI")
    processed_database_uri = os.environ.get("SAGE10K_PROCESSED_DATABASE_URI")
    assert (
        sage10k_database_uri is not None
    ), "Please set the SAGE10k_DATABASE_URI environment variable."
    assert (
        processed_database_uri is not None
    ), "Please set the SAGE10K_PROCESSED_DATABASE_URI environment variable."

    start = time.time()
    sage10k_engine = create_engine(sage10k_database_uri)
    sage10k_session = Session(sage10k_engine)

    processed_engine = create_engine(processed_database_uri)
    drop_database(processed_engine)
    Base.metadata.create_all(bind=processed_engine)

    shelf_contents = ShelfContents.from_raw_objects(
        sage10k_session, ShelfMembershipClassifier()
    )
    print(f"Found {len(shelf_contents.shelf_ids)} shelves among the raw objects.")

    room_ids = list(
        sage10k_session.execute(select(Sage10kObjectDAO.room_id).distinct()).scalars()
    )
    source_ids = list(
        sage10k_session.execute(select(Sage10kObjectDAO.source_id).distinct()).scalars()
    )
    source_id_to_path = build_source_id_to_path()

    # Nothing below opens a database connection before the process pools are
    # created, so nothing forked into a worker is unsafe to share.
    sage10k_session.close()
    sage10k_engine.dispose()

    mesh_measurement_worker_count = _available_worker_count(
        MAX_MESH_MEASUREMENT_WORKER_COUNT
    )
    bounds_by_source_id = measure_meshes_in_parallel(
        source_id_to_path, source_ids, mesh_measurement_worker_count
    )
    measured_count = sum(bounds is not None for bounds in bounds_by_source_id.values())
    print(
        f"Measured {measured_count}/{len(source_ids)} distinct meshes across "
        f"{mesh_measurement_worker_count} workers."
    )

    object_pass_worker_count = _available_worker_count(MAX_OBJECT_PASS_WORKER_COUNT)
    shard_results = process_objects_in_parallel(
        room_ids,
        sage10k_database_uri,
        processed_database_uri,
        source_id_to_path,
        bounds_by_source_id,
        shelf_contents.shelf_ids,
        object_pass_worker_count,
    )
    stored_count = sum(result.stored_count for result in shard_results)
    corrected_count = sum(result.corrected_count for result in shard_results)
    shelf_contents.objects = sorted(
        (obj for result in shard_results for obj in result.shelf_relevant_objects),
        key=lambda obj: obj.id,
    )
    print(
        f"Corrected {corrected_count}/{stored_count} object positions across "
        f"{object_pass_worker_count} workers against {measured_count} cached meshes."
    )

    measurements = MeshMeasurements(
        source_id_to_path=source_id_to_path, _bounds_by_source_id=bounds_by_source_id
    )
    shelves = shelves_with_layers(
        shelf_contents.objects,
        shelf_contents.vertical_extents(measurements),
        shelf_contents.shelf_ids,
        measurements,
    )
    shelf_layers = [layer for shelf in shelves for layer in shelf.layers]
    affinities = object_type_affinities(shelf_layers)
    height_profiles = object_type_height_profiles(shelf_layers)
    print(
        f"Extracted {len(shelf_layers)} layers from {len(shelves)}/"
        f"{len(shelf_contents.shelf_ids)} shelves, {len(affinities)} object-type "
        f"affinities and {len(height_profiles)} height profiles. Shelves and contents "
        f"without a cached mesh are left out, since a layer records offsets and heights "
        f"that an unmeasured mesh would falsify."
    )

    # Layers are stored through their shelf, so the grouping and the
    # bottom-to-top order survive; they remain queryable in their own right.
    processed_session = Session(processed_engine)
    BatchedRecordWriter(session=processed_session, label="shelves").store_all(shelves)
    BatchedRecordWriter(session=processed_session, label="affinities").store_all(
        affinities
    )
    BatchedRecordWriter(session=processed_session, label="height profiles").store_all(
        height_profiles
    )

    print(f"Done in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    preprocess_sage10k_for_training()
