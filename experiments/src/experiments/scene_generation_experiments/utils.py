from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from experiments.orm.ormatic_interface import EGObjectDAO
from experiments.scene_generation_experiments.data_preprocessing import (
    Sage10kSceneDownloader,
    SourceIdNotFoundError,
)
from semantic_digital_twin.scene_generation.scene_schema import MeshCandidate, ObjectType

from semantic_digital_twin.utils import rclpy_installed

DEFAULT_TRAINING_ROOM_COUNT = 1500
"""
Default number of distinct rooms sampled by :func:`sampled_room_ids` for RSPN
training. Selecting rooms first, then loading every object that belongs to
them via :func:`objects_for_rooms`, keeps each selected room's piece
membership complete -- unlike capping the object query directly, which
truncated most rooms' pieces long before their true membership was reached
(the dataset's true median is 23 floor pieces per room, but a flat 50000-row
cap across the whole object table left a median of just 2 pieces per room
actually represented).
"""


@contextlib.contextmanager
def rclpy_node():
    """
    Context manager that initialises an rclpy node and spins it in a background
    thread.

    :raises ValueError: If rclpy is not installed.
    """
    if not rclpy_installed():
        raise ValueError("No ros installed")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("test_node")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    thread.start()
    time.sleep(0.1)
    try:
        yield node
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


def load_all_objects(session: Session) -> list[EGObjectDAO]:
    """
    Load a broad, capped sample of object DAOs, eagerly joining their
    scale/position/orientation, for use as a mesh-candidate pool.

    Deliberately independent of :func:`sampled_room_ids`/
    :func:`objects_for_rooms`: capping the *rooms* selected for RSPN training
    must not also narrow which meshes are available to dress the sampled
    result, so callers building a mesh-candidate pool should use this
    instead of the objects an RSPN-training extractor happened to load.

    :param session: Database session to query objects from.
    :return: Loaded object DAOs.
    """
    return session.scalars(
        select(EGObjectDAO)
        .options(
            joinedload(EGObjectDAO.scale),
            joinedload(EGObjectDAO.position),
            joinedload(EGObjectDAO.orientation),
        )
        .distinct()
        .limit(50000)
    ).all()


def sampled_room_ids(
    session: Session, room_count: int = DEFAULT_TRAINING_ROOM_COUNT
) -> list[str]:
    """
    Return a random sample of up to *room_count* distinct room ids.

    Meant to be followed by :func:`objects_for_rooms`, so a bounded number of
    rooms are selected first and then loaded in full -- rather than capping
    the object query itself, which truncates almost every room's pieces long
    before its true membership is reached.

    :param session: Database session to query room ids from.
    :param room_count: Maximum number of distinct room ids to return.
    :return: Sampled room ids.
    """
    distinct_room_ids = select(EGObjectDAO.room_id).distinct().subquery()
    return list(
        session.scalars(
            select(distinct_room_ids.c.room_id)
            .order_by(func.random())
            .limit(room_count)
        ).all()
    )


def objects_for_rooms(session: Session, room_ids: Sequence[str]) -> list[EGObjectDAO]:
    """
    Load every object DAO belonging to any of *room_ids*, eagerly joining
    scale/position/orientation, with no cap on row count.

    A room's full piece membership must not be truncated, or an RSPN trained
    on the result learns an artificially sparse room composition.

    :param session: Database session to query objects from.
    :param room_ids: Room ids whose member objects should be loaded.
    :return: All matching object DAOs.
    """
    return session.scalars(
        select(EGObjectDAO)
        .where(EGObjectDAO.room_id.in_(room_ids))
        .options(
            joinedload(EGObjectDAO.scale),
            joinedload(EGObjectDAO.position),
            joinedload(EGObjectDAO.orientation),
        )
        .distinct()
    ).all()


def _get_source_ids_for_objects(
    objects: list[EGObjectDAO],
    object_type: ObjectType | None = ObjectType.BOOK,
    downloader: Sage10kSceneDownloader | None = None,
    minimum_candidates: int = 5,
) -> list[MeshCandidate]:
    """
    Build the pool of mesh candidates for objects of *object_type* that have a
    local PLY mesh available.

    :param objects: All loaded object DAOs from the database.
    :param object_type: Only objects whose type equals this value are
        included. Defaults to :attr:`ObjectType.BOOK` to reproduce the
        original book-only behaviour; pass ``None`` to include every
        type.
    :param downloader: When given, scenes are downloaded on demand for
        matching objects whose mesh isn't cached locally yet, until
        *minimum_candidates* distinct meshes are available or every
        matching object has been tried. ``None`` skips downloading, so
        the pool is whatever is already cached.
    :param minimum_candidates: Target number of distinct meshes to have
        available; only consulted when *downloader* is given.
    :return: Pool of mesh candidates, one per matching object with a
        resolvable PLY mesh.
    """
    source_id_to_path = build_source_id_to_path()
    matching_objects = [
        obj for obj in objects if object_type is None or obj.object_type == object_type
    ]
    if downloader is not None:
        _ensure_minimum_mesh_pool(
            matching_objects, source_id_to_path, downloader, minimum_candidates
        )
    return [
        MeshCandidate(
            scene_dir=source_id_to_path[obj.source_id],
            source_id=obj.source_id,
            object_type=obj.object_type,
        )
        for obj in matching_objects
        if obj.source_id in source_id_to_path
    ]


def _ensure_minimum_mesh_pool(
    objects: list[EGObjectDAO],
    source_id_to_path: dict[str, Path],
    downloader: Sage10kSceneDownloader,
    minimum_candidates: int,
) -> None:
    """
    Download scenes for *objects* not yet in *source_id_to_path*, mutating it
    in place, until *minimum_candidates* distinct meshes are cached or every
    object has been tried.

    Not every ``source_id`` resolves in the Sage-10k database (e.g. objects
    from a different data source), so a lookup miss is skipped rather than
    aborting the whole pool.

    :param objects: Candidate objects to download meshes for.
    :param source_id_to_path: Mapping of already-cached source IDs to their
        scene directory; extended in place with newly downloaded ones.
    :param downloader: Resolves a source ID to its scene and downloads it.
    :param minimum_candidates: Target number of distinct meshes to have
        available, among *objects* specifically -- other object types
        already cached in *source_id_to_path* don't count towards it.
    """
    available = {obj.source_id for obj in objects if obj.source_id in source_id_to_path}
    for obj in objects:
        if len(available) >= minimum_candidates:
            return
        if obj.source_id in available:
            continue
        try:
            source_id_to_path[obj.source_id] = downloader.download_scene_for_source_id(
                obj.source_id
            )
        except SourceIdNotFoundError:
            continue
        available.add(obj.source_id)


def build_source_id_to_path(
    scenes_root: Path = Path.home() / "Documents" / "sage-10k-scenes",
) -> dict[str, Path]:
    """
    Scan *scenes_root* and return a mapping from source_id to its scene
    directory.

    Each scene directory is expected to contain an ``objects/`` sub-
    folder with files named ``{source_id}.ply``.

    :param scenes_root: Root directory that contains individual scene
        folders.
    :return:``{source_id: scene_dir}`` for every PLY file found under
        any scene.
    """
    mapping: dict[str, Path] = {}
    for scene_dir in scenes_root.iterdir():
        objects_dir = scene_dir / "objects"
        if not objects_dir.is_dir():
            continue
        for ply_file in objects_dir.glob("*.ply"):
            texture_file = objects_dir / f"{ply_file.stem}_texture.png"
            if texture_file.exists():
                mapping[ply_file.stem] = scene_dir
    return mapping
