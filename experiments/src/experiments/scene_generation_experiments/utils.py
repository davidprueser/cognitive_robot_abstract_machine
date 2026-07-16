from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.orm.ormatic_interface import EGObjectDAO
from semantic_digital_twin.scene_generation.scene_schema import MeshCandidate, ObjectType

from semantic_digital_twin.utils import rclpy_installed


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
    Load all object DAOs from the database, eagerly joining their
    scale/position/orientation.

    :param session: Database session to query objects from.
    :return: All loaded object DAOs.
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


def _get_source_ids_for_objects(
    objects: list[EGObjectDAO],
    object_type: ObjectType | None = ObjectType.BOOK,
) -> list[MeshCandidate]:
    """
    Build the pool of mesh candidates for objects of *object_type* that have a
    local PLY mesh available.

    :param objects: All loaded object DAOs from the database.
    :param object_type: Only objects whose type equals this value are
        included. Defaults to :attr:`ObjectType.BOOK` to reproduce the
        original book-only behaviour; pass ``None`` to include every
        type.
    :return: Pool of mesh candidates, one per matching object with a
        resolvable PLY mesh.
    """
    source_id_to_path = build_source_id_to_path()
    return [
        MeshCandidate(
            scene_dir=source_id_to_path[obj.source_id],
            source_id=obj.source_id,
            object_type=obj.object_type,
        )
        for obj in objects
        if (object_type is None or obj.object_type == object_type)
        and obj.source_id in source_id_to_path
    ]


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
