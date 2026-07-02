from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path
from typing import Callable

from experiments.orm.ormatic_interface import EGObjectDAO
from semantic_digital_twin.scene_generation.scene_schema import ObjectType, BookObjectType

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


def _get_source_ids_for_objects(
    objects: list[EGObjectDAO],
    type_predicate: Callable[[ObjectType], bool] = BookObjectType.contains,
) -> list[tuple[Path, str]]:
    """
    Extract all (scene_dir, source_id) pairs for objects accepted by
    *type_predicate* that have a local PLY mesh available.

    :param objects: All loaded object DAOs from the database.
    :param type_predicate: Called with each object's
        :class:`ObjectType`; only objects for which this returns
        ``True`` are included. Defaults to
        :meth:`BookObjectType.contains` to reproduce the original book-
        only behaviour.
    :return: List of (scene_directory, source_id) pairs.
    """
    source_id_to_path = build_source_id_to_path()
    return [
        (source_id_to_path[obj.source_id], obj.source_id)
        for obj in objects
        if type_predicate(obj.object_type) and obj.source_id in source_id_to_path
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
