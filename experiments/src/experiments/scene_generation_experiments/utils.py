from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from experiments.scene_generation_experiments.data_preprocessing import (
    Sage10kSceneDownloader,
    SourceIdNotFoundError,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    MeshCandidate,
    ObjectType,
)

from semantic_digital_twin.utils import rclpy_installed

if TYPE_CHECKING:
    from experiments.orm.ormatic_interface import EGObjectDAO

MAXIMUM_LEAF_COUNT = 20
"""
Most leaves a fitted circuit may have.

Grounding deep-copies a circuit once per sampled part, so peak memory is circuit
size times part count -- not training-set size. This bound is what keeps a
room of twenty-odd pieces groundable.

Also doubles as the ``min_samples_per_leaf`` fraction (``1 / MAXIMUM_LEAF_COUNT``)
the shelf RSPN is fit with at every level: verified against the live
``sage_processed_data_v2`` row counts (18,437 shelves / 44,609 layers / 124,800
objects), each is large enough that this leaf-count budget is always the binding
constraint, well before any per-row overfitting floor would matter.

.. note::
    Lowering this to 10 was measured to make no difference to grounding memory,
    so it is not the dial for out-of-memory failures. Peak memory tracks the
    number of pieces in the sampled room, not the size of the circuit.
"""

MINIMUM_SAMPLES_PER_QUANTILE = 200
"""
Fewest training rows a continuous variable's Nyga histogram piece may describe, passed
as ``min_samples_per_quantile`` on :class:`~probabilistic_model.probabilistic_circuit.re
lational.rspn.RelationalProbabilisticCircuit`.

Measured against the full sage10k shelf dataset (18,437 shelves / 44,609 layers /
124,800 objects): the library default of 10 let continuous variables (position,
orientation, scale) fragment into thousands of histogram pieces per fitted level,
producing a 68,893-node object-level circuit that made a single :func:`draw_shelf` call
fail to complete a grounding in over 20 minutes. Raising this to 200 shrank that circuit
to 3,532 nodes and brought grounding down to single-digit seconds --
:data:`MAXIMUM_LEAF_COUNT` alone does not bound this, since it only constrains the JPT
tree's own split count, not the per-variable histogram induction nested inside each of
its leaves.

.. note::
    Measured before ``EGObject2D.position``/``.orientation`` were collapsed into
    ``pose: Pose2D``, which reduced its continuous-variable count from five to three.
    Re-verify this bound if that circuit's node count is ever remeasured.
"""


@contextlib.contextmanager
def rclpy_node():
    """
    Context manager that initialises an rclpy node and spins it in a background thread.

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
    object_type: ObjectType | None = ObjectType.BOOK,
    downloader: Sage10kSceneDownloader | None = None,
    minimum_candidates: int = 5,
) -> list[MeshCandidate]:
    """
    Build the pool of mesh candidates for objects of *object_type* that have a local PLY
    mesh available.

    :param objects: All loaded object DAOs from the database.
    :param object_type: Only objects whose type equals this value are included. Defaults
        to :attr:`ObjectType.BOOK` to reproduce the original book-only behaviour; pass
        ``None`` to include every type.
    :param downloader: When given, scenes are downloaded on demand for matching objects
        whose mesh isn't cached locally yet, until *minimum_candidates* distinct meshes
        are available or every matching object has been tried. ``None`` skips
        downloading, so the pool is whatever is already cached.
    :param minimum_candidates: Target number of distinct meshes to have available; only
        consulted when *downloader* is given.
    :return: Pool of mesh candidates, one per matching object with a resolvable PLY
        mesh.
    """
    source_id_to_path = build_source_id_to_path()
    matching_objects = [
        obj
        for obj in objects
        if (object_type is None or obj.object_type == object_type)
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
            native_extents=(obj.scale.y, obj.scale.x, obj.scale.z),
        )
        for obj in matching_objects
        if obj.source_id in source_id_to_path and obj.scale is not None
    ]


def _ensure_minimum_mesh_pool(
    objects: list[EGObjectDAO],
    source_id_to_path: dict[str, Path],
    downloader: Sage10kSceneDownloader,
    minimum_candidates: int,
) -> None:
    """
    Download scenes for *objects* not yet in *source_id_to_path*, mutating it in place,
    until *minimum_candidates* distinct meshes are cached or every object has been
    tried.

    Not every ``source_id`` resolves in the Sage-10k database (e.g. objects from a
    different data source), so a lookup miss is skipped rather than aborting the whole
    pool.

    :param objects: Candidate objects to download meshes for.
    :param source_id_to_path: Mapping of already-cached source IDs to their scene
        directory; extended in place with newly downloaded ones.
    :param downloader: Resolves a source ID to its scene and downloads it.
    :param minimum_candidates: Target number of distinct meshes to have available, among
        *objects* specifically -- other object types already cached in
        *source_id_to_path* don't count towards it.
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
    Scan *scenes_root* and return a mapping from source_id to its scene directory.

    Each scene directory is expected to contain an ``objects/`` sub- folder with files
    named ``{source_id}.ply``.

    :param scenes_root: Root directory that contains individual scene folders.
    :return:``{source_id: scene_dir}`` for every PLY file found under any scene.
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
