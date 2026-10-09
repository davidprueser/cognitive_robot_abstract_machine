"""
Spawned shelves of chair meshes, for testing what happens once real geometry stands.
"""

from __future__ import annotations

from pathlib import Path

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import scene_with_chair_meshes


# %% spawned shelves
def chair_candidate(scenes_root: Path) -> MeshCandidate:
    """
    :param scenes_root: Directory to lay the scene of the chair out in.
    :return: A book-typed mesh candidate backed by the bundled chair mesh.
    """
    scene_directory = scene_with_chair_meshes(scenes_root / "chair_scene", ["chair"])
    return MeshCandidate(scene_directory, "chair", ObjectType.BOOK)


def chair_at(x: float, y: float) -> RelationalCircuitExperimentObject2D:
    """
    :return: A book-typed object at *x*, *y* that spawns as the chair mesh.
    """
    return RelationalCircuitExperimentObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.3, y=0.3, z=0.4),
        pose=Pose2D(x=x, y=y, yaw=0.0),
        source_id="chair",
    )


def spawned_single_layer_shelf(
    objects: list[RelationalCircuitExperimentObject2D],
    candidate: MeshCandidate,
    scale: Scale | None = None,
) -> RelationalCircuitExperimentShelf:
    """
    :param objects: The objects on the only layer.
    :param candidate: The mesh every object spawns as.
    :param scale: The scale of the shelf; a generous four by four metres when omitted.
    :return: The shelf, spawned into a fresh world.
    """
    shelf = RelationalCircuitExperimentShelf(
        scale=scale or Scale(x=4.0, y=4.0, z=2.0),
        layers=[
            RelationalCircuitExperimentShelfLayer(
                objects=objects, theme_dominant_type=ObjectType.BOOK
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
        source_ids=[candidate],
    )
    shelf.spawn(World.create_with_root_body())
    return shelf
