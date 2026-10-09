"""
Synthetic shelves, a shelf model fitted on them, and scenes holding a real mesh, for
testing the shelf experiment without the sage10k dataset.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

from plyfile import PlyData

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
    RelationalCircuitExperimentShelf,
    RelationalCircuitExperimentShelfLayer,
)
from experiments.shelf_generation_experiments.training.shelf_model import (
    ShelfModel,
    ShelfModelFitSettings,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


# %% meshes
def chair_mesh_resources() -> Path:
    """
    :return: The directory holding the chair mesh bundled with semantic_digital_twin.
    """
    return Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"


def scene_with_chair_meshes(
    scene_directory: Path, source_ids: Sequence[str], size_factor: float = 1.0
) -> Path:
    """
    Lay out a scene directory holding the bundled chair mesh once per source id.

    :param scene_directory: The directory to lay the scene out in.
    :param source_ids: The source ids to store the mesh under.
    :param size_factor: How much to shrink or grow the chair, which is about 0.7 by 0.74
        by 0.88 metres.
    :return: *scene_directory*.
    """
    objects_directory = scene_directory / "objects"
    objects_directory.mkdir(parents=True)
    chair = PlyData.read(str(chair_mesh_resources() / "chair.ply"))
    for axis in ("x", "y", "z"):
        chair["vertex"].data[axis] *= size_factor
    for source_id in source_ids:
        chair.write(str(objects_directory / f"{source_id}.ply"))
        shutil.copy(
            chair_mesh_resources() / "chair_texture.png",
            objects_directory / f"{source_id}_texture.png",
        )
    return scene_directory


# %% synthetic shelves
def random_object(
    generator: random.Random, object_type: ObjectType, index: int
) -> RelationalCircuitExperimentObject2D:
    """
    :param generator: The random generator to draw from.
    :param object_type: The type of the object.
    :param index: Distinguishes the source id of the object.
    :return: An object whose scale and pose are drawn independently of each other.
    """
    return RelationalCircuitExperimentObject2D(
        object_type=object_type,
        scale=Scale(
            x=generator.uniform(0.1, 0.2),
            y=generator.uniform(0.1, 0.2),
            z=generator.uniform(0.1, 0.2),
        ),
        pose=Pose2D(
            x=generator.uniform(-0.3, 0.3),
            y=generator.uniform(-0.3, 0.3),
            yaw=generator.uniform(-3.0, 3.0),
        ),
        source_id=f"{object_type.value}_{index}",
    )


def random_shelf(
    generator: random.Random,
    theme: ObjectType,
    object_types: Sequence[ObjectType],
    layer_count: int,
    objects_per_layer: int,
) -> RelationalCircuitExperimentShelf:
    """
    :param generator: The random generator to draw from.
    :param theme: The dominant object type of the shelf.
    :param object_types: The types the objects of every layer cycle through.
    :param layer_count: How many layers the shelf has.
    :param objects_per_layer: How many objects every layer holds.
    :return: A shelf with evenly spaced layers of independently drawn objects.
    """
    shelf_height = generator.uniform(1.0, 1.4)
    return RelationalCircuitExperimentShelf(
        scale=Scale(
            x=generator.uniform(0.8, 1.0),
            y=generator.uniform(0.8, 1.0),
            z=shelf_height,
        ),
        layers=[
            RelationalCircuitExperimentShelfLayer(
                objects=[
                    random_object(
                        generator,
                        object_types[(layer_index + slot) % len(object_types)],
                        slot,
                    )
                    for slot in range(objects_per_layer)
                ],
                theme_dominant_type=theme,
                height_above_shelf_base=shelf_height
                * (layer_index + 1)
                / (layer_count + 1),
                relative_height=(layer_index + 1) / (layer_count + 1),
                vertical_clearance=generator.uniform(0.25, 0.35),
            )
            for layer_index in range(layer_count)
        ],
        theme_dominant_type=theme,
    )


def random_shelves(
    themes: Sequence[ObjectType] = (ObjectType.BOOK, ObjectType.BOTTLE),
    object_types: Sequence[ObjectType] = (
        ObjectType.BOOK,
        ObjectType.BOTTLE,
        ObjectType.BOX,
    ),
    shelves_per_theme: int = 4,
    layer_count: int = 2,
    objects_per_layer: int = 2,
    seed: int = 0,
) -> list[RelationalCircuitExperimentShelf]:
    """
    :return: Reproducible synthetic shelves, *shelves_per_theme* of every theme.
    """
    generator = random.Random(seed)
    return [
        random_shelf(generator, theme, object_types, layer_count, objects_per_layer)
        for theme in themes
        for _ in range(shelves_per_theme)
    ]


# %% fitted model
def single_leaf_fit_settings() -> ShelfModelFitSettings:
    """
    :return: Fit settings under which no leaf can be split in two, since each half
        would hold less than the share of rows a leaf needs, so every level of the
        circuit is a single product of independent distributions.
    """
    return ShelfModelFitSettings(min_samples_per_leaf=0.99)


def factorized_shelf_model(
    shelves: Sequence[RelationalCircuitExperimentShelf] | None = None,
) -> ShelfModel:
    """
    :param shelves: The shelves to fit on; :func:`random_shelves` when omitted.
    :return: A shelf model fitted on *shelves* with :func:`single_leaf_fit_settings`.
    """
    return ShelfModel.fit(
        shelves if shelves is not None else random_shelves(),
        single_leaf_fit_settings(),
    )
