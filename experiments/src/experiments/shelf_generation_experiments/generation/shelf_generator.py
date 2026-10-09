from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.shelf_generation_experiments.generation.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.shelf_generation_experiments.generation.pre_spawn_resolver import (
    PreSpawnLayoutResolver,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.training.processed_database import (
    ProcessedShelfDatabase,
)
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


# %% generated shelf
@dataclass
class GeneratedShelf:
    """
    A shelf sampled from a shelf model and spawned free of collisions.
    """

    shelf: RelationalCircuitExperimentShelf
    """
    The spawned shelf; an object dropped during repair has no annotation.
    """

    dropped_object_count: int
    """
    How many sampled objects were dropped because they could not be placed without a
    collision.
    """

    @property
    def standing_object_count(self) -> int:
        """
        How many objects stand on the shelf.
        """
        return sum(
            object_.annotation is not None
            for layer in self.shelf.layers
            for object_ in layer.objects
        )


# %% generation
@dataclass
class ShelfGenerator:
    """
    Samples shelves from a shelf model, dresses them with real meshes and spawns them
    free of collisions.
    """

    model: ShelfModel
    """
    The model shelves are sampled from.
    """

    database: ProcessedShelfDatabase
    """
    The processed database the meshes of sampled objects are looked up in.
    """

    scenes_root: Path
    """
    Directory holding the downloaded scenes whose meshes can be spawned.
    """

    slab_thickness: float = 0.02
    """
    Thickness, in metres, of every layer slab.
    """

    corpus_wall_thickness: float = 0.03
    """
    Thickness, in metres, of the shelf corpus walls.
    """

    def generate(
        self,
        query: Match[RelationalCircuitExperimentShelf],
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> GeneratedShelf:
        """
        Sample a shelf answering *query* and spawn it into *world*.

        Objects are dressed with a mesh of their own type that fits their layer; an
        object without one is left out. Overlapping objects are redrawn before
        spawning, and whatever still collides once the real meshes stand is dropped.

        :param query: The underspecified shelf to sample, for example from
            :meth:`~experiments.shelf_generation_experiments.shelf_schema.
            RelationalCircuitExperimentShelf.underspecified_query`.
        :param world: The world to spawn the shelf into; its collisions are checked
            against everything already in it.
        :param parent: The entity the shelf is placed under; the root of *world* when
            omitted.
        :param parent_T_self: Where the shelf stands in the frame of *parent*.
        :raises NoSolutionFound: If the model gives *query* no support.
        :return: The spawned shelf.
        """
        shelf: RelationalCircuitExperimentShelf = next(
            iter(self.model.shelf_backend().evaluate(query))
        )
        shelf.source_ids = self.model.coarsening.coarsen_mesh_candidates(
            self.database.mesh_candidates(
                self.model.coarsening.stored_object_types_of(shelf), self.scenes_root
            )
        )

        corpus, layers = shelf._spawn_corpus_and_slabs(
            world,
            parent=parent,
            parent_T_self=parent_T_self,
            slab_thickness=self.slab_thickness,
            corpus_wall_thickness=self.corpus_wall_thickness,
        )
        pre_spawn_resolver = PreSpawnLayoutResolver.for_shelf(
            shelf, layers, self.model, corpus
        )
        RelationalCircuitExperimentShelf.spawn_objects(
            world, corpus, layers, pre_spawn_resolver.resolve()
        )
        in_world_resolver = InWorldLayoutResolver(shelf=shelf)
        return GeneratedShelf(
            shelf=in_world_resolver.resolve(),
            dropped_object_count=pre_spawn_resolver.dropped_object_count
            + in_world_resolver.dropped_body_count,
        )
