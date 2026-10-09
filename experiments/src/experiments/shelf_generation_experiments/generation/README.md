# Step 3: Shelf generation

Generation samples a shelf from the shelf model, dresses its objects with real meshes and
spawns it into a world free of collisions.

```python
from experiments.shelf_generation_experiments.generation.shelf_generator import (
    ShelfGenerator,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.utils import ObjectType

generator = ShelfGenerator(model=model, database=database, scenes_root=scenes_root)
generated = generator.generate(
    RelationalCircuitExperimentShelf.underspecified_query(ObjectType.BOOK, [3, 3, 3]),
    world,
    parent_T_self=shelf_pose,
)
```

The query fixes the theme, which can also be left to the model with `...`, and how many
objects each layer holds. Everything else is drawn from the model: the shelf's scale,
where its layers sit, and every object's type, scale and pose.

## What happens in `ShelfGenerator.generate`

1. **Sampling.** The shelf circuit answers the query with one sampled shelf.
2. **Dressing.** For every object, `RelationalCircuitExperimentShelf.match_meshes` picks a
   mesh of the object's type whose real size is close to the sampled one and fits under
   the layer above. An object without such a mesh is left out.
3. **Spawning the corpus.** The corpus and every layer slab are spawned. The objects are
   not spawned yet.
4. **Repairing before spawning.** `pre_spawn_resolver.PreSpawnLayoutResolver` checks the
   objects of every layer for overlapping footprints, using the real size of each
   matched mesh. It moves objects that leave their layer back inside it. Overlapping
   objects are redrawn from the layer circuit, holding their type and size, the layer
   and its other objects as evidence, and constrained to the free space the others
   leave. What still overlaps after the repair passes is dropped.
   `colliding_pairs.CollidingPairs` chooses which member of each overlapping pair moves.
5. **Spawning the objects.** The remaining objects are spawned with their meshes and
   seated on their slabs.
6. **Repairing in the world.** `in_world_resolver.InWorldLayoutResolver` checks the real
   meshes for what footprints could not foresee. It drops every object that collides
   with another or with the corpus, or no longer rests on its slab.

`GeneratedShelf` reports how many objects stand on the shelf and how many were dropped.

## Tests

`test/experiments_test/shelf_circuit_experiment/test_generation/`
