# Step 2: Training

Training fits a relational probabilistic circuit over the processed shelves.

## Reading the processed database

`processed_database.ProcessedShelfDatabase` is the read side of preprocessing:

- `shelves()` loads every stored shelf with its layers and objects.
- `mesh_candidates(object_types, scenes_root)` lists the stored objects of the given
  types whose mesh is available under `scenes_root`, as `MeshCandidate`s carrying the
  mesh's real size. Generation dresses sampled objects with these meshes.

## Coarsening object types

`object_type_coarsening.ObjectTypeCoarsening` keeps the most frequent object types, and
separately the most frequent shelf themes, and relabels every other type as
`ObjectType.OTHER`. A type that is common among objects is not necessarily common as a
shelf's theme, which is why the two are counted apart. The coarsening also relabels mesh
candidates so their types match what the circuit samples. It resolves a sampled `OTHER`
back to the stored types it stands for.

## Fitting

`shelf_model.ShelfModel.fit(shelves, settings)` coarsens the shelves and fits the circuit
over `RelationalCircuitExperimentShelf`. Its layers and their objects become exchangeable
parts, each fitted as a template of its own. `ShelfModelFitSettings` controls the size
of the circuit:

| Setting | Effect |
| --- | --- |
| `min_samples_per_leaf` | The smallest share of a level's rows a leaf holds, at the shelf, layer and object level. This allows at most `1 / min_samples_per_leaf` leaves per level. |
| `min_samples_per_quantile` | The fewest rows a histogram piece of a continuous variable describes. |
| `object_type_keep_count` | How many object types and themes are kept before coarsening. |

Grounding copies a part's circuit once per part of a queried shelf, so these settings
bound the memory a query needs.

A `ShelfModel` keeps its circuit and its coarsening together, since the circuit's
object-type domain is fixed by the types the coarsening kept. `save` and `load` store
both as JSON. `shelf_backend()` and `layer_backend()` hand out `ProbabilisticBackend`s
over the shelf and the single-layer circuit. These backends register the name
aliases that let queries name an object's pose by `pose.x`/`pose.y`, while the circuit
read them through the pose's ORM mapping as `pose.position.x`/`pose.position.y`.

```python
from experiments.shelf_generation_experiments.training.processed_database import (
    ProcessedShelfDatabase,
)
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel

database = ProcessedShelfDatabase.from_environment()
model = ShelfModel.fit(database.shelves())
model.save(path)
```

## Tests

`test/experiments_test/shelf_circuit_experiment/test_training/`
