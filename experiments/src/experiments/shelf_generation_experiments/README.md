# Shelf Circuit Experiment

This experiment learns what real shelves look like from the [SAGE-10k](https://huggingface.co/datasets/nvidia/SAGE-10k)
indoor scene dataset, and uses what it learned to

1. **generate** new, plausible shelves: their size, their layers and what stands on them,
2. **place** an object where it most likely belongs on a shelf, and
3. let a **robot tidy** an object from a table onto a generated shelf.

The model behind all three is a single *relational probabilistic circuit* (RSPN,
`probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`)
fitted over the shelf schema. Shelves, layers and objects are modelled at their own level, and
the levels are tied together through aggregation statistics (how many layers a shelf has,
how many objects a layer holds). Every question the experiment asks is an underspecified
EQL query answered by krrood's `ProbabilisticBackend` over that circuit.

## The shelf schema

`shelf_schema.py` defines the three levels the circuit models:

| Class | What it describes |
| --- | --- |
| `RelationalCircuitExperimentShelf` | A shelf's scale, its dominant object type (*theme*) and its layers. |
| `RelationalCircuitExperimentShelfLayer` | One level of a shelf: where it sits (height above the base, relative height, clearance above it), the shelf's theme and the objects on it. |
| `RelationalCircuitExperimentObject2D` | An object on a layer: its type, scale and planar pose relative to the shelf. |

Each class can spawn itself into a `semantic_digital_twin` world, and builds the
queries asked about it: `underspecified_query`, `evidence_query` and `placement_query`.
`shelf_schema_aggregations.py` declares the aggregation statistics that connect the levels.

## Sub-experiments

The experiment runs as a pipeline. Every step has its own sub-package and tutorial:

| Step | Sub-package | Tutorial | Produces |
| --- | --- | --- | --- |
| 1. Preprocessing | `preprocessing/` | [preprocessing/README.md](preprocessing/README.md) | The raw database of SAGE-10k layouts, and the processed database of shelves and objects. |
| 2. Training | `training/` | [training/README.md](training/README.md) | A fitted `ShelfModel`. |
| 3. Shelf generation | `generation/` | [generation/README.md](generation/README.md) | Collision-free shelves spawned into a world. |
| 4. Object placement | `placement/` | [placement/README.md](placement/README.md) | The layer and pose an object most likely belongs at. |
| 5. Shelf tidying demo | `tidying_demo/` | [tidying_demo/README.md](tidying_demo/README.md) | A simulated robot putting an object onto a generated shelf. |

## Setup

The experiment reads the dataset and its databases from four environment variables
(`dataset_environment.DatasetEnvironmentVariable`):

| Variable | Meaning |
| --- | --- |
| `SAGE10K_LAYOUTS_ROOT` | Directory with one sub-directory per downloaded layout, each holding exactly one `layout_*.json` file. |
| `SAGE10k_DATABASE_URI` | SQLAlchemy URI of the raw database the layouts are imported into. |
| `SAGE10K_PROCESSED_DATABASE_URI` | SQLAlchemy URI of the processed database preprocessing writes. |
| `SAGE10K_SCENES_ROOT` | Directory with one sub-directory per downloaded scene, each with an `objects/` folder holding `<source_id>.ply` meshes and `<source_id>_texture.png` textures. |

An entry point stops with `MissingEnvironmentVariableError` if a variable it needs is not
set.

The ORM interfaces have to be generated before anything touches a database:

```bash
python scripts/regenerate_all_orm.py
```

## Running the whole pipeline

```bash
# 1. preprocessing
python -m experiments.shelf_generation_experiments.preprocessing.raw_layout_import
python -m experiments.shelf_generation_experiments.preprocessing.preprocess_sage10k

# 2.-5. the demo fits the model on its first run, then generates, places and tidies
python -m experiments.shelf_generation_experiments.tidying_demo.demo
```

## Tests

Every sub-experiment is tested in `test/experiments_test/shelf_circuit_experiment/`, in a
sub-package of the same name (`test_preprocessing`, `test_training`, `test_generation`,
`test_placement`, `test_tidying_demo`). The tests need no dataset: `shelf_dataset.py`
builds synthetic shelves, fits a fully factorized shelf model on them, and lays out
scenes holding the chair mesh bundled with `semantic_digital_twin`.

```bash
pytest --orm-build never test/experiments_test/shelf_circuit_experiment
```
