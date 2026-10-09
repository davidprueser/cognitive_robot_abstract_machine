# Step 1: Preprocessing

Preprocessing turns the raw SAGE-10k layouts into the shelves the circuit is fitted on.
Every correction is applied once here, so that fitting is a plain read from a database.

## 1a. Importing the raw layouts

`raw_layout_import.RawLayoutImport` drops and recreates the raw database, parses every
layout under `SAGE10K_LAYOUTS_ROOT` with `semantic_digital_twin`'s `Sage10kDatasetLoader`
and stores the resulting scenes, committing them in batches.

```bash
export SAGE10K_LAYOUTS_ROOT=~/sage-10k-layouts
export SAGE10k_DATABASE_URI=postgresql://user@localhost/sage10k
python -m experiments.shelf_generation_experiments.preprocessing.raw_layout_import
```

## 1b. Extracting shelves

`preprocess_sage10k.Sage10kPreprocessingRun` reads the raw objects and writes a processed
copy of them to `SAGE10K_PROCESSED_DATABASE_URI`:

- **Unified object types.** `classification.ObjectTypeClassifier` maps the free-form
  type strings of the dataset (`"book2"`, `"bookchair8eba7fdc"`, ...) onto the
  generalized `ObjectType` categories. `ShelfMembershipClassifier` decides which
  furniture counts as a shelf at all.
- **Mesh-centred positions.** `mesh_measurement.MeshMeasurements` moves every object's
  recorded position onto the centre of its mesh. The meshes are looked up under
  `SAGE10K_SCENES_ROOT`.
- **Shelves and layers.** `ShelfExtractor` groups what stands on each shelf into layers,
  drops objects overhanging their shelf, and expresses every object's pose in the shelf's
  content frame. It records each shelf's dominant object type as its theme.

Every object is stored as a `PreprocessedObject`, and every shelf as a
`RelationalCircuitExperimentShelf` with its layers and objects. The object pass is split
by room across worker processes.

```bash
export SAGE10K_PROCESSED_DATABASE_URI=postgresql://user@localhost/sage10k_processed
export SAGE10K_SCENES_ROOT=~/sage-10k-scenes
python -m experiments.shelf_generation_experiments.preprocessing.preprocess_sage10k
```

Both entry points drop the database they write, so a re-run replaces the stored data
rather than adding to it.

## Tests

`test/experiments_test/shelf_circuit_experiment/test_preprocessing/`
