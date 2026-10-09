# Step 5: Shelf tidying demo

The demo puts the previous steps together. A simulated HSR robot picks an object up from
a table and puts it where a generated shelf's model says it belongs.

```bash
source /opt/ros/jazzy/setup.bash
export SAGE10K_PROCESSED_DATABASE_URI=postgresql://user@localhost/sage10k_processed
export SAGE10K_SCENES_ROOT=~/sage-10k-scenes
python -m experiments.shelf_generation_experiments.tidying_demo.demo
```

## What `ShelfTidyingDemo.run` does

1. **Model.** It loads the shelf model from `model_path`. On the first run it fits the
   model on the processed database and stores it there.
   `tidying_demo/models/` is ignored by git.
2. **Object.** It picks a type that the model both themes shelves by and places
   objects of, and that has a standing mesh (taller than wide). Of that type, it picks a
   mesh low enough for some layer. If none fits, `NoFittingObjectError` is raised.
3. **Shelf.** It generates a shelf with `objects_per_layer` objects per layer, placed at
   `shelf_pose`, and publishes the world for `visualization_backend`.
4. **Scene.** It spawns the robot, a floor holding the shelf and a table holding the
   object.
5. **Placement.** It asks where on the shelf the object belongs, with its thin side
   towards the open face.
6. **Tidying.** It runs `shelf_tidying.ShelfTidyingAction`:
   - pick the object up and place it with coraplex's `TransportAction`,
   - in between, drive along a route that `FloorNavigation` plans around everything
     standing on the floor, towards where `ShelfFront` says the robot stands in front of
     the shelf's open face.

   The object is grasped through `GraspableShelfObject`, which grasps at the centre of
   the mesh rather than at its origin, since that origin lies at the bottom.

`run` returns a `TidiedObject` with the shelf, the placement and the object's body.

## Visualization

`visualization.VisualizationBackend` selects the viewer:

- `RVIZ` publishes ordinary markers, which RViz2 loads from the file system.
- `FOXGLOVE` publishes through `FoxgloveVizMarkerPublisher`, for a browser connected
  through `foxglove_bridge`. Browsers cannot load `file://` meshes, so generated meshes
  are converted to glTF in `foxglove_bridge`'s share directory and referred to by
  `package://` resources, and installed meshes are referred to by their package.

## Tests

`test/experiments_test/shelf_circuit_experiment/test_tidying_demo/`. `test_demo.py` runs
the whole demo against a synthetic model and database.
