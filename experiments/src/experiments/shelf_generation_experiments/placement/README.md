# Step 4: Object placement

Placement asks the shelf model where an object most likely belongs on a spawned shelf.

```python
from experiments.shelf_generation_experiments.placement.shelf_placement import (
    ShelfPlacement,
)

placement = ShelfPlacement(shelf=shelf, model=model).most_likely_placement(
    held_object, yaw=0.0
)
placement.layer  # the layer it belongs on
placement.placed_object.pose  # where on that layer, relative to the shelf corpus
placement.log_density  # how likely that placement is
```

## How a placement is found

Every layer is asked separately:

1. The free space of the layer comes from `HasSupportingSurface.planar_free_space`. Its
   height is that of the held object, and it is enlarged by the object's half size, so
   whatever stands on the layer, and the corpus walls, are kept clear.
   A layer without free space is skipped.
2. `RelationalCircuitExperimentShelfLayer.placement_query` builds a query for the layer.
   It holds the layer's own attributes and the objects standing on it as evidence. The
   held object's type and size are fixed, and its pose is constrained to the free space.
3. `ProbabilisticBackend.evaluate_mode` answers the query with its most likely instance
   and that instance's log-density. A layer whose evidence the model gives no support
   is skipped.

The placement with the highest log-density wins. If no layer has room,
`NoShelfPlacementError` is raised.

The yaw of the circuit is close to uniform, so its most likely yaw means little.
`most_likely_placement(..., yaw=...)` pins it instead.

The circuit passes only aggregation statistics from a layer to its objects. What the
held object is therefore decides *where on a layer* it goes, but not *which* layer: the
layers compete through how typical their own attributes and free space are.

## Tests

`test/experiments_test/shelf_circuit_experiment/test_placement/`
