from __future__ import annotations

from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionDetectorModelUpdater,
    CollisionDetectorStateUpdater,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)


# %% stopping
def test_a_stopped_detector_is_no_longer_notified_by_its_world(cylinder_bot_world):
    """
    A detector built for one check registers callbacks the world keeps alive, so without
    stopping it every later change keeps paying for it.
    """
    detector = FCLCollisionDetector(_world=cylinder_bot_world)

    detector.stop()

    assert detector.world_model_updater not in (
        CollisionDetectorModelUpdater.all_callbacks_of_this_type_from_world(
            cylinder_bot_world
        )
    )
    assert detector.world_state_updater not in (
        CollisionDetectorStateUpdater.all_callbacks_of_this_type_from_world(
            cylinder_bot_world
        )
    )
