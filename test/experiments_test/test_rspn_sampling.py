from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.scene_generation_experiments.rspn_sampling import (
    free_object_slot,
    build_layer_query,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


def _typed_object(object_type: ObjectType, object_id: str) -> EGObject2D:
    return EGObject2D(
        object_type=object_type,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=object_id,
    )


# ---------------------------------------------------------------------------
# free_object_slot - free floor object sampling query
# ---------------------------------------------------------------------------


def test_free_object_slot_leaves_the_whole_pose_free() -> None:
    """
    Free floor objects always sit upright without tilting; ``Pose2D`` has no roll/pitch
    dimensions to pin in the first place (unlike the old ``EGRotation``), so x, y, and
    yaw are all left underspecified for the RSPN to sample.
    """
    pose = free_object_slot().kwargs["pose"]

    assert pose.kwargs["x"] is ...
    assert pose.kwargs["y"] is ...
    assert pose.kwargs["yaw"] is ...


# ---------------------------------------------------------------------------
# build_layer_query - conditioning on fixed objects during RSPN sampling
# ---------------------------------------------------------------------------


def test_build_layer_query_frees_resampled_scale_and_pose() -> None:
    """
    build_layer_query must condition only the fixed objects' scale and pose, leaving a
    free slot's scale, position, and orientation all free to be redrawn.

    Conditioning a resampled slot on its own scale pins the query to the
    single training example that combination of evidence (its own scale plus
    every fixed neighbour's exact pose) came from, collapsing the RSPN's
    posterior for that slot's position back to its original, still-colliding
    value -- observed as a repair pass that redraws the exact same pose every
    time and so can never actually resolve a collision. Regression test for
    that collapse.
    """
    query = build_layer_query(
        ObjectType.BOTTLE,
        [_typed_object(ObjectType.BOOK, "fixed")],
        1,
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    conditioned_positions = [name for name in conditioned_names if "pose.x" in name]
    conditioned_scales = [
        name for name in conditioned_names if "objects[" in name and "scale.x" in name
    ]
    # Only the one fixed object's pose and scale are conditioned; the
    # free slot's are left entirely free.
    assert len(conditioned_positions) == 1
    assert len(conditioned_scales) == 1
