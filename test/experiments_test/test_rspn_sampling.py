from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.scene_generation_experiments.rspn_sampling import (
    _free_object_slot,
    build_layer_query,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.scene_generation.scene_schema import (
    ShelfType,
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    ObjectType,
)


def _typed_object(object_type: ObjectType, object_id: str) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=object_type,
        scale=EGScale(height=0.1, length=0.1, width=0.1),
        position=EGPoint2D(x=0.0, y=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id=object_id,
        shelf_type=ShelfType.BOOKCASE,
    )


# ---------------------------------------------------------------------------
# _free_object_slot - free floor object sampling query
# ---------------------------------------------------------------------------


def test_free_object_slot_pins_roll_and_pitch_to_upright() -> None:
    """
    Free floor objects always sit upright without tilting (only yaw varies), so roll and
    pitch must be fixed evidence rather than left underspecified.

    A degenerate (always-constant) circuit dimension left underspecified leaks the
    query's ``...`` placeholder straight through the sample instead of resolving it to a
    number, so only yaw -- which genuinely varies in the training data -- may be left
    for the RSPN to sample.
    """
    orientation = _free_object_slot(ShelfType.BOOKCASE).kwargs["orientation"]

    assert orientation.kwargs["x"] == 0.0
    assert orientation.kwargs["y"] == 0.0
    assert orientation.kwargs["z"] is ...


# ---------------------------------------------------------------------------
# build_layer_query - conditioning on EGScale and fixed objects during
# RSPN sampling
# ---------------------------------------------------------------------------


def test_build_layer_query_conditions_scale_when_given() -> None:
    """
    build_layer_query must register a given scale's width and length as conditioning
    assignments so the RSPN draws positions that are appropriate for that specific
    scale.
    """
    target_scale = EGScale(width=0.5, length=0.3, height=0.02)
    query = build_layer_query(ShelfType.BOOKCASE, free_count=2, scale=target_scale)
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert any("scale.width" in name for name in conditioned_names)
    assert any("scale.length" in name for name in conditioned_names)


def test_build_layer_query_leaves_scale_free_without_one() -> None:
    """
    build_layer_query must leave scale as a free variable when none is given so the RSPN
    samples scale from its marginal -- the reference layer for the fixed-scale workflow
    is obtained this way.
    """
    query = build_layer_query(ShelfType.BOOKCASE, free_count=2)
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    assert not any("scale.width" in name for name in conditioned_names)
    assert not any("scale.length" in name for name in conditioned_names)


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
        ShelfType.BOOKCASE,
        [_typed_object(ObjectType.BOOK, "fixed")],
        1,
        EGScale(width=0.5, length=0.3, height=0.02),
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    conditioned_positions = [name for name in conditioned_names if "position.x" in name]
    # "objects[" scopes to per-object scale, excluding the layer's own
    # (always-fixed) EGShelfLayer.scale.width.
    conditioned_scales = [
        name
        for name in conditioned_names
        if "objects[" in name and "scale.width" in name
    ]
    # Only the one fixed object's position and scale are conditioned; the
    # free slot's are left entirely free.
    assert len(conditioned_positions) == 1
    assert len(conditioned_scales) == 1
