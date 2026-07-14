from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiments.scene_generation_experiments.collision_resolution import (
    _find_colliding_indices,
    _out_of_bounds_indices,
    resolve_shelf_collisions,
)
from experiments.scene_generation_experiments.exceptions import ShelfLayoutResolutionError
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGShelfLayer,
    EGScale,
    ObjectType,
)


def _book(object_id: str, x: float, y: float) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=EGScale(height=0.2, length=0.1, width=0.03),
        position=EGPoint2D(x=x, y=y),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="book_src",
    )


@pytest.fixture
def colliding_layer() -> EGShelfLayer:
    """
    Two overlapping books on one layer, so a resample pass is required to
    resolve the collision.
    """
    return EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[_book("book_0", 0.0, 0.0), _book("book_1", 0.01, 0.01)],
    )


def test_find_colliding_indices_is_deterministic_for_a_simple_pair(
    colliding_layer: EGShelfLayer,
) -> None:
    """
    Repeated calls on the same colliding layout must always resample the same
    index -- the choice must depend only on the collision graph (which pair
    of indices collides), not on the arbitrary body_a/body_b order the
    underlying collision detector happens to report on a given call.
    """
    results = {frozenset(_find_colliding_indices(colliding_layer)) for _ in range(30)}
    assert results == {frozenset({1})}


def test_find_colliding_indices_picks_the_minimal_set_for_a_star_collision() -> None:
    """
    When one object collides with two others that do not collide with each
    other, discarding just the shared object resolves every collision -- that
    minimal, single-index set must be returned consistently, not a larger set
    that happens to also be a valid (but wasteful) cover.
    """
    hub_and_two_others = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            _book("hub", 0.0, 0.0),
            _book("left", -0.02, 0.0),
            _book("right", 0.02, 0.0),
        ],
    )
    results = {frozenset(_find_colliding_indices(hub_and_two_others)) for _ in range(30)}
    assert results == {frozenset({0})}


def test_resolve_shelf_collisions_does_not_recheck_the_same_layer_state_twice(
    colliding_layer: EGShelfLayer,
) -> None:
    """
    resolve_shelf_collisions must not run collision detection twice against the
    same, unchanged layer state within one repair pass -- once to decide the
    layer needs repair, and again to compute which indices to resample.

    For one layer that needs exactly one resample round, collision detection
    must run exactly twice overall: once to find the collision, once more
    afterwards to confirm the resampled layer is clean.
    """
    resolved_layer = EGShelfLayer(
        scale=colliding_layer.scale,
        objects=[colliding_layer.objects[0], _book("book_1", 0.15, 0.0)],
    )

    original_find_colliding_indices = _find_colliding_indices
    call_count = 0

    def counting_wrapper(layer: EGShelfLayer) -> set[int]:
        nonlocal call_count
        call_count += 1
        return original_find_colliding_indices(layer)

    with patch(
        "experiments.scene_generation_experiments.collision_resolution._find_colliding_indices",
        side_effect=counting_wrapper,
    ), patch(
        "experiments.scene_generation_experiments.collision_resolution.ProbabilisticBackend"
    ) as backend_class:
        backend_class.return_value.evaluate.return_value = [resolved_layer]
        result = resolve_shelf_collisions([colliding_layer], rspn=MagicMock())

    assert call_count == 2
    assert not _find_colliding_indices(result[0])


# ---------------------------------------------------------------------------
# Layer/wall boundary checks
# ---------------------------------------------------------------------------


def _object_2d(
    x: float,
    y: float,
    yaw: float,
    scale: EGScale,
    object_id: str = "obj",
) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=scale,
        position=EGPoint2D(x=x, y=y),
        orientation=EGRotation(x=0.0, y=0.0, z=yaw),
        source_id="book_src",
    )


def test_out_of_bounds_indices_flags_object_hanging_off_the_layer_edge() -> None:
    """
    An object positioned past the layer's usable width must be flagged, even
    though it doesn't collide with anything else.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[_object_2d(x=0.3, y=0.0, yaw=0.0, scale=EGScale(height=0.2, length=0.1, width=0.03))],
    )
    assert _out_of_bounds_indices(layer) == {0}


def test_out_of_bounds_indices_does_not_flag_object_within_bounds() -> None:
    """
    A centered, appropriately-sized object must not be flagged.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[_object_2d(x=0.0, y=0.0, yaw=0.0, scale=EGScale(height=0.2, length=0.1, width=0.03))],
    )
    assert _out_of_bounds_indices(layer) == set()


def test_out_of_bounds_indices_uses_axis_aligned_footprint_regardless_of_yaw() -> None:
    """
    The bounds check must use the object's axis-aligned width/length regardless
    of its sampled yaw, not a yaw-rotated corner check.

    A yaw-rotated corner check was tried and measured against real training
    data: it flagged up to ~57% of ground-truth placements as out-of-bounds
    (vs. ~8% for the axis-aligned check), which a resample loop drawing from
    that same distribution can never reliably satisfy. Only the
    axis-aligned footprint is checked, so a rotated object's yaw must not
    change whether it is flagged.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=0.3, width=0.4),
        objects=[
            _object_2d(x=0.0, y=0.0, yaw=45.0, scale=EGScale(height=0.2, length=0.25, width=0.03))
        ],
    )
    assert _out_of_bounds_indices(layer) == set()


def test_resolve_shelf_collisions_resamples_out_of_bounds_objects() -> None:
    """
    An object that hangs off the layer edge must be resampled by
    resolve_shelf_collisions even though it does not collide with any other
    object -- out-of-bounds and pairwise-collision violations are both
    routed through the same repair loop.
    """
    layer_scale = EGScale(height=0.02, length=0.3, width=0.4)
    out_of_bounds_layer = EGShelfLayer(
        scale=layer_scale,
        objects=[_object_2d(x=0.3, y=0.0, yaw=0.0, scale=EGScale(height=0.2, length=0.1, width=0.03))],
    )
    fixed_layer = EGShelfLayer(
        scale=layer_scale,
        objects=[_object_2d(x=0.0, y=0.0, yaw=0.0, scale=EGScale(height=0.2, length=0.1, width=0.03))],
    )

    with patch(
        "experiments.scene_generation_experiments.collision_resolution.ProbabilisticBackend"
    ) as backend_class:
        backend_class.return_value.evaluate.return_value = [fixed_layer]
        result = resolve_shelf_collisions([out_of_bounds_layer], rspn=MagicMock())

    assert result[0] == fixed_layer
    assert not _out_of_bounds_indices(result[0])


def test_resolve_shelf_collisions_raises_after_max_passes_when_unsatisfiable() -> None:
    """
    An object that can never fit within the layer (wider than the layer itself)
    must not spin the repair loop forever -- resolve_shelf_collisions must give
    up and raise once max_passes is exhausted.
    """
    layer_scale = EGScale(height=0.02, length=0.3, width=0.4)
    unfixable_layer = EGShelfLayer(
        scale=layer_scale,
        objects=[_object_2d(x=0.0, y=0.0, yaw=0.0, scale=EGScale(height=0.1, length=0.1, width=1.0))],
    )

    with patch(
        "experiments.scene_generation_experiments.collision_resolution.ProbabilisticBackend"
    ) as backend_class:
        backend_class.return_value.evaluate.return_value = [unfixable_layer]
        with pytest.raises(ShelfLayoutResolutionError):
            resolve_shelf_collisions([unfixable_layer], rspn=MagicMock(), max_passes=3)
