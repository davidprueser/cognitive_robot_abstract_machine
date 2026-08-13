from __future__ import annotations

from collections.abc import Sequence

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    EGShelfLayer,
)


def probabilistic_backend(rspn: RelationalProbabilisticCircuit) -> ProbabilisticBackend:
    """
    Build a single-sample probabilistic backend over *rspn*.

    Centralises the registry-plus-backend wiring shared by the generation pipelines and
    the in-world resolvers, so they all draw exactly one sample per query evaluation.

    :param rspn: The fitted circuit to sample from.
    :return: A backend that draws one sample per query evaluation.
    """
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rspn)
    return ProbabilisticBackend(model_registry=registry, number_of_samples=1)


def _fixed_object_slot(object_2d: EGObject2D):
    """
    Build an EGObject2D query slot whose spatial fields are pinned to
    *object_2d* as conditioning evidence.

    The object_type is left underspecified to avoid enum-to-float conversion
    issues in the RSPN sampling backend.

    :param object_2d: The object whose position, scale, and orientation are
        fixed.
    :return: A partially-underspecified EGObject2D holding *object_2d*'s pose.
    """
    return a(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=...,
        scale=object_2d.scale,
        position=object_2d.position,
        orientation=object_2d.orientation,
        source_id=None,
    )


def _free_object_slot():
    """
    Build a fully underspecified EGObject2D query slot with all spatial fields free.

    Roll and pitch are fixed to ``0.0`` rather than left underspecified: floor objects
    always sit upright without tilting, so those two circuit dimensions are constant
    across every training example, and leaving a constant dimension underspecified lets
    the RSPN sampling backend leak the query's placeholder straight through instead of
    resolving it. Only yaw genuinely varies and is left for the RSPN to sample.

    :return: An underspecified EGObject2D with position, scale, and yaw unset.
    """
    return a(EGObject2D)(
        id=None,
        room_id=None,
        place_id=None,
        object_type=...,
        scale=a(EGScale)(width=..., length=..., height=...),
        position=a(EGPoint2D)(x=..., y=...),
        orientation=a(EGRotation)(x=0.0, y=0.0, z=...),
        source_id=None,
    )


def build_layer_query(
    fixed_objects: Sequence[EGObject2D] = (),
    free_count: int = 0,
    scale: EGScale | None = None,
):
    """
    Build an EGShelfLayer query that keeps *fixed_objects*' spatial fields as
    conditioning evidence and leaves *free_count* fresh object slots fully
    underspecified.

    Used both to draw a layer from scratch (no fixed objects) and to redraw a layer's
    offending objects while holding its others in place. Conditioning a resampled slot
    on its own scale, in addition to the other objects' exact poses, pins the query to
    the single training example that combination of evidence came from, collapsing the
    RSPN's posterior for that slot's position back to its original, still-colliding
    value -- so a fixed object's scale is carried as evidence but a free slot's scale
    never is. Free slots are appended after the fixed ones, so the caller reads freshly
    drawn objects off the tail of the result.

    :param fixed_objects: Objects whose full pose is held as evidence.
    :param free_count: Number of fully-underspecified object slots to draw.
    :param scale: The layer dimensions to condition on. When ``None``, the layer's own
        scale is left free and sampled from the RSPN marginal -- used to draw a
        reference layer whose scale can then be passed here for subsequent layers.
    :return: An underspecified EGShelfLayer query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    scale_argument = (
        scale if scale is not None else a(EGScale)(width=..., length=..., height=...)
    )
    return a(EGShelfLayer)(
        scale=scale_argument,
        objects=[_fixed_object_slot(object_2d) for object_2d in fixed_objects]
        + [_free_object_slot() for _ in range(free_count)],
        # Left free so the layer's height carries whatever the objects drawn
        # onto it imply -- a layer of books is drawn low, one of display pieces
        # high -- rather than being pinned by the caller.
        height_above_shelf_base=...,
        relative_height=...,
        vertical_clearance=...,
    )
