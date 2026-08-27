from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Optional, List

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import NoSolutionFound
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.variable import Variable

from experiments.scene_generation_experiments.exceptions import (
    UndrawableShelfError,
    UnknownShelfVariableError,
)
from krrood.utils import get_class_and_attribute_name
from semantic_digital_twin.scene_generation.scene_schema_aggregations import (
    EGShelfAggregations,
    EGShelfLayerAggregations,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


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

    :param object_2d: The object whose pose and scale are fixed.
    :return: A partially-underspecified EGObject2D holding *object_2d*'s pose.
    """
    return a(EGObject2D)(
        object_type=...,
        scale=object_2d.scale,
        pose=object_2d.pose,
        source_id=None,
    )


def free_object_slot():
    """
    Build a fully underspecified EGObject2D query slot with all spatial fields free.

    Only yaw genuinely varies and is left for the RSPN to sample; ``x`` and ``y`` are
    likewise free. :class:`~semantic_digital_twin.spatial_types.spatial_types.Pose2D`
    carries no roll/pitch dimensions to pin, since floor objects always sit upright
    without tilting.

    The shelf's theme is pinned rather than left free: it is what decides which objects
    are drawn, and only the fields a slot carries itself reach the distribution the
    object is drawn from.

    :return: An underspecified EGObject2D with scale and pose unset.
    """
    return a(EGObject2D)(
        object_type=...,
        scale=a(Scale)(x=..., y=..., z=...),
        pose=a(Pose2D)(x=..., y=..., yaw=...),
        source_id=None,
    )


def build_layer_query(
    theme_dominant_type: ObjectType,
    fixed_objects: Sequence[EGObject2D] = (),
    free_count: int = 0,
):
    """
    Build an EGShelfLayer query that keeps *fixed_objects*' spatial fields as
    conditioning evidence and leaves *free_count* fresh object slots fully
    underspecified.

    Used both to draw a layer from scratch (no fixed objects) and to redraw a layer's
    offending objects while holding its others in place. Free slots are appended after
    the fixed ones, so the caller reads freshly drawn objects off the tail of the
    result.

    :param theme_dominant_type: The shelf's dominant object type, held as evidence so
        the objects drawn onto it are the ones a shelf of that theme holds.
    :param fixed_objects: Objects whose full pose is held as evidence.
    :param free_count: Number of fully-underspecified object slots to draw.
    :return: An underspecified EGShelfLayer query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    return a(EGShelfLayer)(
        objects=[_fixed_object_slot(object_2d) for object_2d in fixed_objects]
        + [free_object_slot() for _ in range(free_count)],
        theme_dominant_type=theme_dominant_type,
        height_above_shelf_base=...,
        relative_height=...,
        vertical_clearance=...,
    )


def evaluate_first_supported(backend: ProbabilisticBackend, *queries):
    """
    Evaluate *queries* in order, returning the first sample the RSPN has support for.

    Each query is expected to hold strictly less evidence than the one before
    it, so the search walks outwards from the most informative conditioning to
    the least. In practice it is neighbour poses that go unsupported and abort
    the whole layout if the search stops early: conditioning a resample on
    every already-placed neighbour's exact pose pins the query to a region of
    zero probability mass, and the neighbours drift further from the training
    distribution with each repair pass.

    :param backend: The backend to evaluate the queries against.
    :param queries: Progressively less-conditioned forms of the same query.
    :raises NoSolutionFound: If the circuit supports none of them.
    :return: The first sample from whichever query found a solution.
    """
    for query in queries[:-1]:
        try:
            return next(iter(backend.evaluate(query)))
        except NoSolutionFound:
            continue
    return next(iter(backend.evaluate(queries[-1])))


def build_theme_shelf_query(
    theme_dominant_type: ObjectType,
    objects_per_layer: List[int],
):
    """
    Sample a shelf's layer count and each layer's object count from
    *relational_probabilistic_circuit* and build the EGShelf query they imply.

    How many layers there are fixes how many slots the query needs, so it is drawn from
    the shelf's own distribution first. Each layer's own object count is drawn the same
    way, from :class:`LayerObjectCountSampler`, rather than taken as a caller-chosen
    constant -- that is what lets a book-themed shelf's layers come out as full as book-
    themed layers were trained on.

    The query this returns is not guaranteed to have support in the circuit: a layer
    count or per-layer object count can carry marginal mass while the shelf it implies
    has none, since the grounded query conditions on the count, the theme and the layer
    structure together, which is stricter than any one count's own marginal. Callers
    that need a query the circuit can actually answer should retry this on
    :class:`~krrood.entity_query_language.exceptions.NoSolutionFound`, as :func:`draw_shelf`
    does.

    :param theme_dominant_type: The shelf's dominant object type to draw.
    :param objects_per_layer: How many objects each layer of the shelf should hold.
    :return: An underspecified EGShelf query ready for :class:`ProbabilisticBackend`
        evaluation.
    """
    return a(EGShelf)(
        scale=a(Scale)(x=..., y=..., z=...),
        layers=[
            a(EGShelfLayer)(
                objects=[
                    a(EGObject2D)(
                        object_type=...,
                        scale=a(Scale)(x=..., y=..., z=...),
                        pose=a(Pose2D)(x=..., y=..., yaw=...),
                        source_id=None,
                    )
                    for _ in range(count)
                ],
                theme_dominant_type=theme_dominant_type,
                height_above_shelf_base=...,
                relative_height=...,
                vertical_clearance=...,
            )
            for count in objects_per_layer
        ],
        theme_dominant_type=theme_dominant_type,
    )
