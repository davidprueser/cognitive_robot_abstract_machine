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
    EGPoint2D,
    EGRotation,
    EGScale,
    EGShelf,
    EGShelfLayer,
    ObjectType,
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
        theme_dominant_type=object_2d.theme_dominant_type,
    )


def free_object_slot(theme_dominant_type: ObjectType):
    """
    Build a fully underspecified EGObject2D query slot with all spatial fields free.

    Roll and pitch are fixed to ``0.0`` rather than left underspecified: floor objects
    always sit upright without tilting, so those two circuit dimensions are constant
    across every training example, and leaving a constant dimension underspecified lets
    the RSPN sampling backend leak the query's placeholder straight through instead of
    resolving it. Only yaw genuinely varies and is left for the RSPN to sample.

    The shelf's theme is pinned rather than left free: it is what decides which objects
    are drawn, and only the fields a slot carries itself reach the distribution the
    object is drawn from.

    :param theme_dominant_type: The shelf's dominant object type.
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
        theme_dominant_type=theme_dominant_type,
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
        + [free_object_slot(theme_dominant_type) for _ in range(free_count)],
        theme_dominant_type=theme_dominant_type,
        # Left free so the layer's height carries whatever the objects drawn
        # onto it imply -- a layer of books is drawn low, one of display pieces
        # high -- rather than being pinned by the caller.
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


_DRAW_ATTEMPTS = 8
"""
How many shelves to try before reporting that none can be drawn.

Each attempt redraws the layer count, so this bounds a rejection sample rather than a
retry of the same request; a handful covers the counts a shelf of a given theme
plausibly takes.
"""


@dataclass(frozen=True)
class ShelfDimensions:
    """
    The shelf-level values a shelf's layers have to agree with.

    Drawn before the layers because both are evidence for them: the footprint is
    what object positions are conditioned on, and the count is how many layers
    the query must ask for.
    """

    scale: EGScale
    """
    Outer dimensions of the shelf, shared by every one of its layers.
    """

    layer_count: int
    """
    How many layers the shelf holds.
    """


def _find_variable(circuit: ProbabilisticCircuit, name: str) -> Variable:
    """
    Find *circuit*'s variable with the given name.

    :param circuit: The class-level circuit to search.
    :param name: The variable's full name.
    :return: The matching variable.
    :raises UnknownShelfVariableError: If the circuit does not model it.
    """
    matches = [variable for variable in circuit.variables if variable.name == name]
    if not matches:
        raise UnknownShelfVariableError(
            variable_name=name,
            modelled_variables=sorted(v.name for v in circuit.variables),
        )
    return matches[0]


@dataclass(frozen=True)
class ShelfDimensionSampler:
    """
    Draws a shelf's own dimensions and layer count from a fitted circuit.

    A query grounds exactly as many parts as it has slots, so the number of layers
    cannot emerge from the draw -- it has to be known before the query is built.
    The footprint has the same problem for a different reason: left free, each
    layer draws its own and they disagree.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The fitted circuit, rooted at :class:`EGShelf`.
    """

    def sample(self, theme_dominant_type: ObjectType) -> ShelfDimensions:
        """
        Draw the dimensions and layer count of a shelf of the given kind.

        :param theme_dominant_type: The shelf's dominant object type to draw.
        :return: The drawn shelf-level values.
        :raises UnknownShelfVariableError: If the circuit models neither the theme nor
            the layer count, which means it predates them.
        """
        [count_feature] = EGShelfAggregations.symbolic_features_of_field("layers")
        type_variable = _find_variable(
            self.relational_probabilistic_circuit.class_probabilistic_circuit,
            get_class_and_attribute_name(EGShelf.__name__, "theme_dominant_type"),
        )
        conditioned, _ = (
            self.relational_probabilistic_circuit.class_probabilistic_circuit.conditional(
                {type_variable: theme_dominant_type}
            )
        )
        wanted = [count_feature._name_] + [
            get_class_and_attribute_name(
                get_class_and_attribute_name(EGShelf.__name__, "scale"), dimension
            )
            for dimension in ("width", "length", "height")
        ]
        variables = [
            variable for variable in conditioned.variables if variable.name in wanted
        ]
        [drawn] = conditioned.marginal(variables).sample(1)
        by_name = dict(zip((v.name for v in variables), drawn))
        return ShelfDimensions(
            scale=EGScale(
                width=float(by_name[wanted[1]]),
                length=float(by_name[wanted[2]]),
                height=float(by_name[wanted[3]]),
            ),
            layer_count=max(1, round(float(by_name[wanted[0]]))),
        )


@dataclass(frozen=True)
class LayerObjectCountSampler:
    """
    Draws how many objects a layer holds from the fitted per-layer template circuit.

    :class:`EGShelfLayerAggregations` records this as an aggregation statistic over each
    observed layer's objects, so a fitted circuit already knows how many objects layers
    of each theme tend to hold -- sampling it here is what lets a book-themed shelf's
    layers come out as full as book-themed layers were trained on, and a tool-themed
    shelf's as sparse as tool-themed layers were, instead of every layer getting the
    same caller-chosen count regardless of theme.
    """

    layer_template_circuit: RelationalProbabilisticCircuit
    """
    The fitted circuit rooted at :class:`EGShelfLayer`, i.e. the "layers" exchangeable
    distribution template's ``template_distribution``.
    """

    _THEME_VARIABLE_NAME: ClassVar[str] = "theme_dominant_type"
    """
    Unlike a root-level circuit's own variables (e.g. ``EGShelf.theme_dominant_type``),
    a nested exchangeable template's own variables carry no owning-class prefix, so this
    is the bare name the fitted circuit actually calls the field.
    """

    def sample(self, theme_dominant_type: ObjectType) -> int:
        """
        Draw the object count of one layer of the given theme.

        :param theme_dominant_type: The shelf's dominant object type.
        :return: The drawn object count, never negative.
        :raises UnknownShelfVariableError: If the circuit models neither the layer's
            theme nor its object count, which means it predates them.
        """
        [count_feature] = EGShelfLayerAggregations.symbolic_features_of_field("objects")
        circuit = self.layer_template_circuit.class_probabilistic_circuit
        type_variable = _find_variable(circuit, self._THEME_VARIABLE_NAME)
        conditioned, _ = circuit.conditional({type_variable: theme_dominant_type})
        count_variable = _find_variable(conditioned, count_feature._name_)
        [drawn] = conditioned.marginal([count_variable]).sample(1)
        return max(0, round(float(drawn[0])))


def build_shelf_query(
    theme_dominant_type: ObjectType,
    objects_per_layer: Sequence[int],
):
    """
    Build an EGShelf query of ``len(objects_per_layer)`` layers, the i-th holding
    ``objects_per_layer[i]`` underspecified objects.

    Written as one nested query rather than composed from separate per-layer and
    per-object builders, so what is pinned as evidence versus left for the RSPN to
    sample is visible in a single place.

    The shelf's theme is held as evidence at every level -- shelf, layer and object.
    It is denormalized onto the layers and objects, and leaving it free there would let
    the drawn layers and objects disagree with the shelf they belong to -- a book-
    themed shelf whose layers or objects claim to be bottle-themed. Object id,
    room_id, place_id and source_id are pinned to ``None`` rather than left
    underspecified: they are not modelled by the circuit, so there is nothing to
    sample. ``object_type`` is left underspecified on every object to avoid enum-to-
    float conversion issues in the RSPN sampling backend. Roll and pitch are pinned to
    ``0.0`` rather than left underspecified: floor objects always sit upright without
    tilting, so those two circuit dimensions are constant across every training
    example, and leaving a constant dimension underspecified lets the RSPN sampling
    backend leak the query's placeholder straight through instead of resolving it.
    Only yaw genuinely varies and is left for the RSPN to sample. A layer's own height
    fields are left free so they carry whatever the objects drawn onto it imply -- a
    layer of books is drawn low, one of display pieces high -- rather than being
    pinned by the caller.

    :param theme_dominant_type: The shelf's dominant object type to draw.
    :param objects_per_layer: Number of objects to draw onto each layer, one entry per
        layer; its length is the number of layers the shelf gets.
    :return: An underspecified EGShelf query ready for :class:`ProbabilisticBackend`
        evaluation.
    """
    return a(EGShelf)(
        scale=a(EGScale)(width=..., length=..., height=...),
        layers=[
            a(EGShelfLayer)(
                objects=[
                    a(EGObject2D)(
                        id=None,
                        room_id=None,
                        place_id=None,
                        object_type=...,
                        scale=a(EGScale)(width=..., length=..., height=...),
                        position=a(EGPoint2D)(x=..., y=...),
                        orientation=a(EGRotation)(x=0.0, y=0.0, z=...),
                        source_id=None,
                        theme_dominant_type=theme_dominant_type,
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


def build_theme_shelf_query(
    relational_probabilistic_circuit: RelationalProbabilisticCircuit,
    theme_dominant_type: ObjectType,
    objects_per_layer: Optional[List[int]] = None,
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

    :param relational_probabilistic_circuit: The fitted circuit, rooted at EGShelf.
    :param theme_dominant_type: The shelf's dominant object type to draw.
    :param layer_count: Overrides the drawn number of layers when given.
    :return: An underspecified EGShelf query ready for :class:`ProbabilisticBackend`
        evaluation.
    """
    # sampler = ShelfDimensionSampler(relational_probabilistic_circuit)
    # object_count_sampler = LayerObjectCountSampler(
    #     relational_probabilistic_circuit.exchangeable_distribution_templates[
    #         "layers"
    #     ].template_distribution
    # )
    # drawn_layer_count = sampler.sample(theme_dominant_type).layer_count
    # objects_per_layer = objects_per_layer or [
    #     object_count_sampler.sample(theme_dominant_type)
    #     for _ in range(drawn_layer_count)
    # ]
    return build_shelf_query(theme_dominant_type, objects_per_layer)


def evaluate_shelf_query(
    relational_probabilistic_circuit: RelationalProbabilisticCircuit,
    query,
) -> EGShelf:
    """
    Draw one sample of *query* from *relational_probabilistic_circuit*.

    :param relational_probabilistic_circuit: The fitted circuit, rooted at EGShelf.
    :param query: An EGShelf query, e.g. built with :func:`build_shelf_query` or
        :func:`build_theme_shelf_query`.
    :raises NoSolutionFound: If the circuit gives *query* no probability.
    :return: The sampled shelf.
    """
    backend = probabilistic_backend(relational_probabilistic_circuit)
    return next(iter(backend.evaluate(query)))


def draw_shelf(
    relational_probabilistic_circuit: RelationalProbabilisticCircuit,
    theme_dominant_type: ObjectType,
    layer_count: Optional[int] = None,
) -> EGShelf:
    """
    Draw one coherent shelf of the given theme.

    The shelf's own scale -- a layer carries none of its own -- is drawn jointly with
    its layers' contents by the query itself, so it agrees with what was placed on it.

    :param relational_probabilistic_circuit: The fitted circuit, rooted at EGShelf.
    :param theme_dominant_type: The shelf's dominant object type to draw.
    :param layer_count: Overrides the drawn number of layers when given.
    :return: A shelf whose contents agree with its drawn scale and layer count.
    """
    backend = probabilistic_backend(relational_probabilistic_circuit)

    # Drawn by rejection; see :func:`build_theme_shelf_query` for why a freshly
    # sampled query can still turn out unsupported. The counts are redrawn each
    # attempt, since that is what the query rejects.
    for _ in range(_DRAW_ATTEMPTS):
        query = build_theme_shelf_query(
            relational_probabilistic_circuit, theme_dominant_type, layer_count
        )
        try:
            return next(iter(backend.evaluate(query)))
        except NoSolutionFound:
            continue
    raise UndrawableShelfError(
        requested_theme=theme_dominant_type.value,
        requested_layer_count=layer_count,
        attempts=_DRAW_ATTEMPTS,
    )
