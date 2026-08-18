from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
from typing import ClassVar, Optional

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
    ShelfType,
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
        shelf_type=object_2d.shelf_type,
    )


def _free_object_slot(shelf_type: ShelfType):
    """
    Build a fully underspecified EGObject2D query slot with all spatial fields free.

    Roll and pitch are fixed to ``0.0`` rather than left underspecified: floor objects
    always sit upright without tilting, so those two circuit dimensions are constant
    across every training example, and leaving a constant dimension underspecified lets
    the RSPN sampling backend leak the query's placeholder straight through instead of
    resolving it. Only yaw genuinely varies and is left for the RSPN to sample.

    The kind of shelf is pinned rather than left free: it is what decides which
    objects are drawn, and only the fields a slot carries itself reach the
    distribution the object is drawn from.

    :param shelf_type: Kind of shelf the object is being drawn into.
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
        shelf_type=shelf_type,
    )


def build_layer_query(
    shelf_type: ShelfType,
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

    :param shelf_type: Kind of shelf the layer belongs to, held as evidence so the
        objects drawn onto it are the ones that kind of shelf holds.
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
        + [_free_object_slot(shelf_type) for _ in range(free_count)],
        shelf_type=shelf_type,
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
    the least. Two kinds of evidence go unsupported in practice, and both abort
    the whole layout if the search stops early:

    - **Neighbour poses.** Conditioning a resample on every already-placed
      neighbour's exact pose pins the query to a region of zero probability
      mass, and the neighbours drift further from the training distribution
      with each repair pass.
    - **The layer's own scale.** Every layer of a shelf is conditioned on the
      reference layer's drawn scale (see :func:`build_layer_query`),
      so a later layer's objects are resampled against a scale the circuit only
      ever saw paired with a different layer's contents. Relaxing only the
      neighbours keeps that scale pinned and fails again.

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
retry of the same request; a handful covers the counts a kind of shelf plausibly takes.
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

    def sample(self, shelf_type: ShelfType) -> ShelfDimensions:
        """
        Draw the dimensions and layer count of a shelf of the given kind.

        :param shelf_type: Kind of shelf to draw.
        :return: The drawn shelf-level values.
        :raises UnknownShelfVariableError: If the circuit models neither the shelf type
            nor the layer count, which means it predates them.
        """
        [count_feature] = EGShelfAggregations.symbolic_features_of_field("layers")
        type_variable = _find_variable(
            self.relational_probabilistic_circuit.class_probabilistic_circuit,
            get_class_and_attribute_name(EGShelf.__name__, "shelf_type"),
        )
        conditioned, _ = (
            self.relational_probabilistic_circuit.class_probabilistic_circuit.conditional(
                {type_variable: shelf_type}
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
    of each kind of shelf tend to hold -- sampling it here is what lets a bookcase's
    layers come out as full as bookcase layers were trained on, and a cabinet's as
    sparse as cabinet layers were, instead of every layer getting the same caller-chosen
    count regardless of shelf type.
    """

    layer_template_circuit: RelationalProbabilisticCircuit
    """
    The fitted circuit rooted at :class:`EGShelfLayer`, i.e. the "layers" exchangeable
    distribution template's ``template_distribution``.
    """

    _SHELF_TYPE_VARIABLE_NAME: ClassVar[str] = "shelf_type"
    """
    Unlike a root-level circuit's own variables (e.g. ``EGShelf.shelf_type``), a nested
    exchangeable template's own variables carry no owning-class prefix, so this is the
    bare name the fitted circuit actually calls the field.
    """

    def sample(self, shelf_type: ShelfType) -> int:
        """
        Draw the object count of one layer of the given kind of shelf.

        :param shelf_type: Kind of shelf the layer belongs to.
        :return: The drawn object count, never negative.
        :raises UnknownShelfVariableError: If the circuit models neither the layer's
            shelf type nor its object count, which means it predates them.
        """
        [count_feature] = EGShelfLayerAggregations.symbolic_features_of_field("objects")
        circuit = self.layer_template_circuit.class_probabilistic_circuit
        type_variable = _find_variable(circuit, self._SHELF_TYPE_VARIABLE_NAME)
        conditioned, _ = circuit.conditional({type_variable: shelf_type})
        count_variable = _find_variable(conditioned, count_feature._name_)
        [drawn] = conditioned.marginal([count_variable]).sample(1)
        return max(0, round(float(drawn[0])))


def build_shelf_query(
    shelf_type: ShelfType,
    layer_footprint: Optional[EGScale],
    objects_per_layer: Sequence[int],
):
    """
    Build an EGShelf query of ``len(objects_per_layer)`` layers, the i-th holding
    ``objects_per_layer[i]`` underspecified objects.

    The kind of shelf is held as evidence at both levels. It is denormalized onto
    the layers, and leaving it free there would let the drawn layers disagree with
    the shelf they belong to -- a bookcase whose layers claim to be a cabinet's.

    Every layer is pinned to the shelf's own footprint, which is how real shelves
    are built and what the training data records. Left free, each layer draws its
    own footprint: the slabs then disagree with each other, and the object
    positions drawn against them no longer suit the surface they are spawned on.

    The footprint has to come from a layer that was actually drawn, not from the
    shelf's own sampled dimensions: with real training data the scale is continuous,
    so an exact value drawn from the shelf's distribution has no counterpart in the
    layers' and pinning them to it leaves the query with no solution.

    :param shelf_type: Kind of shelf to draw.
    :param layer_footprint: Footprint every layer is pinned to, taken from a layer
        drawn beforehand. Left free when ``None``, which is how that first draw is
        made.
    :param objects_per_layer: Number of objects to draw onto each layer, one entry
        per layer; its length is the number of layers the shelf gets.
    :return: An underspecified EGShelf query ready for
        :class:`ProbabilisticBackend` evaluation.
    """
    return a(EGShelf)(
        scale=a(EGScale)(width=..., length=..., height=...),
        layers=[
            build_layer_query(shelf_type, free_count=count, scale=layer_footprint)
            for count in objects_per_layer
        ],
        shelf_type=shelf_type,
    )


def draw_shelf(
    relational_probabilistic_circuit: RelationalProbabilisticCircuit,
    shelf_type: ShelfType,
    layer_count: Optional[int] = None,
) -> EGShelf:
    """
    Draw one coherent shelf of the given kind.

    Four things have to agree and none can be settled by the query alone. How many
    layers there are fixes how many slots the query needs, so it is drawn from the
    shelf's own distribution first. Each layer's own object count is drawn the same way,
    from :class:`LayerObjectCountSampler`, rather than taken as a caller-chosen constant
    -- that is what lets a bookcase's layers come out as full as bookcase layers were
    trained on. The footprint every layer shares is taken from a reference layer, since
    only a value the layers' own distribution produced can be pinned on them. The
    shelf's height comes from the shelf, which is what makes a bookcase tall and a
    cabinet low.

    :param relational_probabilistic_circuit: The fitted circuit, rooted at EGShelf.
    :param shelf_type: Kind of shelf to draw.
    :param layer_count: Overrides the drawn number of layers when given.
    :return: A shelf whose layers agree with it in footprint and count.
    """
    backend = probabilistic_backend(relational_probabilistic_circuit)
    sampler = ShelfDimensionSampler(relational_probabilistic_circuit)
    object_count_sampler = LayerObjectCountSampler(
        relational_probabilistic_circuit.exchangeable_distribution_templates[
            "layers"
        ].template_distribution
    )

    # Drawn by rejection. The layer count comes from the shelf's own distribution,
    # but a count can carry mass there while the shelf it implies has none -- the
    # grounded query conditions on the count, the kind of shelf and the layer
    # structure together, which is stricter than the count's own marginal. Asking
    # again is how a draw from the feasible conditional is obtained; the count is
    # redrawn each time, since that is what the query rejects.
    for _ in range(_DRAW_ATTEMPTS):
        dimensions = sampler.sample(shelf_type)
        drawn_layer_count = (
            layer_count if layer_count is not None else dimensions.layer_count
        )
        objects_per_layer = [
            object_count_sampler.sample(shelf_type) for _ in range(drawn_layer_count)
        ]
        free_query = build_shelf_query(shelf_type, None, objects_per_layer)
        try:
            # The footprint every layer shares has to be a value the layers' own
            # distribution produced, and only a full shelf draw resolves a layer:
            # the layer template carries a latent from its parent that a bare layer
            # query never determines.
            footprint = next(iter(backend.evaluate(free_query))).layers[0].scale
            shelf = evaluate_first_supported(
                backend,
                build_shelf_query(shelf_type, footprint, objects_per_layer),
                free_query,
            )
        except NoSolutionFound:
            continue
        shelf.layers = [
            dataclasses.replace(layer, scale=shelf.layers[0].scale)
            for layer in shelf.layers
        ]
        footprint = shelf.layers[0].scale
        break
    else:
        raise UndrawableShelfError(
            shelf_type=shelf_type.value,
            requested_layer_count=layer_count,
            attempts=_DRAW_ATTEMPTS,
        )

    shelf.scale = EGScale(
        width=footprint.width, length=footprint.length, height=dimensions.scale.height
    )
    return shelf
