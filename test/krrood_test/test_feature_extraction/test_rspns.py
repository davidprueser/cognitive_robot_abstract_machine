import json

import random

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a, an
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.exceptions import IntractableError
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    TestExParts,
)


@pytest.fixture
def scenario():
    objects = [
        SceneObject(type=SceneObjectType.TABLE),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
    ]
    room = SceneRoom(
        position=KRROODPosition(x=2.0, y=1.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects[:3],
    )
    room2 = SceneRoom(
        position=KRROODPosition(x=4.0, y=3.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )
    return to_dao(room), to_dao(room2)


@pytest.fixture
def relational_probabilistic_circuit(scenario):
    room_dao, room2_dao = scenario
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([room_dao, room2_dao])
    return model


@pytest.fixture
def room_query_4():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.resolve()
    return query


def test_ground_before_fit_raises(room_query_4):
    model = RelationalProbabilisticCircuit(SceneRoom)
    with pytest.raises(CircuitNotFittedError):
        model.ground(room_query_4)


def test_fit_class_circuit_is_valid(relational_probabilistic_circuit):
    assert relational_probabilistic_circuit.class_probabilistic_circuit is not None
    assert relational_probabilistic_circuit.class_probabilistic_circuit.is_valid()


def test_fit_class_circuit_has_room_scalar_variables(relational_probabilistic_circuit):
    names = {
        v.name
        for v in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.position.y" in names
    assert "SceneRoom.position.z" in names
    assert "SceneRoom.orientation.x" in names
    assert "SceneRoom.orientation.y" in names
    assert "SceneRoom.orientation.z" in names
    assert "SceneRoom.orientation.w" in names


def test_fit_class_circuit_has_aggregation_variable(relational_probabilistic_circuit):
    names = {
        v.name
        for v in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert "SceneRoomAggregations.total_count()" in names


def test_fit_creates_exchangeable_template_for_objects(
    relational_probabilistic_circuit,
):
    assert (
        "objects"
        in relational_probabilistic_circuit.exchangeable_distribution_templates
    )
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    assert template.template_distribution.class_probabilistic_circuit is not None


def test_fit_exchangeable_template_class_is_the_domain_class(
    relational_probabilistic_circuit,
):
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    assert template.template_distribution.class_ is SceneObject


def test_fit_exchangeable_template_latent_is_total_count(
    relational_probabilistic_circuit,
):
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    latent_names = {v.name for v in template.latent_variables}
    assert "SceneRoomAggregations.total_count()" in latent_names


def test_fit_exchangeable_template_models_object_type(relational_probabilistic_circuit):
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    pc = template.template_distribution.class_probabilistic_circuit
    names = {v.name for v in pc.variables}
    assert "type" in names


def test_ground_circuit_is_valid(relational_probabilistic_circuit, room_query_4):
    model = relational_probabilistic_circuit.ground(room_query_4)
    assert model.is_valid()


def test_ground_has_per_object_type_variables(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {v.name for v in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_preserves_room_scalar_variables(
    relational_probabilistic_circuit, room_query_4
):
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.orientation.w" in names


def test_ground_integrates_out_unavailable_aggregates(
    relational_probabilistic_circuit, room_query_4
):
    """
    ``chair_count`` and ``table_count`` cannot be determined from the underspecified
    query, so the Monte-Carlo path must integrate them out: they must not survive as
    variables, while the object-type variables remain.
    """
    model = relational_probabilistic_circuit.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" not in names
    assert "SceneRoomAggregations.table_count()" not in names
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_with_unavailable_aggregate_is_valid(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    assert relational_probabilistic_circuit.ground(room_query_4).is_valid()


def test_non_positive_sample_count_raises_when_integration_needed(
    relational_probabilistic_circuit, room_query_4
):
    """
    Monte-Carlo integration cannot be disabled: a non-positive sample count is rejected
    when undetermined aggregates must be integrated out.
    """
    relational_probabilistic_circuit.monte_carlo_sample_count = 0
    with pytest.raises(InvalidMonteCarloSampleCountError):
        relational_probabilistic_circuit.ground(room_query_4)


def test_monte_carlo_sample_count_controls_mixture_size(
    relational_probabilistic_circuit, room_query_4
):
    """
    Drawing more samples discovers more distinct aggregate values, each adding an
    exchangeable-distribution instance (and its sum units) to the mixture.
    """
    np.random.seed(0)
    relational_probabilistic_circuit.monte_carlo_sample_count = 1
    single = sum(
        1
        for n in relational_probabilistic_circuit.ground(room_query_4).nodes()
        if isinstance(n, SumUnit)
    )
    np.random.seed(0)
    relational_probabilistic_circuit.monte_carlo_sample_count = 50
    many = sum(
        1
        for n in relational_probabilistic_circuit.ground(room_query_4).nodes()
        if isinstance(n, SumUnit)
    )
    assert many > single


@pytest.fixture
def deserialized_relational_probabilistic_circuit(relational_probabilistic_circuit):
    """
    The circuit after a round-trip through actual JSON text.

    Going through :func:`json.dumps` and :func:`json.loads` rather than only through the
    intermediate dict is what exposes encoding losses such as integer node keys becoming
    strings.
    """
    return from_json(json.loads(json.dumps(to_json(relational_probabilistic_circuit))))


def test_deserialization_restores_class(deserialized_relational_probabilistic_circuit):
    assert isinstance(
        deserialized_relational_probabilistic_circuit, RelationalProbabilisticCircuit
    )
    assert deserialized_relational_probabilistic_circuit.class_ is SceneRoom


def test_deserialization_restores_class_circuit_variables(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    original_names = {
        v.name
        for v in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    restored_names = {
        v.name
        for v in deserialized_relational_probabilistic_circuit.class_probabilistic_circuit.variables
    }
    assert restored_names == original_names


def test_deserialization_restores_exchangeable_templates(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    assert (
        deserialized_relational_probabilistic_circuit.exchangeable_distribution_templates.keys()
        == relational_probabilistic_circuit.exchangeable_distribution_templates.keys()
    )
    template = deserialized_relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]
    latent_names = {v.name for v in template.latent_variables}
    assert latent_names == {
        v.name
        for v in relational_probabilistic_circuit.exchangeable_distribution_templates[
            "objects"
        ].latent_variables
    }


def test_deserialized_circuit_grounds_to_the_same_variables(
    relational_probabilistic_circuit,
    deserialized_relational_probabilistic_circuit,
    room_query_4,
):
    np.random.seed(0)
    original = relational_probabilistic_circuit.ground(room_query_4)
    np.random.seed(0)
    restored = deserialized_relational_probabilistic_circuit.ground(room_query_4)
    assert restored.is_valid()
    assert {v.name for v in restored.variables} == {v.name for v in original.variables}


def test_deserialized_circuit_preserves_likelihoods(
    relational_probabilistic_circuit, deserialized_relational_probabilistic_circuit
):
    """
    The class distribution itself must be preserved numerically, not only structurally.
    """
    samples = relational_probabilistic_circuit.class_probabilistic_circuit.sample(10)
    assert np.allclose(
        relational_probabilistic_circuit.class_probabilistic_circuit.log_likelihood(
            samples
        ),
        deserialized_relational_probabilistic_circuit.class_probabilistic_circuit.log_likelihood(
            samples
        ),
    )


def test_ground_variable_count_scales_with_query_size(relational_probabilistic_circuit):
    query_2 = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(2)],
    )
    query_2.resolve()
    query_4 = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query_4.resolve()
    assert len(relational_probabilistic_circuit.ground(query_4).variables) > len(
        relational_probabilistic_circuit.ground(query_2).variables
    )


def _many_rooms(count: int) -> list:
    random_x = random.Random(0)
    random_y = random.Random(1)
    return [
        to_dao(
            SceneRoom(
                position=KRROODPosition(
                    x=random_x.uniform(0.0, 100.0),
                    y=random_y.uniform(0.0, 100.0),
                    z=0.0,
                ),
                orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
                objects=[SceneObject(type=SceneObjectType.CHAIR)],
            )
        )
        for _ in range(count)
    ]


def test_min_samples_per_leaf_is_forwarded_to_exchangeable_part_templates():
    """
    The bound must also apply to recursively fitted exchangeable parts (e.g. a room's
    ``objects``), since those are the circuits that get deep-copied once per grounded
    instance during sampling.
    """
    model = RelationalProbabilisticCircuit(SceneRoom, min_samples_per_leaf=0.1).fit(
        _many_rooms(50)
    )
    template = model.exchangeable_distribution_templates["objects"]
    assert template.template_distribution.min_samples_per_leaf == 0.1


def test_min_samples_per_leaf_callable_resolves_per_level(scenario):
    """
    A callable ``min_samples_per_leaf`` must be invoked once per fitted level, each time
    with *that level's own* row count rather than the parent's: the room-level circuit
    here is fit on 2 rows (one per room), while its ``objects`` exchangeable part is fit
    on the 7 rows pooled from both rooms' objects (3 + 4).

    A single
    precomputed fraction calibrated against one of those counts miscalibrates the
    other -- passing a callable instead is what lets each level derive its own bound.
    """
    room_dao, room2_dao = scenario
    seen_row_counts = []

    def bound_for(row_count: int) -> float:
        seen_row_counts.append(row_count)
        return 0.5

    RelationalProbabilisticCircuit(SceneRoom, min_samples_per_leaf=bound_for).fit(
        [room_dao, room2_dao]
    )

    assert seen_row_counts == [2, 7]


def test_min_samples_per_leaf_callable_is_resolved_to_a_number_after_fit(scenario):
    """
    After ``fit`` returns, ``min_samples_per_leaf`` must hold the resolved number it was
    fitted with, not the callable strategy: a function isn't JSON-serializable, and
    ``TrainedArbitraryShelfModel.save`` (and any other caller of
    :func:`~krrood.adapters.json_serializer.to_json`) needs to round-trip a fitted
    circuit whichever form ``min_samples_per_leaf`` was originally given in.
    """
    room_dao, room2_dao = scenario
    model = RelationalProbabilisticCircuit(
        SceneRoom, min_samples_per_leaf=lambda row_count: 0.5
    ).fit([room_dao, room2_dao])

    assert model.min_samples_per_leaf == 0.5
    restored = from_json(json.loads(json.dumps(to_json(model))))
    assert restored.min_samples_per_leaf == 0.5


def test_min_samples_per_quantile_defaults_to_the_nyga_library_default():
    """
    A caller that never touches ``min_samples_per_quantile`` must see the same histogram
    granularity as before this field existed, since
    :func:`~probabilistic_model.learning.jpt.variables.infer_variables_from_dataframe`
    itself defaults to 10.
    """
    model = RelationalProbabilisticCircuit(SceneRoom)

    assert model.min_samples_per_quantile == 10


def test_min_samples_per_quantile_is_forwarded_to_exchangeable_part_templates():
    """
    The bound must also apply to recursively fitted exchangeable parts (e.g. a room's
    ``objects``), since those are the circuits that get deep-copied once per grounded
    instance during sampling.
    """
    model = RelationalProbabilisticCircuit(SceneRoom, min_samples_per_quantile=200).fit(
        _many_rooms(50)
    )
    template = model.exchangeable_distribution_templates["objects"]
    assert template.template_distribution.min_samples_per_quantile == 200


def test_min_samples_per_quantile_bounds_continuous_variable_leaf_count():
    """
    Regression test for a real incident: fitting the full sage10k shelf dataset (18,437
    shelves / 44,609 layers / 124,800 objects) with
    :func:`~probabilistic_model.learning.jpt.variables.infer_variables_from_dataframe`'s
    library default of 10 produced a 68,893-node object-level circuit and made
    ``draw_shelf`` unable to complete a single grounding in over 20 minutes, because
    every continuous variable (position, orientation, scale) fragmented into thousands
    of Nyga histogram pieces, each of which grounding deep-copies once per sampled part.

    Raising ``min_samples_per_quantile`` must measurably shrink the fitted circuit, the
    same lever that took it back down to seconds.
    """
    rooms = _many_rooms(50)
    fine = RelationalProbabilisticCircuit(SceneRoom, min_samples_per_quantile=2).fit(
        rooms
    )
    coarse = RelationalProbabilisticCircuit(SceneRoom, min_samples_per_quantile=20).fit(
        rooms
    )

    assert len(coarse.class_probabilistic_circuit.nodes()) < len(
        fine.class_probabilistic_circuit.nodes()
    )


# ---- Group A -- exchangeable parts nested two levels deep ----
#
# Nothing in production fits a class whose exchangeable part itself has an
# exchangeable part. The prefixing and joint-dataframe code anticipates it (and
# documents the bugs it caused), but no test pins it, so these do.


def _nested_scenario(room_counts: list[int]) -> list:
    """
    Build ``TestExParts`` instances whose rooms each hold objects.

    The first instance decides the shape of the whole fit: ``_fit_exchangeable_part``
    reads the child type off ``instances[0]`` and ``_process_many_to_many`` skips empty
    collections, so the leading instance must populate every collection in the chain.

    :param room_counts: Object count for each room of each built instance.
    :return: One data access object per requested instance.
    """
    return [
        to_dao(
            TestExParts(
                objects=[SceneObject(type=SceneObjectType.TABLE)],
                rooms=[
                    SceneRoom(
                        position=KRROODPosition(x=float(index), y=1.0, z=0.0),
                        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
                        objects=[
                            SceneObject(type=SceneObjectType.CHAIR)
                            for _ in range(object_count)
                        ],
                    )
                    for index, object_count in enumerate(room_counts)
                ],
            )
        )
        for _ in range(4)
    ]


@pytest.fixture
def nested_relational_probabilistic_circuit():
    model = RelationalProbabilisticCircuit(TestExParts)
    model.fit(_nested_scenario([2, 3]))
    return model


@pytest.fixture
def nested_query():
    query = a(TestExParts)(
        objects=[a(SceneObject)(type=...)],
        rooms=[
            a(SceneRoom)(
                position=a(KRROODPosition)(x=..., y=..., z=...),
                orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
                objects=[a(SceneObject)(type=...) for _ in range(2)],
            )
            for _ in range(2)
        ],
    )
    query.resolve()
    return query


def test_nested_exchangeable_part_is_fitted_as_its_own_template(
    nested_relational_probabilistic_circuit,
):
    """
    A room's ``objects`` must become a template *inside* the rooms template.

    Asserting on the inner template specifically -- rather than on the outer one -- is
    what separates "depth-2 recursion broken" from "a sibling template broken", since
    ``fit`` builds a template for every collection with aggregation features.
    """
    rooms_template = (
        nested_relational_probabilistic_circuit.exchangeable_distribution_templates[
            "rooms"
        ]
    )
    inner = rooms_template.template_distribution.exchangeable_distribution_templates
    assert "objects" in inner
    assert (
        inner["objects"].template_distribution.class_probabilistic_circuit is not None
    )


def test_grounding_a_nested_query_yields_a_single_rooted_circuit(
    nested_relational_probabilistic_circuit, nested_query
):
    """
    A depth-2 mount that fails to connect surfaces as a circuit with more than one root,
    which is the failure the part-prefix renaming exists to prevent.
    """
    np.random.seed(0)
    grounded = nested_relational_probabilistic_circuit.ground(nested_query)
    assert grounded.is_valid()


def test_grounded_nested_circuit_models_a_variable_per_inner_part(
    nested_relational_probabilistic_circuit, nested_query
):
    """
    Each object of each room needs its own variable, addressed by the full path through
    both levels -- otherwise the two rooms' objects share variables and the inner
    distribution collapses.
    """
    np.random.seed(0)
    grounded = nested_relational_probabilistic_circuit.ground(nested_query)
    names = {v.name for v in grounded.variables}
    for room_index in range(2):
        for object_index in range(2):
            assert (
                f"TestExParts.rooms[{room_index}].objects[{object_index}].type" in names
            )


def test_a_nested_query_samples_back_into_an_instance(
    nested_relational_probabilistic_circuit, nested_query
):
    """
    Grounding is only useful if the sample can be written back through the match tree,
    which is the step that consumes the prefixed variable names.
    """
    np.random.seed(0)
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=nested_relational_probabilistic_circuit
        ),
        number_of_samples=1,
    )
    sampled = next(iter(backend.evaluate(nested_query)))
    assert len(sampled.rooms) == 2
    for room in sampled.rooms:
        assert len(room.objects) == 2


# ---- Group B -- conditioning a query on an enum literal ----
#
# Pinning an enum on a query slot is avoided in production because of
# "enum-to-float conversion issues in the RSPN sampling backend". Whether that
# still holds decides whether a categorical field can be used as a conditioning
# variable at all, so it is pinned here rather than worked around.
#
# Fitting encodes an enum column as ``hash(member)``, and ``Enum.__hash__``
# hashes the member *name*, so the encoded values change with PYTHONHASHSEED.
# Values of that magnitude do not survive the circuit's numeric pipeline: a
# column of small integers keeps its values exactly, while hash-sized ones come
# back as different numbers entirely, leaving nothing to condition against.
# These two tests therefore fail for almost every seed and pass for the rare one
# whose hashes happen to survive -- the seed dependence is the bug, not flake.


def _typed_objects(counts: dict) -> list:
    """
    Build one data access object per requested object of each type.

    Every enum member has to appear, or the fitted variable's domain covers only the
    observed subset and conditioning on the missing member is a different failure.

    :param counts: How many objects to build for each object type.
    :return: The objects, as data access objects.
    """
    return [
        to_dao(SceneObject(type=object_type))
        for object_type, count_of_type in counts.items()
        for _ in range(count_of_type)
    ]


@pytest.fixture
def object_type_circuit():
    model = RelationalProbabilisticCircuit(SceneObject)
    model.fit(_typed_objects({SceneObjectType.CHAIR: 30, SceneObjectType.TABLE: 30}))
    return model


def test_a_query_pinned_to_an_enum_member_samples_that_member(object_type_circuit):
    """
    Conditioning a query on a categorical literal must actually restrict the samples.

    A backend that ignores the literal still returns a sample, so asserting only that
    sampling succeeded would pass while the condition silently did nothing.
    """
    np.random.seed(0)
    query = a(SceneObject)(type=SceneObjectType.CHAIR)
    query.resolve()
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=object_type_circuit
        ),
        number_of_samples=10,
    )
    sampled = list(backend.evaluate(query))
    assert sampled
    assert all(obj.type is SceneObjectType.CHAIR for obj in sampled)


def test_conditioning_the_class_circuit_directly_restricts_an_enum_variable(
    object_type_circuit,
):
    """
    The circuit-level conditioning path is exercised separately from the query path.

    Sampling a part count conditioned on a category uses this path directly, so it has
    to be known-good even if the query path turns out not to be. The member itself is
    the value the variable is defined over, so no encoding step stands between the
    caller and the condition.
    """
    circuit = object_type_circuit.class_probabilistic_circuit
    [type_variable] = [v for v in circuit.variables if v.name.endswith("type")]
    conditioned, probability = circuit.conditional(
        {type_variable: SceneObjectType.CHAIR}
    )
    assert probability > 0.0
    assert conditioned is not None


# ---- Group C -- evaluate_mode_with_log_density ----


def test_evaluate_mode_with_log_density_returns_one_scored_instance(
    relational_probabilistic_circuit, room_query_4
):
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=relational_probabilistic_circuit
        )
    )
    [(instance, log_density)] = list(
        backend.evaluate_mode_with_log_density(room_query_4)
    )
    assert isinstance(instance, SceneRoom)
    assert np.isfinite(log_density)


def test_evaluate_mode_with_log_density_score_matches_conditioning_plus_mode(
    object_type_circuit,
):
    """
    The reported score must equal the conditioning step's own log-probability plus the.

    conditioned model's mode log-density, recomputed independently through the model API
    -- not the backend -- so a future change to how the two are combined is caught.
    """
    query = a(SceneObject)(type=SceneObjectType.CHAIR)
    query.resolve()
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=object_type_circuit
        )
    )
    parameters = UnderspecifiedParameters(query)
    model = backend.model_registry.get_model(parameters)
    conditioned, probability = model.conditional(
        parameters.conditioning_assignments_from_literal_values
    )
    _, expected_mode_log_density = conditioned.log_mode()
    expected_score = expected_mode_log_density + np.log(probability)

    [(instance, log_density)] = list(backend.evaluate_mode_with_log_density(query))
    assert instance.type is SceneObjectType.CHAIR
    assert log_density == pytest.approx(expected_score)


def test_evaluate_mode_with_log_density_falls_back_to_sampling_when_mode_is_intractable(
    monkeypatch,
):
    """
    ``log_mode`` raising :class:`IntractableError` must not propagate -- the answer
    falls back to the highest-likelihood point among the fallback samples instead, its
    score built from that point's own log-likelihood rather than a mode log-density.
    """

    class _StubTruncatedModel:
        variables = ["x"]

        def log_mode(self):
            raise IntractableError(model=self)

        def sample(self, amount):
            return np.array([[1.0], [2.0], [3.0]])

        def log_likelihood(self, samples):
            return np.array([-5.0, -1.0, -3.0])

    class _StubParameters:
        def construct_instance_from_model_sample(self, variables, sample):
            return float(sample[0])

    backend = ProbabilisticBackend()
    monkeypatch.setattr(
        backend,
        "_condition_and_truncate",
        lambda expression: (_StubParameters(), _StubTruncatedModel(), 2.0),
    )

    query = a(SceneObject)(type=...)
    [(instance, log_density)] = list(backend.evaluate_mode_with_log_density(query))
    assert instance == 2.0  # the sample at index 1, the highest log-likelihood
    assert log_density == pytest.approx(-1.0 + 2.0)
