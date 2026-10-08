from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from random_events.interval import closed

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    NoSolutionFound,
)
from krrood.entity_query_language.factories import a, variable
from krrood.parametrization.model_registries import (
    DictRegistry,
    RelationalCircuitRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    leaf,
)

from ...dataset import ormatic_interface  # type: ignore
from ...dataset.example_classes import SceneObject, SceneObjectType


# %% domain
@dataclass
class Slider:
    """
    Something with one continuous setting.
    """

    position: float


# %% fixtures
@pytest.fixture
def overlapping_mixture_backend() -> ProbabilisticBackend:
    """
    A backend over a mixture of two overlapping uniform distributions of
    :attr:`Slider.position`; the mixture is not deterministic, and its density is
    highest where the two overlap.
    """
    query = a(Slider)(position=...)
    query.resolve()
    [position] = UnderspecifiedParameters(query).variables.values()
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for lower, upper in ((0.0, 2.0), (1.0, 3.0)):
        root.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=position, interval=closed(lower, upper).simple_sets[0]
                ),
                circuit,
            ),
            log_weight=np.log(0.5),
        )
    return ProbabilisticBackend(model_registry=DictRegistry({Slider: circuit}))


@pytest.fixture
def object_type_circuit() -> RelationalProbabilisticCircuit:
    return RelationalProbabilisticCircuit(SceneObject).fit(
        [SceneObject(type=SceneObjectType.CHAIR) for _ in range(20)]
        + [SceneObject(type=SceneObjectType.TABLE) for _ in range(40)]
    )


# %% exact mode
def test_mode_score_adds_the_evidence_log_probability_to_the_mode_log_density(
    object_type_circuit,
) -> None:
    """
    The score is the log-probability of the evidence plus the restricted model's mode
    log-density, recomputed here through the model itself rather than the backend.
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
    conditioned, evidence_log_probability = model.log_conditional(
        parameters.conditioning_assignments_from_literal_values
    )
    _, mode_log_density = conditioned.log_mode()

    result = backend.evaluate_mode(query)

    assert result.instance.type is SceneObjectType.CHAIR
    assert result.log_density == pytest.approx(
        mode_log_density + evidence_log_probability
    )


# %% sampled mode
def test_mode_of_a_non_deterministic_model_is_its_most_likely_sample(
    overlapping_mixture_backend,
) -> None:
    query = a(Slider)(position=...)
    query.resolve()

    result = overlapping_mixture_backend.evaluate_mode(query)

    assert 1.0 <= result.instance.position <= 2.0
    assert result.log_density == pytest.approx(np.log(0.5))


# %% invalid queries
def test_mode_of_evidence_without_support_is_none(
    overlapping_mixture_backend,
) -> None:
    query = a(Slider)(position=5.0)
    query.resolve()

    assert overlapping_mixture_backend.evaluate_mode(query) is None


# %% single generation
def test_generate_one_generates_an_instance_within_the_evidence(
    overlapping_mixture_backend,
) -> None:
    query = a(Slider)(position=...)
    query.where(query.position > 2.5)

    generated = overlapping_mixture_backend.generate_one(query)

    assert 2.5 < generated.position <= 3.0


def test_generate_one_of_evidence_without_support_is_none(
    overlapping_mixture_backend,
) -> None:
    query = a(Slider)(position=5.0)
    query.resolve()

    assert overlapping_mixture_backend.generate_one(query) is None


def test_evaluation_of_evidence_without_support_has_no_solution(
    overlapping_mixture_backend,
) -> None:
    query = a(Slider)(position=5.0)
    query.resolve()

    with pytest.raises(NoSolutionFound):
        list(overlapping_mixture_backend.evaluate(query))


def test_mode_is_only_evaluated_for_a_match(overlapping_mixture_backend) -> None:
    with pytest.raises(GenerativeBackendQueryIsNotUnderspecifiedVariable):
        overlapping_mixture_backend.evaluate_mode(variable(Slider, domain=None))


def test_generate_one_only_generates_for_a_match(overlapping_mixture_backend) -> None:
    with pytest.raises(GenerativeBackendQueryIsNotUnderspecifiedVariable):
        overlapping_mixture_backend.generate_one(variable(Slider, domain=None))
