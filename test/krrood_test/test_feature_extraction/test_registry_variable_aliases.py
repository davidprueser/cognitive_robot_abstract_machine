from __future__ import annotations

import random

import numpy as np
import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import FlatPose, FlatPoseHolder


# %% fixtures
@pytest.fixture
def holder_circuit() -> RelationalProbabilisticCircuit:
    """
    A circuit fitted on holders placed with x in [0, 1] and y in [10, 11], whose pose
    variables are named after :class:`FlatPoseMapping`'s nested position.
    """
    generator = random.Random(0)
    return RelationalProbabilisticCircuit(FlatPoseHolder).fit(
        [
            FlatPoseHolder(
                pose=FlatPose(
                    x=generator.uniform(0.0, 1.0),
                    y=generator.uniform(10.0, 11.0),
                    heading=generator.uniform(-1.0, 1.0),
                )
            )
            for _ in range(30)
        ]
    )


@pytest.fixture
def free_holder_query():
    query = a(FlatPoseHolder)(pose=a(FlatPose)(x=..., y=..., heading=...))
    query.resolve()
    return query


def _sampled_holder(
    circuit: RelationalProbabilisticCircuit,
    query,
    variable_name_aliases: dict[str, str],
) -> FlatPoseHolder:
    np.random.seed(0)
    backend = ProbabilisticBackend(
        model_registry=RelationalCircuitRegistry(
            relational_probabilistic_circuit=circuit,
            variable_name_aliases=variable_name_aliases,
        ),
        number_of_samples=1,
    )
    return next(iter(backend.evaluate(query)))


# %% aliases
def test_an_aliased_variable_reaches_the_query(holder_circuit, free_holder_query):
    holder = _sampled_holder(
        holder_circuit,
        free_holder_query,
        {"pose.position.x": "pose.x", "pose.position.y": "pose.y"},
    )

    assert 0.0 <= holder.pose.x <= 1.0
    assert 10.0 <= holder.pose.y <= 11.0


def test_a_variable_named_after_the_mapping_misses_the_query_without_an_alias(
    holder_circuit, free_holder_query
):
    holder = _sampled_holder(holder_circuit, free_holder_query, {})

    assert holder.pose.x is ...
    assert -1.0 <= holder.pose.heading <= 1.0


# %% renaming
def test_grounded_variables_become_the_query_variables(
    holder_circuit, free_holder_query
):
    """
    A grounded variable already qualified by the queried class is renamed to the very
    variable the query holds, rather than being qualified a second time.
    """
    parameters = UnderspecifiedParameters(free_holder_query)
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=holder_circuit,
        variable_name_aliases={
            "pose.position.x": "pose.x",
            "pose.position.y": "pose.y",
        },
    )

    grounded = registry.get_model(parameters)

    assert {id(variable) for variable in grounded.variables} == {
        id(variable) for variable in parameters.variables.values()
    }
