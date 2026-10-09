from __future__ import annotations

import numpy as np
import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    TestExParts,
)


# %% scenario
def _nested_scenario(room_counts: list[int]) -> list:
    """
    Build ``TestExParts`` instances whose rooms each hold objects.

    The first instance decides the shape of the whole fit: ``_fit_exchangeable_part``
    reads the child type off ``instances[0]`` and ``_process_many_to_many`` skips empty
    collections, so the leading instance must populate every collection in the chain.

    :param room_counts: Object count for each room of each built instance.
    :return: The requested instances.
    """
    return [
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
        for _ in range(4)
    ]


# %% fixtures
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


# %% fitting
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


# %% grounding
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


# %% sampling
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
