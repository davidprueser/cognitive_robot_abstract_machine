from __future__ import annotations

import inspect
import random

from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
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
from .test_nested_exchangeable_parts import _nested_scenario


# %% scenario
def _instances_with_scattered_rooms(count: int) -> list[TestExParts]:
    """
    :param count: How many instances to build.
    :return: Instances whose rooms stand at pseudo-random, reproducible positions, so
        the room template has continuous variables worth splitting into pieces.
    """
    position_generator = random.Random(0)
    return [
        TestExParts(
            objects=[SceneObject(type=SceneObjectType.TABLE)],
            rooms=[
                SceneRoom(
                    position=KRROODPosition(
                        x=position_generator.uniform(0.0, 100.0),
                        y=position_generator.uniform(0.0, 100.0),
                        z=0.0,
                    ),
                    orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
                    objects=[SceneObject(type=SceneObjectType.CHAIR)],
                )
            ],
        )
        for _ in range(count)
    ]


# %% learning methods of nested parts
def test_part_learning_method_reaches_a_part_nested_two_levels_deep() -> None:
    """
    A part is fitted with the learning method named for it, however deep it is nested.
    """
    rooms_method = JointProbabilityTree(min_samples_per_leaf=0.5)
    objects_method = JointProbabilityTree(min_samples_per_leaf=0.25)

    model = RelationalProbabilisticCircuit(
        TestExParts,
        part_learning_methods={"rooms": rooms_method, "objects": objects_method},
    ).fit(_nested_scenario([2, 3]))

    rooms_distribution = model.exchangeable_distribution_templates[
        "rooms"
    ].template_distribution
    nested_objects_distribution = (
        rooms_distribution.exchangeable_distribution_templates[
            "objects"
        ].template_distribution
    )
    assert rooms_distribution.learning_method is rooms_method
    assert nested_objects_distribution.learning_method is objects_method


# %% histogram granularity
def test_min_samples_per_quantile_defaults_to_that_of_variable_inference() -> None:
    model = RelationalProbabilisticCircuit(TestExParts)

    inference_default = (
        inspect.signature(infer_variables_from_dataframe)
        .parameters["min_samples_per_quantile"]
        .default
    )
    assert model.min_samples_per_quantile == inference_default


def test_min_samples_per_quantile_reaches_a_part_nested_two_levels_deep() -> None:
    model = RelationalProbabilisticCircuit(
        TestExParts, min_samples_per_quantile=200
    ).fit(_nested_scenario([2, 3]))

    rooms_distribution = model.exchangeable_distribution_templates[
        "rooms"
    ].template_distribution
    nested_objects_distribution = (
        rooms_distribution.exchangeable_distribution_templates[
            "objects"
        ].template_distribution
    )
    assert rooms_distribution.min_samples_per_quantile == 200
    assert nested_objects_distribution.min_samples_per_quantile == 200


def test_min_samples_per_quantile_bounds_the_size_of_a_part_template() -> None:
    """
    Fewer, wider histogram pieces per continuous variable give a smaller part circuit,
    and grounding copies that circuit once per part.

    The rooms are fitted without splits, so every histogram is induced over all rooms
    and its granularity alone decides the circuit's size.
    """
    instances = _instances_with_scattered_rooms(50)

    def rooms_circuit_size(min_samples_per_quantile: int) -> int:
        model = RelationalProbabilisticCircuit(
            TestExParts,
            part_learning_methods={
                "rooms": JointProbabilityTree(min_samples_per_leaf=len(instances))
            },
            min_samples_per_quantile=min_samples_per_quantile,
        ).fit(instances)
        rooms_distribution = model.exchangeable_distribution_templates[
            "rooms"
        ].template_distribution
        return len(rooms_distribution.class_probabilistic_circuit.nodes())

    assert rooms_circuit_size(20) < rooms_circuit_size(2)
