from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import singleton
from random_events.product_algebra import SimpleEvent

from krrood.entity_query_language.factories import a, count, entity, flat_variable
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    ImpossibleStatisticsError,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    SceneRoomAggregations,
)

# %% a circuit fitted on rooms with one or three chairs


def room_with_chairs(chair_count: int, position_x: float) -> SceneRoom:
    """
    :param chair_count: How many of the room's three objects are chairs.
    :param position_x: Where the room is.
    :return: A room of three objects, the rest of which are tables.
    """
    return SceneRoom(
        position=KRROODPosition(x=position_x, y=0.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=[SceneObject(type=SceneObjectType.CHAIR)] * chair_count
        + [SceneObject(type=SceneObjectType.TABLE)] * (3 - chair_count),
    )


@pytest.fixture
def chair_count_circuit() -> RelationalProbabilisticCircuit:
    random_generator = np.random.default_rng(0)
    rooms = [
        room_with_chairs(chair_count, float(random_generator.uniform(0, 5)))
        for chair_count in (1, 3)
        for _ in range(20)
    ]
    return RelationalProbabilisticCircuit().fit(rooms)


def room_query(*object_types: SceneObjectType):
    """
    :param object_types: The type of each object, ``...`` where it is left open.
    :return: A query for a room of these objects.
    """
    return a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=object_type) for object_type in object_types],
    )


def with_at_least_two_chairs(room):
    """
    :param room: A room query.
    :return: The query, restricted to rooms holding at least two chairs.
    """
    object_type = flat_variable(room.objects).type
    return room.having(
        entity(count(object_type)).where(object_type == SceneObjectType.CHAIR) >= 2
    ).grouped_by(room)


def chair_count_probability(grounded, chair_count: int) -> float:
    """
    :param grounded: A grounded circuit retaining the chair count.
    :param chair_count: A number of chairs.
    :return: The probability the grounded circuit gives that number of chairs.
    """
    [chair_count_variable] = [
        variable
        for variable in grounded.variables
        if variable.name
        == SceneRoomAggregations.symbolic_feature_of(
            SceneRoomAggregations.chair_count
        )._name_
    ]
    marginal = grounded.marginal([chair_count_variable])
    return marginal.probability(
        SimpleEvent.from_data(
            {chair_count_variable: singleton(chair_count)}
        ).as_composite_set()
    )


# %% conditioning on having conditions


def test_having_conditions_restrict_the_grounded_statistic(chair_count_circuit):
    query = with_at_least_two_chairs(room_query(..., ..., ...))
    query.resolve()
    np.random.seed(0)

    grounded = chair_count_circuit.ground(query)

    assert chair_count_probability(grounded, 3) == pytest.approx(1.0)


def test_stated_objects_bound_the_grounded_statistic(chair_count_circuit):
    query = room_query(SceneObjectType.TABLE, ..., ...)
    query.resolve()
    np.random.seed(0)

    grounded = chair_count_circuit.ground(query)

    assert chair_count_probability(grounded, 1) == pytest.approx(1.0)


def test_having_conditions_outside_the_stated_bounds_are_impossible(
    chair_count_circuit,
):
    query = with_at_least_two_chairs(room_query(SceneObjectType.TABLE, ..., ...))
    query.resolve()

    with pytest.raises(ImpossibleStatisticsError):
        chair_count_circuit.ground(query)


def test_having_conditions_violated_by_a_determined_statistic_are_impossible(
    chair_count_circuit,
):
    query = with_at_least_two_chairs(
        room_query(SceneObjectType.TABLE, SceneObjectType.TABLE, SceneObjectType.CHAIR)
    )
    query.resolve()

    with pytest.raises(ImpossibleStatisticsError):
        chair_count_circuit.ground(query)
