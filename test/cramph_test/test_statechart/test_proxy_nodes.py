"""
Tests for :class:`~cramph.ownership.ProxyNode`, which stands in for a leaf another
process owns, in the process that ticks the statechart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

import pytest
from typing_extensions import Any

import krrood.symbolic_math.symbolic_math as sm
from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.exceptions import (
    IrreproducibleEndBehaviourError,
    OwnedCompositeNodeError,
    UnknownProxyNodeError,
)
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.nodes_for_testing import (
    NodeFailingOnObservingFalse,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.ownership import (
    NodeOwnership,
    OwnedNodeLifeCycle,
    OwnedNodeObservation,
    OwningProcessConnection,
    ProxyNode,
)
from cramph.statechart import Statechart, StatechartJSONKey
from krrood.symbolic_math.symbolic_math import Scalar
from krrood.utils import get_full_class_name
from semantic_digital_twin.world import World

# %% processes and their connection


class ProcessName(StrEnum):
    """
    The processes sharing the statecharts of these tests.
    """

    TICKING = "ticking"
    """
    The process ticking every statechart of these tests.
    """

    PERCEPTION = "perception"
    """
    A process owning some of the leaves.
    """


class UnimportableClass(StrEnum):
    """
    Class names a document names in place of the classes of the owned leaves, so that
    deserializing such a leaf as itself fails.
    """

    DETECTOR = "unimportable_perception_package.DetectObjects"
    """
    Stands for a heavy class only the perception process can import.
    """


@dataclass(kw_only=True)
class ConnectionRecordingLifeCycles(OwningProcessConnection):
    """
    Keeps every life cycle state it is handed, in order, instead of sending it.
    """

    sent: list[OwnedNodeLifeCycle] = field(default_factory=list)
    """
    The life cycle states handed to this connection, in order.
    """

    def send_life_cycle(self, *, life_cycle: OwnedNodeLifeCycle) -> None:
        self.sent.append(life_cycle)


@dataclass(eq=False, repr=False, kw_only=True)
class NodeSucceedingOnAConditionOfItsOwn(NodeObservingAFixedValue):
    """
    A leaf that ends on a condition of its own, beyond observing True or False.
    """

    @property
    def inherent_success_condition(self) -> Scalar:
        return sm.logic_and(self.observes_true, self.last_observed_true)


# %% fixtures and helpers


@pytest.fixture()
def connection() -> ConnectionRecordingLifeCycles:
    """
    :return: The connection of the ticking process to the owning processes.
    """
    return ConnectionRecordingLifeCycles()


@pytest.fixture()
def ownership(connection: ConnectionRecordingLifeCycles) -> NodeOwnership:
    """
    :return: The ownership of the ticking process.
    """
    return NodeOwnership(process_name=ProcessName.TICKING, connection=connection)


@pytest.fixture()
def ticking_context(
    statechart_context: StatechartContext, ownership: NodeOwnership
) -> StatechartContext:
    """
    :return: The context of the ticking process, the one :func:`statechart_executor`
        ticks in.
    """
    statechart_context.add_extension(extension=ownership)
    return statechart_context


def _document(*, nodes: list[StatechartNode]) -> dict[str, Any]:
    """
    :param nodes: The top level nodes a sender puts into its statechart.
    :return: The JSON document of the sender's statechart, as a receiver gets it.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_nodes(nodes=nodes)
    return json.loads(json.dumps(statechart.to_json()))


def _document_naming_unimportable_classes(
    *, nodes: list[StatechartNode]
) -> dict[str, Any]:
    """
    :param nodes: The top level nodes a sender puts into its statechart.
    :return: :func:`_document` of `nodes`, naming :attr:`UnimportableClass.DETECTOR` as
        the class of every node of a class the perception process owns in these tests.
    """
    text = json.dumps(_document(nodes=nodes))
    for owned_class in (NodeSucceedingOnObservingTrue, NodeFailingOnObservingFalse):
        text = text.replace(
            f'"{get_full_class_name(owned_class)}"', f'"{UnimportableClass.DETECTOR}"'
        )
    return json.loads(text)


def _detector() -> NodeSucceedingOnObservingTrue:
    """
    :return: A leaf the perception process owns, which succeeds once it observes True.
    """
    return NodeSucceedingOnObservingTrue(
        name="find_cup",
        observation=ObservationStateValue.TRUE,
        owning_process=ProcessName.PERCEPTION,
    )


def _started_proxy(
    *,
    proxied: StatechartNode,
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> ProxyNode:
    """
    :param proxied: A leaf the perception process owns.
    :return: The proxy of `proxied` in a statechart of the ticking process, started by
        one tick.
    """
    statechart = Statechart.from_json(
        data=_document_naming_unimportable_classes(nodes=[proxied]),
        context=ticking_context,
    )
    statechart_executor.compile(statechart=statechart)
    statechart_executor.tick()
    return statechart.nodes[0]


# %% which leaves become proxies


def test_a_leaf_another_process_owns_becomes_a_proxy_without_its_class(
    ticking_context: StatechartContext,
) -> None:
    statechart = Statechart.from_json(
        data=_document_naming_unimportable_classes(nodes=[_detector()]),
        context=ticking_context,
    )

    assert type(statechart.nodes[0]) is ProxyNode


def test_a_leaf_below_a_composite_becomes_a_proxy_and_the_composite_stays(
    ticking_context: StatechartContext,
) -> None:
    statechart = Statechart.from_json(
        data=_document_naming_unimportable_classes(
            nodes=[
                Sequence(
                    name="fetch",
                    nodes=[
                        NodeObservingAFixedValue(
                            name="drive_to_table",
                            observation=ObservationStateValue.TRUE,
                        ),
                        _detector(),
                    ],
                )
            ]
        ),
        context=ticking_context,
    )

    assert [type(node) for node in statechart.nodes] == [
        Sequence,
        NodeObservingAFixedValue,
        ProxyNode,
    ]


def test_a_leaf_this_process_owns_keeps_its_class(
    ticking_context: StatechartContext,
) -> None:
    statechart = Statechart.from_json(
        data=_document(
            nodes=[
                NodeSucceedingOnObservingTrue(
                    observation=ObservationStateValue.TRUE,
                    owning_process=ProcessName.TICKING,
                )
            ]
        ),
        context=ticking_context,
    )

    assert type(statechart.nodes[0]) is NodeSucceedingOnObservingTrue


def test_a_context_without_ownership_keeps_the_class_of_an_owned_leaf(
    statechart_context: StatechartContext,
) -> None:
    statechart = Statechart.from_json(
        data=_document(nodes=[_detector()]), context=statechart_context
    )

    assert type(statechart.nodes[0]) is NodeSucceedingOnObservingTrue


def test_a_composite_refuses_an_owning_process() -> None:
    with pytest.raises(OwnedCompositeNodeError):
        Sequence(nodes=[], owning_process=ProcessName.PERCEPTION)


def test_a_leaf_ending_on_a_condition_of_its_own_cannot_be_sent_owned() -> None:
    with pytest.raises(IrreproducibleEndBehaviourError):
        _document(
            nodes=[
                NodeSucceedingOnAConditionOfItsOwn(
                    observation=ObservationStateValue.TRUE,
                    owning_process=ProcessName.PERCEPTION,
                )
            ]
        )


def test_a_proxy_writes_the_json_of_the_leaf_it_stands_for(
    ticking_context: StatechartContext,
) -> None:
    document = _document_naming_unimportable_classes(nodes=[_detector()])
    statechart = Statechart.from_json(data=document, context=ticking_context)

    assert (
        json.loads(json.dumps(statechart.to_json()))[StatechartJSONKey.NODES]
        == document[StatechartJSONKey.NODES]
    )


# %% what a proxy observes


def test_a_proxy_observes_unknown_until_its_owner_reports(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> None:
    proxy = _started_proxy(
        proxied=_detector(),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )

    for _ in range(3):
        statechart_executor.tick()

    assert proxy.observation_state == ObservationStateValue.UNKNOWN


def test_a_proxy_succeeds_once_its_owner_reports_true(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    ownership: NodeOwnership,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    proxy = _started_proxy(
        proxied=_detector(),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )

    ownership.take_observation(
        observation=OwnedNodeObservation(
            node_id=connection.sent[0].node_id,
            run=connection.sent[0].run,
            observation=ObservationStateValue.TRUE,
        )
    )
    statechart_executor.tick()

    assert proxy.life_cycle_state == LifeCycleValue.SUCCEEDED


def test_a_proxy_of_a_leaf_ended_by_its_owner_keeps_running_on_true(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    ownership: NodeOwnership,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    proxy = _started_proxy(
        proxied=NodeObservingAFixedValue(
            observation=ObservationStateValue.TRUE,
            owning_process=ProcessName.PERCEPTION,
        ),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )

    ownership.take_observation(
        observation=OwnedNodeObservation(
            node_id=connection.sent[0].node_id,
            run=connection.sent[0].run,
            observation=ObservationStateValue.TRUE,
        )
    )
    statechart_executor.tick()

    assert proxy.life_cycle_state == LifeCycleValue.RUNNING


def test_a_proxy_of_a_leaf_failing_on_false_fails_once_its_owner_reports_false(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    ownership: NodeOwnership,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    proxy = _started_proxy(
        proxied=NodeFailingOnObservingFalse(
            observation=ObservationStateValue.FALSE,
            owning_process=ProcessName.PERCEPTION,
        ),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )

    ownership.take_observation(
        observation=OwnedNodeObservation(
            node_id=connection.sent[0].node_id,
            run=connection.sent[0].run,
            observation=ObservationStateValue.FALSE,
        )
    )
    statechart_executor.tick()

    assert proxy.life_cycle_state == LifeCycleValue.FAILED


def test_a_proxy_ignores_an_observation_of_its_earlier_run(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    ownership: NodeOwnership,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    proxy = _started_proxy(
        proxied=NodeFailingOnObservingFalse(
            observation=ObservationStateValue.FALSE,
            owning_process=ProcessName.PERCEPTION,
        ),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )
    first_run = connection.sent[0]
    proxy.statechart.request_reset(node=proxy)
    statechart_executor.tick()
    statechart_executor.tick()

    ownership.take_observation(
        observation=OwnedNodeObservation(
            node_id=first_run.node_id,
            run=first_run.run,
            observation=ObservationStateValue.FALSE,
        )
    )
    statechart_executor.tick()

    assert proxy.life_cycle_state == LifeCycleValue.RUNNING


def test_an_observation_of_a_node_without_a_proxy_is_refused(
    ownership: NodeOwnership,
) -> None:
    with pytest.raises(UnknownProxyNodeError):
        ownership.take_observation(
            observation=OwnedNodeObservation(
                node_id="a node of no statechart here",
                run=1,
                observation=ObservationStateValue.TRUE,
            )
        )


# %% what a proxy sends


def test_a_proxy_sends_each_life_cycle_state_of_a_run_once(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    ownership: NodeOwnership,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    _started_proxy(
        proxied=_detector(),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )
    node_id = connection.sent[0].node_id
    ownership.take_observation(
        observation=OwnedNodeObservation(
            node_id=node_id, run=1, observation=ObservationStateValue.TRUE
        )
    )
    statechart_executor.tick()

    assert connection.sent == [
        OwnedNodeLifeCycle(
            owning_process=ProcessName.PERCEPTION,
            node_id=node_id,
            run=1,
            life_cycle=LifeCycleValue.RUNNING,
        ),
        OwnedNodeLifeCycle(
            owning_process=ProcessName.PERCEPTION,
            node_id=node_id,
            run=1,
            life_cycle=LifeCycleValue.SUCCEEDED,
        ),
    ]


def test_a_proxy_starting_paused_sends_paused_once(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    detector = _detector()
    detector.pause_condition = sm.Scalar.const_true()

    _started_proxy(
        proxied=detector,
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )

    assert [life_cycle.life_cycle for life_cycle in connection.sent] == [
        LifeCycleValue.PAUSED
    ]


def test_a_restarted_proxy_sends_its_next_run(
    ticking_context: StatechartContext,
    statechart_executor: StatechartExecutor,
    connection: ConnectionRecordingLifeCycles,
) -> None:
    proxy = _started_proxy(
        proxied=_detector(),
        ticking_context=ticking_context,
        statechart_executor=statechart_executor,
    )
    proxy.statechart.request_reset(node=proxy)
    statechart_executor.tick()
    statechart_executor.tick()

    assert [
        (life_cycle.run, life_cycle.life_cycle) for life_cycle in connection.sent
    ] == [
        (1, LifeCycleValue.RUNNING),
        (1, LifeCycleValue.NOT_STARTED),
        (2, LifeCycleValue.RUNNING),
    ]
