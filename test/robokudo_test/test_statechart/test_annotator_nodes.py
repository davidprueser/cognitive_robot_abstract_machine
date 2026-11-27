"""
Tests for :class:`~robokudo.statechart.nodes.AnnotatorNode`, which annotates the CAS on
the tick thread, and :class:`~robokudo.statechart.nodes.ThreadedAnnotatorNode`, which
annotates it in a thread of its own without holding up the ticks.
"""

from __future__ import annotations

# %% Imports

import threading

import pytest
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.exceptions import NodesMissingContextExtensionsError
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from semantic_digital_twin.world import World

from robokudo.statechart.cas_access import CASAccess

from ._annotator_mimics import (
    AnnotationError,
    AnnotationReleasedByTheTest,
    AnnotatorRaising,
    AnnotatorRecordingItsCAS,
    ThreadedAnnotatorReleasedByTheTest,
    tick_until,
)

# %% running one annotator


def run_until_ended(
    *, statechart_executor: StatechartExecutor, annotator: StatechartNode
) -> None:
    """
    Tick a statechart holding only `annotator` until `annotator` ended.
    """
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=annotator)
    statechart_executor.compile(statechart=statechart)
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: annotator.life_cycle_state.is_terminal,
    )


def released_annotation(*, could_annotate: bool = True) -> AnnotationReleasedByTheTest:
    """
    :param could_annotate: What the call returns.
    :return: A call of a threaded annotator that returns as soon as it starts.
    """
    annotation = AnnotationReleasedByTheTest(could_annotate=could_annotate)
    annotation.release.set()
    return annotation


# %% annotating on the tick thread


def test_an_annotator_that_could_annotate_succeeds_on_the_cas_of_its_context(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    annotator = AnnotatorRecordingItsCAS()

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotator.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert annotator.annotated_cas_instances == [cas_access.cas]


def test_an_annotator_that_could_not_annotate_fails(
    statechart_executor: StatechartExecutor,
) -> None:
    annotator = AnnotatorRecordingItsCAS(could_annotate=False)

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotator.life_cycle_state == LifeCycleValue.FAILED


def test_an_annotator_annotates_on_the_tick_thread(
    statechart_executor: StatechartExecutor,
) -> None:
    annotator = AnnotatorRecordingItsCAS()

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotator.annotating_threads == [threading.current_thread()]


def test_an_error_of_an_annotator_is_raised_out_of_the_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    with pytest.raises(AnnotationError):
        run_until_ended(
            statechart_executor=statechart_executor, annotator=AnnotatorRaising()
        )


def test_an_annotator_is_compiled_only_in_a_context_holding_the_cas() -> None:
    statechart_executor = StatechartExecutor(context=StatechartContext(world=World()))
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=AnnotatorRecordingItsCAS())

    with pytest.raises(NodesMissingContextExtensionsError):
        statechart_executor.compile(statechart=statechart)


# %% annotating in a thread


def test_a_threaded_annotator_that_could_annotate_succeeds_on_the_cas_of_its_context(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    annotation = released_annotation()
    annotator = ThreadedAnnotatorReleasedByTheTest(annotations=[annotation])

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotator.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert annotation.cas is cas_access.cas


def test_a_threaded_annotator_that_could_not_annotate_fails(
    statechart_executor: StatechartExecutor,
) -> None:
    annotator = ThreadedAnnotatorReleasedByTheTest(
        annotations=[released_annotation(could_annotate=False)]
    )

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotator.life_cycle_state == LifeCycleValue.FAILED


def test_a_threaded_annotator_annotates_in_a_thread_of_its_own(
    statechart_executor: StatechartExecutor,
) -> None:
    annotation = released_annotation()
    annotator = ThreadedAnnotatorReleasedByTheTest(annotations=[annotation])

    run_until_ended(statechart_executor=statechart_executor, annotator=annotator)

    assert annotation.thread is not threading.current_thread()


def test_an_error_of_a_threaded_annotator_is_raised_out_of_the_tick(
    statechart_executor: StatechartExecutor,
) -> None:
    annotation = released_annotation()
    annotation.error = AnnotationError()
    annotator = ThreadedAnnotatorReleasedByTheTest(annotations=[annotation])

    with pytest.raises(AnnotationError):
        run_until_ended(statechart_executor=statechart_executor, annotator=annotator)


def test_the_ticks_go_on_while_a_threaded_annotator_annotates(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    An annotation still running neither holds up the ticks nor ends the node.
    """
    annotation = AnnotationReleasedByTheTest()
    annotator = ThreadedAnnotatorReleasedByTheTest(annotations=[annotation])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=annotator)
    statechart_executor.compile(statechart=statechart)

    for _ in range(5):
        statechart_executor.tick()

    assert not annotation.release.is_set()
    assert annotator.life_cycle_state == LifeCycleValue.RUNNING
    assert annotator.observation_state == ObservationStateValue.UNKNOWN
    annotation.release.set()
