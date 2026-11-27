"""
Tests for :class:`~robokudo.statechart.nodes.PipelineNode`, which runs annotators one
after another on a fresh CAS every time it starts, and for
:class:`~robokudo.statechart.cas_access.CASAccess`, which holds that CAS.
"""

from __future__ import annotations

# %% Imports

import pytest
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart

from robokudo.statechart.cas_access import CASAccess
from robokudo.statechart.exceptions import SecondPipelineError
from robokudo.statechart.nodes import PipelineNode

from ._annotator_mimics import (
    AnnotationReleasedByTheTest,
    AnnotatorRecordingItsCAS,
    SwitchSetByTheTest,
    ThreadedAnnotatorReleasedByTheTest,
    tick_until,
)

# %% fresh CAS


def test_every_fresh_cas_takes_the_next_id() -> None:
    cas_access = CASAccess()

    cas_access.start_new_cas()
    first_id = cas_access.cas.cas_id
    cas_access.start_new_cas()

    assert cas_access.cas.cas_id == first_id + 1


def test_a_pipeline_puts_a_fresh_cas_in_when_it_starts(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    cas_before_the_start = cas_access.cas
    annotator = AnnotatorRecordingItsCAS()
    pipeline = PipelineNode(nodes=[annotator])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=pipeline)
    statechart_executor.compile(statechart=statechart)

    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state.is_terminal,
    )

    assert pipeline.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert annotator.annotated_cas_instances == [cas_access.cas]
    assert cas_access.cas is not cas_before_the_start


def test_the_first_annotator_of_a_pipeline_starts_on_the_fresh_cas(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    """
    A threaded annotator starting in the same tick as its pipeline takes the CAS the
    pipeline put in, not the one before it.
    """
    cas_before_the_start = cas_access.cas
    annotation = AnnotationReleasedByTheTest()
    annotation.release.set()
    annotator = ThreadedAnnotatorReleasedByTheTest(annotations=[annotation])
    pipeline = PipelineNode(nodes=[annotator])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=pipeline)
    statechart_executor.compile(statechart=statechart)

    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state.is_terminal,
    )

    assert annotation.cas is cas_access.cas
    assert annotation.cas is not cas_before_the_start


def test_a_reset_pipeline_puts_a_fresh_cas_in(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    annotator = AnnotatorRecordingItsCAS()
    pipeline = PipelineNode(nodes=[annotator])
    reset = SwitchSetByTheTest()
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=reset)
    statechart.add_node(node=pipeline)
    pipeline.start_condition = reset.observes_false
    pipeline.reset_condition = reset.observes_true
    statechart_executor.compile(statechart=statechart)
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state == LifeCycleValue.SUCCEEDED,
    )
    annotated_cas = cas_access.cas

    reset.observation = ObservationStateValue.TRUE
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state == LifeCycleValue.NOT_STARTED,
    )

    assert cas_access.cas is not annotated_cas


# %% restarts


def test_a_restarted_threaded_annotator_keeps_only_the_outcome_on_the_latest_cas(
    statechart_executor: StatechartExecutor, cas_access: CASAccess
) -> None:
    """
    The annotation a restart superseded worked on the CAS of the earlier start, and its
    outcome does not decide the annotator's, although it ends last.
    """
    superseded_annotation = AnnotationReleasedByTheTest(could_annotate=False)
    latest_annotation = AnnotationReleasedByTheTest()
    annotator = ThreadedAnnotatorReleasedByTheTest(
        annotations=[superseded_annotation, latest_annotation]
    )
    pipeline = PipelineNode(nodes=[annotator])
    restart = SwitchSetByTheTest()
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=restart)
    statechart.add_node(node=pipeline)
    pipeline.start_condition = restart.observes_false
    pipeline.reset_condition = restart.observes_true
    statechart_executor.compile(statechart=statechart)
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: annotator.started_annotations == 1,
    )

    restart.observation = ObservationStateValue.TRUE
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state == LifeCycleValue.NOT_STARTED,
    )
    restart.observation = ObservationStateValue.FALSE
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: annotator.started_annotations == 2,
    )
    latest_annotation.release.set()
    latest_annotation.wait_until_ended()
    superseded_annotation.release.set()
    superseded_annotation.wait_until_ended()
    tick_until(
        statechart_executor=statechart_executor,
        condition=lambda: pipeline.life_cycle_state.is_terminal,
    )

    assert annotator.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert latest_annotation.cas is cas_access.cas
    assert superseded_annotation.cas is not cas_access.cas


# %% one pipeline per statechart


def test_a_second_pipeline_in_the_same_context_is_refused(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=PipelineNode(nodes=[AnnotatorRecordingItsCAS()]))
    statechart.add_node(node=PipelineNode(nodes=[AnnotatorRecordingItsCAS()]))

    with pytest.raises(SecondPipelineError):
        statechart_executor.compile(statechart=statechart)
