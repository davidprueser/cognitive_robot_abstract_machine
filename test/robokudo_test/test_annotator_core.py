from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from py_trees.common import Status

import robokudo.annotators.core
from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.utils.error_handling import (
    get_blackboard_exception,
    raise_to_blackboard,
)

# %% annotators the tests run


@dataclass(kw_only=True)
class SucceedingAnnotator(BaseAnnotator):
    """
    Returns success without touching anything else.
    """

    compute_calls: int = field(default=0, init=False)
    """
    How often :meth:`compute` ran.
    """

    def compute(self) -> Status:
        self.compute_calls += 1
        return Status.SUCCESS


@dataclass(kw_only=True)
class ReportingAnnotator(BaseAnnotator):
    """
    Sets its own feedback message.
    """

    message: str = "found two cups"
    """
    The feedback message :meth:`compute` sets.
    """

    def compute(self) -> Status:
        self.feedback_message = self.message
        return Status.SUCCESS


@dataclass
class ComputationFailed(Exception):
    """
    The error the failing annotators raise.
    """


@dataclass(kw_only=True)
class FailingAnnotator(BaseAnnotator):
    """
    Raises from :meth:`compute`.
    """

    error: ComputationFailed = field(default_factory=ComputationFailed)
    """
    The error :meth:`compute` raises.
    """

    def compute(self) -> Status:
        raise self.error


@dataclass(kw_only=True)
class ConfiguredAnnotator(BaseAnnotator):
    """
    Has a configuration field with a mutable default.
    """

    class_names: list[str] = field(default_factory=list)
    """
    A configuration value each instance owns.
    """

    def compute(self) -> Status:
        return Status.SUCCESS


@dataclass(kw_only=True)
class ThreadedSucceedingAnnotator(ThreadedAnnotator):
    """
    Returns success from its worker thread.
    """

    def compute(self) -> Status:
        return Status.SUCCESS


@dataclass(kw_only=True)
class ThreadedFailingAnnotator(ThreadedAnnotator):
    """
    Raises from its worker thread.
    """

    error: ComputationFailed = field(default_factory=ComputationFailed)
    """
    The error :meth:`compute` raises.
    """

    def compute(self) -> Status:
        raise self.error


def update_until_finished(annotator: ThreadedAnnotator) -> Status:
    """
    Tick a threaded annotator until its computation finished.
    """
    deadline = time.monotonic() + 5.0
    status = annotator.update()
    while status == Status.RUNNING and time.monotonic() < deadline:
        time.sleep(0.01)
        status = annotator.update()
    return status


# %% construction


def test_name_defaults_to_the_class_name() -> None:
    assert SucceedingAnnotator().name == SucceedingAnnotator.__name__


def test_name_is_the_behaviour_name_in_the_tree() -> None:
    annotator = SucceedingAnnotator(name="CupDetector")

    assert annotator.qualified_name.endswith(annotator.name)


def test_fields_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        SucceedingAnnotator("CupDetector")


def test_unknown_configuration_is_rejected() -> None:
    with pytest.raises(TypeError):
        SucceedingAnnotator(threshold=0.5)


def test_instances_with_equal_configuration_stay_distinct_tree_nodes() -> None:
    first = ConfiguredAnnotator()
    second = ConfiguredAnnotator()

    assert first != second
    assert len({first, second}) == 2


def test_instances_do_not_share_mutable_defaults() -> None:
    first = ConfiguredAnnotator()
    second = ConfiguredAnnotator()

    first.class_names.append("cup")

    assert second.class_names == []


# %% update


def test_update_returns_the_status_compute_returns() -> None:
    assert SucceedingAnnotator().update() == Status.SUCCESS


def test_update_reports_the_processing_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([1.0, 1.5])
    monkeypatch.setattr(
        robokudo.annotators.core, "default_timer", lambda: next(timestamps)
    )
    annotator = SucceedingAnnotator()

    annotator.update()

    assert annotator.feedback_message == "Processing took 0.5000s"


def test_update_keeps_the_feedback_compute_sets() -> None:
    annotator = ReportingAnnotator()

    annotator.update()

    assert annotator.feedback_message == annotator.message


def test_update_records_an_error_of_compute_and_fails() -> None:
    annotator = FailingAnnotator()

    status = annotator.update()

    assert status == Status.FAILURE
    assert get_blackboard_exception() is annotator.error


def test_update_skips_compute_while_an_error_is_recorded() -> None:
    raise_to_blackboard(ComputationFailed())
    annotator = SucceedingAnnotator()

    status = annotator.update()

    assert status == Status.FAILURE
    assert annotator.compute_calls == 0


# %% threaded update


def test_threaded_update_runs_until_compute_finished() -> None:
    annotator = ThreadedSucceedingAnnotator()

    assert annotator.update() == Status.RUNNING
    assert update_until_finished(annotator) == Status.SUCCESS


def test_threaded_update_records_an_error_of_compute_and_fails() -> None:
    annotator = ThreadedFailingAnnotator()

    status = update_until_finished(annotator)

    assert status == Status.FAILURE
    assert get_blackboard_exception() is annotator.error


def test_threaded_update_starts_a_new_computation_after_one_finished() -> None:
    annotator = ThreadedSucceedingAnnotator()
    update_until_finished(annotator)

    assert annotator.update() == Status.RUNNING
