"""
Annotator nodes with fixed, simple behaviour, a switch the test sets, and ticking until
a condition holds, for the tests of RoboKudo's statechart nodes.
"""

from __future__ import annotations

# %% Imports

import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta

from cramph.data_types import ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.node import EndedByOwner, StatechartNode
from typing_extensions import Callable

from robokudo.cas import CAS
from robokudo.statechart.nodes import AnnotatorNode, ThreadedAnnotatorNode

# %% errors the annotators raise


class AnnotationError(Exception):
    """
    An error an annotator raises while it annotates.
    """


# %% annotating on the tick thread


@dataclass(eq=False, repr=False, kw_only=True)
class AnnotatorRecordingItsCAS(AnnotatorNode):
    """
    Records every CAS it annotates and the thread it annotates in, and reports a fixed
    outcome.
    """

    could_annotate: bool = True
    """
    What :meth:`annotate` returns.
    """

    annotated_cas_instances: list[CAS] = field(default_factory=list)
    """
    The CAS of every call of :meth:`annotate`, in order.
    """

    annotating_threads: list[threading.Thread] = field(default_factory=list)
    """
    The thread of every call of :meth:`annotate`, in order.
    """

    def annotate(self, *, cas: CAS) -> bool:
        self.annotated_cas_instances.append(cas)
        self.annotating_threads.append(threading.current_thread())
        return self.could_annotate


@dataclass(eq=False, repr=False, kw_only=True)
class AnnotatorRaising(AnnotatorNode):
    """
    Raises :class:`AnnotationError` whenever it annotates.
    """

    def annotate(self, *, cas: CAS) -> bool:
        raise AnnotationError()


# %% annotating in a thread


@dataclass(kw_only=True)
class AnnotationReleasedByTheTest:
    """
    One call of :meth:`ThreadedAnnotatorReleasedByTheTest.annotate`, held until the test
    releases it.
    """

    release: threading.Event = field(default_factory=threading.Event)
    """
    Set by the test to let the call end.
    """

    could_annotate: bool = True
    """
    What the call returns once released, unless it raises :attr:`error`.
    """

    error: BaseException | None = None
    """
    What the call raises once released; None if it returns.
    """

    cas: CAS | None = None
    """
    The CAS the call annotates, once it started.
    """

    thread: threading.Thread | None = None
    """
    The thread the call runs in, once it started.
    """

    def wait_until_ended(self) -> None:
        """
        Block until the call's thread ended, its outcome kept or discarded.
        """
        self.thread.join(timeout=10)


@dataclass(eq=False, repr=False, kw_only=True)
class ThreadedAnnotatorReleasedByTheTest(ThreadedAnnotatorNode):
    """
    A threaded annotator whose n-th call of :meth:`annotate` waits until the test
    releases :attr:`annotations` [n].
    """

    annotations: list[AnnotationReleasedByTheTest]
    """
    The calls this annotator goes through, in the order they start.
    """

    started_annotations: int = field(default=0, init=False)
    """
    How many calls have started.
    """

    def annotate(self, *, cas: CAS) -> bool:
        annotation = self.annotations[self.started_annotations]
        self.started_annotations += 1
        annotation.cas = cas
        annotation.thread = threading.current_thread()
        annotation.release.wait(timeout=10)
        if annotation.error is not None:
            raise annotation.error
        return annotation.could_annotate


# %% a switch the test sets


@dataclass(eq=False, repr=False, kw_only=True)
class SwitchSetByTheTest(EndedByOwner, StatechartNode):
    """
    A node observing what the test last set, so other nodes' transitions can follow it.
    """

    observation: ObservationStateValue = ObservationStateValue.FALSE
    """
    What this node observes on its next tick.
    """

    def on_tick(self) -> ObservationStateValue:
        """
        Observe what the test last set in :attr:`observation`.
        """
        return self.observation


# %% ticking


def tick_until(
    *,
    statechart_executor: StatechartExecutor,
    condition: Callable[[], bool],
    timeout: timedelta = timedelta(seconds=10),
) -> None:
    """
    Tick until `condition` holds, giving annotator threads time to run.

    :raises TimeoutError: If it does not hold within `timeout`.
    """
    deadline = time.monotonic() + timeout.total_seconds()
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError()
        statechart_executor.tick()
