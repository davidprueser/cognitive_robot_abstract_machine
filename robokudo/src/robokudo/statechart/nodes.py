"""
The nodes of a RoboKudo statechart: a pipeline that puts in a fresh CAS whenever it
starts, and annotators that work on that CAS on the tick thread or in a thread of their
own.
"""

from __future__ import annotations

# %% Imports

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.composites import Sequence
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValue
from cramph.node import FailsOnObservingFalse, StatechartNode, SucceedsOnObservingTrue
from cramph.threaded_nodes import ThreadedNode
from krrood.patterns.classproperty import classproperty

from robokudo.cas import CAS
from robokudo.statechart.cas_access import CASAccess

# %% nodes working on the CAS


@dataclass(eq=False, repr=False, kw_only=True)
class CASNode(StatechartNode):
    """
    A node that works on the current CAS of its statechart.

    A node that moves data into the CAS or out of it, such as one reading a camera frame
    or one writing results into the world, is one of these rather than an
    :class:`AnnotatorNode`, and is ticked on the tick thread, because the world is not
    thread-safe.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        Differs from :meth:`~cramph.node.StatechartNode.required_context_extensions` in
        also requiring :class:`~robokudo.statechart.cas_access.CASAccess`.
        """
        return super().required_context_extensions + (CASAccess,)

    @property
    def cas_access(self) -> CASAccess:
        """
        :return: The extension of this node's context that holds the CAS.
        """
        return self.context.require_extension(extension_type=CASAccess)

    @property
    def cas(self) -> CAS:
        """
        :return: The CAS this node's statechart works on now.
        """
        return self.cas_access.cas


# %% annotators


@dataclass(eq=False, repr=False, kw_only=True)
class AnnotatorNode(SucceedsOnObservingTrue, FailsOnObservingFalse, CASNode, ABC):
    """
    Annotates the CAS once when it is ticked, on the tick thread.

    It succeeds if it could annotate the CAS and fails if it could not; an error it
    raises is raised out of the tick.
    """

    @abstractmethod
    def annotate(self, *, cas: CAS) -> bool:
        """
        Reads from `cas` and adds what this annotator finds to it, without touching the
        world.

        :param cas: The CAS to annotate.
        :return: Whether this annotator could annotate `cas`.
        """

    @staticmethod
    def outcome_observation(*, could_annotate: bool) -> ObservationStateValue:
        """
        :param could_annotate: What :meth:`annotate` returned.
        :return: What an annotator observes once :meth:`annotate` returned that.
        """
        if could_annotate:
            return ObservationStateValue.TRUE
        return ObservationStateValue.FALSE

    def on_tick(self) -> ObservationStateValue:
        """
        Annotate the current CAS and observe whether that succeeded.
        """
        return self.outcome_observation(could_annotate=self.annotate(cas=self.cas))


@dataclass(eq=False, repr=False, kw_only=True)
class ThreadedAnnotatorNode(AnnotatorNode, ThreadedNode, ABC):
    """
    Annotates the CAS of its start in a thread of its own, and observes unknown until
    that returned, so a slow annotator does not hold up the ticks.

    A restart supersedes a run that is still going: that run finishes on the CAS of its
    own start, which the pipeline replaced, and its outcome is discarded.
    """

    _started_cas: CAS | None = field(default=None, init=False, repr=False)
    """
    The CAS that was current when this node last started.
    """

    def on_start(self) -> None:
        """
        Take the current CAS, then start :meth:`run`, see
        :meth:`~cramph.threaded_nodes.ThreadedNode.on_start`.
        """
        self._started_cas = self.cas
        super().on_start()

    def run(self) -> bool:
        return self.annotate(cas=self._started_cas)

    def on_tick(self) -> ObservationStateValue:
        """
        Observe whether the annotation succeeded, once it returned or raised.

        :raises BaseException: What :meth:`annotate` raised.
        """
        if not self.has_finished:
            return ObservationStateValue.UNKNOWN
        if self._error is not None:
            raise self._error
        return self.outcome_observation(could_annotate=self._result)


# %% pipelines


@dataclass(repr=False, eq=False, kw_only=True)
class PipelineNode(Sequence, CASNode):
    """
    Runs annotators one after another on a fresh CAS, which it puts in whenever it
    starts or resets.

    A statechart holds at most one pipeline, because every node of it works on the same
    CAS.
    """

    def set_up(self) -> None:
        """
        Register this pipeline as the one that fills the CAS of its context.

        :raises ~robokudo.statechart.exceptions.SecondPipelineError: If another pipeline
            already fills it.
        """
        super().set_up()
        self.cas_access.register_pipeline(pipeline=self)

    def on_start(self) -> None:
        """
        Put in a fresh CAS before the first annotator starts on it.
        """
        super().on_start()
        self.cas_access.start_new_cas()

    def on_reset(self) -> None:
        """
        Put in a fresh CAS, so nothing reads what the run before left.
        """
        super().on_reset()
        self.cas_access.start_new_cas()
