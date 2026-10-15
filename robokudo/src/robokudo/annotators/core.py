"""
Base classes of RoboKudo annotators.

An annotator is a dataclass: its init fields are its configuration and its other fields
are the state it builds while running. Annotators implement
:meth:`BaseAnnotator.compute`; the base classes run it, time it and record its errors.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from timeit import default_timer

from py_trees.behaviour import Behaviour
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from typing_extensions import Any

import robokudo.pipeline
from robokudo.annotators.outputs import AnnotatorOutputs, AnnotatorOutputStruct
from robokudo.cas import CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import ParentPipelineMissing
from robokudo.utils.error_handling import has_blackboard_exception, raise_to_blackboard
from robokudo.utils.tree import find_parent_of_type

# %% pipeline members


class PipelineMember:
    """
    A behaviour that works on the CAS of the pipeline it is part of.
    """

    def get_cas(self) -> CAS:
        """
        :return: The CAS of the pipeline this behaviour is part of.
        """
        return self.get_parent_pipeline().cas

    def get_parent_pipeline(self) -> robokudo.pipeline.Pipeline:
        """
        :return: The pipeline this behaviour is part of.
        :raises ParentPipelineMissing: If it is not part of a pipeline.
        """
        pipeline = find_parent_of_type(self, robokudo.pipeline.Pipeline)
        if pipeline is None:
            raise ParentPipelineMissing(behaviour_name=self.name)
        return pipeline


# %% annotators


@dataclass(kw_only=True, eq=False)
class BaseAnnotator(Behaviour, PipelineMember):
    """
    A behaviour that analyzes the CAS of its pipeline and annotates it.

    Each subclass is a ``@dataclass(kw_only=True)``. Its :attr:`name` defaults to the
    subclass name, and two instances are always distinct tree nodes, whatever their
    fields.

    Annotators may obtain RoboKudo's shared ROS node at any point in their lifecycle,
    including while its executor is spinning, and may create and use ROS entities on it.
    The application owns that node; annotators must not destroy it or shut down its
    context.
    """

    name: str = "BaseAnnotator"
    """
    Name of the annotator in the behaviour tree and in the annotations it creates.
    """

    visualize: bool = True
    """
    Whether the annotator writes its results to the visualization outputs.
    """

    robokudo_logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(PACKAGE_NAME),
        init=False,
        repr=False,
    )
    """
    The logger of the RoboKudo package.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Default :attr:`name` to the subclass name and keep identity equality, before
        ``@dataclass`` processes the subclass.
        """
        super().__init_subclass__(**kwargs)
        cls.__eq__ = object.__eq__
        cls.__hash__ = object.__hash__
        annotations = cls.__dict__.get("__annotations__", {})
        if "name" in annotations:
            return
        cls.__annotations__ = {"name": "str", **annotations}
        cls.name = field(default=cls.__name__, kw_only=True)

    def __post_init__(self) -> None:
        Behaviour.__init__(self, self.name)

    def compute(self) -> Status:
        """
        Analyze the CAS and annotate it; an annotator that overrides nothing succeeds
        without changing the CAS.

        :return: The status of the annotator after this tick.
        """
        return Status.SUCCESS

    def update(self) -> Status:
        """
        Run :meth:`compute` and report its processing time as feedback, unless
        :meth:`compute` set a feedback message itself.

        An error :meth:`compute` raises is recorded on the blackboard and fails the
        annotator. While an error is recorded, :meth:`compute` does not run.

        :return: The status :meth:`compute` returned, or failure.
        """
        if has_blackboard_exception():
            return Status.FAILURE
        self.feedback_message = ""
        start = default_timer()
        try:
            status = self.compute()
        except Exception as error:
            return self.fail_with(error)
        self.report_processing_time(default_timer() - start)
        return status

    def fail_with(self, error: BaseException) -> Status:
        """
        Record an error of :meth:`compute` on the blackboard.

        :param error: The error :meth:`compute` raised.
        :return: Failure.
        """
        self.robokudo_logger.error("%s failed", self.name, exc_info=error)
        raise_to_blackboard(error)
        return Status.FAILURE

    def report_processing_time(self, seconds: float) -> None:
        """
        Set the processing time as feedback, unless :meth:`compute` set a feedback
        message.

        :param seconds: How long :meth:`compute` took.
        """
        if self.feedback_message == "":
            self.feedback_message = f"Processing took {seconds:.4f}s"

    def setup(self, **kwargs: Any) -> bool:
        """
        Perform delayed initialization.

        This method is called for initialization tasks after the constructor has been
        called. Setup may run after the shared ROS node's executor has started. An
        annotator may obtain that node with
        :meth:`semantic_digital_twin.adapters.ros.node_registry.ROSNodeRegistry.get`
        and create or use ROS entities on it. See :class:`BaseAnnotator` for its
        ownership contract.
        """
        return True

    def key_callback(self, key: int) -> None:
        """
        Handle a key pressed in the visualizer window.

        :param key: Key code of the pressed key.
        """

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: int
    ) -> None:
        """
        Handle a mouse event in the visualizer window.

        :param event: The event type (see ``cv2.EVENT_*``).
        :param x: The x coordinate of the mouse pointer.
        :param y: The y coordinate of the mouse pointer.
        :param flags: Flags passed by OpenCV.
        :param param: Parameters passed by OpenCV.
        """

    def get_annotator_outputs(self) -> AnnotatorOutputs:
        """
        :return: The visualization outputs of all annotators in the pipeline.
        """
        annotator_output_pipeline_map_buffer = Blackboard().get(
            "annotator_output_pipeline_map_buffer"
        )
        return annotator_output_pipeline_map_buffer.map[self.get_parent_pipeline().name]

    def get_annotator_output_struct(self) -> AnnotatorOutputStruct:
        """
        :return: The visualization output of this annotator, created on first access.
        """
        annotator_outputs = self.get_annotator_outputs()
        if self.name not in annotator_outputs.outputs:
            annotator_outputs.init_annotator(self.name)
        return annotator_outputs.outputs[self.name]


@dataclass(kw_only=True)
class ThreadedAnnotator(BaseAnnotator):
    """
    An annotator whose :meth:`compute` runs in a worker thread, so that ticking the tree
    does not block while it runs.
    """

    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1),
        init=False,
        repr=False,
    )
    """
    The worker thread :meth:`compute` runs in.
    """

    computation: Future[Status] | None = field(default=None, init=False, repr=False)
    """
    The running computation, or ``None`` if none was started since the last finished.
    """

    computation_start: float = field(default=0.0, init=False, repr=False)
    """
    When the running computation started, as :func:`timeit.default_timer` time.
    """

    def update(self) -> Status:
        """
        Start :meth:`compute` in the worker thread, report running while it runs and
        then its result, as :meth:`BaseAnnotator.update` does.

        :return: Running, the status :meth:`compute` returned, or failure.
        """
        if self.computation is None:
            if has_blackboard_exception():
                return Status.FAILURE
            self.feedback_message = ""
            self.computation_start = default_timer()
            self.computation = self.executor.submit(self.compute)
            return Status.RUNNING
        if not self.computation.done():
            return Status.RUNNING
        computation, self.computation = self.computation, None
        error = computation.exception()
        if error is not None:
            return self.fail_with(error)
        self.report_processing_time(default_timer() - self.computation_start)
        return computation.result()
