"""
Testing annotators for RoboKudo.

This module provides annotators for testing and debugging purposes.
It supports:

* Empty annotator for baseline testing
* Failing annotator for error handling testing
* Simulated processing delays
* Visual output generation
* Status feedback testing

The module is used for:

* System testing
* Error handling verification
* Performance testing
* Debug visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field

import copy
import time
import datetime

import cv2
import numpy as np
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.cas import CAS, CASViews


@dataclass(kw_only=True)
class SlowAnnotator(ThreadedAnnotator):
    """
    A slow annotator that demonstrates long-running processing using ThreadedAnnotator.

    This annotator simulates a time-consuming process by adding a configurable delay
    and generating visual output. It is useful for:

    * Testing thread handling
    * Performance monitoring
    * Timeout behavior verification
    * Visual feedback testing
    """

    sleep_seconds: float = 1.0
    """
    How long each computation sleeps.
    """

    def compute(self) -> Status:
        """
        Perform the main computation with artificial delay.

        This method:

        * Retrieves the color image from CAS
        * Adds timestamp and visual markers
        * Sleeps for configured duration
        * Sets feedback message

        :return: Success status after completion
        """
        self.robokudo_logger.debug(
            "%s.compute(): Start doing the heavy stuff" % self.__class__.__name__
        )

        vis = copy.deepcopy(self.get_cas().get(CASViews.COLOR_IMAGE))

        vis = cv2.putText(
            vis,
            str(datetime.datetime.now()),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        vis = cv2.rectangle(vis, (400, 400), (650, 650), (0, 0, 255), 2)
        self.get_annotator_output_struct().set_image(vis)

        time.sleep(self.sleep_seconds)

        self.robokudo_logger.debug(
            "%s.compute(): Stop doing the heavy stuff" % self.__class__.__name__
        )
        self.feedback_message = "SlowAnnotator was successful."

        return Status.SUCCESS


@dataclass(kw_only=True)
class EmptyAnnotator(BaseAnnotator):
    """
    Basic annotator that generates empty visual output.

    This annotator creates a black image and sets it as output. Used for testing
    visualization and output handling.
    """

    def compute(self) -> Status:
        """
        Generate empty visual output.

        Creates a black image and sets it as the annotator output.

        :return: SUCCESS status
        """
        vis = np.zeros((640, 480, 3), dtype="uint8")
        self.get_annotator_output_struct().set_image(vis)

        return Status.SUCCESS


@dataclass(kw_only=True)
class FailingAnnotator(ThreadedAnnotator):
    """
    Annotator that simulates failures for testing with a long-running annotator using
    the ThreadedAnnotator class.

    This annotator alternates between success and failure states, with configurable
    delays. Used for testing error handling.
    """

    counter: int = field(default=0, init=False)
    """
    Counts the computations, to alternate between success and failure.
    """

    def compute(self) -> Status:
        """
        Simulate processing with alternating success/failure.

        Sleeps for a fixed duration and alternates between success and failure states
        based on an internal counter.

        :return: SUCCESS or FAILURE status
        """
        self.robokudo_logger.debug(
            "%s.compute(): Start doing the heavy stuff" % self.__class__.__name__
        )
        time.sleep(0.5)  # in s

        if self.counter == 1:
            self.feedback_message = "I failed horribly!"
            self.counter = 0
            return Status.FAILURE
        else:
            self.counter += 1
            self.feedback_message = "Action successful"
            return Status.SUCCESS

        self.robokudo_logger.debug(
            "%s.compute(): Stop doing the heavy stuff" % self.__class__.__name__
        )


@dataclass(kw_only=True)
class FakeCollectionReaderAnnotator(BaseAnnotator):
    """
    A simulated collection reader for testing pipeline behavior.

    This annotator simulates a collection reader by:

    * Waiting for a configurable number of iterations
    * Creating synthetic CAS data
    * Managing feedback messages
    * Testing pipeline flow control

    Used for:

    * Pipeline integration testing
    * Flow control verification
    * Feedback message handling
    * CAS creation testing
    """

    counter: int = field(default=0, init=False)
    """
    Number of ticks since the reader started over; data arrives on the third.
    """

    def initialise(self) -> None:
        """
        Initialize the reader state.

        Called on first tick and whenever status changes from non-running. Resets
        counter and clears feedback messages for all children in sequence.
        """
        self.robokudo_logger.debug("%s.initialise()" % (self.__class__.__name__))
        self.counter = 0

        for child in self.parent.children:
            child.feedback_message = ""

    def compute(self) -> Status:
        """
        Update the reader state and generate synthetic data.

        This method:

        * Increments internal counter
        * Creates new CAS data after 3 iterations
        * Sets appropriate feedback messages
        * Updates processing status

        :return: Current processing status (RUNNING or SUCCESS)
        """
        self.counter += 1
        new_status = Status.SUCCESS if self.counter == 3 else Status.RUNNING
        if new_status == Status.SUCCESS:
            self.feedback_message = "Got sensor data!"
            # Create a new CAS
            blackboard = Blackboard()
            cas = CAS()
            cas.percepts = [time.time()]
            blackboard.set("CAS", cas)

        else:
            self.feedback_message = "Waiting for incoming sensor data ..."
        self.robokudo_logger.debug(
            "%s.update()[%s->%s][%s]"
            % (self.__class__.__name__, self.status, new_status, self.feedback_message)
        )
        return new_status
