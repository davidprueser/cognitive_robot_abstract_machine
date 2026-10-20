"""
Decorator utilities for RoboKudo annotators.
"""

from __future__ import annotations

import timeit
from collections import defaultdict

from typing_extensions import Callable, Any
from robokudo.annotators.core import BaseAnnotator


def timer_decorator(function: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Log execution time of decorated function.

    :param function: Function to be timed
    :return: Wrapped function that logs execution time

    :Example:

    .. code-block:: python

        @timer_decorator
        def my_function():
            # Function code here
            pass

    .. note::
        Uses robokudo_logger if available, otherwise prints to stdout
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = None
        class_name = None

        if args and hasattr(args[0], "robokudo_logger"):
            logger = args[0].robokudo_logger
            class_name = args[0].__class__
        elif "self" in kwargs and hasattr(kwargs["self"], "robokudo_logger"):
            logger = kwargs["self"].robokudo_logger
            class_name = kwargs["self"].__class__

        start_time = timeit.default_timer()
        result = function(*args, **kwargs)
        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        if class_name:
            log_message = f"Function '{class_name}.{function.__name__}' took {processing_time:.3f} seconds to execute."
        else:
            log_message = f"Function '{function.__name__}' took {processing_time:.3f} seconds to execute."

        if logger:
            logger.info(log_message)
        else:
            print(log_message)

        return result

    return wrapper


def record_time(function: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Record execution time of annotator method.

    :param function: Annotator method to time
    :return: Wrapped method that records execution time

    :Example:

    .. code-block:: python

        @record_time
        def compute(self):
            # Method code here
            pass

    .. note::
        Only works on annotator instance methods
    """
    function_name = function.__name__

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        self = args[0]
        if not hasattr(self, "_times"):
            setattr(self, "_times", defaultdict(list))
        start_time = timeit.default_timer()
        result = function(*args, **kwargs)
        end_time = timeit.default_timer()
        processing_time = end_time - start_time
        # self._times[function_name].append(processing_time)
        self._times[function_name] = processing_time
        return result

    return wrapper


def publish_variables(function: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Publish annotator variables after method execution.

    :param function: Annotator method to wrap
    :return: Wrapped method that publishes variables
    :raises AssertionError: If not used on BaseAnnotator instance method

    :Example:

    .. code-block:: python

        @publish_variables
        def update(self):
            # Method code here
            pass

    .. note::
        Only works on BaseAnnotator instance methods
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = function(*args, **kwargs)
        assert isinstance(args[0], BaseAnnotator)
        args[0].publish_variables()
        return result

    return wrapper
