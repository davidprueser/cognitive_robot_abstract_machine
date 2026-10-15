"""
Decorator utilities for RoboKudo annotators.
"""

from __future__ import annotations

import timeit

from typing_extensions import Callable, Any


def timer_decorator(func: Callable[[...], Any]) -> Callable[[...], Any]:
    """
    Log execution time of decorated function.

    :param func: Function to be timed
    :return: Wrapped function that logs execution time

    :Example:

    .. code-block:: python

        @timer_decorator
        def my_function():
            # Function code here
            pass

    .. note::
        Uses rk_logger if available, otherwise prints to stdout
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = None
        class_name = None

        if args and hasattr(args[0], "rk_logger"):
            logger = args[0].rk_logger
            class_name = args[0].__class__
        elif "self" in kwargs and hasattr(kwargs["self"], "rk_logger"):
            logger = kwargs["self"].rk_logger
            class_name = kwargs["self"].__class__

        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        if class_name:
            log_msg = f"Function '{class_name}.{func.__name__}' took {processing_time:.3f} seconds to execute."
        else:
            log_msg = f"Function '{func.__name__}' took {processing_time:.3f} seconds to execute."

        if logger:
            logger.info(log_msg)
        else:
            print(log_msg)

        return result

    return wrapper
