"""
Lambda function annotator for RoboKudo.

This module provides an annotator for executing arbitrary functions.
It supports:

* Dynamic function execution
* Custom function arguments
* Flexible parameter passing
* Generic function handling

The module is used for:

* Custom processing
* Dynamic behavior
* Testing and debugging
* Quick prototyping
"""

from __future__ import annotations

from dataclasses import dataclass, field

from py_trees.common import Status
from typing_extensions import Any, Callable

from robokudo.annotators.core import BaseAnnotator


@dataclass(kw_only=True)
class LambdaFunctionAnnotator(BaseAnnotator):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments, allowing
    for dynamic behavior definition without creating new annotator classes.
    """

    function: Callable[..., Any] | None = None
    """
    The function to run; nothing runs if ``None``.

    It receives the annotator first.
    """

    arguments: tuple[Any, ...] = ()
    """
    Further positional arguments of :attr:`function`.
    """

    keyword_arguments: dict[str, Any] = field(default_factory=dict)
    """
    Keyword arguments of :attr:`function`.
    """

    def compute(self) -> Status:
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument, followed
        by any configured positional and keyword arguments.

        :return: SUCCESS status
        """
        if self.function is not None:
            self.function(self, *self.arguments, **self.keyword_arguments)

        return Status.SUCCESS
