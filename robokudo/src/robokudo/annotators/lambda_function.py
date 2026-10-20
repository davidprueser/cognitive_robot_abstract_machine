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

from py_trees.common import Status
from typing_extensions import Callable, Dict, Optional, Tuple

from robokudo.annotators.core import BaseAnnotator


class LambdaFunctionAnnotator(BaseAnnotator):
    """
    Annotator for executing arbitrary functions.

    This annotator executes a provided function with configurable arguments, allowing
    for dynamic behavior definition without creating new annotator classes.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for lambda function annotator.
        """

        class Parameters:
            """
            Parameter container for function configuration.

            :type function: callable
            :type function_arguments: tuple
            :type function_keyword_arguments: dict
            """

            def __init__(self) -> None:
                self.function: Optional[Callable] = None
                self.function_arguments: Optional[Tuple] = None
                self.function_keyword_arguments: Optional[Dict] = None

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "LambdaFunctionAnnotator",
        descriptor: LambdaFunctionAnnotator.Descriptor | None = None,
    ) -> None:
        """
        Initialize the lambda function annotator.

        :param name: Annotator name
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """
        Execute the configured function.

        The function is called with the annotator instance as first argument, followed
        by any configured positional and keyword arguments.

        :return: SUCCESS status
        """
        function = self.descriptor.parameters.function

        if function:
            function_arguments = self.descriptor.parameters.function_arguments or []
            function_keyword_arguments = (
                self.descriptor.parameters.function_keyword_arguments or {}
            )

            function(self, *function_arguments, **function_keyword_arguments)

        return Status.SUCCESS
