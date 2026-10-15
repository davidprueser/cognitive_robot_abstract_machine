"""
Filter annotator for RoboKudo.

This module provides an annotator for filtering annotations based on custom conditions.
It supports:

* Dynamic filtering through callable functions
* Custom filter arguments
* In-place annotation list modification
* Flexible condition evaluation

The module is used for:

* Annotation filtering
* Data preprocessing
* Result refinement
* Conditional processing
"""

from __future__ import annotations

from dataclasses import dataclass, field

from py_trees.common import Status
from typing_extensions import Any, Callable

from robokudo.annotators.core import BaseAnnotator


@dataclass(kw_only=True)
class FilterAnnotator(BaseAnnotator):
    """
    Annotator for applying filter conditions to annotations.

    This annotator applies a provided filter function to the current set of annotations,
    modifying the annotation list in-place based on the filter results.
    """

    function: Callable[..., Any] | None = None
    """
    Decides whether an annotation is kept; all are kept if ``None``.

    It receives the annotation first.
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
        Apply the filter function to current annotations.

        The filter function is applied to each annotation with the configured arguments.
        Only annotations that pass the filter are kept.

        :return: SUCCESS status
        """
        if self.function is None:
            return Status.SUCCESS
        self.get_cas().annotations = [
            annotation
            for annotation in self.get_cas().annotations
            if self.function(annotation, *self.arguments, **self.keyword_arguments)
        ]

        return Status.SUCCESS
