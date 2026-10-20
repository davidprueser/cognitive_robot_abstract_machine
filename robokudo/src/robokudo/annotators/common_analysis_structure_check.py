"""
Common analysis structure condition checking utilities.

This module provides annotators for checking conditions in the common analysis structure.
The annotators follow these principles:

* Return SUCCESS if condition is true
* Return FAILURE if condition is false
* Never return RUNNING status

.. note::
   For RUNNING status checks, use CommonAnalysisStructureCondition from common_analysis_structure_condition.py instead.
"""

import functools

from py_trees.common import Status
from typing_extensions import Callable, Type, Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructure
from robokudo.exceptions import (
    CommonAnalysisStructureCheckConfigurationError,
    CommonAnalysisStructureCheckFailed,
)
from robokudo.types.scene import ObjectHypothesis


class CommonAnalysisStructureCheckFunc(BaseAnnotator):
    """
    Function-based common analysis structure condition checker.

    Evaluates a given function that checks conditions in the common analysis structure and returns:

    * SUCCESS if function returns True
    * FAILURE if function returns False

    .. warning::
       Will raise an exception if initialized without a function.
    """

    def __init__(
        self,
        name: str = "CommonAnalysisStructureCheckFunc",
        func: Optional[Callable[[CommonAnalysisStructure], bool]] = None,
        raise_with_str: str = "",
    ) -> None:
        """
        Initialize the common analysis structure condition checker.

        :param name: Name of this node in the behavior tree
        :param func: Function that evaluates common analysis structure conditions, must return bool
        :param raise_with_str: Error message to raise on failure, empty string disables
            raising
        :raises CommonAnalysisStructureCheckConfigurationError: If func is None
        """
        super(CommonAnalysisStructureCheckFunc, self).__init__(name=name)
        if func is None:
            raise CommonAnalysisStructureCheckConfigurationError(component_name=name)
        self.func = func
        self.raise_with_str = raise_with_str

    def update(self) -> Status:
        """
        Check the common analysis structure condition.

        :return: SUCCESS if condition is True, FAILURE if False
        :raises CommonAnalysisStructureCheckFailed: If raise_with_str is set and condition is False
        """
        common_analysis_structure = self.get_common_analysis_structure()
        if self.func(common_analysis_structure):
            return Status.SUCCESS
        else:
            if self.raise_with_str != "":
                raise CommonAnalysisStructureCheckFailed(reason=self.raise_with_str)
            else:
                return Status.FAILURE


def any_of_type_present(
    annotation_type: Type, common_analysis_structure: CommonAnalysisStructure
) -> bool:
    """
    Check if any annotations of a specific type exist in the common analysis structure.

    :param annotation_type: Type of annotation to check for
    :param common_analysis_structure: Common Analysis System instance
    :return: True if at least one annotation exists, False otherwise
    """
    annotations = common_analysis_structure.filter_annotations_by_type(annotation_type)
    return len(annotations) > 0


class CommonAnalysisStructureCheckAnnotationTypeExists(
    CommonAnalysisStructureCheckFunc
):
    """
    Check for existence of specific annotation types.

    Specialized CommonAnalysisStructureCheckFunc that:

    * Checks for presence of specific annotation types
    * Returns SUCCESS if at least one annotation exists
    * Returns FAILURE if no annotations exist
    """

    def __init__(
        self,
        name: str = "CommonAnalysisStructureCheckAnnotationTypeExists",
        annotation_type: Optional[Type] = None,
        raise_with_str: str = "",
    ) -> None:
        """
        Initialize annotation type checker.

        :param name: Name of this node in the behavior tree
        :param annotation_type: Type of annotation to check for
        :param raise_with_str: Error message to raise on failure, empty string disables
            raising. This is useful in scenarios where you want to raise an error which
            will be sent via the Query Interface to the caller.
        """
        func = functools.partial(any_of_type_present, annotation_type)
        super(CommonAnalysisStructureCheckAnnotationTypeExists, self).__init__(
            name=name, func=func, raise_with_str=raise_with_str
        )


class CommonAnalysisStructureCheckOHExists(
    CommonAnalysisStructureCheckAnnotationTypeExists
):
    """
    Check for existence of ObjectHypothesis annotations.

    Specialized CommonAnalysisStructureCheckAnnotationTypeExists that:

    * Specifically checks for ObjectHypothesis annotations
    * Returns SUCCESS if any ObjectHypothesis exists
    * Returns FAILURE if no ObjectHypothesis exists
    """

    def __init__(
        self,
        name: str = "CommonAnalysisStructureCheckOHExists",
        raise_with_str: str = "",
    ) -> None:
        """
        Initialize ObjectHypothesis checker.

        :param name: Name of this node in the behavior tree
        :param raise_with_str: Error message to raise on failure, empty string disables
            raising
        """
        super(CommonAnalysisStructureCheckOHExists, self).__init__(
            name=name, annotation_type=ObjectHypothesis, raise_with_str=raise_with_str
        )
