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


class CommonAnalysisStructureCheckFunction(BaseAnnotator):
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
        name: str = "CommonAnalysisStructureCheckFunction",
        function: Optional[Callable[[CommonAnalysisStructure], bool]] = None,
        failure_message: str = "",
    ) -> None:
        """
        Initialize the common analysis structure condition checker.

        :param name: Name of this node in the behavior tree
        :param function: Function that evaluates common analysis structure conditions, must return bool
        :param failure_message: Error message to raise on failure, empty string disables
            raising
        :raises CommonAnalysisStructureCheckConfigurationError: If function is None
        """
        super(CommonAnalysisStructureCheckFunction, self).__init__(name=name)
        if function is None:
            raise CommonAnalysisStructureCheckConfigurationError(component_name=name)
        self.function = function
        self.failure_message = failure_message

    def update(self) -> Status:
        """
        Check the common analysis structure condition.

        :return: SUCCESS if condition is True, FAILURE if False
        :raises CommonAnalysisStructureCheckFailed: If failure_message is set and condition is False
        """
        common_analysis_structure = self.get_common_analysis_structure()
        if self.function(common_analysis_structure):
            return Status.SUCCESS
        else:
            if self.failure_message != "":
                raise CommonAnalysisStructureCheckFailed(reason=self.failure_message)
            else:
                return Status.FAILURE


def any_of_type_present(
    annotation_type: Type, common_analysis_structure: CommonAnalysisStructure
) -> bool:
    """
    Check if any annotations of a specific type exist in the common analysis structure.

    :param annotation_type: Type of annotation to check for
    :param common_analysis_structure: Common analysis structure instance
    :return: True if at least one annotation exists, False otherwise
    """
    annotations = common_analysis_structure.filter_annotations_by_type(annotation_type)
    return len(annotations) > 0


class CommonAnalysisStructureCheckAnnotationTypeExists(
    CommonAnalysisStructureCheckFunction
):
    """
    Check for existence of specific annotation types.

    Specialized CommonAnalysisStructureCheckFunction that:

    * Checks for presence of specific annotation types
    * Returns SUCCESS if at least one annotation exists
    * Returns FAILURE if no annotations exist
    """

    def __init__(
        self,
        name: str = "CommonAnalysisStructureCheckAnnotationTypeExists",
        annotation_type: Optional[Type] = None,
        failure_message: str = "",
    ) -> None:
        """
        Initialize annotation type checker.

        :param name: Name of this node in the behavior tree
        :param annotation_type: Type of annotation to check for
        :param failure_message: Error message to raise on failure, empty string disables
            raising. This is useful in scenarios where you want to raise an error which
            will be sent via the Query Interface to the caller.
        """
        function = functools.partial(any_of_type_present, annotation_type)
        super(CommonAnalysisStructureCheckAnnotationTypeExists, self).__init__(
            name=name, function=function, failure_message=failure_message
        )


class CommonAnalysisStructureCheckObjectHypothesisExists(
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
        name: str = "CommonAnalysisStructureCheckObjectHypothesisExists",
        failure_message: str = "",
    ) -> None:
        """
        Initialize ObjectHypothesis checker.

        :param name: Name of this node in the behavior tree
        :param failure_message: Error message to raise on failure, empty string disables
            raising
        """
        super(CommonAnalysisStructureCheckObjectHypothesisExists, self).__init__(
            name=name, annotation_type=ObjectHypothesis, failure_message=failure_message
        )
