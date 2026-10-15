"""
CAS condition checking utilities.

This module provides annotators for checking conditions in the Common Analysis System (CAS).
The annotators follow these principles:

* Return SUCCESS if condition is true
* Return FAILURE if condition is false
* Never return RUNNING status

.. note::
   For RUNNING status checks, use CASCondition from cas_condition.py instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from py_trees.common import Status
from typing_extensions import Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.exceptions import CASCheckFailed
from robokudo.types.core import Annotation
from robokudo.types.scene import ObjectHypothesis

# %% checks


@dataclass(kw_only=True)
class CASCheck(BaseAnnotator, ABC):
    """
    Succeeds if a condition on the CAS holds and fails otherwise.
    """

    failure_reason: str = ""
    """
    If set, a condition that does not hold raises :class:`CASCheckFailed` with this
    reason instead of failing, so that the query interface reports it to the caller.
    """

    @abstractmethod
    def holds(self, cas: CAS) -> bool:
        """
        :param cas: The CAS to check.
        :return: Whether the condition holds.
        """

    def compute(self) -> Status:
        """
        :return: Success if the condition holds, failure otherwise.
        :raises CASCheckFailed: If it does not hold and :attr:`failure_reason` is set.
        """
        if self.holds(self.get_cas()):
            return Status.SUCCESS
        if self.failure_reason != "":
            raise CASCheckFailed(reason=self.failure_reason)
        return Status.FAILURE


@dataclass(kw_only=True)
class CASCheckFunc(CASCheck):
    """
    Checks the condition a given function evaluates.
    """

    check: Callable[[CAS], bool]
    """
    Evaluates the condition on the CAS.
    """

    def holds(self, cas: CAS) -> bool:
        return self.check(cas)


@dataclass(kw_only=True)
class CASCheckAnnotationTypeExists(CASCheck):
    """
    Checks that the CAS holds at least one annotation of a type.
    """

    annotation_type: type[Annotation]
    """
    The type of annotation that has to be present.
    """

    def holds(self, cas: CAS) -> bool:
        return len(cas.filter_annotations_by_type(self.annotation_type)) > 0


@dataclass(kw_only=True)
class CASCheckOHExists(CASCheckAnnotationTypeExists):
    """
    Checks that the CAS holds at least one object hypothesis.
    """

    annotation_type: type[Annotation] = ObjectHypothesis
