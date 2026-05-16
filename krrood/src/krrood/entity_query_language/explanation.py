from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass, field
from functools import wraps
from types import ModuleType
from typing import Any, List, Optional, Type, Callable, Union
from uuid import UUID

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.operators.core_logical_operators import LogicalOperator

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult, SymbolicExpression,
    )
    from krrood.entity_query_language.query.query import Query


def filter_stack(
        stack: List[inspect.FrameInfo], internal_package: Optional[str] = None
) -> List[inspect.FrameInfo]:
    """
    Filter the stack to remove external libraries and optionally keep only a specific package.

    :param stack: The stack to filter.
    :param internal_package: The name of the package to focus on.
    :return: The filtered stack.
    """
    filtered = []
    for frame in stack:
        path = frame.filename
        # Exclude standard library/external packages
        if "site-packages" in path or "dist-packages" in path:
            continue

        # If a specific package is requested, filter further
        if internal_package and internal_package not in path:
            continue

        filtered.append(frame)
    return filtered


@dataclass
class MonitoredRegistry:
    """
    Registry for monitoring EQL object creation stacks.
    Acts as a class decorator and provides lookup methods.
    """
    _monitored: set[type] = field(default_factory=set)
    """
    Set of classes that are currently being monitored.
    """

    def __call__(self, cls: Type) -> Type:
        """
        Decorate a class to automatically record its creation stack.

        :param cls: The class to monitor.
        :return: The monitored class.
        """
        cls._is_monitored_ = True
        self._monitored.add(cls)

        original_post_init = getattr(cls, "__post_init__", lambda self: None)

        @wraps(original_post_init)
        def new_post_init(self, *args, **kwargs):
            raw_stack = inspect.stack()[1:]
            self._creation_stack = filter_stack(raw_stack)
            original_post_init(self, *args, **kwargs)

        cls.__post_init__ = new_post_init
        return cls

    def get_stack(self, instance: Any) -> Optional[List[inspect.FrameInfo]]:
        """
        Retrieve the creation stack for a monitored instance.

        :param instance: The instance to retrieve the stack for.
        :return: The creation stack, or None if not monitored.
        """
        if not self.is_monitored(type(instance)):
            return None
        return instance._creation_stack

    def is_monitored(self, target: Union[Type, Callable]) -> bool:
        """
        Check whether a class or callable is monitored.

        :param target: The class or callable to check.
        :return: True if monitored, False otherwise.
        """
        # Check the registry set first, then fall back to the marker attribute
        return target in self._monitored or bool(getattr(target, "_is_monitored_", False))

    def unregister(self, cls: Type) -> None:
        """
        Remove a class from monitoring.

        :param cls: The class to stop monitoring.
        """
        self._monitored.discard(cls)
        if hasattr(cls, "_is_monitored_"):
            del cls._is_monitored_

    @property
    def monitored_classes(self) -> tuple[type, ...]:
        """Return an immutable snapshot of all monitored classes."""
        return tuple(self._monitored)


# Singleton instance — use this as the decorator
monitored = MonitoredRegistry()


@dataclass
class ConditionAndBindings:
    """
    Represents a condition and its associated bindings in the inference process.
    """
    condition: SymbolicExpression
    """
    The condition expression.
    """
    bindings: dict[UUID, Any]
    """
    A dictionary mapping UUIDs of condition children to their corresponding bindings.
    """

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.condition} ({','.join(str(child) for child in self.condition._children_)})"


@dataclass
class InferenceExplanation:
    """
    Explanation of how an instance was created through inference.
    """

    instance: Any
    """
    The instance that was created.
    """
    query_node: SymbolicExpression
    """
    The query node that was used to create the instance.
    """
    stack: List[inspect.FrameInfo]
    """
    The stack trace at the point of creation.
    """
    query_root: Optional[Query] = None
    """
    The root of the query that was used to create the instance.
    """
    satisfied_condition_ids: Optional[frozenset] = None
    """
    A frozenset of UUIDs of condition expressions that were satisfied (truth value = True)
    during the evaluation that produced this instance. None if no condition information is available.
    """
    operation_result: Optional[OperationResult] = None
    """
    The full :class:`OperationResult` from the evaluation iteration that produced this instance.
    Contains bindings, all_bindings, is_false, operand, previous_operation_result, and
    satisfied_condition_ids. None if no result information is available.
    """

    def get_satisfied_conditions_as_string(self) -> str:
        """
        Returns a string representation of the satisfied conditions, joined by ' AND '.
        """
        return '\nAND '.join(str(c) for c in self.get_satisfied_conditions_and_their_bindings())

    def get_satisfied_conditions_and_their_bindings(self) -> List[ConditionAndBindings]:
        """
        Retrieve the list of satisfied condition expressions along with their bindings.

        :return: A list of :class:`ConditionAndBindings` objects, each containing a satisfied condition expression and
        its corresponding bindings. Returns an empty list if no satisfaction data is available.
        """
        if self.operation_result is None or not self.operation_result.satisfied_condition_ids:
            return []

        satisfied_conditions = []
        for condition_id in self.operation_result.satisfied_condition_ids:
            condition_expr = self.query_root._get_expression_by_id_(condition_id)
            if isinstance(condition_expr, (LogicalOperator, )):
                continue
            if condition_expr is not None:
                satisfied_conditions.append(ConditionAndBindings(condition_expr, self.operation_result.all_bindings))
        return satisfied_conditions

    def condition_graph(self):
        """
        Build a QueryGraph of the full query tree with satisfaction data overlaid.

        Each ``QueryNode`` carries an ``is_satisfied`` flag grounded directly on
        the satisfied condition IDs.  Unsatisfied condition subtrees are also
        marked as *faded* for visualization purposes.

        :return: A :class:`QueryGraph` instance, or None if no conditions exist
            or no satisfaction data is available.
        """
        if self.query_root is None or not self.satisfied_condition_ids:
            return None
        from krrood.entity_query_language.query_graph import QueryGraph

        return QueryGraph(
            self.query_root,
            satisfied_condition_ids=self.satisfied_condition_ids,
        )

    def as_string(
            self, focus_package: Optional[str | ModuleType] = None
    ) -> str:
        """
        Convert an InferenceExplanation into a human-readable string.

        :param focus_package: Optional package name to filter the stack further.
        :return: A formatted string explaining the inference.
        """
        if isinstance(focus_package, ModuleType):
            focus_package = focus_package.__name__
        # Allow further filtering at explanation time
        display_stack = filter_stack(self.stack, internal_package=focus_package)

        formatted_stack = []
        for frame_info in display_stack:
            formatted_stack.append(
                f'  File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function}\n'
                f'    {frame_info.code_context[0].strip() if frame_info.code_context else "???"}\n'
            )

        stack_str = "".join(formatted_stack[:10])  # Limit to 10 frames

        return (
            f"Instance {self.instance} was created by inference variable: {self.query_node}\n"
            f"Part of query: {self.query_root}\n"
            f"Call stack at definition:\n{stack_str}"
        )


# Dictionary to store inference explanations for instances.
# Uses weak references to allow instances to be garbage collected.
INFERENCE_RECORD: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def register_inference(
        instance: Any, variable_node: SymbolicExpression, result: Optional[OperationResult] = None
) -> None:
    """
    Register an instance created via inference into the internal records.

    :param instance: The instance to record.
    :param variable_node: The variable node that produced the instance.
    :param result: The OperationResult from the evaluation, carrying satisfied condition IDs.
    """
    if not monitored.is_monitored(type(variable_node)):
        return

    satisfied_ids = result.satisfied_condition_ids if result else None
    explanation = InferenceExplanation(
        instance=instance,
        query_node=variable_node,
        stack=monitored.get_stack(variable_node) or [],
        query_root=variable_node._root_,
        satisfied_condition_ids=satisfied_ids,
        operation_result=result,
    )
    try:
        INFERENCE_RECORD[instance] = explanation
    except TypeError:
        pass


def explain_inference(instance: Any) -> Optional[InferenceExplanation]:
    """
    Retrieve the explanation of how the given instance was created through inference.

    :param instance: The instance to explain.
    :return: An InferenceExplanation object if found, otherwise None.
    """
    try:
        return INFERENCE_RECORD.get(instance)
    except TypeError:
        return None
