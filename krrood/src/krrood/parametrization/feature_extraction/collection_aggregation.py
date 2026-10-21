from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from typing_extensions import Any, Callable

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    FlatVariable,
    MappedVariable,
)
from krrood.entity_query_language.factories import and_, entity
from krrood.entity_query_language.operators.aggregators import (
    AggregationBounds,
    Aggregator,
    PartiallyKnownValues,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.operators.core_logical_operators import (
    AND,
    OR,
    Not,
)
from krrood.entity_query_language.query.query import Entity
from krrood.parametrization.feature_extraction.exceptions import (
    AggregationIsNotOverACollection,
)

# %% conditions on one element


class ConditionOutcome(Enum):
    """
    How a condition on one element turns out, as far as the element is stated.
    """

    SATISFIED = auto()
    """
    The stated element satisfies the condition.
    """

    VIOLATED = auto()
    """
    The stated element violates the condition.
    """

    UNDECIDED = auto()
    """
    The element leaves open what the condition depends on.
    """

    def conjoined_with(self, other: ConditionOutcome) -> ConditionOutcome:
        """
        :param other: The outcome of another condition on the same element.
        :return: The outcome of both conditions holding.
        """
        if ConditionOutcome.VIOLATED in (self, other):
            return ConditionOutcome.VIOLATED
        if self is ConditionOutcome.SATISFIED and other is ConditionOutcome.SATISFIED:
            return ConditionOutcome.SATISFIED
        return ConditionOutcome.UNDECIDED

    def disjoined_with(self, other: ConditionOutcome) -> ConditionOutcome:
        """
        :param other: The outcome of another condition on the same element.
        :return: The outcome of either condition holding.
        """
        return self.negated.conjoined_with(other.negated).negated

    @property
    def negated(self) -> ConditionOutcome:
        """
        :return: The outcome of the condition not holding.
        """
        if self is ConditionOutcome.SATISFIED:
            return ConditionOutcome.VIOLATED
        if self is ConditionOutcome.VIOLATED:
            return ConditionOutcome.SATISFIED
        return ConditionOutcome.UNDECIDED


@dataclass
class AttributePath:
    """
    A sequence of attributes followed from a value, compared by the attribute names.
    """

    steps: tuple[Attribute, ...] = field(compare=False)
    """
    The attributes, in the order they are followed.
    """

    attribute_names: tuple[str, ...] = field(init=False)
    """
    The names of the attributes, in the order they are followed.
    """

    def __post_init__(self):
        self.attribute_names = tuple(step._attribute_name_ for step in self.steps)

    @classmethod
    def from_steps(
        cls, steps: list[MappedVariable], query: SymbolicExpression
    ) -> AttributePath:
        """
        :param steps: Steps of an access path.
        :param query: The query the steps are taken from.
        :return: The path through the steps.
        :raises AggregationIsNotOverACollection: If a step is not an attribute.
        """
        if not all(isinstance(step, Attribute) for step in steps):
            raise AggregationIsNotOverACollection(query)
        return cls(tuple(steps))

    def value_stated_by(self, start: Any) -> Any:
        """
        :param start: A value, or a pattern describing one.
        :return: The value the path leads to, or ``...`` when a value on the way is not
            stated.
        """
        current = start
        for step in self.steps:
            if current is ...:
                return ...
            current = next(iter(step._apply_mapping_on_external_value_(current)), ...)
        return current


@dataclass
class ElementCondition(ABC):
    """
    A condition on one element of a collection.
    """

    @abstractmethod
    def outcome_for(self, element: Any) -> ConditionOutcome:
        """
        :param element: The element, a pattern describing it, or ``...`` when it is not
            stated.
        :return: How the condition turns out for the element.
        """

    @staticmethod
    def from_expression(
        expression: SymbolicExpression, element: FlatVariable, query: SymbolicExpression
    ) -> ElementCondition:
        """
        :param expression: A condition of a query on the elements of a collection.
        :param element: The variable standing for one element.
        :param query: The query the condition belongs to.
        :return: The condition as a condition on one element.
        :raises AggregationIsNotOverACollection: If the condition is not a combination
            of comparisons between an attribute of the element and a literal.
        """
        if isinstance(expression, AND):
            return ElementConjunction(
                ElementCondition.from_expression(expression.left, element, query),
                ElementCondition.from_expression(expression.right, element, query),
            )
        if isinstance(expression, OR):
            return ElementDisjunction(
                ElementCondition.from_expression(expression.left, element, query),
                ElementCondition.from_expression(expression.right, element, query),
            )
        if isinstance(expression, Not):
            return ElementNegation(
                ElementCondition.from_expression(expression._child_, element, query)
            )
        if isinstance(expression, Comparator) and expression._is_literal_comparator_():
            return ElementComparison.from_comparator(expression, element, query)
        raise AggregationIsNotOverACollection(query)


@dataclass
class ElementComparison(ElementCondition):
    """
    A comparison between an attribute of the element and a literal.
    """

    attribute_path: AttributePath
    """
    The path from the element to the compared value.
    """

    operation: Callable[[Any, Any], bool]
    """
    The comparison, taking the element's value first.
    """

    literal: Any
    """
    The value the element's value is compared to.
    """

    @classmethod
    def from_comparator(
        cls, comparator: Comparator, element: FlatVariable, query: SymbolicExpression
    ) -> ElementComparison:
        """
        :param comparator: A comparison between a chain through the element and a
            literal.
        :param element: The variable standing for one element.
        :param query: The query the comparison belongs to.
        :return: The comparison on one element.
        :raises AggregationIsNotOverACollection: If the chain does not run through the
            element.
        """
        path = comparator.left._access_path_
        element_indices = [index for index, step in enumerate(path) if step is element]
        if not element_indices:
            raise AggregationIsNotOverACollection(query)
        return cls(
            AttributePath.from_steps(path[element_indices[0] + 1 :], query),
            comparator.operation,
            comparator.right._value_,
        )

    def outcome_for(self, element: Any) -> ConditionOutcome:
        value = self.attribute_path.value_stated_by(element)
        if value is ...:
            return ConditionOutcome.UNDECIDED
        if self.operation(value, self.literal):
            return ConditionOutcome.SATISFIED
        return ConditionOutcome.VIOLATED


@dataclass
class ElementConjunction(ElementCondition):
    """
    Two conditions that both have to hold.
    """

    left: ElementCondition
    """
    The first condition.
    """

    right: ElementCondition
    """
    The second condition.
    """

    def outcome_for(self, element: Any) -> ConditionOutcome:
        return self.left.outcome_for(element).conjoined_with(
            self.right.outcome_for(element)
        )


@dataclass
class ElementDisjunction(ElementCondition):
    """
    Two conditions of which at least one has to hold.
    """

    left: ElementCondition
    """
    The first condition.
    """

    right: ElementCondition
    """
    The second condition.
    """

    def outcome_for(self, element: Any) -> ConditionOutcome:
        return self.left.outcome_for(element).disjoined_with(
            self.right.outcome_for(element)
        )


@dataclass
class ElementNegation(ElementCondition):
    """
    A condition that must not hold.
    """

    condition: ElementCondition
    """
    The negated condition.
    """

    def outcome_for(self, element: Any) -> ConditionOutcome:
        return self.condition.outcome_for(element).negated


# %% an aggregation over the elements of one collection


@dataclass
class CollectionAggregation:
    """
    An aggregation of a value of each element of one collection of a subject, over the
    elements a condition holds for.

    Two collection aggregations are equal when they aggregate the same way along the
    same attributes under the same condition, whichever variable they are written over.
    """

    aggregator: Aggregator = field(compare=False)
    """
    The aggregator applied to the values.
    """

    collection_path: AttributePath
    """
    The path from the subject to the collection.
    """

    value_path: AttributePath
    """
    The path from an element to its aggregated value.
    """

    condition: ElementCondition | None = None
    """
    The condition an element has to satisfy to be aggregated, if any.
    """

    aggregator_type: type[Aggregator] = field(init=False)
    """
    The kind of aggregation.
    """

    def __post_init__(self):
        self.aggregator_type = type(self.aggregator)

    @classmethod
    def from_query(cls, query: Entity | Aggregator) -> CollectionAggregation:
        """
        :param query: An aggregator over the elements of a collection, or a query
            selecting one and filtering the elements.
        :return: The aggregation the query states.
        :raises AggregationIsNotOverACollection: If the query aggregates something
            other than attributes of the elements of one collection, filtered by
            comparisons of their attributes to literals.
        """
        query = query if isinstance(query, Entity) else entity(query)
        aggregator = query.selected_aggregator
        is_grouped = (
            query._grouped_by_builder_ is not None
            and query._grouped_by_builder_.variables_to_group_by
        )
        if (
            aggregator is None
            or is_grouped
            or query._having_builder_ is not None
            or query._ordered_by_builder_ is not None
        ):
            raise AggregationIsNotOverACollection(query)
        aggregated = aggregator._child_
        if not isinstance(aggregated, MappedVariable):
            raise AggregationIsNotOverACollection(query)
        path = aggregated._access_path_
        element_indices = [
            index for index, step in enumerate(path) if isinstance(step, FlatVariable)
        ]
        if len(element_indices) != 1:
            raise AggregationIsNotOverACollection(query)
        [element_index] = element_indices
        conditions = query._where_builder_.conditions if query._where_builder_ else ()
        return cls(
            aggregator,
            AttributePath.from_steps(path[:element_index], query),
            AttributePath.from_steps(path[element_index + 1 :], query),
            (
                ElementCondition.from_expression(
                    and_(*conditions), path[element_index], query
                )
                if conditions
                else None
            ),
        )

    def values_stated_by(self, subject: Any) -> PartiallyKnownValues:
        """
        :param subject: The subject, or a pattern describing it.
        :return: The values the aggregation ranges over, as far as the subject states
            them.
        """
        collection = self.collection_path.value_stated_by(subject)
        if collection is ...:
            return PartiallyKnownValues(is_size_unstated=True)
        values = PartiallyKnownValues()
        for element in collection:
            outcome = (
                self.condition.outcome_for(element)
                if self.condition is not None
                else ConditionOutcome.SATISFIED
            )
            if outcome is ConditionOutcome.VIOLATED:
                continue
            values.add(
                self.value_path.value_stated_by(element),
                is_certainly_included=outcome is ConditionOutcome.SATISFIED,
            )
        return values

    def bounds_stated_by(self, subject: Any) -> AggregationBounds:
        """
        :param subject: The subject, or a pattern describing it.
        :return: The closed range the result of the aggregation lies in, as far as the
            subject states the aggregated values.
        :raises AggregatorHasNoBounds: If the aggregation cannot be bounded.
        """
        return self.aggregator.bounds_of(self.values_stated_by(subject))
