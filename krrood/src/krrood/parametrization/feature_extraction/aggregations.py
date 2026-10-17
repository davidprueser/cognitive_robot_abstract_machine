from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import (
    Callable,
    Optional,
    Type,
    Any,
    ClassVar,
    TypeVar,
    get_args,
)

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable as SymbolicVariable
from krrood.entity_query_language.operators.aggregators import (
    AggregationBounds,
    Aggregator,
)
from krrood.entity_query_language.query.query import Entity
from krrood.parametrization.feature_extraction.collection_aggregation import (
    CollectionAggregation,
)
from krrood.parametrization.feature_extraction.exceptions import (
    InvalidAggregatedFieldError,
    MissingFieldNameError,
    StatisticValueTypeNotDeclared,
)
from krrood.patterns.role import Role
from krrood.entity_query_language.factories import variable
from krrood.symbol_graph.helpers import get_method_return_type
from krrood.utils import recursive_subclasses, get_generic_type_parameters

TAggregatedObject = TypeVar("TAggregatedObject", bound=object)
"""
The type of the domain object an :class:`AggregationStatistic` is a role of.
"""


@dataclass
class _AggregationStatisticDescriptor:
    """
    Descriptor returned by :func:`aggregation_statistic`.

    When Python constructs the enclosing class, it calls :meth:`__set_name__` with the
    owner class and the attribute name. At that point the function is registered and the
    descriptor replaces itself with the bare function so normal method-call semantics
    are preserved.
    """

    field_name: str
    """
    The exchangeable-part field this statistic aggregates over.
    """

    func: Callable
    """
    The wrapped statistic method.
    """

    def __set_name__(self, owner: Type[AggregationStatistic], name: str) -> None:
        if "aggregation_registry" not in owner.__dict__:
            owner.aggregation_registry = defaultdict(list)
        owner.aggregation_registry[self.field_name].append(self.func)
        setattr(owner, name, self.func)


def aggregation_statistic(
    aggregated_field: Attribute,
) -> Callable[[Callable], _AggregationStatisticDescriptor]:
    """
    Marks a method as an aggregation statistic for an exchangeable-part field.

    The containing class is discovered automatically via
    ``__set_name__`` — there is no need to pass it explicitly.

    The method returns the query computing the statistic, written over the role
    taker, and is annotated with the type of the statistic's value, e.g.
    ``-> Query[int]``.

    Example: ``@aggregation_statistic(variable(Shelf).layers)`` marks a statistic over
    the ``layers`` of a ``Shelf``, such as one returning
    ``entity(count(flat_variable(self.layers)))``.

    :param aggregated_field: A symbolic attribute of a variable, naming the field this
        statistic aggregates for.
    :raises InvalidAggregatedFieldError: If ``aggregated_field`` is not a direct
        attribute of a variable.
    """
    if not (
        isinstance(aggregated_field, Attribute)
        and isinstance(aggregated_field._child_, SymbolicVariable)
    ):
        raise InvalidAggregatedFieldError(aggregated_field=aggregated_field)

    def decorator(func: Callable) -> _AggregationStatisticDescriptor:
        return _AggregationStatisticDescriptor(
            field_name=aggregated_field._attribute_name_, func=func
        )

    return decorator


@lru_cache(maxsize=None)
def get_aggregation_class(owner: Type) -> Optional[Type[AggregationStatistic]]:
    """
    Returns the most specific :class:`AggregationStatistic` subclass for ``owner``.

    Walks the MRO of ``owner`` from most specific to least specific, returning the first
    subclass of :class:`AggregationStatistic` whose role taker type matches that ancestor.
    This means that if ``B`` extends ``A`` and only ``AggregationStatistic[A]`` exists,
    a lookup for ``B`` will return it.

    :param owner: The domain class to look up.
    :return: The most specific matching subclass, or ``None`` if none has been defined.
    """
    subclasses = list(recursive_subclasses(AggregationStatistic))
    for ancestor in owner.__mro__:
        for subclass in subclasses:
            bound = get_generic_type_parameters(subclass, AggregationStatistic)
            if bound and bound[0] == ancestor:
                return subclass
    return None


@dataclass(eq=False)
class AggregationStatistic(Role[TAggregatedObject]):
    """
    Base class for aggregation statistics over a domain object's exchangeable-
    part fields.

    An aggregation statistic is a :class:`~krrood.patterns.role.Role` of the domain
    object it describes: it adds statistics to the object without changing its type or
    identity, and the object's own attributes are readable through it.

    Subclasses bind ``TAggregatedObject`` to the type of the role taker and declare one
    or more methods, each annotated with :func:`aggregation_statistic` and returning the
    query that computes the statistic over the role taker.  Discovery
    happens automatically via :func:`get_aggregation_class` — no explicit registration
    is required.

    Set :attr:`field_name` to scope :attr:`aggregation_features` and related methods to
    a single exchangeable-part field.

    .. note::
        Each owner class may have at most one ``AggregationStatistic`` subclass, which must
        handle all of its exchangeable-part fields.  Shared logic across owner types should
        be extracted to an intermediate abstract subclass whose concrete children each bind
        their own role taker type.  The intermediate class declares its own bounded
        TypeVar ``S = TypeVar("S", bound=Base)`` and inherits from
        ``AggregationStatistic[S]``; when all owner types share a common supertype
        ``Base``, this lets it access ``self.role_taker`` with the full interface of
        ``Base`` without losing type safety in the concrete subclasses.
    """

    field_name: Optional[str] = None
    """
    The exchangeable-part field this instance is scoped to.

    Must be set before accessing :attr:`aggregation_features`.
    """

    aggregation_registry: ClassVar[dict[str, list[Callable]]] = defaultdict(list)
    """
    Methods marked with :func:`aggregation_statistic` that are defined directly on this
    class.

    Keys are field names; values are the registered callables.  Each subclass receives
    its own registry via :meth:`__init_subclass__` so registrations never bleed across
    unrelated branches of the hierarchy.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "aggregation_registry" not in cls.__dict__:
            cls.aggregation_registry = defaultdict(list)

    @classmethod
    def _registered_names_for_field(cls, field_name: str) -> set[str]:
        """
        Collect the names of all statistics registered for ``field_name`` across the MRO
        so inherited registrations are included.

        :param field_name: The exchangeable-part field to look up.
        :return: Names of all callables registered under ``field_name``.
        """
        names: set[str] = set()
        for ancestor in cls.__mro__:
            if not issubclass(ancestor, AggregationStatistic):
                continue
            for func in ancestor.aggregation_registry[field_name]:
                names.add(func.__name__)
        return names

    @classmethod
    def aggregation_features_of_field(cls, field_name: str) -> list[Callable]:
        """
        All methods on this class marked with :func:`aggregation_statistic` for
        ``field_name``.

        :param field_name: The exchangeable-part field to look up.
        :return: The marked callable methods for the given field, sorted alphabetically
            by name.
        """
        registered = cls._registered_names_for_field(field_name)
        return [
            func
            for _, func in inspect.getmembers(cls, predicate=inspect.isfunction)
            if func.__name__ in registered
        ]

    @classmethod
    def symbolic_features_of_field(cls, field_name: str) -> list[MappedVariable]:
        """
        Symbolic variables for the statistic methods that aggregate ``field_name``.

        Requires no domain object, so the features of an already fitted model can be
        recreated from the class alone.

        :param field_name: The exchangeable-part field to look up.
        :return: One :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
            per matching statistic method, in alphabetical order.
        """
        return [
            cls.symbolic_feature_of(statistic)
            for statistic in cls.aggregation_features_of_field(field_name)
        ]

    @classmethod
    def symbolic_feature_of(cls, statistic: Callable) -> MappedVariable:
        """
        :param statistic: One of this class's statistics.
        :return: The symbolic variable standing for the statistic's value, named the
            way the variable of a fitted model is.
        """
        feature = getattr(variable(cls, []), statistic.__name__)()
        feature._type_ = cls.value_type_of(statistic.__name__)
        return feature

    @classmethod
    def value_type_of(cls, statistic_name: str) -> type:
        """
        :param statistic_name: The name of one of this class's statistics.
        :return: The type of the statistic's value.
        :raises StatisticValueTypeNotDeclared: If the statistic is not annotated to
            return a query of one value type.
        """
        return_type = get_method_return_type(cls, statistic_name)
        value_types = get_args(return_type)
        if len(value_types) != 1:
            raise StatisticValueTypeNotDeclared(cls, statistic_name)
        return value_types[0]

    @classmethod
    def statistics(cls) -> list[Callable]:
        """
        :return: The methods on this class marked with :func:`aggregation_statistic`,
            for every field.
        """
        field_names = {
            field_name
            for ancestor in cls.__mro__
            if issubclass(ancestor, AggregationStatistic)
            for field_name in ancestor.aggregation_registry
        }
        return [
            statistic
            for field_name in sorted(field_names)
            for statistic in cls.aggregation_features_of_field(field_name)
        ]

    @classmethod
    @lru_cache
    def collection_aggregation_of(cls, statistic: Callable) -> CollectionAggregation:
        """
        :param statistic: One of this class's statistics.
        :return: The aggregation the statistic computes.
        :raises AggregationIsNotOverACollection: If the statistic's query aggregates
            something other than the elements of one collection.
        """
        symbolic_role = cls(role_taker=variable(cls.get_role_taker_type(), []))
        return CollectionAggregation.from_query(statistic(symbolic_role))

    @classmethod
    def statistic_defining(cls, aggregation: Entity | Aggregator) -> Callable | None:
        """
        :param aggregation: An aggregation over the elements of a collection of the
            role taker, or a query selecting one.
        :return: The statistic computing the same aggregation, or ``None`` if none does.
        :raises AggregationIsNotOverACollection: If the aggregation is not over the
            elements of one collection.
        """
        collection_aggregation = CollectionAggregation.from_query(aggregation)
        for statistic in cls.statistics():
            if cls.collection_aggregation_of(statistic) == collection_aggregation:
                return statistic
        return None

    @classmethod
    def bounds_stated_by(
        cls, subject: Any, field_name: str
    ) -> dict[str, AggregationBounds]:
        """
        :param subject: The role taker, or a pattern describing it.
        :param field_name: The exchangeable-part field whose statistics are bounded.
        :return: The closed range each statistic of the field lies in, as far as the
            subject states the aggregated values, by the name of the statistic's
            symbolic feature.
        """
        return {
            feature._name_: cls.collection_aggregation_of(statistic).bounds_stated_by(
                subject
            )
            for statistic, feature in zip(
                cls.aggregation_features_of_field(field_name),
                cls.symbolic_features_of_field(field_name),
            )
        }

    @classmethod
    def statistic_of(cls, feature: MappedVariable) -> Callable:
        """
        :param feature: The symbolic feature of one of this class's statistics.
        :return: The statistic the feature stands for.
        """
        return cls._statistics_by_feature_name_()[feature._name_]

    @classmethod
    @lru_cache
    def _statistics_by_feature_name_(cls) -> dict[str, Callable]:
        """
        :return: This class's statistics, by the name of their symbolic feature.
        """
        return {
            cls.symbolic_feature_of(statistic)._name_: statistic
            for statistic in cls.statistics()
        }

    def value_of(self, statistic: Callable) -> Any:
        """
        :param statistic: One of this class's statistics.
        :return: The value the statistic takes for the role taker.
        """
        [value] = statistic(self).tolist()
        return value

    @property
    def aggregation_features(self) -> list[Callable]:
        """
        All methods on this class marked with :func:`aggregation_statistic` for
        :attr:`field_name`.

        :return: The marked callable methods for the scoped field, sorted alphabetically
            by name.
        :raises MissingFieldNameError: If :attr:`field_name` was not provided.
        """
        if self.field_name is None:
            raise MissingFieldNameError()
        return type(self).aggregation_features_of_field(self.field_name)

    def symbolic_aggregation_features(self) -> list[MappedVariable]:
        """
        Symbolic variables for statistic methods that aggregate
        :attr:`field_name`.

        :return: One :class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable`
            per matching statistic method, in alphabetical order.
        :raises MissingFieldNameError: If :attr:`field_name` was not provided.
        """
        if self.field_name is None:
            raise MissingFieldNameError()
        return type(self).symbolic_features_of_field(self.field_name)

    def apply_mapping(self) -> dict[str, Any]:
        """
        Evaluates every statistic for :attr:`field_name` against this instance.

        :return: A mapping from each statistic method name to its computed value.
        """
        return {
            statistic.__name__: self.value_of(statistic)
            for statistic in self.aggregation_features
        }
