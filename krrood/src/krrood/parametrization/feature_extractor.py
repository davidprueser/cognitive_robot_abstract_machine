from __future__ import annotations

import enum
from collections import deque
from dataclasses import field
from typing import Tuple, Type

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import MANYTOONE, ONETOMANY
from typing_extensions import TYPE_CHECKING
import inspect
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
)
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column, is_data_column
from random_events.variable import compatible_types

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


@dataclass
class AggregationStatistic(ABC):
    """
    Base class for aggregation statistics over a list of exchangeable domain objects.

    Subclasses declare one field holding the list of items to aggregate and
    expose one public method per statistic. Each method receives ``self`` and
    returns a scalar value.
    """

    aggregation_object: List[Any]
    """
    The items over which statistics are to be computed.
    """

    def __post_init__(self):
        if not self.aggregation_object:
            raise ValueError("Aggregation object must not be empty")

    @property
    def symbolic_aggregation_features(self) -> List[MappedVariable]:
        """
        Symbolic variables corresponding to each aggregation statistic method.

        Dict-returning methods are expanded into one variable per key.
        :return: One ``MappedVariable`` per scalar statistic, typed as ``float``.
        """
        symbolic_aggregations = []
        aggregation_variable = variable(type(self), [])
        for function in self.aggregation_features:
            function_variable = getattr(aggregation_variable, function.__name__)()
            symbolic_aggregations.append(function_variable)
        return symbolic_aggregations

    @property
    def aggregation_features(self) -> List[Any]:
        """
        All public, non-base statistic methods defined on the concrete subclass.

        Dict-returning methods are flattened into per-key entries and recorded
        in ``_internal_aggregation_mapping`` so symbolic lookup can recover
        the original method name.
        :return: A list of bound or flattened statistic callables.
        """
        cls_functions = inspect.getmembers(self.__class__, predicate=inspect.isfunction)
        aggregations = []
        for name, function in cls_functions:
            if (
                name.startswith("__")
                or name.startswith("_")
                or function is AggregationStatistic.apply_mapping
            ):
                continue
            else:
                aggregations.append(function)

        return aggregations

    def apply_mapping(self) -> List:
        """
        Evaluates every symbolic aggregation feature against this instance.

        :return: One concrete value per entry in ``symbolic_aggregation_features``.
        """
        return [
            feature.apply_mapping_on_external_root(self)
            for feature in self.symbolic_aggregation_features
        ]


class AggregationRegistry:
    """
    Class-level registry mapping ``(owner_class, attribute_name)`` pairs to
    their ``AggregationStatistic`` subclass.

    The registry is write-protected from outside this module: all entries are
    created exclusively through the ``@aggregation_for`` decorator.
    """

    _registry: Dict[Tuple[Type, str], Type[AggregationStatistic]] = {}
    """
    The registry mapping ``(owner_class, attribute_name)`` pairs to their
    ``AggregationStatistic`` subclass.
    """

    @classmethod
    def _register(
        cls,
        owner: Type,
        attribute_name: str,
        aggregation_cls: Type[AggregationStatistic],
    ) -> None:
        """
        Registers an aggregation class for the given owner field.
        :param owner: The domain class that owns the exchangeable-part field.
        :param attribute_name: The field name on ``owner``.
        :param aggregation_cls: The ``AggregationStatistic`` subclass to register for the pair.
        """
        cls._registry[(owner, attribute_name)] = aggregation_cls

    @classmethod
    def get(cls, owner: Type, attribute_name: str) -> Type[AggregationStatistic]:
        """
        Returns the aggregation class registered for the given owner field.

        :param owner: The domain class that owns the exchangeable-part field.
        :param attribute_name: The field name on ``owner``.
        :return: The registered ``AggregationStatistic`` subclass.
        :raises KeyError: If no aggregation class has been registered for the pair.
        """
        key = (owner, attribute_name)
        if key not in cls._registry:
            raise KeyError(
                f"No aggregation class registered for "
                f"{owner.__name__}.{attribute_name}. "
                f"Use @aggregation_for({owner.__name__!r}, {attribute_name!r}) "
                f"to register one."
            )
        return cls._registry[key]

    @classmethod
    def get_fields_for(cls, owner: Type) -> List[str]:
        """
        Returns the names of all fields on ``owner`` that have a registered aggregation class.

        :param owner: The domain class to query.
        :return: Field names registered for ``owner``, in insertion order.
        """
        return [attr for (owner_cls, attr) in cls._registry if owner_cls is owner]


def aggregation_for(*owner_attribute_pairs: Tuple[Type, str]):
    """
    Class decorator that registers an ``AggregationStatistic`` subclass in the
    ``AggregationRegistry`` for one or more ``(owner, attribute_name)`` pairs.

    :param owner_attribute_pairs: One or more ``(owner_class, attribute_name)`` tuples.
    """

    def wrapper(
        aggregation_cls: Type[AggregationStatistic],
    ) -> Type[AggregationStatistic]:
        for owner, attribute_name in owner_attribute_pairs:
            AggregationRegistry._register(owner, attribute_name, aggregation_cls)
        return aggregation_cls

    return wrapper


@dataclass
class HasExchangeablePartAggregations(ABC):
    """
    Mixin for domain classes whose exchangeable-part fields have aggregation
    classes registered via ``@aggregation_for``.

    Subclasses must be dataclasses. Any field whose ``(owner, name)`` pair
    appears in the ``AggregationRegistry`` is validated to be a list at
    instance creation time.
    """

    def __post_init__(self) -> None:
        """
        Validates that every registered exchangeable-part field holds a list.

        :raises TypeError: If a registered field is not a list at instance creation.
        """
        for field_name in AggregationRegistry.get_fields_for(type(self)):
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise TypeError(
                    f"{self.__class__.__name__}.{field_name} must be a list, "
                    f"got {type(value).__name__}."
                )

    def get_aggregation_class_by_part_name(
        self, part_name: str
    ) -> AggregationStatistic:
        """
        Instantiates and returns the aggregation class registered for the named field.

        :param part_name: The name of the exchangeable-part field.
        :return: An ``AggregationStatistic`` initialised with the field's current value.
        :raises KeyError: If no aggregation class is registered for ``part_name``.
        """
        aggregation_cls = AggregationRegistry.get(type(self), part_name)
        return aggregation_cls(getattr(self, part_name))


@dataclass
class FeatureExtractor:
    """
    Extracts symbolic features from DAO instances, including scalar attributes,
    unique-part sub-trees, and aggregation statistics over exchangeable parts.

    Prefer ``FeatureExtractor.from_instances`` for construction; the direct
    constructor is for cases where the feature list is already known.
    """

    features: List[MappedVariable]
    """
    Symbolic variables representing every extractable feature, in traversal order.
    """

    aggregations: Dict[MappedVariable, str] = field(default_factory=dict, init=False)
    """
    Maps each aggregation feature variable to the exchangeable-part field name it came from.
    """

    def __post_init__(self):
        if not self.features:
            raise ValueError(
                "No features provided. If list of instances available, use `FeatureExtractor.from_instances` for instantiation."
            )

    @classmethod
    def from_instances(cls, instances: List[DataAccessObject]) -> FeatureExtractor:
        """
        Create a new feature extractor from the given instances.
        :param instances: The instances to create the feature extractor from.
        :return: A new feature extractor.
        """
        if not instances:
            raise ValueError("No instances provided")

        dao_state = FromDataAccessObjectState()
        root = variable(type(instances[0].from_dao(dao_state)), [])
        extractor = cls.__new__(cls)
        extractor.aggregations = {}
        extractor.features = extractor._extract_features(instances[0], root)
        return extractor

    def _extract_features(
        self, example_instance: DataAccessObject, symbolic_root: Variable
    ) -> List[MappedVariable]:
        """
        Traverses the DAO object graph breadth-first and collects all features.

        :param example_instance: A representative DAO instance that defines the schema.
        :param symbolic_root: The root symbolic variable for the traversal.
        :return: All discovered feature variables in traversal order.
        """
        result = []
        seen = set()
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen.add(id(current_instance))

            specification = EntityCompositionDescriptor(type(current_instance))

            result.extend(
                self._process_attributes(
                    current_instance, current_symbolic, specification
                )
            )
            result.extend(
                self._process_exchangeable_parts(
                    current_instance, current_symbolic, specification
                )
            )

            queue.extend(
                self._process_unique_parts(
                    current_instance, current_symbolic, specification
                )
            )

        return result

    @staticmethod
    def _process_attributes(
        instance: DataAccessObject,
        symbolic_root: Variable,
        specification: EntityCompositionDescriptor,
    ) -> List[MappedVariable]:
        """
        Collects symbolic variables for all scalar data columns of ``instance``.

        Columns whose value is not a compatible primitive type are skipped.
        :param instance: The DAO instance to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param specification: The RSPN specification describing the instance's schema.
        :return: One typed ``MappedVariable`` per compatible scalar attribute.
        """
        result = []
        for attribute in specification.attributes:
            value = getattr(instance, attribute.key)

            if not isinstance(value, compatible_types):
                continue

            symbolic_attribute = getattr(symbolic_root, attribute.name)
            symbolic_attribute._type_ = get_python_type_from_sqlalchemy_column(
                attribute
            )
            result.append(symbolic_attribute)
        return result

    @staticmethod
    def _process_unique_parts(
        instance: DataAccessObject,
        symbolic_root: Variable,
        specification: EntityCompositionDescriptor,
    ) -> deque[Any]:
        """
        Enqueues non-null unique-part (many-to-one) relations for further traversal.

        :param instance: The DAO instance to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param specification: The RSPN specification describing the instance's schema.
        :return: ``(child_instance, child_symbolic)`` pairs ready for BFS expansion.
        """
        queue = deque()
        for part in specification.unique_parts:
            value = getattr(instance, part)

            if value is None:
                continue

            queue.append((value, getattr(symbolic_root, part)))
        return queue

    def _process_exchangeable_parts(
        self, current_instance, current_symbolic, specification
    ):
        """
        Collects aggregation statistic variables for all one-to-many relations of ``current_instance``.

        Also records each aggregation variable in ``self.aggregations`` so
        ``apply_mapping`` can route it to the correct ``AggregationStatistic``
        at evaluation time.
        :param current_instance: The DAO instance whose exchangeable parts are processed.
        :param current_symbolic: Unused; kept for signature consistency with other process methods.
        :param specification: The RSPN specification describing the instance's schema.
        :return: Symbolic aggregation feature variables for all exchangeable parts.
        :raises ValueError: If the domain class does not implement ``HasExchangeablePartAggregations``.
        """
        result = []
        domain_object = current_instance.from_dao()
        domain_class = type(domain_object)

        for exchangeable_part in specification.exchangeable_parts:
            if not isinstance(domain_object, HasExchangeablePartAggregations):
                raise ValueError(
                    f"{domain_class.__name__} has exchangeable part '{exchangeable_part}' "
                    f"but does not implement HasPartAggregations."
                )
            aggregation_instance = domain_object.get_aggregation_class_by_part_name(
                exchangeable_part
            )
            for aggregation in aggregation_instance.symbolic_aggregation_features:
                self.aggregations[aggregation] = exchangeable_part
                result.append(aggregation)

        return result

    def apply_mapping(self, instance: DataAccessObject) -> List:
        """
        Extracts the mapped values for each feature from the given instance.
        :param instance: The instance to extract features from.
        :return: A list of mapped values.
        """
        result = []
        for feature in self.features:
            if feature in self.aggregations:
                part_name = self.aggregations[feature]
                domain_object = instance.from_dao()
                aggregation_instance = domain_object.get_aggregation_class_by_part_name(
                    part_name
                )
                result.append(
                    feature.apply_mapping_on_external_root(aggregation_instance)
                )
            else:
                result.append(feature.apply_mapping_on_external_root(instance))
        return result

    def create_dataframe(self, instances: List[DataAccessObject]) -> pd.DataFrame:
        """
        Create a dataframe from the given instances.
        :param instances: The instances to create the dataframe from.
        :return: A dataframe containing the mapped values for each feature.
        """
        result = []
        for instance in instances:
            result.append(self.apply_mapping(instance))
        features_names = [feature._name_ for feature in self.features]
        return pd.DataFrame(columns=features_names, data=result)

    def create_dataframe_for_exchangeable_parts_with_aggregations(
        self,
        instances: List[DataAccessObject],
        aggregations: List[MappedVariable],
        agg_values: List[List] = None,
    ) -> pd.DataFrame:
        """
        Create a dataframe from the given instances.
        :param instances: The child instances (exchangeable part objects).
        :param aggregations: Aggregation features whose names form the aggregation columns.
        :param agg_values: Pre-computed aggregation values per instance, one list per row.
        :return: A dataframe with one row per child object and columns for its attributes and the aggregation statistics.
        """
        if not instances:
            return pd.DataFrame()

        specification = EntityCompositionDescriptor(type(instances[0]))
        instance_attr_names = [col.key for col in specification.attributes]
        agg_names = [aggregation._name_ for aggregation in aggregations]

        if agg_values is None:
            agg_values = [[] for _ in instances]

        result = [
            agg_row + [getattr(instance, name) for name in instance_attr_names]
            for instance, agg_row in zip(instances, agg_values)
        ]
        return pd.DataFrame(columns=agg_names + instance_attr_names, data=result)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe for JointProbabilityTrees by converting boolean columns to integers and enum columns to hashes.
        :param df: The dataframe to preprocess.
        :return: The dataframe in a JPT compatible format.
        """
        feature_map = dict(zip(df.columns, self.features))
        for column in df.columns:
            feature = feature_map[column]
            if feature._type_ is bool:
                df[column] = df[column].astype(int)
            elif isinstance(feature._type_, enum.EnumType):
                df[column] = df[column].apply(lambda x: hash(x))
            elif feature._type_ not in compatible_types and feature._type_ is not None:
                raise TypeError(
                    f"Unsupported type {feature._type_} for column {column}"
                )
        return df


@dataclass
class EntityCompositionDescriptor:
    """
    Describes the composition of a domain class in terms of its scalar attributes, unique-part relations, and exchangeable-part relations.
    It is constructed from a DAO class' SQLAlchemy mapper.
    """

    dao_class: Type[DataAccessObject] = field(init=True)
    """
    The DAO class whose SQLAlchemy mapper is inspected.
    """

    def __post_init__(self):
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []

        mapper = sqlalchemy.inspection.inspect(self.dao_class)

        for relationship in mapper.relationships:
            if relationship.direction == MANYTOONE:
                self.unique_parts.append(relationship.key)
            # not many to many since we have the association table
            elif relationship.direction == ONETOMANY:
                self.exchangeable_parts.append(relationship.key)
        for column in mapper.columns:
            if is_data_column(column) and column not in mapper.relationships:
                self.attributes.append(column)
