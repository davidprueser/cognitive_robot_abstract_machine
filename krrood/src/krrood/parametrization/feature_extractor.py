from __future__ import annotations

import enum
import inspect
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import List, Type, Any, Dict, Optional

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import MANYTOONE, ONETOMANY
from typing_extensions import TYPE_CHECKING
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import (
    Annotated,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from krrood.entity_query_language.core.mapped_variable import MappedVariable, Apply
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.utils import is_iterable
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column, is_data_column
from random_events.variable import compatible_types

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


@dataclass(frozen=True)
class AggregatedBy:
    """
    Field-level annotation that declares which AggregationStatistic subclass
    is responsible for a list field.

    Usage::
        objects: Annotated[list[SceneObject], AggregatedBy(SceneObjectAggregations)]
    """

    aggregation_class: type


@dataclass
class AggregationStatistic(ABC):
    """
    Base for all aggregation classes. Subclasses are dataclasses whose only
    field is the list of items to aggregate, and whose methods (excluding
    dunder and this base method) are the statistics.
    """

    aggregation_object: List[Any]

    _internal_aggregation_mapping: Dict[Any, Any] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        if not self.aggregation_object:
            raise ValueError("Aggregation object must not be empty")
        self.variable = variable(type(self), [])

    @property
    def symbolic_aggregation_features(self) -> List[MappedVariable]:
        result = []
        for function in self.aggregation_features:
            if function in self._internal_aggregation_mapping:
                original_function_name = self._internal_aggregation_mapping[function]
                function_variable = getattr(self.variable, original_function_name)()
                function_variable._type = float
                result.append(function_variable)
            else:
                function_variable = getattr(self.variable, function.__name__)()
                function_variable._type = float
                result.append(function_variable)
        return result

    @property
    def aggregation_features(self) -> List[Any]:
        cls_functions = inspect.getmembers(self.__class__, predicate=inspect.isfunction)
        aggregations = []
        for name, function in cls_functions:
            if name.startswith("__") and name.endswith("__"):
                continue
            if name.startswith("_"):
                continue
            if function is AggregationStatistic.aggregation_features:
                continue
            if function is AggregationStatistic.apply_mapping:
                continue
            called_function = function(self)
            if isinstance(called_function, dict):
                aggregations.extend(
                    self._process_nested_aggregation_statistics(
                        called_function, function
                    )
                )
            else:
                aggregations.append(function)

        return aggregations

    def _process_nested_aggregation_statistics(
        self, aggregation_statistics, original_function
    ) -> Dict[str, Any]:
        result = {}
        for key, val in aggregation_statistics.items():
            if isinstance(key, Enum):
                key = key.value
            function_name = f"{original_function.__name__}_{key}"
            self._internal_aggregation_mapping[function_name] = (
                original_function.__name__
            )
            result[function_name] = val

        return result

    def apply_mapping(self):
        return [
            feature.apply_mapping_on_external_root(self)
            for feature in self.symbolic_aggregation_features
        ]


@dataclass
class HasExchangeablePartAggregations(ABC):
    """
    Mixin for domain classes that use ``Annotated[list[X], AggregatedBy(Y)]``
    to declare aggregation intent on their fields.

    Validation runs once at class definition time (``__init_subclass__``) and
    again at instance creation time (``__post_init__``).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._validate_aggregation_annotations()

    @classmethod
    def _validate_aggregation_annotations(cls) -> None:
        """
        Called at class definition time. Checks:
        - AggregatedBy targets are proper AggregationStatistic subclasses
        - The annotated field is actually a list type
        """
        hints = get_type_hints(cls, include_extras=True)
        for field_name, hint in hints.items():
            aggregation_marker = _extract_aggregated_by(hint)
            if aggregation_marker is None:
                continue

            # Must target an AggregationStatistic subclass
            if not (
                isinstance(aggregation_marker.aggregation_class, type)
                and issubclass(
                    aggregation_marker.aggregation_class, AggregationStatistic
                )
            ):
                raise TypeError(
                    f"{cls.__name__}.{field_name}: AggregatedBy target "
                    f"{aggregation_marker.aggregation_class} must be a subclass of AggregationStatistic."
                )

            # The field itself must be a list
            inner = get_args(hint)[0]  # unwrap Annotated -> actual type
            if get_origin(inner) not in (list, List):
                raise TypeError(
                    f"{cls.__name__}.{field_name}: AggregatedBy requires a list field, "
                    f"got {inner!r}."
                )

    def __post_init__(self) -> None:
        """
        Called at instance creation time. Checks that each AggregatedBy field
        is actually a list (not None, not a scalar).
        """
        hints = get_type_hints(self.__class__, include_extras=True)
        for field_name, hint in hints.items():
            if _extract_aggregated_by(hint) is None:
                continue
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise TypeError(
                    f"{self.__class__.__name__}.{field_name} must be a list, "
                    f"got {type(value).__name__}."
                )

    def get_aggregation_class_by_part_name(
        self, part_name: str
    ) -> Optional[AggregationStatistic]:
        """
        Returns an instantiated AggregationStatistic for the named field,
        or raises KeyError if the field has no AggregatedBy annotation.
        :param part_name: The name of the field to get the aggregation class for.
        :return: An instantiated AggregationStatistic.
        """
        mapping = self._build_part_mapping()
        if part_name not in mapping:
            raise KeyError(
                f"No AggregatedBy annotation found for field '{part_name}' "
                f"on {self.__class__.__name__}. "
                f"Annotated fields: {list(mapping.keys())}"
            )
        return mapping[part_name]

    def _build_part_mapping(self) -> Dict[str, AggregationStatistic]:
        hints = get_type_hints(self.__class__, include_extras=True)
        result = {}
        for field_name, hint in hints.items():
            marker = _extract_aggregated_by(hint)
            if marker is None:
                continue
            items = getattr(self, field_name)
            result[field_name] = marker.aggregation_class(items)
        return result


def _extract_aggregated_by(hint: Any) -> Optional[AggregatedBy]:
    """Returns the AggregatedBy marker from an Annotated hint, or None."""
    if get_origin(hint) is not Annotated:
        return None
    for meta in get_args(hint)[1:]:
        if isinstance(meta, AggregatedBy):
            return meta
    return None


@dataclass
class FeatureExtractor:
    """
    A class to extract features from a given class. Features are all attributes of the class, propagating custom types/objects down. The features are represented as symbolic variables.
    A feature extractor provides additional knowledge about the class.
    """

    features: List[MappedVariable]
    """
    The features extracted from the class/instances.
    """

    aggregations: Dict[MappedVariable, str] = field(default_factory=dict, init=False)

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
        result = []
        seen = set()
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen.add(id(current_instance))

            specification = RelationalSumProductNetworkSpecification(
                type(current_instance)
            )

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
        specification: RelationalSumProductNetworkSpecification,
    ) -> List[MappedVariable]:
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
        specification: RelationalSumProductNetworkSpecification,
    ) -> deque[Any]:
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

        specification = RelationalSumProductNetworkSpecification(type(instances[0]))
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
class RelationalSumProductNetworkSpecification:
    """
    Specification used to learn a RelationalSumProductNetwork from a class.
    It contains information about the attributes, unique parts, exchangeable parts, and relations of the class.
    These are determined by the relationships and columns of the DAO class.
    """

    spec: Type[DataAccessObject] = field(init=True)
    """
    The wrapped class that is supposed to be an RSPN.
    """

    def __post_init__(self):
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(self.spec)

        for relationship in mapper.relationships:
            if relationship.direction == MANYTOONE:
                self.unique_parts.append(relationship.key)
            # not many to many since we have the association table
            elif relationship.direction == ONETOMANY:
                self.exchangeable_parts.append(relationship.key)
        for column in mapper.columns:
            if is_data_column(column) and column not in mapper.relationships:
                self.attributes.append(column)
