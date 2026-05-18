from __future__ import annotations

import enum
import inspect
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Type, Any, Dict, Optional

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import MANYTOONE, ONETOMANY
from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.core.mapped_variable import MappedVariable, Apply
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column, is_data_column
from random_events.variable import compatible_types

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


def _to_domain(dao: Any) -> Any:
    return dao.from_dao()


@dataclass
class HasAggregationStatistics(ABC):

    def get_aggregation_statistics(self):
        """
        Get the aggregation statistics for this feature extractor.
        :return: A dictionary containing the aggregation statistics.
        """
        cls_functions = inspect.getmembers(self.__class__, predicate=inspect.isfunction)
        return {
            name: function(self)
            for name, function in cls_functions
            if function is not HasAggregationStatistics.get_aggregation_statistics
            and not (name.startswith("__") and name.endswith("__"))
        }


@dataclass
class HasPartAggregations(ABC):
    """
    Mixin for domain classes that have exchangeable parts (one-to-many relationships) requiring
    aggregation statistics. Subclasses declare which :class:`HasAggregationStatistics` class is
    responsible for each named part.
    """

    @classmethod
    @abstractmethod
    def aggregation_class_for_part(
        cls, part_name: str
    ) -> Optional[Type[HasAggregationStatistics]]:
        """
        :param part_name: The name of an exchangeable part on this class.
        :return: The aggregation statistics class for that part, or ``None`` if the part needs no aggregation.
        """
        ...


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
            if not isinstance(domain_object, HasPartAggregations):
                raise ValueError(
                    f"{domain_class.__name__} has exchangeable part '{exchangeable_part}' "
                    f"but does not implement HasPartAggregations."
                )
            agg_class = domain_class.aggregation_class_for_part(exchangeable_part)
            if agg_class is None:
                raise ValueError(
                    f"No aggregation class registered for part '{exchangeable_part}' "
                    f"in {domain_class.__name__}."
                )

            collection = getattr(domain_object, exchangeable_part)

            # Composition chain:
            # current_symbolic → Apply(_to_domain) → Attribute(part) → Apply(agg_class)
            #   → Attribute(method) → Call() → Index(key)
            to_domain_node = Apply(_child_=current_symbolic, _callable_=_to_domain)
            to_domain_node._type_ = domain_class
            collection_node = getattr(to_domain_node, exchangeable_part)
            agg_node = Apply(_child_=collection_node, _callable_=agg_class)
            agg_node._type_ = agg_class

            agg_instance = agg_class(collection)
            for (
                method_name,
                method_result,
            ) in agg_instance.get_aggregation_statistics().items():
                if not isinstance(method_result, dict):
                    continue
                method_call_node = getattr(agg_node, method_name)()
                for key, val in method_result.items():
                    feature = method_call_node[key]
                    feature._type_ = type(val)
                    result.append(feature)

        return result

    def apply_mapping(self, instance: DataAccessObject) -> List:
        """
        Extracts the mapped values for each feature from the given instance.
        :param instance: The instance to extract features from.
        :return: A list of mapped values.
        """
        return [
            feature.apply_mapping_on_external_root(instance)
            for feature in self.features
        ]

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
