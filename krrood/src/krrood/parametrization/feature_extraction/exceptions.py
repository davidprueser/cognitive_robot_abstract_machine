from dataclasses import dataclass
from typing_extensions import Type
from krrood.exceptions import DataclassException


@dataclass
class MissingBaseClassForClassWithExchangeableParts(DataclassException, TypeError):
    """
    Exception raised when a class has exchangeable parts (iterables as attributes), but it does not inherit from ``HasExchangeablePartAggregations``.

    This exception is raised only in FeatureExtraction/MachineLearning Scenarios, where aggregations are required to prepare exchangeable parts for ML models.
    """

    clazz: Type

    def __post_init__(self):
        self.message = f"Class {self.clazz} has exchangeable parts but does not inherit from HasExchangeablePartAggregations"
