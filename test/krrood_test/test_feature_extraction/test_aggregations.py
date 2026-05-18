from dataclasses import dataclass
from typing import List

import pytest

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extractor import (
    HasSceneGenerationAggregations,
    FeatureExtractor,
)
from semantic_digital_twin.adapters.adaptive_environment_generation.schema import (
    EGObject,
    EGSize,
    EGPosition,
    EGOrientation,
)
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.example_classes import SceneObject, SceneRoom


@pytest.fixture
def example_scenario():
    obj1 = SceneObject(
        type="table",
    )
    obj2 = SceneObject(
        type="chair",
    )
    return obj1, obj2


def test_single_aggregation(example_scenario):
    obj1, obj2 = example_scenario
    room = SceneRoom([obj1, obj2])
    aggregations = room.get_aggregation_statistics()
    assert aggregations["object_count_features"] == {"table": 1, "chair": 1}
    assert len(aggregations) == 2


def test_multiple_aggregations(example_scenario):
    obj1, obj2 = example_scenario
    room = SceneRoom([obj1, obj2])


def test_feature_extraction_with_aggregation_statistics(example_scenario):
    obj1, obj2 = example_scenario
    room = SceneRoom([obj1, obj2])
    extractor = FeatureExtractor.from_instances([to_dao(room)])
