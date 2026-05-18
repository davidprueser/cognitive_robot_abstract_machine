from dataclasses import dataclass
from typing import List

import pytest

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extractor import FeatureExtractor
from krrood.entity_query_language.core.mapped_variable import Index
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.example_classes import SceneObject, SceneObjectAggregations, SceneRoom


@pytest.fixture
def example_scenario():
    obj1 = SceneObject(type="table")
    obj2 = SceneObject(type="chair")
    return obj1, obj2


def test_single_aggregation(example_scenario):
    obj1, obj2 = example_scenario
    agg = SceneObjectAggregations([obj1, obj2])
    aggregations = agg.get_aggregation_statistics()
    assert aggregations["object_count_features"] == {"table": 1, "chair": 1}
    assert len(aggregations) == 1


def test_multiple_aggregations(example_scenario):
    obj1, obj2 = example_scenario
    agg = SceneObjectAggregations([obj1, obj2, SceneObject(type="table")])
    result = agg.object_count_features()
    assert result["table"] == 2
    assert result["chair"] == 1


def test_feature_extraction_with_aggregation_statistics(example_scenario):
    obj1, obj2 = example_scenario
    room = SceneRoom([obj1, obj2])
    extractor = FeatureExtractor.from_instances([to_dao(room)])

    agg_features = [f for f in extractor.features if isinstance(f, Index)]
    assert len(agg_features) == 2

    names = {f._name_ for f in agg_features}
    assert any("table" in n for n in names)
    assert any("chair" in n for n in names)

    values = extractor.apply_mapping(to_dao(room))
    assert 1 in values
