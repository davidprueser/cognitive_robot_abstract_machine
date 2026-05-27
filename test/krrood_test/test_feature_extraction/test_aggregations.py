import pytest

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.feature_extractor import (
    FeatureExtractor,
)
from krrood.entity_query_language.core.mapped_variable import Call
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.example_classes import (
    SceneObject,
    SceneRoom,
    KRROODPosition,
    KRROODOrientation,
    SceneObjectType,
    TestExParts,
)


@pytest.fixture
def example_scenario():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2],
    )
    return room


def test_single_aggregation(example_scenario):
    room = example_scenario
    aggregation_instance = room.get_aggregation_class_by_part_name("objects")
    aggregations = aggregation_instance.symbolic_aggregation_features
    values = aggregation_instance.apply_mapping()
    assert len(aggregations) == 3
    assert values == [1, 1, 2]


def test_feature_extraction_with_aggregation_statistics(example_scenario):
    room = example_scenario
    extractor = FeatureExtractor.from_instances([to_dao(room)])

    agg_features = [f for f in extractor.features if isinstance(f, Call)]
    assert len(agg_features) == 3

    names = {f._name_ for f in agg_features}
    assert any("table" in n for n in names)
    assert any("chair" in n for n in names)

    values = extractor.apply_mapping(to_dao(room))
    assert 1 in values


def test_multiple_exchangeable_parts():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2],
    )
    room2 = SceneRoom(
        position=KRROODPosition(1, 1, 1),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1],
    )
    test_ex_parts = TestExParts(objects=[obj1, obj2], rooms=[room, room2])

    extractor = FeatureExtractor.from_instances([to_dao(test_ex_parts)])
    assert len([f for f in extractor.features if isinstance(f, Call)]) == 4
    assert extractor.apply_mapping(to_dao(test_ex_parts)) == [1, 1, 2, 2]
