from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.object_type_classifier import (
    ObjectTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


@pytest.fixture
def classifier() -> ObjectTypeClassifier:
    return ObjectTypeClassifier()


@pytest.mark.parametrize(
    "raw_type, expected_object_type",
    [
        ("book2", ObjectType.BOOK),
        ("Book2", ObjectType.BOOK),
        ("BOOKCHAIR6", ObjectType.CHAIR),
        ("bookshelf", ObjectType.SHELF),
        ("pottedplant", ObjectType.PLANT),
        ("floorlamp", ObjectType.LAMP),
        ("printer", ObjectType.PRINTER),
        ("smartphone", ObjectType.PHONE),
        ("showcase", ObjectType.DISPLAYCASE),
        ("dishwasher", ObjectType.DISHWASHER),
        ("chair_1", ObjectType.CHAIR),
    ],
)
def test_classify_maps_raw_type_to_expected_object_type(
    classifier: ObjectTypeClassifier, raw_type: str, expected_object_type: ObjectType
) -> None:
    assert classifier.classify(raw_type) == expected_object_type


def test_classify_falls_back_to_other_for_unrecognized_type(
    classifier: ObjectTypeClassifier,
) -> None:
    assert classifier.classify("xyzzy_totally_unknown_object") == ObjectType.OTHER
