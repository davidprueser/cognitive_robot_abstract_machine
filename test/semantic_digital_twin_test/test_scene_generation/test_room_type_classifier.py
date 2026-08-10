from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.room_type_classifier import (
    RoomTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import RoomType


@pytest.fixture
def classifier() -> RoomTypeClassifier:
    return RoomTypeClassifier()


@pytest.mark.parametrize(
    "raw_type, expected_room_type",
    [
        ("kitchen", RoomType.KITCHEN),
        ("residential_kitchen", RoomType.KITCHEN),
        ("commercial kitchen", RoomType.KITCHEN),
        ("master bedroom", RoomType.BEDROOM),
        ("children's room", RoomType.BEDROOM),
        ("nursery", RoomType.NURSERY),
        ("living room", RoomType.LIVING_ROOM),
        ("living_room", RoomType.LIVING_ROOM),
        ("dining room", RoomType.DINING_ROOM),
        ("bathroom", RoomType.BATHROOM),
        ("office", RoomType.OFFICE),
        ("Office", RoomType.OFFICE),
        ("corridor", RoomType.CORRIDOR),
        ("hallway", RoomType.CORRIDOR),
        ("grocery store", RoomType.GROCERY_STORE),
        ("grocery_store", RoomType.GROCERY_STORE),
        ("clothing store", RoomType.CLOTHING_STORE),
        ("bookstore", RoomType.STORE),
        ("bakery", RoomType.BAKERY),
        ("gym_fitness_center", RoomType.GYM),
        ("hospital patient room", RoomType.PATIENT_ROOM),
        ("operating room", RoomType.OPERATING_ROOM),
        ("prison cell", RoomType.PRISON_CELL),
        ("wine cellar", RoomType.WINE_CELLAR),
        ("waiting room", RoomType.LOBBY),
        ("art studio", RoomType.STUDIO),
        ("TV studio", RoomType.STUDIO),
    ],
)
def test_classify_maps_raw_type_to_expected_room_type(
    classifier: RoomTypeClassifier, raw_type: str, expected_room_type: RoomType
) -> None:
    assert classifier.classify(raw_type) == expected_room_type


@pytest.mark.parametrize(
    "raw_type, expected_room_type",
    [
        # "restaurant_dining_area" must not fall through to DINING_ROOM.
        ("restaurant_dining_area", RoomType.RESTAURANT),
        ("buffet dining room", RoomType.RESTAURANT),
        # A room named after the building it sits in keeps the building.
        ("warehouse office", RoomType.WAREHOUSE),
        ("greenhouse living room", RoomType.GREENHOUSE),
        # Medical rooms are more specific than the generic office/room they contain.
        ("dentist_office", RoomType.EXAMINATION_ROOM),
        ("hospital examination room", RoomType.EXAMINATION_ROOM),
        # "cellar" must not be swallowed by the prison keyword, and vice versa.
        ("prison cell bedroom", RoomType.PRISON_CELL),
        # "Baroque warehouse" contains "bar" but is a warehouse.
        ("Baroque warehouse", RoomType.WAREHOUSE),
        # "shoe store" is apparel retail, not the generic store category.
        ("shoe store", RoomType.CLOTHING_STORE),
        # "workshop" contains "shop" and must not be classified as retail.
        ("workshop", RoomType.WORKSHOP),
        ("craft workshop", RoomType.WORKSHOP),
    ],
)
def test_classify_resolves_ambiguous_compound_names_by_specificity(
    classifier: RoomTypeClassifier, raw_type: str, expected_room_type: RoomType
) -> None:
    assert classifier.classify(raw_type) == expected_room_type


def test_classify_falls_back_to_other_for_unrecognized_type(
    classifier: RoomTypeClassifier,
) -> None:
    assert classifier.classify("xyzzy_totally_unknown_room") == RoomType.OTHER
