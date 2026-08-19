from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.shelf_membership_classifier import (
    ShelfMembershipClassifier,
)


@pytest.fixture
def classifier() -> ShelfMembershipClassifier:
    return ShelfMembershipClassifier()


@pytest.mark.parametrize(
    "raw_type",
    [
        "shelf",
        "wallshelf",
        "rack",
        "shelving",
        "bookshelf",
        "bookcase",
        "cabinet",
        "storagecabinet",
        "sideboard",
        "console",
        "credenza",
        "SHELF",
        "BookShelf",
    ],
)
def test_shelf_like_furniture_is_recognized(
    classifier: ShelfMembershipClassifier, raw_type: str
) -> None:
    """
    Any of the modelled keyword groups, in any casing, is recognized as shelf-like.
    """
    assert classifier.is_shelf_like(raw_type)


@pytest.mark.parametrize(
    "raw_type",
    ["dresser", "wardrobe", "closet", "displaycase", "table", "chair", "bed", "sofa"],
)
def test_furniture_outside_the_modelled_types_is_not_classified(
    classifier: ShelfMembershipClassifier, raw_type: str
) -> None:
    """
    ``is_shelf_like`` doubles as the membership gate deciding what enters training, so
    it answers ``False`` rather than a catch-all match.

    Dressers, wardrobes and display cases are storage but were deliberately left out of
    the modelled keywords, and a catch-all would pull them plus every table and chair
    back in.
    """
    assert not classifier.is_shelf_like(raw_type)
