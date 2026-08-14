from __future__ import annotations

import pytest

from semantic_digital_twin.scene_generation.scene_schema import ShelfType
from semantic_digital_twin.scene_generation.shelf_type_classifier import (
    ShelfTypeClassifier,
)


@pytest.fixture
def classifier() -> ShelfTypeClassifier:
    return ShelfTypeClassifier()


@pytest.mark.parametrize(
    "raw_type, expected_shelf_type",
    [
        ("shelf", ShelfType.OPEN_SHELF),
        ("wallshelf", ShelfType.OPEN_SHELF),
        ("rack", ShelfType.OPEN_SHELF),
        ("shelving", ShelfType.OPEN_SHELF),
        ("bookshelf", ShelfType.BOOKCASE),
        ("bookcase", ShelfType.BOOKCASE),
        ("cabinet", ShelfType.CABINET),
        ("storagecabinet", ShelfType.CABINET),
        ("sideboard", ShelfType.SIDEBOARD),
        ("console", ShelfType.SIDEBOARD),
        ("SHELF", ShelfType.OPEN_SHELF),
        ("BookShelf", ShelfType.BOOKCASE),
    ],
)
def test_a_raw_name_classifies_as_the_type_it_names(
    classifier: ShelfTypeClassifier, raw_type: str, expected_shelf_type: ShelfType
) -> None:
    """
    The headline keyword of each type maps to it, whatever the casing.
    """
    assert classifier.classify(raw_type) is expected_shelf_type


def test_a_bookshelf_is_not_swept_into_the_open_shelf_category(
    classifier: ShelfTypeClassifier,
) -> None:
    """
    ``"bookshelf"`` contains ``"shelf"``, so the more specific bookcase keywords have to
    be tested first.

    Getting this order wrong silently merges the single largest well-populated type into
    the generic one, and the two differ in exactly the way the model is meant to learn
    -- bookcases average far more layers than open shelves.
    """
    assert classifier.classify("bookshelf") is ShelfType.BOOKCASE
    assert classifier.classify("bookshelf2") is ShelfType.BOOKCASE
    assert classifier.classify("book_shelf") is ShelfType.BOOKCASE


@pytest.mark.parametrize(
    "raw_type",
    ["dresser", "wardrobe", "closet", "displaycase", "table", "chair", "bed", "sofa"],
)
def test_furniture_outside_the_modelled_types_is_not_classified(
    classifier: ShelfTypeClassifier, raw_type: str
) -> None:
    """
    ``classify`` doubles as the membership gate deciding what enters training, so it
    answers ``None`` rather than a catch-all member.

    Dressers, wardrobes and display cases are storage but were deliberately left out of
    the modelled types, and a catch-all would pull them plus every table and chair back
    in.
    """
    assert classifier.classify(raw_type) is None


def test_every_shelf_type_is_reachable(classifier: ShelfTypeClassifier) -> None:
    """
    A member no keyword can produce would be a category the model can be asked for but
    never trained on, which conditioning cannot report as impossible.
    """
    reachable = {
        classifier.classify(keyword)
        for _, keywords in ShelfTypeClassifier._KEYWORDS_BY_TYPE
        for keyword in keywords
    }
    assert reachable == set(ShelfType)
