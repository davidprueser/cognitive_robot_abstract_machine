from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from semantic_digital_twin.scene_generation.scene_schema import ShelfType


@dataclass(frozen=True)
class ShelfTypeClassifier:
    """
    Maps the free-form furniture names found in the raw sage10k dataset (e.g.
    ``"bookshelf2"``, ``"storagecabinet"``) onto the modelled :class:`ShelfType`
    categories.

    Matching is a case-insensitive, ordered keyword lookup, the first matching
    category winning, mirroring
    :class:`~semantic_digital_twin.scene_generation.object_type_classifier.ObjectTypeClassifier`.
    Compound names decide the ordering: ``"bookshelf"`` contains ``"shelf"``, so
    the bookcase keywords are tested before the open-shelf ones.

    This also serves as the gate deciding which furniture enters training, which
    is why an unrecognised name yields ``None`` instead of a catch-all member: a
    catch-all would admit every table and chair in the dataset as a shelf.
    """

    _KEYWORDS_BY_TYPE: ClassVar[tuple[tuple[ShelfType, tuple[str, ...]], ...]] = (
        (ShelfType.BOOKCASE, ("bookshelf", "bookcase", "book_shelf", "book_case")),
        (ShelfType.CABINET, ("cabinet",)),
        (ShelfType.SIDEBOARD, ("sideboard", "console", "credenza")),
        (ShelfType.OPEN_SHELF, ("shelf", "shelv", "rack")),
    )
    """
    Keywords identifying each shelf type, in the order they are tested.

    Ordered most specific first: a later entry's keyword may be a substring of an
    earlier entry's name.
    """

    def classify(self, raw_type: str) -> Optional[ShelfType]:
        """
        Classify a raw furniture name.

        :param raw_type: The dataset's free-form name for the furniture.
        :return: The matching shelf type, or ``None`` when the name does not describe
            furniture of a modelled type.
        """
        normalized_type = raw_type.lower()
        for shelf_type, keywords in self._KEYWORDS_BY_TYPE:
            if any(keyword in normalized_type for keyword in keywords):
                return shelf_type
        return None
