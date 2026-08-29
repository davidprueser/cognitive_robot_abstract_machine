from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ShelfMembershipClassifier:
    """
    Decides whether a free-form furniture name from the raw sage10k dataset (e.g.
    ``"bookshelf2"``, ``"storagecabinet"``) describes shelf-like storage furniture at
    all.

    Matching is a case-insensitive substring lookup against a fixed keyword set. This
    is the gate deciding which furniture enters training as a shelf -- a name outside
    the keyword set answers ``False`` rather than being admitted as some catch-all
    kind of shelf, which would let every table and chair in the dataset in.

    A shelf's kind is no longer classified from its furniture name; see
    :attr:`~semantic_digital_twin.scene_generation.scene_schema.EGShelf.theme_dominant_type`,
    which is derived from what is actually placed on the shelf instead.
    """

    _KEYWORDS: ClassVar[tuple[str, ...]] = (
        "bookshelf",
        "bookcase",
        "book_shelf",
        "book_case",
        "cabinet",
        "sideboard",
        "console",
        "credenza",
        "shelf",
        "shelv",
        "rack",
    )
    """
    Keywords identifying shelf-like furniture, matched as substrings of the raw name.
    """

    def is_shelf_like(self, raw_type: str) -> bool:
        """
        Decide whether a raw furniture name describes shelf-like storage furniture.

        :param raw_type: The dataset's free-form name for the furniture.
        :return: ``True`` when the name matches a modelled shelf-like keyword.
        """
        normalized_type = raw_type.lower()
        return any(keyword in normalized_type for keyword in self._KEYWORDS)
