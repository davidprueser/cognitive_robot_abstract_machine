from __future__ import annotations

from enum import IntEnum

from krrood.ormatic.utils import classproperty
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    FirstFloor,
    GroundFloor,
    SecondFloor,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Level

# %% storeys


class Storey(IntEnum):
    """
    A storey of the office building, valued as the number of storeys below it.
    """

    GROUND_FLOOR = 0
    """
    The storey at street level, holding the lab elevator room.
    """

    FIRST_FLOOR = 1
    """
    The storey right above the ground floor.
    """

    SECOND_FLOOR = 2
    """
    The storey right above the first floor.
    """

    @classproperty
    def height(cls) -> float:
        """
        :return: The distance between the floors of two consecutive storeys, in metres,
            as the floor plans of the building give it.
        """
        return 3.45

    @property
    def elevation(self) -> float:
        """
        :return: The height of this storey's floor above the ground floor's.
        """
        return self.value * Storey.height

    @property
    def level_type(self) -> type[Level]:
        """
        :return: The level annotation standing for this storey.
        """
        match self:
            case Storey.GROUND_FLOOR:
                return GroundFloor
            case Storey.FIRST_FLOOR:
                return FirstFloor
            case Storey.SECOND_FLOOR:
                return SecondFloor
