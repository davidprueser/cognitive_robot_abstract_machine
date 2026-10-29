from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

from krrood.adapters.json_serializer import from_json
from krrood.patterns.path_enumeration import PathEnumeration

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    RectilinearRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)

# %% frames


class OutlineFrame(Enum):
    """
    A frame of the lab elevator room or of its hallway that the outline of an office
    room is given in.
    """

    ELEVATOR_ROOM = auto()
    """
    The outline frame of the lab elevator room.
    """

    HALLWAY_EAST_STRETCH = auto()
    """
    The outline frame of the hallway's stretch running east from the elevator room.
    """

    HALLWAY_SOUTH_STRETCH = auto()
    """
    The outline frame of the hallway's long stretch running south.
    """


# %% rooms


@dataclass
class OfficeRoom:
    """
    A room or hallway of the office, as traced from the floor plan of its storey: its
    floor, walls and doorways, with its outline given in a frame of the lab map.
    """

    number: str
    """
    The room's number on the floor plan, such as ``1.53``.
    """

    outline_frame: OutlineFrame
    """
    The frame the outline of :attr:`layout` is given in.
    """

    layout: RectilinearRoom
    """
    The room's floor, walls and doorways.
    """


class FloorPlanFile(PathEnumeration):
    """
    The files holding the floor plan of each storey of the office.
    """

    GROUND_FLOOR = Path(__file__).parent / "floor_plans" / "ground_floor.json"
    """
    The floor plan of the ground floor.
    """

    FIRST_FLOOR = Path(__file__).parent / "floor_plans" / "first_floor.json"
    """
    The floor plan of the first floor.
    """

    SECOND_FLOOR = Path(__file__).parent / "floor_plans" / "second_floor.json"
    """
    The floor plan of the second floor.
    """


@dataclass
class FloorPlan:
    """
    The rooms and hallways of one storey of the office that are reached from the lab
    elevator room and its hallway, traced from the building's floor plan of that
    storey.

    The rooms' floors lie at height zero in their outline frames; the storey's
    elevation lifts them.
    """

    storey: Storey
    """
    The storey the rooms are on.
    """

    rooms: list[OfficeRoom]
    """
    The rooms and hallways of the storey.
    """

    @classmethod
    def read(cls, file: FloorPlanFile) -> FloorPlan:
        """
        :param file: The file holding the floor plan.
        :return: The floor plan stored in ``file``.
        """
        return from_json(json.loads(file.read_text()))
