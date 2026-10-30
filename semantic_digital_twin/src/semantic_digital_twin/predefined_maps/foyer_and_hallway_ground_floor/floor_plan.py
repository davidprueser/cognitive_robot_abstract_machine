from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

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
class StoreyHighRoom(RectilinearRoom):
    """
    A room whose walls stand as high as the storey it is on.
    """

    wall_height: float = Storey.height
    """
    Height of every wall above the floor, reaching the floor of the storey above.
    """


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
    The rooms of the storey, each standing on a floor of its own.
    """

    hallway: list[OfficeRoom]
    """
    The pieces of the storey's hallway, which stand on one floor together.
    """
