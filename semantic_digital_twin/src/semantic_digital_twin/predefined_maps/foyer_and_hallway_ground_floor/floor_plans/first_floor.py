from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.floor_plan import (
    FloorPlan,
    OfficeRoom,
    OutlineFrame,
    StoreyHighRoom,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.rectilinear_room import (
    Doorway,
    OpenEdge,
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.spatial_types import Point2

# %% first floor


@dataclass
class FirstFloorPlan(FloorPlan):
    """
    The rooms and hallways of the first floor, as the building's floor plan of that
    storey shows them.
    """

    storey: Storey = Storey.FIRST_FLOOR

    rooms: list[OfficeRoom] = field(
        default_factory=lambda: [
            OfficeRoom(
                number="1.18b",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.27, 27.313),
                        Point2(-4.67, 27.313),
                        Point2(-4.67, 27.793),
                        Point2(-3.21, 27.793),
                        Point2(-3.21, 29.673),
                        Point2(-6.27, 29.673),
                    ],
                    doorways=[
                        Doorway(Point2(-3.21, 28.714), Point2(-3.21, 29.605)),
                        Doorway(Point2(-6.27, 29.6), Point2(-6.27, 28.893)),
                        Doorway(Point2(-6.27, 28.586), Point2(-6.27, 27.873)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18b.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 28.793),
                        Point2(-6.39, 28.793),
                        Point2(-6.39, 29.673),
                        Point2(-7.97, 29.673),
                    ],
                    doorways=[Doorway(Point2(-6.39, 28.893), Point2(-6.39, 29.6))],
                ),
            ),
            OfficeRoom(
                number="1.18b.2",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 27.793),
                        Point2(-6.39, 27.793),
                        Point2(-6.39, 28.673),
                        Point2(-7.97, 28.673),
                    ],
                    doorways=[Doorway(Point2(-6.39, 27.873), Point2(-6.39, 28.586))],
                ),
            ),
            OfficeRoom(
                number="1.18c",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.27, 29.933),
                        Point2(-3.21, 29.933),
                        Point2(-3.21, 31.793),
                        Point2(-6.27, 31.793),
                    ],
                    doorways=[
                        Doorway(Point2(-3.21, 30.83), Point2(-3.21, 31.719)),
                        Doorway(Point2(-6.27, 31.713), Point2(-6.27, 31.013)),
                        Doorway(Point2(-6.27, 30.723), Point2(-6.27, 30.013)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18c.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 30.913),
                        Point2(-6.39, 30.913),
                        Point2(-6.39, 31.793),
                        Point2(-7.97, 31.793),
                    ],
                    doorways=[Doorway(Point2(-6.39, 31.013), Point2(-6.39, 31.713))],
                ),
            ),
            OfficeRoom(
                number="1.18c.2",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 29.933),
                        Point2(-6.39, 29.933),
                        Point2(-6.39, 30.773),
                        Point2(-7.97, 30.773),
                    ],
                    doorways=[Doorway(Point2(-6.39, 30.013), Point2(-6.39, 30.723))],
                ),
            ),
            OfficeRoom(
                number="1.18d",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 33.92),
                        Point2(-7.922, 33.92),
                        Point2(-7.922, 35.5),
                        Point2(-9.822, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-9.722, 33.92), Point2(-9.015, 33.92)),
                        Doorway(Point2(-8.752, 33.92), Point2(-7.999, 33.92)),
                        Doorway(Point2(-7.998, 35.5), Point2(-8.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18d.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 35.64),
                        Point2(-7.922, 35.64),
                        Point2(-7.922, 37.02),
                        Point2(-9.822, 37.02),
                    ],
                    doorways=[
                        Doorway(Point2(-8.822, 35.64), Point2(-7.998, 35.64)),
                        Doorway(Point2(-7.997, 37.02), Point2(-8.881, 37.02)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18d.2",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.882, 32.02),
                        Point2(-7.922, 32.02),
                        Point2(-7.922, 33.78),
                        Point2(-8.882, 33.78),
                    ],
                    doorways=[Doorway(Point2(-7.999, 33.78), Point2(-8.752, 33.78))],
                ),
            ),
            OfficeRoom(
                number="1.18d.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 32.02),
                        Point2(-8.982, 32.02),
                        Point2(-8.982, 33.78),
                        Point2(-9.822, 33.78),
                    ],
                    doorways=[Doorway(Point2(-9.015, 33.78), Point2(-9.722, 33.78))],
                ),
            ),
            OfficeRoom(
                number="1.18e",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-12.322, 33.92),
                        Point2(-9.942, 33.92),
                        Point2(-9.942, 35.5),
                        Point2(-12.322, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-11.742, 33.92), Point2(-11.043, 33.92)),
                        Doorway(Point2(-10.802, 33.92), Point2(-10.03, 33.92)),
                        Doorway(Point2(-10.03, 35.5), Point2(-10.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18e.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 35.64),
                        Point2(-9.942, 35.64),
                        Point2(-9.942, 37.02),
                        Point2(-11.842, 37.02),
                    ],
                    doorways=[
                        Doorway(Point2(-10.822, 35.64), Point2(-10.03, 35.64)),
                        Doorway(Point2(-10.025, 37.02), Point2(-10.913, 37.02)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.18e.2",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-10.902, 32.02),
                        Point2(-9.942, 32.02),
                        Point2(-9.942, 33.78),
                        Point2(-10.902, 33.78),
                    ],
                    doorways=[Doorway(Point2(-10.03, 33.78), Point2(-10.802, 33.78))],
                ),
            ),
            OfficeRoom(
                number="1.18e.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 32.02),
                        Point2(-11.002, 32.02),
                        Point2(-11.002, 33.78),
                        Point2(-11.842, 33.78),
                    ],
                    doorways=[Doorway(Point2(-11.043, 33.78), Point2(-11.742, 33.78))],
                ),
            ),
            OfficeRoom(
                number="1.20",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.782, 35.22),
                        Point2(-5.202, 35.22),
                        Point2(-5.202, 36.94),
                        Point2(-7.782, 36.94),
                    ],
                    doorways=[
                        Doorway(Point2(-5.202, 35.97), Point2(-5.202, 36.872)),
                        Doorway(Point2(-6.664, 36.94), Point2(-7.55, 36.94)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.21",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.062, 35.22),
                        Point2(-2.782, 35.22),
                        Point2(-2.782, 36.94),
                        Point2(-5.062, 36.94),
                    ],
                    doorways=[Doorway(Point2(-5.062, 36.872), Point2(-5.062, 35.97))],
                ),
            ),
            OfficeRoom(
                number="1.22",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.442, 33.92),
                        Point2(-12.442, 33.92),
                        Point2(-12.442, 35.64),
                        Point2(-11.982, 35.64),
                        Point2(-11.982, 37.02),
                        Point2(-13.442, 37.02),
                    ],
                    doorways=[Doorway(Point2(-12.432, 37.02), Point2(-13.319, 37.02))],
                ),
            ),
            OfficeRoom(
                number="1.23b",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.46, -5.327),
                        Point2(-3.46, -7.187),
                        Point2(-3.56, -7.187),
                        Point2(-3.56, -8.987),
                        Point2(-3.46, -8.987),
                        Point2(-3.46, -10.547),
                        Point2(0.03, -10.547),
                        Point2(0.03, -5.327),
                    ],
                    doorways=[Doorway(Point2(-1.449, -5.327), Point2(-2.58, -5.327))],
                    open_edges=[
                        OpenEdge(Point2(-3.46, -5.327), Point2(-3.46, -7.187)),
                        OpenEdge(Point2(-3.46, -8.987), Point2(-3.46, -10.547)),
                        OpenEdge(Point2(0.03, -10.547), Point2(0.03, -5.327)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.23c",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-27.142, 38.97),
                        Point2(-27.142, 36.37),
                        Point2(-21.652, 36.37),
                        Point2(-21.652, 38.97),
                        Point2(-23.002, 38.97),
                        Point2(-23.702, 38.97),
                        Point2(-23.702, 39.512),
                        Point2(-24.842, 39.512),
                        Point2(-24.842, 38.97),
                        Point2(-25.652, 38.97),
                    ],
                    doorways=[
                        Doorway(Point2(-27.142, 38.45), Point2(-27.142, 37.45)),
                        Doorway(Point2(-21.652, 37.447), Point2(-21.652, 38.77)),
                    ],
                    open_edges=[
                        OpenEdge(Point2(-21.652, 38.97), Point2(-23.002, 38.97)),
                        OpenEdge(Point2(-23.702, 39.512), Point2(-24.842, 39.512)),
                        OpenEdge(Point2(-25.652, 38.97), Point2(-27.142, 38.97)),
                        OpenEdge(Point2(-27.142, 36.37), Point2(-21.652, 36.37)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.24",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.162, 32.02),
                        Point2(-2.782, 32.02),
                        Point2(-2.782, 33.85),
                        Point2(-5.162, 33.85),
                    ],
                    doorways=[
                        Doorway(Point2(-2.782, 32.172), Point2(-2.782, 33.17)),
                        Doorway(Point2(-5.162, 33.17), Point2(-5.162, 32.169)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.44",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.782, 32.02),
                        Point2(-5.282, 32.02),
                        Point2(-5.282, 33.85),
                        Point2(-7.782, 33.85),
                    ],
                    doorways=[Doorway(Point2(-5.282, 32.169), Point2(-5.282, 33.17))],
                ),
            ),
            OfficeRoom(
                number="1.47",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.25, 26.263),
                        Point2(-3.21, 26.263),
                        Point2(-3.21, 27.653),
                        Point2(-4.31, 27.653),
                        Point2(-4.31, 27.193),
                        Point2(-6.25, 27.193),
                    ],
                    doorways=[Doorway(Point2(-3.21, 26.308), Point2(-3.21, 27.195))],
                ),
            ),
            OfficeRoom(
                number="1.49",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-32.342, 37.06),
                        Point2(-31.342, 37.06),
                        Point2(-31.342, 32.12),
                        Point2(-27.362, 32.12),
                        Point2(-27.362, 38.84),
                        Point2(-32.342, 38.84),
                    ],
                    doorways=[
                        Doorway(Point2(-31.342, 35.197), Point2(-31.342, 33.92)),
                        Doorway(Point2(-27.362, 37.45), Point2(-27.362, 38.45)),
                        Doorway(Point2(-28.824, 38.84), Point2(-29.707, 38.84)),
                        Doorway(Point2(-32.342, 38.67), Point2(-32.342, 37.361)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.50",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-40.562, 32.1),
                        Point2(-31.482, 32.1),
                        Point2(-31.482, 36.92),
                        Point2(-40.562, 36.92),
                    ],
                    doorways=[Doorway(Point2(-31.482, 33.92), Point2(-31.482, 35.197))],
                ),
            ),
            OfficeRoom(
                number="1.51",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-40.602, 37.04),
                        Point2(-32.482, 37.04),
                        Point2(-32.482, 38.98),
                        Point2(-31.482, 38.98),
                        Point2(-31.482, 43.89),
                        Point2(-40.602, 43.89),
                    ],
                    doorways=[Doorway(Point2(-32.482, 37.361), Point2(-32.482, 38.67))],
                    open_edges=[
                        OpenEdge(Point2(-40.602, 37.04), Point2(-32.482, 37.04))
                    ],
                ),
            ),
            OfficeRoom(
                number="1.52",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-31.342, 38.99),
                        Point2(-27.352, 38.99),
                        Point2(-27.352, 43.89),
                        Point2(-31.342, 43.89),
                    ],
                    doorways=[Doorway(Point2(-29.707, 38.99), Point2(-28.824, 38.99))],
                ),
            ),
            OfficeRoom(
                number="1.53",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.452, 38.98),
                        Point2(-17.592, 38.98),
                        Point2(-17.592, 43.89),
                        Point2(-21.452, 43.89),
                    ],
                    doorways=[Doorway(Point2(-19.914, 38.98), Point2(-19.04, 38.98))],
                ),
            ),
            OfficeRoom(
                number="1.54",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-17.452, 38.98),
                        Point2(-13.472, 38.98),
                        Point2(-13.472, 43.89),
                        Point2(-17.452, 43.89),
                    ],
                    doorways=[Doorway(Point2(-15.808, 38.98), Point2(-14.934, 38.98))],
                ),
            ),
            OfficeRoom(
                number="1.55",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.342, 38.97),
                        Point2(-9.352, 38.97),
                        Point2(-9.352, 43.89),
                        Point2(-13.342, 43.89),
                    ],
                    doorways=[Doorway(Point2(-10.952, 38.97), Point2(-9.552, 38.97))],
                ),
            ),
            OfficeRoom(
                number="1.56",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.202, 38.98),
                        Point2(-5.222, 38.98),
                        Point2(-5.222, 43.89),
                        Point2(-9.202, 43.89),
                    ],
                    doorways=[Doorway(Point2(-7.543, 38.98), Point2(-6.664, 38.98))],
                ),
            ),
            OfficeRoom(
                number="1.57",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.082, 38.98),
                        Point2(-1.102, 38.98),
                        Point2(-1.102, 43.89),
                        Point2(-5.082, 43.89),
                    ],
                    doorways=[Doorway(Point2(-3.539, 38.98), Point2(-2.664, 38.98))],
                ),
            ),
            OfficeRoom(
                number="1.58",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-0.952, 38.92),
                        Point2(-0.002, 38.92),
                        Point2(-0.002, 37.27),
                        Point2(-0.752, 37.27),
                        Point2(-0.752, 36.04),
                        Point2(4.228, 36.04),
                        Point2(4.228, 43.89),
                        Point2(-0.952, 43.89),
                    ],
                    doorways=[Doorway(Point2(-0.002, 38.562), Point2(-0.002, 37.68))],
                ),
            ),
            OfficeRoom(
                number="1.59",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-17.462, 32.12),
                        Point2(-13.582, 32.12),
                        Point2(-13.582, 37.03),
                        Point2(-17.462, 37.03),
                    ],
                    doorways=[
                        Doorway(Point2(-14.93, 37.03), Point2(-15.818, 37.03)),
                        Doorway(Point2(-17.462, 35.943), Point2(-17.462, 35.065)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.60",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.452, 32.12),
                        Point2(-17.602, 32.12),
                        Point2(-17.602, 37.03),
                        Point2(-21.452, 37.03),
                    ],
                    doorways=[
                        Doorway(Point2(-17.602, 35.065), Point2(-17.602, 35.943)),
                        Doorway(Point2(-19.036, 37.03), Point2(-19.923, 37.03)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.75",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 31.873),
                        Point2(3.68, 31.873),
                        Point2(3.68, 35.843),
                        Point2(-1.24, 35.843),
                    ],
                    doorways=[Doorway(Point2(-1.24, 34.402), Point2(-1.24, 33.518))],
                ),
            ),
            OfficeRoom(
                number="1.76",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 27.743),
                        Point2(3.68, 27.743),
                        Point2(3.68, 31.733),
                        Point2(-1.24, 31.733),
                    ],
                    doorways=[Doorway(Point2(-1.24, 30.164), Point2(-1.24, 29.277))],
                ),
            ),
            OfficeRoom(
                number="1.77",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 23.643),
                        Point2(3.68, 23.643),
                        Point2(3.68, 27.613),
                        Point2(-1.24, 27.613),
                    ],
                    doorways=[Doorway(Point2(-1.24, 25.82), Point2(-1.24, 24.937))],
                ),
            ),
            OfficeRoom(
                number="1.78",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 19.523),
                        Point2(3.68, 19.523),
                        Point2(3.68, 23.493),
                        Point2(-1.24, 23.493),
                    ],
                    doorways=[Doorway(Point2(-1.24, 21.941), Point2(-1.24, 21.055))],
                ),
            ),
            OfficeRoom(
                number="1.79",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 15.403),
                        Point2(3.68, 15.403),
                        Point2(3.68, 19.353),
                        Point2(-1.24, 19.353),
                    ],
                    doorways=[Doorway(Point2(-1.24, 17.878), Point2(-1.24, 16.992))],
                ),
            ),
            OfficeRoom(
                number="1.80",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 11.253),
                        Point2(3.68, 11.253),
                        Point2(3.68, 15.233),
                        Point2(-1.24, 15.233),
                    ],
                    doorways=[Doorway(Point2(-1.24, 13.666), Point2(-1.24, 12.78))],
                ),
            ),
            OfficeRoom(
                number="1.81",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 7.133),
                        Point2(3.68, 7.133),
                        Point2(3.68, 11.103),
                        Point2(-1.24, 11.103),
                    ],
                    doorways=[Doorway(Point2(-1.24, 9.539), Point2(-1.24, 8.655))],
                ),
            ),
            OfficeRoom(
                number="1.82",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 3.013),
                        Point2(3.68, 3.013),
                        Point2(3.68, 7.003),
                        Point2(-1.24, 7.003),
                    ],
                    doorways=[Doorway(Point2(-1.24, 5.444), Point2(-1.24, 4.558))],
                ),
            ),
            OfficeRoom(
                number="1.83",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, -5.097),
                        Point2(3.68, -5.097),
                        Point2(3.68, 2.863),
                        Point2(-1.24, 2.863),
                    ],
                    doorways=[
                        Doorway(Point2(-1.24, 1.38), Point2(-1.24, 0.493)),
                        Doorway(Point2(-1.24, -2.72), Point2(-1.24, -3.603)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.85",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, -5.097),
                        Point2(-3.18, -5.097),
                        Point2(-3.18, 2.863),
                        Point2(-8.1, 2.863),
                    ],
                    doorways=[
                        Doorway(Point2(-3.18, -3.604), Point2(-3.18, -2.714)),
                        Doorway(Point2(-3.18, 0.492), Point2(-3.18, 1.381)),
                    ],
                ),
            ),
            OfficeRoom(
                number="1.87",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 3.013),
                        Point2(-3.18, 3.013),
                        Point2(-3.18, 7.003),
                        Point2(-8.1, 7.003),
                    ],
                    doorways=[Doorway(Point2(-3.18, 4.56), Point2(-3.18, 5.442))],
                ),
            ),
            OfficeRoom(
                number="1.88",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 7.133),
                        Point2(-3.18, 7.133),
                        Point2(-3.18, 11.103),
                        Point2(-8.1, 11.103),
                    ],
                    doorways=[Doorway(Point2(-3.18, 8.655), Point2(-3.18, 9.543))],
                ),
            ),
            OfficeRoom(
                number="1.89",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 11.253),
                        Point2(-3.18, 11.253),
                        Point2(-3.18, 15.233),
                        Point2(-8.1, 15.233),
                    ],
                    doorways=[Doorway(Point2(-3.18, 12.777), Point2(-3.18, 13.668))],
                ),
            ),
            OfficeRoom(
                number="1.90",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 15.403),
                        Point2(-3.18, 15.403),
                        Point2(-3.18, 19.353),
                        Point2(-8.1, 19.353),
                    ],
                    doorways=[Doorway(Point2(-3.18, 16.994), Point2(-3.18, 17.876))],
                ),
            ),
            OfficeRoom(
                number="1.91",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 19.523),
                        Point2(-3.18, 19.523),
                        Point2(-3.18, 23.493),
                        Point2(-8.1, 23.493),
                    ],
                    doorways=[Doorway(Point2(-3.18, 21.058), Point2(-3.18, 21.945))],
                ),
            ),
            OfficeRoom(
                number="1.92",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 23.643),
                        Point2(-3.18, 23.643),
                        Point2(-3.18, 25.923),
                        Point2(-6.56, 25.923),
                        Point2(-6.56, 27.653),
                        Point2(-8.1, 27.653),
                    ],
                    doorways=[Doorway(Point2(-3.18, 24.936), Point2(-3.18, 25.826))],
                ),
            ),
        ]
    )

    hallway: list[OfficeRoom] = field(
        default_factory=lambda: [
            OfficeRoom(
                number="1.19",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.502, 37.16),
                        Point2(-2.522, 37.16),
                        Point2(-2.522, 36.07),
                        Point2(-0.902, 36.07),
                        Point2(-0.902, 37.27),
                        Point2(-0.002, 37.27),
                        Point2(-0.002, 38.84),
                        Point2(-21.502, 38.84),
                    ],
                    doorways=[
                        Doorway(Point2(-19.923, 37.16), Point2(-19.036, 37.16)),
                        Doorway(Point2(-15.818, 37.16), Point2(-14.93, 37.16)),
                        Doorway(Point2(-13.319, 37.16), Point2(-12.432, 37.16)),
                        Doorway(Point2(-10.913, 37.16), Point2(-10.025, 37.16)),
                        Doorway(Point2(-8.881, 37.16), Point2(-7.997, 37.16)),
                        Doorway(Point2(-7.55, 37.16), Point2(-6.664, 37.16)),
                        Doorway(Point2(-0.002, 37.68), Point2(-0.002, 38.562)),
                        Doorway(Point2(-2.664, 38.84), Point2(-3.539, 38.84)),
                        Doorway(Point2(-6.664, 38.84), Point2(-7.543, 38.84)),
                        Doorway(Point2(-9.44, 38.84), Point2(-10.952, 38.84)),
                        Doorway(Point2(-14.934, 38.84), Point2(-15.808, 38.84)),
                        Doorway(Point2(-19.04, 38.84), Point2(-19.914, 38.84)),
                        Doorway(Point2(-21.502, 38.77), Point2(-21.502, 37.447)),
                    ],
                    open_edges=[OpenEdge(Point2(-2.522, 36.07), Point2(-0.902, 36.07))],
                ),
            ),
            OfficeRoom(
                number="1.19a",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.04, 11.063),
                        Point2(-1.4, 11.063),
                        Point2(-1.4, 35.843),
                        Point2(-3.04, 35.843),
                    ],
                    doorways=[
                        Doorway(Point2(-2.63, 11.063), Point2(-1.42, 11.063)),
                        Doorway(Point2(-1.4, 12.78), Point2(-1.4, 13.666)),
                        Doorway(Point2(-1.4, 16.992), Point2(-1.4, 17.878)),
                        Doorway(Point2(-1.4, 21.055), Point2(-1.4, 21.941)),
                        Doorway(Point2(-1.4, 24.937), Point2(-1.4, 25.82)),
                        Doorway(Point2(-1.4, 29.277), Point2(-1.4, 30.164)),
                        Doorway(Point2(-1.4, 33.518), Point2(-1.4, 34.402)),
                        Doorway(Point2(-3.04, 33.163), Point2(-3.04, 32.165)),
                        Doorway(Point2(-3.04, 31.719), Point2(-3.04, 30.83)),
                        Doorway(Point2(-3.04, 29.605), Point2(-3.04, 28.714)),
                        Doorway(Point2(-3.04, 27.195), Point2(-3.04, 26.308)),
                        Doorway(Point2(-3.04, 25.826), Point2(-3.04, 24.936)),
                        Doorway(Point2(-3.04, 21.945), Point2(-3.04, 21.058)),
                        Doorway(Point2(-3.04, 17.876), Point2(-3.04, 16.994)),
                        Doorway(Point2(-3.04, 13.668), Point2(-3.04, 12.777)),
                    ],
                    open_edges=[OpenEdge(Point2(-1.4, 35.843), Point2(-3.04, 35.843))],
                ),
            ),
            OfficeRoom(
                number="1.19b",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.04, -5.087),
                        Point2(-1.4, -5.087),
                        Point2(-1.4, 10.913),
                        Point2(-3.04, 10.913),
                    ],
                    doorways=[
                        Doorway(Point2(-2.58, -5.087), Point2(-1.449, -5.087)),
                        Doorway(Point2(-1.4, -3.603), Point2(-1.4, -2.72)),
                        Doorway(Point2(-1.4, 0.493), Point2(-1.4, 1.38)),
                        Doorway(Point2(-1.4, 4.558), Point2(-1.4, 5.444)),
                        Doorway(Point2(-1.4, 8.655), Point2(-1.4, 9.539)),
                        Doorway(Point2(-1.42, 10.913), Point2(-2.63, 10.913)),
                        Doorway(Point2(-3.04, 9.543), Point2(-3.04, 8.655)),
                        Doorway(Point2(-3.04, 5.442), Point2(-3.04, 4.56)),
                        Doorway(Point2(-3.04, 1.381), Point2(-3.04, 0.492)),
                        Doorway(Point2(-3.04, -2.714), Point2(-3.04, -3.604)),
                    ],
                ),
            ),
        ]
    )
