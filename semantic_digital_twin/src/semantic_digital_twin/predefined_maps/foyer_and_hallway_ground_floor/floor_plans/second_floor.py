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

# %% second floor


@dataclass
class SecondFloorPlan(FloorPlan):
    """
    The rooms and hallways of the second floor, as the building's floor plan of that
    storey shows them.
    """

    storey: Storey = Storey.SECOND_FLOOR

    rooms: list[OfficeRoom] = field(
        default_factory=lambda: [
            OfficeRoom(
                number="2.16d",
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
                        Doorway(Point2(-3.21, 28.714), Point2(-3.21, 29.603)),
                        Doorway(Point2(-6.27, 29.603), Point2(-6.27, 28.893)),
                        Doorway(Point2(-6.27, 28.586), Point2(-6.27, 27.873)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16d.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 28.793),
                        Point2(-6.39, 28.793),
                        Point2(-6.39, 29.673),
                        Point2(-7.97, 29.673),
                    ],
                    doorways=[Doorway(Point2(-6.39, 28.893), Point2(-6.39, 29.603))],
                ),
            ),
            OfficeRoom(
                number="2.16d.2",
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
                number="2.16e",
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
                        Doorway(Point2(-6.27, 31.714), Point2(-6.27, 31.013)),
                        Doorway(Point2(-6.27, 30.723), Point2(-6.27, 30.013)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16e.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 30.913),
                        Point2(-6.39, 30.913),
                        Point2(-6.39, 31.793),
                        Point2(-7.97, 31.793),
                    ],
                    doorways=[Doorway(Point2(-6.39, 31.013), Point2(-6.39, 31.714))],
                ),
            ),
            OfficeRoom(
                number="2.16e.2",
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
                number="2.16f",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 33.92),
                        Point2(-7.922, 33.92),
                        Point2(-7.922, 35.5),
                        Point2(-9.822, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-9.722, 33.92), Point2(-9.019, 33.92)),
                        Doorway(Point2(-8.752, 33.92), Point2(-8.004, 33.92)),
                        Doorway(Point2(-7.996, 35.5), Point2(-8.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16f.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 35.64),
                        Point2(-7.922, 35.64),
                        Point2(-7.922, 37.02),
                        Point2(-9.822, 37.02),
                    ],
                    doorways=[
                        Doorway(Point2(-8.822, 35.64), Point2(-7.996, 35.64)),
                        Doorway(Point2(-8.001, 37.02), Point2(-8.886, 37.02)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16f.2",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.882, 32.02),
                        Point2(-7.922, 32.02),
                        Point2(-7.922, 33.78),
                        Point2(-8.882, 33.78),
                    ],
                    doorways=[Doorway(Point2(-8.004, 33.78), Point2(-8.752, 33.78))],
                ),
            ),
            OfficeRoom(
                number="2.16f.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 32.02),
                        Point2(-8.982, 32.02),
                        Point2(-8.982, 33.78),
                        Point2(-9.822, 33.78),
                    ],
                    doorways=[Doorway(Point2(-9.019, 33.78), Point2(-9.722, 33.78))],
                ),
            ),
            OfficeRoom(
                number="2.16g",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-12.322, 33.92),
                        Point2(-9.942, 33.92),
                        Point2(-9.942, 35.5),
                        Point2(-12.322, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-11.742, 33.92), Point2(-11.046, 33.92)),
                        Doorway(Point2(-10.802, 33.92), Point2(-10.03, 33.92)),
                        Doorway(Point2(-10.027, 35.5), Point2(-10.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16g.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 35.64),
                        Point2(-9.942, 35.64),
                        Point2(-9.942, 37.02),
                        Point2(-11.842, 37.02),
                    ],
                    doorways=[
                        Doorway(Point2(-10.822, 35.64), Point2(-10.027, 35.64)),
                        Doorway(Point2(-10.031, 37.02), Point2(-10.918, 37.02)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.16g.2",
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
                number="2.16g.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 32.02),
                        Point2(-11.002, 32.02),
                        Point2(-11.002, 33.78),
                        Point2(-11.842, 33.78),
                    ],
                    doorways=[Doorway(Point2(-11.046, 33.78), Point2(-11.742, 33.78))],
                ),
            ),
            OfficeRoom(
                number="2.18b",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.51, -5.327),
                        Point2(-3.51, -7.187),
                        Point2(-3.61, -7.187),
                        Point2(-3.61, -8.987),
                        Point2(-3.51, -8.987),
                        Point2(-3.51, -10.547),
                        Point2(-1.4, -10.547),
                        Point2(-1.4, -5.327),
                    ],
                    doorways=[Doorway(Point2(-1.452, -5.327), Point2(-2.585, -5.327))],
                    open_edges=[
                        OpenEdge(Point2(-3.51, -5.327), Point2(-3.51, -7.187)),
                        OpenEdge(Point2(-3.51, -8.987), Point2(-3.51, -10.547)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.18c",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-27.142, 38.97),
                        Point2(-27.142, 37.13),
                        Point2(-21.652, 37.13),
                        Point2(-21.652, 38.97),
                        Point2(-23.002, 38.97),
                        Point2(-23.702, 38.97),
                        Point2(-23.702, 39.512),
                        Point2(-24.842, 39.512),
                        Point2(-24.842, 38.97),
                        Point2(-25.652, 38.97),
                    ],
                    doorways=[
                        Doorway(Point2(-24.832, 37.13), Point2(-23.945, 37.13)),
                        Doorway(Point2(-21.652, 37.455), Point2(-21.652, 38.451)),
                    ],
                    open_edges=[
                        OpenEdge(Point2(-21.652, 38.97), Point2(-23.002, 38.97)),
                        OpenEdge(Point2(-23.702, 39.512), Point2(-24.842, 39.512)),
                        OpenEdge(Point2(-25.652, 38.97), Point2(-27.142, 38.97)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.50",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-27.142, 32.02),
                        Point2(-21.662, 32.02),
                        Point2(-21.662, 37.03),
                        Point2(-27.142, 37.03),
                    ],
                    doorways=[Doorway(Point2(-23.945, 37.03), Point2(-24.832, 37.03))],
                ),
            ),
            OfficeRoom(
                number="2.58",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.452, 38.99),
                        Point2(-17.602, 38.99),
                        Point2(-17.602, 43.89),
                        Point2(-21.452, 43.89),
                    ],
                    doorways=[Doorway(Point2(-18.574, 38.99), Point2(-17.688, 38.99))],
                ),
            ),
            OfficeRoom(
                number="2.59",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-17.452, 38.99),
                        Point2(-13.472, 38.99),
                        Point2(-13.472, 43.89),
                        Point2(-17.452, 43.89),
                    ],
                    doorways=[Doorway(Point2(-14.447, 38.99), Point2(-13.566, 38.99))],
                ),
            ),
            OfficeRoom(
                number="2.60",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.342, 39.23),
                        Point2(-9.352, 39.23),
                        Point2(-9.352, 43.89),
                        Point2(-13.342, 43.89),
                    ],
                    doorways=[Doorway(Point2(-10.326, 39.23), Point2(-9.439, 39.23))],
                ),
            ),
            OfficeRoom(
                number="2.61",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.202, 38.99),
                        Point2(-5.222, 38.99),
                        Point2(-5.222, 43.89),
                        Point2(-9.202, 43.89),
                    ],
                    doorways=[Doorway(Point2(-6.199, 38.99), Point2(-5.314, 38.99))],
                ),
            ),
            OfficeRoom(
                number="2.62",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.082, 38.92),
                        Point2(-0.852, 38.92),
                        Point2(-0.852, 36.05),
                        Point2(4.208, 36.05),
                        Point2(4.208, 43.89),
                        Point2(-5.082, 43.89),
                    ],
                    doorways=[Doorway(Point2(-4.941, 38.92), Point2(-4.056, 38.92))],
                ),
            ),
            OfficeRoom(
                number="2.64",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-17.392, 32.12),
                        Point2(-13.712, 32.12),
                        Point2(-13.712, 37.03),
                        Point2(-17.392, 37.03),
                    ],
                    doorways=[Doorway(Point2(-15.061, 37.03), Point2(-15.945, 37.03))],
                ),
            ),
            OfficeRoom(
                number="2.65",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.472, 32.12),
                        Point2(-17.542, 32.12),
                        Point2(-17.542, 37.03),
                        Point2(-21.472, 37.03),
                    ],
                    doorways=[Doorway(Point2(-17.687, 37.03), Point2(-18.574, 37.03))],
                ),
            ),
            OfficeRoom(
                number="2.67",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.342, 33.92),
                        Point2(-12.442, 33.92),
                        Point2(-12.442, 35.64),
                        Point2(-11.982, 35.64),
                        Point2(-11.982, 37.02),
                        Point2(-13.342, 37.02),
                    ],
                    doorways=[Doorway(Point2(-12.437, 37.02), Point2(-13.322, 37.02))],
                ),
            ),
            OfficeRoom(
                number="2.71",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.062, 35.22),
                        Point2(-2.782, 35.22),
                        Point2(-2.782, 36.94),
                        Point2(-5.062, 36.94),
                    ],
                    doorways=[Doorway(Point2(-5.062, 36.862), Point2(-5.062, 35.97))],
                ),
            ),
            OfficeRoom(
                number="2.75",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 31.873),
                        Point2(3.66, 31.873),
                        Point2(3.66, 35.823),
                        Point2(-1.24, 35.823),
                    ],
                    doorways=[Doorway(Point2(-1.24, 32.915), Point2(-1.24, 32.027))],
                ),
            ),
            OfficeRoom(
                number="2.76",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 27.743),
                        Point2(3.66, 27.743),
                        Point2(3.66, 31.733),
                        Point2(-1.24, 31.733),
                    ],
                    doorways=[Doorway(Point2(-1.24, 31.653), Point2(-1.24, 30.767))],
                ),
            ),
            OfficeRoom(
                number="2.77",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 23.643),
                        Point2(3.66, 23.643),
                        Point2(3.66, 27.613),
                        Point2(-1.24, 27.613),
                    ],
                    doorways=[Doorway(Point2(-1.24, 24.661), Point2(-1.24, 23.782))],
                ),
            ),
            OfficeRoom(
                number="2.78",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 19.523),
                        Point2(3.66, 19.523),
                        Point2(3.66, 23.493),
                        Point2(-1.24, 23.493),
                    ],
                    doorways=[Doorway(Point2(-1.24, 23.405), Point2(-1.24, 22.518))],
                ),
            ),
            OfficeRoom(
                number="2.79",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 15.383),
                        Point2(3.66, 15.383),
                        Point2(3.66, 19.353),
                        Point2(-1.24, 19.353),
                    ],
                    doorways=[Doorway(Point2(-1.24, 16.412), Point2(-1.24, 15.53))],
                ),
            ),
            OfficeRoom(
                number="2.80",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 11.253),
                        Point2(3.66, 11.253),
                        Point2(3.66, 15.233),
                        Point2(-1.24, 15.233),
                    ],
                    doorways=[Doorway(Point2(-1.24, 15.154), Point2(-1.24, 14.269))],
                ),
            ),
            OfficeRoom(
                number="2.81",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 7.133),
                        Point2(3.66, 7.133),
                        Point2(3.66, 11.103),
                        Point2(-1.24, 11.103),
                    ],
                    doorways=[Doorway(Point2(-1.24, 8.164), Point2(-1.24, 7.278))],
                ),
            ),
            OfficeRoom(
                number="2.82",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, 3.013),
                        Point2(3.66, 3.013),
                        Point2(3.66, 7.003),
                        Point2(-1.24, 7.003),
                    ],
                    doorways=[Doorway(Point2(-1.24, 6.905), Point2(-1.24, 6.021))],
                ),
            ),
            OfficeRoom(
                number="2.83",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.24, -2.477),
                        Point2(3.66, -2.477),
                        Point2(3.66, 2.863),
                        Point2(-1.24, 2.863),
                    ],
                    doorways=[Doorway(Point2(-1.24, -1.461), Point2(-1.24, -2.344))],
                ),
            ),
            OfficeRoom(
                number="2.84",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.26, -10.547),
                        Point2(3.66, -10.547),
                        Point2(3.66, -2.637),
                        Point2(-1.26, -2.637),
                    ],
                    doorways=[Doorway(Point2(-1.26, -2.719), Point2(-1.26, -3.604))],
                ),
            ),
            OfficeRoom(
                number="2.85",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, -5.097),
                        Point2(-3.2, -5.097),
                        Point2(-3.2, 0.123),
                        Point2(-8.1, 0.123),
                    ],
                    doorways=[Doorway(Point2(-3.2, -3.605), Point2(-3.2, -2.72))],
                ),
            ),
            OfficeRoom(
                number="2.86",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 0.263),
                        Point2(-3.2, 0.263),
                        Point2(-3.2, 4.243),
                        Point2(-8.1, 4.243),
                    ],
                    doorways=[Doorway(Point2(-3.2, 3.269), Point2(-3.2, 4.157))],
                ),
            ),
            OfficeRoom(
                number="2.87",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 4.403),
                        Point2(-3.2, 4.403),
                        Point2(-3.2, 8.383),
                        Point2(-8.1, 8.383),
                    ],
                    doorways=[Doorway(Point2(-3.2, 7.396), Point2(-3.2, 8.28))],
                ),
            ),
            OfficeRoom(
                number="2.88",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 8.523),
                        Point2(-3.2, 8.523),
                        Point2(-3.2, 12.503),
                        Point2(-8.1, 12.503),
                    ],
                    doorways=[Doorway(Point2(-3.2, 8.654), Point2(-3.2, 9.539))],
                ),
            ),
            OfficeRoom(
                number="2.89",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 12.623),
                        Point2(-3.2, 12.623),
                        Point2(-3.2, 16.623),
                        Point2(-8.1, 16.623),
                    ],
                    doorways=[Doorway(Point2(-3.2, 15.529), Point2(-3.2, 16.416))],
                ),
            ),
            OfficeRoom(
                number="2.90",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 16.763),
                        Point2(-3.2, 16.763),
                        Point2(-3.2, 20.743),
                        Point2(-8.1, 20.743),
                    ],
                    doorways=[Doorway(Point2(-3.2, 19.767), Point2(-3.2, 20.655))],
                ),
            ),
            OfficeRoom(
                number="2.91",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.1, 20.883),
                        Point2(-3.2, 20.883),
                        Point2(-3.2, 25.923),
                        Point2(-6.56, 25.923),
                        Point2(-6.56, 27.653),
                        Point2(-8.1, 27.653),
                    ],
                    doorways=[Doorway(Point2(-3.2, 21.027), Point2(-3.2, 21.915))],
                ),
            ),
            OfficeRoom(
                number="2.93",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.782, 32.02),
                        Point2(-5.282, 32.02),
                        Point2(-5.282, 33.85),
                        Point2(-7.782, 33.85),
                    ],
                    doorways=[Doorway(Point2(-5.282, 32.176), Point2(-5.282, 33.17))],
                ),
            ),
            OfficeRoom(
                number="2.96",
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
                    doorways=[Doorway(Point2(-3.21, 26.31), Point2(-3.21, 27.195))],
                ),
            ),
            OfficeRoom(
                number="2.97",
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
                        Doorway(Point2(-5.162, 33.17), Point2(-5.162, 32.176)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.97a",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.782, 35.22),
                        Point2(-5.202, 35.22),
                        Point2(-5.202, 36.94),
                        Point2(-7.782, 36.94),
                    ],
                    doorways=[
                        Doorway(Point2(-5.202, 35.97), Point2(-5.202, 36.862)),
                        Doorway(Point2(-6.669, 36.94), Point2(-7.55, 36.94)),
                    ],
                ),
            ),
        ]
    )

    hallway: list[OfficeRoom] = field(
        default_factory=lambda: [
            OfficeRoom(
                number="2.21a",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-21.552, 37.16),
                        Point2(-2.802, 37.16),
                        Point2(-2.802, 36.02),
                        Point2(-0.892, 36.02),
                        Point2(-0.892, 38.82),
                        Point2(-21.552, 38.82),
                    ],
                    doorways=[
                        Doorway(Point2(-18.574, 37.16), Point2(-17.687, 37.16)),
                        Doorway(Point2(-15.945, 37.16), Point2(-15.061, 37.16)),
                        Doorway(Point2(-13.322, 37.16), Point2(-12.437, 37.16)),
                        Doorway(Point2(-10.918, 37.16), Point2(-10.031, 37.16)),
                        Doorway(Point2(-8.886, 37.16), Point2(-8.001, 37.16)),
                        Doorway(Point2(-7.55, 37.16), Point2(-6.669, 37.16)),
                        Doorway(Point2(-1.855, 36.02), Point2(-0.912, 36.02)),
                        Doorway(Point2(-0.892, 37.672), Point2(-0.892, 38.562)),
                        Doorway(Point2(-4.056, 38.82), Point2(-4.941, 38.82)),
                        Doorway(Point2(-5.314, 38.82), Point2(-6.199, 38.82)),
                        Doorway(Point2(-9.439, 38.82), Point2(-10.326, 38.82)),
                        Doorway(Point2(-13.566, 38.82), Point2(-14.447, 38.82)),
                        Doorway(Point2(-17.688, 38.82), Point2(-18.574, 38.82)),
                        Doorway(Point2(-21.552, 38.451), Point2(-21.552, 37.455)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.21b",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.04, 12.713),
                        Point2(-1.4, 12.713),
                        Point2(-1.4, 35.813),
                        Point2(-3.04, 35.813),
                    ],
                    doorways=[
                        Doorway(Point2(-2.63, 12.713), Point2(-1.42, 12.713)),
                        Doorway(Point2(-1.4, 14.269), Point2(-1.4, 15.154)),
                        Doorway(Point2(-1.4, 15.53), Point2(-1.4, 16.412)),
                        Doorway(Point2(-1.4, 22.518), Point2(-1.4, 23.405)),
                        Doorway(Point2(-1.4, 23.782), Point2(-1.4, 24.661)),
                        Doorway(Point2(-1.4, 30.767), Point2(-1.4, 31.653)),
                        Doorway(Point2(-1.4, 32.027), Point2(-1.4, 32.915)),
                        Doorway(Point2(-1.42, 35.813), Point2(-2.983, 35.813)),
                        Doorway(Point2(-3.04, 33.163), Point2(-3.04, 32.165)),
                        Doorway(Point2(-3.04, 31.719), Point2(-3.04, 30.83)),
                        Doorway(Point2(-3.04, 29.603), Point2(-3.04, 28.714)),
                        Doorway(Point2(-3.04, 27.195), Point2(-3.04, 26.31)),
                        Doorway(Point2(-3.04, 21.915), Point2(-3.04, 21.027)),
                        Doorway(Point2(-3.04, 20.655), Point2(-3.04, 19.767)),
                        Doorway(Point2(-3.04, 16.416), Point2(-3.04, 15.529)),
                    ],
                ),
            ),
            OfficeRoom(
                number="2.21c",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-3.04, -5.087),
                        Point2(-1.4, -5.087),
                        Point2(-1.4, 12.593),
                        Point2(-3.04, 12.593),
                    ],
                    doorways=[
                        Doorway(Point2(-2.585, -5.087), Point2(-1.452, -5.087)),
                        Doorway(Point2(-1.4, -3.604), Point2(-1.4, -2.719)),
                        Doorway(Point2(-1.4, -2.344), Point2(-1.4, -1.461)),
                        Doorway(Point2(-1.4, 6.021), Point2(-1.4, 6.905)),
                        Doorway(Point2(-1.4, 7.278), Point2(-1.4, 8.164)),
                        Doorway(Point2(-1.42, 12.593), Point2(-2.63, 12.593)),
                        Doorway(Point2(-3.04, 9.539), Point2(-3.04, 8.654)),
                        Doorway(Point2(-3.04, 8.28), Point2(-3.04, 7.396)),
                        Doorway(Point2(-3.04, 4.157), Point2(-3.04, 3.269)),
                        Doorway(Point2(-3.04, -2.72), Point2(-3.04, -3.605)),
                    ],
                ),
            ),
        ]
    )
