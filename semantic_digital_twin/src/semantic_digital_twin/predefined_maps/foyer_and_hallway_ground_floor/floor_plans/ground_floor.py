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
)
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.spatial_types import Point2

# %% ground floor


@dataclass
class GroundFloorPlan(FloorPlan):
    """
    The rooms and hallways of the ground floor, as the building's floor plan of that
    storey shows them.
    """

    storey: Storey = Storey.GROUND_FLOOR

    rooms: list[OfficeRoom] = field(
        default_factory=lambda: [
            OfficeRoom(
                number="0.30",
                outline_frame=OutlineFrame.ELEVATOR_ROOM,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-4.29, 0.048),
                        Point2(-0.51, 0.048),
                        Point2(-0.51, 6.758),
                        Point2(-4.29, 6.758),
                    ],
                    doorways=[
                        Doorway(Point2(-0.51, 5.35), Point2(-0.51, 6.63)),
                        Doorway(Point2(-4.29, 2.038), Point2(-4.29, 1.049)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.31",
                outline_frame=OutlineFrame.ELEVATOR_ROOM,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.55, 0.048),
                        Point2(-4.44, 0.048),
                        Point2(-4.44, 11.698),
                        Point2(-13.55, 11.698),
                    ],
                    doorways=[
                        Doorway(Point2(-4.44, 1.049), Point2(-4.44, 2.038)),
                        Doorway(Point2(-4.44, 9.41), Point2(-4.44, 10.916)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.32",
                outline_frame=OutlineFrame.ELEVATOR_ROOM,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-4.29, 6.918),
                        Point2(-1.17, 6.918),
                        Point2(-1.17, 11.838),
                        Point2(-4.29, 11.838),
                    ],
                    doorways=[Doorway(Point2(-4.29, 10.916), Point2(-4.29, 9.41))],
                ),
            ),
            OfficeRoom(
                number="0.34",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-17.312, 39.0),
                        Point2(-13.472, 39.0),
                        Point2(-13.472, 43.93),
                        Point2(-17.312, 43.93),
                    ],
                    doorways=[Doorway(Point2(-13.572, 39.0), Point2(-14.692, 39.0))],
                ),
            ),
            OfficeRoom(
                number="0.34.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.332, 39.0),
                        Point2(-11.4, 39.0),
                        Point2(-11.4, 39.7),
                        Point2(-9.352, 39.7),
                        Point2(-9.352, 43.93),
                        Point2(-13.332, 43.93),
                    ],
                    doorways=[Doorway(Point2(-9.447, 39.7), Point2(-10.442, 39.7))],
                ),
            ),
            OfficeRoom(
                number="0.35",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.232, 39.7),
                        Point2(-7.25, 39.7),
                        Point2(-7.25, 39.0),
                        Point2(-1.212, 39.0),
                        Point2(-1.212, 43.93),
                        Point2(-9.232, 43.93),
                    ],
                    doorways=[Doorway(Point2(-8.061, 39.7), Point2(-9.059, 39.7))],
                ),
            ),
            OfficeRoom(
                number="0.36",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-0.892, 39.0),
                        Point2(0.05, 39.0),
                        Point2(0.05, 37.17),
                        Point2(-0.532, 37.17),
                        Point2(-0.532, 36.04),
                        Point2(4.198, 36.04),
                        Point2(4.198, 43.93),
                        Point2(-0.892, 43.93),
                    ],
                    doorways=[Doorway(Point2(0.05, 37.485), Point2(0.05, 38.746))],
                ),
            ),
            OfficeRoom(
                number="0.38",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-15.962, 34.78),
                        Point2(-13.592, 34.78),
                        Point2(-13.592, 37.0),
                        Point2(-15.962, 37.0),
                    ],
                    doorways=[Doorway(Point2(-14.717, 37.0), Point2(-13.716, 37.0))],
                ),
            ),
            OfficeRoom(
                number="0.39",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-13.342, 33.92),
                        Point2(-12.442, 33.92),
                        Point2(-12.442, 35.64),
                        Point2(-11.982, 35.64),
                        Point2(-11.982, 37.0),
                        Point2(-13.342, 37.0),
                    ],
                    doorways=[Doorway(Point2(-13.32, 37.0), Point2(-12.431, 37.0))],
                ),
            ),
            OfficeRoom(
                number="0.40",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-12.322, 33.92),
                        Point2(-9.942, 33.92),
                        Point2(-9.942, 35.5),
                        Point2(-12.322, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-11.742, 33.92), Point2(-11.044, 33.92)),
                        Doorway(Point2(-10.802, 33.92), Point2(-10.031, 33.92)),
                        Doorway(Point2(-10.026, 35.5), Point2(-10.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.40.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 35.64),
                        Point2(-9.942, 35.64),
                        Point2(-9.942, 37.0),
                        Point2(-11.842, 37.0),
                    ],
                    doorways=[
                        Doorway(Point2(-10.822, 35.64), Point2(-10.026, 35.64)),
                        Doorway(Point2(-10.918, 37.0), Point2(-10.032, 37.0)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.40.2",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-10.902, 32.2),
                        Point2(-9.942, 32.2),
                        Point2(-9.942, 33.78),
                        Point2(-10.902, 33.78),
                    ],
                    doorways=[Doorway(Point2(-10.031, 33.78), Point2(-10.802, 33.78))],
                ),
            ),
            OfficeRoom(
                number="0.40.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-11.842, 32.2),
                        Point2(-11.002, 32.2),
                        Point2(-11.002, 33.78),
                        Point2(-11.842, 33.78),
                    ],
                    doorways=[Doorway(Point2(-11.044, 33.78), Point2(-11.742, 33.78))],
                ),
            ),
            OfficeRoom(
                number="0.41",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 33.92),
                        Point2(-7.922, 33.92),
                        Point2(-7.922, 35.5),
                        Point2(-9.822, 35.5),
                    ],
                    doorways=[
                        Doorway(Point2(-9.722, 33.92), Point2(-9.016, 33.92)),
                        Doorway(Point2(-8.752, 33.92), Point2(-7.998, 33.92)),
                        Doorway(Point2(-8.006, 35.5), Point2(-8.822, 35.5)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.41.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 35.64),
                        Point2(-7.922, 35.64),
                        Point2(-7.922, 37.0),
                        Point2(-9.822, 37.0),
                    ],
                    doorways=[
                        Doorway(Point2(-8.822, 35.64), Point2(-8.006, 35.64)),
                        Doorway(Point2(-8.887, 37.0), Point2(-8.006, 37.0)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.41.2",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.882, 32.2),
                        Point2(-7.922, 32.2),
                        Point2(-7.922, 33.78),
                        Point2(-8.882, 33.78),
                    ],
                    doorways=[Doorway(Point2(-7.998, 33.78), Point2(-8.752, 33.78))],
                ),
            ),
            OfficeRoom(
                number="0.41.3",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-9.822, 32.2),
                        Point2(-8.982, 32.2),
                        Point2(-8.982, 33.78),
                        Point2(-9.822, 33.78),
                    ],
                    doorways=[Doorway(Point2(-9.016, 33.78), Point2(-9.722, 33.78))],
                ),
            ),
            OfficeRoom(
                number="0.42",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.592, 35.58),
                        Point2(-5.262, 35.58),
                        Point2(-5.262, 37.0),
                        Point2(-7.592, 37.0),
                    ],
                    doorways=[
                        Doorway(Point2(-5.262, 35.973), Point2(-5.262, 36.844)),
                        Doorway(Point2(-7.55, 37.0), Point2(-6.664, 37.0)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.43",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.112, 35.16),
                        Point2(-2.772, 35.16),
                        Point2(-2.772, 37.0),
                        Point2(-5.112, 37.0),
                    ],
                    doorways=[Doorway(Point2(-5.112, 36.844), Point2(-5.112, 35.973))],
                ),
            ),
            OfficeRoom(
                number="0.45",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.25, 33.263),
                        Point2(3.7, 33.263),
                        Point2(3.7, 35.823),
                        Point2(-1.25, 35.823),
                    ],
                    doorways=[Doorway(Point2(-1.25, 33.587), Point2(-1.25, 34.474))],
                ),
            ),
            OfficeRoom(
                number="0.45.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.25, 29.6),
                        Point2(-0.45, 29.6),
                        Point2(-0.45, 27.763),
                        Point2(3.7, 27.763),
                        Point2(3.7, 33.123),
                        Point2(-1.25, 33.123),
                    ],
                    doorways=[Doorway(Point2(-0.45, 27.89), Point2(-0.45, 28.897))],
                ),
            ),
            OfficeRoom(
                number="0.46",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.25, 21.3),
                        Point2(-0.42, 21.3),
                        Point2(-0.42, 19.503),
                        Point2(3.7, 19.503),
                        Point2(3.7, 27.6),
                        Point2(-1.25, 27.6),
                    ],
                    doorways=[Doorway(Point2(-0.42, 19.65), Point2(-0.42, 21.08))],
                ),
            ),
            OfficeRoom(
                number="0.47",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.25, 14.183),
                        Point2(3.7, 14.183),
                        Point2(3.7, 19.363),
                        Point2(-0.42, 19.363),
                        Point2(-0.42, 17.5),
                        Point2(-1.25, 17.5),
                    ],
                    doorways=[Doorway(Point2(-0.42, 17.767), Point2(-0.42, 19.278))],
                ),
            ),
            OfficeRoom(
                number="0.49",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-1.25, -5.097),
                        Point2(3.7, -5.097),
                        Point2(3.7, 13.833),
                        Point2(-1.25, 13.833),
                    ],
                    doorways=[
                        Doorway(Point2(-1.25, 3.138), Point2(-1.25, 4.651)),
                        Doorway(Point2(-1.25, 1.269), Point2(-1.25, 2.783)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.50",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.14, -5.097),
                        Point2(-3.2, -5.097),
                        Point2(-3.2, 1.0),
                        Point2(-3.92, 1.0),
                        Point2(-3.92, 4.8),
                        Point2(-3.2, 4.8),
                        Point2(-3.2, 8.353),
                        Point2(-8.14, 8.353),
                    ],
                    doorways=[
                        Doorway(Point2(-3.92, 2.783), Point2(-3.92, 1.776)),
                        Doorway(Point2(-3.92, 4.58), Point2(-3.92, 3.153)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.52",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.14, 8.523),
                        Point2(-3.2, 8.523),
                        Point2(-3.2, 17.5),
                        Point2(-3.85, 17.5),
                        Point2(-3.85, 19.363),
                        Point2(-8.14, 19.363),
                    ],
                    doorways=[Doorway(Point2(-3.85, 19.278), Point2(-3.85, 18.273))],
                ),
            ),
            OfficeRoom(
                number="0.53",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.14, 19.503),
                        Point2(-3.85, 19.503),
                        Point2(-3.85, 21.3),
                        Point2(-3.2, 21.3),
                        Point2(-3.2, 24.773),
                        Point2(-8.14, 24.773),
                    ],
                    doorways=[Doorway(Point2(-3.85, 21.08), Point2(-3.85, 19.65))],
                ),
            ),
            OfficeRoom(
                number="0.66",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.18, -10.547),
                        Point2(3.69, -10.547),
                        Point2(3.69, -5.3),
                        Point2(-8.18, -5.3),
                    ],
                    doorways=[Doorway(Point2(-3.0, -5.3), Point2(-1.45, -5.3))],
                ),
            ),
            OfficeRoom(
                number="0.69",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.27, 29.933),
                        Point2(-3.2, 29.933),
                        Point2(-3.2, 31.793),
                        Point2(-6.27, 31.793),
                    ],
                    doorways=[
                        Doorway(Point2(-3.2, 31.714), Point2(-3.2, 30.823)),
                        Doorway(Point2(-6.27, 31.703), Point2(-6.27, 31.013)),
                        Doorway(Point2(-6.27, 30.716), Point2(-6.27, 30.013)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.69.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 30.913),
                        Point2(-6.39, 30.913),
                        Point2(-6.39, 31.793),
                        Point2(-7.97, 31.793),
                    ],
                    doorways=[Doorway(Point2(-6.39, 31.013), Point2(-6.39, 31.703))],
                ),
            ),
            OfficeRoom(
                number="0.69.2",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 29.933),
                        Point2(-6.39, 29.933),
                        Point2(-6.39, 30.773),
                        Point2(-7.97, 30.773),
                    ],
                    doorways=[Doorway(Point2(-6.39, 30.013), Point2(-6.39, 30.716))],
                ),
            ),
            OfficeRoom(
                number="0.70",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.27, 27.313),
                        Point2(-4.67, 27.313),
                        Point2(-4.67, 27.793),
                        Point2(-3.2, 27.793),
                        Point2(-3.2, 29.673),
                        Point2(-6.27, 29.673),
                    ],
                    doorways=[
                        Doorway(Point2(-3.2, 29.601), Point2(-3.2, 28.707)),
                        Doorway(Point2(-6.27, 29.593), Point2(-6.27, 28.893)),
                        Doorway(Point2(-6.27, 28.579), Point2(-6.27, 27.873)),
                    ],
                ),
            ),
            OfficeRoom(
                number="0.70.1",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 28.793),
                        Point2(-6.39, 28.793),
                        Point2(-6.39, 29.673),
                        Point2(-7.97, 29.673),
                    ],
                    doorways=[Doorway(Point2(-6.39, 28.893), Point2(-6.39, 29.593))],
                ),
            ),
            OfficeRoom(
                number="0.70.2",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.97, 27.793),
                        Point2(-6.39, 27.793),
                        Point2(-6.39, 28.673),
                        Point2(-7.97, 28.673),
                    ],
                    doorways=[Doorway(Point2(-6.39, 27.873), Point2(-6.39, 28.579))],
                ),
            ),
            OfficeRoom(
                number="0.71",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-6.26, 26.153),
                        Point2(-4.41, 26.153),
                        Point2(-4.41, 26.273),
                        Point2(-3.2, 26.273),
                        Point2(-3.2, 27.653),
                        Point2(-4.51, 27.653),
                        Point2(-4.51, 27.173),
                        Point2(-6.26, 27.173),
                    ],
                    doorways=[Doorway(Point2(-3.2, 27.194), Point2(-3.2, 26.299))],
                ),
            ),
            OfficeRoom(
                number="0.72",
                outline_frame=OutlineFrame.HALLWAY_SOUTH_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-8.14, 24.923),
                        Point2(-3.2, 24.923),
                        Point2(-3.2, 26.013),
                        Point2(-6.38, 26.013),
                        Point2(-6.38, 27.653),
                        Point2(-8.14, 27.653),
                    ],
                    doorways=[Doorway(Point2(-3.2, 25.945), Point2(-3.2, 25.046))],
                ),
            ),
            OfficeRoom(
                number="0.73",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-5.162, 32.02),
                        Point2(-2.782, 32.02),
                        Point2(-2.782, 33.85),
                        Point2(-5.162, 33.85),
                    ],
                    doorways=[
                        Doorway(Point2(-2.782, 33.17), Point2(-2.782, 32.162)),
                        Doorway(Point2(-5.162, 33.17), Point2(-5.162, 32.16)),
                    ],
                ),
            ),
            OfficeRoom(
                number="T.0.4.1",
                outline_frame=OutlineFrame.HALLWAY_EAST_STRETCH,
                layout=StoreyHighRoom(
                    outline=[
                        Point2(-7.782, 32.02),
                        Point2(-5.282, 32.02),
                        Point2(-5.282, 33.85),
                        Point2(-7.782, 33.85),
                    ],
                    doorways=[Doorway(Point2(-5.282, 32.16), Point2(-5.282, 33.17))],
                ),
            ),
        ]
    )

    hallway: list[OfficeRoom] = field(default_factory=list)
