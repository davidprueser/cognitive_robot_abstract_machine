import math
from collections import Counter
from dataclasses import dataclass, field

import pytest
from typing_extensions import Iterator

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.navigation import (
    ElevatorNavigation,
    PathPlanningNavigateAction,
)
from experiments.tiago_elevator_demo.demo import TiagoElevatorRideDemonstration
from experiments.tiago_room_navigation_demo.demo import (
    TiagoIAIOfficeRoomNavigationDemonstration,
    TiagoRoomNavigationDemonstration,
)
from experiments.tiago_room_navigation_demo.exceptions import (
    NoElevatorError,
    RoomOnNoLevelError,
    TargetRoomNotReachedError,
)
from experiments.tiago_room_navigation_demo.room_navigation import (
    NavigateToRoomAction,
)
from experiments.tiago_room_navigation_demo.trials import (
    RoomNavigationTrialResult,
    RoomNavigationTrials,
    RoomNavigationTrialSummary,
    RoomTrip,
    RoomTripDraw,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.predefined_maps.arise_office import AriseOfficeRoom
from semantic_digital_twin.predefined_maps.foyer_and_hallway_ground_floor.storey import (
    Storey,
)
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Elevator,
    FirstFloor,
    Floor,
    GroundFloor,
    Level,
    Room,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
    Point3,
)
from semantic_digital_twin.world_description.geometry import Scale

# %% Tiago standing in the lobby


@dataclass(kw_only=True)
class LevelRecordingRoomNavigationDemonstration(TiagoRoomNavigationDemonstration):
    """
    A room navigation demonstration that records the level Tiago stands on whenever a
    leg of the trip is built.
    """

    levels_when_built: list[Level | None] = field(default_factory=list)
    """
    The level Tiago stood on as each leg was built, in building order.
    """

    def build_plan(self, context: Context) -> PlanNode:
        self.levels_when_built.append(Level.under(context.robot.root))
        return super().build_plan(context)


@pytest.fixture
def lobby_demonstration() -> Iterator[TiagoElevatorRideDemonstration]:
    demonstration = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    )
    yield demonstration
    demonstration.tear_down()


@pytest.fixture
def lobby_context(lobby_demonstration) -> Context:
    return lobby_demonstration.build_context(
        lobby_demonstration.build_simulated_world()
    )


def planned_steps(action: NavigateToRoomAction, context: Context) -> list:
    """
    :return: The actions ``action`` expands into for the robot of ``context``.
    """
    execute_single(action, context=context)
    return [child.designator for child in action._action_plan.children]


# %% planning the way to a room


def test_a_room_on_the_robots_level_is_driven_to_directly(
    lobby_demonstration, lobby_context
):
    east_office = lobby_demonstration.office.get_room(
        lobby_context.world, GroundFloor, AriseOfficeRoom.EAST_OFFICE
    )

    [drive] = planned_steps(NavigateToRoomAction(east_office), lobby_context)

    assert isinstance(drive, PathPlanningNavigateAction)
    assert drive.target.to_np() == pytest.approx(east_office.floor_centre.to_np())


def test_a_room_on_another_level_is_approached_by_riding_the_elevator_to_its_level(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    west_office = lobby_demonstration.office.get_room(
        world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
    )
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    action = NavigateToRoomAction(west_office)

    steps = planned_steps(action, lobby_context)

    assert [type(step) for step in steps] == [
        PathPlanningNavigateAction,
        ElevatorNavigation,
    ]
    drive_to_elevator, ride = steps
    assert drive_to_elevator.target.to_np() == pytest.approx(
        elevator.pose_in_front_of_doors(action.elevator_waiting_distance).to_np()
    )
    assert ride.target_floor is world.get_semantic_annotations_by_type(FirstFloor)[0]


@pytest.mark.parametrize(
    "level_type, room",
    [
        (GroundFloor, AriseOfficeRoom.EAST_OFFICE),
        (FirstFloor, AriseOfficeRoom.WEST_OFFICE),
    ],
)
def test_every_drive_to_a_room_keeps_the_obstacle_clearance(
    lobby_demonstration, lobby_context, level_type, room
):
    target_room = lobby_demonstration.office.get_room(
        lobby_context.world, level_type, room
    )
    action = NavigateToRoomAction(target_room, obstacle_clearance=0.15)

    drives = [
        step
        for step in planned_steps(action, lobby_context)
        if isinstance(step, PathPlanningNavigateAction)
    ]

    assert [drive.obstacle_clearance for drive in drives] == [action.obstacle_clearance]


def test_the_ride_to_a_room_keeps_the_elevator_exit_clearance(
    lobby_demonstration, lobby_context
):
    west_office = lobby_demonstration.office.get_room(
        lobby_context.world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
    )
    action = NavigateToRoomAction(west_office, elevator_exit_clearance=0.7)

    [ride] = [
        step
        for step in planned_steps(action, lobby_context)
        if isinstance(step, ElevatorNavigation)
    ]

    assert ride.exit_clearance == action.elevator_exit_clearance


def test_the_demonstration_drives_with_its_obstacle_clearance(
    lobby_demonstration, lobby_context
):
    east_office = lobby_demonstration.office.get_room(
        lobby_context.world, GroundFloor, AriseOfficeRoom.EAST_OFFICE
    )
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        target_room_name=east_office.name.name,
        obstacle_clearance=0.15,
    )

    [navigation] = demonstration.build_plan(lobby_context).children

    assert navigation.designator.obstacle_clearance == demonstration.obstacle_clearance


def test_the_demonstration_rides_with_its_elevator_exit_clearance(
    lobby_demonstration, lobby_context
):
    west_office = lobby_demonstration.office.get_room(
        lobby_context.world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
    )
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        target_room_name=west_office.name.name,
        elevator_exit_clearance=0.7,
    )

    [navigation] = demonstration.build_plan(lobby_context).children

    assert (
        navigation.designator.elevator_exit_clearance
        == demonstration.elevator_exit_clearance
    )


@pytest.mark.parametrize(
    "level_type, room",
    [
        (GroundFloor, AriseOfficeRoom.EAST_OFFICE),
        (FirstFloor, AriseOfficeRoom.WEST_OFFICE),
    ],
)
def test_the_elevator_operator_takes_tiago_to_the_target_rooms_level(
    arise_office, level_type, room
):
    world = TiagoElevatorRideDemonstration(
        used_robot=Tiago, execution_type=ExecutionType.KINEMATICALLY_SIMULATED
    ).build_simulated_world()
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        target_room_name=arise_office.get_room(world, level_type, room).name.name,
    )

    demonstration.build_context(world)

    assert (
        demonstration.elevator_operator.floor
        is world.get_semantic_annotations_by_type(level_type)[0]
    )
    demonstration.tear_down()


@pytest.mark.parametrize(
    "level_type, room, floor_index",
    [
        (GroundFloor, AriseOfficeRoom.EAST_OFFICE, 0),
        (FirstFloor, AriseOfficeRoom.WEST_OFFICE, 1),
    ],
)
def test_the_destination_floor_counts_the_target_rooms_level_from_the_lowest(
    arise_office, arise_office_world, level_type, room, floor_index
):
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        execution_type=ExecutionType.KINEMATICALLY_SIMULATED,
        target_room_name=arise_office.get_room(
            arise_office_world, level_type, room
        ).name.name,
    )

    assert demonstration.destination_floor_index(arise_office_world) == floor_index


def test_a_room_on_no_level_cannot_be_planned_for(lobby_context):
    world = lobby_context.world
    basement_floor = Floor.get_annotation_specification(
        "basement_floor",
        Floor.get_default_root_kinematic_structure_entity_specification(
            scale=Scale(2, 2, 0)
        ),
    ).spawn(world, parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-3))
    basement = Room(name=PrefixedName("basement"), floor=basement_floor)
    with world.modify_world():
        world.add_semantic_annotation(basement)

    with pytest.raises(RoomOnNoLevelError):
        planned_steps(NavigateToRoomAction(basement), lobby_context)


def test_a_room_on_another_level_cannot_be_planned_for_without_an_elevator(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    with world.modify_world():
        world.remove_semantic_annotation(
            world.get_semantic_annotations_by_type(Elevator)[0]
        )
    landing = lobby_demonstration.office.get_room(
        world, FirstFloor, AriseOfficeRoom.LANDING
    )

    with pytest.raises(NoElevatorError):
        planned_steps(NavigateToRoomAction(landing), lobby_context)


def test_path_planning_lays_its_path_out_over_the_level_the_robot_is_on(
    lobby_demonstration, lobby_context
):
    world = lobby_context.world
    east_office = lobby_demonstration.office.get_room(
        world, GroundFloor, AriseOfficeRoom.EAST_OFFICE
    )
    action = PathPlanningNavigateAction(east_office.floor_centre)

    execute_single(action, context=lobby_context)

    assert action._ground is world.get_semantic_annotations_by_type(GroundFloor)[0]


# %% driving to a room


@pytest.mark.parametrize(
    "level_type, room",
    [
        (GroundFloor, AriseOfficeRoom.WEST_OFFICE),
        (FirstFloor, AriseOfficeRoom.EAST_OFFICE),
    ],
)
def test_tiago_reaches_the_target_room(
    rclpy_node, arise_office, arise_office_world, level_type, room
):
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        target_room_name=arise_office.get_room(
            arise_office_world, level_type, room
        ).name.name,
    )

    world = demonstration.run()

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    assert (
        Floor.under(robot.root)
        is world.get_semantic_annotation_by_name(demonstration.target_room_name).floor
    )


def test_the_drive_into_a_room_on_another_level_is_planned_after_the_ride(
    rclpy_node, arise_office, arise_office_world
):
    """
    A leg is laid out from where Tiago stands when it is built, so the drive into the
    room must be built once the ride has brought Tiago to the room's level.
    """
    demonstration = LevelRecordingRoomNavigationDemonstration(
        used_robot=Tiago,
        target_room_name=arise_office.get_room(
            arise_office_world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
        ).name.name,
    )

    world = demonstration.run()

    assert demonstration.levels_when_built == [
        world.get_semantic_annotations_by_type(GroundFloor)[0],
        world.get_semantic_annotations_by_type(FirstFloor)[0],
    ]


def test_tiago_outside_the_target_room_fails_the_demonstration(
    arise_office, arise_office_world
):
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        target_room_name=arise_office.get_room(
            arise_office_world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
        ).name.name,
    )

    world = demonstration.build_simulated_world()

    with pytest.raises(TargetRoomNotReachedError):
        demonstration.conclude(world)
    demonstration.tear_down()


# %% placing Tiago in the start room


@pytest.fixture
def lobby_room_navigation(
    arise_office, arise_office_world
) -> Iterator[TiagoRoomNavigationDemonstration]:
    demonstration = TiagoRoomNavigationDemonstration(
        used_robot=Tiago,
        target_room_name=arise_office.get_room(
            arise_office_world, FirstFloor, AriseOfficeRoom.WEST_OFFICE
        ).name.name,
    )
    yield demonstration
    demonstration.tear_down()


def test_tiago_is_placed_in_the_middle_of_a_start_room_on_another_level(
    arise_office, lobby_room_navigation
):
    world = lobby_room_navigation.build_simulated_world()
    start_room = arise_office.get_room(world, FirstFloor, AriseOfficeRoom.EAST_OFFICE)
    lobby_room_navigation.start_room_name = start_room.name.name

    lobby_room_navigation.place_robot_in_start_room(world)

    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    assert robot.root.global_transform.position.to_np()[:3] == pytest.approx(
        lobby_room_navigation.start_pose(world).position.to_np()[:3]
    )
    assert Floor.under(robot.root) is start_room.floor


def test_tiago_left_in_the_cabin_is_placed_back_on_the_map(
    arise_office, lobby_room_navigation
):
    world = lobby_room_navigation.build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    world.move_branch(
        robot.root, world.get_semantic_annotations_by_type(Elevator)[0].root
    )

    lobby_room_navigation.place_robot_in_start_room(world)

    assert robot.root.parent_kinematic_structure_entity is world.root


# %% drawing trips


@pytest.fixture
def iai_office_trip_draw(iai_office, iai_office_name) -> RoomTripDraw:
    return RoomTripDraw(
        office=iai_office, building_name=iai_office_name, number_of_trips=50, seed=0
    )


def test_a_trip_ends_on_another_level_than_it_starts_on(
    iai_office_world, iai_office_trip_draw
):
    trips = iai_office_trip_draw.trips(iai_office_world)

    levels_of_trips = [
        [
            Level.under(
                iai_office_world.get_semantic_annotation_by_name(room_name).floor.root
            )
            for room_name in (trip.start_room_name, trip.target_room_name)
        ]
        for trip in trips
    ]
    assert len(trips) == iai_office_trip_draw.number_of_trips
    assert all(start is not target for start, target in levels_of_trips)


def test_the_same_seed_draws_the_same_trips(iai_office_world, iai_office_trip_draw):
    assert iai_office_trip_draw.trips(iai_office_world) == iai_office_trip_draw.trips(
        iai_office_world
    )


def test_only_plain_rooms_a_robot_may_be_sent_into_are_drawn(
    iai_office, iai_office_name, iai_office_world, iai_office_trip_draw
):
    drawn_room_names = {
        room_name
        for trip in iai_office_trip_draw.trips(iai_office_world)
        for room_name in (trip.start_room_name, trip.target_room_name)
    }

    assert drawn_room_names <= {
        room.name.name
        for room in iai_office.navigation_targets(iai_office_world, iai_office_name)
        if type(room) is Room
    }


# %% trips through the IAI's office


@pytest.fixture
def iai_office_room_navigation() -> Iterator[TiagoIAIOfficeRoomNavigationDemonstration]:
    demonstration = TiagoIAIOfficeRoomNavigationDemonstration(used_robot=Tiago)
    yield demonstration
    demonstration.tear_down()


def test_tiago_finds_a_way_between_the_elevator_and_every_room_trips_are_drawn_from(
    iai_office_room_navigation,
):
    world = iai_office_room_navigation.build_simulated_world()
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    elevator = world.get_semantic_annotations_by_type(Elevator)[0]
    base_radius = robot.mobile_base.base_radius
    robot_height = (
        robot.as_bounding_box_collection_in_frame(robot.root).bounding_box().scale.z
    )
    footprint_offsets = [(0.0, 0.0)] + [
        (base_radius * math.cos(angle), base_radius * math.sin(angle))
        for angle in (index * math.pi / 4 for index in range(8))
    ]
    rooms = [
        world.get_semantic_annotation_by_name(room_name)
        for room_name in RoomTripDraw(
            office=iai_office_room_navigation.office,
            building_name=iai_office_room_navigation.building_name,
        ).room_levels(world)
    ]
    rooms_off_their_floor = []
    for level in elevator.floors:
        elevator.drive_to_floor(level)
        free_space = level.planar_free_space(
            max_height=robot_height,
            bloat_obstacles=base_radius + iai_office_room_navigation.obstacle_clearance,
        )
        for room in rooms:
            if Level.under(room.floor.root) is not level:
                continue
            waiting_spot = Point2.from_pose(
                world.transform(
                    elevator.pose_in_front_of_doors(
                        NavigateToRoomAction(room).elevator_waiting_distance
                    ),
                    world.root,
                )
            )
            centre = room.floor_centre
            if not all(
                room.floor.covers(
                    Point3(
                        float(centre.x) + x,
                        float(centre.y) + y,
                        float(centre.z),
                        reference_frame=world.root,
                    )
                )
                for x, y in footprint_offsets
            ):
                rooms_off_their_floor.append(room.name.name)
            room_spot = Point2.from_pose(centre)
            free_space.path_from_to(waiting_spot, room_spot)
            free_space.path_from_to(room_spot, waiting_spot)

    assert rooms_off_their_floor == []


def test_tiago_drives_between_every_pair_of_floors_in_one_world(
    rclpy_node, iai_office, iai_office_room_navigation
):
    """
    Each trip after the first starts on a floor the cabin does not serve, so Tiago has
    to call it.
    """
    world = iai_office_room_navigation.acquire_world()
    room_names = {
        storey: iai_office.get_room(
            world, iai_office_room_navigation.building_name, storey, number
        ).name.name
        for storey, number in (
            (Storey.GROUND_FLOOR, "0.31"),
            (Storey.FIRST_FLOOR, "1.54"),
            (Storey.SECOND_FLOOR, "2.59"),
        )
    }
    trips = [
        RoomTrip(room_names[Storey.GROUND_FLOOR], room_names[Storey.FIRST_FLOOR]),
        RoomTrip(room_names[Storey.SECOND_FLOOR], room_names[Storey.GROUND_FLOOR]),
        RoomTrip(room_names[Storey.FIRST_FLOOR], room_names[Storey.SECOND_FLOOR]),
    ]
    trials = RoomNavigationTrials(demonstration=iai_office_room_navigation)

    results = [trials.perform_trial(world, trip) for trip in trips]

    assert [result.failure for result in results] == [None] * len(trips)


# %% failing trials


def test_a_failing_trial_is_recorded_instead_of_raised(lobby_room_navigation):
    world = lobby_room_navigation.build_simulated_world()
    trip = RoomTrip(lobby_room_navigation.start_room_name, "no_such_room")

    result = RoomNavigationTrials(demonstration=lobby_room_navigation).perform_trial(
        world, trip
    )

    assert result.trip is trip
    assert isinstance(result.failure, WorldEntityNotFoundError)


# %% overview of the trials


@pytest.fixture
def summary_with_one_failure() -> RoomNavigationTrialSummary:
    return RoomNavigationTrialSummary(
        [
            RoomNavigationTrialResult(RoomTrip("lobby", "office")),
            RoomNavigationTrialResult(
                RoomTrip("lobby", "landing"), TargetRoomNotReachedError("landing")
            ),
            RoomNavigationTrialResult(RoomTrip("office", "landing")),
        ]
    )


def test_summary_counts_successes_and_failures(summary_with_one_failure):
    assert summary_with_one_failure.number_of_successes == 2
    assert summary_with_one_failure.number_of_failures == 1


def test_summary_groups_failures_by_type(summary_with_one_failure):
    assert summary_with_one_failure.failures_by_type == Counter(
        {TargetRoomNotReachedError.__name__: 1}
    )


def test_summary_groups_trials_by_target_room(summary_with_one_failure):
    office, failed_landing, landing = summary_with_one_failure.results
    assert summary_with_one_failure.trials_by_target_room == {
        "office": [office],
        "landing": [failed_landing, landing],
    }


def test_report_names_each_failed_trial_with_why_it_failed(summary_with_one_failure):
    assert (
        f"#2 from lobby to landing: {TargetRoomNotReachedError.__name__}"
        in summary_with_one_failure.report()
    )
