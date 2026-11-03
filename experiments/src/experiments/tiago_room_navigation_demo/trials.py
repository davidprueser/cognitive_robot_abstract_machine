"""
Repeated trials of :class:`TiagoRoomNavigationDemonstration` between rooms drawn from an
office, each to a room on another level than the one it starts on, with an overview of
how many succeeded and why the others failed.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field

from coraplex.datastructures.enums import ExecutionType
from experiments.tiago_room_navigation_demo.demo import (
    TiagoIAIOfficeRoomNavigationDemonstration,
    TiagoRoomNavigationDemonstration,
)
from semantic_digital_twin.predefined_maps.elevator_building import ElevatorBuilding
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Level,
    Room,
)
from semantic_digital_twin.world import World

# %% trips


@dataclass
class RoomTrip:
    """
    A drive from one room to another.
    """

    start_room_name: str
    """
    The name of the room the robot starts in.
    """

    target_room_name: str
    """
    The name of the room the robot is sent to.
    """


@dataclass
class RoomTripDraw:
    """
    Draws trips between the rooms of an office a robot may be sent into, each to a room
    on another level than the one it starts on.

    Only rooms annotated as nothing more specific than a
    :class:`~semantic_digital_twin.semantic_annotations.semantic_annotations.Room` are
    drawn.
    """

    office: ElevatorBuilding
    """
    The office the rooms are drawn from.
    """

    building_name: str
    """
    The prefix the office is spawned with.
    """

    number_of_trips: int = 50
    """
    How many trips to draw.
    """

    seed: int | None = None
    """
    Seed of the draw, or None to draw differently each time.
    """

    def room_levels(self, world: World) -> dict[str, str]:
        """
        :param world: The world the office was spawned into.
        :return: The name of the level each room that may be drawn lies on, by the
            room's name.
        """
        return {
            room.name.name: Level.under(room.floor.root).name.name
            for room in self.office.navigation_targets(world, self.building_name)
            if type(room) is Room
        }

    def trips(self, world: World) -> list[RoomTrip]:
        """
        :param world: The world the office was spawned into.
        :return: :attr:`number_of_trips` trips, each starting in a room drawn uniformly
            from every room and ending in a room drawn uniformly from the rooms on the
            other levels.
        """
        room_levels = self.room_levels(world)
        room_names = sorted(room_levels)
        sampler = random.Random(self.seed)
        trips = []
        for _ in range(self.number_of_trips):
            start_room_name = sampler.choice(room_names)
            target_room_name = sampler.choice(
                [
                    room_name
                    for room_name in room_names
                    if room_levels[room_name] != room_levels[start_room_name]
                ]
            )
            trips.append(RoomTrip(start_room_name, target_room_name))
        return trips


# %% outcomes


@dataclass
class RoomNavigationTrialResult:
    """
    The outcome of one trip.
    """

    trip: RoomTrip
    """
    The trip the robot was sent on.
    """

    failure: Exception | None = None
    """
    What made the trial fail, or None if the robot reached the target room.
    """

    @property
    def succeeded(self) -> bool:
        """
        Whether the robot reached the target room.
        """
        return self.failure is None


@dataclass
class RoomNavigationTrialSummary:
    """
    The outcomes of a series of trips.
    """

    results: list[RoomNavigationTrialResult] = field(default_factory=list)
    """
    The outcome of each trial, in the order the trials ran.
    """

    @property
    def number_of_successes(self) -> int:
        """
        How many trials reached their target room.
        """
        return sum(result.succeeded for result in self.results)

    @property
    def number_of_failures(self) -> int:
        """
        How many trials did not reach their target room.
        """
        return len(self.results) - self.number_of_successes

    @property
    def failures_by_type(self) -> Counter[str]:
        """
        How many trials failed with each type of failure, by the type's name.
        """
        return Counter(
            type(result.failure).__name__
            for result in self.results
            if not result.succeeded
        )

    @property
    def trials_by_target_room(self) -> dict[str, list[RoomNavigationTrialResult]]:
        """
        The trials to each target room, by the room's name.
        """
        trials = {}
        for result in self.results:
            trials.setdefault(result.trip.target_room_name, []).append(result)
        return trials

    def report(self) -> str:
        """
        :return: A human-readable overview of how many trials succeeded, per target
            room and in total, and why the others failed.
        """
        lines = [
            f"Room navigation: {len(self.results)} trials, "
            f"{self.number_of_successes} succeeded, {self.number_of_failures} failed.",
            "Per target room:",
        ]
        for room_name, trials in sorted(self.trials_by_target_room.items()):
            successes = sum(trial.succeeded for trial in trials)
            lines.append(f"  {room_name}: {successes}/{len(trials)} succeeded")
        if self.number_of_failures:
            lines.append("Failures by type:")
            for failure_type, count in self.failures_by_type.most_common():
                lines.append(f"  {failure_type}: {count}")
            lines.append("Failed trials:")
            for number, result in enumerate(self.results, start=1):
                if result.succeeded:
                    continue
                message_lines = str(result.failure).splitlines() or [""]
                lines.append(
                    f"  #{number} from {result.trip.start_room_name} to "
                    f"{result.trip.target_room_name}: "
                    f"{type(result.failure).__name__}: {message_lines[0]}"
                )
        return "\n".join(lines)


# %% trials


@dataclass
class RoomNavigationTrials:
    """
    Sends the robot of a demonstration on trips drawn from its office, one after
    another, all in the one world the demonstration acquires.
    """

    demonstration: TiagoRoomNavigationDemonstration
    """
    The demonstration that drives each trip; its start and target room are set per trip.
    """

    number_of_trials: int = 50
    """
    How many trips to drive.
    """

    seed: int | None = None
    """
    Seed of the draw of the trips, or None to draw differently each time.
    """

    def run(self) -> RoomNavigationTrialSummary:
        """
        Draw the trips and drive every one, carrying on after a failed one.

        :return: The outcomes of the trials.
        """
        try:
            world = self.demonstration.acquire_world()
            if not self.demonstration.is_scene_populated(world):
                self.demonstration.populate_scene(world)
            trips = RoomTripDraw(
                office=self.demonstration.office,
                building_name=self.demonstration.building_name,
                number_of_trips=self.number_of_trials,
                seed=self.seed,
            ).trips(world)
            return RoomNavigationTrialSummary(
                [self.perform_trial(world, trip) for trip in trips]
            )
        finally:
            self.demonstration.tear_down()

    def perform_trial(self, world: World, trip: RoomTrip) -> RoomNavigationTrialResult:
        """
        Drive the robot from the start to the target room of ``trip`` in ``world``.

        .. note:: Any exception the trial raises fails the trial instead of the series.

        :param world: The world the demonstration acquired and populated.
        :param trip: The trip to drive.
        :return: The outcome of the drive.
        """
        self.demonstration.start_room_name = trip.start_room_name
        self.demonstration.target_room_name = trip.target_room_name
        try:
            self.demonstration.perform(world)
        except Exception as failure:
            logging.getLogger(__name__).exception(
                f"The trial from {trip.start_room_name} to {trip.target_room_name} "
                f"failed."
            )
            return RoomNavigationTrialResult(trip, failure)
        return RoomNavigationTrialResult(trip)


def main(
    number_of_trials: int = 50,
    seed: int | None = None,
    execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED,
) -> None:
    """
    Drive Tiago on trips drawn from the IAI's office and print their overview.

    :param number_of_trials: How many trips to drive.
    :param seed: Seed of the draw of the trips.
    :param execution_type: Whether to drive the real (or Gazebo) robot or simulate it.
    """
    summary = RoomNavigationTrials(
        demonstration=TiagoIAIOfficeRoomNavigationDemonstration(
            used_robot=Tiago, execution_type=execution_type
        ),
        number_of_trials=number_of_trials,
        seed=seed,
    ).run()
    print(summary.report())


if __name__ == "__main__":
    main()
