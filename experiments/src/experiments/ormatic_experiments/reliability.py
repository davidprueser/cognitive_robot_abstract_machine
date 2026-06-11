import pathlib
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import plotly.graph_objects as go
import tqdm
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

import pycram.orm.ormatic_interface  # type: ignore  # noqa: F401
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from pycram.datastructures.dataclasses import Context
from pycram.motion_executor import simulated_robot
from pycram.orm.ormatic_interface import Base, PlanMappingDAO  # type: ignore
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

_REPO_ROOT = pathlib.Path(__file__).parents[4]
_APARTMENT_URDF = _REPO_ROOT / "pycram" / "resources" / "worlds" / "apartment.urdf"


def _build_pr2_world() -> World:
    urdf_parser = URDFParser.from_file(file_path=PR2.get_ros_file_path())
    world = urdf_parser.parse()
    PR2.from_world(world)

    with world.modify_world():
        old_root = world.root
        map_body = Body(name=PrefixedName("map"))
        odom = Body(name=PrefixedName("odom_combined"))
        map_C_odom = Connection6DoF.create_with_dofs(world, map_body, odom)
        world.add_connection(map_C_odom)
        drive = OmniDrive.create_with_dofs(parent=odom, child=old_root, world=world)
        world.add_connection(drive)
        drive.has_hardware_interface = True

    return world


def build_cram_world():
    """Build a PR2+apartment world and return ``(world, pr2, context)``."""
    pr2_world = _build_pr2_world()
    apartment = URDFParser.from_file(str(_APARTMENT_URDF)).parse()
    pr2_world.merge_world(apartment)
    pr2_world.get_body_by_name("base_footprint").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.3, 2, 0)
    )
    pr2 = pr2_world.get_semantic_annotations_by_type(PR2)[0]
    ctx = Context(
        pr2_world, pr2, evaluate_conditions=False, query_backend=ProbabilisticBackend()
    )
    return pr2_world, pr2, ctx


_MAX_TRIES = 10


def _random_underspecified_action(world: World):
    """Return a random underspecified action limited to at most ``_MAX_TRIES`` attempts."""
    choice = random.randint(0, 2)
    if choice == 0:
        action = underspecified(NavigateAction)(
            target_location=underspecified(Pose.from_xyz_rpy)(
                x=...,
                y=...,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=...,
                reference_frame=world.root,
            ),
            keep_joint_states=True,
        )
    elif choice == 1:
        action = underspecified(MoveTorsoAction)(torso_state=...)
    else:
        action = underspecified(ParkArmsAction)(arm=...)
    action.expression.limit(_MAX_TRIES)
    return action


def create_random_plan(world: World, ctx: Context, n_actions: int):
    """Create a sequential plan with *n_actions* randomly chosen underspecified actions."""
    actions = [_random_underspecified_action(world) for _ in range(n_actions)]
    return sequential(actions, context=ctx).plan


@dataclass
class ORMaticReliabilityExperimentResult(ExperimentResult):
    """Raw timings from a single plan-creation, execution and DB-write cycle."""

    plan_size: int
    """Number of actions in the plan."""
    world_building_duration: float
    """Seconds to build the PR2+apartment world."""
    plan_execution_duration: float
    """Seconds to execute the plan under simulated_robot."""
    to_dao_duration: float
    """Seconds for to_dao() to serialise the plan to a DAO."""
    writing_to_database_duration: float
    """Combined seconds for session.add(dao) + session.commit()."""
    reading_from_database_duration: float
    """Seconds for session.scalars(select(PlanMappingDAO)).one()."""
    reconstruction_duration: float
    """Seconds for fetched_plan.from_dao()."""


@dataclass
class ORMaticReliabilityAggregateResult(ExperimentResult):
    """Aggregated timings across multiple reliability experiment runs."""

    plan_size: int
    """Number of actions in each plan in this aggregate."""
    world_building_duration: MeanAndStandardDeviation
    plan_execution_duration: MeanAndStandardDeviation
    to_dao_duration: MeanAndStandardDeviation
    writing_to_database_duration: MeanAndStandardDeviation
    reading_from_database_duration: MeanAndStandardDeviation
    reconstruction_duration: MeanAndStandardDeviation


def reliability_experiment(
    plan_size: int,
    world: World,
    pr2,
    ctx: Context,
    world_building_duration: float,
) -> ORMaticReliabilityExperimentResult:
    """
    Run a single reliability iteration: create and execute a plan, then write it
    to an in-memory SQLite database, measuring each phase separately.

    :param plan_size: Number of actions to include in the random plan.
    :param world: The pre-built world to create the plan in.
    :param pr2: The PR2 robot view.
    :param ctx: The execution context.
    :param world_building_duration: Pre-measured world building time (s).
    :return: Timing breakdown for this single run.
    """
    plan = create_random_plan(world, ctx, plan_size)

    t0 = time.perf_counter()
    with simulated_robot:
        plan.perform()
    plan_execution_duration = time.perf_counter() - t0

    t0 = time.perf_counter()
    dao = to_dao(plan)
    to_dao_duration = time.perf_counter() - t0

    engine = create_engine("sqlite:///:memory:")
    session = sessionmaker(engine)()
    Base.metadata.create_all(bind=session.bind)

    t0 = time.perf_counter()
    session.add(dao)
    session.commit()
    writing_to_database_duration = time.perf_counter() - t0

    t0 = time.perf_counter()
    fetched = session.scalars(select(PlanMappingDAO)).one()
    reading_from_database_duration = time.perf_counter() - t0

    t0 = time.perf_counter()
    fetched.from_dao()
    reconstruction_duration = time.perf_counter() - t0

    drop_database(session.bind)
    session.close()
    engine.dispose()

    return ORMaticReliabilityExperimentResult(
        plan_size=plan_size,
        world_building_duration=round(world_building_duration, 3),
        plan_execution_duration=round(plan_execution_duration, 3),
        to_dao_duration=round(to_dao_duration, 3),
        writing_to_database_duration=round(writing_to_database_duration, 3),
        reading_from_database_duration=round(reading_from_database_duration, 3),
        reconstruction_duration=round(reconstruction_duration, 3),
    )


def run_reliability_experiment(
    plan_size: int,
    iterations: int = 10,
) -> tuple[ORMaticReliabilityAggregateResult, List[ORMaticReliabilityExperimentResult]]:
    """
    Build the world once, then run *iterations* reliability experiment cycles
    for plans of *plan_size* actions and aggregate the results.

    :param plan_size: Number of actions per plan.
    :param iterations: Number of independent iterations to aggregate.
    :return: Tuple of (aggregate result, raw per-iteration results).
    """
    t0 = time.perf_counter()
    world, pr2, ctx = build_cram_world()
    world_building_duration = time.perf_counter() - t0

    raw = []
    for _ in range(iterations):
        state = deepcopy(world.state._data)
        result = reliability_experiment(
            plan_size, world, pr2, ctx, world_building_duration
        )
        world.state._data[:] = state
        world.notify_state_change()
        raw.append(result)

    aggregate = ORMaticReliabilityAggregateResult(
        plan_size=plan_size,
        world_building_duration=MeanAndStandardDeviation.from_measurements(
            [r.world_building_duration for r in raw]
        ),
        plan_execution_duration=MeanAndStandardDeviation.from_measurements(
            [r.plan_execution_duration for r in raw]
        ),
        to_dao_duration=MeanAndStandardDeviation.from_measurements(
            [r.to_dao_duration for r in raw]
        ),
        writing_to_database_duration=MeanAndStandardDeviation.from_measurements(
            [r.writing_to_database_duration for r in raw]
        ),
        reading_from_database_duration=MeanAndStandardDeviation.from_measurements(
            [r.reading_from_database_duration for r in raw]
        ),
        reconstruction_duration=MeanAndStandardDeviation.from_measurements(
            [r.reconstruction_duration for r in raw]
        ),
    )
    return aggregate, raw


def plot_reliability(
    raw_results: List[ORMaticReliabilityExperimentResult],
) -> go.Figure:
    """
    Produce a grouped violin plot of plan size vs duration for each timing phase.

    :param raw_results: All per-iteration :class:`ORMaticReliabilityExperimentResult`
                        instances across every plan size.
    :return: A Plotly figure ready for display or export.
    """
    phases = [
        ("plan_execution_duration", "Plan Execution"),
        ("to_dao_duration", "to_dao()"),
        ("writing_to_database_duration", "Write to DB"),
        ("reading_from_database_duration", "Read from DB"),
        ("reconstruction_duration", "from_dao()"),
    ]

    traces = []
    for field, label in phases:
        traces.append(
            go.Violin(
                x=[str(r.plan_size) for r in raw_results],
                y=[getattr(r, field) for r in raw_results],
                name=label,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        violinmode="group",
        xaxis_title="Plan Size (actions)",
        yaxis_title="Duration (s)",
        title="ORMatic Reliability: Plan Size vs Timing",
    )
    return fig


def main():
    aggregate_results = []
    all_raw: List[ORMaticReliabilityExperimentResult] = []
    for plan_size in tqdm.tqdm(
        [
            1,
            3,
            5,
            10,
            20,
        ]
    ):
        aggregate, raw = run_reliability_experiment(plan_size, iterations=10)
        aggregate_results.append(aggregate)
        all_raw.extend(raw)

    table = ExperimentsTable(aggregate_results)
    print(TypstRenderer(table).render_table())
    plot_reliability(all_raw).show()


if __name__ == "__main__":
    main()
