"""
A plan as the Plan Builder writes it: reading it from the builder's form, writing it
back, and turning it into the coraplex actions that carry it out.
"""

from __future__ import annotations

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState

from cramera.model_catalog import BuilderStep
from cramera.plan_steps import (
    BuilderPlan,
    LookAt,
    MalformedPlanError,
    MoveTorso,
    ObjectStepInPlanError,
    ParkArms,
    StepField,
)

from .test_live_bridge import world_with


def context_on(world) -> Context:
    """
    The context a step is resolved against, for a scene with no robot in it.
    """
    return Context(world=world, robot=None)


def step(step_type: BuilderStep, **parameters) -> dict:
    """
    One step in the builder's form.
    """
    return {StepField.TYPE: step_type.value, StepField.PARAMETERS: parameters}


def read(*steps) -> BuilderPlan:
    return BuilderPlan.from_payload(list(steps))


# %% reading a plan


class TestReadingAPlan:
    def test_a_plan_keeps_the_order_its_steps_were_written_in(self):
        plan = read(
            step(BuilderStep.PARK_ARMS, arm="BOTH"),
            step(BuilderStep.MOVE_TORSO, torso="HIGH"),
        )
        assert [type(found) for found in plan.steps] == [ParkArms, MoveTorso]
        assert plan.steps[0].arm is Arms.BOTH
        assert plan.steps[1].torso_state is TorsoState.HIGH

    def test_a_plan_without_steps_is_a_plan_of_nothing(self):
        assert BuilderPlan.from_payload([]).steps == []

    def test_a_step_of_an_unknown_type_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read({StepField.TYPE: "make_coffee", StepField.PARAMETERS: {}})

    def test_a_step_acting_on_an_object_is_refused(self):
        with pytest.raises(ObjectStepInPlanError):
            read(step(BuilderStep.TRANSPORT, object="milk.stl", arm="LEFT"))

    def test_an_arm_the_robot_does_not_have_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.PARK_ARMS, arm="THIRD"))

    def test_a_torso_state_that_is_not_one_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.MOVE_TORSO, torso="SLIGHTLY_UP"))

    def test_a_coordinate_that_is_not_a_number_is_refused(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.NAVIGATE, x="over there", y=1.0, z=0.0, yaw=0.0))

    def test_a_point_to_look_at_needs_all_three_coordinates(self):
        with pytest.raises(MalformedPlanError):
            read(step(BuilderStep.LOOK_AT, x=1.0, y=2.0))


# %% writing a plan back


@pytest.mark.parametrize(
    "written",
    [
        step(BuilderStep.PARK_ARMS, arm="LEFT"),
        step(BuilderStep.MOVE_TORSO, torso="LOW"),
        step(BuilderStep.NAVIGATE, x=2.6, y=1.8, z=0.0, yaw=1.57),
        step(BuilderStep.LOOK_AT, x=1.0, y=-0.5, z=1.2),
    ],
)
def test_a_step_reads_back_as_it_was_written(written):
    plan = read(written)

    assert read(*plan.to_payload()) == plan


# %% the actions a plan performs


class TestActionsAPlanPerforms:
    def test_parking_arms_parks_the_named_arm(self):
        action = (
            read(step(BuilderStep.PARK_ARMS, arm="LEFT"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, ParkArmsAction)
        assert action.arm is Arms.LEFT

    def test_moving_the_torso_moves_it_to_the_named_state(self):
        action = (
            read(step(BuilderStep.MOVE_TORSO, torso="LOW"))
            .steps[0]
            .action(context_on(world_with()))
        )
        assert isinstance(action, MoveTorsoAction)
        assert action.torso_state is TorsoState.LOW

    def test_navigating_goes_to_the_given_place(self):
        world = world_with()
        action = (
            read(step(BuilderStep.NAVIGATE, x=2.6, y=1.8, z=0.0, yaw=1.57))
            .steps[0]
            .action(context_on(world))
        )
        assert isinstance(action, NavigateAction)
        assert action.target_location.to_position().to_np()[:3] == pytest.approx(
            [2.6, 1.8, 0.0]
        )

    def test_looking_at_a_point_aims_at_that_point(self):
        world = world_with()
        [look] = read(step(BuilderStep.LOOK_AT, x=1.0, y=-0.5, z=1.2)).steps
        action = look.action(context_on(world))

        assert isinstance(action, LookAtAction)
        assert isinstance(look, LookAt)
        assert action.target.to_position().to_np()[:3] == pytest.approx(
            [look.target.x, look.target.y, look.target.z]
        )
        assert action.target.reference_frame is world.root
