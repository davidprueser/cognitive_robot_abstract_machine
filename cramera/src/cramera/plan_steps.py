"""
A plan as the Plan Builder writes it: the ordered steps one robot performs.

The builder hands its steps over as data rather than as the Python it also generates, so
what a plan can ask for stays bounded, and a step naming an arm the robot has not got is
refused when it is read instead of failing inside a motion. Each step reads itself from
the builder's form of it and writes itself back, and turns into the coraplex action
carrying it out.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from cramera.model_catalog import BuilderStep
from typing_extensions import Any, ClassVar, Dict, List, Type

# %% the builder's form of a step


class StepField(StrEnum):
    """
    The keys a step is written with in the builder's form of it.
    """

    TYPE = "type"
    PARAMETERS = "params"


class StepParameter(StrEnum):
    """
    The parameters a step's builder form names.
    """

    ARM = "arm"
    TORSO = "torso"
    X = "x"
    Y = "y"
    Z = "z"
    YAW = "yaw"


class MalformedPlanError(Exception):
    """
    Raised when the builder's form of a plan cannot be read.
    """


def _number(parameters: Dict[str, Any], key: StepParameter) -> float:
    """
    One finite coordinate of a step.

    :param parameters: The step's parameters.
    :param key: The parameter to read.
    :raises MalformedPlanError: If the value is missing or not a finite number.
    """
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedPlanError(f"{key!r} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise MalformedPlanError(f"{key!r} must be finite, got {value!r}")
    return float(value)


def _member(enumeration: Type, parameters: Dict[str, Any], key: StepParameter) -> Any:
    """
    The enum member a step names.

    :param enumeration: The enumeration the name has to belong to.
    :param parameters: The step's parameters.
    :param key: The parameter holding the member's name.
    :raises MalformedPlanError: If the name is not one of the enumeration's members.
    """
    name = parameters.get(key)
    if name not in enumeration.__members__:
        raise MalformedPlanError(
            f"{key!r} must be one of {list(enumeration.__members__)}, got {name!r}"
        )
    return enumeration[name]


# %% places a step names


@dataclass(frozen=True)
class Point:
    """
    A point in the world's frame, as the builder's scene lets one be set.
    """

    x: float
    """
    Position along the world's x axis, in metres.
    """

    y: float
    """
    Position along the world's y axis, in metres.
    """

    z: float
    """
    Height above the world's origin, in metres.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Point:
        """
        :param parameters: A step's parameters.
        :return: The point they name.
        :raises MalformedPlanError: If a coordinate is missing or unusable.
        """
        return cls(
            x=_number(parameters, StepParameter.X),
            y=_number(parameters, StepParameter.Y),
            z=_number(parameters, StepParameter.Z),
        )

    def to_parameters(self) -> Dict[str, float]:
        """
        :return: The point in a step's parameters.
        """
        return {
            StepParameter.X: self.x,
            StepParameter.Y: self.y,
            StepParameter.Z: self.z,
        }

    def pose(self, world: World) -> Pose:
        """
        :param world: The world the point is expressed in.
        :return: A pose at the point, with no turn.
        """
        return Pose.from_xyz_rpy(self.x, self.y, self.z, reference_frame=world.root)


@dataclass(frozen=True)
class LevelPose(Point):
    """
    A pose given as a place and a heading, with no roll or pitch.
    """

    yaw: float = 0.0
    """
    Heading about the vertical axis, in radians.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> LevelPose:
        point = Point.from_parameters(parameters)
        return cls(
            x=point.x,
            y=point.y,
            z=point.z,
            yaw=_number(parameters, StepParameter.YAW),
        )

    def to_parameters(self) -> Dict[str, float]:
        return {**super().to_parameters(), StepParameter.YAW: self.yaw}

    def pose(self, world: World) -> Pose:
        return Pose.from_xyz_rpy(
            self.x, self.y, self.z, yaw=self.yaw, reference_frame=world.root
        )


# %% the steps themselves


@dataclass(frozen=True)
class PlanStep(ABC):
    """
    One step of a plan the builder wrote.
    """

    STEP_TYPE: ClassVar[BuilderStep]
    """
    What the step is called in the builder's form of it.
    """

    @classmethod
    @abstractmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> PlanStep:
        """
        Read the step off its parameters.

        :param parameters: The step's parameters.
        :raises MalformedPlanError: If a parameter is missing or unusable.
        """

    @abstractmethod
    def to_parameters(self) -> Dict[str, Any]:
        """
        :return: The step's parameters, as :meth:`from_parameters` reads them.
        """

    @abstractmethod
    def action(self, context: Context) -> ActionDescription:
        """
        The coraplex action carrying this step out.

        :param context: The running scene, whose world the step's poses are resolved in.
        """

    def to_payload(self) -> Dict[str, Any]:
        """
        :return: The builder's form of the step.
        """
        return {
            StepField.TYPE: self.STEP_TYPE.value,
            StepField.PARAMETERS: self.to_parameters(),
        }


@dataclass(frozen=True)
class ParkArms(PlanStep):
    """
    Bring an arm back to its parked pose.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.PARK_ARMS

    arm: Arms
    """
    The arm to park.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> ParkArms:
        return cls(arm=_member(Arms, parameters, StepParameter.ARM))

    def to_parameters(self) -> Dict[str, Any]:
        return {StepParameter.ARM: self.arm.name}

    def action(self, context: Context) -> ActionDescription:
        return ParkArmsAction(self.arm)


@dataclass(frozen=True)
class MoveTorso(PlanStep):
    """
    Raise or lower the torso.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.MOVE_TORSO

    torso_state: TorsoState
    """
    The height the torso is moved to.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> MoveTorso:
        return cls(torso_state=_member(TorsoState, parameters, StepParameter.TORSO))

    def to_parameters(self) -> Dict[str, Any]:
        return {StepParameter.TORSO: self.torso_state.name}

    def action(self, context: Context) -> ActionDescription:
        return MoveTorsoAction(self.torso_state)


@dataclass(frozen=True)
class Navigate(PlanStep):
    """
    Drive the robot's base somewhere.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.NAVIGATE

    target: LevelPose
    """
    Where the robot drives to, and which way it ends up facing.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> Navigate:
        return cls(target=LevelPose.from_parameters(parameters))

    def to_parameters(self) -> Dict[str, Any]:
        return self.target.to_parameters()

    def action(self, context: Context) -> ActionDescription:
        return NavigateAction(self.target.pose(context.world))


@dataclass(frozen=True)
class LookAt(PlanStep):
    """
    Point the robot's default camera at a point.
    """

    STEP_TYPE: ClassVar[BuilderStep] = BuilderStep.LOOK_AT

    target: Point
    """
    The point looked at.
    """

    @classmethod
    def from_parameters(cls, parameters: Dict[str, Any]) -> LookAt:
        return cls(target=Point.from_parameters(parameters))

    def to_parameters(self) -> Dict[str, Any]:
        return self.target.to_parameters()

    def action(self, context: Context) -> ActionDescription:
        return LookAtAction(self.target.pose(context.world))


# %% a whole plan


@dataclass(frozen=True)
class ObjectStepInPlanError(MalformedPlanError):
    """
    Raised for a step that acts on a carried object, which a plan read on its own does
    not have.
    """

    step_type: BuilderStep
    """
    The step that was asked for.
    """

    def __str__(self) -> str:
        return (
            f"a {self.step_type} step acts on an object, and this plan carries none;"
            f" readable are {[kind.STEP_TYPE.value for kind in STEP_KINDS]}"
        )


STEP_KINDS: List[Type[PlanStep]] = [ParkArms, MoveTorso, Navigate, LookAt]
"""
The steps a plan read on its own can hold.
"""


@dataclass(frozen=True)
class BuilderPlan:
    """
    The steps one robot performs, in order.
    """

    steps: List[PlanStep] = field(default_factory=list)
    """
    The steps to perform, in order.
    """

    @classmethod
    def from_payload(cls, payload: List[Any]) -> BuilderPlan:
        """
        :param payload: The builder's form of the steps.
        :return: The plan they describe.
        :raises MalformedPlanError: If the plan or one of its steps is unusable.
        """
        if not isinstance(payload, list):
            raise MalformedPlanError("a plan is a list of steps")
        return cls(steps=[cls._step(entry) for entry in payload])

    @staticmethod
    def _step(entry: Any) -> PlanStep:
        """
        :param entry: One step in the builder's form.
        :return: The step it describes.
        :raises MalformedPlanError: If the entry names no step a plan can hold.
        """
        if not isinstance(entry, dict):
            raise MalformedPlanError("every step must be an object")
        named = entry.get(StepField.TYPE)
        if named not in BuilderStep.__members__.values():
            raise MalformedPlanError(
                f"{StepField.TYPE!r} must be one of {[kind.value for kind in BuilderStep]},"
                f" got {named!r}"
            )
        by_type = {kind.STEP_TYPE: kind for kind in STEP_KINDS}
        if BuilderStep(named) not in by_type:
            raise ObjectStepInPlanError(BuilderStep(named))
        parameters = entry.get(StepField.PARAMETERS) or {}
        if not isinstance(parameters, dict):
            raise MalformedPlanError(f"{StepField.PARAMETERS!r} must be an object")
        return by_type[BuilderStep(named)].from_parameters(parameters)

    def to_payload(self) -> List[Dict[str, Any]]:
        """
        :return: The builder's form of the steps.
        """
        return [step.to_payload() for step in self.steps]

    def plan(self, context: Context) -> PlanNode:
        """
        :param context: The running scene's context, whose world the steps resolve in.
        :return: The coraplex plan performing these steps.
        """
        actions = [step.action(context) for step in self.steps]
        return sequential(actions, context=context).plan
