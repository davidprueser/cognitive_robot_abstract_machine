"""Plan-builder models and capabilities from CRAM's semantic annotations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path

from typing_extensions import Any, ClassVar, Protocol, get_args

from cramera.payload import CrameraPayload
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.robots.armar7 import Armar7
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.icub3 import ICub3
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.mmp_dresden import MMPDresden
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_part_mixins import (
    HasLeftRightArm,
    HasMobileBase,
)
from semantic_digital_twin.robots.robot_parts import (
    Camera,
    AbstractRobot,
    AbstractRobotPart,
    Arm,
    Torso,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.world import World

ROBOT_ENTRY_POINT_GROUP = "cramera.robots"
"""
The entry point group another installed distribution registers its robot annotations
under, so the plan builder offers robots cram does not ship.
"""


# %% frontend vocabulary


class BuilderStep(StrEnum):
    """Operations offered by the plan builder."""

    PARK_ARMS = "park_arms"
    """Move an arm to its park state."""
    MOVE_TORSO = "move_torso"
    """Move a torso to a named height."""
    NAVIGATE = "navigate"
    """Drive a mobile base to a pose."""
    LOOK_AT = "look_at"
    """Point the default camera at a point."""
    DETECT = "detect"
    """Look for an object of a type and take its pose from perception."""
    TRANSPORT = "transport"
    """Pick up and place an object."""
    PICK = "pick"
    """Pick up an object."""
    PLACE = "place"
    """Place a held object."""


class CatalogField(StrEnum):
    """Keys of the model catalog's browser payload."""

    OK = "ok"
    """Whether the catalog request succeeded."""
    NAME = "name"
    """Human-readable model name."""
    CLASS = "cls"
    """Concrete robot annotation class name."""
    IMPORT = "import"
    """Import statement for the annotation."""
    STEPS = "steps"
    """Supported builder operations."""
    ARMS = "arms"
    """Supported coraplex arm selections."""
    PATH = "path"
    """Local environment description path."""
    KIND = "kind"
    """How an environment comes into a world, an :class:`EnvironmentKind`."""
    ROBOTS = "robots"
    """Registered robot choices."""
    ENVIRONMENTS = "environments"
    """Installed environment choices."""
    MAPS = "maps"
    """Environment choices built by a map class."""
    OBJECTS = "objects"
    """Object choices carrying an annotation class."""
    MESH = "mesh"
    """An object's mesh file name, which names its body."""
    MESH_URL = "mesh_url"
    """The package URL an object's mesh is resolved from."""


class EnvironmentKind(StrEnum):
    """How a builder environment comes into a world."""

    FILE = "file"
    """A world description file coraplex ships, read by the parser its suffix names."""
    MAP = "map"
    """A map class that spawns its furniture into a world, such as the real lab."""


class BuilderArm(StrEnum):
    """Arm selections understood by coraplex actions."""

    LEFT = "LEFT"
    """The robot's left arm."""
    RIGHT = "RIGHT"
    """The robot's right arm."""
    BOTH = "BOTH"
    """Both arms, or the sole available arm."""


# %% robot capabilities


@dataclass
class RobotModel:
    """An annotated robot and the operations its part structure supports."""

    annotation: type[AbstractRobot]
    """The CRAM annotation that owns the robot's description and parts."""

    def part_types(self) -> set[type[AbstractRobot | AbstractRobotPart]]:
        """Collect nested part annotations through CRAM's dataclass introspector.

        :return: Robot and part classes declared by the annotation.
        """
        pending = [self.annotation]
        result = set()
        introspector = DataclassOnlyIntrospector()
        while pending:
            candidate = pending.pop()
            arguments = get_args(candidate)
            if arguments:
                pending.extend(arguments)
                continue
            if (
                not isinstance(candidate, type)
                or not issubclass(candidate, (AbstractRobot, AbstractRobotPart))
                or candidate in result
            ):
                continue
            result.add(candidate)
            pending.extend(
                attribute.field.type for attribute in introspector.discover(candidate)
            )
        return result

    @property
    def steps(self) -> list[BuilderStep]:
        """Return operations supported by the robot's declared semantic parts."""
        parts = self.part_types()
        supported = []
        if any(issubclass(part, Arm) for part in parts):
            supported.extend(
                [
                    BuilderStep.PARK_ARMS,
                    BuilderStep.TRANSPORT,
                    BuilderStep.PICK,
                    BuilderStep.PLACE,
                ]
            )
        if any(issubclass(part, Torso) for part in parts):
            supported.append(BuilderStep.MOVE_TORSO)
        if issubclass(self.annotation, HasMobileBase):
            supported.append(BuilderStep.NAVIGATE)
        if any(issubclass(part, Camera) for part in parts):
            supported.extend([BuilderStep.LOOK_AT, BuilderStep.DETECT])
        return [step for step in BuilderStep if step in supported]

    @property
    def arms(self) -> list[BuilderArm]:
        """Return named arms, or coraplex's default selection for a single arm."""
        if any(issubclass(part, HasLeftRightArm) for part in self.part_types()):
            return list(BuilderArm)
        return [BuilderArm.BOTH] if BuilderStep.PICK in self.steps else []

    def to_payload(self) -> dict[str, Any]:
        """Return the browser's import and capability information."""
        return {
            CatalogField.NAME: self.annotation.__name__,
            CatalogField.CLASS: self.annotation.__name__,
            CatalogField.IMPORT: f"from {self.annotation.__module__} import {self.annotation.__name__}",
            CatalogField.STEPS: self.steps,
            CatalogField.ARMS: self.arms,
        }


# %% installed environments


@dataclass
class EnvironmentModel:
    """An installed world description selectable as a builder environment."""

    path: str
    """Absolute path passed to CRAM's world specification."""

    def to_payload(self) -> dict[str, str]:
        """Return a readable environment name and its description path."""
        return {
            CatalogField.NAME: Path(self.path).stem.replace("_", " "),
            CatalogField.KIND: EnvironmentKind.FILE,
            CatalogField.PATH: self.path,
        }


class WorldMap(Protocol):
    """A map that builds its environment into a world rather than being read from a file."""

    def populate(self, world: World) -> None:
        """Spawn the map's furniture into a world."""

    @staticmethod
    def is_populated(world: World) -> bool:
        """Whether a world already holds the map's furniture."""


@dataclass
class MapEnvironmentModel:
    """A map class selectable as a builder environment.

    A generated demo imports the class and populates its world with it, so a real robot's
    world, which its world server serves holding only the robot, gets the same furniture
    a simulated one is built with.
    """

    map: type[WorldMap]
    """The map class populating a world."""

    name: str
    """The name the browser lists the map under."""

    def to_payload(self) -> dict[str, str]:
        """Return the map's name, its kind and how a generated demo imports it."""
        return {
            CatalogField.NAME: self.name,
            CatalogField.KIND: EnvironmentKind.MAP,
            CatalogField.CLASS: self.map.__name__,
            CatalogField.IMPORT: f"from {self.map.__module__} import {self.map.__name__}",
        }


# %% objects with an annotation class


@dataclass
class TypedObjectModel:
    """An object with an annotation class, which a generated demo spawns as an
    annotation of that class and can therefore detect by it."""

    annotation: type[HasRootBody]
    """The annotation class of the object."""

    mesh: str
    """The object's mesh file name, which also names its body."""

    mesh_url: str
    """The package URL the mesh is resolved from where the demo runs."""

    def to_payload(self) -> dict[str, str]:
        """Return the object's class, its import and where its mesh comes from."""
        return {
            CatalogField.NAME: self.annotation.__name__,
            CatalogField.CLASS: self.annotation.__name__,
            CatalogField.IMPORT: f"from {self.annotation.__module__} import {self.annotation.__name__}",
            CatalogField.MESH: self.mesh,
            CatalogField.MESH_URL: self.mesh_url,
        }


@dataclass
class ModelCatalog(CrameraPayload):
    """CRAM robot annotations and world descriptions available for authoring."""

    robots: list[RobotModel]
    """Registered robot annotations with derived capabilities."""

    worlds_directory: Path
    """Directory containing installed coraplex environment descriptions."""

    ANNOTATIONS: ClassVar[tuple[type[AbstractRobot], ...]] = (
        PR2,
        Garmi,
        HSRB,
        Tiago,
        Stretch,
        Armar7,
        Justin,
        ICub3,
        MMPDresden,
        Tracy,
        DAiSy,
        UnitreeG1,
    )
    """Annotated robots with a concrete model-description provider."""

    MAPS: ClassVar[tuple[MapEnvironmentModel, ...]] = (
        MapEnvironmentModel(ApartmentEnvironment, "real-lab apartment"),
    )
    """Environments built by a map class rather than read from a shipped file."""

    OBJECTS: ClassVar[tuple[TypedObjectModel, ...]] = (
        TypedObjectModel(
            CheezeIt, "cheeze_it.obj", ApartmentEnvironment.mesh_url("cheeze_it.obj")
        ),
    )
    """Objects with an annotation class: the ones a demonstration proved on a robot."""

    @classmethod
    def installed(cls) -> ModelCatalog:
        """Read the installed CRAM packages' model inventory, including the robots other
        installed distributions register under :data:`ROBOT_ENTRY_POINT_GROUP`.

        :return: The authoring catalog for this installation.
        """
        registered = [
            entry_point.load()
            for entry_point in entry_points(group=ROBOT_ENTRY_POINT_GROUP)
        ]
        annotations = list(cls.ANNOTATIONS) + [
            annotation for annotation in registered if annotation not in cls.ANNOTATIONS
        ]
        return cls(
            robots=[RobotModel(annotation) for annotation in annotations],
            worlds_directory=Path(str(files("coraplex"))).parent.parent
            / "resources"
            / "worlds",
        )

    @property
    def robot_types(self) -> dict[str, type[AbstractRobot]]:
        """
        :return: Every offered robot annotation, by the model name the browser lists it
            under.
        """
        return {model.annotation.__name__: model.annotation for model in self.robots}

    @property
    def environments(self) -> list[EnvironmentModel]:
        """Return every URDF world shipped in the environment directory."""
        return [
            EnvironmentModel(str(path))
            for path in sorted(self.worlds_directory.glob("*.urdf"))
        ]

    @property
    def maps(self) -> list[MapEnvironmentModel]:
        """Return every environment a map class builds."""
        return list(self.MAPS)

    @property
    def objects(self) -> list[TypedObjectModel]:
        """Return every object offered with an annotation class."""
        return list(self.OBJECTS)

    def to_payload(self) -> dict[str, Any]:
        """Return robot capabilities and environment choices for the browser."""
        return {
            CatalogField.OK: self.ok,
            CatalogField.ROBOTS: [robot.to_payload() for robot in self.robots],
            CatalogField.ENVIRONMENTS: [
                environment.to_payload() for environment in self.environments
            ],
            CatalogField.MAPS: [environment.to_payload() for environment in self.maps],
            CatalogField.OBJECTS: [model.to_payload() for model in self.objects],
        }
