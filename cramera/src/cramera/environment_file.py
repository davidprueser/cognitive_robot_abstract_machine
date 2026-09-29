"""
The file an environment is described in, read by the parser its format asks for.

A URDF, a Gazebo world and a USD scene each describe an environment robots are put into;
:meth:`EnvironmentFile.from_path` tells them apart by the file's suffix, and each kind
turns itself into the world specification its parser builds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath

from cramera.scene_presentation import ScenePresentation
from krrood.utils import recursive_subclasses
from semantic_digital_twin.adapters.usd.stage_parser import RootPlacement
from semantic_digital_twin.api import (
    RobotSpecification,
    SpawnSpecification,
    WorldSpecification,
)
from typing_extensions import ClassVar, Sequence, Tuple


@dataclass
class UnsupportedEnvironmentFileError(ValueError):
    """
    Raised for a file whose suffix no environment parser reads.
    """

    path: str
    """
    The file that was named as an environment.
    """

    def __str__(self) -> str:
        supported = sorted(
            suffix
            for kind in recursive_subclasses(EnvironmentFile)
            for suffix in kind.SUFFIXES
        )
        return f"{self.path} is no environment file; readable are {supported}"


@dataclass
class EnvironmentFile(ABC):
    """
    An environment described in a file.
    """

    path: str
    """
    Where the file is: a path, or a ``package://`` URL its parser resolves.
    """

    SUFFIXES: ClassVar[Tuple[str, ...]] = ()
    """
    The file endings this kind of environment file is written with.
    """

    @classmethod
    def from_path(cls, path: str) -> EnvironmentFile:
        """
        :param path: A file describing an environment.
        :return: The environment file of the kind its suffix names.
        :raises UnsupportedEnvironmentFileError: If no kind reads that suffix.
        """
        suffix = PurePosixPath(path).suffix.lower()
        for kind in recursive_subclasses(EnvironmentFile):
            if suffix in kind.SUFFIXES:
                return kind(path=path)
        raise UnsupportedEnvironmentFileError(path)

    @abstractmethod
    def specification(
        self,
        robots: Sequence[RobotSpecification] = (),
        objects: Sequence[SpawnSpecification] = (),
    ) -> WorldSpecification:
        """
        :param robots: The robots put into the environment.
        :param objects: What is spawned once the robots are in place.
        :return: The specification of the environment with the robots and objects in it.
        """

    def presentation(self) -> ScenePresentation:
        """
        :return: How the viewer draws the environment: with the furniture palette it
            gives models that ship without a look of their own.
        """
        return ScenePresentation()


@dataclass
class URDFEnvironmentFile(EnvironmentFile):
    """
    An environment described in a URDF.
    """

    SUFFIXES: ClassVar[Tuple[str, ...]] = (".urdf",)

    def specification(
        self,
        robots: Sequence[RobotSpecification] = (),
        objects: Sequence[SpawnSpecification] = (),
    ) -> WorldSpecification:
        return WorldSpecification.from_urdf(
            self.path, robots=list(robots), objects=list(objects)
        )


@dataclass
class GazeboEnvironmentFile(EnvironmentFile):
    """
    An environment described as a Gazebo world or model.
    """

    SUFFIXES: ClassVar[Tuple[str, ...]] = (".world", ".sdf")

    def specification(
        self,
        robots: Sequence[RobotSpecification] = (),
        objects: Sequence[SpawnSpecification] = (),
    ) -> WorldSpecification:
        return WorldSpecification.from_gazebo(
            self.path, robots=list(robots), objects=list(objects)
        )


@dataclass
class USDSceneEnvironmentFile(EnvironmentFile):
    """
    An environment described as a USD stage of separately placed static objects, such as
    a scanned building.
    """

    SUFFIXES: ClassVar[Tuple[str, ...]] = (".usd", ".usda", ".usdc", ".usdz")

    root_placement: RootPlacement = RootPlacement.SCENE_GROUND
    """
    Where the scene's root is placed.

    A scan is captured wherever its scanner stood, often far from its stage's origin, so
    it is stood on its own ground unless it says otherwise.
    """

    def specification(
        self,
        robots: Sequence[RobotSpecification] = (),
        objects: Sequence[SpawnSpecification] = (),
    ) -> WorldSpecification:
        return WorldSpecification.from_usd_scene(
            self.path,
            root_placement=self.root_placement,
            robots=list(robots),
            objects=list(objects),
        )

    def presentation(self) -> ScenePresentation:
        """
        :return: How the viewer draws the scan: as the photographed surfaces its
            materials carry, which no palette may replace.
        """
        return ScenePresentation(preserve_environment_materials=True)
