from __future__ import annotations

from dataclasses import dataclass, field

from krrood.exceptions import DataclassException

from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.world_entity import Connection

# %% writing SDF


@dataclass
class UnsupportedConnectionType(DataclassException):
    """
    Raised when a world holds a connection that no SDF joint expresses.
    """

    connection: Connection = field(kw_only=True)
    """
    The connection that cannot be written.
    """

    supported_types: list[type[Connection]] = field(kw_only=True)
    """
    The connection types that can be written.
    """

    def error_message(self) -> str:
        return (
            f"Connection '{self.connection.name}' is a "
            f"{type(self.connection).__name__}, which no SDF joint expresses."
        )

    def suggest_correction(self) -> str:
        supported_names = ", ".join(
            connection_type.__name__ for connection_type in self.supported_types
        )
        return f"Replace it with one of {supported_names} before writing the world."


@dataclass
class UnsupportedShapeType(DataclassException):
    """
    Raised when a body holds a shape that no SDF geometry expresses.
    """

    shape: Shape = field(kw_only=True)
    """
    The shape that cannot be written.
    """

    def error_message(self) -> str:
        return f"A {type(self.shape).__name__} has no SDF geometry."

    def suggest_correction(self) -> str:
        return "Replace the shape with a box, sphere, cylinder or mesh."


@dataclass
class SharedDegreeOfFreedom(DataclassException):
    """
    Raised when several connections move with one degree of freedom, a coupling that SDF
    joints do not express.
    """

    connections: list[Connection] = field(kw_only=True)
    """
    The connections that share the degree of freedom.
    """

    def error_message(self) -> str:
        connection_names = ", ".join(
            f"'{connection.name}'" for connection in self.connections
        )
        return f"Connections {connection_names} share one degree of freedom."

    def suggest_correction(self) -> str:
        return "Give each connection a degree of freedom of its own."


@dataclass
class DuplicateSDFName(DataclassException):
    """
    Raised when two elements of one SDF scope would carry the same name, which SDF
    rejects.
    """

    name: str = field(kw_only=True)
    """
    The name that occurs more than once.
    """

    scope: str = field(kw_only=True)
    """
    The name of the model or world whose elements share the name.
    """

    def error_message(self) -> str:
        return f"The name '{self.name}' occurs more than once in '{self.scope}'."

    def suggest_correction(self) -> str:
        return (
            "Rename the bodies or connections so that every link and joint of a model, "
            "and every model of the world, has a name of its own."
        )


@dataclass
class ReservedSDFName(DataclassException):
    """
    Raised when an element would carry a name that SDF reserves.
    """

    name: str = field(kw_only=True)
    """
    The reserved name.
    """

    scope: str = field(kw_only=True)
    """
    The name of the model or world the element belongs to.
    """

    def error_message(self) -> str:
        return f"The name '{self.name}' in '{self.scope}' is reserved by SDF."

    def suggest_correction(self) -> str:
        return "Rename the body or connection that carries it."
