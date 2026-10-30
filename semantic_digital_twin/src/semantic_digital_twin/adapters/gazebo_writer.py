from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path

import numpy as np
from krrood.ormatic.utils import classproperty

from semantic_digital_twin.adapters.exceptions import (
    DuplicateSDFName,
    ReservedSDFName,
    SharedDegreeOfFreedom,
    UnsupportedConnectionType,
    UnsupportedShapeType,
)
from semantic_digital_twin.adapters.gazebo import (
    GazeboParser,
    SDFAttribute,
    SDFElement,
    SDFJointType,
    SDFLightType,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Connection,
    KinematicStructureEntity,
)

# %% Gazebo systems


@dataclass(frozen=True)
class GazeboSystemLibrary:
    """
    The library a Gazebo system is loaded from.
    """

    filename: str
    """
    The file name of the library.
    """

    class_name: str
    """
    The fully qualified name of the class in the library that implements the system.
    """


class GazeboSystem(Enum):
    """
    The Gazebo systems a written world loads, which are those Gazebo loads for a world
    that names none.
    """

    PHYSICS = GazeboSystemLibrary("gz-sim-physics-system", "gz::sim::systems::Physics")
    """
    Simulates the physics of the world.
    """

    USER_COMMANDS = GazeboSystemLibrary(
        "gz-sim-user-commands-system", "gz::sim::systems::UserCommands"
    )
    """
    Answers requests to create, move and remove entities.
    """

    SCENE_BROADCASTER = GazeboSystemLibrary(
        "gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"
    )
    """
    Publishes the scene and the state of the world, which the graphical interface shows.
    """


class GazeboPhysicsEngine(StrEnum):
    """
    The physics engines Gazebo simulates a world with, named by their library.
    """

    BULLET_FEATHERSTONE = "gz-physics-bullet-featherstone-plugin"
    """
    Bullet with its Featherstone solver, which loads and steps worlds of many collision
    shapes quickly.
    """

    DARTSIM = "gz-physics-dartsim-plugin"
    """
    DART, the engine Gazebo picks when a world names none, which does not enforce mimic
    joints.
    """


# %% models


@dataclass
class SDFModelContents(ABC):
    """
    The bodies and connections of a world that one SDF model holds.
    """

    connections: list[Connection] = field(default_factory=list, kw_only=True)
    """
    The connections inside the model, parents before children, which become its joints.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the model.
        """

    @property
    @abstractmethod
    def origin_body(self) -> Body:
        """
        The body at whose pose the model is placed.
        """

    @property
    @abstractmethod
    def bodies(self) -> list[Body]:
        """
        The bodies the model holds as links with their shapes.
        """

    @property
    @abstractmethod
    def is_static(self) -> bool:
        """
        Whether Gazebo keeps the model in place without simulating it.
        """

    @property
    def anchors(self) -> list[Body]:
        """
        The bodies of another model that connections of this model are attached to,
        which this model holds as links without shapes, fixed to the world.
        """
        return []

    @property
    def has_moving_joints(self) -> bool:
        """
        Whether any joint of the model moves.
        """
        return any(
            isinstance(connection, ActiveConnection1DOF)
            for connection in self.connections
        )


@dataclass
class SDFRootedModelContents(SDFModelContents):
    """
    A root body and every body below it up to the next free body.
    """

    root: Body = field(kw_only=True)
    """
    The body the model is rooted at.
    """

    @property
    def name(self) -> str:
        """
        The prefix of the root, or the root's name if it has none.
        """
        return self.root.name.prefix or self.root.name.name

    @property
    def origin_body(self) -> Body:
        return self.root

    @property
    def bodies(self) -> list[Body]:
        """
        The root first, then the bodies below it.
        """
        return [self.root] + [connection.child for connection in self.connections]

    @property
    def is_static(self) -> bool:
        """
        Whether the model is rooted at the root of the world and none of its joints
        moves.
        """
        return self.root.parent_connection is None and not self.has_moving_joints

    def separate_moving_parts(
        self,
    ) -> tuple[SDFRootedModelContents, SDFMovingPartsContents]:
        """
        :return: The part of this model that the root holds through fixed connections
            alone, and the moving parts attached to it.
        """
        fixed_bodies = {self.root}
        fixed_connections = []
        moving_connections = []
        for connection in self.connections:
            if (
                isinstance(connection, FixedConnection)
                and connection.parent in fixed_bodies
            ):
                fixed_bodies.add(connection.child)
                fixed_connections.append(connection)
            else:
                moving_connections.append(connection)
        fixed_part = SDFRootedModelContents(
            root=self.root, connections=fixed_connections
        )
        return fixed_part, SDFMovingPartsContents(
            fixed_part=fixed_part, connections=moving_connections
        )


@dataclass
class SDFMovingPartsContents(SDFModelContents):
    """
    The moving parts of a model rooted at the root of the world, hanging from links that
    are fixed to the world where the parts they are attached to are.
    """

    fixed_part: SDFRootedModelContents = field(kw_only=True)
    """
    The part of the model that never moves, which the moving parts are attached to.
    """

    @classproperty
    def name_suffix(cls) -> str:
        """
        The suffix that tells the name of the moving parts from that of the fixed part.
        """
        return "moving_parts"

    @property
    def name(self) -> str:
        """
        The name of the fixed part followed by :attr:`name_suffix`.
        """
        return f"{self.fixed_part.name}_{self.name_suffix}"

    @property
    def origin_body(self) -> Body:
        return self.fixed_part.root

    @property
    def bodies(self) -> list[Body]:
        return [connection.child for connection in self.connections]

    @property
    def is_static(self) -> bool:
        return False

    @property
    def anchors(self) -> list[Body]:
        fixed_bodies = self.fixed_part.bodies
        return list(
            dict.fromkeys(
                connection.parent
                for connection in self.connections
                if connection.parent in fixed_bodies
            )
        )


# %% writing


@dataclass
class GazeboWriter:
    """
    Writes a world as an SDF world that Gazebo loads.

    The root of the world and every body a :class:`Connection6DoF` attaches each root a
    model; fixed, revolute and prismatic connections become the joints inside a model.
    Of the model rooted at the root of the world, the part held through fixed
    connections alone is static; its moving parts form a model of their own, hanging
    from links fixed to the world where the parts they are attached to are. Every other
    model moves freely.

    Of the joints that move with one degree of freedom, the first leads and the others
    mimic it.

    Joints are written at position zero, and free models where they currently are
    relative to the body they are attached to.

    .. note:: Semantic annotations, regions and the armature of joints have no SDF
        counterpart and are not written.
    """

    world: World
    """
    The world to write.
    """

    physics_engine: GazeboPhysicsEngine = GazeboPhysicsEngine.BULLET_FEATHERSTONE
    """
    The physics engine Gazebo simulates the written world with.
    """

    ambient_light: Color = field(default_factory=Color.GRAY)
    """
    The light that reaches every surface alike, whatever stands between the surface and
    a light source.
    """

    sunlight: Color = field(default_factory=lambda: Color(0.7, 0.7, 0.7))
    """
    The light of the sun, which reaches every surface facing it, the inside of buildings
    included, because it casts no shadows.
    """

    sunlight_direction: Vector3 = field(
        default_factory=lambda: Vector3(-0.4, 0.3, -0.9)
    )
    """
    The direction the sunlight shines in, in the frame of the world.
    """

    @classproperty
    def sun_name(cls) -> str:
        """
        The name of the light standing for the sun.
        """
        return "sun"

    @classproperty
    def sdf_version(cls) -> str:
        """
        The SDF version of the written documents, the first to describe mimic joints.
        """
        return "1.11"

    @classproperty
    def default_world_name(cls) -> str:
        """
        The name of the written world when the world has none, which is the name Gazebo
        gives its worlds.
        """
        return "default"

    @classproperty
    def anchor_joint_suffix(cls) -> str:
        """
        The suffix that names the joint fixing an anchor to the world after the anchor.
        """
        return "fixed_to_world"

    def anchor_joint_name(self, anchor: Body) -> str:
        """
        :param anchor: An anchor of a model, see :attr:`SDFModelContents.anchors`.
        :return: The name of the joint fixing the anchor to the world.
        """
        return f"{anchor.name.name}_{self.anchor_joint_suffix}"

    @classproperty
    def supported_connection_types(cls) -> list[type[Connection]]:
        """
        The connection types that a model holds as joints, besides the
        :class:`Connection6DoF` that separates the models.
        """
        return [FixedConnection, RevoluteConnection, PrismaticConnection]

    def write(self, file_path: str) -> None:
        """
        Writes the world to an SDF file.

        :param file_path: The path of the file to write.
        :raises UnsupportedConnectionType: If the world holds a connection no SDF joint
            expresses.
        :raises SharedDegreeOfFreedom: If connections of different models move with one
            degree of freedom.
        :raises DuplicateSDFName: If two links or joints of a model, or two models,
            would carry the same name.
        :raises ReservedSDFName: If a link or joint would carry a name SDF reserves.
        """
        document = ElementTree.ElementTree(self.document_element())
        ElementTree.indent(document)
        document.write(file_path, encoding="utf-8", xml_declaration=True)

    # %% division into models

    @cached_property
    def models(self) -> list[SDFModelContents]:
        """
        The models the world divides into: the part rooted at the root of the world that
        never moves and its moving parts first, then every free model.
        """
        root_model = SDFRootedModelContents(root=self.world.root)
        model_of_body = {self.world.root: root_model}
        for connection in self.world.connections_topologically_sorted:
            if not isinstance(connection.child, Body):
                continue
            if isinstance(connection, Connection6DoF):
                model_of_body[connection.child] = SDFRootedModelContents(
                    root=connection.child
                )
                continue
            if not isinstance(connection, tuple(self.supported_connection_types)):
                raise UnsupportedConnectionType(
                    connection=connection,
                    supported_types=self.supported_connection_types + [Connection6DoF],
                )
            model = model_of_body[connection.parent]
            model.connections.append(connection)
            model_of_body[connection.child] = model
        free_models = list(
            {id(model): model for model in model_of_body.values()}.values()
        )[1:]
        if not root_model.has_moving_joints:
            return [root_model] + free_models
        return list(root_model.separate_moving_parts()) + free_models

    @cached_property
    def root_T_body_at_zero(self) -> dict[KinematicStructureEntity, np.ndarray]:
        """
        The pose of every entity relative to the root of the world, with every joint at
        position zero and every free body where it currently is relative to its parent.
        """
        root_T_body = {self.world.root: np.eye(4)}
        for connection in self.world.connections_topologically_sorted:
            root_T_body[connection.child] = root_T_body[
                connection.parent
            ] @ self.parent_T_child_at_zero(connection)
        return root_T_body

    def parent_T_child_at_zero(self, connection: Connection) -> np.ndarray:
        """
        :param connection: The connection whose transform is computed.
        :return: The pose of the connection's child relative to its parent, at position
            zero for a joint and where the child currently is for a free body.
        """
        if isinstance(connection, Connection6DoF):
            return self.world.compute_forward_kinematics_np(
                connection.parent, connection.child
            )
        return (
            connection.parent_T_connection_expression
            @ connection.connection_T_child_expression
        ).to_np()

    @cached_property
    def leader_by_follower(
        self,
    ) -> dict[ActiveConnection1DOF, ActiveConnection1DOF]:
        """
        The joint that every mimic joint follows: the first joint of its model that
        moves with the same degree of freedom.

        :raises SharedDegreeOfFreedom: If connections of different models move with one
            degree of freedom.
        """
        model_and_leader_by_degree_of_freedom = {}
        leader_by_follower = {}
        for model in self.models:
            for connection in model.connections:
                if not isinstance(connection, ActiveConnection1DOF):
                    continue
                leader_model, leader = model_and_leader_by_degree_of_freedom.setdefault(
                    connection.raw_dof.id, (model, connection)
                )
                if leader is connection:
                    continue
                if leader_model is not model:
                    raise SharedDegreeOfFreedom(connections=[leader, connection])
                leader_by_follower[connection] = leader
        return leader_by_follower

    @staticmethod
    def check_names(names: list[str], scope: str) -> None:
        """
        :param names: The names of the elements of one SDF scope.
        :param scope: The name of the model or world the elements belong to.
        :raises DuplicateSDFName: If a name occurs more than once.
        :raises ReservedSDFName: If a name is reserved by SDF.
        """
        for name, count in Counter(names).items():
            if name == GazeboParser.world_link_name:
                raise ReservedSDFName(name=name, scope=scope)
            if count > 1:
                raise DuplicateSDFName(name=name, scope=scope)

    # %% world and models

    def document_element(self) -> ElementTree.Element:
        """
        :return: The ``sdf`` element holding the world.
        """
        world_name = self.world.name or self.default_world_name
        self.check_names([model.name for model in self.models], world_name)

        document = ElementTree.Element(
            SDFElement.SDF, {SDFAttribute.VERSION: self.sdf_version}
        )
        world_element = ElementTree.SubElement(
            document, SDFElement.WORLD, {SDFAttribute.NAME: world_name}
        )
        world_element.extend([self.system_element(system) for system in GazeboSystem])
        world_element.extend([self.scene_element(), self.sun_element()])
        world_element.extend([self.model_element(model) for model in self.models])
        return document

    def system_element(self, system: GazeboSystem) -> ElementTree.Element:
        """
        :param system: The system to load.
        :return: The ``plugin`` element loading the system, the physics system with
            :attr:`physics_engine`.
        """
        element = ElementTree.Element(
            SDFElement.PLUGIN,
            {
                SDFAttribute.FILENAME: system.value.filename,
                SDFAttribute.NAME: system.value.class_name,
            },
        )
        if system is GazeboSystem.PHYSICS:
            engine = ElementTree.SubElement(element, SDFElement.ENGINE)
            ElementTree.SubElement(engine, SDFElement.FILENAME).text = (
                self.physics_engine
            )
        return element

    def scene_element(self) -> ElementTree.Element:
        """
        :return: The ``scene`` element giving the world its ambient light.
        """
        element = ElementTree.Element(SDFElement.SCENE)
        ElementTree.SubElement(element, SDFElement.AMBIENT).text = self.format_numbers(
            self.ambient_light.to_rgba()
        )
        return element

    def sun_element(self) -> ElementTree.Element:
        """
        :return: The ``light`` element standing for the sun.
        """
        element = ElementTree.Element(
            SDFElement.LIGHT,
            {
                SDFAttribute.NAME: self.sun_name,
                SDFAttribute.TYPE: SDFLightType.DIRECTIONAL,
            },
        )
        ElementTree.SubElement(element, SDFElement.CAST_SHADOWS).text = "false"
        ElementTree.SubElement(element, SDFElement.DIFFUSE).text = self.format_numbers(
            self.sunlight.to_rgba()
        )
        ElementTree.SubElement(element, SDFElement.DIRECTION).text = (
            self.format_numbers(self.sunlight_direction.to_np()[:3])
        )
        return element

    def model_element(self, model: SDFModelContents) -> ElementTree.Element:
        """
        :param model: The model to describe.
        :return: The ``model`` element holding the model's links and joints.
        """
        self.check_names(
            [body.name.name for body in model.anchors + model.bodies]
            + [connection.name.name for connection in model.connections]
            + [self.anchor_joint_name(anchor) for anchor in model.anchors],
            model.name,
        )

        root_T_model = self.root_T_body_at_zero[model.origin_body]
        element = ElementTree.Element(SDFElement.MODEL, {SDFAttribute.NAME: model.name})
        element.append(self.pose_element(root_T_model))
        if model.is_static:
            ElementTree.SubElement(element, SDFElement.STATIC).text = "true"

        model_T_root = np.linalg.inv(root_T_model)
        element.extend(
            [
                self.anchor_link_element(
                    anchor, model_T_root @ self.root_T_body_at_zero[anchor]
                )
                for anchor in model.anchors
            ]
        )
        element.extend(
            [
                self.link_element(body, model_T_root @ self.root_T_body_at_zero[body])
                for body in model.bodies
            ]
        )
        element.extend(
            [self.joint_element(connection) for connection in model.connections]
        )
        element.extend([self.anchor_joint_element(anchor) for anchor in model.anchors])
        return element

    # %% links

    def anchor_link_element(
        self, anchor: Body, model_T_link: np.ndarray
    ) -> ElementTree.Element:
        """
        :param anchor: An anchor of a model, see :attr:`SDFModelContents.anchors`.
        :param model_T_link: The pose of the anchor relative to the model.
        :return: The ``link`` element placing the anchor, without its shapes.
        """
        element = ElementTree.Element(
            SDFElement.LINK, {SDFAttribute.NAME: anchor.name.name}
        )
        element.append(self.pose_element(model_T_link))
        return element

    def link_element(self, body: Body, model_T_link: np.ndarray) -> ElementTree.Element:
        """
        :param body: The body to describe.
        :param model_T_link: The pose of the body relative to its model.
        :return: The ``link`` element describing the body.
        """
        element = ElementTree.Element(
            SDFElement.LINK, {SDFAttribute.NAME: body.name.name}
        )
        element.append(self.pose_element(model_T_link))
        if body.inertial is not None:
            element.append(self.inertial_element(body.inertial))
        element.extend(
            [
                self.visual_element(index, shape)
                for index, shape in enumerate(body.visual)
            ]
        )
        element.extend(
            [
                self.shape_element(SDFElement.COLLISION, index, shape)
                for index, shape in enumerate(body.collision)
            ]
        )
        return element

    def inertial_element(self, inertial: Inertial) -> ElementTree.Element:
        """
        :param inertial: The inertial properties to describe.
        :return: The ``inertial`` element, whose inertia is expressed in the link frame.
        """
        link_T_inertial = np.eye(4)
        link_T_inertial[:3, 3] = inertial.center_of_mass.to_np()[:3]
        inertia = inertial.inertia.data

        element = ElementTree.Element(SDFElement.INERTIAL)
        element.append(self.pose_element(link_T_inertial))
        ElementTree.SubElement(element, SDFElement.MASS).text = self.format_numbers(
            [inertial.mass]
        )
        inertia_element = ElementTree.SubElement(element, SDFElement.INERTIA)
        for tag, value in (
            (SDFElement.IXX, inertia[0, 0]),
            (SDFElement.IXY, inertia[0, 1]),
            (SDFElement.IXZ, inertia[0, 2]),
            (SDFElement.IYY, inertia[1, 1]),
            (SDFElement.IYZ, inertia[1, 2]),
            (SDFElement.IZZ, inertia[2, 2]),
        ):
            ElementTree.SubElement(inertia_element, tag).text = self.format_numbers(
                [value]
            )
        return element

    # %% shapes

    def visual_element(self, index: int, shape: Shape) -> ElementTree.Element:
        """
        :param index: The position of the shape among the visual shapes of its body.
        :param shape: The shape to describe.
        :return: The ``visual`` element describing the shape, painted in its color,
            except for a mesh whose color was never set, which keeps the materials of
            its file.
        """
        element = self.shape_element(SDFElement.VISUAL, index, shape)
        if isinstance(shape, Mesh) and shape.color == Color():
            return element
        rgba = self.format_numbers(shape.color.to_rgba())
        material = ElementTree.SubElement(element, SDFElement.MATERIAL)
        ElementTree.SubElement(material, SDFElement.AMBIENT).text = rgba
        ElementTree.SubElement(material, SDFElement.DIFFUSE).text = rgba
        return element

    def shape_element(
        self, tag: SDFElement, index: int, shape: Shape
    ) -> ElementTree.Element:
        """
        :param tag: Whether the shape is written as a visual or a collision.
        :param index: The position of the shape among the shapes of its kind.
        :param shape: The shape to describe.
        :return: The element placing the shape's geometry on its link.
        """
        element = ElementTree.Element(tag, {SDFAttribute.NAME: f"{tag}_{index}"})
        element.append(self.pose_element(shape.origin.to_np()))
        ElementTree.SubElement(element, SDFElement.GEOMETRY).append(
            self.geometry_element(shape)
        )
        return element

    def geometry_element(self, shape: Shape) -> ElementTree.Element:
        """
        :param shape: The shape to describe.
        :return: The element describing the shape's geometry.
        :raises UnsupportedShapeType: If no SDF geometry describes the shape.
        """
        if isinstance(shape, Box):
            element = ElementTree.Element(SDFElement.BOX)
            ElementTree.SubElement(element, SDFElement.SIZE).text = self.format_numbers(
                [shape.scale.x, shape.scale.y, shape.scale.z]
            )
            return element

        if isinstance(shape, Sphere):
            element = ElementTree.Element(SDFElement.SPHERE)
            ElementTree.SubElement(element, SDFElement.RADIUS).text = (
                self.format_numbers([shape.radius])
            )
            return element

        if isinstance(shape, Cylinder):
            element = ElementTree.Element(SDFElement.CYLINDER)
            ElementTree.SubElement(element, SDFElement.RADIUS).text = (
                self.format_numbers([shape.width / 2])
            )
            ElementTree.SubElement(element, SDFElement.LENGTH).text = (
                self.format_numbers([shape.height])
            )
            return element

        if isinstance(shape, Mesh):
            element = ElementTree.Element(SDFElement.MESH)
            ElementTree.SubElement(element, SDFElement.URI).text = (
                Path(shape.local_file).absolute().as_uri()
            )
            ElementTree.SubElement(element, SDFElement.SCALE).text = (
                self.format_numbers([shape.scale.x, shape.scale.y, shape.scale.z])
            )
            return element

        raise UnsupportedShapeType(shape=shape)

    # %% joints

    def joint_element(self, connection: Connection) -> ElementTree.Element:
        """
        :param connection: The fixed, revolute or prismatic connection to describe.
        :return: The ``joint`` element describing the connection, placed at the
            connection's frame relative to its child.
        """
        element = ElementTree.Element(
            SDFElement.JOINT,
            {
                SDFAttribute.NAME: connection.name.name,
                SDFAttribute.TYPE: self.joint_type_of(connection),
            },
        )
        ElementTree.SubElement(element, SDFElement.PARENT).text = (
            connection.parent.name.name
        )
        ElementTree.SubElement(element, SDFElement.CHILD).text = (
            connection.child.name.name
        )
        element.append(
            self.pose_element(
                connection.connection_T_child_expression.inverse().to_np()
            )
        )
        if isinstance(connection, ActiveConnection1DOF):
            element.append(self.axis_element(connection))
        return element

    @staticmethod
    def joint_type_of(connection: Connection) -> SDFJointType:
        """
        :param connection: The fixed, revolute or prismatic connection.
        :return: The type of the joint the connection is written as.
        """
        if isinstance(connection, FixedConnection):
            return SDFJointType.FIXED
        if isinstance(connection, PrismaticConnection):
            return SDFJointType.PRISMATIC
        if connection.dof.has_position_limits():
            return SDFJointType.REVOLUTE
        return SDFJointType.CONTINUOUS

    def axis_element(self, connection: ActiveConnection1DOF) -> ElementTree.Element:
        """
        :param connection: The revolute or prismatic connection to describe.
        :return: The ``axis`` element holding the connection's axis, limits and dynamics,
            in the coordinates the connection moves in.
        """
        element = ElementTree.Element(SDFElement.AXIS)
        ElementTree.SubElement(element, SDFElement.XYZ).text = self.format_numbers(
            connection.axis.to_np()[:3]
        )

        limits = connection.dof.limits
        limit_element = ElementTree.SubElement(element, SDFElement.LIMIT)
        for tag, value in (
            (SDFElement.LOWER, limits.lower.position),
            (SDFElement.UPPER, limits.upper.position),
            (SDFElement.VELOCITY, limits.upper.velocity),
        ):
            if value is not None:
                ElementTree.SubElement(limit_element, tag).text = self.format_numbers(
                    [value]
                )

        dynamics = ElementTree.SubElement(element, SDFElement.DYNAMICS)
        ElementTree.SubElement(dynamics, SDFElement.DAMPING).text = self.format_numbers(
            [connection.dynamics.damping]
        )
        ElementTree.SubElement(dynamics, SDFElement.FRICTION).text = (
            self.format_numbers([connection.dynamics.dry_friction])
        )

        leader = self.leader_by_follower.get(connection)
        if leader is not None:
            element.append(self.mimic_element(connection, leader))
        return element

    def mimic_element(
        self, follower: ActiveConnection1DOF, leader: ActiveConnection1DOF
    ) -> ElementTree.Element:
        """
        :param follower: The connection that mimics the leader.
        :param leader: The connection that moves with the same degree of freedom as the
            follower and has a joint of its own.
        :return: The ``mimic`` element that moves the follower's joint as the follower
            moves with the leader's.
        """
        element = ElementTree.Element(
            SDFElement.MIMIC, {SDFAttribute.JOINT: leader.name.name}
        )
        for tag, value in (
            (SDFElement.MULTIPLIER, follower.multiplier / leader.multiplier),
            (SDFElement.OFFSET, follower.offset),
            (SDFElement.REFERENCE, leader.offset),
        ):
            ElementTree.SubElement(element, tag).text = self.format_numbers([value])
        return element

    def anchor_joint_element(self, anchor: Body) -> ElementTree.Element:
        """
        :param anchor: An anchor of a model, see :attr:`SDFModelContents.anchors`.
        :return: The ``joint`` element fixing the anchor to the world.
        """
        element = ElementTree.Element(
            SDFElement.JOINT,
            {
                SDFAttribute.NAME: self.anchor_joint_name(anchor),
                SDFAttribute.TYPE: SDFJointType.FIXED,
            },
        )
        ElementTree.SubElement(element, SDFElement.PARENT).text = (
            GazeboParser.world_link_name
        )
        ElementTree.SubElement(element, SDFElement.CHILD).text = anchor.name.name
        return element

    # %% numbers

    def pose_element(self, transform: np.ndarray) -> ElementTree.Element:
        """
        :param transform: The homogeneous transform to describe.
        :return: The ``pose`` element holding the transform as position and roll, pitch
            and yaw.
        """
        roll, pitch, yaw = HomogeneousTransformationMatrix(
            data=transform
        ).rotation_matrix.rpy
        element = ElementTree.Element(SDFElement.POSE)
        element.text = self.format_numbers([*transform[:3, 3], roll, pitch, yaw])
        return element

    @staticmethod
    def format_numbers(values: list[float]) -> str:
        """
        :param values: The numbers to write.
        :return: The numbers separated by spaces, each written exactly.
        """
        return " ".join(repr(float(value)) for value in values)
