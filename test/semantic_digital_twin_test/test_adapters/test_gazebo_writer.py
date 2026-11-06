from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest
from krrood.code_generation.naming import to_snake_case

from semantic_digital_twin.adapters.exceptions import (
    DuplicateSDFName,
    ReservedSDFName,
    SharedDegreeOfFreedom,
    UnsupportedConnectionType,
)
from semantic_digital_twin.adapters.gazebo import (
    GazeboParser,
    SDFAttribute,
    SDFElement,
    SDFJointType,
)
from semantic_digital_twin.adapters.gazebo_writer import (
    CollideBitmask,
    GazeboModelSystem,
    GazeboSystem,
    GazeboWriter,
    StaticLinkDivision,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor, Wall
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import ServoGains
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

from .test_gazebo import (
    GazeboFixturePaths,
    body_named,
    connection_named,
    gazebo_paths,
)

# %% helpers


def write_and_parse(world: World, directory: Path) -> World:
    """
    :param world: The world to write.
    :param directory: The directory the SDF file is written to.
    :return: The world parsed back from the written file.
    """
    file_path = str(directory / "world.sdf")
    GazeboWriter(world=world).write(file_path)
    return GazeboParser.from_file(file_path).parse()


def write_and_read(world: World, directory: Path) -> ElementTree.Element:
    """
    :param world: The world to write.
    :param directory: The directory the SDF file is written to.
    :return: The root element of the written file.
    """
    return write_with(GazeboWriter(world=world), directory)


def write_with(writer: GazeboWriter, directory: Path) -> ElementTree.Element:
    """
    :param writer: The writer, holding the world to write.
    :param directory: The directory the SDF file is written to.
    :return: The root element of the written file.
    """
    file_path = str(directory / "world.sdf")
    writer.write(file_path)
    return ElementTree.parse(file_path).getroot()


def model_element_named(
    document: ElementTree.Element, name: str
) -> ElementTree.Element:
    """
    :param document: The root element of a written file.
    :param name: The name of the model.
    :return: The model element with that name.
    """
    return document.find(
        f"{SDFElement.WORLD}/{SDFElement.MODEL}[@{SDFAttribute.NAME}='{name}']"
    )


@pytest.fixture
def world_with_free_object() -> World:
    """
    A room with a table fixed in it and a cup standing freely on the table.
    """
    world = World()
    with world.modify_world():
        room = Body(name=PrefixedName("room", "house"))
        table = Body(
            name=PrefixedName("table", "house"),
            visual=ShapeCollection([Box(scale=Scale(1.0, 0.6, 0.05))]),
        )
        world.add_connection(
            FixedConnection(
                parent=room,
                child=table,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1.0, z=0.75
                ),
            )
        )
        cup = Body(name=PrefixedName("cup", "cup"))
        cup_connection = Connection6DoF.create_with_dofs(
            world=world, parent=table, child=cup
        )
        world.add_connection(cup_connection)
    cup_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.2, 0.3, 0.05, yaw=0.5, reference_frame=table
    )
    return world


# %% shapes and inertial properties


class TestShapeWriting:
    """
    The shapes and inertial properties of a body, written and parsed back.
    """

    def test_writes_every_shape_type(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        assert [type(shape) for shape in body_named(parsed, "link").visual] == [
            type(shape) for shape in body_named(source, "link").visual
        ]

    def test_writes_shape_dimensions(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        source_box, source_cylinder, source_sphere = body_named(source, "link").visual
        box, cylinder, sphere = body_named(parsed, "link").visual
        assert box.scale == source_box.scale
        assert (cylinder.width, cylinder.height) == (
            source_cylinder.width,
            source_cylinder.height,
        )
        assert sphere.radius == source_sphere.radius

    def test_writes_shape_origins(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        for shape, source_shape in zip(
            body_named(parsed, "link").visual, body_named(source, "link").visual
        ):
            assert np.allclose(shape.origin.to_np(), source_shape.origin.to_np())

    def test_writes_visual_colors(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        assert [shape.color for shape in body_named(parsed, "link").visual] == [
            shape.color for shape in body_named(source, "link").visual
        ]

    def test_writes_collision_shapes(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        assert [type(shape) for shape in body_named(parsed, "link").collision] == [
            type(shape) for shape in body_named(source, "link").collision
        ]

    def test_writes_inertial_properties(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.simple_shapes).parse()
        parsed = write_and_parse(source, tmp_path)
        inertial = body_named(parsed, "link").inertial
        source_inertial = body_named(source, "link").inertial
        assert inertial.mass == source_inertial.mass
        assert np.allclose(
            inertial.center_of_mass.to_np(), source_inertial.center_of_mass.to_np()
        )
        assert np.allclose(inertial.inertia.data, source_inertial.inertia.data)


# %% meshes


class TestMeshWriting:
    """
    Meshes, which are written as references to their files.
    """

    @pytest.fixture
    def world_with_mesh(self) -> World:
        """
        A world of one body drawn with a scaled mesh of the default color.
        """
        mesh_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "semantic_digital_twin",
                "resources",
                "stl",
                "milk.stl",
            )
        )
        world = World()
        with world.modify_world():
            world.add_body(
                Body(
                    name=PrefixedName("milk", "milk"),
                    visual=ShapeCollection(
                        [Mesh(filename=mesh_path, scale=Scale(2.0, 2.0, 2.0))]
                    ),
                )
            )
        return world

    def test_mesh_refers_to_its_file(
        self, world_with_mesh: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(world_with_mesh, tmp_path)
        (mesh,) = body_named(parsed, "milk").visual
        (source_mesh,) = body_named(world_with_mesh, "milk").visual
        assert mesh.filename == str(source_mesh.local_file)
        assert mesh.scale == source_mesh.scale

    def test_mesh_without_color_keeps_the_materials_of_its_file(
        self, world_with_mesh: World, tmp_path: Path
    ) -> None:
        """
        Gazebo paints a visual in its material instead of the materials of its mesh, so
        a mesh whose color was never set must be written without one.
        """
        document = write_and_read(world_with_mesh, tmp_path)
        visual = model_element_named(document, "milk").find(
            f"{SDFElement.LINK}/{SDFElement.VISUAL}"
        )
        assert visual.find(SDFElement.MATERIAL) is None


# %% joints


class TestJointWriting:
    """
    Fixed, revolute and prismatic connections, written as joints and parsed back.
    """

    def test_revolute_connection_stays_revolute(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        parsed = write_and_parse(source, tmp_path)
        assert isinstance(connection_named(parsed, "hinge"), RevoluteConnection)
        assert parsed.validate()

    def test_writes_axis(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        parsed = write_and_parse(source, tmp_path)
        assert np.allclose(
            connection_named(parsed, "hinge").axis.to_np(),
            connection_named(source, "hinge").axis.to_np(),
        )

    def test_writes_limits(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        parsed = write_and_parse(source, tmp_path)
        limits = connection_named(parsed, "hinge").raw_dof.limits
        source_limits = connection_named(source, "hinge").raw_dof.limits
        assert limits.lower.position == source_limits.lower.position
        assert limits.upper.position == source_limits.upper.position
        assert limits.upper.velocity == source_limits.upper.velocity

    def test_writes_dynamics(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        parsed = write_and_parse(source, tmp_path)
        dynamics = connection_named(parsed, "hinge").dynamics
        source_dynamics = connection_named(source, "hinge").dynamics
        assert (dynamics.damping, dynamics.dry_friction) == (
            source_dynamics.damping,
            source_dynamics.dry_friction,
        )

    def test_joint_is_written_at_position_zero(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        """
        SDF places links where they are with every joint at position zero, so a door
        that stands open when it is written must be written closed.
        """
        source = GazeboParser.from_file(gazebo_paths.hinged_door).parse()
        frame_T_closed_door = source.compute_forward_kinematics_np(
            body_named(source, "frame"), body_named(source, "door")
        )
        connection_named(source, "hinge").position = 1.0

        parsed = write_and_parse(source, tmp_path)

        frame_T_door = parsed.compute_forward_kinematics_np(
            body_named(parsed, "frame"), body_named(parsed, "door")
        )
        assert np.allclose(frame_T_door, frame_T_closed_door)

    def test_prismatic_connection_stays_prismatic(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.drawer).parse()
        parsed = write_and_parse(source, tmp_path)
        connection = connection_named(parsed, "slide")
        source_connection = connection_named(source, "slide")
        assert isinstance(connection, PrismaticConnection)
        assert (
            connection.raw_dof.limits.upper.position
            == source_connection.raw_dof.limits.upper.position
        )

    def test_revolute_connection_without_position_limits_is_continuous(
        self, gazebo_paths: GazeboFixturePaths, tmp_path: Path
    ) -> None:
        source = GazeboParser.from_file(gazebo_paths.drawer).parse()
        document = write_and_read(source, tmp_path)
        joint = document.find(
            f"{SDFElement.WORLD}/{SDFElement.MODEL}/"
            f"{SDFElement.JOINT}[@{SDFAttribute.NAME}='spin']"
        )
        assert joint.get(SDFAttribute.TYPE) == SDFJointType.CONTINUOUS

    def test_limits_are_those_of_the_connection_position(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        """
        A connection moves by its degree of freedom shifted by an offset, and its joint
        moves like the connection does.
        """
        connection = mini_world.connections[0]
        connection.raw_dof.limits = DegreeOfFreedomLimits.from_position_range_and_speed(
            lower_position=-1.0, upper_position=1.0, maximum_speed=2.0
        )
        connection.offset = 0.5

        parsed = write_and_parse(mini_world, tmp_path)

        limits = connection_named(parsed, connection.name.name).raw_dof.limits
        assert limits.lower.position == connection.dof.limits.lower.position
        assert limits.upper.position == connection.dof.limits.upper.position

    def test_connection_sharing_a_degree_of_freedom_becomes_a_mimic_joint(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        """
        A connection that moves with the degree of freedom of another one keeps moving
        with it, scaled and offset as before.
        """
        hinge = mini_world.connections[0]
        hinge.offset = 0.2
        with mini_world.modify_world():
            mirrored_hinge = RevoluteConnection(
                parent=hinge.parent,
                child=Body(name=PrefixedName("mirrored_tip")),
                axis=Vector3.Z(),
                multiplier=-1.5,
                offset=0.3,
                raw_dof=hinge.raw_dof,
            )
            mini_world.add_connection(mirrored_hinge)
        hinge.position = 0.7

        parsed = write_and_parse(mini_world, tmp_path)

        parsed_hinge = connection_named(parsed, hinge.name.name)
        parsed_mirrored_hinge = connection_named(parsed, mirrored_hinge.name.name)
        parsed_hinge.position = hinge.position
        assert parsed_mirrored_hinge.raw_dof is parsed_hinge.raw_dof
        assert parsed_mirrored_hinge.position == pytest.approx(mirrored_hinge.position)


# %% models


class TestModelDivision:
    """
    The division of a world into the models of an SDF world.
    """

    def test_free_body_roots_a_model_of_its_own(
        self, world_with_free_object: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(world_with_free_object, tmp_path)
        cup = body_named(parsed, "cup")
        assert isinstance(cup.parent_connection, Connection6DoF)
        assert cup.parent_connection.parent == parsed.root

    def test_free_model_is_placed_where_its_body_is(
        self, world_with_free_object: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(world_with_free_object, tmp_path)
        root_T_cup = parsed.compute_forward_kinematics_np(
            parsed.root, body_named(parsed, "cup")
        )
        source_root_T_cup = world_with_free_object.compute_forward_kinematics_np(
            world_with_free_object.root, body_named(world_with_free_object, "cup")
        )
        assert np.allclose(root_T_cup, source_root_T_cup)

    def test_model_without_moving_joints_is_static(
        self, world_with_free_object: World, tmp_path: Path
    ) -> None:
        document = write_and_read(world_with_free_object, tmp_path)
        model = model_element_named(document, "house")
        assert model.findtext(SDFElement.STATIC) == "true"

    def test_moving_parts_hang_from_a_link_fixed_to_the_world(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        """
        A static model cannot move its joints in Gazebo, so the moving parts of the
        model holding the world's root form a model of their own, fixed to the world
        where they are attached.
        """
        writer = GazeboWriter(world=mini_world)
        fixed_part, moving_parts = writer.models
        document = write_and_read(mini_world, tmp_path)
        fixed_part_element = model_element_named(document, fixed_part.name)
        moving_parts_element = model_element_named(document, moving_parts.name)
        anchor_joint = moving_parts_element.find(
            f"{SDFElement.JOINT}[@{SDFAttribute.NAME}="
            f"'{writer.anchor_joint_name(mini_world.root)}']"
        )
        assert fixed_part_element.findtext(SDFElement.STATIC) == "true"
        assert moving_parts_element.find(SDFElement.STATIC) is None
        assert anchor_joint.get(SDFAttribute.TYPE) == SDFJointType.FIXED
        assert anchor_joint.findtext(SDFElement.PARENT) == GazeboParser.world_link_name
        assert anchor_joint.findtext(SDFElement.CHILD) == mini_world.root.name.name

    def test_model_fixed_to_the_world_stays_in_place(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(mini_world, tmp_path)
        root = body_named(parsed, mini_world.root.name.name)
        assert isinstance(root.parent_connection, FixedConnection)


# %% static part written by kind


def offset_box() -> ShapeCollection:
    """
    :return: A box placed off the origin of its body and turned against it.
    """
    return ShapeCollection(
        [
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(y=0.5, yaw=0.1),
                scale=Scale(1.0, 0.1, 2.0),
            )
        ]
    )


@pytest.fixture
def building_world() -> World:
    """
    A building of two walls, a floor slab and an unannotated pillar fixed to its root.

    The shapes sit off their bodies' origins and the bodies are turned, so a shape's
    pose relative to the model depends on both.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("building", "office"))
        bodies = []
        for name, x, yaw in (
            ("wall_north", 2.0, 0.3),
            ("wall_south", -2.0, -0.4),
            ("slab", 0.0, 0.2),
            ("pillar", 1.0, 0.7),
        ):
            body = Body(
                name=PrefixedName(name, "office"),
                visual=offset_box(),
                collision=offset_box(),
            )
            world.add_connection(
                FixedConnection(
                    parent=root,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=x, z=0.5, yaw=yaw
                    ),
                )
            )
            bodies.append(body)
        wall_north, wall_south, slab, _ = bodies
        world.add_semantic_annotation(Wall(root=wall_north))
        world.add_semantic_annotation(Wall(root=wall_south))
        world.add_semantic_annotation(Floor(root=slab))
    return world


@pytest.fixture
def building_world_with_hinged_lid(building_world: World) -> World:
    """
    The building with a lid swinging on a hinge on its floor slab.
    """
    slab = body_named(building_world, "slab")
    with building_world.modify_world():
        lid = Body(
            name=PrefixedName("lid", "office"),
            collision=ShapeCollection([Box(scale=Scale(1.0, 1.0, 0.05))]),
        )
        building_world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=building_world, parent=slab, child=lid, axis=Vector3.Y()
            )
        )
    return building_world


def root_T_collisions(world: World) -> np.ndarray:
    """
    :param world: A world.
    :return: The positions of all collision shapes of the world relative to its root,
        sorted so that two worlds holding the same shapes give the same array.
    """
    positions = [
        (world.compute_forward_kinematics_np(world.root, body) @ shape.origin.to_np())[
            :3, 3
        ]
        for body in world.bodies
        for shape in body.collision
    ]
    return np.array(sorted(map(tuple, np.round(positions, 9))))


class TestStaticLinksByKind:
    """
    The static part of the model holding the world's root, written as one link per kind
    of body.
    """

    def test_static_part_has_one_link_per_kind_of_body(
        self, building_world: World, tmp_path: Path
    ) -> None:
        """
        Bodies held by no annotation go into the link named after the root.
        """
        writer = GazeboWriter(
            world=building_world, static_link_division=StaticLinkDivision.ONE_PER_KIND
        )
        (fixed_part,) = writer.models
        document = write_with(writer, tmp_path)
        links = model_element_named(document, fixed_part.name).findall(SDFElement.LINK)
        assert {link.get(SDFAttribute.NAME) for link in links} == {
            building_world.root.name.name,
            to_snake_case(Wall.__name__),
            to_snake_case(Floor.__name__),
        }

    def test_static_part_has_no_joints(
        self, building_world: World, tmp_path: Path
    ) -> None:
        writer = GazeboWriter(
            world=building_world, static_link_division=StaticLinkDivision.ONE_PER_KIND
        )
        (fixed_part,) = writer.models
        document = write_with(writer, tmp_path)
        assert model_element_named(document, fixed_part.name).find(
            SDFElement.JOINT
        ) is (None)

    def test_shapes_keep_their_place(
        self, building_world: World, tmp_path: Path
    ) -> None:
        file_path = str(tmp_path / "world.sdf")
        GazeboWriter(
            world=building_world, static_link_division=StaticLinkDivision.ONE_PER_KIND
        ).write(file_path)
        parsed = GazeboParser.from_file(file_path).parse()
        assert np.allclose(root_T_collisions(parsed), root_T_collisions(building_world))

    def test_moving_parts_keep_one_link_per_body(
        self, building_world_with_hinged_lid: World, tmp_path: Path
    ) -> None:
        writer = GazeboWriter(
            world=building_world_with_hinged_lid,
            static_link_division=StaticLinkDivision.ONE_PER_KIND,
        )
        _, moving_parts = writer.models
        document = write_with(writer, tmp_path)
        links = model_element_named(document, moving_parts.name).findall(
            SDFElement.LINK
        )
        assert {link.get(SDFAttribute.NAME) for link in links} == {
            body.name.name for body in moving_parts.anchors + moving_parts.bodies
        }


# %% simulation


@pytest.fixture
def world_with_hinged_lid() -> World:
    """
    A room whose lid swings on a hinge, both with collision shapes.
    """
    world = World()
    with world.modify_world():
        room = Body(
            name=PrefixedName("room", "house"),
            collision=ShapeCollection([Box(scale=Scale(4.0, 4.0, 0.1))]),
        )
        lid = Body(
            name=PrefixedName("lid", "house"),
            collision=ShapeCollection([Box(scale=Scale(1.0, 1.0, 0.05))]),
        )
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world, parent=room, child=lid, axis=Vector3.Y()
            )
        )
    return world


def plugins_named(element: ElementTree.Element, system: GazeboModelSystem) -> list:
    """
    :param element: A model element.
    :param system: The system the plugins load.
    :return: The plugin elements of the model loading system.
    """
    return element.findall(
        f"{SDFElement.PLUGIN}[@{SDFAttribute.NAME}='{system.value.class_name}']"
    )


class TestSimulationSetup:
    """
    What a written world tells Gazebo about simulating it.
    """

    def test_world_loads_the_systems_it_is_given(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        systems = [GazeboSystem.PHYSICS, GazeboSystem.SENSORS, GazeboSystem.IMU]
        document = write_with(GazeboWriter(world=mini_world, systems=systems), tmp_path)
        assert [
            plugin.get(SDFAttribute.NAME)
            for plugin in document.findall(f"{SDFElement.WORLD}/{SDFElement.PLUGIN}")
        ] == [system.value.class_name for system in systems]

    def test_physics_steps_by_the_step_size(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        writer = GazeboWriter(world=mini_world, step_size=0.003)
        document = write_with(writer, tmp_path)
        step_size = document.findtext(
            f"{SDFElement.WORLD}/{SDFElement.PHYSICS}/{SDFElement.MAX_STEP_SIZE}"
        )
        assert float(step_size) == writer.step_size

    def test_default_sunlight_has_no_tint(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        writer = GazeboWriter(world=mini_world)
        document = write_with(writer, tmp_path)
        diffuse = document.findtext(
            f"{SDFElement.WORLD}/{SDFElement.LIGHT}/{SDFElement.DIFFUSE}"
        )
        red, green, blue, _ = (float(channel) for channel in diffuse.split())
        assert [red, green, blue] == [writer.sunlight.value] * 3

    def test_moving_parts_of_the_environment_pass_through_its_static_part(
        self, world_with_hinged_lid: World, tmp_path: Path
    ) -> None:
        """
        The moving parts of the model holding the world's root, such as an elevator's
        cabin, do not collide with its static part, such as the floor slabs the cabin
        passes; anything else collides with both.
        """
        writer = GazeboWriter(world=world_with_hinged_lid)
        fixed_part, moving_parts = writer.models
        document = write_with(writer, tmp_path)

        def bitmasks(model_name: str) -> set[str]:
            return {
                element.text
                for element in model_element_named(document, model_name).iter(
                    SDFElement.COLLIDE_BITMASK
                )
            }

        assert bitmasks(fixed_part.name) == {CollideBitmask.STATIC_ENVIRONMENT}
        assert bitmasks(moving_parts.name) == {CollideBitmask.MOVING_ENVIRONMENT}

    def test_free_model_collides_with_everything(
        self, world_with_free_object: World, tmp_path: Path
    ) -> None:
        cup = body_named(world_with_free_object, "cup")
        cup.collision = ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))], cup)
        document = write_and_read(world_with_free_object, tmp_path)
        cup_model = model_element_named(document, "cup")
        assert cup_model.find(f".//{SDFElement.COLLISION}") is not None
        assert cup_model.find(f".//{SDFElement.COLLIDE_BITMASK}") is None

    def test_position_servo_drives_its_joint_with_its_gains(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        hinge = mini_world.connections[0]
        gains = ServoGains(stiffness=10.0, damping=2.0, torque_limit=5.0)
        servo = PositionServo(name=PrefixedName("hinge_servo"), gains=gains)
        servo.add_dof(hinge.raw_dof)
        with mini_world.modify_world():
            mini_world.add_actuator(servo)
        writer = GazeboWriter(world=mini_world)
        _, moving_parts = writer.models

        document = write_with(writer, tmp_path)

        [controller] = plugins_named(
            model_element_named(document, moving_parts.name),
            GazeboModelSystem.JOINT_POSITION_CONTROLLER,
        )
        assert controller.findtext(SDFElement.JOINT_NAME) == hinge.name.name
        assert controller.findtext(SDFElement.TOPIC) == writer.joint_command_topic(
            moving_parts, hinge
        )
        assert [
            float(controller.findtext(tag))
            for tag in (
                SDFElement.P_GAIN,
                SDFElement.D_GAIN,
                SDFElement.CMD_MAX,
                SDFElement.CMD_MIN,
            )
        ] == [gains.stiffness, gains.damping, gains.torque_limit, -gains.torque_limit]

    def test_model_with_moving_joints_publishes_their_states(
        self, mini_world: World, tmp_path: Path
    ) -> None:
        writer = GazeboWriter(world=mini_world)
        fixed_part, moving_parts = writer.models
        document = write_with(writer, tmp_path)
        publisher = GazeboModelSystem.JOINT_STATE_PUBLISHER
        assert (
            len(
                plugins_named(
                    model_element_named(document, moving_parts.name), publisher
                )
            )
            == 1
        )
        assert not plugins_named(
            model_element_named(document, fixed_part.name), publisher
        )


# %% unsupported worlds


class TestUnsupportedWorlds:
    """
    Worlds that SDF cannot express, each of which is rejected naming what is wrong.
    """

    def test_drive_is_rejected(self, cylinder_bot_world: World, tmp_path: Path) -> None:
        (drive,) = [
            connection
            for connection in cylinder_bot_world.connections
            if isinstance(connection, OmniDrive)
        ]
        with pytest.raises(UnsupportedConnectionType) as error:
            write_and_read(cylinder_bot_world, tmp_path)
        assert error.value.connection is drive

    def test_degree_of_freedom_shared_across_models_is_rejected(
        self, world_with_free_object: World, tmp_path: Path
    ) -> None:
        """
        A mimic joint follows a joint of its own model, so a coupling between the joints
        of two models cannot be written.
        """
        table = body_named(world_with_free_object, "table")
        cup = body_named(world_with_free_object, "cup")
        with world_with_free_object.modify_world():
            drawer_slide = PrismaticConnection.create_with_dofs(
                world=world_with_free_object,
                parent=table,
                child=Body(name=PrefixedName("drawer", "house")),
                axis=Vector3.X(),
            )
            world_with_free_object.add_connection(drawer_slide)
            lid_hinge = RevoluteConnection(
                parent=cup,
                child=Body(name=PrefixedName("lid", "cup")),
                axis=Vector3.Y(),
                raw_dof=drawer_slide.raw_dof,
            )
            world_with_free_object.add_connection(lid_hinge)
        with pytest.raises(SharedDegreeOfFreedom) as error:
            write_and_read(world_with_free_object, tmp_path)
        assert error.value.connections == [drawer_slide, lid_hinge]

    def test_link_and_joint_of_one_name_are_rejected(
        self, world_with_two_bodies: tuple[World, Body, Body], tmp_path: Path
    ) -> None:
        world, parent, child = world_with_two_bodies
        with world.modify_world():
            world.add_connection(
                FixedConnection(parent=parent, child=child, name=child.name)
            )
        with pytest.raises(DuplicateSDFName) as error:
            write_and_read(world, tmp_path)
        assert error.value.name == child.name.name

    def test_body_named_like_the_world_is_rejected(
        self, world_with_two_bodies: tuple[World, Body, Body], tmp_path: Path
    ) -> None:
        world, parent, _ = world_with_two_bodies
        reserved_body = Body(name=PrefixedName(GazeboParser.world_link_name))
        with world.modify_world():
            world.add_connection(FixedConnection(parent=parent, child=reserved_body))
        with pytest.raises(ReservedSDFName) as error:
            write_and_read(world, tmp_path)
        assert error.value.name == GazeboParser.world_link_name


# %% the apartment


class TestApartmentWriting:
    """
    Writing the apartment, a world of meshes, doors and drawers with objects standing
    freely in it.
    """

    def test_writes_every_body(
        self, apartment_world_copy: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(apartment_world_copy, tmp_path)
        assert {body.name.name for body in parsed.bodies} == {
            body.name.name for body in apartment_world_copy.bodies
        } | {parsed.root.name.name}

    def test_every_joint_keeps_its_type(
        self, apartment_world_copy: World, tmp_path: Path
    ) -> None:
        parsed = write_and_parse(apartment_world_copy, tmp_path)
        parsed_types = {
            connection.name.name: type(connection) for connection in parsed.connections
        }
        source_types = {
            connection.name.name: type(connection)
            for connection in apartment_world_copy.connections
            if not isinstance(connection, Connection6DoF)
        }
        assert source_types.items() <= parsed_types.items()

    @pytest.mark.skipif(
        shutil.which("gz") is None, reason="Needs the Gazebo command line tools."
    )
    def test_gazebo_accepts_the_written_world(
        self, apartment_world_copy: World, tmp_path: Path
    ) -> None:
        file_path = str(tmp_path / "apartment.sdf")
        GazeboWriter(world=apartment_world_copy).write(file_path)
        check = subprocess.run(
            ["gz", "sdf", "-k", file_path], capture_output=True, text=True
        )
        assert check.returncode == 0, check.stdout + check.stderr
