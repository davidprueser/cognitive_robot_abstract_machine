import os
from copy import deepcopy

import pytest

from krrood.entity_query_language.symbol_graph import SymbolGraph
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.connections import OmniDrive
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(autouse=True)
def cleanup_after_test():
    # Setup: runs before each krrood_test
    SymbolGraph()
    yield
    SymbolGraph().clear()


@pytest.fixture(autouse=True, scope="session")
def cleanup_ros():
    """
    Fixture to ensure that ROS is properly cleaned up after all tests.
    """
    if os.environ.get("ROS_VERSION") == "2":
        import rclpy

        if not rclpy.ok():
            rclpy.init()
    yield
    if os.environ.get("ROS_VERSION") == "2":
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture(scope="session")
def pr2_world_setup():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    pr2 = os.path.join(urdf_dir, "pr2_calibrated_with_ft.urdf")
    pr2_parser = URDFParser.from_file(file_path=pr2)
    world_with_pr2 = pr2_parser.parse()
    with world_with_pr2.modify_world():
        pr2_root = world_with_pr2.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_pr2.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_with_pr2
        )
        world_with_pr2.add_connection(c_root_bf)

    return world_with_pr2


@pytest.fixture(scope="session")
def hsr_world_setup():
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "pycram",
        "resources",
        "robots",
    )
    hsr = os.path.join(urdf_dir, "hsrb.urdf")
    hsr_parser = URDFParser.from_file(file_path=hsr)
    world_with_hsr = hsr_parser.parse()
    with world_with_hsr.modify_world():
        hsr_root = world_with_hsr.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_with_hsr.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=hsr_root, world=world_with_hsr
        )
        world_with_hsr.add_connection(c_root_bf)

    return world_with_hsr


@pytest.fixture(scope="session")
def apartment_world_setup():
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "pycram",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    apartment_world.merge_world_at_pose(
        milk_world,
        TransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        ),
    )
    apartment_world.merge_world_at_pose(
        cereal_world,
        TransformationMatrix.from_xyz_rpy(
            2.37, 1.8, 1.05, reference_frame=apartment_world.root
        ),
    )
    milk_view = Milk(body=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


@pytest.fixture(scope="session")
def pr2_apartment_world(pr2_world_setup, apartment_world_setup):
    pr2_copy = deepcopy(pr2_world_setup)
    apartment_copy = deepcopy(apartment_world_setup)

    apartment_copy.merge_world_at_pose(
        pr2_copy,
        TransformationMatrix.from_xyz_quaternion(
            1.3, 2, 0, reference_frame=apartment_copy.root
        ),
    )
    return apartment_copy
