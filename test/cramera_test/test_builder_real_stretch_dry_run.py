"""
A demo for the real Stretch in the apartment lab, authored in the Plan Builder and
exported as a RobotDemonstration class, performs its plan as exported.

The plan is the first half of the reference demonstration
(``experiments.real_stretch_apartment_demo``): park, drive to the shelf, look at the
cereal's shelf layer, detect the cereal, pick it up perceiving it first, park, drive to
the bedside table and place it there. The export for the simulated robot is run; the one
for the real robot is checked in its source, since it needs the robot's world server.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from typing_extensions import Any, Dict, List

from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.demonstrations import RobotDemonstration
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from cramera.model_catalog import BuilderStep
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
    ApartmentFurniture,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF

from .test_builder_detect_generation import cereal_object
from .test_builder_real_lab_generation import MAP_ENVIRONMENT_VALUE, generate_demos

CEREAL_SHELF_LAYER_NAME = "shelf_layer2"
"""
The shelf layer the cereal starts on, as in the reference demonstration.
"""

CEREAL_SHELF_LAYER_T_CEREAL = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=-0.1, y=0.0, z=0.115
)
"""
Where the cereal starts on its shelf layer, as in the reference demonstration.
"""

BEDSIDE_TABLE_T_STANDING = HomogeneousTransformationMatrix.from_xyz_rpy(
    0.8, 0, 0, yaw=math.pi
)
"""
Where the robot stands at the bedside table, as in the reference demonstration.
"""

BEDSIDE_TABLE_T_PLACED = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=0.1, z=0.6, yaw=math.pi
)
"""
Where the cereal is put down on the bedside table.

Two centimetres above the reference demonstration's height: that one runs the real robot
without collision avoidance, whereas an exported demo avoids collisions, and the box's
bottom would sink into the table top at the reference height.
"""

PLACEMENT_TOLERANCE = 0.05
"""
How far from the authored pose the placed cereal may come to rest.
"""

ROBOT_START = {"x": 1.0, "y": 1.0, "yaw": 0.0}
"""
Where the Stretch starts, as in the reference demonstration.
"""


# %% the authored scene
def level_pose(frame_T_pose: np.ndarray) -> Dict[str, float]:
    """
    A world-frame matrix as the builder writes a pose: position and heading.
    """
    return {
        "x": round(float(frame_T_pose[0, 3]), 3),
        "y": round(float(frame_T_pose[1, 3]), 3),
        "z": round(float(frame_T_pose[2, 3]), 3),
        "yaw": round(float(math.atan2(frame_T_pose[1, 0], frame_T_pose[0, 0])), 3),
    }


@pytest.fixture(scope="module")
def apartment_furniture_meshes(apartment_meshes) -> None:
    """
    Skip when the apartment package installed lacks a furniture or cereal mesh.
    """
    for mesh in [name.value for name in ApartmentFurniture] + [cereal_object()["mesh"]]:
        if mesh.endswith((".dae", ".obj")) and not os.path.isfile(
            ApartmentEnvironment.mesh_path(mesh)
        ):
            pytest.skip(f"apartment mesh {mesh} is not installed")


@pytest.fixture(scope="module")
def authored_scene(apartment_furniture_meshes) -> Dict[str, Any]:
    """
    The cereal and the plan, with every pose in the world frame as the builder writes
    it, derived from where the map stands its furniture.
    """
    apartment = ApartmentEnvironment().get_world()
    shelf_layer = apartment.get_body_by_name(
        CEREAL_SHELF_LAYER_NAME
    ).global_pose.to_np()
    table = apartment.get_body_by_name(
        ApartmentFurniture.BEDSIDE_TABLE
    ).global_pose.to_np()
    cereal = level_pose(shelf_layer @ CEREAL_SHELF_LAYER_T_CEREAL.to_np())
    shelf_point = level_pose(shelf_layer)
    placed = level_pose(table @ BEDSIDE_TABLE_T_PLACED.to_np())
    cereal_object_placed = {**cereal_object(), **cereal, "roll": 0.0, "pitch": 0.0}
    steps: List[Dict[str, Any]] = [
        {"id": "s1", "type": BuilderStep.PARK_ARMS.value, "params": {"arm": "BOTH"}},
        {
            "id": "s2",
            "type": BuilderStep.NAVIGATE.value,
            "params": {"x": 1.2, "y": 1.2, "z": 0.0, "yaw": round(math.pi, 3)},
        },
        {"id": "s3", "type": BuilderStep.LOOK_AT.value, "params": shelf_point},
        {
            "id": "s4",
            "type": BuilderStep.NAVIGATE.value,
            "params": {"x": 0.8, "y": 0.6, "z": 0.0, "yaw": round(-math.pi / 2, 3)},
        },
        {"id": "s5", "type": BuilderStep.LOOK_AT.value, "params": shelf_point},
        {
            "id": "s6",
            "type": BuilderStep.DETECT.value,
            "params": {"object": cereal_object_placed["mesh"]},
        },
        {
            "id": "s7",
            "type": BuilderStep.PICK.value,
            "params": {
                "object": cereal_object_placed["mesh"],
                "arm": "BOTH",
                "perceive": True,
            },
        },
        {"id": "s8", "type": BuilderStep.PARK_ARMS.value, "params": {"arm": "BOTH"}},
        {
            "id": "s9",
            "type": BuilderStep.NAVIGATE.value,
            "params": level_pose(table @ BEDSIDE_TABLE_T_STANDING.to_np()),
        },
        {
            "id": "s10",
            "type": BuilderStep.PLACE.value,
            "params": {
                "object": cereal_object_placed["mesh"],
                "arm": "BOTH",
                "targetMode": "pose",
                **placed,
            },
        },
    ]
    return {"object": cereal_object_placed, "steps": steps, "placed": placed}


def export_demonstration(
    authored_scene: Dict[str, Any],
    execution_type: ExecutionType,
    monkeypatch: pytest.MonkeyPatch,
) -> type[RobotDemonstration]:
    """
    Generate the class export for the authored scene and import it.

    :param authored_scene: The cereal and the plan.
    :param execution_type: Which robot the exported demonstration drives.
    :param monkeypatch: Registers the generated module while its dataclass is defined.
    :return: The exported demonstration class.
    """
    source = generate_demos(
        MAP_ENVIRONMENT_VALUE,
        execution_type=execution_type,
        steps=authored_scene["steps"],
        objects=[authored_scene["object"]],
        instances=[{"model": Stretch.__name__, **ROBOT_START}],
        name=f"real_stretch_{execution_type.name.lower()}",
    )["class"]
    module = ModuleType(f"builder_real_stretch_{execution_type.name.lower()}")
    module.__file__ = str(
        Path(__file__).resolve().parents[2]
        / "coraplex"
        / "demos"
        / "coraplex_generated"
        / f"{module.__name__}.py"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, module.__file__, "exec"), vars(module))
    return next(
        item
        for item in vars(module).values()
        if isinstance(item, type)
        and issubclass(item, RobotDemonstration)
        and item is not RobotDemonstration
    )


# %% the simulated export performs the plan
@pytest.fixture(scope="module")
def performed_simulated_export(authored_scene) -> Dict[str, Any]:
    """
    The simulated export, run through its own world building, population, context and
    plan, exactly as ``RobotDemonstration.run`` would without a viewer.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        demonstration_type = export_demonstration(
            authored_scene, ExecutionType.SIMULATED, monkeypatch
        )
    demonstration = demonstration_type(used_robot=Stretch, collision_avoidance=True)
    world = demonstration.build_simulated_world()
    assert not demonstration.is_scene_populated(world)
    demonstration.populate_scene(world)
    populated = demonstration.is_scene_populated(world)
    context = demonstration.build_context(world)
    plan = demonstration.build_plan(context)
    with ExecutionEnvironment(
        execution_type=demonstration.execution_type,
        collision_avoidance=demonstration.collision_avoidance,
    ):
        plan.perform()
    return {
        "demonstration": demonstration,
        "world": world,
        "context": context,
        "plan": plan,
        "populated": populated,
    }


def test_the_export_drives_the_simulated_robot_by_default(performed_simulated_export):
    assert (
        performed_simulated_export["demonstration"].execution_type
        is ExecutionType.SIMULATED
    )


def test_the_scene_counts_as_populated_once_the_map_and_the_cereal_are_spawned(
    performed_simulated_export,
):
    world = performed_simulated_export["world"]

    assert performed_simulated_export["populated"]
    assert ApartmentEnvironment.is_populated(world)
    [cereal] = world.get_semantic_annotations_by_type(CheezeIt)
    assert isinstance(cereal.root.parent_connection, Connection6DoF)


def test_the_context_carries_the_stretch_motion_mappings(performed_simulated_export):
    context = performed_simulated_export["context"]
    demonstration = performed_simulated_export["demonstration"]

    assert (
        context.alternative_motion_mappings == demonstration.alternative_motion_mappings
    )
    assert isinstance(context.robot, Stretch)


def test_every_perceiving_and_manipulating_action_succeeds(performed_simulated_export):
    """
    The authored Detect step and the detection the perceiving pick adds before its
    grasp both succeed, as do the pick and the place.
    """
    plan = performed_simulated_export["plan"]

    [pick] = plan.get_nodes_by_designator_type(PickUpAction)
    [place] = plan.get_nodes_by_designator_type(PlaceAction)
    detections = plan.get_nodes_by_designator_type(DetectAction)
    assert pick.designator.perceive_before_grasp is True
    assert len(detections) == 2
    for node in detections + [pick, place]:
        assert node.status is LifeCycleValues.SUCCEEDED, node.designator


def test_the_cereal_comes_to_rest_at_the_authored_place(
    performed_simulated_export, authored_scene
):
    world = performed_simulated_export["world"]
    [cereal] = world.get_semantic_annotations_by_type(CheezeIt)
    placed = authored_scene["placed"]

    assert cereal.root.parent_connection.parent is world.root
    resting = cereal.root.global_pose.to_np()[:3, 3]
    assert (
        np.linalg.norm(resting - np.array([placed["x"], placed["y"], placed["z"]]))
        < PLACEMENT_TOLERANCE
    )


# %% the arm the builder offers for the Stretch
def test_both_arms_name_the_stretchs_one_arm(performed_simulated_export):
    """
    The builder offers ``BOTH`` for a single-arm robot; coraplex resolves it to the one
    arm, the same the reference demonstration names as ``LEFT``.
    """
    robot = performed_simulated_export["context"].robot

    assert ViewManager.get_arm_view(Arms.BOTH, robot) is ViewManager.get_arm_view(
        Arms.LEFT, robot
    )


# %% the real-robot export
def test_the_real_export_populates_a_world_that_holds_only_the_robot(
    authored_scene, monkeypatch
):
    """
    The world a world server serves holds only the robot; the exported demonstration
    spawns the map and the cereal into it, and recognizes them afterwards.
    """
    demonstration_type = export_demonstration(
        authored_scene, ExecutionType.REAL, monkeypatch
    )
    demonstration = demonstration_type(
        used_robot=Stretch, execution_type=ExecutionType.REAL
    )
    world = World.create_with_root_body("map")

    assert not demonstration.is_scene_populated(world)
    demonstration.populate_scene(world)
    assert demonstration.is_scene_populated(world)
    [cereal] = world.get_semantic_annotations_by_type(CheezeIt)
    assert cereal.root.name.name == authored_scene["object"]["mesh"]
