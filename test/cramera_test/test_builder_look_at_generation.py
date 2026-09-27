"""
What the Plan Builder writes for a scene opened from a demo setup: a Look at step as the
coraplex action pointing the robot's camera at its point, and the environment's joints
standing where the setup says.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cramera.model_catalog import BuilderStep, ModelCatalog
from cramera.paths import WEB_ROOT

LOOKED_AT = {"x": 1.5, "y": -0.5, "z": 1.2}
"""
The point the generated plan looks at.
"""

DOOR_POSITIONS = {"lab/wall_T_lab/door_0": 1.2}
"""
Where the opened setup stands the environment's one door.
"""


@pytest.fixture
def looking_demos() -> dict[str, str]:
    """
    Both output styles for one Tracy looking at a point.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    scenario = {
        "robots": ModelCatalog.installed().to_payload()["robots"],
        "instances": [{"model": "Tracy", "x": 0.0, "y": 0.0, "yaw": 0.0}],
        "activeIdentifier": "robot_1",
        "objects": [],
        "captured": {},
        "steps": [{"type": BuilderStep.LOOK_AT.value, "params": LOOKED_AT}],
        "environmentJointPositions": DOOR_POSITIONS,
        "selections": {
            "pb-robot": "Tracy",
            "pb-env": "apartment.urdf",
            "pb-name": "looking",
        },
    }
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).parent / "dataset" / "generate_builder_demo.js"),
            str(WEB_ROOT),
        ],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def calls_named(module: ast.Module, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


@pytest.mark.parametrize("style", ["script", "class"])
def test_a_look_at_step_looks_at_its_point(looking_demos, style) -> None:
    module = ast.parse(looking_demos[style])

    [looking] = calls_named(module, "LookAtAction")
    [target] = looking.args
    assert [ast.literal_eval(argument) for argument in target.args] == [
        LOOKED_AT["x"],
        LOOKED_AT["y"],
        LOOKED_AT["z"],
    ]


@pytest.mark.parametrize("style", ["script", "class"])
def test_the_generated_demo_imports_the_look_at_action(looking_demos, style) -> None:
    module = ast.parse(looking_demos[style])

    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and node.module == "coraplex.robot_plans.actions.core.navigation"
        for alias in node.names
    }
    assert "LookAtAction" in imported


@pytest.mark.parametrize("style", ["script", "class"])
def test_the_environments_joints_stand_where_the_setup_says(
    looking_demos, style
) -> None:
    module = ast.parse(looking_demos[style])

    [scene] = calls_named(module, "RobotScene")
    arguments = {keyword.arg: keyword.value for keyword in scene.keywords}
    assert ast.literal_eval(arguments["environment_joint_positions"]) == DOOR_POSITIONS
