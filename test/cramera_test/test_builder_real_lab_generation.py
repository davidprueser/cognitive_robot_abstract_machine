"""
What the Plan Builder writes for the real lab: a demo whose world is populated by the
apartment map instead of being read from a world file, and a RobotDemonstration class
that drives the real robot or a simulated one.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typing_extensions import Any, Dict, List

from coraplex.datastructures.enums import ExecutionType
from cramera.model_catalog import (
    CatalogField,
    EnvironmentKind,
    ModelCatalog,
)
from cramera.paths import WEB_ROOT
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)

MAP_ENVIRONMENT_VALUE = f"{EnvironmentKind.MAP}:{ApartmentEnvironment.__name__}"
"""
The value the builder's environment select carries for the real-lab apartment.
"""

FILE_ENVIRONMENT_VALUE = "apartment.urdf"
"""
The value the select carries for the apartment world file.
"""


# %% generating through the real page script
def generate_demos(
    environment: str,
    execution_type: ExecutionType = ExecutionType.SIMULATED,
    steps: List[Dict[str, Any]] | None = None,
    objects: List[Dict[str, Any]] | None = None,
    instances: List[Dict[str, Any]] | None = None,
    robot: str = "Stretch",
    name: str = "real_lab",
) -> Dict[str, Any]:
    """
    Run both page generators on one authored scene.

    :param environment: The environment select's value.
    :param execution_type: Which robot the demonstration class drives.
    :param steps: The plan's steps, in the builder's form.
    :param objects: The placed objects, in the builder's form.
    :param instances: The robots standing in the scene, as the page always has at least
        one; ``None`` exercises the single-robot form the harness starts in.
    :param robot: The selected robot model.
    :param name: The demo's name.
    :return: The script and class outputs, or what a generator refused.
    """
    if shutil.which("node") is None:
        pytest.skip("Node.js is required for the browser generator")
    catalog = ModelCatalog.installed().to_payload()
    scenario = {
        "robots": catalog[CatalogField.ROBOTS],
        "catalog": catalog,
        "objects": objects or [],
        "captured": {},
        "steps": steps or [],
        "robotXY": {"x": 1.0, "y": 1.0},
        "selections": {
            "pb-robot": robot,
            "pb-env": environment,
            "pb-execution": execution_type.name,
            "pb-name": name,
        },
    }
    if instances is not None:
        scenario["instances"] = instances
        scenario["activeIdentifier"] = "robot_1"
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


def calls_named(tree: ast.AST, name: str) -> List[ast.Call]:
    """
    Every call of a plain name or of an attribute of that name.
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(name)
    ]


def function_named(module: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


@pytest.fixture(params=[None, [{"model": "Stretch", "x": 1.0, "y": 1.0, "yaw": 0.0}]])
def instances(request: pytest.FixtureRequest) -> List[Dict[str, Any]] | None:
    """
    The scene once as the harness starts it and once as the page authors it, with the
    robot as an instance of a robot scene.
    """
    return request.param


@pytest.fixture
def map_demos(instances) -> Dict[str, Any]:
    """
    Both output styles for a Stretch in the real-lab apartment.
    """
    return generate_demos(MAP_ENVIRONMENT_VALUE, instances=instances)


# %% a world populated by the map rather than read from a file
@pytest.mark.parametrize("style", ["script", "class"])
def test_a_map_environment_is_not_read_from_a_world_file(map_demos, style) -> None:
    module = ast.parse(map_demos[style])

    assert calls_named(module, "from_urdf") == []
    assert "_WORLDS" not in {
        node.id for node in ast.walk(module) if isinstance(node, ast.Name)
    }


@pytest.mark.parametrize("style", ["script", "class"])
def test_a_map_environment_is_imported_and_populates_the_world(
    map_demos, style
) -> None:
    module = ast.parse(map_demos[style])

    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        and node.module == ApartmentEnvironment.__module__
        for alias in node.names
    }
    assert imported == {ApartmentEnvironment.__name__}
    [populating] = calls_named(module, ".populate")
    assert (
        ast.unparse(populating) == f"{ApartmentEnvironment.__name__}().populate(world)"
    )


def test_the_class_populates_the_map_when_spawning_its_scene(map_demos) -> None:
    module = ast.parse(map_demos["class"])

    populate_scene = function_named(module, "populate_scene")
    assert len(calls_named(populate_scene, ".populate")) == 1


def test_the_class_counts_the_scene_as_populated_only_with_the_map(map_demos) -> None:
    module = ast.parse(map_demos["class"])

    populated = function_named(module, "is_scene_populated")
    [checking] = calls_named(populated, ".is_populated")
    assert (
        ast.unparse(checking) == f"{ApartmentEnvironment.__name__}.is_populated(world)"
    )


def test_a_map_environment_has_no_placement_annotations_of_a_file(map_demos) -> None:
    for style in ("script", "class"):
        assert calls_named(ast.parse(map_demos[style]), "PlacementAnnotations") == []


def test_the_robot_scene_of_a_map_environment_is_built_without_a_file() -> None:
    demos = generate_demos(
        MAP_ENVIRONMENT_VALUE,
        instances=[{"model": "Stretch", "x": 1.0, "y": 1.0, "yaw": 0.0}],
    )

    for style in ("script", "class"):
        [building] = calls_named(ast.parse(demos[style]), ".build_world")
        assert ast.unparse(building) == "ROBOT_SCENE.build_world()"


def test_a_single_robot_in_a_map_environment_is_spawned_into_an_empty_world() -> None:
    demos = generate_demos(MAP_ENVIRONMENT_VALUE)

    for style in ("script", "class"):
        [specification] = [
            call
            for call in calls_named(ast.parse(demos[style]), "WorldSpecification")
            if ast.unparse(call.func) == "WorldSpecification"
        ]
        assert [keyword.arg for keyword in specification.keywords] == ["robots"]


def test_a_world_file_is_still_read_as_before() -> None:
    demos = generate_demos(FILE_ENVIRONMENT_VALUE)

    for style in ("script", "class"):
        module = ast.parse(demos[style])
        assert len(calls_named(module, "from_urdf")) == 1
        assert calls_named(module, ".populate") == []


# %% driving the real robot
@pytest.fixture(params=[ExecutionType.SIMULATED, ExecutionType.REAL])
def execution_type(request: pytest.FixtureRequest) -> ExecutionType:
    return request.param


@pytest.fixture
def real_lab_class(execution_type) -> ast.Module:
    """
    The class output for the real lab, driving the robot the fixture names.
    """
    return ast.parse(
        generate_demos(MAP_ENVIRONMENT_VALUE, execution_type=execution_type)["class"]
    )


def test_main_runs_the_demonstration_with_the_chosen_robot(
    real_lab_class, execution_type
) -> None:
    main = function_named(real_lab_class, "main")

    [default] = main.args.defaults
    assert ast.unparse(default) == f"ExecutionType.{execution_type.name}"
    [run] = calls_named(main, ".run")
    keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in run.func.value.keywords
    }
    assert keywords["execution_type"] == "execution_type"


def test_the_context_carries_the_alternative_motion_mappings(real_lab_class) -> None:
    [context] = calls_named(function_named(real_lab_class, "build_context"), "Context")

    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in context.keywords}
    assert keywords["alternative_motion_mappings"] == "self.alternative_motion_mappings"


def test_reasoning_about_a_world_file_is_left_to_a_simulated_run() -> None:
    """
    The world a real robot's world server serves is not reasoned about, as the reference
    demonstration leaves it; a world read from a file is, as before.
    """
    module = ast.parse(generate_demos(FILE_ENVIRONMENT_VALUE)["class"])
    build_context = function_named(module, "build_context")

    [guard] = [
        node
        for node in ast.walk(build_context)
        if isinstance(node, ast.If) and calls_named(node, "WorldReasoner")
    ]
    assert ast.unparse(guard.test) == "self.execution_type is not ExecutionType.REAL"
    assert len(calls_named(guard, "PlacementAnnotations")) == 1


def test_a_map_is_not_reasoned_about(real_lab_class) -> None:
    """
    The map is annotated as it is spawned, and the reference demonstration never reasons
    about it.
    """
    assert calls_named(real_lab_class, "WorldReasoner") == []
    imported = {
        alias.name
        for node in ast.walk(real_lab_class)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "WorldReasoner" not in imported


def test_a_real_robot_in_a_world_file_is_refused() -> None:
    demos = generate_demos(FILE_ENVIRONMENT_VALUE, execution_type=ExecutionType.REAL)

    assert list(demos["class"]) == ["refused"]
    assert demos["class"]["refused"]
    assert isinstance(demos["script"], str)


def test_a_real_robot_is_resolved_by_its_type_in_the_served_world() -> None:
    """
    A world server names the robot as it likes, not by the scene's instance namespace.
    """
    demos = generate_demos(
        MAP_ENVIRONMENT_VALUE,
        execution_type=ExecutionType.REAL,
        instances=[{"model": "Stretch", "x": 1.0, "y": 1.0, "yaw": 0.0}],
    )
    module = ast.parse(demos["class"])

    build_context = function_named(module, "build_context")
    [by_type] = calls_named(build_context, "get_semantic_annotations_by_type")
    assert ast.unparse(by_type.args[0]) == "self.used_robot"
    assert calls_named(build_context, "selected_robot") == []


def test_a_simulated_robot_is_the_selected_instance_of_its_scene() -> None:
    demos = generate_demos(
        MAP_ENVIRONMENT_VALUE,
        execution_type=ExecutionType.SIMULATED,
        instances=[{"model": "Stretch", "x": 1.0, "y": 1.0, "yaw": 0.0}],
    )
    module = ast.parse(demos["class"])

    build_context = function_named(module, "build_context")
    [selected] = calls_named(build_context, "selected_robot")
    assert ast.unparse(selected) == "ROBOT_SCENE.selected_robot(world)"
