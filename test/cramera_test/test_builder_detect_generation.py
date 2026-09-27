"""
What the Plan Builder writes for an object carrying an annotation class: the object is
spawned as an annotation, a Detect step looks for it by that class, and a Pick step can
perceive it before grasping.

An object without a class is written as before.
"""

from __future__ import annotations

import ast

import pytest
from typing_extensions import Any, Dict, List

from cramera.model_catalog import BuilderStep, CatalogField, ModelCatalog
from cramera.paths import WEB_ROOT
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from .test_builder_real_lab_generation import (
    FILE_ENVIRONMENT_VALUE,
    calls_named,
    function_named,
    generate_demos,
)

CEREAL_POSE = {
    "x": 0.88,
    "y": -0.05,
    "z": 0.745,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": -1.571,
}
"""
Where the authored cereal stands.
"""


def cereal_object() -> Dict[str, Any]:
    """
    The cereal as the builder places a typed object: the catalog's class and mesh.
    """
    [model] = ModelCatalog.installed().to_payload()[CatalogField.OBJECTS]
    return {
        "id": "o1",
        "name": model[CatalogField.MESH],
        "mesh": model[CatalogField.MESH],
        "cls": model[CatalogField.CLASS],
        "import": model[CatalogField.IMPORT],
        "meshUrl": model[CatalogField.MESH_URL],
        "color": "#e6c07f",
        **CEREAL_POSE,
    }


def milk_object() -> Dict[str, Any]:
    return {
        "id": "o2",
        "name": "milk.stl",
        "mesh": "milk.stl",
        "x": 2.4,
        "y": 2.2,
        "z": 0.95,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "color": "#e6ecff",
    }


def steps() -> List[Dict[str, Any]]:
    return [
        {
            "id": "s1",
            "type": BuilderStep.DETECT.value,
            "params": {"object": "cheeze_it.obj"},
        },
        {
            "id": "s2",
            "type": BuilderStep.PICK.value,
            "params": {"object": "cheeze_it.obj", "arm": "BOTH", "perceive": True},
        },
        {
            "id": "s3",
            "type": BuilderStep.PLACE.value,
            "params": {
                "object": "cheeze_it.obj",
                "arm": "BOTH",
                "targetMode": "pose",
                "x": 1.9,
                "y": 2.6,
                "z": 0.6,
                "yaw": 0.0,
            },
        },
        {
            "id": "s4",
            "type": BuilderStep.PICK.value,
            "params": {"object": "milk.stl", "arm": "BOTH", "perceive": False},
        },
        {
            "id": "s5",
            "type": BuilderStep.TRANSPORT.value,
            "params": {
                "object": "cheeze_it.obj",
                "arm": "BOTH",
                "targetMode": "pose",
                "x": 1.0,
                "y": 1.0,
                "z": 0.8,
                "yaw": 0.0,
            },
        },
    ]


@pytest.fixture
def typed_demos() -> Dict[str, Any]:
    """
    Both output styles for a Stretch detecting, picking and placing the cereal.
    """
    return generate_demos(
        FILE_ENVIRONMENT_VALUE, steps=steps(), objects=[cereal_object(), milk_object()]
    )


@pytest.fixture
def typed_class(typed_demos) -> ast.Module:
    return ast.parse(typed_demos["class"])


def plan_statements(module: ast.Module) -> Dict[str, str]:
    """
    The assignments of ``build_plan``, by target name.
    """
    return {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in function_named(module, "build_plan").body
        if isinstance(node, ast.Assign)
    }


# %% looking for the object
def test_a_detect_step_looks_for_the_objects_class(typed_class) -> None:
    [detecting] = calls_named(typed_class, "DetectAction")

    assert (
        ast.unparse(detecting)
        == "DetectAction(DetectionTechnique.TYPES, object_sem_annotation=CheezeIt, accept_first_if_multiple=True)"
    )


@pytest.mark.parametrize("style", ["script", "class"])
def test_the_detect_action_and_its_technique_are_imported(typed_demos, style) -> None:
    module = ast.parse(typed_demos[style])

    imported = {
        (node.module, alias.name)
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("coraplex.robot_plans.actions.core.misc", "DetectAction") in imported
    assert ("coraplex.datastructures.enums", "DetectionTechnique") in imported
    assert (CheezeIt.__module__, CheezeIt.__name__) in imported


def test_a_detect_of_an_object_without_a_class_is_refused() -> None:
    demos = generate_demos(
        FILE_ENVIRONMENT_VALUE,
        steps=[
            {
                "id": "s1",
                "type": BuilderStep.DETECT.value,
                "params": {"object": "milk.stl"},
            }
        ],
        objects=[milk_object()],
    )

    assert list(demos["class"]) == ["refused"]
    assert list(demos["script"]) == ["refused"]


# %% picking and placing the annotated object
def test_the_object_is_resolved_once_by_its_class(typed_class) -> None:
    assignments = plan_statements(typed_class)

    assert (
        assignments["_cheeze_it"]
        == "world.get_semantic_annotations_by_type(CheezeIt)[0]"
    )


def test_a_pick_perceives_the_annotated_object_before_grasping(typed_class) -> None:
    picks = [ast.unparse(call) for call in calls_named(typed_class, "PickUpAction")]

    assert (
        "PickUpAction(_cheeze_it, Arms.BOTH, _grasp_s2, perceive_before_grasp=True)"
        in picks
    )


def test_the_grasp_is_taken_from_the_annotations_root(typed_class) -> None:
    assignments = plan_statements(typed_class)

    assert assignments["_grasp_s2"] == (
        "GraspDescription.robot_relative_default("
        "ViewManager.get_end_effector_view(Arms.BOTH, context.robot), "
        "_cheeze_it.root.global_pose, _cheeze_it.root)"
    )


def test_a_pick_of_a_plain_object_is_written_as_before(typed_class) -> None:
    picks = [ast.unparse(call) for call in calls_named(typed_class, "PickUpAction")]
    assignments = plan_statements(typed_class)

    assert "PickUpAction(HasRootBody(root=_pick_s4), Arms.BOTH, _grasp_s4)" in picks
    assert assignments["_pick_s4"] == "world.get_body_by_name('milk.stl')"


def test_a_place_puts_down_the_annotations_root(typed_class) -> None:
    [placing] = calls_named(typed_class, "PlaceAction")

    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in placing.keywords}
    assert keywords["object_designator"] == "_cheeze_it.root"


def test_a_transport_carries_the_annotations_root_without_perceiving(
    typed_class,
) -> None:
    [transporting] = calls_named(typed_class, "TransportAction")

    keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in transporting.keywords
    }
    assert keywords["object_designator"] == "HasRootBody(root=_cheeze_it.root)"
    assert "perceive_before_grasp" not in keywords


# %% spawning the annotated object
def test_the_annotated_object_is_listed_with_its_class_and_mesh(typed_class) -> None:
    [listed] = [
        node.value
        for node in typed_class.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "ANNOTATED_OBJECTS"
    ]

    [cereal] = listed.elts
    assert ast.unparse(cereal.elts[0]) == CheezeIt.__name__
    assert [ast.literal_eval(part) for part in cereal.elts[1:3]] == [
        cereal_object()["mesh"],
        cereal_object()["meshUrl"],
    ]
    assert [ast.literal_eval(part) for part in cereal.elts[3:]] == [
        CEREAL_POSE[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ]


def test_a_plain_object_stays_in_the_mesh_list_and_an_annotated_one_leaves_it(
    typed_class,
) -> None:
    [listed] = [
        node.value
        for node in typed_class.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "OBJECTS"
    ]

    assert [ast.literal_eval(spec)[0] for spec in listed.elts] == ["milk.stl"]


@pytest.mark.parametrize("style", ["script", "class"])
def test_the_annotated_object_is_spawned_free_to_move_from_its_resolved_mesh(
    typed_demos, style
) -> None:
    module = ast.parse(typed_demos[style])

    [spawning] = calls_named(module, ".get_annotation_specification")
    keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in spawning.keywords
    }
    assert (
        keywords["parent_connection_specification"] == "Connection6DoFSpecification()"
    )
    [resolving] = calls_named(spawning, ".resolve")
    assert ast.unparse(resolving) == "CompositePathResolver().resolve(mesh_url)"


def test_the_scene_counts_as_populated_only_with_the_annotated_object(
    typed_class,
) -> None:
    function = function_named(typed_class, "is_scene_populated")
    namespace = {
        "World": World,
        "OBJECTS": [],
        "ANNOTATED_OBJECTS": [(CheezeIt, "cheeze_it.obj")],
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "demo.py", "exec"),
        namespace,
    )
    empty = World.create_with_root_body("map")
    holding = World.create_with_root_body("map")
    with holding.modify_world():
        holding.add_connection(
            FixedConnection(
                parent=holding.root, child=Body(name=PrefixedName("cheeze_it.obj"))
            )
        )

    assert namespace[function.name](None, empty) is False
    assert namespace[function.name](None, holding) is True


# %% what the page offers
def test_the_page_offers_a_detect_block() -> None:
    page_script = (WEB_ROOT / "plan_builder.js").read_text(encoding="utf-8")

    assert "\n    %s: { name: " % BuilderStep.DETECT.value in page_script
