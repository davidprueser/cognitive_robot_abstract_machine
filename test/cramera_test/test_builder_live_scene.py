"""
The Plan Builder's live scene shows the world without performing a plan, so the robots
can be stood elsewhere in any environment, however their motions would fare there.
"""

from __future__ import annotations

import ast
import runpy

import pytest

from coraplex.plans.plan_node import PlanNode
from cramera.body_geometry import DrawnGeometry
from cramera.model_catalog import ModelCatalog
from semantic_digital_twin.robots.pr2 import PR2

from .test_builder_real_lab_generation import generate_demos

# %% a plan without steps


class PlanPerformedError(Exception):
    """
    Raised in place of performing a plan, so a script that performs one fails.
    """


def refuse_to_perform(plan_node: PlanNode) -> None:
    """
    Stand in for performing a plan.

    :param plan_node: The plan that would have been performed.
    :raises PlanPerformedError: Always.
    """
    raise PlanPerformedError(plan_node)


def apartment_path() -> str:
    """
    :return: The apartment the builder offers, as its environment select names it.
    """
    [environment] = [
        environment
        for environment in ModelCatalog.installed().environments
        if environment.path.endswith("apartment.urdf")
    ]
    return environment.path


def test_a_plan_without_steps_brings_up_the_world_and_performs_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The script the live scene runs builds the world, robot included, and performs no
    plan, so a robot standing where none of its motions would be feasible still shows.
    """
    script = generate_demos(apartment_path(), robot=PR2.__name__)["script"]
    path = tmp_path / "live_scene.py"
    path.write_text(script)
    monkeypatch.setenv("CORAPLEX_VISUALIZATION", "NONE")
    monkeypatch.setattr(PlanNode, "perform", refuse_to_perform)

    namespace = runpy.run_path(str(path), run_name="__main__")

    [robot] = namespace["world"].get_semantic_annotations_by_type(PR2)
    assert "plan" not in namespace


# %% how the live scene draws its environment


def drawn_geometry_assignments(script: str) -> list[ast.expr]:
    """
    :param script: A generated script.
    :return: The values it sets the drawn geometry of a bridge's environment to.
    """
    return [
        node.value
        for node in ast.walk(ast.parse(script))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == "environment_geometry"
    ]


def test_an_environment_drawn_as_its_collision_is_drawn_so_in_the_live_scene(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = generate_demos(
        apartment_path(),
        robot=PR2.__name__,
        environment_geometry=DrawnGeometry.COLLISION,
    )["script"]
    path = tmp_path / "live_scene.py"
    path.write_text(script)
    monkeypatch.setenv("CORAPLEX_VISUALIZATION", "NONE")

    runpy.run_path(str(path), run_name="__main__")

    [value] = drawn_geometry_assignments(script)
    assert (
        ast.unparse(value) == f"{DrawnGeometry.__name__}.{DrawnGeometry.COLLISION.name}"
    )


def test_an_environment_drawn_as_it_looks_leaves_the_bridge_as_it_starts() -> None:
    script = generate_demos(apartment_path(), robot=PR2.__name__)["script"]

    assert drawn_geometry_assignments(script) == []
