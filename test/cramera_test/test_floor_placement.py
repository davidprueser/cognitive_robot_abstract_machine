"""
Standing the active robot where the floor of the 3D scene is clicked.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

JAVASCRIPT_TESTS = Path(__file__).parent / "js"


def run_node(source: str) -> None:
    """
    Run one Node test module and fail with its output.

    :param source: The module's file name under the ``js`` directory.
    """
    result = subprocess.run(
        ["node", "--test", str(JAVASCRIPT_TESTS / source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# %% the scene's half and the builder's half
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_scene_answers_a_floor_pick() -> None:
    run_node("test_scene_floor_pick.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_builder_stands_the_robot_where_the_floor_was_clicked() -> None:
    run_node("test_builder_place_robot_by_click.js")
