"""
What the scene panel draws per frame, and what it leaves out.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% the scene panel's drawing cost
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_drawing_cost() -> None:
    """
    Exercise the shadow map, the occlusion chain and the render loop of the actual
    panel.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_scene_drawing_cost.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
