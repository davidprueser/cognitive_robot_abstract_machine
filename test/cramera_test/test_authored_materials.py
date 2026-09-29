"""
Authored scene materials and asynchronous object placement retain their intent.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% authored scene rendering
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_authored_materials() -> None:
    """
    Exercise material preservation and delayed mesh loading in the actual panel.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_authored_materials.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
