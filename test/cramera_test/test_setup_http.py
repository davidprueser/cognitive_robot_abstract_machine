"""
The Plan Builder saves demo setups as new files through the viewer's server and opens
them again.
"""

from __future__ import annotations

import urllib.error

import pytest

from cramera.demo_setup import SetupField, SetupLibrary
from cramera.model_catalog import BuilderStep
from cramera.plan_steps import StepField

from .test_server import get, get_json, post, server  # noqa: F401


@pytest.fixture()
def setups_directory(monkeypatch, tmp_path):
    """
    A scratch data directory, so saved setups land in the test's own space.
    """
    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path / "data"))
    return SetupLibrary().directory


def builder_setup(x: float = 1.0) -> dict:
    """
    A setup in the Plan Builder's form: one PR2 in a scanned lab, looking at a point.
    """
    return {
        SetupField.ENVIRONMENT: {SetupField.PATH: "/lab/world.usda"},
        SetupField.ROBOTS: [
            {
                SetupField.IDENTIFIER: "robot_1",
                SetupField.LABEL: "PR2",
                SetupField.MODEL: "PR2",
                SetupField.X: x,
                SetupField.Y: 2.0,
                SetupField.YAW: 0.0,
                SetupField.STEPS: [
                    {
                        StepField.TYPE: BuilderStep.LOOK_AT,
                        StepField.PARAMETERS: {"x": 1.0, "y": 0.0, "z": 1.0},
                    }
                ],
            }
        ],
    }


def test_a_saved_setup_is_listed(server, setups_directory):  # noqa: F811
    status, answer = post(
        server + "/api/setup/save", {"name": "moved_about", "setup": builder_setup()}
    )

    assert status == 200 and answer["ok"]
    assert get_json(server + "/api/setup/list")["names"] == ["moved_about"]


def test_a_saved_setup_opens_as_it_was_saved(server, setups_directory):  # noqa: F811
    post(server + "/api/setup/save", {"name": "demo", "setup": builder_setup(x=3.5)})

    opened = get_json(server + "/api/setup/open?name=demo")["setup"]

    [robot] = opened["robots"]
    assert robot["x"] == pytest.approx(3.5)
    assert robot["steps"][0]["type"] == BuilderStep.LOOK_AT
    assert opened["environment"]["path"] == "/lab/world.usda"


def test_a_setup_is_saved_as_a_new_file_only(server, setups_directory):  # noqa: F811
    post(server + "/api/setup/save", {"name": "demo", "setup": builder_setup(x=1.0)})

    status, _ = post(
        server + "/api/setup/save", {"name": "demo", "setup": builder_setup(x=9.0)}
    )

    assert status == 409
    opened = get_json(server + "/api/setup/open?name=demo")["setup"]
    assert opened["robots"][0]["x"] == pytest.approx(1.0)


def test_a_robot_model_nobody_installed_is_refused(
    server, setups_directory
):  # noqa: F811
    setup = builder_setup()
    setup["robots"][0]["model"] = "Unheard"

    status, answer = post(server + "/api/setup/save", {"name": "demo", "setup": setup})

    assert status == 400 and not answer["ok"]


def test_a_setup_that_was_never_saved_is_not_found(
    server, setups_directory
):  # noqa: F811
    with pytest.raises(urllib.error.HTTPError) as refused:
        get(server + "/api/setup/open?name=never")

    assert refused.value.code == 404
