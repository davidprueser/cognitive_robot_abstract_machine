from __future__ import annotations

import pytest

from experiments.shelf_generation_experiments.dataset_environment import (
    DatasetEnvironmentVariable,
)
from experiments.shelf_generation_experiments.exceptions import (
    MissingEnvironmentVariableError,
)


# %% reading the environment
def test_a_set_variable_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(DatasetEnvironmentVariable.SCENES_ROOT.value, "/data/scenes")

    assert DatasetEnvironmentVariable.SCENES_ROOT.read() == "/data/scenes"


def test_a_missing_variable_names_itself(monkeypatch) -> None:
    monkeypatch.delenv(DatasetEnvironmentVariable.SCENES_ROOT.value, raising=False)

    with pytest.raises(MissingEnvironmentVariableError) as error:
        DatasetEnvironmentVariable.SCENES_ROOT.read()

    assert error.value.variable_name == DatasetEnvironmentVariable.SCENES_ROOT.value
