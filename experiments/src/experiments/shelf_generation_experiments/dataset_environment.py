from __future__ import annotations

import os
from enum import StrEnum

from experiments.shelf_generation_experiments.exceptions import (
    MissingEnvironmentVariableError,
)


# %% dataset locations
class DatasetEnvironmentVariable(StrEnum):
    """
    The environment variables that locate the sage10k dataset and the databases built
    from it.
    """

    RAW_LAYOUTS_ROOT = "SAGE10K_LAYOUTS_ROOT"
    """
    Directory holding one sub-directory per downloaded sage10k layout, each with exactly
    one ``layout_*.json`` file.
    """

    RAW_DATABASE_URI = "SAGE10k_DATABASE_URI"
    """
    Connection string of the database the raw sage10k layouts are imported into.
    """

    PROCESSED_DATABASE_URI = "SAGE10K_PROCESSED_DATABASE_URI"
    """
    Connection string of the database preprocessing writes its shelves and objects to.
    """

    SCENES_ROOT = "SAGE10K_SCENES_ROOT"
    """
    Directory holding one sub-directory per downloaded sage10k scene, each with an
    ``objects/`` folder of PLY meshes and textures.
    """

    def read(self) -> str:
        """
        :raises MissingEnvironmentVariableError: If the variable is not set.
        :return: The value of this variable in the current environment.
        """
        value = os.environ.get(self.value)
        if value is None:
            raise MissingEnvironmentVariableError(variable_name=self.value)
        return value
