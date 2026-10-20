"""
Storage writer annotator for RoboKudo.

This module provides an annotator for storing sensor data in MongoDB.
It supports:

* MongoDB integration
* common analysis structure data persistence
* View data storage
* Database management
* Configurable database settings

The module is used for:

* Data recording
* Offline processing
* Dataset creation
* Experiment logging
"""

from __future__ import annotations

import copy
import json
from timeit import default_timer

from py_trees.common import Status
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from typing_extensions import Any

import robokudo.world
from robokudo.annotators.core import BaseAnnotator
from robokudo.io.storage import Storage


class StorageWriter(BaseAnnotator):
    """Annotator for storing sensor data in MongoDB.

    This annotator provides methods to store sensor data in a MongoDB database,
    allowing for data recording and offline processing without using ROS bag files.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for storage writer."""

        class Parameters:
            """Parameter container for storage configuration."""

            def __init__(self) -> None:
                self.db_name: str = "rk_scenes"
                """Database name"""

                self.drop_database_on_storage: bool = True
                """Whether to clear database before recording"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "StorageWriter",
        descriptor: StorageWriter.Descriptor | None = None,
    ) -> None:
        """Initialize the storage writer. Minimal one-time init!

        :param name: Annotator name
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.storage = Storage(self.descriptor.parameters.db_name)
        self.storage_preparation_completed: bool = False
        """
        Whether storage setup has already run for this writer.
        """

    def setup(self, **kwargs: Any) -> bool:
        """
        Prepare storage before recording begins.
        It is guarded to execute only once, since setup might be called multiple times
        in the chosen tree executor.

        :param kwargs: Arguments forwarded to BaseAnnotator.setup()
        :return: Whether storage preparation succeeded.
        """
        if not super().setup(**kwargs):
            return False
        if self.storage_preparation_completed:
            return True

        if self.descriptor.parameters.drop_database_on_storage:
            self.storage.drop_database()

        self.storage_preparation_completed = True
        return True

    def update(self) -> Status:
        """Store current common analysis structure data in MongoDB.

        Creates a deep copy of the common analysis structure, flattens it into a dictionary,
        and stores both views and common analysis structure data in the database.

        :return: SUCCESS if storage successful, FAILURE otherwise
        """
        start_timer = default_timer()

        persist_common_analysis_structure = copy.deepcopy(
            self.get_common_analysis_structure()
        )

        flat_common_analysis_structure = (
            self.storage.generate_dict_from_real_common_analysis_structure(
                persist_common_analysis_structure
            )
        )
        flat_common_analysis_structure["view_ids"] = {}

        world = robokudo.world.world_instance()
        snapshot = WorldModelSnapshot(
            modifications=list(
                world.get_world_model_manager().model_modification_blocks
            ),
            ids=list(world.state.keys()),
            states=list(world.state.positions),
        )

        payload = snapshot.to_json()
        json_str = json.dumps(payload)
        flat_common_analysis_structure["world"] = json_str

        # step 1: persist each view
        self.storage.store_views_in_mongo(flat_common_analysis_structure)

        # step 2: persist common_analysis_structure and pointers to each view
        del flat_common_analysis_structure["views"]
        result = self.storage.store_common_analysis_structure_dict(
            flat_common_analysis_structure
        )
        if not result.acknowledged:
            self.robokudo_logger.error(
                f"mongo db error when trying to store common_analysis_structure"
            )
            return Status.FAILURE

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
