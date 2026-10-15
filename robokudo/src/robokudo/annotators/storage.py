"""
Storage writer annotator for RoboKudo.

This module provides an annotator for storing sensor data in MongoDB.
It supports:

* MongoDB integration
* CAS data persistence
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

from dataclasses import dataclass, field

import copy
import json

from py_trees.common import Status
from semantic_digital_twin.adapters.ros.messages import WorldModelSnapshot
from typing_extensions import Any

import robokudo.world
from robokudo.annotators.core import BaseAnnotator
from robokudo.io.storage import Storage


@dataclass(kw_only=True)
class StorageWriter(BaseAnnotator):
    """
    Annotator for storing sensor data in MongoDB.

    This annotator provides methods to store sensor data in a MongoDB database, allowing
    for data recording and offline processing without using ROS bag files.
    """

    db_name: str = "rk_scenes"
    """
    Name of the MongoDB database the CAS data is written to.
    """

    drop_database_on_storage: bool = True
    """
    Whether to clear database before recording.
    """

    storage: Storage = field(init=False, repr=False)
    """
    The database the CAS data is written to.
    """

    storage_preparation_completed: bool = field(default=False, init=False)
    """
    Whether storage setup has already run for this writer.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.storage = Storage(self.db_name)

    def setup(self, **kwargs: Any) -> bool:
        """
        Prepare storage before recording begins. It is guarded to execute only once,
        since setup might be called multiple times in the chosen tree executor.

        :param kwargs: Arguments forwarded to BaseAnnotator.setup()
        :return: Whether storage preparation succeeded.
        """
        if not super().setup(**kwargs):
            return False
        if self.storage_preparation_completed:
            return True

        if self.drop_database_on_storage:
            self.storage.drop_database()

        self.storage_preparation_completed = True
        return True

    def compute(self) -> Status:
        """
        Store current CAS data in MongoDB.

        Creates a deep copy of the CAS, flattens it into a dictionary, and stores both
        views and CAS data in the database.

        :return: SUCCESS if storage successful, FAILURE otherwise
        """
        persist_cas = copy.deepcopy(self.get_cas())

        flat_cas = self.storage.generate_dict_from_real_cas(persist_cas)
        flat_cas["view_ids"] = {}

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
        flat_cas["world"] = json_str

        # step 1: persist each view
        self.storage.store_views_in_mongo(flat_cas)

        # step 2: persist cas and pointers to each view
        del flat_cas["views"]
        result = self.storage.store_cas_dict(flat_cas)
        if not result.acknowledged:
            self.robokudo_logger.error(f"mongo db error when trying to store cas")
            return Status.FAILURE

        return Status.SUCCESS
