"""
Sensor data collection and CAS initialization.

The collection reader is typically the first annotator in a pipeline: once each of its
cameras has new data, it creates a fresh CAS and fills it with that data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from timeit import default_timer

from py_trees.common import Status
from py_trees.display import unicode_tree
from typing_extensions import TYPE_CHECKING, Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.descriptors.camera_configs.components import TfComponent

if TYPE_CHECKING:
    from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
    from robokudo.io.camera_interface import CameraInterface

# %% collection reader


@dataclass(kw_only=True)
class CollectionReaderAnnotator(BaseAnnotator):
    """
    Creates a new CAS from the data of its cameras once each of them has new data.

    A query in the previous CAS is carried over into the new one.
    """

    camera_config: BaseCameraConfig
    """
    Configuration of the camera whose data fills the CAS; it also names the CAS's world
    and camera frames.
    """

    additional_camera_configs: list[BaseCameraConfig] = field(default_factory=list)
    """
    Configurations of further cameras whose data fills the same CAS.
    """

    ticks_before_warning: int = 40
    """
    Number of ticks without new data after which a warning is logged.
    """

    camera_interfaces: list[CameraInterface] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The interface of each configured camera, created in :meth:`setup`.
    """

    waiting_since: float | None = field(default=None, init=False)
    """
    When the reader started waiting for new data, as :func:`timeit.default_timer` time.
    """

    ticks_since_last_data: int = field(default=0, init=False)
    """
    Number of ticks since all cameras last had new data.
    """

    def add_collection_reader(self, camera_config: BaseCameraConfig) -> None:
        """
        Fill the CAS with the data of one more camera.

        Must be called before :meth:`setup`.

        :param camera_config: Configuration of the additional camera.
        """
        self.additional_camera_configs.append(camera_config)

    def setup(self, **kwargs: Any) -> bool:
        """
        Create the interface of each configured camera.
        """
        self.camera_interfaces = [
            camera_config.create_interface()
            for camera_config in [self.camera_config, *self.additional_camera_configs]
        ]
        return super().setup(**kwargs)

    def initialise(self) -> None:
        """
        Clear the feedback of every sibling, since the pipeline starts over.
        """
        for child in self.parent.children:
            child.feedback_message = ""

    def compute(self) -> Status:
        """
        :return: Success once a new CAS was created, running while waiting for data.
        """
        if self.waiting_since is None:
            self.waiting_since = default_timer()

        if not all(interface.has_new_data() for interface in self.camera_interfaces):
            return self.keep_waiting()

        self.ticks_since_last_data = 0
        waited_seconds = default_timer() - self.waiting_since
        self.waiting_since = None
        self.robokudo_logger.debug("Waited for %f seconds", waited_seconds)

        pipeline = self.get_parent_pipeline()
        query = pipeline.cas.query
        pipeline.create_new_cas()
        self.robokudo_logger.debug("%s: new CAS id=%s", self.name, pipeline.cas.cas_id)

        if isinstance(self.camera_config, TfComponent):
            cas = self.get_cas()
            cas.world_frame = self.camera_config.tf_to
            cas.camera_frame = self.camera_config.tf_from

        if query:
            pipeline.cas.query = query

        for interface in self.camera_interfaces:
            interface.set_data(self.get_cas())

        self.feedback_message = (
            f"Got sensor data... Waited for {waited_seconds} seconds"
        )
        self.robokudo_logger.info("Current Behavior Tree:")
        print(unicode_tree(self.parent))
        return Status.SUCCESS

    def keep_waiting(self) -> Status:
        """
        Count a tick without new data and warn if that lasted too long.

        :return: Running.
        """
        self.ticks_since_last_data += 1
        self.feedback_message = "Waiting for incoming sensor data ..."
        if self.ticks_since_last_data >= self.ticks_before_warning:
            self.robokudo_logger.warning(
                "Didn't receive sensor data since %d ticks.", self.ticks_since_last_data
            )
            for interface in self.camera_interfaces:
                self.robokudo_logger.warning(
                    "  CR data available?: %s", interface.has_new_data()
                )
            self.ticks_since_last_data = 0
        return Status.RUNNING
