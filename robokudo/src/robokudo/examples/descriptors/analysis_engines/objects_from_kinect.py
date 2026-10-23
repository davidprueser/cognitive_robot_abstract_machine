"""
Example analysis engine that perceives the objects on a table from the topics of the
PR2's head-mounted Kinect into the semantic digital twin.

The camera is placed where TF reports it relative to ``map``. Without a process serving
a shared world, RoboKudo perceives into an empty world of its own, which then holds
``map``, the camera and every segmented object. A recording of the topics and of TF
stands in for the robot::

    ros2 bag play --loop <recording>
    ros2 run robokudo_ros main _ros_pkg=robokudo.examples _ae=objects_from_kinect
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.tabletop_objects import TabletopObjectsEngine

# %% analysis engine


@dataclass
class AnalysisEngine(TabletopObjectsEngine):
    """
    Segments the objects on the table the Kinect sees and adds them to the perception
    world.
    """

    def name(self) -> str:
        return "objects_from_kinect"

    def camera_descriptor(self) -> CollectionReaderAnnotator.Descriptor:
        return CollectionReaderDescriptorFactory.create_descriptor("kinect")
