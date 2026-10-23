"""
Example analysis engine that perceives the objects on the apartment's shelf through the
head-mounted Kinect of a PR2 Giskard simulates.

The Kinect shows the apartment as it really is, rendered from where the world Giskard
serves holds the camera. Objects the world already knows are moved to where they were
seen, and the others are added to it::

    ros2 launch giskardpy_ros giskardpy_pr2_standalone.launch.py
    python -m robokudo.examples.build_apartment_around_robot
    ros2 run robokudo_ros main _ros_pkg=robokudo.examples _ae=apartment_shelf_from_pr2_kinect
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.tabletop_objects import TabletopObjectsEngine

# %% analysis engine


@dataclass
class AnalysisEngine(TabletopObjectsEngine):
    """
    Segments the objects on the second layer of the apartment's shelf and puts them into
    the world.
    """

    shelf_region: VolumetricBoundingBox = field(
        default_factory=lambda: VolumetricBoundingBox(
            min_x=0.48,
            min_y=-0.28,
            min_z=0.60,
            max_x=1.28,
            max_y=0.05,
            max_z=1.00,
            origin=HomogeneousTransformationMatrix(),
        )
    )
    """
    The region of the world the second layer of the shelf and the objects on it are in,
    without the side and back panels of the shelf.
    """

    resolution: int = 1024
    """
    How many pixels the rendered images are wide and high, as many per degree as the
    Kinect has.
    """

    def name(self) -> str:
        return "apartment_shelf_from_pr2_kinect"

    def camera_descriptor(self) -> CollectionReaderAnnotator.Descriptor:
        return CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer_robot_camera",
            world_descriptor_ros_package="robokudo.examples",
            world_descriptor_name="apartment_reality",
            resolution=self.resolution,
        )

    def crop_descriptor(self) -> PointcloudCropAnnotator.Descriptor:
        descriptor = PointcloudCropAnnotator.Descriptor()
        parameters = descriptor.parameters
        parameters.relative_to_world = True
        parameters.min_x = self.shelf_region.min_x
        parameters.min_y = self.shelf_region.min_y
        parameters.min_z = self.shelf_region.min_z
        parameters.max_x = self.shelf_region.max_x
        parameters.max_y = self.shelf_region.max_y
        parameters.max_z = self.shelf_region.max_z
        return descriptor
