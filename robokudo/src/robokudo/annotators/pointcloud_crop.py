"""
Point cloud cropping and visualization.

This module provides an annotator for:

* Cropping point clouds using axis-aligned bounding boxes
* Supporting both sensor and world coordinate frames
* Generating and visualizing point cloud masks
* Combining masks with color images

.. note::
   Cropping can be done in either sensor coordinates (default) or world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass


import cv2
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.annotator_helper import (
    transform_cloud_from_camera_to_world,
    transform_cloud_from_world_to_camera,
)
from robokudo.utils.o3d_helper import get_mask_from_pointcloud

if TYPE_CHECKING:
    pass


@dataclass(kw_only=True)
class PointcloudCropAnnotator(BaseAnnotator):
    """
    Point cloud cropping using axis-aligned bounding boxes.

    Crop a subset of points from a pointcloud data based on min/max X,Y,Z values.
    The crop is either done in sensor coordinates (default) or relative to the world frame.

    .. warning::
       When using world coordinates, requires valid camera-to-world transform in CAS.
    """

    minimum_x: float = -2.0
    """
    Lower x bound of the crop box, in meters.
    """

    minimum_y: float = -2.0
    """
    Lower y bound of the crop box, in meters.
    """

    minimum_z: float = -9.0
    """
    Lower z bound of the crop box, in meters.
    """

    maximum_x: float = 2.0
    """
    Upper x bound of the crop box, in meters.
    """

    maximum_y: float = 2.0
    """
    Upper y bound of the crop box, in meters.
    """

    maximum_z: float = 3.0
    """
    Upper z bound of the crop box, in meters.
    """

    relative_to_world: bool = False
    """
    Whether the crop box is given in world coordinates instead of camera coordinates.
    """

    def compute(self) -> Status:
        """
        Process and crop point cloud data.

        The method:

        * Loads point cloud and camera data from CAS
        * Optionally transforms to world coordinates
        * Crops using axis-aligned bounding box
        * Generates visualization mask
        * Updates CAS with cropped cloud
        * Creates combined visualization

        :return: SUCCESS after processing
        :raises Exception: If world transform not found when needed
        """
        self.robokudo_logger.warning(f"{self.__class__.__name__} called for update()")

        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        pointcloud_camera_intrinsics = self.get_cas().get(
            CASViews.POINTCLOUD_CAMERA_INTRINSIC
        )
        color_to_depth_ratio = self.get_cas().get(CASViews.COLOR_TO_DEPTH_RATIO)

        #
        # Lookup camera information if the crop should be done relative to the world frame
        #

        if self.relative_to_world:
            try:
                cloud = transform_cloud_from_camera_to_world(self.get_cas(), cloud)
            except Exception as e:
                self.robokudo_logger.warning(
                    f"Couldn't find camera viewpoint in the CAS and relative_to_world is true. "
                    f"Fail. Error: {e}"
                )
                return Status.FAILURE

        #
        # Crop the point cloud
        #
        assert isinstance(cloud, o3d.geometry.PointCloud)

        abb = o3d.geometry.AxisAlignedBoundingBox(
            [
                self.minimum_x,
                self.minimum_y,
                self.minimum_z,
            ],
            [
                self.maximum_x,
                self.maximum_y,
                self.maximum_z,
            ],
        )

        cropped_cloud = cloud.crop(abb)

        # Transform cloud back to camera coordinates if it has been transformed to world before
        if self.relative_to_world:
            cropped_cloud_transformed = transform_cloud_from_world_to_camera(
                self.get_cas(), cropped_cloud
            )
            cropped_cloud = cropped_cloud_transformed

        assert isinstance(cropped_cloud, o3d.geometry.PointCloud)

        self.get_cas().set_ref(CASViews.CLOUD, cropped_cloud)
        self.get_annotator_output_struct().set_geometries(cropped_cloud)

        mask_scale = 1.0 / color_to_depth_ratio.horizontal
        mask = get_mask_from_pointcloud(
            cropped_cloud,
            color,
            pointcloud_camera_intrinsics,
            mask_scale_factor=mask_scale,
            crop_to_ref=True,
        )
        combined_mask_color = cv2.addWeighted(color, 0.5, mask, 0.5, 0)
        self.get_annotator_output_struct().set_image(combined_mask_color)

        return Status.SUCCESS
