"""
Image preprocessing and point cloud generation.

This module provides an annotator for:

* Converting RGB-D camera data to point clouds
* Handling color and depth image synchronization
* Managing camera intrinsic parameters
* Providing visualization modes for debugging

.. note::
   Depth values are expected to be in millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import copy
from enum import IntEnum

import cv2
import numpy
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import ColorToDepthRatioMissing
from robokudo.utils.annotator_helper import scale_camera_intrinsics
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

if TYPE_CHECKING:
    pass


# %% display modes


class PreprocessorDisplayMode(IntEnum):
    """
    Which image the visualization of the preprocessor shows; the key with the member's
    value selects it.
    """

    COLOR = 1
    DEPTH = 2


# %% annotator


@dataclass(kw_only=True)
class ImagePreprocessorAnnotator(BaseAnnotator):
    """
    RGB-D image preprocessor and point cloud generator.

    This annotator:

    * Converts RGB-D camera data to point clouds
    * Handles color/depth image synchronization
    * Manages camera intrinsic parameters
    * Provides visualization modes for debugging

    .. warning::
       Requires properly configured camera intrinsics and color-to-depth ratio.
    """

    maximum_depth: float = 9.0
    """
    Distance in meters beyond which depth measurements are left out of the cloud.
    """

    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic | None = field(
        default=None, init=False, repr=False
    )
    """
    Intrinsics of the camera at the resolution of the depth image.
    """

    display_mode: PreprocessorDisplayMode = field(
        default=PreprocessorDisplayMode.COLOR, init=False
    )
    """
    Which image the visualization currently shows.
    """

    def compute(self) -> Status:
        """
        Process RGB-D images and generate point cloud.

        The method:

        * Loads color and depth images from CAS
        * Synchronizes and scales images
        * Converts to Open3D format
        * Generates point cloud
        * Updates visualization based on view mode

        :return: SUCCESS after processing
        :raises ColorToDepthRatioMissing: If color-to-depth ratio is not set
        """
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.camera_intrinsics = copy.deepcopy(
            self.get_cas().get(CASViews.CAMERA_INTRINSIC)
        )

        if self.display_mode == PreprocessorDisplayMode.DEPTH:
            self.get_annotator_output_struct().set_image(depth)
        else:
            self.get_annotator_output_struct().set_image(color)

        scale_camera_intrinsics(self)

        resized_color = None
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                self.get_cas(), color
            )
        except ColorToDepthRatioMissing:
            self.robokudo_logger.error(
                "No color to depth ratio set by your camera driver! Can't preprocess."
            )
            raise

        # o3d expects color images in RGB order
        color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
        o3d_color = o3d.geometry.Image(color_rgb)
        o3d_depth = None
        try:
            o3d_depth = o3d.geometry.Image(
                depth
            )  # Please note that depth values should be in mm
        except RuntimeError:
            # Even though you might have a uint16 already, it might be required to explicitly cast to uint16
            # (Note: the byteorder field might be different after the cast to make o3d happy)
            o3d_depth = o3d.geometry.Image(depth.astype(numpy.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            convert_rgb_to_intensity=False,
            depth_trunc=self.maximum_depth,
        )

        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.camera_intrinsics
        )

        self.get_cas().set(CASViews.POINTCLOUD_CAMERA_INTRINSIC, self.camera_intrinsics)

        self.get_cas().set_ref(CASViews.CLOUD, cloud)
        self.get_annotator_output_struct().set_geometries(
            cloud
        )  # the visualizer can only show non-NaN pointclouds

        return Status.SUCCESS

    def key_callback(self, key: int) -> None:
        """
        Handle keyboard input for view mode switching.

        :param key: ASCII value of pressed key

        Available keys:

        * '1': Switch to color view mode
        * '2': Switch to depth view mode
        """
        for display_mode in PreprocessorDisplayMode:
            if key == ord(str(display_mode.value)):
                self.display_mode = display_mode
