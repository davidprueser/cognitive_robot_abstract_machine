"""
Image-based object cluster extraction.

This module provides functionality for extracting object clusters from color images using HSV color segmentation.
The main class :class:`ImageClusterExtractor` implements color-based segmentation and contour detection to identify
object clusters in RGB images.

Key features:

* HSV color space thresholding
* Contour detection and filtering
* 3D point cloud generation from depth data
* ROI and mask generation
* Query-based color parameter adjustment
* Visualization of detected clusters
"""

from __future__ import annotations

from dataclasses import dataclass, field

import copy
from enum import Enum, IntEnum

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import ColorToDepthRatioMissing, ImageContourMissing
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import scale_camera_intrinsics
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image

if TYPE_CHECKING:
    import numpy.typing as npt


# %% color ranges


@dataclass(frozen=True)
class HSVRange:
    """
    A box in the HSV color space of OpenCV's full-range hue (0-255).

    A minimum hue above the maximum hue wraps around the 0/255 seam of the hue wheel.
    """

    minimum: tuple[int, int, int]
    """
    Lowest hue, saturation and value inside the range.
    """

    maximum: tuple[int, int, int]
    """
    Highest hue, saturation and value inside the range.
    """


class NamedHSVRange(Enum):
    """
    The HSV range of each color a query can ask for.
    """

    BLUE = HSVRange(minimum=(150, 130, 85), maximum=(200, 255, 255))
    RED = HSVRange(minimum=(245, 150, 95), maximum=(15, 255, 255))
    """
    Red sits at the 0/255 seam of the hue wheel, so its hue range wraps around.
    """


class ImageDisplayMode(IntEnum):
    """
    What the visualization of the annotator shows; the key with the member's value
    selects it.
    """

    MASKED_OBJECT = 1
    """
    The color image masked to the detected objects.
    """

    DEPTH_MASK = 2
    """
    The depth mask of the last detected object.
    """


# %% annotator


@dataclass(kw_only=True)
class ImageClusterExtractor(BaseAnnotator):
    """
    Extract object clusters from images using color segmentation.

    This annotator performs the following steps:

    * Converts RGB image to HSV color space
    * Applies HSV thresholding based on configured parameters
    * Detects and filters contours based on size
    * Generates point clouds from depth data for each contour
    * Creates ObjectHypothesis annotations with ROIs and masks
    * Provides visualization of detected clusters

    The HSV thresholds can be adjusted dynamically based on color queries.
    """

    hsv_range: HSVRange = NamedHSVRange.BLUE.value
    """
    The color range of the objects to extract, unless a query asks for a color.
    """

    erosion_iterations: int = 2
    """
    How often each object mask is eroded, to drop background pixels the imperfect
    calibration between the color and depth camera adds.
    """

    minimum_contour_area: int = 1000
    """
    Area in pixels a contour must exceed to become an object.
    """

    outlier_removal: bool = True
    """
    Whether to remove statistical outliers from each object cloud.
    """

    outlier_removal_neighbor_count: int = 20
    """
    Number of neighbors the statistical outlier removal considers per point.
    """

    outlier_removal_standard_deviation_ratio: float = 2.0
    """
    Standard deviations from the mean neighbor distance beyond which a point is an
    outlier.
    """

    maximum_object_count: int = 2
    """
    Number of the largest contours that become objects.
    """

    minimum_point_count: int = 62
    """
    Number of points an object cloud must have to become an object hypothesis.
    """

    query_hsv_range: HSVRange | None = field(default=None, init=False)
    """
    The color range the query of the current CAS asks for, if any.
    """

    hsv: npt.NDArray | None = field(default=None, init=False, repr=False)
    """
    The current color image in HSV, scaled to the depth image resolution.
    """

    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic | None = field(
        default=None, init=False, repr=False
    )
    """
    Intrinsics of the camera at the resolution of the depth image.
    """

    display_mode: ImageDisplayMode = field(
        default=ImageDisplayMode.MASKED_OBJECT, init=False
    )
    """
    What the visualization currently shows.
    """

    def adjust_hsv_threshold_to_query(self) -> None:
        """
        Use the color range of the first color the query of the CAS asks for, if it
        names one of :class:`NamedHSVRange`.
        """
        self.query_hsv_range = None
        if not self.get_cas().contains(CASViews.QUERY):
            return
        query = self.get_cas().get(CASViews.QUERY)
        if not query or len(query.obj.color) == 0:
            return
        color_name = query.obj.color[0].upper()
        if color_name in NamedHSVRange.__members__:
            self.query_hsv_range = NamedHSVRange[color_name].value

    @property
    def active_hsv_range(self) -> HSVRange:
        """
        The color range of the objects extracted from the current CAS.
        """
        if self.query_hsv_range is not None:
            return self.query_hsv_range
        return self.hsv_range

    def _threshold_hsv(
        self,
        hsv_min: tuple[int, int, int],
        hsv_max: tuple[int, int, int],
    ) -> npt.NDArray:
        """
        Threshold ``self.hsv`` to the given bounds, wrapping the hue channel around the
        0/255 seam when ``hsv_min``'s hue exceeds ``hsv_max``'s.

        True red sits exactly at that seam, so a single contiguous hue range only ever
        catches one side of it (confirmed live: a plain ``[215,255]`` range matched 124
        of 101760 pixels on a red object; wrapping recovered 1748 by also matching the
        pixels that wrapped to just above 0).

        :param hsv_min: Lower HSV bound.
        :param hsv_max: Upper HSV bound. A hue lower than ``hsv_min``'s signals
            wraparound.
        :return: Binary mask of pixels within the (possibly wrapped) bounds.
        """
        if hsv_min[0] <= hsv_max[0]:
            return cv2.inRange(self.hsv, hsv_min, hsv_max)

        low = cv2.inRange(
            self.hsv,
            (0, hsv_min[1], hsv_min[2]),
            (hsv_max[0], hsv_max[1], hsv_max[2]),
        )
        high = cv2.inRange(
            self.hsv,
            (hsv_min[0], hsv_min[1], hsv_min[2]),
            (255, hsv_max[1], hsv_max[2]),
        )
        return cv2.bitwise_or(low, high)

    def compute(self) -> Status:
        """
        Process input images to detect and annotate object clusters.

        The method:

        * Scales color image to match depth image
        * Converts to HSV and applies thresholding
        * Detects and filters contours
        * Generates point clouds for each contour
        * Creates ObjectHypothesis annotations
        * Generates visualization output

        :return: SUCCESS if clusters found, FAILURE if no clusters
        :raises ImageContourMissing: If no contours are found
        """

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        self.camera_intrinsics = copy.deepcopy(
            self.get_cas().get(CASViews.CAMERA_INTRINSIC)
        )

        # Scale the image down so that it matches the depth image size
        resized_color = None
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                self.get_cas(), color
            )
            scale_camera_intrinsics(self)
        except ColorToDepthRatioMissing:
            self.robokudo_logger.error(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation."
            )
            raise

        self.hsv = cv2.cvtColor(resized_color, cv2.COLOR_BGR2HSV_FULL)

        self.adjust_hsv_threshold_to_query()

        # Apply the HSV threshold on the image and find contours on the resultant binary image
        hsv_mask = self._threshold_hsv(
            self.active_hsv_range.minimum, self.active_hsv_range.maximum
        )
        contours, hierarchy = cv2.findContours(
            image=hsv_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # Fail if no contours have been found
            raise ImageContourMissing(context="image cluster extraction")

        # Visualization purposes
        result = copy.deepcopy(resized_color)
        result = cv2.bitwise_and(result, result, mask=hsv_mask)

        # Calculate the areas spanned by the Contours in order to find the largest ones
        # Filter the contours based on hierarchy
        contours = [
            contour for i, contour in enumerate(contours) if hierarchy[0][i][3] == -1
        ]
        contour_areas = np.asarray([cv2.contourArea(c) for c in contours])
        filtered_contour_areas = [
            area for area in contour_areas if area > self.minimum_contour_area
        ]
        sorted_areas = sorted(filtered_contour_areas, reverse=True)
        largest_elements = sorted_areas[: self.maximum_object_count]
        filtered_contours = [
            contours[i]
            for i, area in enumerate(contour_areas)
            if area in largest_elements
        ]

        # Draw a 'mask' based on the contours
        # This will completely mask out the area inside a contour instead of just drawing contour points
        # List to store ObjectHypothesis instances
        object_hypotheses = []
        visualized_geometries = []
        for i, contour in enumerate(filtered_contours):
            biggest_contour_mask = np.zeros_like(depth, dtype=np.uint8)
            cv2.drawContours(
                image=biggest_contour_mask,
                contours=[contour],
                contourIdx=-1,
                color=255,
                thickness=cv2.FILLED,
            )

            # Apply Erosion to remove background points which might be included due to imperfect calibration between
            # RGB and Depth images
            kernel = np.ones((5, 5), np.uint8)
            biggest_contour_mask = cv2.erode(
                biggest_contour_mask,
                kernel,
                iterations=self.erosion_iterations,
            )
            biggest_contour_mask_rgb = cv2.cvtColor(
                biggest_contour_mask, cv2.COLOR_GRAY2BGR
            )

            # Cluster creation
            # Here we'll exploit, that the depth creation in open3d will ignore depth values of 0
            # https://github.com/isl-org/Open3D/issues/1662
            color_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
            depth_masked = copy.deepcopy(depth)

            depth_masked = np.where(
                biggest_contour_mask == 255, depth_masked, 0
            )  # mask all depth values

            o3d_color = o3d.geometry.Image(color_rgb)
            o3d_depth = o3d.geometry.Image(
                depth_masked
            )  # Please note that depth values should be in mm
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, convert_rgb_to_intensity=False, depth_trunc=9.0
            )

            cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.camera_intrinsics
            )

            if self.outlier_removal:
                cloud, indices_after_outlier = cloud.remove_statistical_outlier(
                    nb_neighbors=self.outlier_removal_neighbor_count,
                    std_ratio=self.outlier_removal_standard_deviation_ratio,
                )
            visualized_geometries.append(cloud)

            if len(cloud.points) >= self.minimum_point_count:
                # Create an ObjectHypothesis instance

                object_hypothesis = ObjectHypothesis()
                # Set ObjectHypothesis attributes
                object_hypothesis.id = str(i)
                object_hypothesis.source = self.name
                object_hypothesis.points = cloud

                # Calculate bounding rectangle for the current contour
                x, y, w, h = cv2.boundingRect(contour)
                # Set ROI attributes for the ObjectHypothesis
                object_hypothesis.roi.roi.pos.x = x
                object_hypothesis.roi.roi.pos.y = y
                object_hypothesis.roi.roi.width = w
                object_hypothesis.roi.roi.height = h

                # TODO Generate proper Masks

                # TODO Allow multiple objects to be detected

                object_hypotheses.append(object_hypothesis)
                self.get_cas().annotations.append(object_hypothesis)

        #
        # Create visualization Output
        #
        if self.display_mode == ImageDisplayMode.DEPTH_MASK:
            visualization_img = copy.deepcopy(biggest_contour_mask_rgb)
        else:
            visualization_img = copy.deepcopy(result)

        for oh in object_hypotheses:
            assert isinstance(oh, ObjectHypothesis)
            oh_roi = oh.roi.roi
            upper_left = (oh_roi.pos.x, oh_roi.pos.y)
            upper_left_text = (oh_roi.pos.x, oh_roi.pos.y - 5)

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(
                visualization_img,
                f"ROI-{oh.id}({len(oh.points.points)})",
                upper_left_text,
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )
            visualization_img = cv2.rectangle(
                visualization_img,
                upper_left,
                (oh_roi.pos.x + oh_roi.width, oh_roi.pos.y + oh_roi.height),
                (0, 0, 255),
                2,
            )

        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(visualized_geometries)
        return Status.SUCCESS

    def key_callback(self, key: int) -> None:
        """
        Handle keyboard input to change visualization mode.

        :param key: ASCII value of pressed key
        """
        for display_mode in ImageDisplayMode:
            if key == ord(str(display_mode.value)):
                self.display_mode = display_mode
