"""
Robokudo Color Analysis Module.

This module provides functionality for semantic color analysis of object hypotheses.
It analyzes RGB regions of interest (ROIs) and masks to determine dominant colors
using HSV color space segmentation.

.. warning::
   The non-numba version of color counting may contain bugs in color counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import copy
from enum import Enum

import cv2
import numba
import numpy as np
import numpy.ma
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import SemanticColor
from robokudo.types.cv import Rect
from robokudo.types.scene import AnalyzableAnnotation
from robokudo.utils.cv_helper import crop_image

if TYPE_CHECKING:
    import numpy.typing as npt


class Color(Enum):
    """
    Enumeration of semantic colors used for classification.

    * Primary colors: RED, GREEN, BLUE
    * Secondary colors: YELLOW, CYAN, MAGENTA
    * Grayscale: WHITE, BLACK, GREY
    """

    RED = (0,)
    YELLOW = (1,)
    GREEN = (2,)
    CYAN = (3,)
    BLUE = (4,)
    MAGENTA = (5,)
    WHITE = (6,)
    BLACK = (7,)
    GREY = (8,)

    @property
    def bgr(self) -> tuple[int, int, int]:
        """
        The color the visualization draws this color with, in OpenCV's BGR order.

        Only the visualization uses it; the classification uses the hue ranges.
        """
        red, green, blue = {
            Color.RED: (255, 0, 0),
            Color.YELLOW: (255, 255, 0),
            Color.GREEN: (0, 255, 0),
            Color.CYAN: (0, 255, 255),
            Color.BLUE: (0, 0, 255),
            Color.MAGENTA: (255, 0, 255),
            Color.WHITE: (255, 255, 255),
            Color.BLACK: (0, 0, 0),
            Color.GREY: (127, 127, 127),
        }[self]
        return blue, green, red


@dataclass(kw_only=True)
class ClusterColorAnnotator(BaseAnnotator):
    """
    Calculate the semantic color for every Object Hypothesis (cluster) that has a RGB
    ROI and a Mask.

    This annotator analyzes object hypotheses by:

    * Converting RGB image regions to HSV color space
    * Using HSV thresholds to classify pixels into semantic color categories
    * Counting pixel distributions for each color
    * Annotating objects with their dominant colors

    The color classification uses configurable thresholds for:

    * Hue ranges for primary/secondary colors
    * Saturation threshold for chromatic colors
    * Value thresholds for black/white/grey
    """

    chromatic_color_count: int = 6
    """
    Number of equally wide hue ranges the chromatic colors divide the hue circle into.
    """

    compile_on_setup: bool = False
    """
    Whether to compile the numba color counting in :meth:`setup` instead of on the first
    tick.
    """

    minimum_chromatic_value: int = 60
    """
    HSV value a pixel must exceed to count as a chromatic color.
    """

    minimum_chromatic_saturation: int = 60
    """
    HSV saturation a pixel must exceed to count as a chromatic color.
    """

    maximum_black_value: int = 60
    """
    Highest HSV value of an achromatic pixel that counts as black.
    """

    minimum_white_value: int = 120
    """
    HSV value an achromatic pixel must exceed to count as white; the ones in between
    count as grey.
    """

    minimum_annotated_color_ratio: float = 0.2
    """
    Share of the analyzed pixels a color must exceed to be annotated.
    """

    analysis_scope: type[AnalyzableAnnotation] = AnalyzableAnnotation
    """
    Type of the annotations whose colors are analyzed.
    """

    cluster_color_info: list[list[tuple[Color, int, float]]] = field(
        default_factory=list, init=False
    )
    """
    For each analyzed annotation of the current tick, its colors with pixel count and
    ratio, the most frequent first.
    """

    cluster_rois: list[Rect] = field(default_factory=list, init=False)
    """
    The region of each analyzed annotation of the current tick.
    """

    @property
    def hue_range(self) -> float:
        """
        Width of the hue range of each chromatic color.
        """
        return 256 / self.chromatic_color_count

    @property
    def color_hue_positions(self) -> dict[Color, float]:
        """
        The hue each chromatic color's range ends at.
        """
        hue_interval_ends = [
            i * self.hue_range + self.hue_range / 2 + 0.5
            for i in range(0, self.chromatic_color_count)
        ]
        chromatic_colors = [
            Color.RED,
            Color.YELLOW,
            Color.GREEN,
            Color.CYAN,
            Color.BLUE,
            Color.MAGENTA,
        ]
        return dict(zip(chromatic_colors, hue_interval_ends))

    def setup(self, **kwargs: Any) -> bool:
        """
        Compile the numba color counting if :attr:`compile_on_setup` is set.
        """
        if self.compile_on_setup:
            self.count_colors_numba(np.array([[[0, 0, 0]]]), np.array([[0]]))
        return super().setup(**kwargs)

    def count_colors_numba(
        self, hsv_image: npt.NDArray, mask: npt.NDArray
    ) -> tuple[int, dict[Color, int]]:
        """
        Wrapper function for the numba version of the count colors method.

        The wrapper handles the pythonic/object style data and prepares everything for
        the requirements of the numba method.

        :param hsv_image: Image in HSV color space
        :param mask: Binary mask indicating pixels to analyze
        :return: A tuple with the number of analyzed pixels (given by the mask) as well
            as a dict with the counted colors. The key is a Color enum from this module.
        """
        counted_pixel_sum, color_count = ClusterColorAnnotator.count_colors_numba_impl(
            hsv_image,
            mask,
            len(Color),
            self.minimum_chromatic_saturation,
            self.minimum_chromatic_value,
            self.maximum_black_value,
            self.minimum_white_value,
            self.chromatic_color_count,
            self.hue_range,
        )

        color_count_map = {}
        for c in Color:
            color_count_map[c] = color_count[c.value[0]]

        return counted_pixel_sum, color_count_map

    @staticmethod
    @numba.njit
    def count_colors_numba_impl(
        hsv_image: npt.NDArray,
        mask: npt.NDArray,
        total_amount_of_colors: int,
        min_saturation_color: int,
        min_value_color: int,
        max_value_black: int,
        min_value_white: int,
        number_of_colors: int,
        color_range: float,
    ) -> tuple[int, npt.NDArray]:
        """
        Iterate over the masked pixels on a given hsv image to count how many pixels can
        be assigned to the defined color ranges. Requires the numba library and passes
        the required parameters to avoid class access and object-mode from numba.

        :param hsv_image: Image in HSV color space.
        :param mask: Binary mask indicating pixels to analyze.
        :param total_amount_of_colors: Total number of color categories.
        :param min_saturation_color: Minimum saturation for chromatic colors.
        :param min_value_color: Minimum value for chromatic colors.
        :param max_value_black: Maximum value for black classification.
        :param min_value_white: Minimum value for white classification.
        :param number_of_colors: Number of hue divisions.
        :param color_range: Size of each hue division.
        :return: A tuple with the number of analyzed pixels (given by the mask) as well
            as a numpy array with the counted colors. The id refers the indices of the
            Color enum from this module.
        """
        width, height, channels = hsv_image.shape
        counted_pixel_sum = 0

        hue_intervals = [
            i * color_range + color_range / 2 + 0.5 for i in range(0, number_of_colors)
        ]

        # Amount of defined Colors (including black/white/grey)
        color_count = np.zeros(total_amount_of_colors, dtype=numpy.int32)

        for i in range(width):
            for j in range(height):
                if mask[i][j] != 255:
                    continue
                counted_pixel_sum += 1
                hue, sat, val = hsv_image[i, j]
                # Assume check is done in the right order to not mix up indices
                if sat > min_saturation_color and val > min_value_color:
                    if hue < hue_intervals[Color.RED.value[0]]:
                        color_count[Color.RED.value[0]] += 1
                    elif hue < hue_intervals[Color.YELLOW.value[0]]:
                        color_count[Color.YELLOW.value[0]] += 1
                    elif hue < hue_intervals[Color.GREEN.value[0]]:
                        color_count[Color.GREEN.value[0]] += 1
                    elif hue < hue_intervals[Color.CYAN.value[0]]:
                        color_count[Color.CYAN.value[0]] += 1
                    elif hue < hue_intervals[Color.BLUE.value[0]]:
                        color_count[Color.BLUE.value[0]] += 1
                    elif hue < hue_intervals[Color.MAGENTA.value[0]]:
                        color_count[Color.MAGENTA.value[0]] += 1
                    else:
                        color_count[Color.RED.value[0]] += 1
                elif val <= max_value_black:
                    color_count[Color.BLACK.value[0]] += 1
                elif val > min_value_white:
                    color_count[Color.WHITE.value[0]] += 1
                else:
                    color_count[Color.GREY.value[0]] += 1
        return counted_pixel_sum, color_count

    def count_colors(
        self, hsv_image: npt.NDArray, mask: npt.NDArray
    ) -> tuple[int, dict]:
        """
        Non-numba version of the count_colors_numba_impl method.

        Use only if you can't use numba.

        .. warning::
           Seems to contain a bug in color counting.

        :param hsv_image: Image in HSV color space
        :param mask: Binary mask indicating pixels to analyze
        :return: A tuple with the number of analyzed pixels and a dict with color counts
        """
        all_image_pixel_values_in_mask = hsv_image[mask == 255]

        # indices
        # hue = 0
        # sat = 1
        # val = 2

        # https://stackoverflow.com/questions/7662458/how-to-split-an-array-according-to-a-condition-in-numpy
        def split(
            arr: npt.NDArray, cond: npt.NDArray
        ) -> tuple[npt.NDArray, npt.NDArray]:
            return arr[cond], arr[~cond]

        # Input matrix
        im = all_image_pixel_values_in_mask

        # We'll now create multiple pixel lists (pl) in a divide and conquer fashion.
        # Effectively reducing the number of pixels we have to analyze
        # We try to visit pixels as few times as possible.
        pl_color, pl_bw_range = split(
            im,
            np.logical_and(
                im[:, 1] > self.minimum_chromatic_saturation,
                im[:, 2] > self.minimum_chromatic_value,
            ),
        )

        pl_black, pl_grey_white = split(
            pl_bw_range, pl_bw_range[:, 2] <= self.maximum_black_value
        )
        pl_white, pl_grey = split(
            pl_grey_white,
            pl_grey_white[:, 2] > self.minimum_white_value,
        )

        # Check on the HUE channel the color and reduce the list of pixels piece-by-piece
        pl_red, pl_rest = split(
            pl_color, pl_color[:, 0] < self.color_hue_positions[Color.RED]
        )
        pl_yellow, pl_rest = split(
            pl_rest, pl_rest[:, 0] < self.color_hue_positions[Color.YELLOW]
        )
        pl_green, pl_rest = split(
            pl_rest, pl_rest[:, 0] < self.color_hue_positions[Color.GREEN]
        )
        pl_cyan, pl_rest = split(
            pl_rest, pl_rest[:, 0] < self.color_hue_positions[Color.CYAN]
        )
        pl_blue, pl_rest = split(
            pl_rest, pl_rest[:, 0] < self.color_hue_positions[Color.BLUE]
        )
        pl_magenta, pl_rest = split(
            pl_rest, pl_rest[:, 0] < self.color_hue_positions[Color.MAGENTA]
        )
        pl_red = np.append(
            pl_red, pl_rest
        )  # the rest can be considered as red in the HUE scale

        color_count = {
            Color.RED: len(pl_red),
            Color.YELLOW: len(pl_yellow),
            Color.GREEN: len(pl_green),
            Color.CYAN: len(pl_cyan),
            Color.BLUE: len(pl_blue),
            Color.MAGENTA: len(pl_magenta),
            Color.WHITE: len(pl_white),
            Color.BLACK: len(pl_black),
            Color.GREY: len(pl_grey),
        }

        return len(all_image_pixel_values_in_mask), color_count

    def compute(self) -> Status:
        """
        Process current scene to analyze colors of object hypotheses.

        Steps:

        * Gets current point cloud and color image
        * Creates color annotations for each object hypothesis
        * Updates visualization with color information
        * Prepares 3D visualization geometries

        :return: SUCCESS after processing is complete
        """
        cloud = self.get_cas().get(CASViews.CLOUD)
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        # List for visualization purposes
        self.cluster_color_info = []
        self.cluster_rois = []

        self.create_color_annotations(color)

        visualization_img = copy.deepcopy(color)
        visualization_img = self.draw_visualization(visualization_img)

        # 3D Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_struct().set_geometries(vis_geometries)
        self.get_annotator_output_struct().set_image(visualization_img)

        return Status.SUCCESS

    def create_color_annotations(self, color: npt.NDArray) -> None:
        """
        Create color annotations for all object hypotheses.

        For each object hypothesis with a valid ROI mask:

        * Extract color image region and mask
        * Convert to HSV color space
        * Count pixel colors using numba-optimized implementation
        * Sort colors by frequency
        * Add color annotations for colors above ratio threshold
        * Store color information for visualization

        :param color: Input color image
        """
        # Iterate over everything that is an Object hypothesis and calculate the colors
        object_hypotheses = self.get_cas().filter_annotations_by_type(
            self.analysis_scope
        )

        for object_hypothesis in object_hypotheses:
            if object_hypothesis.roi.mask is None:
                continue

            roi = object_hypothesis.roi.roi

            cluster_color_img = color[
                roi.pos.y : roi.pos.y + roi.height, roi.pos.x : roi.pos.x + roi.width
            ]

            cluster_mask = object_hypothesis.roi.mask

            # If you need to manually check the ROI color calcuation, you can overwrite ROI images:
            # cluster_color_img = np.zeros((300, 300, 3), np.uint8)
            # cluster_color_img[:] = self.color_bgr_values[ClusterColorAnnotator.Color.GREY]
            # cluster_mask = np.zeros((300, 300), np.uint8)
            # cluster_mask[:] = 255  # take everything

            color_height, color_width, color_dim = cluster_color_img.shape
            mask_height, mask_width = cluster_mask.shape

            assert color_height == mask_height
            assert color_width == mask_width

            # Convert to HSV_FULL. FULL is important, # because otherwise you just get H=0...180
            cluster_hsv_color_img = cv2.cvtColor(
                cluster_color_img, cv2.COLOR_BGR2HSV_FULL
            )

            # Uncomment the following line if you want to use the non-numba version (roughly 100% slower)
            # color_count_result = self.count_colors(cluster_hsv_color_img, cluster_mask)
            total_pixel_count, color_count_dict = self.count_colors_numba(
                cluster_hsv_color_img, cluster_mask
            )

            # Sort list by color count. Highest value comes first.
            # Also add the ratio of the color
            color_count_result_sorted_by_count = sorted(
                [
                    (
                        color_name,
                        color_pixel_count,
                        color_pixel_count / total_pixel_count,
                    )
                    for color_name, color_pixel_count in color_count_dict.items()
                ],
                key=lambda tup: tup[1],
                reverse=True,
            )

            # Add Color Annotation if ratio of color is large
            for (
                color_name,
                color_pixel_count,
                color_ratio,
            ) in color_count_result_sorted_by_count:
                if color_ratio > self.minimum_annotated_color_ratio:
                    color_annotation = SemanticColor()
                    color_annotation.color = str(color_name.name.lower())
                    color_annotation.source = self.name
                    color_annotation.ratio = color_ratio
                    object_hypothesis.annotations.append(color_annotation)
                else:
                    break

            # Prepare visualization
            self.cluster_color_info.append(color_count_result_sorted_by_count)
            self.cluster_rois.append(roi)

    def draw_visualization(self, visualization_img: npt.NDArray) -> npt.NDArray:
        """
        Draw visualization of detected colors on the image.

        For each ROI:

        * Draw bounding box in dominant color
        * Add text label with ROI index and dominant color name
        * Create color histogram showing distribution of detected colors

        :param visualization_img: Image to draw visualization on
        :return: Image with visualization overlays
        """
        for i, roi in enumerate(self.cluster_rois):
            cci = self.cluster_color_info[i]
            dominant_color = cci[0][0]
            assert isinstance(dominant_color, Color)
            dominant_color_value = dominant_color.bgr

            upper_left = (roi.pos.x, roi.pos.y)
            upper_left_text = (roi.pos.x, roi.pos.y - 5)

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_img = cv2.putText(
                visualization_img,
                f"{i}: {dominant_color.name}",
                upper_left_text,
                font,
                0.5,
                dominant_color_value,
                1,
                2,
            )
            visualization_img = cv2.rectangle(
                visualization_img,
                upper_left,
                (roi.pos.x + roi.width, roi.pos.y + roi.height),
                dominant_color_value,
                2,
            )

            # Prepare Histogram
            histogram_size = roi.width, 10
            histogram_upper_left = (roi.pos.x, roi.pos.y + roi.height + 1)
            histogram_lower_right = (
                roi.pos.x + histogram_size[0],
                roi.pos.y + roi.height + 1 + histogram_size[1],
            )
            histogram_region = crop_image(
                visualization_img, histogram_upper_left, histogram_size
            )

            histogram_region[:] = (255, 255, 255)

            # Draw the border of the Histogram
            visualization_img = cv2.rectangle(
                visualization_img,
                histogram_upper_left,
                histogram_lower_right,
                (0, 0, 0),
                1,
            )

            # Draw the actual histogram
            # Get the ratios of the colors, calculate their width and draw them into the region of the image
            # where the histogram is located.
            start = 0
            for elem in cci:
                # Triple: Name, Count, Ratio
                hist_color = elem[0]
                assert isinstance(hist_color, Color)
                ratio = elem[2]
                hist_color_width = histogram_size[0] * ratio
                hist_color_rect = (
                    int(start + 0.5),
                    0,
                    int(hist_color_width + 0.5),
                    histogram_size[1],
                )
                start += hist_color_width
                cv2.rectangle(
                    histogram_region,
                    hist_color_rect,
                    hist_color.bgr,
                    thickness=-1,
                )
        return visualization_img
