"""
Robokudo Color Analysis Module.

This module provides functionality for semantic color analysis of object hypotheses.
It analyzes RGB regions of interest (regions of interest) and masks to determine dominant colors
using HSV color space segmentation.

.. warning::
   The non-numba version of color counting may contain bugs in color counting.
"""

from __future__ import annotations

import copy
from enum import Enum
from timeit import default_timer

import cv2
import numba
import numpy as np
import numpy.ma
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Dict, List, Tuple, Type

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import SemanticColor
from robokudo.types.cv import Rectangle
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
    # COUNT # Use len(Color) instead


class ClusterColorAnnotator(BaseAnnotator):
    """
    Calculate the semantic color for every Object Hypothesis (cluster) that has a RGB
    region of interest and a Mask.

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

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for the ClusterColorAnnotator.

        Defines parameters that control the color analysis behavior:

        * Number of color divisions in hue space
        * Thresholds for color classification
        * Minimum ratio for color annotation
        """

        class Parameters:
            """
            Parameters for color analysis configuration.
            """

            def __init__(self) -> None:
                self.number_of_colors: int = 6
                self.color_range: float = 256 / self.number_of_colors
                self.just_in_time_compile_on_initialization: bool = False
                self.min_value_color: int = 60
                self.min_saturation_color: int = 60
                self.max_value_black: int = 60
                self.min_value_white: int = 120
                self.ratio_annotation_threshold: float = 0.2
                self.analysis_scope: Type = AnalyzableAnnotation

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    # To easily define this, we use RGB here and then flip it. OpenCV in python uses BGR per default.
    # Please note, that this color will not be used for the segmentation!
    # The segmentation is mainly done with self.color_hue_positions
    color_name_to_blue_green_red_values: Dict[Color, Tuple[int, ...]] = {
        Color.RED: tuple(reversed((255, 0, 0))),
        Color.YELLOW: tuple(reversed((255, 255, 0))),
        Color.GREEN: tuple(reversed((0, 255, 0))),
        Color.CYAN: tuple(reversed((0, 255, 255))),
        Color.BLUE: tuple(reversed((0, 0, 255))),
        Color.MAGENTA: tuple(reversed((255, 0, 255))),
        Color.WHITE: tuple(reversed((255, 255, 255))),
        Color.BLACK: tuple(reversed((0, 0, 0))),
        Color.GREY: tuple(reversed((127, 127, 127))),
    }

    def __init__(
        self,
        name: str = "ClusterColorAnnotator",
        descriptor: ClusterColorAnnotator.Descriptor | None = None,
    ) -> None:
        """
        Default construction.

        Minimal one-time init!
        :param name: Name of the annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)

        # This list defines the range of a certain color.
        hue_intervals = [
            i * self.descriptor.parameters.color_range
            + self.descriptor.parameters.color_range / 2
            + 0.5
            for i in range(0, self.descriptor.parameters.number_of_colors)
        ]

        self.color_hue_positions = {
            Color.RED: hue_intervals[0],
            Color.YELLOW: hue_intervals[1],
            Color.GREEN: hue_intervals[2],
            Color.CYAN: hue_intervals[3],
            Color.BLUE: hue_intervals[4],
            Color.MAGENTA: hue_intervals[5],
        }

        self.cluster_color_info: List[List[Tuple[Color, int, float]]] = []
        self.cluster_regions_of_interest: List[Rectangle] = []

        # Trigger JIT by calling method first time with empty data
        if self.descriptor.parameters.just_in_time_compile_on_initialization:
            compile_start_time = default_timer()
            hsv_image = np.array([[[0, 0, 0]]])
            mask = np.array([[0]])
            self.count_colors_numba(hsv_image, mask)
            compile_end_time = default_timer()
            self.robokudo_logger.info(
                f"Pre-compilation took {(compile_end_time - compile_start_time):.4f}s"
            )

    def count_colors_numba(
        self, hsv_image: npt.NDArray, mask: npt.NDArray
    ) -> Tuple[int, Dict[Color, int]]:
        """
        Wrapper function for the numba version of the count colors method.

        The wrapper handles the pythonic/object style data and prepares everything for
        the requirements of the numba method.

        :param hsv_image: Image in HSV color space
        :param mask: Binary mask indicating pixels to analyze
        :return: A tuple with the number of analyzed pixels (given by the mask) as well
            as a dict with the counted colors. The key is a Color enum from this module.
        """
        counted_pixel_sum, color_count = (
            ClusterColorAnnotator.count_colors_numba_implementation(
                hsv_image,
                mask,
                len(Color),
                self.descriptor.parameters.min_saturation_color,
                self.descriptor.parameters.min_value_color,
                self.descriptor.parameters.max_value_black,
                self.descriptor.parameters.min_value_white,
                self.descriptor.parameters.number_of_colors,
                self.descriptor.parameters.color_range,
            )
        )

        color_count_map = {}
        for color in Color:
            color_count_map[color] = color_count[color.value[0]]

        return counted_pixel_sum, color_count_map

    @staticmethod
    @numba.njit
    def count_colors_numba_implementation(
        hsv_image: npt.NDArray,
        mask: npt.NDArray,
        total_amount_of_colors: int,
        min_saturation_color: int,
        min_value_color: int,
        max_value_black: int,
        min_value_white: int,
        number_of_colors: int,
        color_range: float,
    ) -> Tuple[int, npt.NDArray]:
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
                hue, saturation, value = hsv_image[i, j]
                # Assume check is done in the right order to not mix up indices
                if saturation > min_saturation_color and value > min_value_color:
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
                elif value <= max_value_black:
                    color_count[Color.BLACK.value[0]] += 1
                elif value > min_value_white:
                    color_count[Color.WHITE.value[0]] += 1
                else:
                    color_count[Color.GREY.value[0]] += 1
        return counted_pixel_sum, color_count

    def count_colors(
        self, hsv_image: npt.NDArray, mask: npt.NDArray
    ) -> Tuple[int, dict]:
        """
        Non-numba version of the count_colors_numba_implementation method.

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
        # saturation = 1
        # value = 2

        # https://stackoverflow.com/questions/7662458/how-to-split-an-array-according-to-a-condition-in-numpy
        def split(
            array: npt.NDArray, condition: npt.NDArray
        ) -> Tuple[npt.NDArray, npt.NDArray]:
            return array[condition], array[~condition]

        # Input matrix
        pixel_values = all_image_pixel_values_in_mask

        # We'll now create multiple pixel lists (pl) in a divide and conquer fashion.
        # Effectively reducing the number of pixels we have to analyze
        # We try to visit pixels as few times as possible.
        pixels_color, pixels_black_and_white_range = split(
            pixel_values,
            np.logical_and(
                pixel_values[:, 1] > self.descriptor.parameters.min_saturation_color,
                pixel_values[:, 2] > self.descriptor.parameters.min_value_color,
            ),
        )

        pixels_black, pixels_grey_white = split(
            pixels_black_and_white_range,
            pixels_black_and_white_range[:, 2]
            <= self.descriptor.parameters.max_value_black,
        )
        pixels_white, pixels_grey = split(
            pixels_grey_white,
            pixels_grey_white[:, 2] > self.descriptor.parameters.min_value_white,
        )

        # Check on the HUE channel the color and reduce the list of pixels piece-by-piece
        pixels_red, pixels_rest = split(
            pixels_color, pixels_color[:, 0] < self.color_hue_positions[Color.RED]
        )
        pixels_yellow, pixels_rest = split(
            pixels_rest, pixels_rest[:, 0] < self.color_hue_positions[Color.YELLOW]
        )
        pixels_green, pixels_rest = split(
            pixels_rest, pixels_rest[:, 0] < self.color_hue_positions[Color.GREEN]
        )
        pixels_cyan, pixels_rest = split(
            pixels_rest, pixels_rest[:, 0] < self.color_hue_positions[Color.CYAN]
        )
        pixels_blue, pixels_rest = split(
            pixels_rest, pixels_rest[:, 0] < self.color_hue_positions[Color.BLUE]
        )
        pixels_magenta, pixels_rest = split(
            pixels_rest, pixels_rest[:, 0] < self.color_hue_positions[Color.MAGENTA]
        )
        pixels_red = np.append(
            pixels_red, pixels_rest
        )  # the rest can be considered as red in the HUE scale

        color_count = {
            Color.RED: len(pixels_red),
            Color.YELLOW: len(pixels_yellow),
            Color.GREEN: len(pixels_green),
            Color.CYAN: len(pixels_cyan),
            Color.BLUE: len(pixels_blue),
            Color.MAGENTA: len(pixels_magenta),
            Color.WHITE: len(pixels_white),
            Color.BLACK: len(pixels_black),
            Color.GREY: len(pixels_grey),
        }

        return len(all_image_pixel_values_in_mask), color_count

    def update(self) -> Status:
        """
        Process current scene to analyze colors of object hypotheses.

        Steps:

        * Gets current point cloud and color image
        * Creates color annotations for each object hypothesis
        * Updates visualization with color information
        * Prepares 3D visualization geometries

        :return: SUCCESS after processing is complete
        """
        start_timer = default_timer()

        cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )
        color = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )

        # List for visualization purposes
        self.cluster_color_info = []
        self.cluster_regions_of_interest = []

        self.create_color_annotations(color)

        visualization_image = copy.deepcopy(color)
        visualization_image = self.draw_visualization(visualization_image)

        # 3D Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_structure().set_geometries(vis_geometries)
        self.get_annotator_output_structure().set_image(visualization_image)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def create_color_annotations(self, color: npt.NDArray) -> None:
        """
        Create color annotations for all object hypotheses.

        For each object hypothesis with a valid region of interest mask:

        * Extract color image region and mask
        * Convert to HSV color space
        * Count pixel colors using numba-optimized implementation
        * Sort colors by frequency
        * Add color annotations for colors above ratio threshold
        * Store color information for visualization

        :param color: Input color image
        """
        # Iterate over everything that is an Object hypothesis and calculate the colors
        object_hypotheses = (
            self.get_common_analysis_structure().filter_annotations_by_type(
                self.descriptor.parameters.analysis_scope
            )
        )

        for object_hypothesis in object_hypotheses:
            if object_hypothesis.region_of_interest.mask is None:
                continue

            rectangle = object_hypothesis.region_of_interest.rectangle

            cluster_color_image = color[
                rectangle.position.y : rectangle.position.y + rectangle.height,
                rectangle.position.x : rectangle.position.x + rectangle.width,
            ]

            cluster_mask = object_hypothesis.region_of_interest.mask

            # If you need to manually check the region of interest color calcuation, you can overwrite region of interest images:
            # cluster_color_image = np.zeros((300, 300, 3), np.uint8)
            # cluster_color_image[:] = self.color_bgr_values[ClusterColorAnnotator.Color.GREY]
            # cluster_mask = np.zeros((300, 300), np.uint8)
            # cluster_mask[:] = 255  # take everything

            color_height, color_width, color_channel_count = cluster_color_image.shape
            mask_height, mask_width = cluster_mask.shape

            assert color_height == mask_height
            assert color_width == mask_width

            # Convert to HSV_FULL. FULL is important, # because otherwise you just get H=0...180
            cluster_hsv_color_image = cv2.cvtColor(
                cluster_color_image, cv2.COLOR_BGR2HSV_FULL
            )

            # Uncomment the following line if you want to use the non-numba version (roughly 100% slower)
            # color_count_result = self.count_colors(cluster_hsv_color_image, cluster_mask)
            total_pixel_count, color_count_dict = self.count_colors_numba(
                cluster_hsv_color_image, cluster_mask
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
                key=lambda pair: pair[1],
                reverse=True,
            )

            # Add Color Annotation if ratio of color is large
            for (
                color_name,
                color_pixel_count,
                color_ratio,
            ) in color_count_result_sorted_by_count:
                if color_ratio > self.descriptor.parameters.ratio_annotation_threshold:
                    color_annotation = SemanticColor()
                    color_annotation.color = str(color_name.name.lower())
                    color_annotation.source = self.name
                    color_annotation.ratio = color_ratio
                    object_hypothesis.annotations.append(color_annotation)
                else:
                    break

            # Prepare visualization
            self.cluster_color_info.append(color_count_result_sorted_by_count)
            self.cluster_regions_of_interest.append(rectangle)

    def draw_visualization(self, visualization_image: npt.NDArray) -> npt.NDArray:
        """
        Draw visualization of detected colors on the image.

        For each region of interest:

        * Draw bounding box in dominant color
        * Add text label with region of interest index and dominant color name
        * Create color histogram showing distribution of detected colors

        :param visualization_image: Image to draw visualization on
        :return: Image with visualization overlays
        """
        for i, rectangle in enumerate(self.cluster_regions_of_interest):
            cluster_color_information = self.cluster_color_info[i]
            dominant_color = cluster_color_information[0][0]
            assert isinstance(dominant_color, Color)
            dominant_color_value = self.color_name_to_blue_green_red_values[
                dominant_color
            ]

            upper_left = (rectangle.position.x, rectangle.position.y)
            upper_left_text = (
                rectangle.position.x,
                rectangle.position.y - 5,
            )

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_image = cv2.putText(
                visualization_image,
                f"{i}: {dominant_color.name}",
                upper_left_text,
                font,
                0.5,
                dominant_color_value,
                1,
                2,
            )
            visualization_image = cv2.rectangle(
                visualization_image,
                upper_left,
                (
                    rectangle.position.x + rectangle.width,
                    rectangle.position.y + rectangle.height,
                ),
                dominant_color_value,
                2,
            )

            # Prepare Histogram
            histogram_size = rectangle.width, 10
            histogram_upper_left = (
                rectangle.position.x,
                rectangle.position.y + rectangle.height + 1,
            )
            histogram_lower_right = (
                rectangle.position.x + histogram_size[0],
                rectangle.position.y + rectangle.height + 1 + histogram_size[1],
            )
            histogram_region = crop_image(
                visualization_image, histogram_upper_left, histogram_size
            )

            histogram_region[:] = (255, 255, 255)

            # Draw the border of the Histogram
            visualization_image = cv2.rectangle(
                visualization_image,
                histogram_upper_left,
                histogram_lower_right,
                (0, 0, 0),
                1,
            )

            # Draw the actual histogram
            # Get the ratios of the colors, calculate their width and draw them into the region of the image
            # where the histogram is located.
            start = 0
            for element in cluster_color_information:
                # Triple: Name, Count, Ratio
                histogram_color = element[0]
                assert isinstance(histogram_color, Color)
                ratio = element[2]
                histogram_color_width = histogram_size[0] * ratio
                histogram_color_rectangle = (
                    int(start + 0.5),
                    0,
                    int(histogram_color_width + 0.5),
                    histogram_size[1],
                )
                start += histogram_color_width
                cv2.rectangle(
                    histogram_region,
                    histogram_color_rectangle,
                    self.color_name_to_blue_green_red_values[histogram_color],
                    thickness=-1,
                )
        return visualization_image
