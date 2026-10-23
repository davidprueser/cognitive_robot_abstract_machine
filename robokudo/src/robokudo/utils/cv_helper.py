"""
OpenCV helper utilities for Robokudo.

This module provides helper functions for working with OpenCV images and operations.
It supports:

* Image cropping and region of interest handling
* ROS-OpenCV image conversion
* Color space conversions
* Bounding box operations
* Image scaling and resizing
* Drawing utilities
"""

from __future__ import annotations

import cv2
import numpy as np
from typing_extensions import TYPE_CHECKING, Tuple

from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.exceptions import ColorToDepthRatioMissing
from robokudo.types.cv import ImageRegionOfInterest

if TYPE_CHECKING:
    import cv2.typing as cv2t
    import numpy.typing as npt
    from robokudo.common_analysis_structure import CommonAnalysisStructure


def crop_image(image: npt.NDArray, xy: tuple, width_height: tuple) -> npt.NDArray:
    """
    Crop region from image using coordinates and dimensions.

    Based on https://stackoverflow.com/a/67799880

    :param image: Input image
    :param xy: Top-left corner coordinates (x,y)
    :param width_height: Width and height of crop region (w,h)
    :return: Cropped image region
    """
    x, y = xy
    width, height = width_height

    x0, y0 = max(xy[0], 0), max(xy[1], 0)
    x1, y1 = min(x + width, image.shape[1]), min(y + height, image.shape[0])

    crop = image[y0:y1, x0:x1]
    return crop


def crop_image_region_of_interest(
    image: npt.NDArray, region_of_interest: ImageRegionOfInterest
) -> npt.NDArray:
    """
    Crop region from image using region of interest object.

    :param image: Input image
    :param region_of_interest: Region of interest
    :return: Cropped image region
    """
    return crop_image(
        image,
        (
            region_of_interest.rectangle.position.x,
            region_of_interest.rectangle.position.y,
        ),
        (region_of_interest.rectangle.width, region_of_interest.rectangle.height),
    )


def get_scaled_color_image_for_depth_image(
    common_analysis_structure: CommonAnalysisStructure, color_image: npt.NDArray
) -> npt.NDArray:
    """
    Scale color image to match depth image resolution.

    If the color-to-depth ratio is not 1,1, we will usually scale the color image to the
    same size the depth image has. This method will get check if a conversion is
    required and will return the correct size. This is usually called before
    o3d.geometry.RGBDImage.create_from_color_and_depth is called, so that depth and
    color image match.

    :param common_analysis_structure: The common analysis structure holding the color-to-depth ratio.
    :param color_image: The color image to be adjusted
    :return: A resized copy of the color image if a resize necessary. Otherwise, the
        unchanged color_image is returned.
    :raises ColorToDepthRatioMissing: If color-to-depth ratio not set in common analysis structure
    """
    color_to_depth_ratio = common_analysis_structure.get(
        CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO
    )

    if color_to_depth_ratio is None:
        raise ColorToDepthRatioMissing(
            operation="scale color image to depth image resolution"
        )

    if color_to_depth_ratio.is_identity:
        return color_image
    return cv2.resize(
        color_image,
        None,
        fx=color_to_depth_ratio.horizontal,
        fy=color_to_depth_ratio.vertical,
        interpolation=cv2.INTER_NEAREST,
    )


def get_hsv_for_red_green_blue_color(
    red_green_blue: Tuple[int, int, int],
) -> npt.NDArray:
    """
    Convert RGB color to HSV color space.

    :param red_green_blue: RGB color values (r,g,b)
    :return: HSV color values
    """
    return cv2.cvtColor(np.uint8([[list(red_green_blue)]]), cv2.COLOR_RGB2HSV_FULL)


def get_hsv_for_blue_green_red_color(
    blue_green_red: Tuple[int, int, int],
) -> npt.NDArray:
    """
    Convert BGR color to HSV color space.

    :param blue_green_red: BGR color values (b,g,r)
    :return: HSV color values
    """
    return cv2.cvtColor(np.uint8([[list(blue_green_red)]]), cv2.COLOR_BGR2HSV_FULL)


def draw_rectangle_around_center(
    binary_image: npt.NDArray, x: int, y: int, width: int, height: int, value: int = 255
) -> npt.NDArray:
    """
    Draw filled rectangle centered at given point.

    :param binary_image: Binary image to draw on
    :param x: Center x coordinate
    :param y: Center y coordinate
    :param width: Rectangle width
    :param height: Rectangle height
    :param value: Fill value for rectangle
    :return: Image with drawn rectangle
    """
    if width <= 0 or height <= 0:
        return binary_image
    # Consider x and y as the center point of the interest and draw rectangle around it
    center = (int(x), int(y))  # Cast to int for safety
    first_corner = (center[0] - width // 2, center[1] - height // 2)
    second_corner = (center[0] + width // 2, center[1] + height // 2)
    return cv2.rectangle(
        img=binary_image,
        pt1=first_corner,
        pt2=second_corner,
        color=value,
        thickness=cv2.FILLED,
    )


def clamp_bounding_rectangle(
    bounding_rectangle: cv2t.Rect, image_width: int, image_height: int
) -> cv2t.Rect:
    """
    Clamp bounding rectangle to image boundaries.

    :param bounding_rectangle: Input bounding rectangle (x,y,w,h)
    :param image_width: Image width
    :param image_height: Image height
    :return: Clamped bounding rectangle
    """
    x1, y1, width, height = bounding_rectangle
    x2, y2 = x1 + width, y1 + height

    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))

    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return x1, y1, x2 - x1, y2 - y1


def rectangle_outside_image(
    bounding_rectangle: cv2t.Rect, image_width: int, image_height: int
) -> bool:
    """
    Check whether a bounding rectangle is completely outside an image.

    A bounding rectangle is considered outside if it has no overlapping area with the
    image. Rectangles that only touch the image border (zero-area intersection) are
    treated as outside.

    :param bounding_rectangle: Bounding rectangle to check (x, y, w, h)
    :param image_width: Image width
    :param image_height: Image height
    :return: True if the bounding rectangle lies completely outside the image, False
        otherwise
    """
    x, y, width, height = bounding_rectangle

    # Rectangle edges
    rectangle_left = x
    rectangle_right = x + width
    rectangle_top = y
    rectangle_bottom = y + height

    # Image edges
    image_left = 0
    image_right = image_width
    image_top = 0
    image_bottom = image_height

    # Completely outside if one separating axis exists
    return (
        rectangle_right <= image_left  # left of image
        or rectangle_left >= image_right  # right of image
        or rectangle_bottom <= image_top  # above image
        or rectangle_top >= image_bottom  # below image
    )


def sanity_checks_bounding_rectangles(
    bounding_rectangle: cv2t.Rect, image_width: int, image_height: int
) -> bool:
    """
    Check if bounding rectangle is valid for image dimensions.

    Check basic rules of a boundingRect to reject bad ones. Might happen if projections
    are done on objects that are out-of-view etc.

    :param bounding_rectangle: Bounding rectangle to check (x,y,w,h)
    :param image_width: Image width
    :param image_height: Image height
    :return: True if rules have been passed, False otherwise
    """
    # Check if region of interest starts out of bounds - In that case, we'll skip this Object Hypothesis
    if bounding_rectangle[0] > image_width or bounding_rectangle[1] > image_height:
        return False

    # Check if the estimated Width and Height of the bounding box are even bigger than the image
    if bounding_rectangle[2] > image_width or bounding_rectangle[3] > image_height:
        return False

    return True


def adjust_region_of_interest(
    image: npt.NDArray, region_of_interest: Tuple[int, int, int, int], offset: int
) -> Tuple[int, int, int, int]:
    """
    Adjusts the region of interest by a given offset while respecting image boundaries, keeping the
    same center point.

    :param image: The image (OpenCV format).
    :param region_of_interest: A tuple (x, y, width, height) representing the bounding box.
    :param offset: The amount by which to grow or shrink the region of interest (can be negative).
    :return: A tuple (new_x, new_y, new_width, new_height) representing the adjusted
        region of interest.
    """
    x, y, width, height = region_of_interest

    # Calculate the center of the current region of interest
    center_x = x + width // 2
    center_y = y + height // 2

    # Calculate the new width and height
    new_width = max(1, width + 2 * offset)
    new_height = max(1, height + 2 * offset)

    # Calculate the new top-left position
    new_x = center_x - new_width // 2
    new_y = center_y - new_height // 2

    # Ensure the new region of interest fits within the image boundaries
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_width > image.shape[1]:
        new_x = image.shape[1] - new_width
    if new_y + new_height > image.shape[0]:
        new_y = image.shape[0] - new_height

    # Adjust the width and height if region of interest is at the boundary
    if new_x < 0:
        new_width += new_x
        new_x = 0
    if new_y < 0:
        new_height += new_y
        new_y = 0

    return new_x, new_y, new_width, new_height


def adjust_image_region_of_interest(
    image: npt.NDArray, image_region_of_interest: ImageRegionOfInterest, offset: int
) -> None:
    """
    Adjusts the ImageRegionOfInterest's region of interest by a given offset while respecting image boundaries,
    keeping the same center point. This method uses the adjust_region_of_interest function.

    :param image: The image this region of interest is relative to. Necessary to respect the boundaries
        properly.
    :param image_region_of_interest: An object of type ImageRegionOfInterest.
    :param offset: The amount by which to grow or shrink the region of interest (can be negative).
    :return: None. The ImageRegionOfInterest's region of interest is adjusted in place.
    """
    # Extract the current region of interest as a tuple (x, y, width, height)
    region_of_interest_tuple = (
        image_region_of_interest.rectangle.position.x,
        image_region_of_interest.rectangle.position.y,
        image_region_of_interest.rectangle.width,
        image_region_of_interest.rectangle.height,
    )

    # Use the adjust_region_of_interest function to get the new region of interest
    new_x, new_y, new_width, new_height = adjust_region_of_interest(
        image, region_of_interest_tuple, offset
    )

    # Update the ImageRegionOfInterest object with the new values
    image_region_of_interest.rectangle.position.x = new_x
    image_region_of_interest.rectangle.position.y = new_y
    image_region_of_interest.rectangle.width = new_width
    image_region_of_interest.rectangle.height = new_height


def adjust_mask(mask: npt.NDArray, offset: int, fill_value: int = 0) -> npt.NDArray:
    """
    Adjusts the mask by a given offset.

    When expanding, the new areas are filled with the specified fill value.
    :param mask: The mask corresponding to the region of interest (same dimensions as the region of interest).
    :param offset: The amount by which to grow or shrink the mask (can be negative).
    :param fill_value: The value to fill new areas when expanding the mask.
    :return: The adjusted mask.
    """
    height, width = mask.shape

    # Calculate the new dimensions for the mask
    new_width = max(1, width + 2 * offset)
    new_height = max(1, height + 2 * offset)

    if offset > 0:
        # Growing the mask
        new_mask = np.full((new_height, new_width), fill_value, dtype=mask.dtype)
        new_mask[offset : offset + height, offset : offset + width] = mask
    elif offset < 0:
        # Shrinking the mask
        new_mask = mask[-offset : -offset + new_height, -offset : -offset + new_width]
    else:
        new_mask = mask

    return new_mask
