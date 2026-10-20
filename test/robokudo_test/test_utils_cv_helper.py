from __future__ import annotations
import copy

import numpy as np
import open3d as o3d
import pytest
from cv2 import boundingRect

from robokudo.types.cv import ColorToDepthRatio
import robokudo.common_analysis_structure
from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.pipeline import Pipeline
from robokudo.types.cv import ImageROI
from robokudo.utils.cv_helper import (
    adjust_roi,
    get_hsv_for_rgb_color,
    crop_image,
    get_hsv_for_bgr_color,
    get_scaled_color_image_for_depth_image,
    draw_rectangle_around_center,
    clamp_bounding_rect,
    sanity_checks_bounding_rects,
    adjust_image_roi,
    adjust_mask,
    crop_image_roi,
)


class TestUtilsCVHelper(object):
    @pytest.fixture()
    def annotator_in_pipeline(self) -> BaseAnnotator:
        root = Pipeline("Sequence")
        annotator = ImagePreprocessorAnnotator("1")
        root.add_child(annotator)
        return annotator

    @pytest.fixture()
    def kinect_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.set_intrinsics(
            width=1024, height=1280, fx=1050.0, fy=1050.0, cx=639.5, cy=479.5
        )
        return camera_intrinsics

    @pytest.mark.parametrize(
        ["image_dimensions_xy", "crop_xy", "crop_width_height", "expected_yx_shape"],
        [
            # Crop exceeds image boundaries right and bottom
            ((256, 256), (0, 0), (257, 257), (256, 256)),
            # Image is same size as crop (valid)
            ((256, 256), (0, 0), (256, 256), (256, 256)),
            # Image is larger than crop (valid)
            ((256, 256), (128, 128), (64, 64), (64, 64)),
            # Crop exceeds image boundaries right
            ((256, 256), (128, 128), (256, 64), (64, 128)),
            # Crop exceeds image boundaries bottom
            ((256, 256), (128, 128), (64, 256), (128, 64)),
            # Crop exceeds image boundaries left
            ((256, 256), (-1, 128), (64, 256), (128, 63)),
            # Crop exceeds image boundaries top
            ((256, 256), (128, -1), (256, 64), (63, 128)),
        ],
    )
    def test_crop_image(
        self,
        image_dimensions_xy: tuple[int, int],
        crop_xy: tuple[int, int],
        crop_width_height: tuple[int, int],
        expected_yx_shape: tuple[int, int],
    ) -> None:
        image = np.zeros((image_dimensions_xy[0], image_dimensions_xy[1], 3), np.uint8)
        cropped_image = crop_image(image.copy(), crop_xy, crop_width_height)
        assert cropped_image.shape[:2] == expected_yx_shape

    @pytest.mark.parametrize(
        ["image_dimensions_xy", "crop_xy", "crop_width_height", "expected_yx_shape"],
        [
            # Crop exceeds image boundaries right and bottom
            ((256, 256), (0, 0), (257, 257), (256, 256)),
            # Image is same size as crop (valid)
            ((256, 256), (0, 0), (256, 256), (256, 256)),
            # Image is larger than crop (valid)
            ((256, 256), (128, 128), (64, 64), (64, 64)),
            # Crop exceeds image boundaries right
            ((256, 256), (128, 128), (256, 64), (64, 128)),
            # Crop exceeds image boundaries bottom
            ((256, 256), (128, 128), (64, 256), (128, 64)),
            # Crop exceeds image boundaries left
            ((256, 256), (-1, 128), (64, 256), (128, 63)),
            # Crop exceeds image boundaries top
            ((256, 256), (128, -1), (256, 64), (63, 128)),
        ],
    )
    def test_crop_image_roi(
        self,
        image_dimensions_xy: tuple[int, int],
        crop_xy: tuple[int, int],
        crop_width_height: tuple[int, int],
        expected_yx_shape: tuple[int, int],
    ) -> None:
        image = np.zeros((image_dimensions_xy[0], image_dimensions_xy[1], 3), np.uint8)
        region_of_interest = robokudo.types.cv.ImageROI()
        region_of_interest.roi.pos.x = crop_xy[0]
        region_of_interest.roi.pos.y = crop_xy[1]
        region_of_interest.roi.width = crop_width_height[0]
        region_of_interest.roi.height = crop_width_height[1]
        cropped_image = crop_image_roi(image.copy(), region_of_interest)
        assert cropped_image.shape[:2] == expected_yx_shape

    @pytest.mark.parametrize(
        "scale_factor",
        [
            (1.0, 1.0),  # No scaling
            (2.0, 2.0),  # Upscaling
            (0.5, 0.5),  # Downscaling
            (3.0, 2.0),  # Non-uniform scaling
            (2.0, 3.0),  # Non-uniform scaling
        ],
    )
    def test_get_scaled_color_image_for_depth_image(
        self,
        scale_factor: tuple[float, float],
        annotator_in_pipeline: BaseAnnotator,
        kinect_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    ) -> None:
        annotator_in_pipeline.descriptor.parameters.global_with_depth = True
        common_analysis_structure = (
            annotator_in_pipeline.get_common_analysis_structure()
        )
        width, height = kinect_intrinsics.width, kinect_intrinsics.height
        scale_x, scale_y = scale_factor

        image = np.zeros((height, width, 3), dtype=np.uint8)

        common_analysis_structure.set(
            robokudo.common_analysis_structure.CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO,
            ColorToDepthRatio(horizontal=scale_x, vertical=scale_y),
        )

        annotator_in_pipeline.camera_intrinsics = copy.deepcopy(kinect_intrinsics)

        scaled_image = get_scaled_color_image_for_depth_image(
            common_analysis_structure, image
        )
        assert scaled_image.shape[:2] == (
            height * scale_y,
            width * scale_x,
        ), f"Image should be scaled to {scale_x}, {scale_y}"

    def test_get_scaled_color_image_for_depth_image_no_depth_ratio(
        self, annotator_in_pipeline: BaseAnnotator
    ) -> None:
        common_analysis_structure = (
            annotator_in_pipeline.get_common_analysis_structure()
        )
        common_analysis_structure.set(
            robokudo.common_analysis_structure.CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO,
            None,
        )
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        assert pytest.raises(
            RuntimeError,
            get_scaled_color_image_for_depth_image,
            common_analysis_structure,
            image,
        )

    @pytest.mark.parametrize(
        ["red_green_blue", "hsv"],
        [
            # All cannels the same
            ((0, 0, 0), (0, 0, 0)),
            ((255, 255, 255), (0, 0, 255)),
            ((int(255 / 2), int(255 / 2), int(255 / 2)), (0, 0, int(255 / 2))),
            # Only red
            ((255, 0, 0), (0, 255, 255)),
            # Only green
            ((0, 255, 0), (85, 255, 255)),
            # Only blue
            ((0, 0, 255), (171, 255, 255)),
        ],
    )
    def test_get_hsv_for_rgb_color(
        self, red_green_blue: tuple[int, int, int], hsv: tuple[int, int, int]
    ) -> None:
        hsv_from_red_green_blue = get_hsv_for_rgb_color(red_green_blue)
        assert np.all(hsv == hsv_from_red_green_blue)

    @pytest.mark.parametrize(
        ["blue_green_red", "hsv"],
        [
            # All cannels the same
            ((0, 0, 0), (0, 0, 0)),
            ((255, 255, 255), (0, 0, 255)),
            ((int(255 / 2), int(255 / 2), int(255 / 2)), (0, 0, int(255 / 2))),
            # Only red
            ((0, 0, 255), (0, 255, 255)),
            # Only green
            ((0, 255, 0), (85, 255, 255)),
            # Only blue
            ((255, 0, 0), (171, 255, 255)),
        ],
    )
    def test_get_hsv_for_bgr_color(
        self, blue_green_red: tuple[int, int, int], hsv: tuple[int, int, int]
    ) -> None:
        hsv_from_blue_green_red = get_hsv_for_bgr_color(blue_green_red)
        assert np.all(hsv == hsv_from_blue_green_red)

    @pytest.mark.parametrize(
        ["center", "size", "expected_points"],
        [
            # Entire image rectangle
            ((256, 256), (512, 512), (512 * 512)),
            # Rectangle larger than image
            ((256, 256), (513, 513), (512 * 512)),
            # Zero width rectangle
            ((256, 256), (0, 128), 0),
            # Zero height rectangle
            ((256, 256), (128, 0), 0),
            # Zero width and height rectangle
            ((256, 256), (0, 0), 0),
            # One width and height rectangle
            ((256, 256), (1, 1), 1),
            # Even width and height rectangle (center pixel has to be added)
            ((256, 256), (128, 128), (129 * 129)),
            # Even width and height rectangle (center pixel possible)
            ((256, 256), (127, 127), (127 * 127)),
            # Even width and height with center outside image boundaries
            (
                (513, 513),
                (128, 128),
                ((128 / 2) - 1) * ((128 / 2) - 1),
            ),  # Added center row/col outside of image
            # Uneven width and height with center outside image boundaries
            (
                (513, 513),
                (127, 127),
                ((127 // 2) - 1) * ((127 // 2) - 1),
            ),  # Center row/col outside of image
            # Even width and height with center outside image boundaries
            (
                (-1, -1),
                (128, 128),
                ((128 / 2) * (128 / 2)),
            ),  # Added center row/col outside of image
            # Uneven width and height with center outside image boundaries
            (
                (-1, -1),
                (127, 127),
                (127 // 2) * (127 // 2),
            ),  # Center row/col outside of image
        ],
    )
    def test_draw_rectangle_around_center(
        self, center: tuple[int, int], size: tuple[int, int], expected_points: int
    ) -> None:
        x, y = center
        width, height = size

        image = np.zeros((512, 512, 3), dtype=np.uint8)
        modified_image = draw_rectangle_around_center(
            image, x, y, width, height, value=(255, 0, 0)
        )

        assert (
            np.sum(modified_image != 0) == expected_points
        ), f"Expected {expected_points} pixels to be set"
        assert image.shape == modified_image.shape, "Image shape changed"

    @pytest.mark.parametrize(
        ["points", "adjusted"],
        [
            # No adjustment needed
            ((128, 128, 128, 128), (128, 128, 128, 128)),
            # Adjust width
            ((128, 128, 512, 128), (128, 128, 512 - 128, 128)),
            # Adjust height
            ((128, 128, 128, 512), (128, 128, 128, 512 - 128)),
            # Adjust x and width
            ((-64, 128, 128, 128), (0, 128, 64, 128)),
            ((-128, 128, 128, 128), (0, 128, 0, 128)),
            # Adjust y and height
            ((128, -64, 128, 128), (128, 0, 128, 64)),
            ((128, -128, 128, 128), (128, 0, 128, 0)),
            # Adjust all
            ((-64, -64, 768, 768), (0, 0, 512, 512)),
        ],
    )
    def test_clamp_bounding_rect(
        self, points: tuple[int, int, int, int], adjusted: tuple[int, int, int, int]
    ) -> None:
        image_width = 512
        image_height = 512

        result = clamp_bounding_rect(points, image_width, image_height)

        assert np.all(result == adjusted)

        x, y, width, height = result

        assert 0 <= x + width <= image_width
        assert 0 <= y + height <= image_height

        assert 0 <= x < image_width
        assert 0 <= y < image_height

    @pytest.mark.parametrize(
        ["points", "valid"],
        [
            # Valid boundingRect
            (np.array([(128, 128), (192, 192)]), True),
            # Invalid x
            (np.array([(-4096, 128), (128, 128)]), False),
            # Invalid y
            (np.array([(128, -4096), (128, 128)]), False),
            # Invalid w
            (np.array([(0, 0), (4096, 128)]), False),
            # Invalid h
            (np.array([(0, 0), (128, 4096)]), False),
            # All Invalid
            (np.array([(-4096, -4096), (4096, 4096)]), False),
            # Roi starts oob
            (np.array([(4096, 4096), (4096, 4096)]), False),
        ],
    )
    def test_sanity_checks_bounding_rects(
        self, points: np.ndarray, valid: bool
    ) -> None:
        image_width = 512
        image_height = 512

        rectangle = boundingRect(points)

        assert (
            sanity_checks_bounding_rects(rectangle, image_width, image_height) == valid
        )

    @pytest.mark.parametrize(
        ["image_dimensions", "region_of_interest", "offset"],
        [
            # Roi no change
            ((512, 512), (64, 64, 128, 128), 0),
            # Roi expanded within image boundaries
            ((512, 512), (64, 64, 128, 128), 10),
            # Roi shrunk within image boundaries
            ((512, 512), (64, 64, 128, 128), -10),
        ],
    )
    def test_adjust_roi_within_boundaries(
        self,
        image_dimensions: tuple[int, int],
        region_of_interest: tuple[int, int, int, int],
        offset: int,
    ) -> None:
        width, height = image_dimensions
        image = np.zeros((height, width, 3), dtype=np.uint8)

        new_x, new_y, new_width, new_height = adjust_roi(
            image, region_of_interest, offset
        )

        assert new_x == region_of_interest[0] - offset
        assert new_y == region_of_interest[1] - offset
        assert new_width == region_of_interest[2] + 2 * offset
        assert new_height == region_of_interest[3] + 2 * offset

    @pytest.mark.parametrize(
        ["image_dimensions", "region_of_interest", "offset"],
        [
            # Roi shrink to minimum
            ((512, 512), (40, 40, 20, 20), -10),
            # Roi shrink below minimum
            ((512, 512), (40, 40, 20, 20), -35),
            # Roi expanded beyond left
            ((512, 1024), (0, 64, 256, 128), 10),
            # Roi expanded beyond right
            ((512, 1024), (256, 64, 256, 128), 10),
            # Roi expanded beyond top
            ((1024, 512), (64, 0, 128, 256), 10),
            # Roi expanded beyond bottom
            ((1024, 512), (64, 256, 128, 256), 10),
            # Roi expanded beyond left and right
            ((512, 1024), (0, 64, 512, 128), 10),
            # Roi expanded beyond top and bottom
            ((1024, 512), (64, 0, 128, 512), 10),
            # Roi expanded beyond top, left, bottom and right
            ((512, 512), (0, 0, 512, 512), 10),
        ],
    )
    def test_adjust_roi_boundary_conditions(
        self,
        image_dimensions: tuple[int, int],
        region_of_interest: tuple[int, int, int, int],
        offset: int,
    ) -> None:
        width, height = image_dimensions
        image = np.zeros((height, width, 3), dtype=np.uint8)

        new_x, new_y, new_width, new_height = adjust_roi(
            image, region_of_interest, offset
        )

        assert (
            0 <= new_x < width
        ), f"{region_of_interest} x should be expanded by {offset} within image width"
        assert (
            0 <= new_y < height
        ), f"{region_of_interest} y should be expanded by {offset} within image width"
        assert (
            1 <= new_width <= width
        ), f"{region_of_interest} width should be at least 1 and within image width"
        assert (
            1 <= new_height <= height
        ), f"{region_of_interest} width should be at least 1 and within image width"
        assert (
            1 <= new_x + new_width <= width
        ), f"{region_of_interest} width should be expanded by {offset} within image width"
        assert (
            1 <= new_y + new_height <= height
        ), f"{region_of_interest} height should be expanded by {offset} within image width"

    @pytest.mark.parametrize(
        ["image_dimensions", "region_of_interest", "offset"],
        [
            # Roi no change
            ((512, 512), (64, 64, 128, 128), 0),
            # Roi expanded within image boundaries
            ((512, 512), (64, 64, 128, 128), 10),
            # Roi shrunk within image boundaries
            ((512, 512), (64, 64, 128, 128), -10),
        ],
    )
    def test_adjust_image_roi_within_boundaries(
        self,
        image_dimensions: tuple[int, int],
        region_of_interest: tuple[int, int, int, int],
        offset: int,
    ) -> None:
        width, height = image_dimensions

        image_region_of_interest = ImageROI()
        image_region_of_interest.roi.pos.x, image_region_of_interest.roi.pos.y = (
            region_of_interest[0],
            region_of_interest[1],
        )
        image_region_of_interest.roi.width, image_region_of_interest.roi.height = (
            region_of_interest[2],
            region_of_interest[3],
        )

        image = np.zeros((height, width, 3), dtype=np.uint8)

        adjust_image_roi(image, image_region_of_interest, offset)

        assert image_region_of_interest.roi.pos.x == region_of_interest[0] - offset
        assert image_region_of_interest.roi.pos.y == region_of_interest[1] - offset
        assert image_region_of_interest.roi.width == region_of_interest[2] + 2 * offset
        assert image_region_of_interest.roi.height == region_of_interest[3] + 2 * offset

    @pytest.mark.parametrize(
        ["image_dimensions", "region_of_interest", "offset"],
        [
            # Roi shrink to minimum
            ((512, 512), (40, 40, 20, 20), -10),
            # Roi shrink below minimum
            ((512, 512), (40, 40, 20, 20), -35),
            # Roi expanded beyond left
            ((512, 1024), (0, 64, 256, 128), 10),
            # Roi expanded beyond right
            ((512, 1024), (256, 64, 256, 128), 10),
            # Roi expanded beyond top
            ((1024, 512), (64, 0, 128, 256), 10),
            # Roi expanded beyond bottom
            ((1024, 512), (64, 256, 128, 256), 10),
            # Roi expanded beyond left and right
            ((512, 1024), (0, 64, 512, 128), 10),
            # Roi expanded beyond top and bottom
            ((1024, 512), (64, 0, 128, 512), 10),
            # Roi expanded beyond top, left, bottom and right
            ((512, 512), (0, 0, 512, 512), 10),
        ],
    )
    def test_adjust_image_roi_boundary_conditions(
        self,
        image_dimensions: tuple[int, int],
        region_of_interest: tuple[int, int, int, int],
        offset: int,
    ) -> None:
        width, height = image_dimensions
        offset = offset

        image_region_of_interest = ImageROI()
        image_region_of_interest.roi.pos.x, image_region_of_interest.roi.pos.y = (
            region_of_interest[0],
            region_of_interest[1],
        )
        image_region_of_interest.roi.width, image_region_of_interest.roi.height = (
            region_of_interest[2],
            region_of_interest[3],
        )

        image = np.zeros((height, width, 3), dtype=np.uint8)

        adjust_image_roi(image, image_region_of_interest, offset)

        assert (
            0 <= image_region_of_interest.roi.pos.x <= width
        ), f"{region_of_interest} x should be expanded by {offset} within image width"
        assert (
            0 <= image_region_of_interest.roi.pos.y <= height
        ), f"{region_of_interest} y should be expanded by {offset} within image width"
        assert (
            1 <= image_region_of_interest.roi.width <= width
        ), f"{region_of_interest} width should be at least 1 and within image width"
        assert (
            1 <= image_region_of_interest.roi.height <= height
        ), f"{region_of_interest} width should be at least 1 and within image width"
        assert (
            1
            <= image_region_of_interest.roi.pos.x + image_region_of_interest.roi.width
            <= width
        ), f"{region_of_interest} width should be expanded by {offset} within image width"
        assert (
            1
            <= image_region_of_interest.roi.pos.y + image_region_of_interest.roi.height
            <= height
        ), f"{region_of_interest} height should be expanded by {offset} within image width"

    @pytest.mark.parametrize(
        ["mask", "offset", "fill_value"],
        [
            # Unchanged empty mask
            (np.empty((0, 0), dtype=np.uint8), 0, 0),
            # Expand empty mask without fill
            (np.empty((0, 0), dtype=np.uint8), 1, 0),
            # Expand empty mask with fill
            (np.empty((0, 0), dtype=np.uint8), 1, 1),
            # Unchanged zero mask
            (np.zeros((512, 512), dtype=np.uint8), 0, 0),
            # Expand zero mask without
            (np.zeros((512, 512), dtype=np.uint8), 10, 0),
            # Shrink zero mask without
            (np.zeros((512, 512), dtype=np.uint8), -10, 0),
        ],
    )
    def test_adjust_mask(self, mask: np.ndarray, offset: int, fill_value: int) -> None:
        new_mask = adjust_mask(mask, offset, fill_value=fill_value)
        assert np.all(new_mask == fill_value)
        assert np.all(new_mask.shape[:2] == np.array(mask.shape[:2]) + (offset * 2))

    def test_adjust_mask_shrink_empty_mask(self) -> None:
        mask = np.empty((0, 0), dtype=np.uint8)
        new_mask = adjust_mask(mask, -1, fill_value=0)
        assert np.all(new_mask == 0)
        assert np.all(new_mask.shape[:2] == np.array(mask.shape[:2]))

    def test_adjust_mask_unchanged_random_mask(self) -> None:
        mask = np.random.randint(0, 2, size=(512, 512), dtype=np.uint8)
        new_mask = adjust_mask(mask, 0, fill_value=0)
        assert np.array_equal(mask, new_mask)
        assert np.all(new_mask.shape[:2] == mask.shape[:2])

    def test_adjust_mask_expand_random_mask(self) -> None:
        offset = 10
        fill = 0
        mask = np.random.randint(0, 2, size=(512, 512), dtype=np.uint8)
        new_mask = adjust_mask(mask, 10, fill_value=fill)
        assert np.array_equal(
            new_mask[offset : offset + mask.shape[0], offset : offset + mask.shape[1]],
            mask,
        )
        assert np.all(new_mask.shape[:2] == np.array(mask.shape[:2]) + 20)
        # Newly added areas should be filled with fill
        assert np.all(new_mask[0:offset, :] == fill)  # Top
        assert np.all(new_mask[:, 0:offset] == fill)  # Left
        assert np.all(new_mask[-offset:, :] == fill)  # Bottom
        assert np.all(new_mask[:, -offset:] == fill)  # Right

    def test_adjust_mask_shrink_random_mask(self) -> None:
        offset = -10
        mask = np.random.randint(0, 2, size=(512, 512), dtype=np.uint8)
        new_mask = adjust_mask(mask, offset, fill_value=0)

        absolute_offset = abs(offset)
        expected_slice = (
            slice(absolute_offset, mask.shape[0] - absolute_offset),
            slice(absolute_offset, mask.shape[1] - absolute_offset),
        )
        assert np.array_equal(new_mask, mask[expected_slice])
        assert np.all(new_mask.shape[:2] == np.array(mask.shape[:2]) - 20)
