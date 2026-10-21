from __future__ import annotations

import colorsys
import copy

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, List, Sequence

from robokudo.annotators.core import ThreadedAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import SIFTAnnotation
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio


class SIFTAnnotator(ThreadedAnnotator):
    """
    Annotator for SIFT feature extraction and matching using OpenCV.
    """

    def __init__(
        self,
        name: str = "SIFTAnnotator",
    ) -> None:
        super().__init__(name)

        self._sift: cv2.SIFT = cv2.SIFT_create()
        """
        SIFT feature extractor.
        """

    def compute(self) -> Status:
        """
        Compute the SIFT features of the current image and match them to the last image.

        :return: The status of the computation.
        """
        common_analysis_structure = self.get_common_analysis_structure()

        object_hypotheses: List[ObjectHypothesis] = (
            common_analysis_structure.filter_annotations_by_type(ObjectHypothesis)
        )
        if len(object_hypotheses) == 0:
            return Status.FAILURE

        intrinsic = common_analysis_structure.camera_intrinsic
        if intrinsic is None:
            return Status.FAILURE

        color_to_depth_ratio = common_analysis_structure.color_to_depth_ratio
        if color_to_depth_ratio is None:
            return Status.FAILURE

        color_image = common_analysis_structure.get_copy(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        depth_image = common_analysis_structure.get_copy(
            CommonAnalysisStructureViews.DEPTH_IMAGE
        )
        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        vis_image = color_image.copy()

        all_keypoints: List[cv2.KeyPoint] = []
        for i, object_hypothesis in enumerate(object_hypotheses):
            region_of_interest = (
                object_hypothesis.region_of_interest.rectangle.get_corner_points()
            )

            region_of_interest_image = grey_image[
                region_of_interest[1] : region_of_interest[3],
                region_of_interest[0] : region_of_interest[2],
            ]
            if (
                object_hypothesis.region_of_interest.mask.shape[:2]
                != region_of_interest_image.shape[:2]
            ):
                region_of_interest_mask = object_hypothesis.region_of_interest.mask[
                    region_of_interest[1] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[2],
                ]
            else:
                region_of_interest_mask = object_hypothesis.region_of_interest.mask

            keypoints, descriptors = self._sift.detectAndCompute(
                region_of_interest_image, region_of_interest_mask
            )

            for k in keypoints:
                k.pt = (
                    k.pt[0] + region_of_interest[0],
                    k.pt[1] + region_of_interest[1],
                )

            object_hypothesis.annotations.append(
                SIFTAnnotation.from_cv2(
                    keypoints=keypoints,
                    descriptors=descriptors,
                )
            )

            all_keypoints.extend(keypoints)

            vis_image = self._draw_visualization(
                color_image,
                vis_image,
                keypoints,
            )

        point_cloud = self._keypoints_to_point_cloud(
            all_keypoints, depth_image, intrinsic, color_to_depth_ratio
        )
        self.get_annotator_output_structure().set_image(vis_image)
        self.get_annotator_output_structure().set_geometries(point_cloud)
        return Status.SUCCESS

    def _draw_visualization(
        self,
        color_image: npt.NDArray[np.uint8],
        vis_image: npt.NDArray[np.uint8],
        current_keypoints: Sequence[cv2.KeyPoint],
    ) -> npt.NDArray[np.uint8]:
        """
        Add visualizations to the image.

        :param color_image: The current color image.
        :param vis_image: The image to add visualizations to.
        :param current_keypoints: The keypoints from the current image.
        :return: The image with visualizations.
        """
        return cv2.drawKeypoints(
            color_image,
            current_keypoints,
            vis_image,
            flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
            | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )

    def _keypoints_to_point_cloud(
        self,
        keypoints: List[cv2.KeyPoint],
        depth_image: npt.NDArray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        color_to_depth_ratio: ColorToDepthRatio,
        depth_scale: float = 1000.0,
        max_depth: float = 3.0,
    ) -> o3d.geometry.PointCloud:
        """
        Project the keypoints to a 3D point cloud.

        :param keypoints: The keypoints to project to 3D.
        :param depth_image: The depth image used for projection.
        :param intrinsics: The camera intrinsics used for projection.
        :param color_to_depth_ratio: Size of the depth image relative to the color
            image.
        :param depth_scale: The depth scale of the depth image.
        :param max_depth: The maximum depth to project.
        :return: The 3D point cloud.
        """
        intrinsic_matrix = intrinsics.intrinsic_matrix
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        points_3d = []
        responses = []
        for keypoint in keypoints:
            u = round(keypoint.pt[0] * color_to_depth_ratio.horizontal)
            v = round(keypoint.pt[1] * color_to_depth_ratio.vertical)
            if not (0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]):
                continue

            z = depth_image[v, u] / depth_scale
            if z <= 0 or z > max_depth:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d.append([x, y, z])
            responses.append(keypoint.response)

        colors = self._keypoint_colors(keypoints, np.array(responses))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            np.array(points_3d, dtype=np.float32)
        )
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return point_cloud

    @staticmethod
    def _keypoint_colors(
        keypoints: List[cv2.KeyPoint],
        responses: npt.NDArray,
    ) -> npt.NDArray[np.float64]:
        """
        Get red_green_blue colors for keypoints based on their responses and angles.

        :param keypoints: The keypoints.
        :param responses: The keypoints responses.
        :return: The red_green_blue colors.
        """

        def _normalize(array: npt.NDArray) -> npt.NDArray:
            low, high = array.min(), array.max()
            return (array - low) / (high - low + 1e-8)

        values = 0.4 + 0.6 * _normalize(responses)

        red_green_blue = []
        for keypoint, v in zip(keypoints, values):
            seed = hash(
                (
                    round(keypoint.pt[0], 1),
                    round(keypoint.pt[1], 1),
                    round(keypoint.size, 1),
                )
            )
            random_generator: npt.NDArray[np.float32] = np.random.default_rng(
                abs(seed) % (2**32)
            )

            hue = random_generator.uniform(0.0, 1.0)
            red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, v)
            red_green_blue.append([red, green, blue])

        return np.array(red_green_blue, dtype=np.float64)
