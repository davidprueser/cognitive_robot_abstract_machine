from __future__ import annotations

import copy
from enum import Enum

import cv2
import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import (
    PoseAnnotation,
    TruncatedSignedDistanceFunctionAnnotation,
)
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.comparators import TranslationComparator
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.o3d_helper import scale_o3d_camera_intrinsics
from robokudo.utils.transform import get_transform_matrix_from_quaternion

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio


class TruncatedSignedDistanceFunctionAnnotatorVisualizationModes(Enum):
    """Enumeration of supported visualization modes for truncated signed distance function volumes."""

    NONE = "none"
    MESH = "mesh"
    POINTCLOUD = "pointcloud"
    VOXEL_POINTCLOUD = "voxel_pointcloud"


class TruncatedSignedDistanceFunctionAnnotator(ThreadedAnnotator):
    """Annotator for per-object truncated signed distance function volume integration.

    .. note::
        The annotator requires a camera to world transform and object poses to integrate the truncated signed distance function volumes.
        To segment objects out of the image a mask is required too. Higher quality masks yield better results.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for the truncated signed distance function annotator."""

        class Parameters(BaseAnnotator.Descriptor.Parameters):
            """Parameters for configuring the truncated signed distance function annotator."""

            def __init__(self) -> None:
                self.voxel_length: float = 0.005
                """Voxel length for the truncated signed distance function volume of each object."""

                self.signed_distance_function_truncation: float = 0.04
                """Signed distance function truncation distance for the truncated signed distance function volume of each object."""

                self.depth_scale: float = 1000.0
                """Depth scale for the truncated signed distance function volume integration."""

                self.depth_truncation: float = 3.0
                """Depth truncation distance for the truncated signed distance function volume integration."""

                self.geometry_visualization_mode: (
                    TruncatedSignedDistanceFunctionAnnotatorVisualizationModes
                ) = TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.MESH
                """Whether to visualize the truncated signed distance function as a mesh or a point cloud. """

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "TruncatedSignedDistanceFunctionAnnotator",
        descriptor: TruncatedSignedDistanceFunctionAnnotator.Descriptor | None = None,
    ) -> None:
        super().__init__(name, descriptor)

        self.translation_comparator = TranslationComparator(
            weight=1.0, max_distance=0.4
        )
        """Translation comparator for comparing world positions between pipeline iterations."""

        self.tracked_objects: List[Dict[str, Any]] = []
        """A list of tracked object data, including the bounding box and the truncated signed distance function volume."""

    def find_tracked_object(
        self, pose: PoseAnnotation, threshold: float = 0.75
    ) -> Optional[Dict[str, Any]]:
        """Check whether the given bounding box is similar to one that has been seen before.

        :param pose: The pose to check
        :param threshold: The similarity threshold between 0.0 and 1.0 for considering the bounding box similar.
        :return: The most similar tracked object or none if no similar object was found.
        """
        best_sim: float = 0.0
        best_object: Optional[Dict[str, Any]] = None
        for tracked_object in self.tracked_objects:
            sim = self.translation_comparator.compute_similarity(
                pose.translation, tracked_object["pose"].translation
            )
            if sim > best_sim:
                best_object = tracked_object
                best_sim = sim
        return best_object if best_sim > threshold else None

    def compute(self) -> Status:
        """Generate a truncated signed distance function volume for each object hypothesis or integrate into existing volumes.

        :return: Whether the truncated signed distance function integration was successful or not.
        """
        common_analysis_structure = self.get_common_analysis_structure()
        if common_analysis_structure is None:
            return Status.FAILURE

        object_hypotheses: List[ObjectHypothesis] = (
            common_analysis_structure.filter_annotations_by_type(ObjectHypothesis)
        )
        if len(object_hypotheses) == 0:
            return Status.FAILURE

        color_to_depth_ratio: ColorToDepthRatio = common_analysis_structure.get(
            CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO
        )

        # Only process object_hypotheses with pose
        object_hypothesis_data: List[
            Tuple[ObjectHypothesis, PoseAnnotation, Tuple[int, int, int, int]]
        ] = []
        for object_hypothesis in object_hypotheses:
            poses = common_analysis_structure.filter_by_type(
                PoseAnnotation, object_hypothesis.annotations
            )
            if len(poses) > 0:
                region_of_interest: Tuple[int, int, int, int] = (
                    color_to_depth_ratio.depth_region(
                        object_hypothesis.region_of_interest.rectangle
                    ).get_corner_points()
                )
                object_hypothesis_data.append(
                    (object_hypothesis, poses[0], region_of_interest)
                )
            else:
                self.robokudo_logger.debug(
                    f"No pose found for object {object_hypothesis.id}, skipping..."
                )

        if len(object_hypothesis_data) == 0:
            return Status.FAILURE

        camera_intrinsic: o3d.camera.PinholeCameraIntrinsic = (
            scale_o3d_camera_intrinsics(
                common_analysis_structure.get(
                    CommonAnalysisStructureViews.CAMERA_INTRINSIC
                ),
                color_to_depth_ratio.horizontal,
                color_to_depth_ratio.vertical,
            )
        )
        color_image: npt.NDArray[np.uint8] = get_scaled_color_image_for_depth_image(
            common_analysis_structure,
            copy.deepcopy(
                self.get_common_analysis_structure().get(
                    CommonAnalysisStructureViews.COLOR_IMAGE
                )
            ),
        )
        depth_image: npt.NDArray[np.float32] = copy.deepcopy(
            self.get_common_analysis_structure().get(
                CommonAnalysisStructureViews.DEPTH_IMAGE
            )
        )

        geometries: List[o3d.geometry.Geometry] = []
        masks: List[npt.NDArray[np.uint8]] = []
        for i, (object_hypothesis, pose, region_of_interest) in enumerate(
            object_hypothesis_data
        ):
            mask: npt.NDArray[np.uint8] = object_hypothesis.region_of_interest.mask
            if mask is None:
                self.robokudo_logger.debug(
                    f"No mask found for object {object_hypothesis.id}, skipping..."
                )
                continue

            # Mask should be or (0, 1)
            if mask.max() == 255:
                mask = np.clip(mask, 0, 1)

            mask = get_scaled_color_image_for_depth_image(
                common_analysis_structure, mask
            )

            if mask.shape[:2] == depth_image.shape[:2]:
                masked_depth = depth_image * mask
                masked_color = color_image * mask[..., np.newaxis]
            else:
                masked_depth = np.zeros_like(depth_image, dtype=np.float32)
                masked_depth[
                    region_of_interest[1] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[2],
                ] = (
                    depth_image[
                        region_of_interest[1] : region_of_interest[3],
                        region_of_interest[0] : region_of_interest[2],
                    ]
                    * mask
                )

                masked_color = np.zeros_like(color_image, dtype=np.uint8)
                masked_color[
                    region_of_interest[1] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[2],
                ] = (
                    color_image[
                        region_of_interest[1] : region_of_interest[3],
                        region_of_interest[0] : region_of_interest[2],
                    ]
                    * mask[..., np.newaxis]
                )

            masks.append(mask)

            # Convert BGR (cv2 default) to RGB (open3d default) in-place
            cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB, dst=masked_color)

            # Create RGBD image from masked data
            rgbd_masked: o3d.geometry.RGBDImage = (
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color=o3d.geometry.Image(masked_color),
                    depth=o3d.geometry.Image(masked_depth),
                    depth_scale=self.descriptor.parameters.depth_scale,
                    depth_trunc=self.descriptor.parameters.depth_truncation,
                    convert_rgb_to_intensity=False,
                )
            )

            pose_in_camera = get_transform_matrix_from_quaternion(
                pose.rotation, pose.translation
            )

            tracked_object = self.find_tracked_object(pose)
            if tracked_object is not None:
                volume = tracked_object["volume"]
                tracked_object["pose"] = pose
            else:
                volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=self.descriptor.parameters.voxel_length,
                    sdf_trunc=self.descriptor.parameters.signed_distance_function_truncation,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                )
                self.tracked_objects.append(
                    {
                        "pose": pose,
                        "volume": volume,
                    }
                )

            # Integrate with pose in camera coordinates
            volume.integrate(
                rgbd_masked,
                intrinsic=camera_intrinsic,
                extrinsic=pose_in_camera,
            )

            volume_an = TruncatedSignedDistanceFunctionAnnotation(source=self.name)
            volume_an.volume = volume
            volume_an.transform = pose_in_camera
            object_hypothesis.annotations.append(volume_an)

            vis_mode = self.descriptor.parameters.geometry_visualization_mode
            if (
                vis_mode
                != TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.NONE
            ):
                visualization_geometries = self.get_visualization_geometries(volume_an)
                geometries.extend(visualization_geometries)

        vis_image = self.get_visualization_image(
            color_image, object_hypothesis_data, masks
        )
        self.get_annotator_output_structure().set_image(vis_image)

        vis_mode = self.descriptor.parameters.geometry_visualization_mode
        if vis_mode != TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.NONE:
            self.robokudo_logger.debug(
                f"Visualizing geometry for {len(geometries) // 3} objects"
            )
            self.get_annotator_output_structure().set_geometries(geometries)

        return Status.SUCCESS

    def get_visualization_image(
        self,
        image: npt.NDArray[np.uint8],
        object_hypothesis_data: List[
            Tuple[ObjectHypothesis, PoseAnnotation, Tuple[int, int, int, int]]
        ],
        masks: List[npt.NDArray[np.uint8]],
    ) -> npt.NDArray[np.uint8]:
        """Get an image containing the bounding boxes and masks of objects.

        :param image: The image to add the visualizations to.
        :param object_hypothesis_data: The object hypotheses to visualize.
        :param masks: The masks to visualize.
        :return: The image with the visualizations.
        """
        self.robokudo_logger.debug(
            f"Visualizing image for {len(object_hypothesis_data)} objects"
        )

        overlay = np.zeros_like(image, dtype=np.uint8)
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 0-1
        for i, (object_hypothesis, _, region_of_interest) in enumerate(
            object_hypothesis_data
        ):
            cv2.rectangle(
                image,
                (region_of_interest[0], region_of_interest[1]),
                (region_of_interest[2], region_of_interest[3]),
                (0, 255, 0),
                2,
            )

            mask = masks[i]  # (H, W)
            if mask.shape[:2] != image.shape[:2]:
                combined_mask[
                    region_of_interest[1] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[2],
                ] = np.maximum(
                    combined_mask[
                        region_of_interest[1] : region_of_interest[3],
                        region_of_interest[0] : region_of_interest[2],
                    ],
                    mask,
                )
            else:
                combined_mask = np.maximum(combined_mask, mask)

        # Set the green channel (index 1 in RGB)
        overlay[:, :, 1] = 255 * combined_mask
        image = cv2.addWeighted(image, 1 - 0.5, overlay.astype(np.uint8), 0.5, 0)

        return image

    def get_visualization_geometries(
        self, volume: TruncatedSignedDistanceFunctionAnnotation
    ) -> List[o3d.geometry.Geometry]:
        """Get visualization geometries for the truncated signed distance function volume depending on the visualization mode.

        :param volume: The truncated signed distance function volume annotation to visualize.
        :return: List of visualization geometries for the given volume annotation.
        """
        vis_mode = self.descriptor.parameters.geometry_visualization_mode
        if (
            vis_mode
            == TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.POINTCLOUD
        ):
            return [volume.get_point_cloud(), volume.get_coordinate_frame()]
        elif (
            vis_mode
            == TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.VOXEL_POINTCLOUD
        ):
            return [volume.get_voxel_point_cloud(), volume.get_coordinate_frame()]
        elif (
            vis_mode == TruncatedSignedDistanceFunctionAnnotatorVisualizationModes.MESH
        ):
            return [volume.get_mesh(), volume.get_coordinate_frame()]
        else:
            self.robokudo_logger.warning(
                f"Visualization mode '{vis_mode}' not supported!"
            )
            return []
