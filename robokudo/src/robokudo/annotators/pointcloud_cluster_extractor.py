"""Point cloud clustering and object detection.

This module provides annotators for:

* Extracting object clusters from point clouds above detected planes
* Generating regions of interest and masks for detected clusters
* Visualizing detected clusters with unique colors

The module uses DBSCAN clustering and oriented bounding boxes (oriented bounding box) for object detection.

.. note::
   The clustering process requires a plane annotation in the common analysis structure to function.
"""

from __future__ import annotations

import copy
import logging
from timeit import default_timer

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from py_trees.common import Status
from scipy.spatial.transform import Rotation as R
from typing_extensions import TYPE_CHECKING, List, Tuple

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.exceptions import (
    EmptyPointCloud,
    PlaneModelMissing,
    PointCloudTooSmallForClustering,
)
from robokudo.types.annotation import Plane
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import draw_bounding_boxes_from_object_hypotheses
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import concatenate_clouds, put_obb_on_target_obb
from robokudo.utils.transform import get_transform_matrix

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio

DILATION_KERNEL = np.ones((3, 3), np.uint8)


def generate_region_of_interest_with_mask_from_points(
    image_height: int,
    image_width: int,
    pointcloud_camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    cloud: o3d.geometry.PointCloud,
    color_to_depth_ratio: ColorToDepthRatio,
) -> Tuple[ImageROI, npt.NDArray]:
    """Generate region of interest and mask from projected point cloud points.

    Projects 3D points to image plane and creates:

    * region of interest bounding the projected points
    * Binary mask of projected points
    * Dilated mask to account for point sparsity

    :param image_height: Height of the target image
    :param image_width: Width of the target image
    :param pointcloud_camera_intrinsics: Camera intrinsic parameters
    :param cloud: Point cloud to project
    :param color_to_depth_ratio: Size of the depth image relative to the color image.
    :return: Tuple of (region of interest with mask, full image mask)
    """
    # TODO: Use robokudo.utils.o3d_helper.get_mask_from_pointcloud

    # Project 3D points to image plane for region of interest generation
    # This will project the matched points into the full-size input image
    k = pointcloud_camera_intrinsics.intrinsic_matrix
    cropped_3d_points = np.asarray(cloud.points)
    uvd = cropped_3d_points @ k.T
    x = (uvd[:, 0] / uvd[:, 2]).astype(int)
    y = (uvd[:, 1] / uvd[:, 2]).astype(int)

    x_scaled, y_scaled = color_to_depth_ratio.color_pixels(np.stack((x, y), axis=1)).T

    full_mask = np.zeros((image_height, image_width), np.uint8)
    full_mask[y_scaled, x_scaled] = 255
    region_of_interest = ImageROI()
    region_of_interest.roi.pos.x = x_scaled.min()
    region_of_interest.roi.pos.y = y_scaled.min()
    region_of_interest.roi.width = x_scaled.max() - x_scaled.min()
    region_of_interest.roi.height = y_scaled.max() - y_scaled.min()

    region_of_interest.mask = full_mask[
        region_of_interest.roi.pos.y : region_of_interest.roi.pos.y
        + region_of_interest.roi.height,
        region_of_interest.roi.pos.x : region_of_interest.roi.pos.x
        + region_of_interest.roi.width,
    ]
    region_of_interest.mask = cv2.dilate(
        region_of_interest.mask, DILATION_KERNEL, iterations=1
    )

    return region_of_interest, full_mask


def cluster_points(
    input_cloud: o3d.geometry.PointCloud,
    epsilon: float,
    min_points: int,
    robokudo_logger: logging.Logger,
) -> List[int]:
    """Cluster points using DBSCAN algorithm.

    :param input_cloud: Point cloud to cluster
    :param epsilon: DBSCAN epsilon parameter (density threshold)
    :param min_points: Minimum points for a cluster
    :param robokudo_logger: Logger instance for progress reporting
    :return: List of point indices for each cluster

    .. note::
       Points labeled as noise (label=-1) are ignored in the output.
    """
    # Only print progress of DBSCAN when in logging level DEBUG
    if robokudo_logger.getEffectiveLevel() < logging.getLevelNamesMapping()["DEBUG"]:
        print_progress = True
    else:
        print_progress = False
    labels = np.array(
        input_cloud.cluster_dbscan(
            eps=epsilon, min_points=min_points, print_progress=print_progress
        )
    )

    max_label = labels.max()
    robokudo_logger.debug(f"point cloud has {max_label + 1} clusters")
    all_cluster_indices = []
    # DBScan returns label==-1 for all points that appear to be noise
    # This will be ignored
    for label in range(0, labels.max() + 1):
        cluster_indices = np.where(labels == label)[0]
        all_cluster_indices.append(cluster_indices)

    return all_cluster_indices


class PointCloudClusterExtractor(ThreadedAnnotator):
    """Annotator for object detection.

    The main idea is to look for the Plane annotation in the common analysis structure extract the points that belong to the plane,
    constructs an oriented bounding box around it, pull it up to create volume above the plane and
    then extract all the inlier points. After that, a clustering process is executed
    and each cluster is annotated in the common analysis structure.

    .. warning::
       Requires a Plane annotation in the common analysis structure to function.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for point cloud clustering."""

        class Parameters:
            """Parameters for configuring cluster extraction."""

            def __init__(self) -> None:
                self.dbscan_min_cluster_count: int = 90
                """Minimum points for DBSCAN cluster"""

                self.min_cluster_count: int = 1000
                """Minimum total points for valid cluster"""

                self.min_on_plane_point_count: int = 90
                """Minimum points above plane"""

                self.dbscan_epsilon: float = 0.04
                """DBSCAN epsilon parameter (density threshold)"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "PointCloudClusterExtractor",
        descriptor: PointCloudClusterExtractor.Descriptor | None = None,
    ) -> None:
        """Initialize the cluster extractor.

        :param name: The name of this annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)

    def points_and_indices_from_plane(
        self,
        plane_model: Plane,
        cloud: o3d.geometry.PointCloud,
    ) -> Tuple[
        o3d.geometry.OrientedBoundingBox,
        o3d.geometry.OrientedBoundingBox,
        o3d.geometry.PointCloud,
        List[int],
        o3d.geometry.PointCloud,
    ]:
        """Extract points above a detected plane.

        Based on an input cloud and a plane model, extract all the points that are above that plane in the input cloud.
        Returns various data structures for further analysis and visualization.

        :param plane_model:
        :param cloud:
        :return: A tuple consisting of:
            plane_oriented_bounding_box = The oriented bounding box around the points belonging to the plane_model in cloud
            on_plane_oriented_bounding_box = The oriented bounding box put above plane_oriented_bounding_box, with the same width and depth, but extended upwards
            on_plane_cloud = Points found in on_plane_oriented_bounding_box
            on_plane_cloud_indices = Indices of Points found in on_plane_oriented_bounding_box
            outlier_cloud = All points that do not belong to the plane/table itself.
        :raises PointCloudTooSmallForClustering: If insufficient points are found above the plane

        .. note::
           The function ensures the plane normal points upward by rotating
           the oriented bounding box if necessary.
        """

        plane_model = plane_model
        plane_cloud = cloud.select_by_index(plane_model.inliers)

        # This cloud will contain everything that doesn't belong to the plane
        # cloud and was in the analysis of the PlaneAnnotation.
        outlier_cloud = cloud.select_by_index(plane_model.inliers, invert=True)
        plane_oriented_bounding_box = plane_cloud.get_oriented_bounding_box()
        assert isinstance(plane_oriented_bounding_box, o3d.geometry.OrientedBoundingBox)

        # Do we need to fix the Plane oriented bounding box orientation?
        # The normal should point 'up' from the detected plane
        plane_oriented_bounding_box_normal_vector = np.matmul(
            np.array([0, 0, 1]), plane_oriented_bounding_box.R
        )  # transform normal vector with plane rotation
        # Dot product between the vector pointing to the plane center from 0,0,0 and the plane oriented bounding box normal
        label = np.dot(
            plane_oriented_bounding_box.center,
            plane_oriented_bounding_box_normal_vector,
        )
        # print(f"Dot product between VP and Plane oriented bounding box Normal: {label}")
        if label > 0:
            # fix the oriented bounding box orientation. It is pointing downwards now
            # Rotate by 180 deg around X
            rotation = R.from_euler("x", np.radians(180)).as_matrix()
            new_oriented_bounding_box_rotation = np.matmul(
                plane_oriented_bounding_box.R, rotation
            )
            plane_oriented_bounding_box.R = new_oriented_bounding_box_rotation

        # Plane oriented bounding box should be good now (normal points up)
        # Begin the actual segmentation

        # Construct oriented bounding box which is used to segment objects on the plane
        # We need to transform it so that it is on top of the found plane
        above_plane_oriented_bounding_box = o3d.geometry.OrientedBoundingBox()
        above_plane_oriented_bounding_box.extent = np.array(
            [
                plane_oriented_bounding_box.extent[0],
                plane_oriented_bounding_box.extent[1],
                0.8,
            ]
        )

        on_plane_oriented_bounding_box = put_obb_on_target_obb(
            above_plane_oriented_bounding_box, plane_oriented_bounding_box
        )
        on_plane_cloud = outlier_cloud.crop(on_plane_oriented_bounding_box)
        on_plane_cloud_indices = (
            on_plane_oriented_bounding_box.get_point_indices_within_bounding_box(
                outlier_cloud.points
            )
        )
        assert isinstance(on_plane_cloud, o3d.geometry.PointCloud)
        on_plane_point_count = len(on_plane_cloud.points)
        if on_plane_point_count < self.descriptor.parameters.min_on_plane_point_count:
            self.feedback_message = (
                f"above plane point cloud has not enough "
                f"points ({on_plane_point_count} "
                f"< {self.descriptor.parameters.min_on_plane_point_count}). Skipping"
            )
            raise PointCloudTooSmallForClustering(
                point_count=on_plane_point_count,
                minimum_point_count=(
                    self.descriptor.parameters.min_on_plane_point_count
                ),
                context="above-plane point cloud clustering",
            )

        return (
            plane_oriented_bounding_box,
            on_plane_oriented_bounding_box,
            on_plane_cloud,
            on_plane_cloud_indices,
            outlier_cloud,
        )

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """Process point cloud to detect and annotate object clusters.

        The method:

        * Extracts points above detected plane
        * Clusters points using DBSCAN
        * Creates ObjectHypothesis annotations for each cluster
        * Generates regions of interest and masks for visualization
        * Visualizes clusters with unique colors

        :return: SUCCESS if clusters are found, FAILURE if no clusters or errors
        :raises PlaneModelMissing: If no plane model exists in common analysis structure
        :raises PointCloudTooSmallForClustering: If insufficient points are found above the plane
        """
        start_timer = default_timer()
        self.robokudo_logger.info("PCE Start")
        cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )
        color_to_depth_ratio = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO
        )
        pointcloud_camera_intrinsics = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.POINTCLOUD_CAMERA_INTRINSIC
        )
        assert isinstance(
            pointcloud_camera_intrinsics, o3d.camera.PinholeCameraIntrinsic
        )

        color = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        height, width, channel_count = color.shape
        visualization_mask = np.zeros((height, width), np.uint8)
        plane_models = self.get_common_analysis_structure().filter_annotations_by_type(
            Plane
        )
        if plane_models is None or plane_models == []:
            self.feedback_message = (
                "No plane model in common analysis structure. Aborting."
            )
            raise PlaneModelMissing(context="point cloud clustering")

        plane_model: Plane = plane_models[0]

        # Segment the pointcloud first. Get the points inside the volume above the given plane and related variables
        (
            plane_oriented_bounding_box,
            on_plane_oriented_bounding_box,
            on_plane_cloud,
            on_plane_cloud_indices,
            outlier_cloud,
        ) = self.points_and_indices_from_plane(plane_model=plane_model, cloud=cloud)

        # Cluster the points above the plane to get a list of point indices for each cluster
        all_cluster_indices = cluster_points(
            input_cloud=on_plane_cloud,
            epsilon=self.descriptor.parameters.dbscan_epsilon,
            min_points=self.descriptor.parameters.dbscan_min_cluster_count,
            robokudo_logger=self.robokudo_logger,
        )

        color_map = plt.get_cmap("tab20")
        clusters = {}
        all_visualization_clouds = []
        object_hypotheses = []
        cluster_index = 0
        for cluster_indices in all_cluster_indices:
            if len(cluster_indices) < self.descriptor.parameters.min_cluster_count:
                self.robokudo_logger.debug(
                    f"Skipping cluster {cluster_index} with {len(cluster_indices)} points"
                )
                continue

            # Work the indices back from our crop towards our original pointcloud
            # This will allow us to reference the full Cloud in the common analysis structure
            # directly, and we don't need to store partial point clouds
            cluster_indices_in_outlier_cloud = np.take(
                on_plane_cloud_indices, cluster_indices
            )
            cluster_cloud = outlier_cloud.select_by_index(
                list(cluster_indices_in_outlier_cloud)
            )

            clusters[cluster_index] = {
                "cloud": cluster_cloud,
                "vis_cloud": copy.deepcopy(cluster_cloud),
            }

            clusters[cluster_index]["vis_cloud"].paint_uniform_color(
                color_map.colors[(cluster_index % 20)]
            )  # limit to 0-19 range due to map size
            all_visualization_clouds.append(clusters[cluster_index]["vis_cloud"])

            self.robokudo_logger.debug(
                f"Extracted cluster {cluster_index} with {len(cluster_cloud.points)} points"
            )

            # Add each cluster to common analysis structure
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.source = self.name
            object_hypothesis.points = cluster_cloud
            object_hypothesis.point_indices = [cluster_indices_in_outlier_cloud]

            object_hypothesis.id = str(cluster_index)
            object_hypothesis.roi, full_mask_cluster = (
                generate_region_of_interest_with_mask_from_points(
                    image_height=height,
                    image_width=width,
                    pointcloud_camera_intrinsics=pointcloud_camera_intrinsics,
                    cloud=cluster_cloud,
                    color_to_depth_ratio=color_to_depth_ratio,
                )
            )
            # Add the image pixel belonging to the cluster to the mask containing all found objects
            visualization_mask[full_mask_cluster == 255] = 255

            self.get_common_analysis_structure().annotations.append(object_hypothesis)
            object_hypotheses.append(object_hypothesis)
            cluster_index += 1

        if len(object_hypotheses) == 0:
            self.robokudo_logger.warning(f"No Clusters have been found.")
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return Status.FAILURE

        # Visualization
        # 3D
        visualization_cloud = concatenate_clouds(all_visualization_clouds)

        plane_normal = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        t = get_transform_matrix(
            plane_oriented_bounding_box.R, plane_oriented_bounding_box.center
        )
        plane_normal.transform(t)

        x = [
            {"name": "Visualization cloud PCE", "geometry": visualization_cloud},
            {
                "name": "Plane oriented bounding box",
                "geometry": plane_oriented_bounding_box,
            },
            {"name": "Plane Normal", "geometry": plane_normal},
            {
                "name": "On Plane oriented bounding box",
                "geometry": on_plane_oriented_bounding_box,
            },
            {"name": "Cloud", "geometry": cloud},
        ]

        self.get_annotator_output_structure().set_geometries(x)

        visualization_mask = cv2.dilate(
            visualization_mask, DILATION_KERNEL, iterations=1
        )
        # make mask a 3 channel mask in order to make it work with addWeighted
        mask3d = np.stack(
            (visualization_mask, visualization_mask, visualization_mask), axis=2
        )

        # Overlay the mask with the input RGB image
        visualization_image = cv2.addWeighted(color, 0.5, mask3d, 0.5, 0)
        draw_bounding_boxes_from_object_hypotheses(
            visualization_image,
            object_hypotheses,
            lambda object_hypothesis: f"ROI-{object_hypothesis.id}({len(object_hypothesis.points.points)})",
        )

        self.get_annotator_output_structure().set_image(visualization_image)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.robokudo_logger.info("PCE Done")
        return Status.SUCCESS


class NaivePointCloudClusterExtractor(ThreadedAnnotator):
    """Annotator for object detection. Simply takes an input cloud and clusters it.

    .. note::
       Unlike PointCloudClusterExtractor, this does not require a plane model.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for naive point cloud clustering."""

        class Parameters:
            """Parameters for configuring cluster extraction."""

            def __init__(self) -> None:
                self.dbscan_min_cluster_count: int = 90
                """Minimum points for DBSCAN cluster"""

                self.min_cluster_count: int = 1000
                """Minimum total points for valid cluster"""

                self.min_on_plane_point_count: int = 90
                """Minimum points above plane"""

                self.dbscan_epsilon: float = 0.04
                """DBSCAN epsilon parameter (density threshold)"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "NaivePointCloudClusterExtractor",
        descriptor: NaivePointCloudClusterExtractor.Descriptor | None = None,
    ) -> None:
        """Initialize the naive cluster extractor.

        :param name: The name of this annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """Process point cloud to detect and annotate object clusters.

        The method:

        * Clusters points directly using DBSCAN
        * Creates ObjectHypothesis annotations for each cluster
        * Generates regions of interest and masks for visualization
        * Visualizes clusters with unique colors

        :return: SUCCESS if clusters found, FAILURE if no clusters or errors
        :raises EmptyPointCloud: If the input cloud is empty
        """
        start_timer = default_timer()
        cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )
        color_to_depth_ratio = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO
        )
        pointcloud_camera_intrinsics = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.POINTCLOUD_CAMERA_INTRINSIC
        )
        assert isinstance(
            pointcloud_camera_intrinsics, o3d.camera.PinholeCameraIntrinsic
        )

        color = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        height, width, channel_count = color.shape
        visualization_mask = np.zeros((height, width), np.uint8)

        if len(cloud.points) == 0:
            self.feedback_message = "Input cloud is empty - Can't compute clustering"
            raise EmptyPointCloud(context="point cloud clustering")

        # Cluster the points above the plane to get a list of point indices for each cluster
        all_cluster_indices = cluster_points(
            input_cloud=cloud,
            epsilon=self.descriptor.parameters.dbscan_epsilon,
            min_points=self.descriptor.parameters.dbscan_min_cluster_count,
            robokudo_logger=self.robokudo_logger,
        )

        color_map = plt.get_cmap("tab20")
        clusters = {}
        all_visualization_clouds = []
        object_hypotheses = []
        cluster_index = 0
        for cluster_indices in all_cluster_indices:
            if len(cluster_indices) < self.descriptor.parameters.min_cluster_count:
                self.robokudo_logger.debug(
                    f"Skipping cluster {cluster_index} with {len(cluster_indices)} points"
                )
                continue

            cluster_cloud = cloud.select_by_index(list(cluster_indices))

            clusters[cluster_index] = {
                "cloud": cluster_cloud,
                "vis_cloud": copy.deepcopy(cluster_cloud),
            }

            clusters[cluster_index]["vis_cloud"].paint_uniform_color(
                color_map.colors[(cluster_index % 20)]
            )  # limit to 0-19 range due to map size
            all_visualization_clouds.append(clusters[cluster_index]["vis_cloud"])

            self.robokudo_logger.debug(
                f"Extracted cluster {cluster_index} with {len(cluster_cloud.points)} points"
            )

            # Add each cluster to common analysis structure
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.source = self.name
            object_hypothesis.points = cluster_cloud
            object_hypothesis.point_indices = [cluster_indices]

            object_hypothesis.id = str(cluster_index)
            object_hypothesis.roi, full_mask_cluster = (
                generate_region_of_interest_with_mask_from_points(
                    image_height=height,
                    image_width=width,
                    pointcloud_camera_intrinsics=pointcloud_camera_intrinsics,
                    cloud=cluster_cloud,
                    color_to_depth_ratio=color_to_depth_ratio,
                )
            )
            # Add the image pixel belonging to the cluster to the mask containing all found objects
            visualization_mask[full_mask_cluster == 255] = 255

            self.get_common_analysis_structure().annotations.append(object_hypothesis)
            object_hypotheses.append(object_hypothesis)
            cluster_index += 1

        if len(object_hypotheses) == 0:
            self.robokudo_logger.warning(f"No Clusters have been found.")
            end_timer = default_timer()
            self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
            return Status.FAILURE

        # Visualization
        # 3D
        visualization_cloud = concatenate_clouds(all_visualization_clouds)

        x = [
            {"name": "Visualization cloud PCE", "geometry": visualization_cloud},
        ]

        self.get_annotator_output_structure().set_geometries(x)

        visualization_mask = cv2.dilate(
            visualization_mask, DILATION_KERNEL, iterations=1
        )
        # make mask a 3-channel mask in order to make it work with addWeighted
        mask3d = np.stack(
            (visualization_mask, visualization_mask, visualization_mask), axis=2
        )

        # Overlay the mask with the input RGB image
        visualization_image = cv2.addWeighted(color, 0.5, mask3d, 0.5, 0)
        draw_bounding_boxes_from_object_hypotheses(
            visualization_image,
            object_hypotheses,
            lambda object_hypothesis: f"ROI-{object_hypothesis.id}({len(object_hypothesis.points.points)})",
        )

        self.get_annotator_output_structure().set_image(visualization_image)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        self.robokudo_logger.info("PCE Stop")
        return Status.SUCCESS
