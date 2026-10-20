"""
Object pose visualization for RoboKudo.

This module provides an annotator for visualizing the poses of detected objects in both
2D (image overlays) and 3D (coordinate frames) representations.
"""

from __future__ import annotations

from timeit import default_timer

import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.transform import get_transform_matrix_from_q


class ObjectPoseVisualizer(BaseAnnotator):
    """
    Annotator for visualizing object poses in the common analysis structure.

    This annotator creates visualizations of object poses by:

    * Drawing pose information on the color image
    * Creating 3D coordinate frames at each object's pose
    * Displaying the coordinate frames alongside the point cloud
    """

    def __init__(self, name: str = "ObjectPoseVisualizer") -> None:
        """
        Initialize the object pose visualizer.

        :param name: Name of the annotator instance
        """
        super().__init__(name)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """
        Update the visualization with current object poses.

        Creates visualizations containing:

        * Color image with pose information
        * 3D coordinate frames for each object's pose
        * Point cloud data

        :return: SUCCESS after creating visualizations
        """
        start_timer = default_timer()

        visualization_image = self.get_common_analysis_structure().get_copy(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )

        object_hypotheses = (
            self.get_common_analysis_structure().filter_annotations_by_type(
                ObjectHypothesis
            )
        )

        geometries_to_visualize = []
        object_id = 0
        for object_hypothesis in object_hypotheses:
            for object_hypothesis_annotation in object_hypothesis.annotations:
                if isinstance(object_hypothesis_annotation, PoseAnnotation):
                    cluster_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.2
                    )
                    pose_transform = get_transform_matrix_from_q(
                        np.asarray(object_hypothesis_annotation.rotation),
                        np.asarray(object_hypothesis_annotation.translation),
                    )
                    cluster_frame.transform(pose_transform)

                    geometries_to_visualize.append(
                        {"name": f"Pose-Obj-{object_id}", "geometry": cluster_frame}
                    )

            object_id += 1

        self.get_annotator_output_structure().set_image(visualization_image)

        visualization_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        visualization_geometries.extend(geometries_to_visualize)
        self.get_annotator_output_structure().set_geometries(visualization_geometries)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
