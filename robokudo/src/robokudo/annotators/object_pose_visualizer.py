"""
Object pose visualization for RoboKudo.

This module provides an annotator for visualizing the poses of detected objects in both
2D (image overlays) and 3D (coordinate frames) representations.
"""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_transform_matrix_from_q


@dataclass(kw_only=True)
class ObjectPoseVisualizer(BaseAnnotator):
    """
    Annotator for visualizing object poses in the CAS.

    This annotator creates visualizations of object poses by:

    * Drawing pose information on the color image
    * Creating 3D coordinate frames at each object's pose
    * Displaying the coordinate frames alongside the point cloud
    """

    def compute(self) -> Status:
        """
        Update the visualization with current object poses.

        Creates visualizations containing:

        * Color image with pose information
        * 3D coordinate frames for each object's pose
        * Point cloud data

        :return: SUCCESS after creating visualizations
        """
        visualization_img = self.get_cas().get_copy(CASViews.COLOR_IMAGE)
        cloud = self.get_cas().get(CASViews.CLOUD)

        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        geometries_to_visualize = []
        object_id = 0
        for oh in object_hypotheses:
            for oh_anno in oh.annotations:
                if isinstance(oh_anno, PoseAnnotation):
                    cluster_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.2
                    )
                    pose_transform = get_transform_matrix_from_q(
                        np.asarray(oh_anno.rotation),
                        np.asarray(oh_anno.translation),
                    )
                    cluster_frame.transform(pose_transform)

                    geometries_to_visualize.append(
                        {"name": f"Pose-Obj-{object_id}", "geometry": cluster_frame}
                    )

            object_id += 1

        self.get_annotator_output_struct().set_image(visualization_img)

        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(geometries_to_visualize)
        self.get_annotator_output_struct().set_geometries(vis_geometries)

        return Status.SUCCESS
