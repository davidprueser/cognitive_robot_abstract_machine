"""
3D position estimation for object hypotheses.

This module provides an annotator for:

* Calculating 3D positions for object hypotheses
* Supporting different analysis scopes
* Computing centroids from point clouds
* Generating visualization markers

The module uses:

* Point cloud centroid computation
* Covariance analysis
* Open3D visualization tools
* Flexible annotation types

.. note::
   Can analyze either ObjectHypothesis or CloudAnnotation data.
"""

from __future__ import annotations

from dataclasses import dataclass


import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.types import annotation, scene


@dataclass(kw_only=True)
class ClusterPositionAnnotator(BaseAnnotator):
    """
    3D position estimation for object hypotheses.

    This annotator:

    * Calculates 3D positions from point clouds
    * Supports multiple analysis scopes
    * Computes centroids and covariance
    * Creates position annotations
    * Generates visualization markers

    .. note::
       Can process either ObjectHypothesis or CloudAnnotation data.
    """

    analysis_scope: type = scene.ObjectHypothesis
    """
    Type of data to perform position estimation on(e.g. ObjectHypothesis or
    CloudAnnotation)
    """

    visualizer_point_radius: float = 0.04
    """
    Radius of centroid sphere markers in meters.
    """

    def compute(self) -> Status:
        """
        Process object hypotheses and estimate positions.

        The method:

        * Loads point cloud from CAS
        * For each object hypothesis:
          * Gets appropriate point cloud data
          * Computes centroid and covariance
          * Creates position annotation
          * Creates visualization marker

        :return: SUCCESS after processing
        """
        cloud = self.get_cas().get(CASViews.CLOUD)
        centroids_to_visualize: list[list[float]] = []

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            scene.ObjectHypothesis
        )

        for object_hypothesis in object_hypotheses:
            if object_hypothesis.points is None:
                continue

            if self.analysis_scope == annotation.CloudAnnotation:
                o_clouds: list[annotation.CloudAnnotation] = CAS.filter_by_type(
                    annotation.CloudAnnotation,
                    object_hypothesis.annotations,
                )
                if len(o_clouds) == 0:
                    self.robokudo_logger.warning(
                        "CloudAnnotation mode, but no CloudAnnotation found on object"
                    )
                    continue

                cluster_cloud = o_clouds[0].points
            else:
                cluster_cloud = object_hypothesis.points
            centroid, covariance = cluster_cloud.compute_mean_and_covariance()

            position = self.position_annotation_from_centroid(centroid)
            object_hypothesis.annotations.append(position)

            self.add_centroid_to_vis(centroid, centroids_to_visualize)

        # Visualization
        vis_geometries = [
            {"name": "Cloud", "geometry": cloud},
        ]
        vis_geometries.extend(centroids_to_visualize)

        self.get_annotator_output_struct().set_geometries(vis_geometries)

        return Status.SUCCESS

    def position_annotation_from_centroid(
        self, centroid: list[float]
    ) -> annotation.PositionAnnotation:
        """
        Create position annotation from centroid.

        :param centroid: 3D centroid coordinates
        :return: Position annotation with centroid as translation
        """
        position = annotation.PositionAnnotation()
        position.source = type(self).__name__
        position.translation = centroid
        return position

    def add_centroid_to_vis(
        self, centroid: list[float], centroids_to_visualize: list[list[float]]
    ) -> None:
        """
        Add centroid visualization marker.

        Creates a colored sphere at the centroid position.

        :param centroid: 3D centroid coordinates
        :param centroids_to_visualize: List to append visualization marker to
        """
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.visualizer_point_radius
        )  # in meters
        centroid_sphere.paint_uniform_color([255, 0, 0])
        centroid_sphere.translate(centroid)
        centroids_to_visualize.append(centroid_sphere)
