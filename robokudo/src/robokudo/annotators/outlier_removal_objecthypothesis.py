"""
Statistical outlier removal and clustering for object hypotheses.

This module implements statistical outlier removal based on standard deviation and
number of neighbors in point clouds, followed by clustering to refine object hypotheses.
The implementation works in-place to minimize memory usage.

Key features:

* Statistical outlier removal using point neighborhoods
* DBSCAN clustering for object refinement
* In-place point cloud modification
* Configurable parameters per object class
* Visualization of processed clusters

Authors:
* Sorin Arion
* Naser Azizi
"""

from __future__ import annotations

from dataclasses import dataclass, field


import numpy as np
import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

"""
This module implements a statistical outlierremoval based on the standard deviation and
number of neighbors in a point cloud. Afterwards a clustering is applied, to get for each 
object-hypothesis the actual object. This implementation works in place, such that there is no 
remaining copy of the old state for some point cloud. 

Authors: Sorin Arion, Naser Azizi 
"""

_PYTYPE_TO_ROS_FIELD: dict[type, str] = {
    int: "integer_value",
    float: "double_value",
    str: "string_value",
    bool: "bool_value",
    list: "string_array_value",  # rclpy supports string arrays only
}


@dataclass(kw_only=True)
class OutlierRemovalOnObjectHypothesisAnnotator(BaseAnnotator):
    """
    Annotator for statistical outlier removal and clustering refinement.

    This annotator processes object hypotheses by:

    * Removing statistical outliers from point clouds
    * Clustering remaining points using DBSCAN
    * Selecting largest cluster as refined object
    * Updating object hypothesis point clouds in-place
    * Providing visualization of processed clusters

    The processing can be selectively disabled for specific object classes.
    """

    dbscan_neighbors: int = 90
    """
    Minimum points to form a cluster (DBSCAN min_samples)
    """

    dbscan_epsilon: float = 0.02
    """
    DBSCAN neighborhood size.
    """

    stat_neighbors: int = 200
    """
    Number of neighbors for statistical analysis.
    """

    stat_std: float = 0.5
    """
    Standard deviation threshold for outlier removal.
    """

    skip_removal_on_classes: list[str] = field(default_factory=list)
    """
    List of class names to skip processing.

    If you want to skip the complete outlier removal process on certain classes, you can
    add a list of strings here. The values are case-sensitive!
    """

    def compute(self) -> Status:
        """
        Process object hypotheses to remove outliers and refine clusters.

        :return: SUCCESS if processing completed, raises Exception if no clusters found
        :raises Exception: If no clusters are found after processing
        """
        pcd_cluster = self.cluster_statistical_outlierremoval_pcd()

        if not pcd_cluster:
            # TODO This should not be decided by this Annotator, but by the Result Generator - Refactor and test AEs
            self.robokudo_logger.warning(f"No Clusters have been found.")
            self.feedback_message = f"No clusters have been found"
            raise Exception("No Clusters have been found.")
        return Status.SUCCESS

    def cluster_statistical_outlierremoval_pcd(self) -> bool:
        """
        Perform outlier removal and clustering on each object hypothesis.

        :return: True, if at least one of the object hypotheses could be optimized.
            False otherwise.
        """
        annotations = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
        vis_geometries = []

        optimized_one_cluster = False

        for annotation in annotations:
            if annotation.points is None:
                continue

            if len(self.skip_removal_on_classes) > 0:
                classes: list[Classification] = CAS.filter_by_type(
                    Classification, annotation.annotations
                )

                found_class = False
                for c in classes:
                    if c.classname in self.skip_removal_on_classes:
                        found_class = True
                        optimized_one_cluster = True  # this cluster could probably been optimized, but is ignored by config
                        continue
                if found_class:
                    continue

            pcd = annotation.points
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=self.stat_neighbors,
                std_ratio=self.stat_std,
            )

            pcd = pcd.select_by_index(ind)

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                labels = np.array(
                    pcd.cluster_dbscan(
                        eps=self.dbscan_epsilon,
                        min_points=self.dbscan_neighbors,
                        print_progress=True,
                    )
                )
            # We pick the biggest cluster, assuming that its point cloud represents
            # an actual object and not noise
            try:
                cluster_sizes = []
                max_label = labels.max()
                for val in range(0, max_label + 1):
                    cluster_sizes.append(np.where(labels == val)[0].shape[0])
                best_cluster = np.argmax(np.asarray(cluster_sizes))

            except ValueError as e:
                # We couldn't optimize THIS object hypothesis, but maybe there are other ones that already have been
                # optimized in previous iterations or we will find optimizables ones next
                continue
                # return False

            optimized_one_cluster = True

            cluster_indices = np.where(labels == best_cluster)[0]
            clustered_pcd = pcd.select_by_index(cluster_indices)
            # Replace old state with clustered point cloud
            annotation.points = clustered_pcd
            vis_geometries.append(clustered_pcd)

        if not optimized_one_cluster:
            return False

        visualization_img = self.get_cas().get(CASViews.COLOR_IMAGE)
        self.get_annotator_output_struct().set_image(visualization_img)
        self.get_annotator_output_struct().set_geometries(vis_geometries)

        return True
