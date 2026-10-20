"""Statistical outlier removal and clustering for object hypotheses.

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

from timeit import default_timer

import numpy as np
import open3d as o3d
from py_trees.common import Status
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from typing_extensions import Dict, List, Type

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import (
    CommonAnalysisStructure,
    CommonAnalysisStructureViews,
)
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry

"""
This module implements a statistical outlierremoval based on the standard deviation and
number of neighbors in a point cloud. Afterwards a clustering is applied, to get for each 
object-hypothesis the actual object. This implementation works in place, such that there is no 
remaining copy of the old state for some point cloud. 

Authors: Sorin Arion, Naser Azizi 
"""

_PYTHON_TYPE_TO_ROS_FIELD: Dict[Type, str] = {
    int: "integer_value",
    float: "double_value",
    str: "string_value",
    bool: "bool_value",
    list: "string_array_value",  # rclpy supports string arrays only
}


class OutlierRemovalOnObjectHypothesisAnnotator(BaseAnnotator):
    """Annotator for statistical outlier removal and clustering refinement.

    This annotator processes object hypotheses by:

    * Removing statistical outliers from point clouds
    * Clustering remaining points using DBSCAN
    * Selecting largest cluster as refined object
    * Updating object hypothesis point clouds in-place
    * Providing visualization of processed clusters

    The processing can be selectively disabled for specific object classes.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for outlier removal and clustering."""

        class Parameters:
            """Parameter container for outlier removal configuration."""

            def __init__(self) -> None:
                # TODO Rename the dbscan_neighbors parameter. It is not about neighbors, but it is the MINIMUM
                self.dbscan_neighbors: int = 90
                """Minimum points to form a cluster (DBSCAN min_samples)"""

                self.dbscan_epsilon: float = 0.02
                """DBSCAN neighborhood size"""

                self.statistical_neighbor_count: int = 200
                """Number of neighbors for statistical analysis"""

                self.statistical_standard_deviation_ratio: float = 0.5
                """Standard deviation threshold for outlier removal"""

                self.skip_removal_on_classes: List[str] = []
                """List of class names to skip processing
                
                If you want to skip the complete outlier removal process on certain classes, you can add a list of
                strings here. The values are case-sensitive!
                """

        parameters = Parameters()

    def __init__(
        self,
        name: str = "OutlierRemovalOnObjectHypothesis",
        descriptor: OutlierRemovalOnObjectHypothesisAnnotator.Descriptor | None = None,
    ) -> None:
        """Initialize the outlier removal annotator.

        :param name: Name of this annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.node = ROSNodeRegistry().get()

        for parameter_name, default_value in vars(self.descriptor.parameters).items():
            self.node.declare_parameter(f"{self.name}/{parameter_name}", default_value)

        self.node.add_on_set_parameters_callback(self.parameters_callback)

        self.robokudo_logger.info(
            "OutlierRemovalNode initialized with current parameters"
        )

    def parameters_callback(self, parameters: List[Parameter]) -> SetParametersResult:
        for parameter in parameters:
            if hasattr(self.descriptor.parameters, parameter.name):
                setattr(self.descriptor.parameters, parameter.name, parameter.value)

        self.robokudo_logger.info("Received reconf call: " + str(parameters))
        return SetParametersResult(successful=True)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Process object hypotheses to remove outliers and refine clusters.

        :return: SUCCESS if processing completed, raises Exception if no clusters found
        :raises Exception: If no clusters are found after processing
        """

        start_timer = default_timer()
        point_cloud_cluster = self.cluster_statistical_outlier_removal_point_cloud()

        if not point_cloud_cluster:
            # TODO This should not be decided by this Annotator, but by the Result Generator - Refactor and test analysis engines
            self.robokudo_logger.warning(f"No Clusters have been found.")
            self.feedback_message = f"No clusters have been found"
            raise Exception("No Clusters have been found.")
        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def cluster_statistical_outlier_removal_point_cloud(self) -> bool:
        """Perform outlier removal and clustering on each object hypothesis.

        :return: True, if at least one of the object hypotheses could be optimized. False otherwise.
        """
        annotations = self.get_common_analysis_structure().filter_annotations_by_type(
            ObjectHypothesis
        )
        visualization_geometries = []

        optimized_one_cluster = False

        for annotation in annotations:
            if annotation.points is None:
                continue

            if len(self.descriptor.parameters.skip_removal_on_classes) > 0:
                classes: List[Classification] = CommonAnalysisStructure.filter_by_type(
                    Classification, annotation.annotations
                )

                found_class = False
                for classification in classes:
                    if (
                        classification.classname
                        in self.descriptor.parameters.skip_removal_on_classes
                    ):
                        found_class = True
                        optimized_one_cluster = True  # this cluster could probably been optimized, but is ignored by config
                        continue
                if found_class:
                    continue

            point_cloud = annotation.points
            inlier_cloud, inlier_indices = point_cloud.remove_statistical_outlier(
                nb_neighbors=self.descriptor.parameters.statistical_neighbor_count,
                std_ratio=self.descriptor.parameters.statistical_standard_deviation_ratio,
            )

            point_cloud = point_cloud.select_by_index(inlier_indices)

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as verbosity_context:
                labels = np.array(
                    point_cloud.cluster_dbscan(
                        eps=self.descriptor.parameters.dbscan_epsilon,
                        min_points=self.descriptor.parameters.dbscan_neighbors,
                        print_progress=True,
                    )
                )
            # We pick the biggest cluster, assuming that its point cloud represents
            # an actual object and not noise
            try:
                cluster_sizes = []
                max_label = labels.max()
                for label in range(0, max_label + 1):
                    cluster_sizes.append(np.where(labels == label)[0].shape[0])
                best_cluster = np.argmax(np.asarray(cluster_sizes))

            except ValueError as exception:
                # We couldn't optimize THIS object hypothesis, but maybe there are other ones that already have been
                # optimized in previous iterations or we will find optimizables ones next
                continue
                # return False

            optimized_one_cluster = True

            cluster_indices = np.where(labels == best_cluster)[0]
            clustered_point_cloud = point_cloud.select_by_index(cluster_indices)
            # Replace old state with clustered point cloud
            annotation.points = clustered_point_cloud
            visualization_geometries.append(clustered_point_cloud)

        if not optimized_one_cluster:
            return False

        visualization_image = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        self.get_annotator_output_structure().set_image(visualization_image)
        self.get_annotator_output_structure().set_geometries(visualization_geometries)

        return True
