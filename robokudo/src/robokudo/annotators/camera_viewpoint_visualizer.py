"""
Camera viewpoint visualization for RoboKudo.

This module provides an annotator for visualizing camera viewpoints and reference frames
in 3D space using Open3D.
"""

import open3d as o3d
from py_trees.common import Status

import robokudo.utils.annotator_helper
from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews


class CameraViewpointVisualizer(BaseAnnotator):
    """
    Annotator for visualizing camera viewpoints and reference frames.

    This annotator displays the reference frame set in the viewpoint (e.g., /map frame)
    along with the point cloud data. It creates a 3D visualization showing the
    coordinate frame and the point cloud in the same space.

    The annotator will fail if the required viewpoint transform cannot be found in the
    common analysis structure.
    """

    def __init__(self, name: str = "CameraViewpointVisualizer") -> None:
        """
        Initialize the camera viewpoint visualizer.

        :param name: Name of the annotator instance
        """
        super().__init__(name)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)

    def update(self) -> Status:
        """
        Update the visualization with the current viewpoint and point cloud data.

        Creates a visualization containing:

        * A coordinate frame representing the world reference frame
        * The current point cloud data

        :return: SUCCESS if visualization was created, FAILURE if camera to world transform not found
        :raises AssertionError: If viewpoint transform is of a wrong type
        """
        try:
            world_to_camera_transform = (
                robokudo.utils.annotator_helper.get_world_to_camera_transform_matrix(
                    self.get_common_analysis_structure()
                )
            )
        except KeyError as error:
            self.robokudo_logger.warning(
                f"Couldn't find viewpoint in the common analysis structure: {error}"
            )
            return Status.FAILURE

        cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        world_frame.transform(world_to_camera_transform)

        geometries = [
            {"name": "World frame", "geometry": world_frame},
            {"name": "Cloud", "geometry": cloud},
        ]
        self.get_annotator_output_structure().set_geometries(geometries)

        return Status.SUCCESS
