"""
Tests for finding the points that stand on the plane a cluster extractor clusters above.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

# robokudo.pipeline must be imported before robokudo.annotators.outputs: importing
# outputs first trips a circular import between it and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.types.annotation import Plane

# %% points on the plane


class TestPointsOnThePlane:
    def test_object_on_a_layer_seen_straight_ahead_stands_on_the_plane(
        self,
    ) -> None:
        # In camera coordinates, x right, y down and z forward: a shelf layer below a
        # camera looking straight ahead, and a box standing on it.
        random_state = np.random.default_rng(0)
        layer_points = np.column_stack(
            [
                random_state.uniform(-0.4, 0.4, 4000),
                0.85 + random_state.normal(0.0, 0.005, 4000),
                random_state.uniform(1.15, 1.45, 4000),
            ]
        )
        object_points = np.column_stack(
            [
                random_state.uniform(0.0, 0.07, 300),
                random_state.uniform(0.65, 0.83, 300),
                random_state.uniform(1.25, 1.32, 300),
            ]
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.vstack([layer_points, object_points])
        )
        layer = Plane(inliers=list(range(len(layer_points))))

        (
            _,
            _,
            on_plane_cloud,
            _,
            _,
        ) = PointCloudClusterExtractor().points_and_indices_from_plane(
            plane_model=layer, cloud=cloud
        )

        assert len(on_plane_cloud.points) == len(object_points)
