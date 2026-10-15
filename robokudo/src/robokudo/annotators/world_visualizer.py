"""
World visualization for RoboKudo.

This annotator visualizes predefined objects and regions from a world descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np
import open3d as o3d
import py_trees
from semantic_digital_twin.world_description.world_entity import Body, Region

import robokudo.annotators.core
import robokudo.utils.annotator_helper
import robokudo.utils.error_handling
import robokudo.utils.o3d_helper
import robokudo.world as rk_world
from robokudo.cas import CASViews
from robokudo.utils.region import region_obb_in_camera_coordinates
from robokudo.world_descriptor import PredefinedObject


@dataclass(kw_only=True)
class WorldVisualizer(robokudo.annotators.core.ThreadedAnnotator):
    """
    Visualize predefined objects and regions from the shared world.
    """

    visualize_objects: bool = True

    visualize_regions: bool = True

    @staticmethod
    def _body_size(body: Body) -> np.ndarray | None:
        shape_collection = (
            body.collision
            if body.collision is not None and len(body.collision) > 0
            else body.visual
        )
        if shape_collection is None or len(shape_collection) == 0:
            return None
        bb = shape_collection.as_bounding_box_collection_in_frame(body).bounding_box()
        return np.array([bb.scale.x, bb.scale.y, bb.scale.z], dtype=float)

    def compute(self) -> py_trees.common.Status:
        cloud = self.get_cas().get(CASViews.CLOUD)
        runtime_world = rk_world.world_instance()

        try:
            world_to_camera_transform = (
                robokudo.utils.annotator_helper.get_world_to_camera_transform_matrix(
                    self.get_cas()
                )
            )
        except KeyError as err:
            self.robokudo_logger.warning(f"Couldn't find viewpoint in the CAS: {err}")
            return py_trees.common.Status.FAILURE

        visualized_geometries = []

        if self.visualize_objects:
            predefined_objects = runtime_world.get_semantic_annotations_by_type(
                PredefinedObject
            )
            for predefined_object in predefined_objects:
                body = predefined_object.body
                if body is None:
                    continue
                size = self._body_size(body)
                if size is None:
                    continue
                world_T_body = body.global_pose.to_np()
                camera_T_body = world_to_camera_transform @ world_T_body
                obb = robokudo.utils.o3d_helper.get_obb_from_size_and_transform(
                    size, camera_T_body
                )
                visualized_geometries.append({"name": str(body.name), "geometry": obb})

        if self.visualize_regions:
            regions = runtime_world.get_kinematic_structure_entity_by_type(Region)
            for region in regions:
                obb = region_obb_in_camera_coordinates(
                    runtime_world, region, world_to_camera_transform
                )
                visualized_geometries.append(
                    {"name": str(region.name), "geometry": obb}
                )

        visualized_geometries.append({"name": "cloud", "geometry": cloud})

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        visualized_geometries.append(
            {
                "name": "world_frame",
                "geometry": world_frame.transform(world_to_camera_transform),
            }
        )

        self.get_annotator_output_struct().set_geometries(visualized_geometries)

        return py_trees.common.Status.SUCCESS
