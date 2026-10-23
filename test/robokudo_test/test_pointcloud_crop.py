"""
Tests for cropping a point cloud relative to the world frame.

``PointcloudCropAnnotator``'s own docstring documents two supported modes: cropping in
sensor coordinates (the default) or relative to the world frame. The world-relative mode
transforms the cloud into world coordinates, crops it there, then transforms the result
back into camera coordinates -- but an early, unconditional ``return Status.FAILURE``
made every step after the transform unreachable, so the mode never actually cropped
anything even when the camera-to-world transform was available.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest
from py_trees.blackboard import Blackboard
from py_trees.common import Status

# robokudo.pipeline must be imported before robokudo.annotators.outputs: importing
# outputs first trips a circular import between it and robokudo.annotators.core.
from robokudo.types.cv import ColorToDepthRatio
import robokudo.pipeline
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap, AnnotatorOutputs
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.common_analysis_structure import (
    CommonAnalysisStructure,
    CommonAnalysisStructureViews,
)
from robokudo.pipeline import Pipeline
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@pytest.fixture()
def camera_intrinsics() -> o3d.camera.PinholeCameraIntrinsic:
    """
    Intrinsics sized so the test cloud's points project well inside the color image.
    """
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=200, height=200, fx=100.0, fy=100.0, cx=100.0, cy=100.0
    )
    return intrinsics


@pytest.fixture()
def common_analysis_structure_with_cloud_at_world_origin(
    camera_intrinsics: o3d.camera.PinholeCameraIntrinsic,
) -> CommonAnalysisStructure:
    """
    A common analysis structure whose cloud sits at the world origin, one metre in front
    of the camera, with an identity camera-to-world transform (so world and camera
    coordinates coincide here, keeping the fixture's geometry simple).
    """
    common_analysis_structure = CommonAnalysisStructure()
    common_analysis_structure.camera_to_world_transform = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.0,
            pos_y=0.0,
            pos_z=0.0,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
    )

    points_inside_the_crop = np.array([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]])
    points_outside_the_crop = np.array([[5.0, 5.0, 1.0], [-5.0, -5.0, 1.0]])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.vstack([points_inside_the_crop, points_outside_the_crop])
    )

    common_analysis_structure.set_reference(CommonAnalysisStructureViews.CLOUD, cloud)
    common_analysis_structure.set(
        CommonAnalysisStructureViews.COLOR_IMAGE,
        np.zeros((200, 200, 3), dtype=np.uint8),
    )
    common_analysis_structure.set(
        CommonAnalysisStructureViews.POINTCLOUD_CAMERA_INTRINSIC, camera_intrinsics
    )
    common_analysis_structure.set(
        CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio()
    )
    return common_analysis_structure


@pytest.fixture()
def crop_annotator_in_pipeline(
    common_analysis_structure_with_cloud_at_world_origin: CommonAnalysisStructure,
) -> PointcloudCropAnnotator:
    """
    A ``PointcloudCropAnnotator`` configured to crop relative to the world frame, wired
    up with just enough pipeline/blackboard state for ``update()`` to run.
    """
    descriptor = PointcloudCropAnnotator.Descriptor()
    descriptor.parameters.relative_to_world = True
    descriptor.parameters.min_x = -1.0
    descriptor.parameters.max_x = 1.0
    descriptor.parameters.min_y = -1.0
    descriptor.parameters.max_y = 1.0
    descriptor.parameters.min_z = 0.0
    descriptor.parameters.max_z = 2.0

    pipeline = Pipeline("TestPipeline")
    pipeline.common_analysis_structure = (
        common_analysis_structure_with_cloud_at_world_origin
    )
    annotator = PointcloudCropAnnotator(descriptor=descriptor)
    pipeline.add_child(annotator)

    output_map = AnnotatorOutputPerPipelineMap()
    output_map.map[pipeline.name] = AnnotatorOutputs()
    Blackboard().set("annotator_output_pipeline_map_buffer", output_map)

    return annotator


def test_world_relative_crop_succeeds_when_the_transform_is_available(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
) -> None:
    status = crop_annotator_in_pipeline.update()

    assert status == Status.SUCCESS


def test_world_relative_crop_actually_crops_the_cloud(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
) -> None:
    crop_annotator_in_pipeline.update()

    cropped_cloud = crop_annotator_in_pipeline.get_common_analysis_structure().get(
        CommonAnalysisStructureViews.CLOUD
    )

    assert len(cropped_cloud.points) == 2


def test_world_relative_crop_fails_without_a_camera_to_world_transform(
    crop_annotator_in_pipeline: PointcloudCropAnnotator,
) -> None:
    crop_annotator_in_pipeline.get_common_analysis_structure().camera_to_world_transform = (
        None
    )

    status = crop_annotator_in_pipeline.update()

    assert status == Status.FAILURE
