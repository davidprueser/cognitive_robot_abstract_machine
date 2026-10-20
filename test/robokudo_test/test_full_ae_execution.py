from __future__ import annotations
from pathlib import Path

import numpy as np
import py_trees
import pytest

from robokudo.types.cv import ColorToDepthRatio
from robokudo import world as rk_world
import robokudo.common_analysis_structure
import robokudo.defs
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.descriptors.camera_configs.config_semdt_raytracer
import robokudo.io.file_reader_interface
import robokudo.io.semdt_raytracer_camera_interface
import robokudo.io.storage_reader_interface
import robokudo.behaviours.clear_annotator_outputs
import robokudo.pipeline
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.data_downloader
import robokudo.utils.tree_execution
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.annotators.shape_estimator import ShapeEstimatorAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.world_descriptor import PredefinedObject
from semantic_digital_twin.world_description.geometry import Mesh
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from robokudo.common_analysis_structure import CommonAnalysisStructure
    from typing_extensions import Iterator
    from rclpy.node import Node


@pytest.fixture
def clean_semantic_world() -> Iterator[None]:
    rk_world.get_object_belief_states().clear()
    rk_world.init_world_with_entity_tracker()
    yield
    rk_world.get_object_belief_states().clear()
    rk_world.init_world_with_entity_tracker()


class TestFullAEExecution(object):
    def test_run_simple_ae_successfully(self, node: Node) -> None:
        # rclpy.init()
        # node = Node(robokudo.defs.TEST_ROS_NODE_NAME)

        cr_fr_config = CollectionReaderDescriptorFactory.create_descriptor(
            "file_reader",
            loop=False,
            target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
            kinect_height_fix_mode=True,
            color_to_depth_ratio=ColorToDepthRatio(horizontal=0.5, vertical=0.5),
            static_camera_transform_enabled=True,
            static_world_frame="map",
            static_camera_frame="camera",
        )

        # Restrict FOV of pointcloud to robustly get only one object
        pc_crop_config = (
            robokudo.annotators.pointcloud_crop.PointcloudCropAnnotator.Descriptor()
        )
        pc_crop_config.parameters.min_x = -0.3
        pc_crop_config.parameters.max_x = 0.3

        seq = robokudo.pipeline.Pipeline("TestPipeline")
        seq.add_children(
            [
                robokudo.behaviours.clear_annotator_outputs.ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=cr_fr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=pc_crop_config),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )

        tree_result = robokudo.utils.tree_execution.run_tree_once(seq, node)
        assert tree_result is py_trees.common.Status.SUCCESS

        # We should have found a plane and an object
        assert len(seq.common_analysis_structure.annotations) > 0
        types_of_annotations = list(
            map(type, seq.common_analysis_structure.annotations)
        )

        assert types_of_annotations.count(robokudo.types.annotation.Plane) == 1
        assert types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 1

    def test_run_file_reader_ae_synchronizes_semantic_digital_twin_belief_state(
        self, node: Node, clean_semantic_world: None
    ) -> None:
        cr_fr_config = CollectionReaderDescriptorFactory.create_descriptor(
            "file_reader",
            loop=False,
            target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
            kinect_height_fix_mode=True,
            color_to_depth_ratio=ColorToDepthRatio(horizontal=0.5, vertical=0.5),
            static_camera_transform_enabled=True,
            static_world_frame="map",
            static_camera_frame="camera",
        )

        pc_crop_config = (
            robokudo.annotators.pointcloud_crop.PointcloudCropAnnotator.Descriptor()
        )
        pc_crop_config.parameters.min_x = -0.3
        pc_crop_config.parameters.max_x = 0.3

        seq = robokudo.pipeline.Pipeline("SemanticDigitalTwinFileReaderTestPipeline")
        seq.add_children(
            [
                robokudo.behaviours.clear_annotator_outputs.ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=cr_fr_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=pc_crop_config),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
                SemanticDigitalTwinConnector(),
            ]
        )

        belief_uuids_after_each_successful_common_analysis_structure = []

        def assert_current_common_analysis_structure_updates_single_belief(
            common_analysis_structure: CommonAnalysisStructure,
        ) -> None:
            object_hypotheses = common_analysis_structure.filter_annotations_by_type(
                robokudo.types.scene.ObjectHypothesis
            )
            assert (
                len(object_hypotheses) == 1
            ), "Each successful common analysis structure should already contain the object hypothesis to be synchronized."

            object_beliefs = list(rk_world.get_object_belief_states().values())
            assert (
                len(object_beliefs) == 1
            ), "After each successful connector run, the current object should map to exactly one belief."

            object_belief = object_beliefs[0]
            assert (
                object_belief.latest_hypothesis is object_hypotheses[0]
            ), "The belief observed at common analysis structure success should point to that common analysis structure's object hypothesis."
            belief_uuids_after_each_successful_common_analysis_structure.append(
                object_belief.uuid
            )

        successful_common_analysis_structure_instances = robokudo.utils.tree_execution.run_tree_until_successful_common_analysis_structure_count(
            seq,
            node,
            expected_common_analysis_structure_count=2,
            max_iterations=120,
            tick_rate=20,
            on_successful_common_analysis_structure=assert_current_common_analysis_structure_updates_single_belief,
        )

        assert (
            len(successful_common_analysis_structure_instances) == 2
        ), "The file-reader integration fixture should yield the two available successful common analysis structure instances."
        assert (
            len(belief_uuids_after_each_successful_common_analysis_structure) == 2
        ), "The integration test should inspect the belief state immediately after both successful common analysis structure instances."
        assert (
            belief_uuids_after_each_successful_common_analysis_structure[0]
            == belief_uuids_after_each_successful_common_analysis_structure[1]
        ), "The second percept should update the same ObjectBeliefState created from the first percept."

        common_analysis_structure_ids = [
            common_analysis_structure.common_analysis_structure_id
            for common_analysis_structure in successful_common_analysis_structure_instances
        ]
        assert len(set(common_analysis_structure_ids)) == len(
            common_analysis_structure_ids
        ), "The repeated runner should count distinct common analysis structure ids, not repeated ticks of the same common analysis structure."

        first_common_analysis_structure, second_common_analysis_structure = (
            successful_common_analysis_structure_instances
        )
        same_color_image = np.array_equal(
            first_common_analysis_structure.get(
                robokudo.common_analysis_structure.CommonAnalysisStructureViews.COLOR_IMAGE
            ),
            second_common_analysis_structure.get(
                robokudo.common_analysis_structure.CommonAnalysisStructureViews.COLOR_IMAGE
            ),
        )
        same_depth_image = np.array_equal(
            first_common_analysis_structure.get(
                robokudo.common_analysis_structure.CommonAnalysisStructureViews.DEPTH_IMAGE
            ),
            second_common_analysis_structure.get(
                robokudo.common_analysis_structure.CommonAnalysisStructureViews.DEPTH_IMAGE
            ),
        )
        assert not (
            same_color_image and same_depth_image
        ), "The two successful common analysis structure instances should come from distinct RGB-D percepts."

        all_hypotheses = []
        for common_analysis_structure in successful_common_analysis_structure_instances:
            types_of_annotations = list(
                map(type, common_analysis_structure.annotations)
            )
            assert (
                types_of_annotations.count(robokudo.types.annotation.Plane) == 1
            ), "The file-reader baseline should detect one support plane per percept."
            assert (
                types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 1
            ), "The cropped file-reader baseline should produce one object hypothesis per percept."

            object_hypotheses = common_analysis_structure.filter_annotations_by_type(
                robokudo.types.scene.ObjectHypothesis
            )
            object_hypothesis = object_hypotheses[0]
            all_hypotheses.append(object_hypothesis)

            pose_annotations = common_analysis_structure.filter_by_type(
                robokudo.types.annotation.PoseAnnotation,
                object_hypothesis.annotations,
            )
            bbox_annotations = common_analysis_structure.filter_by_type(
                robokudo.types.annotation.BoundingBox3DAnnotation,
                object_hypothesis.annotations,
            )
            stamped_pose_annotations = common_analysis_structure.filter_by_type(
                robokudo.types.annotation.StampedPoseAnnotation,
                object_hypothesis.annotations,
            )

            assert len(pose_annotations) == 1
            assert len(bbox_annotations) == 1
            assert len(stamped_pose_annotations) == 1
            assert (
                stamped_pose_annotations[0].source
                == SemanticDigitalTwinConnector.__name__
            )
            assert stamped_pose_annotations[0].frame == "map"

        object_beliefs = list(rk_world.get_object_belief_states().values())
        assert (
            len(object_beliefs) == 1
        ), "Both observations of the static object should update one stable object belief."

        object_belief = object_beliefs[0]
        assert (
            object_belief.latest_hypothesis is all_hypotheses[-1]
        ), "The object belief should point to the latest associated object hypothesis."
        assert (
            object_belief.latest_pose is not None
        ), "The object belief should expose the latest pose from its associated hypothesis."
        assert (
            object_belief.latest_bbox_3d is not None
        ), "The object belief should expose the latest 3D bounding box from its associated hypothesis."
        assert (
            object_belief.body.parent_connection is not None
        ), "The object belief body should be attached to the SemDT world frame."
        assert (
            len(object_belief.body.visual.shapes) == 1
        ), "The object belief body should contain one visual shape derived from the latest bounding box."

    def test_run_semdt_raytracer_ae_successfully(self, node: Node) -> None:
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            world_descriptor_name="world_semdt_raytracer_cylinders",
            resolution=128,
        )

        plane_desc = PlaneAnnotator.Descriptor()
        plane_desc.parameters.distance_threshold = 0.01
        plane_desc.parameters.random_seed = 0

        cluster_desc = PointCloudClusterExtractor.Descriptor()
        cluster_desc.parameters.dbscan_min_cluster_count = 8
        cluster_desc.parameters.min_cluster_count = 20
        cluster_desc.parameters.min_on_plane_point_count = 10
        cluster_desc.parameters.eps = 0.05

        shape_desc = ShapeEstimatorAnnotator.Descriptor()
        shape_desc.parameters.minimum_point_count = 20
        shape_desc.parameters.distance_threshold = 0.015
        shape_desc.parameters.cylinder_minimum_inlier_ratio = 0.5
        shape_desc.parameters.cuboid_minimum_inlier_ratio = 0.3
        shape_desc.parameters.log_candidate_metrics = False
        shape_desc.parameters.log_rejection_reasons = False

        seq = robokudo.pipeline.Pipeline("SemDTRayTracerTestPipeline")
        seq.add_children(
            [
                robokudo.behaviours.clear_annotator_outputs.ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_desc),
                PointCloudClusterExtractor(descriptor=cluster_desc),
                ClusterColorAnnotator(),
                ClusterPoseBBAnnotator(),
                ShapeEstimatorAnnotator(descriptor=shape_desc),
            ]
        )

        tree_result = robokudo.utils.tree_execution.run_tree_once(
            seq, node, max_iterations=80, tick_rate=20
        )
        assert tree_result is py_trees.common.Status.SUCCESS

        types_of_annotations = list(
            map(type, seq.common_analysis_structure.annotations)
        )
        assert types_of_annotations.count(robokudo.types.annotation.Plane) == 1
        assert types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 2

        object_hypotheses = seq.common_analysis_structure.filter_annotations_by_type(
            robokudo.types.scene.ObjectHypothesis
        )
        dominant_colors = {
            colors[0].color
            for object_hypothesis in object_hypotheses
            if (
                colors := seq.common_analysis_structure.filter_by_type(
                    robokudo.types.annotation.SemanticColor,
                    object_hypothesis.annotations,
                )
            )
        }
        assert dominant_colors == {"red", "blue"}

        object_shapes = [
            seq.common_analysis_structure.filter_by_type(
                robokudo.types.annotation.Shape,
                object_hypothesis.annotations,
            )
            for object_hypothesis in object_hypotheses
        ]
        assert [len(shapes) for shapes in object_shapes] == [1, 1]
        assert all(
            isinstance(shapes[0], robokudo.types.annotation.Cylinder)
            for shapes in object_shapes
        )
        assert all(len(shapes[0].inliers) > 0 for shapes in object_shapes)

    def test_run_semdt_raytracer_tabletop_ae_loads_mesh_world_descriptor(
        self, node: Node
    ) -> None:
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            world_descriptor_name="world_semdt_raytracer_tabletop",
            resolution=128,
        )

        plane_desc = PlaneAnnotator.Descriptor()
        plane_desc.parameters.distance_threshold = 0.01
        plane_desc.parameters.random_seed = 0

        cluster_desc = PointCloudClusterExtractor.Descriptor()
        cluster_desc.parameters.dbscan_min_cluster_count = 8
        cluster_desc.parameters.min_cluster_count = 20
        cluster_desc.parameters.min_on_plane_point_count = 10
        cluster_desc.parameters.eps = 0.05

        seq = robokudo.pipeline.Pipeline("SemDTRayTracerTabletopTestPipeline")
        seq.add_children(
            [
                robokudo.behaviours.clear_annotator_outputs.ClearAnnotatorOutputs(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_desc),
                PointCloudClusterExtractor(descriptor=cluster_desc),
                ClusterColorAnnotator(),
                ClusterPoseBBAnnotator(),
            ]
        )

        tree_result = robokudo.utils.tree_execution.run_tree_once(
            seq, node, max_iterations=80, tick_rate=20
        )
        assert tree_result is py_trees.common.Status.SUCCESS

        types_of_annotations = list(
            map(type, seq.common_analysis_structure.annotations)
        )
        assert types_of_annotations.count(robokudo.types.annotation.Plane) == 1
        assert types_of_annotations.count(robokudo.types.scene.ObjectHypothesis) == 4
