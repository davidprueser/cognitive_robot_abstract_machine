"""
Analysis engine for simulated RGB-D input from semantic digital twin RayTracer with
query functionality.

This pipeline mirrors the standard tabletop segmentation flow but uses the
`semdt_raytracer` camera descriptor, which renders color/depth images from a
world descriptor instead of reading a physical camera stream.
"""

from __future__ import annotations

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_pose_bounding_box import (
    ClusterPoseBoundingBoxAnnotator,
)
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator, GenerateQueryResult
from robokudo.behaviours.action_server_checks import ActionServerCheck
from robokudo.descriptors.factories.collection_reader_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_initialization
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semantic_digital_twin_raytracer_query_demo"

    def implementation(self) -> Pipeline:
        raytracer_config = CollectionReaderDescriptorFactory.create_descriptor(
            "semdt_raytracer",
            world_descriptor_name="world_semantic_digital_twin_raytracer_tabletop",
        )
        plane_descriptor = PlaneAnnotator.Descriptor()
        plane_descriptor.parameters.distance_threshold = 0.01

        query_result_descriptor = GenerateQueryResult.Descriptor()
        query_result_descriptor.parameters.filter_by_query = True

        pipeline = Pipeline("SemDTRayTracerPipeline")
        pipeline.add_children(
            [
                pipeline_initialization(),
                QueryAnnotator(),
                CollectionReaderAnnotator(descriptor=raytracer_config),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(descriptor=plane_descriptor),
                PointCloudClusterExtractor(),
                ClusterColorAnnotator(),
                ClusterPoseBoundingBoxAnnotator(),
                GenerateQueryResult(descriptor=query_result_descriptor),
                ActionServerCheck(),
            ]
        )
        return pipeline
