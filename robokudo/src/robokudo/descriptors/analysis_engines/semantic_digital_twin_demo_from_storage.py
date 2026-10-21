from robokudo.annotators.cluster_pose_bounding_box import (
    ClusterPoseBoundingBoxAnnotator,
)
from robokudo.world import world_instance
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import (
    OutlierRemovalOnObjectHypothesisAnnotator,
)
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.semantic_world_connector import SemanticDigitalTwinConnector
from robokudo.descriptors.factories.collection_reader_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_initialization
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "semantic_digital_twin_demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a tabletop segmentation pipeline that synchronizes detections to semantic digital twin.
        """
        tf_publisher = TFPublisher(
            _world=world_instance(), node=ROSNodeRegistry().get()
        )

        storage_reader_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
            "mongo"
        )

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_initialization(),
                CollectionReaderAnnotator(descriptor=storage_reader_descriptor),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterColorHistogramAnnotator(),
                OutlierRemovalOnObjectHypothesisAnnotator(),
                ClusterPoseBoundingBoxAnnotator(),
                SemanticDigitalTwinConnector(),
            ]
        )
        return seq
