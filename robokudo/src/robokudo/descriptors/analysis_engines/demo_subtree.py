from __future__ import annotations
from robokudo.descriptors.camera_configs.config_kinect_robot_wo_transform import (
    KinectWoTfCameraConfig,
)
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.cluster_color_histogram import ClusterColorHistogramAnnotator
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.analysis_engines.subtree_tabletop_object_localization import (
    Subtree as TTLocalizationSubtree,
)
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.pipeline import Pipeline
from robokudo.idioms import pipeline_init


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "demo_subtree"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline that does tabletop segmentation.
        """
        kinect_config = KinectWoTfCameraConfig()

        seq = Pipeline("RWPipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(camera_config=kinect_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                PointcloudCropAnnotator(),
                TTLocalizationSubtree().implementation(),
                ClusterColorAnnotator(),
                ClusterColorHistogramAnnotator(),
                ClusterPositionAnnotator(),
                # Additional annotators (e.g., QueryAnnotator, ActionServerCheck) can be added if needed.
            ]
        )
        return seq
