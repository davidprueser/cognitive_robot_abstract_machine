from __future__ import annotations
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.descriptors.camera_configs.config_realsense import RealsenseCameraConfig
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    def name(self) -> str:
        return "realsense"

    def implementation(self) -> Pipeline:
        """
        Create a basic pipeline that does tabletop segmentation.
        """
        kinect_config = RealsenseCameraConfig()

        seq = Pipeline("RealSensePipeline")
        seq.add_children(
            [
                # PipelineTrigger(),
                pipeline_init(),
                CollectionReaderAnnotator(camera_config=kinect_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
            ]
        )
        return seq
