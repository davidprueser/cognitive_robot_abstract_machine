from __future__ import annotations
from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine detecting objects in stored camera data with YOLO.

    The detections are segmented with SAM (precision mode) and their colors annotated.
    """

    def name(self) -> str:
        """
        :return: The name identifier of this analysis engine
        """
        return "yolo_demo_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline detecting and segmenting objects in data read from MongoDB.

        :return: The configured pipeline
        """
        cr_storage_config = MongoCameraConfig()

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(camera_config=cr_storage_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                YoloAnnotator(
                    pretrained_model="yolov8m.pt", threshold=0.2, precision_mode=True
                ),
                ClusterColorAnnotator(),
            ]
        )
        return seq
