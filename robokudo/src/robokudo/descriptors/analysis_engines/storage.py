"""
Analysis engine that records Kinect camera data to storage.
"""

from __future__ import annotations

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.descriptors.camera_configs.config_kinect_robot import KinectCameraConfig
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.storage import StorageWriter

from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline


class AnalysisEngine(AnalysisEngineInterface):
    """
    Records preprocessed Kinect camera data with StorageWriter.
    """

    def name(self) -> str:
        return "storage"

    def implementation(self) -> Pipeline:
        kinect_config = KinectCameraConfig()

        seq = Pipeline()
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(camera_config=kinect_config),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                StorageWriter(),
            ]
        )
        return seq
