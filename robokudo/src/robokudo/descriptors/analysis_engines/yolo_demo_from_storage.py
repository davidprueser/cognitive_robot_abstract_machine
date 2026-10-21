from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_color import ClusterColorAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.factories.collection_reader_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.idioms import pipeline_initialization
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
        storage_reader_descriptor = CollectionReaderDescriptorFactory.create_descriptor(
            "mongo"
        )

        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.pretrained_model = "yolov8m.pt"
        yolo_descriptor.parameters.threshold = 0.2
        yolo_descriptor.parameters.precision_mode = True

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_initialization(),
                CollectionReaderAnnotator(descriptor=storage_reader_descriptor),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                YoloAnnotator(descriptor=yolo_descriptor),
                ClusterColorAnnotator(),
            ]
        )
        return seq
