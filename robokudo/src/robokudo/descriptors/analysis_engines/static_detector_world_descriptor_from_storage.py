from __future__ import annotations
from py_trees.behaviour import Behaviour

import robokudo.analysis_engine
import robokudo.behaviours.clear_errors
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.idioms
import robokudo.io.storage_reader_interface
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.location import LocationAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.static_object_detector import (
    StaticObjectDetectorAnnotator,
    StaticObjectMode,
)
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapAnnotator,
)


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):
    def name(self) -> str:
        return "static_detector_world_descriptor_from_storage"

    def implementation(self) -> Behaviour:
        cr_storage_config = MongoCameraConfig()

        seq = robokudo.pipeline.Pipeline("StoragePipeline")
        seq.add_children(
            [
                robokudo.idioms.pipeline_init(),
                CollectionReaderAnnotator(camera_config=cr_storage_config),
                WorldDescriptorBootstrapAnnotator(
                    world_descriptor_name="world_iai_kitchen20"
                ),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                StaticObjectDetectorAnnotator(
                    mode=StaticObjectMode.WORLD_DESCRIPTOR,
                    class_names=["cereal", "milk"],
                    create_bounding_box_annotation=True,
                    create_pose_annotation=True,
                ),
                ObjectHypothesisVisualizer(visualize_full_cloud=True),
            ]
        )
        return seq
