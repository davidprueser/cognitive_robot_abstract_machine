"""
Analysis engine for object detection and world visualization from stored data.

This module provides an analysis engine that demonstrates object detection and
world visualization using stored camera data. It combines static object
detection with various visualization components to display object hypotheses,
world descriptor entities, and color information.

The pipeline implements the following functionality:
- Reading stored camera data from MongoDB
- Bootstrapping static world descriptor entities into the shared world
- Image preprocessing
- Static object detection with predefined parameters
- Visualization of object hypotheses
- Visualization of world descriptor entities
- Color analysis of detected objects

.. note::
    This engine uses predefined object detection parameters optimized for a
    specific use case (mug detection). Adjust the parameters for other objects
    or scenarios.
"""

from __future__ import annotations

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.annotators.cluster_color import ClusterColorAnnotator

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.object_hypothesis_visualizer import ObjectHypothesisVisualizer
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapAnnotator,
)
from robokudo.annotators.world_visualizer import WorldVisualizer
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.annotators.static_object_detector import StaticObjectDetectorAnnotator


class AnalysisEngine(AnalysisEngineInterface):
    """
    Analysis engine for object detection and world descriptor visualization.

    This class implements a pipeline that combines static object detection with
    various visualization components. It processes stored camera data to detect
    objects and visualize their properties and associated world descriptor entities.

    The pipeline includes:
    - Collection reader for accessing stored data
    - Image preprocessing
    - Static object detection with predefined parameters
    - Object hypothesis visualization
    - World descriptor visualization
    - Color analysis and visualization

    .. note::
        The static object detector is configured with specific parameters for
        mug detection. These parameters include bounding box dimensions and
        pose information that should be adjusted for different objects.
    """

    def name(self) -> str:
        """
        Get the name of the analysis engine.

        :return: The name identifier of this analysis engine
        """
        return "world_descriptor_from_storage"

    def implementation(self) -> Pipeline:
        """
        Create a pipeline for object detection and world descriptor visualization.

        This method constructs a processing pipeline that reads stored camera data,
        performs object detection with predefined parameters, and visualizes the
        results including object hypotheses, world descriptor entities, and color information.

        The static object detector is configured with specific parameters for a mug:
        - Bounding box: 397x126 pixels with size 49x106
        - Position: (0.202, -0.109, 1.096)
        - Rotation: Quaternion (0.575, 0.666, -0.360, 0.310)

        :return: The configured pipeline for object detection and visualization
        """
        cr_storage_config = MongoCameraConfig()

        seq = Pipeline("StoragePipeline")
        seq.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(camera_config=cr_storage_config),
                WorldDescriptorBootstrapAnnotator(
                    world_descriptor_name="world_iai_kitchen20"
                ),
                ImagePreprocessorAnnotator(name="ImagePreprocessor"),
                StaticObjectDetectorAnnotator(
                    bounding_box_x=397,
                    bounding_box_y=126,
                    bounding_box_width=49,
                    bounding_box_height=106,
                    create_pose_annotation=True,
                    pose_use_euler_angles=False,
                    class_name="Mug",
                ),
                ObjectHypothesisVisualizer(),
                WorldVisualizer(),
                ClusterColorAnnotator(),
            ]
        )
        return seq
