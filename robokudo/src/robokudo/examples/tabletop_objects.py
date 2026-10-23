"""
Example analysis engines that add the objects on a table to the semantic digital twin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline

# %% tabletop objects


class TabletopObjectsEngine(AnalysisEngineInterface, ABC):
    """
    Segments the objects on the table a camera sees and adds them to the perception
    world.
    """

    @abstractmethod
    def camera_descriptor(self) -> CollectionReaderAnnotator.Descriptor:
        """
        :return: The descriptor of the reader of the camera that sees the table, placing
            the camera in the perception world.
        """

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("TabletopObjectsPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=self.camera_descriptor()),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
            ]
        )
        return pipeline
