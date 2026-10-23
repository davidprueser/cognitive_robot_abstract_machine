"""
Example analysis engines that add the objects on a table to the semantic digital twin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from robokudo.analysis_engine import AnalysisEngineInterface
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.core import BaseAnnotator
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

    def crop_descriptor(self) -> PointcloudCropAnnotator.Descriptor:
        """
        :return: The descriptor of the crop to the region the table is in.
        """
        return PointcloudCropAnnotator.Descriptor()

    def object_classifiers(self) -> list[BaseAnnotator]:
        """
        :return: The annotators that classify the located objects; none by default.
        """
        return []

    def implementation(self) -> Pipeline:
        pipeline = Pipeline("TabletopObjectsPipeline")
        pipeline.add_children(
            [
                pipeline_init(),
                CollectionReaderAnnotator(descriptor=self.camera_descriptor()),
                ImagePreprocessorAnnotator("ImagePreprocessor"),
                PointcloudCropAnnotator(descriptor=self.crop_descriptor()),
                PlaneAnnotator(),
                PointCloudClusterExtractor(),
                ClusterPoseBBAnnotator(),
                *self.object_classifiers(),
            ]
        )
        return pipeline
