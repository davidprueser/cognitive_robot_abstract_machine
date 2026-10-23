"""
Tests for the example analysis engine that classifies the objects it perceives from the
topics of the Kinect with a YOLO model.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.descriptors.analysis_engines.yolo_objects_from_kinect import (
    AnalysisEngine,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline
from robokudo.utils.module_loader import ModuleLoader

# %% pipeline


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """
    :return: The pipeline of the analysis engine, with a camera interface that
        subscribes to nothing.
    """
    # Keeps the camera configuration real while nothing subscribes to the topics.
    monkeypatch.setitem(
        CollectionReaderDescriptorFactory._camera_interface_types,
        "kinect",
        CameraInterface,
    )
    return AnalysisEngine().implementation()


# %% classification


class TestClassification:
    def test_yolo_classifies_the_located_objects(self, pipeline: Pipeline) -> None:
        child_types = [type(child) for child in pipeline.children]

        assert child_types.index(YoloAnnotator) > child_types.index(
            ClusterPoseBBAnnotator
        )

    def test_yolo_classifies_existing_hypotheses_with_the_configured_model(
        self, pipeline: Pipeline
    ) -> None:
        [yolo] = [
            child for child in pipeline.children if isinstance(child, YoloAnnotator)
        ]
        parameters = yolo.descriptor.parameters
        analysis_engine = AnalysisEngine()
        assert (
            parameters.classify_existing_hypotheses,
            parameters.pretrained_model,
            parameters.threshold,
        ) == (
            True,
            analysis_engine.yolo_model,
            analysis_engine.minimum_detection_confidence,
        )


# %% loading


class TestLoading:
    def test_main_loads_the_example_from_the_examples_package(self) -> None:
        analysis_engine = ModuleLoader().load_ae(
            ros_pkg_name="robokudo.examples", module_name="yolo_objects_from_kinect"
        )

        # The loader reloads an imported module, so the class is a fresh copy.
        assert (
            type(analysis_engine).__module__,
            type(analysis_engine).__qualname__,
        ) == (AnalysisEngine.__module__, AnalysisEngine.__qualname__)
