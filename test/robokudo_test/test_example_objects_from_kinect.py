"""
Tests for the example analysis engine that perceives the topics of the Kinect into the
world.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.descriptors.camera_configs.config_kinect_robot import (
    KinectCameraConfig,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.descriptors.analysis_engines.objects_from_kinect import (
    AnalysisEngine,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.utils.module_loader import ModuleLoader

# %% camera configuration


class TestCameraConfiguration:
    def test_kinect_is_placed_where_tf_reports_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Keeps the camera configuration real while nothing subscribes to the topics.
        monkeypatch.setitem(
            CollectionReaderDescriptorFactory._camera_interface_types,
            "kinect",
            CameraInterface,
        )

        pipeline = AnalysisEngine().implementation()

        [reader] = [
            child
            for child in pipeline.children
            if isinstance(child, CollectionReaderAnnotator)
        ]
        camera_config = reader.descriptor.parameters.camera_config
        assert isinstance(camera_config, KinectCameraConfig)
        assert camera_config.lookup_viewpoint


# %% loading


class TestLoading:
    def test_main_loads_the_example_from_the_examples_package(self) -> None:
        analysis_engine = ModuleLoader().load_ae(
            ros_pkg_name="robokudo.examples", module_name="objects_from_kinect"
        )

        # The loader reloads an imported module, so the class is a fresh copy.
        assert (
            type(analysis_engine).__module__,
            type(analysis_engine).__qualname__,
        ) == (AnalysisEngine.__module__, AnalysisEngine.__qualname__)
