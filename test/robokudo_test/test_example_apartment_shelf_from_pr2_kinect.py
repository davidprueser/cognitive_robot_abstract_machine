"""
Tests for the example analysis engine that perceives the apartment's shelf through the
Kinect of a PR2 Giskard simulates.
"""

from __future__ import annotations

import pytest

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.descriptors.camera_configs.config_semdt_raytracer_robot_camera import (
    SemDTRayTracerRobotCameraConfig,
)
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.examples.descriptors.analysis_engines.apartment_shelf_from_pr2_kinect import (
    AnalysisEngine,
)
from robokudo.io.camera_interface import CameraInterface
from robokudo.pipeline import Pipeline
from robokudo.utils.module_loader import ModuleLoader

# %% pipeline


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    """
    :return: The pipeline of the analysis engine, with a camera interface that renders
        nothing.
    """
    # Keeps the camera configuration real while nothing renders the scene.
    monkeypatch.setitem(
        CollectionReaderDescriptorFactory._camera_interface_types,
        "semdt_raytracer_robot_camera",
        CameraInterface,
    )
    return AnalysisEngine().implementation()


# %% camera configuration


class TestCameraConfiguration:
    def test_apartment_is_rendered_from_the_kinect_of_the_robot(
        self, pipeline: Pipeline
    ) -> None:
        [reader] = [
            child
            for child in pipeline.children
            if isinstance(child, CollectionReaderAnnotator)
        ]
        camera_config = reader.descriptor.parameters.camera_config
        assert isinstance(camera_config, SemDTRayTracerRobotCameraConfig)
        assert (
            camera_config.world_descriptor_ros_package,
            camera_config.world_descriptor_name,
        ) == ("robokudo.examples", "apartment_reality")


# %% region of interest


class TestRegionOfInterest:
    def test_cloud_is_cropped_to_the_shelf_region_of_the_world(
        self, pipeline: Pipeline
    ) -> None:
        [crop] = [
            child
            for child in pipeline.children
            if isinstance(child, PointcloudCropAnnotator)
        ]
        parameters = crop.descriptor.parameters
        shelf_region = AnalysisEngine().shelf_region
        assert parameters.relative_to_world
        assert (
            parameters.min_x,
            parameters.min_y,
            parameters.min_z,
            parameters.max_x,
            parameters.max_y,
            parameters.max_z,
        ) == (
            shelf_region.min_x,
            shelf_region.min_y,
            shelf_region.min_z,
            shelf_region.max_x,
            shelf_region.max_y,
            shelf_region.max_z,
        )


# %% loading


class TestLoading:
    def test_main_loads_the_example_from_the_examples_package(self) -> None:
        analysis_engine = ModuleLoader().load_ae(
            ros_pkg_name="robokudo.examples",
            module_name="apartment_shelf_from_pr2_kinect",
        )

        # The loader reloads an imported module, so the class is a fresh copy.
        assert (
            type(analysis_engine).__module__,
            type(analysis_engine).__qualname__,
        ) == (AnalysisEngine.__module__, AnalysisEngine.__qualname__)
