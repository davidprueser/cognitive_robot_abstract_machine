"""
Tests that the demo analysis engine perceives in a recorded rosbag what it perceived
before RoboKudo's port to statecharts.

Runs only where the bag is present; CI covers the same annotators on stored frames.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import pytest
from py_trees.common import Status
from rclpy.node import Node
from semantic_digital_twin.robots.robot_parts import Camera

from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.descriptors.analysis_engines.demo import AnalysisEngine
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Plane
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.ros_frames import normalize_ros_frame_id
from robokudo.utils.tree_execution import run_tree_once

from .dataset.breakfast_table_rosbag import (
    BreakfastTableRosbag,
    DemoPassOutcome,
    RecordedKinectFrame,
    RecordedKinectFramePublisher,
)

pytestmark = pytest.mark.skipif(
    not BreakfastTableRosbag().is_available,
    reason="the breakfast table rosbag is only present on developer machines",
)


# %% One demo pass per recorded frame
@dataclass(frozen=True, kw_only=True)
class DemoPass:
    """
    One pass of the demo analysis engine over one recorded frame.
    """

    pipeline: Pipeline
    """
    The demo's pipeline after the pass.
    """

    status: Status
    """
    Status the pass ended with.
    """

    frame: RecordedKinectFrame
    """
    The frame the pass read.
    """


@pytest.fixture(scope="module")
def breakfast_table_rosbag() -> BreakfastTableRosbag:
    """
    The recording the demo is run on.
    """
    return BreakfastTableRosbag()


@pytest.fixture(
    params=BreakfastTableRosbag().demo_pass_outcomes,
    ids=lambda outcome: f"frame {outcome.frame_index}",
)
def expected_outcome(request: pytest.FixtureRequest) -> DemoPassOutcome:
    """
    What the demo perceives in one of the recorded frames.
    """
    return request.param


@pytest.fixture
def demo_pass(
    node: Node,
    breakfast_table_rosbag: BreakfastTableRosbag,
    expected_outcome: DemoPassOutcome,
) -> DemoPass:
    """
    Run the demo once over the frame of ``expected_outcome``, with its plane fitting
    seeded as the outcome was recorded.
    """
    pipeline = AnalysisEngine().implementation()
    plane_annotator = next(
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, PlaneAnnotator)
    )
    plane_annotator.descriptor.parameters.random_seed = (
        breakfast_table_rosbag.plane_random_seed
    )
    collection_reader = next(
        behaviour
        for behaviour in pipeline.iterate()
        if isinstance(behaviour, CollectionReaderAnnotator)
    )
    camera_config = collection_reader.descriptor.parameters.camera_config
    frame = breakfast_table_rosbag.read_frames(camera_config=camera_config)[
        expected_outcome.frame_index
    ]
    publisher = RecordedKinectFramePublisher(
        node=node, frame=frame, camera_config=camera_config
    )
    status = run_tree_once(pipeline, node)
    publisher.stop()
    return DemoPass(pipeline=pipeline, status=status, frame=frame)


# %% What the demo perceives
def test_demo_pass_succeeds(demo_pass: DemoPass) -> None:
    """
    The demo completes a pass over each recorded frame.
    """
    assert demo_pass.status is Status.SUCCESS


def test_demo_finds_the_table_plane(
    demo_pass: DemoPass, expected_outcome: DemoPassOutcome
) -> None:
    """
    The demo fits the recorded plane to the table, with the recorded inliers.
    """
    planes = demo_pass.pipeline.cas.filter_annotations_by_type(Plane)
    assert [len(plane.inliers) for plane in planes] == [
        expected_outcome.plane_inlier_count
    ]
    assert tuple(planes[0].model) == pytest.approx(expected_outcome.plane_model)


def test_demo_finds_the_objects_on_the_table(
    demo_pass: DemoPass, expected_outcome: DemoPassOutcome
) -> None:
    """
    The demo finds as many clusters on the table as it did before the port.
    """
    object_hypotheses = demo_pass.pipeline.cas.filter_annotations_by_type(
        ObjectHypothesis
    )
    assert len(object_hypotheses) == expected_outcome.object_hypothesis_count


def test_demo_leaves_only_the_camera_in_the_world(demo_pass: DemoPass) -> None:
    """
    Without a camera pose the demo puts no object into the world, which holds only the
    camera the frame was taken with.
    """
    world = demo_pass.pipeline.world
    camera_frame = normalize_ros_frame_id(demo_pass.frame.camera_info.header.frame_id)
    assert [str(body.name) for body in world.bodies] == [camera_frame]
    assert [
        (type(annotation), str(annotation.root.name))
        for annotation in world.semantic_annotations
    ] == [(Camera, camera_frame)]
