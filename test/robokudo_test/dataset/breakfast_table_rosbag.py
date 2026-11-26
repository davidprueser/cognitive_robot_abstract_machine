"""
A rosbag of a PR2's head Kinect looking at a breakfast table, and what the demo analysis
engine perceives in it.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import rosbag2_py
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.timer import Timer
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, CompressedImage
from typing_extensions import Any

from robokudo.descriptors.camera_configs.components import RGBDComponent


# %% Recorded frames
@dataclass(frozen=True, kw_only=True)
class RecordedKinectFrame:
    """
    One RGB-D frame of the recording, as the messages a Kinect camera interface
    synchronizes into a frame.
    """

    color_image: CompressedImage
    """
    JPEG compressed color image.
    """

    depth_image: CompressedImage
    """
    Depth image recorded closest in time to :attr:`color_image`.
    """

    camera_info: CameraInfo
    """
    Calibration of the color camera, stamped like :attr:`color_image`.
    """


@dataclass(kw_only=True)
class RecordedKinectFramePublisher:
    """
    Publishes one recorded frame periodically on the topics of a camera configuration,
    until stopped, so a subscription that connects late still receives it.
    """

    node: Node
    """
    Node the frame is published from.
    """

    frame: RecordedKinectFrame
    """
    Frame that is published.
    """

    camera_config: RGBDComponent
    """
    Configuration naming the topics the frame is published on.
    """

    publishing_period_seconds: float = 0.2
    """
    Time between two publications of the frame.
    """

    queue_size: int = 1
    """
    Number of messages each publisher keeps for a subscription that has not taken them.
    """

    color_image_publisher: Publisher = field(init=False)
    """
    Publisher of :attr:`RecordedKinectFrame.color_image`.
    """

    depth_image_publisher: Publisher = field(init=False)
    """
    Publisher of :attr:`RecordedKinectFrame.depth_image`.
    """

    camera_info_publisher: Publisher = field(init=False)
    """
    Publisher of :attr:`RecordedKinectFrame.camera_info`.
    """

    timer: Timer = field(init=False)
    """
    Timer that publishes the frame every :attr:`publishing_period_seconds`.
    """

    def __post_init__(self) -> None:
        """
        Start publishing :attr:`frame` every :attr:`publishing_period_seconds`.
        """
        self.color_image_publisher = self.node.create_publisher(
            CompressedImage, self.camera_config.topic_color, self.queue_size
        )
        self.depth_image_publisher = self.node.create_publisher(
            CompressedImage, self.camera_config.topic_depth, self.queue_size
        )
        self.camera_info_publisher = self.node.create_publisher(
            CameraInfo, self.camera_config.topic_camera_info, self.queue_size
        )
        self.timer = self.node.create_timer(
            self.publishing_period_seconds, self.publish_frame
        )

    def publish_frame(self) -> None:
        """
        Publish every message of :attr:`frame` once.
        """
        self.color_image_publisher.publish(self.frame.color_image)
        self.depth_image_publisher.publish(self.frame.depth_image)
        self.camera_info_publisher.publish(self.frame.camera_info)

    def stop(self) -> None:
        """
        Stop publishing and remove the timer and publishers from :attr:`node`.
        """
        self.node.destroy_timer(self.timer)
        self.node.destroy_publisher(self.color_image_publisher)
        self.node.destroy_publisher(self.depth_image_publisher)
        self.node.destroy_publisher(self.camera_info_publisher)


# %% Demo analysis engine outcomes
@dataclass(frozen=True, kw_only=True)
class DemoPassOutcome:
    """
    What one pass of :class:`robokudo.descriptors.analysis_engines.demo.AnalysisEngine`
    perceives in one recorded frame, with its plane fitting seeded by
    :attr:`BreakfastTableRosbag.plane_random_seed`.
    """

    frame_index: int
    """
    Position of the frame among the recorded color images.
    """

    plane_model: tuple[float, float, float, float]
    """
    Coefficients ``a, b, c, d`` of the table plane ``ax + by + cz + d = 0`` in the
    camera frame.
    """

    plane_inlier_count: int
    """
    Number of points of the cloud that lie on the table plane.
    """

    object_hypothesis_count: int
    """
    Number of clusters found on the table.
    """


# %% Breakfast table rosbag
@dataclass(frozen=True, kw_only=True)
class BreakfastTableRosbag:
    """
    A PR2's head Kinect looking at a table holding pancake mix, corn flakes, milk, a
    coffee can and a pancake maker: 55 color frames with their calibration, depth
    images and ``/tf``, recorded in 2014.
    """

    bag_directory: Path = field(
        default_factory=lambda: Path.home() / "workspaces" / "cram_ws" / "test"
    )
    """
    Directory holding the bag and its ``metadata.yaml``; absent on CI.
    """

    storage_identifier: str = "sqlite3"
    """
    Storage plugin the bag was written with.
    """

    serialization_format: str = "cdr"
    """
    Format the messages are serialized in.
    """

    plane_random_seed: int = 0
    """
    Seed of the demo's plane fitting that :attr:`demo_pass_outcomes` were recorded with.
    """

    demo_pass_outcomes: tuple[DemoPassOutcome, ...] = (
        DemoPassOutcome(
            frame_index=0,
            plane_model=(
                -0.007845388536900033,
                0.9012572625692269,
                0.4332133383737495,
                -0.9294657859583481,
            ),
            plane_inlier_count=53907,
            object_hypothesis_count=5,
        ),
        DemoPassOutcome(
            frame_index=27,
            plane_model=(
                -0.009556651823730316,
                0.9021196554904538,
                0.4313801080065062,
                -0.9285329308680839,
            ),
            plane_inlier_count=60096,
            object_hypothesis_count=5,
        ),
        DemoPassOutcome(
            frame_index=54,
            plane_model=(
                -0.006991041720279372,
                0.902527628110046,
                0.4305752034589541,
                -0.9292890134754517,
            ),
            plane_inlier_count=68846,
            object_hypothesis_count=6,
        ),
    )
    """
    What the demo perceives in the first, the middle and the last frame.
    """

    @property
    def is_available(self) -> bool:
        """
        Whether the bag is present on this machine.
        """
        return (self.bag_directory / "metadata.yaml").is_file()

    def read_frames(self, *, camera_config: RGBDComponent) -> list[RecordedKinectFrame]:
        """
        Read the recorded frames from the topics ``camera_config`` names, one per color
        image, in recording order.
        """
        messages_per_topic = self.read_messages(
            topics=[
                camera_config.topic_color,
                camera_config.topic_depth,
                camera_config.topic_camera_info,
            ]
        )
        camera_info_per_stamp = {
            Time.from_msg(camera_info.header.stamp).nanoseconds: camera_info
            for camera_info in messages_per_topic[camera_config.topic_camera_info]
        }
        depth_images = messages_per_topic[camera_config.topic_depth]
        frames = []
        for color_image in messages_per_topic[camera_config.topic_color]:
            color_stamp = Time.from_msg(color_image.header.stamp).nanoseconds
            closest_depth_image = min(
                depth_images,
                key=lambda depth_image: abs(
                    Time.from_msg(depth_image.header.stamp).nanoseconds - color_stamp
                ),
            )
            frames.append(
                RecordedKinectFrame(
                    color_image=color_image,
                    depth_image=closest_depth_image,
                    camera_info=camera_info_per_stamp[color_stamp],
                )
            )
        return frames

    def read_messages(self, *, topics: list[str]) -> dict[str, list[Any]]:
        """
        Read the messages recorded on ``topics``, per topic in recording order.
        """
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(
                uri=str(self.bag_directory), storage_id=self.storage_identifier
            ),
            rosbag2_py.ConverterOptions(
                input_serialization_format=self.serialization_format,
                output_serialization_format=self.serialization_format,
            ),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        message_type_per_topic = {
            topic.name: get_message(topic.type)
            for topic in reader.get_all_topics_and_types()
        }
        messages_per_topic = {topic: [] for topic in topics}
        while reader.has_next():
            topic, serialized_message, _ = reader.read_next()
            messages_per_topic[topic].append(
                deserialize_message(serialized_message, message_type_per_topic[topic])
            )
        return messages_per_topic
