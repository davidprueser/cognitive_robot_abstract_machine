"""
Annotation writer classes for RoboKudo.

This module provides annotators for writing and publishing annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import json
import os
import shutil

from py_trees.common import Status
from typing_extensions import Any
from rclpy.publisher import Publisher
from std_msgs.msg import String

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.cas_annotation_codecs import serialize_annotations
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry


@dataclass(kw_only=True)
class AnnotationStorageWriter(BaseAnnotator):
    """
    Annotator for writing annotations to storage in JSON format.

    This annotator writes the current CAS annotations to files in a specified directory,
    using JSON serialization.
    """

    basic_path: str = "annotations"
    """
    Base directory for storing annotations.
    """

    suffix: str = "json"
    """
    File extension for annotation files.
    """

    counter: int = field(default=-1, init=False)
    """
    Number of the file written last; -1 before the first.
    """

    def compute(self) -> Status:
        """
        Write current CAS annotations to storage.

        Serializes the current annotations to JSON and writes them to a file in the
        configured directory. Files are numbered sequentially.

        :return: SUCCESS after writing annotations
        """
        # Encode annotations via KRROOD serializer and dump to JSON.
        json_data_string = json.dumps(serialize_annotations(self.get_cas().annotations))

        dir_path = os.path.join(self.basic_path, self.name)

        # increase count
        self.counter += 1

        if self.counter == 0 and os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)

        json_path = os.path.join(dir_path, "{}.{}".format(self.counter, self.suffix))
        with open(json_path, "w") as f:
            f.write(json_data_string)

        return Status.SUCCESS


@dataclass(kw_only=True)
class AnnotationPublisherWriter(BaseAnnotator):
    """
    Annotator for publishing annotations via ROS topics.

    This annotator publishes the current CAS annotations as JSON-encoded strings over a
    ROS topic.
    """

    topic_name: str = "/annotations"
    """
    Name of the ROS topic to publish on.
    """

    publisher: Publisher | None = field(default=None, init=False)
    """
    ROS publisher of the annotations, created in :meth:`setup`.
    """

    def setup(self, **kwargs: Any) -> bool:
        """
        Create the ROS publisher of the annotations.
        """
        node = ROSNodeRegistry().get()
        self.publisher = node.create_publisher(String, self.topic_name, 10)
        return super().setup(**kwargs)

    def compute(self) -> Status:
        """
        Publish current CAS annotations.

        Serializes the current annotations to JSON and publishes them on the configured
        ROS topic.

        :return: SUCCESS after publishing annotations
        """
        # Encode annotations via KRROOD serializer and dump to JSON.
        json_data_string = json.dumps(serialize_annotations(self.get_cas().annotations))

        if self.publisher is None:
            raise RuntimeError("Publisher is not initialized. Call setup() first.")
        self.publisher.publish(String(data=json_data_string))

        return Status.SUCCESS
