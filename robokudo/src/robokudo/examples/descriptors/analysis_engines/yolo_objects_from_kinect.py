"""
Example analysis engine that perceives the objects on a table from the topics of the
PR2's head-mounted Kinect into the semantic digital twin and classifies them with a YOLO
model.

An object whose class names a perceivable semantic annotation, such as ``Noodles``, gets
that annotation; any other object stays a ``PerceivedObject`` labelled with the class.
The weights of :attr:`AnalysisEngine.yolo_model` have to be in
``robokudo/src/robokudo/models``. A recording of the topics and of TF stands in for the
robot, and the server prints the semantic annotation of every object::

    python -m robokudo.examples.serve_shared_world
    ros2 bag play --loop <recording>
    ros2 run robokudo_ros main _ros_pkg=robokudo.examples _ae=yolo_objects_from_kinect
"""

from __future__ import annotations

from dataclasses import dataclass

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator
from robokudo.examples.descriptors.analysis_engines import objects_from_kinect

# %% analysis engine


@dataclass
class AnalysisEngine(objects_from_kinect.AnalysisEngine):
    """
    Segments the objects on the table the Kinect sees, classifies them with a YOLO model
    and adds them to the perception world.
    """

    yolo_model: str = "yolov8n.pt"
    """
    The file name of the YOLO weights in RoboKudo's models directory.
    """

    minimum_detection_confidence: float = 0.7
    """
    How confident a YOLO detection has to be to classify an object.
    """

    def name(self) -> str:
        return "yolo_objects_from_kinect"

    def object_classifiers(self) -> list[BaseAnnotator]:
        descriptor = YoloAnnotator.Descriptor()
        descriptor.parameters.pretrained_model = self.yolo_model
        descriptor.parameters.threshold = self.minimum_detection_confidence
        descriptor.parameters.classify_existing_hypotheses = True
        return [YoloAnnotator(descriptor=descriptor)]
