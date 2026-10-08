"""
Object hypotheses as a perception pipeline reports them, for tests that consume them.
"""

from __future__ import annotations

import numpy as np

from robokudo.types.annotation import BoundingBox3DAnnotation, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis


def make_object_hypothesis(
    *,
    roi: tuple[int, int, int, int] = (10, 15, 20, 25),
    translation: tuple[float, float, float] = (0.0, 0.0, 1.0),
    extent: tuple[float, float, float] = (0.1, 0.2, 0.3),
    source: str = "test_source",
) -> ObjectHypothesis:
    """
    An object hypothesis with an image region, a pose and a 3D bounding box.

    :param roi: Image region as x, y, width and height.
    :param translation: Position of the object in camera coordinates.
    :param extent: Lengths of the bounding box along x, y and z.
    :param source: Name of the annotator the annotations claim to come from.
    :return: The hypothesis.
    """
    hypothesis = ObjectHypothesis()

    x, y, width, height = roi
    hypothesis.roi.roi.pos.x = x
    hypothesis.roi.roi.pos.y = y
    hypothesis.roi.roi.width = width
    hypothesis.roi.roi.height = height
    hypothesis.roi.mask = np.ones((height, width), dtype=np.uint8)

    pose = PoseAnnotation()
    pose.source = source
    pose.translation = list(translation)
    pose.rotation = [0.0, 0.0, 0.0, 1.0]
    hypothesis.annotations.append(pose)

    bounding_box = BoundingBox3DAnnotation()
    bounding_box.source = source
    bounding_box.pose.translation = list(translation)
    bounding_box.pose.rotation = [0.0, 0.0, 0.0, 1.0]
    bounding_box.x_length, bounding_box.y_length, bounding_box.z_length = extent
    hypothesis.annotations.append(bounding_box)

    return hypothesis
