from __future__ import annotations

from dataclasses import dataclass, field

from visualization_msgs.msg import MarkerArray

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


# %% helpers
@dataclass(eq=False)
class RecordingVizMarkerPublisher(VizMarkerPublisher):
    """
    A marker publisher that records every marker array it is about to publish.
    """

    published: list[MarkerArray] = field(default_factory=list, init=False)
    """
    The marker arrays handed to :meth:`publish_markers`, in order.
    """

    def publish_markers(self) -> None:
        self.published.append(self.markers)
        super().publish_markers()


# %% construction
def test_repr_does_not_raise_after_construction(rclpy_node, cylinder_bot_world):
    """
    The generated ``__repr__`` reads every declared field, so a field the constructor
    never sets makes ``repr`` raise instead of describing the publisher.
    """
    publisher = VizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)

    assert repr(publisher).startswith(VizMarkerPublisher.__name__)


# %% publishing
def test_every_model_change_is_published_through_publish_markers(
    rclpy_node, cylinder_bot_world
):
    publisher = RecordingVizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)
    published_before = len(publisher.published)

    publisher.notify_model_change()

    assert len(publisher.published) == published_before + 1
    assert publisher.published[-1] is publisher.markers
