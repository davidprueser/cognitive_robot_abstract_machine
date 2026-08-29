from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


def test_repr_does_not_raise_after_construction(rclpy_node, cylinder_bot_world):
    """
    The dataclass-generated ``__repr__`` reads every declared field, including
    ``_publisher``.

    If ``__post_init__`` stores the created publisher under a different attribute name,
    that field is never set and ``repr()`` raises instead of describing the instance.
    """
    publisher = VizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)

    repr(publisher)
