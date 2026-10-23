"""
Example server of a shared world for RoboKudo to join, standing in for the process that
executes robot motions.

The world holds only the ``map`` frame. RoboKudo started while it runs fetches it, and
every object it perceives is added to this world as well; the server prints the semantic
annotation of each and shows the world in RViz, with its tf frames below
:attr:`SharedWorldServer.tf_frame_prefix` so they never claim the robot's frames::

    python -m robokudo.examples.serve_shared_world
    ros2 run robokudo_ros main _ros_pkg=robokudo.examples _ae=shared_world_from_files
"""

from __future__ import annotations

from dataclasses import dataclass, field

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.tf_publisher import TfFrameNames, TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.examples.new_semantic_annotation_reporter import (
    NewSemanticAnnotationReporter,
)

# %% served world


@dataclass
class SharedWorldServer:
    """
    A world served to the processes that join it and kept in sync with them.
    """

    node: Node
    """
    The node the world is served and synchronized through.
    """

    world_frame: str = "map"
    """
    The name of the root frame of the served world.
    """

    tf_frame_prefix: str = "shared_world/"
    """
    Put in front of the tf frame name of every body of :attr:`world`.
    """

    world: World = field(init=False)
    """
    The served world, holding only :attr:`world_frame` until others change it.
    """

    reporter: NewSemanticAnnotationReporter = field(init=False)
    """
    Prints the semantic annotations other processes add to the world.
    """

    server: FetchWorldServer = field(init=False)
    """
    Hands the world to processes that join.
    """

    synchronizer: WorldSynchronizer = field(init=False)
    """
    Keeps the world in sync with the processes that joined.
    """

    tf_publisher: TFPublisher = field(init=False)
    """
    Publishes the tf tree of the world, with the root of the world joined to
    :attr:`world_frame`.
    """

    marker_publisher: VizMarkerPublisher = field(init=False)
    """
    Shows the world in RViz.
    """

    def __post_init__(self) -> None:
        self.world = World()
        with self.world.modify_world():
            self.world.add_body(Body(name=PrefixedName(self.world_frame)))
        self.reporter = NewSemanticAnnotationReporter(world=self.world)
        self.server = FetchWorldServer(node=self.node, world=self.world)
        self.synchronizer = WorldSynchronizer(node=self.node, _world=self.world)
        self.tf_publisher = TFPublisher(
            _world=self.world,
            node=self.node,
            frame_names=TfFrameNames(prefix=self.tf_frame_prefix),
        )
        self.marker_publisher = VizMarkerPublisher(
            _world=self.world, node=self.node, tf_publisher=self.tf_publisher
        )

    def close(self) -> None:
        """
        Stop serving, synchronizing and showing the world.
        """
        self.marker_publisher.stop()
        self.tf_publisher.stop()
        self.synchronizer.close()
        self.server.close()


# %% entry point


def main(report_period: float = 1.0) -> None:
    """
    Serve the world until ROS shuts down, as it does on SIGINT and SIGTERM.

    :param report_period: How often, in seconds, to report the semantic annotations not
        reported yet.
    """
    rclpy.init()
    node = rclpy.create_node("shared_world_server")
    served_world = SharedWorldServer(node=node)
    node.create_timer(report_period, served_world.reporter.report)
    # One thread runs the callbacks one at a time; unlike the single-threaded
    # executor, this one ends its spin quietly when ROS shuts down.
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(node)
    print(
        f"Serving a world with root '{served_world.world_frame}'. Start RoboKudo now."
    )
    try:
        executor.spin()
    finally:
        served_world.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
