"""
Example process that builds the apartment around the robot Giskard spawned, standing in
for whoever maps the robot's surroundings.

It joins the world Giskard serves, localizes the robot in front of the apartment's shelf
and builds the apartment around it, with the objects the robot already knows slightly
misplaced. It then prints every body perception adds to the world::

    ros2 launch giskardpy_ros giskardpy_pr2_standalone.launch.py
    python -m robokudo.examples.build_apartment_around_robot
    ros2 run robokudo_ros main _ros_pkg=robokudo.examples _ae=apartment_shelf_from_pr2_kinect
"""

from __future__ import annotations

from datetime import timedelta
from threading import Thread

import rclpy
from rclpy.executors import MultiThreadedExecutor
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

from robokudo.examples.apartment_scene import ApartmentScene
from robokudo.examples.new_body_reporter import NewBodyReporter

# %% entry point


def main(
    odometry_frame: str = "odom_combined",
    world_discovery_timeout: timedelta = timedelta(minutes=5),
    report_period: float = 1.0,
) -> None:
    """
    Build the apartment around the robot and report the bodies perception adds until ROS
    shuts down, as it does on SIGINT and SIGTERM.

    :param odometry_frame: The name of the odometry frame of the robot.
    :param world_discovery_timeout: How long to wait for Giskard to serve its world,
        long enough for it to load the robot.
    :param report_period: How often, in seconds, to report the bodies not reported yet.
    """
    rclpy.init()
    node = rclpy.create_node("apartment_builder")
    # One thread runs the callbacks one at a time; unlike the single-threaded
    # executor, this one ends its spin quietly when ROS shuts down.
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    print("Waiting for Giskard to serve its world.")
    world = fetch_world_from_service(
        node, timeout_seconds=world_discovery_timeout.total_seconds()
    )
    synchronizer = WorldSynchronizer(node=node, _world=world)
    ApartmentScene().describe(world, odometry_frame=odometry_frame)
    reporter = NewBodyReporter(world=world)
    node.create_timer(report_period, reporter.report)
    print("Built the apartment around the robot. Start RoboKudo now.")
    try:
        executor_thread.join()
    finally:
        synchronizer.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
