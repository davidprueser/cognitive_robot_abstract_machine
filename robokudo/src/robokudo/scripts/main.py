#!/usr/bin/env python3

from __future__ import annotations

from robokudo.world import world_instance
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)

import argparse
import logging
import os
import sys
import time
import traceback
from threading import Thread

# For time measurements or additional logic
from timeit import default_timer as timer

import rclpy
import rclpy.impl.logging_severity
import rclpy.logging
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.parameter import Parameter
from typing_extensions import TYPE_CHECKING

# RoboKudo imports
from robokudo.annotators.query import QueryActionServer
from robokudo.definitions import LOGGING_IDENTIFIER_MAIN_EXECUTABLE, PACKAGE_NAME
from robokudo.garden import grow_tree
from robokudo.identifier import BlackboardIdentifier
from robokudo.utils.logging_configuration import configure_logging
from robokudo.utils.module_loader import ModuleLoader
from robokudo.utils.tree import setup_with_descendants_robokudo
from robokudo.vis.cv_visualizer import CVVisualizer
from robokudo.vis.ros_visualizer import AllAnnotatorROSVisualizer, SharedROSVisualizer

if TYPE_CHECKING:
    from py_trees_ros.trees import BehaviourTree

# Silence some TensorFlow GPU logs if needed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_analysis_engine(
    analysis_engine_name: str,
    analysis_engine_root: BehaviourTree,
    tickrate: int = 20,
) -> None:
    """
    Run an Analysis Engine (analysis engine) by periodically ticking the Behavior Tree.
    """
    logger = logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE)
    logger.info(f"Running analysis engine named '{analysis_engine_name}'...")

    blackboard = Blackboard()
    blackboard.set("common_analysis_structure", None)
    tick_count = 0

    def tick_tree() -> bool:
        nonlocal tick_count
        try:
            logger.debug(f"--------- Tick {tick_count} ---------")
            start = timer()
            analysis_engine_root.tick()
            end = timer()
            logger.debug(f"Tick took {end - start:.4f} seconds")
            if (
                analysis_engine_root.root.children
                and analysis_engine_root.root.children[0].status == Status.FAILURE
            ):
                # If your top-level child fails, maybe shut down
                rclpy.shutdown()
                return False
            tick_count += 1
        except Exception as exception:
            logger.error(f"Exception: {exception}")
            logger.error("Traceback:\n" + traceback.format_exc())
        return True

    interval = 1.0 / tickrate
    last_tick = time.monotonic()

    while True:
        current_time = time.monotonic()
        elapsed = current_time - last_tick

        if elapsed >= interval:
            if not tick_tree():
                break

            last_tick = current_time
        else:
            time.sleep(interval - elapsed)


def main() -> None:
    """
    Entry point for the RoboKudo system, setting up ROS, parsing arguments, loading the
    requested Analysis Engine, and spinning the ROS executors.
    """
    # 1. Parse CLI arguments (prefix_chars='_'):
    parser = argparse.ArgumentParser(prefix_chars="_")
    parser.add_argument(
        "_ae",
        dest="ae",
        type=str,
        nargs="?",
        const=1,
        default="demo",
        help="Analysis Engine to run (module name in descriptors/analysis_engines/).",
    )
    parser.add_argument(
        "_ros_pkg",
        dest="ros_pkg",
        type=str,
        nargs="?",
        const=1,
        default=PACKAGE_NAME,
        help="ROS package name containing the analysis engine (default: robokudo).",
    )
    parser.add_argument(
        "_headless", action="store_true", help="If set, runs without a GUI."
    )
    parser.set_defaults(headless=False)
    parser.add_argument(
        "_no3d",
        action="store_true",
        help=(
            "If set, runs the GUI without the Open3D viewer, keeping the OpenCV and "
            "ROS visualizers. Use this if the Open3D viewer crashes in your display "
            "environment."
        ),
    )
    parser.set_defaults(no3d=False)
    parser.add_argument(
        "_nodesuffix",
        dest="nodesuffix",
        type=str,
        nargs="?",
        const=1,
        default="",
        help="A suffix to add to the ROS node name.",
    )
    parser.add_argument(
        "_tickrate",
        dest="tickrate",
        type=int,
        nargs="?",
        const=1,
        default=5,
        help="Rate (Hz) to tick the Behavior Tree.",
    )
    parser.add_argument(
        "_debugmode",
        action="store_true",
        help="If set, the rcply root logger will be set to DEBUG log level which will yield many ROS-related debug messages.",
    )
    parser.set_defaults(debugmode=False)
    args = parser.parse_args()

    # 2. Initialize RCL
    rclpy.init(args=sys.argv)

    if args.debugmode:
        rclpy.logging.set_logger_level(
            "", rclpy.impl.logging_severity.LoggingSeverity.DEBUG
        )

    # 3. Logging setup
    logger = logging.getLogger(LOGGING_IDENTIFIER_MAIN_EXECUTABLE)

    logging_configuration_file = (
        ModuleLoader.get_module_path(PACKAGE_NAME) / "logging_levels.yaml"
    )
    configure_logging(logging_config_file_name=str(logging_configuration_file))

    # 4. Create a main ROS node
    node_name = PACKAGE_NAME + args.nodesuffix
    node_registry = ROSNodeRegistry()
    node1 = rclpy.create_node(
        node_name,
        parameter_overrides=[
            Parameter(
                "default_snapshot_stream", rclpy.parameter.Parameter.Type.BOOL, True
            ),
            Parameter(
                "default_snapshot_period", rclpy.parameter.Parameter.Type.DOUBLE, 2.0
            ),
        ],
    )
    node_registry.register(node1)
    logger.info(f"Created node: {node_name}")

    # 5. Create any action servers or supporting nodes
    query_action_server = QueryActionServer(name="query")
    blackboard = Blackboard()
    blackboard.set(BlackboardIdentifier.QUERY_SERVER, query_action_server)
    blackboard.set(
        BlackboardIdentifier.QUERY_SERVER_IN_PIPELINE, False
    )  # Ownership in Pipeline has to be declared first

    # 6. Start executors in separate threads
    executor_main = SingleThreadedExecutor()
    action_server_executor = (
        MultiThreadedExecutor()
    )  # Necessary to handle long-running goals AND incoming preempts

    executor_main.add_node(node1)
    action_server_executor.add_node(query_action_server)

    def spin_executor(executor: rclpy.Executor) -> None:
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass

    thread_main = Thread(target=spin_executor, args=(executor_main,), daemon=True)
    action_server_thread = Thread(
        target=spin_executor, args=(action_server_executor,), daemon=True
    )
    thread_main.start()
    action_server_thread.start()

    # 7. Dynamically load the requested Analysis Engine (analysis engine) using the **refactored** ModuleLoader
    loader = ModuleLoader()
    logger.info(f"Loading analysis engine '{args.ae}' from package '{args.ros_pkg}'...")
    loaded_analysis_engine = loader.load_analysis_engine(
        ros_package_name=args.ros_pkg, module_name=args.ae
    )

    # 8. Build your Behavior Tree from the loaded analysis engine
    #    (Assuming loaded_analysis_engine.implementation() returns a py_trees root or something similar)
    visualizer_types = None
    if args.no3d:
        visualizer_types = [
            CVVisualizer,
            SharedROSVisualizer,
            AllAnnotatorROSVisualizer,
        ]
    analysis_engine_root = grow_tree(
        loaded_analysis_engine.implementation(),
        node=node1,
        include_gui=not args.headless,
        visualizer_types=visualizer_types,
    )

    # If you have a custom version of `setup_with_descendants`, call it:
    setup_with_descendants_robokudo(analysis_engine_root)

    try:
        # 9. Start ticking the Behavior Tree
        run_analysis_engine(
            analysis_engine_name=args.ae,
            analysis_engine_root=analysis_engine_root,
            tickrate=args.tickrate,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received; shutting down.")
    finally:
        # 10. Shutdown executors cleanly
        executor_main.shutdown()
        action_server_executor.shutdown()

        # 11. Wait for shutdown
        thread_main.join()
        action_server_thread.join()

        # 12. Clean up nodes
        node_registry.clear(node1)
        node1.destroy_node()
        query_action_server.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
