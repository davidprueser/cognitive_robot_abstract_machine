"""
Tree execution utilities for RoboKudo.

This module provides utilities for executing behavior trees in ROS environments.
It supports:

* One-shot tree execution with GUI option
* Rate-limited tree ticking
* Status monitoring and completion detection
* ROS time handling

Dependencies:

* py_trees for behavior tree functionality
* rospy for ROS integration
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import rclpy
from py_trees.blackboard import Blackboard
from py_trees.common import Status
from py_trees.decorators import OneShot
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from typing_extensions import TYPE_CHECKING, Optional

from robokudo.garden import grow_tree
from robokudo.utils.tree import setup_with_descendants_rk
from robokudo.identifier import BlackboardIdentifier

if TYPE_CHECKING:
    from py_trees.behaviour import Behaviour
    from py_trees_ros.trees import BehaviourTree
    from robokudo.annotators.query import QueryActionServer
    from robokudo.common_analysis_structure import CommonAnalysisStructure

_T = TypeVar("_T")


def _query_is_active() -> bool:
    """
    Check whether the current behavior tree is handling a query.
    """
    blackboard = Blackboard()
    if not blackboard.exists(BlackboardIdentifier.QUERY_SERVER):
        return False

    query_server: QueryActionServer = blackboard.get(BlackboardIdentifier.QUERY_SERVER)
    return query_server.is_active()


def _tick_tree_until(
    ae_root: BehaviourTree,
    node: Node,
    stop_condition: Callable[[int], Optional[_T]],
    tick_rate: int,
) -> Optional[_T]:
    """
    Tick an already setup tree until ``stop_condition`` returns a result.
    """
    tick_count = 0
    result = None
    tick_period_seconds = 1.0 / tick_rate
    executor = MultiThreadedExecutor(num_threads=2)
    executor_nodes = []

    try:
        executor.add_node(node)
        executor_nodes.append(node)

        # Fetch the Action Server, if present. It's a separate node.
        blackboard = Blackboard()
        if blackboard.exists(BlackboardIdentifier.QUERY_SERVER):
            query_server_node = blackboard.get(BlackboardIdentifier.QUERY_SERVER)
            executor.add_node(query_server_node)
            executor_nodes.append(query_server_node)

        # Spin only these nodes until we're done; don't touch global rclpy lifecycle.
        while result is None and rclpy.ok(context=node.context):
            tick_started_at = time.monotonic()
            ae_root.tick()
            tick_count += 1
            result = stop_condition(tick_count)

            if result is not None:
                break

            # print("spinning node", node.get_name(), id(node)) # in case of missed goal callbacks, debug spinned node
            executor.spin_once(timeout_sec=tick_period_seconds)
            elapsed_seconds = time.monotonic() - tick_started_at
            remaining_seconds = tick_period_seconds - elapsed_seconds
            if remaining_seconds > 0.0:
                time.sleep(remaining_seconds)

        return result
    finally:
        for executor_node in executor_nodes:
            executor.remove_node(executor_node)
        executor.shutdown()


def run_tree_once(
    tree: Behaviour,
    node: Node,
    include_gui: bool = False,
    max_iterations: int = 500,
    tick_rate: int = 5,
) -> Optional[Status]:
    """
    Execute a behavior tree once with monitoring.

    This function:
    * Grows the tree with optional GUI
    * Sets up all descendants
    * Ticks the tree at specified rate
    * Monitors for completion via OneShot decorator
    * Handles ROS time exceptions

    :param tree: Root behavior of tree to execute
    :param node: a ros node
    :param include_gui: Whether to show GUI visualization, defaults to False
    :param max_iterations: Maximum number of ticks before timeout, defaults to 500
    :param tick_rate: Rate to tick tree in Hz, defaults to 5
    :return: Final tree status (SUCCESS/FAILURE) or None if timed out
    """
    ae_root = grow_tree(tree, node, include_gui=include_gui, run_once=True)
    setup_with_descendants_rk(ae_root)

    assert isinstance(ae_root.root, OneShot)
    one_shot = ae_root.root

    def stop_condition(tick_count: int) -> Optional[Status]:
        if tick_count >= max_iterations:
            return one_shot.status

        if one_shot.status is Status.SUCCESS and _query_is_active():
            return None

        if one_shot.status in (Status.SUCCESS, Status.FAILURE):
            return one_shot.status
        return None

    return _tick_tree_until(ae_root, node, stop_condition, tick_rate)


def run_tree_until_successful_common_analysis_structure_count(
    tree: Behaviour,
    node: Node,
    expected_common_analysis_structure_count: int,
    include_gui: bool = False,
    max_iterations: int = 500,
    tick_rate: int = 20,
    on_successful_common_analysis_structure: Optional[
        Callable[[CommonAnalysisStructure], None]
    ] = None,
) -> list:
    """
    Execute a tree until a requested number of distinct successful common analysis structures exist.

    The tree is setup once without a ``OneShot`` wrapper so continuously running
    pipelines can process multiple percepts using the same ROS node.
    ``on_successful_common_analysis_structure`` is called immediately after each new successful common analysis structure is
    observed.
    """
    if expected_common_analysis_structure_count < 0:
        raise ValueError(
            "expected_common_analysis_structure_count must be non-negative"
        )
    if expected_common_analysis_structure_count == 0:
        return []

    ae_root = grow_tree(tree, node, include_gui=include_gui, run_once=False)
    setup_with_descendants_rk(ae_root)

    seen_common_analysis_structure_ids = set()
    successful_common_analysis_structure_instances = []

    def stop_condition(tick_count: int) -> Optional[list]:
        if tree.status is Status.SUCCESS:
            common_analysis_structure = tree.common_analysis_structure
            if (
                common_analysis_structure.common_analysis_structure_id
                not in seen_common_analysis_structure_ids
            ):
                seen_common_analysis_structure_ids.add(
                    common_analysis_structure.common_analysis_structure_id
                )
                successful_common_analysis_structure_instances.append(
                    common_analysis_structure
                )
                if on_successful_common_analysis_structure is not None:
                    on_successful_common_analysis_structure(common_analysis_structure)

        if (
            len(successful_common_analysis_structure_instances)
            >= expected_common_analysis_structure_count
        ):
            return successful_common_analysis_structure_instances

        if tick_count >= max_iterations:
            raise TimeoutError(
                f"Expected {expected_common_analysis_structure_count} successful common analysis structures, "
                f"got {len(successful_common_analysis_structure_instances)} after {tick_count} ticks"
            )

        return None

    result = _tick_tree_until(ae_root, node, stop_condition, tick_rate)
    return (
        result if result is not None else successful_common_analysis_structure_instances
    )
