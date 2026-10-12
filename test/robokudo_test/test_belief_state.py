from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from krrood.adapters.json_serializer import from_json, to_json
from robokudo.belief_state import BeliefState
from robokudo.types.belief_state import ObjectBeliefState
from semantic_digital_twin.adapters.ros.messages import WorldUpdate
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% helpers


def add_object(
    world: World, parent: Body, origin: HomogeneousTransformationMatrix
) -> Connection6DoF:
    """
    Add a box that is free to move relative to ``parent`` and place it at ``origin``.
    """
    body = Body(name=PrefixedName(f"object_{uuid.uuid4().hex[:8]}"))
    body.visual.append(
        Box(
            origin=HomogeneousTransformationMatrix(reference_frame=body),
            scale=Scale(x=0.1, y=0.2, z=0.3),
        )
    )
    with world.modify_world():
        world.add_body(body)
        connection = Connection6DoF.create_with_dofs(
            world=world, parent=parent, child=body
        )
        world.add_connection(connection)
    connection.origin = origin
    return connection


def apply_world_update(update_json: str, world: World) -> None:
    """
    Apply a serialized :class:`WorldUpdate` the way a process receiving it would.
    """
    update: WorldUpdate = from_json(
        json.loads(update_json),
        **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
    )
    with world.modify_world():
        update.modification_block.modifications.apply(world)
    for degree_of_freedom_id, position in zip(
        update.state_update.ids, update.state_update.states
    ):
        world.state[degree_of_freedom_id].position = position
    world.notify_state_change()


def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    """
    Wait until ``condition`` holds or ``timeout`` seconds passed.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


@pytest.fixture
def belief_state() -> BeliefState:
    belief_state = BeliefState()
    belief_state.add_camera_frame(world_frame="map", camera_frame="camera")
    return belief_state


@pytest.fixture
def spinning_node(node: Node) -> Iterator[Node]:
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown()
    thread.join(timeout=2.0)


@dataclass
class SynchronizedWorlds:
    """
    The belief state of RoboKudo and the world of another process, kept in step.
    """

    belief_state: BeliefState
    other_world: World
    other_synchronizer: WorldSynchronizer


@pytest.fixture
def synchronized_worlds(spinning_node: Node) -> Iterator[SynchronizedWorlds]:
    topic_name = f"/robokudo_test/world_sync_{uuid.uuid4().hex}"
    belief_world = World(name="belief_world")
    belief_state = BeliefState(
        world=belief_world,
        synchronizer=WorldSynchronizer(
            node=spinning_node,
            _world=belief_world,
            topic_name=topic_name,
            defer_incoming_updates=True,
        ),
    )
    other_world = World(name="other_world")
    other_synchronizer = WorldSynchronizer(
        node=spinning_node, _world=other_world, topic_name=topic_name
    )
    yield SynchronizedWorlds(
        belief_state=belief_state,
        other_world=other_world,
        other_synchronizer=other_synchronizer,
    )
    belief_state.synchronizer.close()
    other_synchronizer.close()


# %% the changes of a perception run


def test_unchanged_run_has_no_changes(belief_state: BeliefState) -> None:
    snapshot = belief_state.start_run()

    update = belief_state.changes_since(snapshot)

    assert len(update.modification_block.modifications) == 0
    assert update.state_update.ids == []


def test_object_added_in_a_run_is_reproduced_from_its_changes(
    belief_state: BeliefState,
) -> None:
    world = belief_state.world
    world_body = world.get_body_by_name("map")
    other_world = copy.deepcopy(world)
    snapshot = belief_state.start_run()
    connection = add_object(
        world,
        world_body,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.5, y=-0.2, z=0.8, yaw=0.3, reference_frame=world_body
        ),
    )

    apply_world_update(
        json.dumps(to_json(belief_state.changes_since(snapshot))), other_world
    )

    other_body = other_world.get_kinematic_structure_entity_by_id(connection.child.id)
    assert other_body.visual.shapes[0].scale == connection.child.visual.shapes[0].scale
    np.testing.assert_allclose(
        other_world.compute_forward_kinematics_np(
            other_world.get_body_by_name("map"), other_body
        ),
        world.compute_forward_kinematics_np(world_body, connection.child),
        atol=1e-9,
    )


def test_object_moved_in_a_run_changes_only_its_moved_degrees_of_freedom(
    belief_state: BeliefState,
) -> None:
    world = belief_state.world
    world_body = world.get_body_by_name("map")
    connection = add_object(
        world,
        world_body,
        HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, reference_frame=world_body),
    )
    snapshot = belief_state.start_run()

    connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.4, reference_frame=world_body
    )
    update = belief_state.changes_since(snapshot)

    assert len(update.modification_block.modifications) == 0
    assert dict(
        zip(update.state_update.ids, update.state_update.states)
    ) == pytest.approx({connection.x.id: 0.4})


# %% camera


def test_camera_frame_is_added_once(belief_state: BeliefState) -> None:
    belief_state.add_camera_frame(world_frame="map", camera_frame="camera")

    world = belief_state.world
    assert len(world.get_bodies_by_name("camera")) == 1
    assert (
        len(
            world.get_connections_by_name(
                BeliefState.camera_connection_name("map", "camera")
            )
        )
        == 1
    )


def test_camera_pose_is_set_in_the_world(belief_state: BeliefState) -> None:
    world = belief_state.world
    world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, z=1.5, pitch=0.4
    )

    belief_state.set_camera_pose(
        world_frame="map", camera_frame="camera", world_T_camera=world_T_camera
    )

    np.testing.assert_allclose(
        world.compute_forward_kinematics_np(
            world.get_body_by_name("map"), world.get_body_by_name("camera")
        ),
        world_T_camera.to_np(),
        atol=1e-9,
    )


def test_camera_belonging_to_another_structure_keeps_its_place() -> None:
    belief_state = BeliefState()
    world = belief_state.world
    world_body = Body(name=PrefixedName("map"))
    robot_body = Body(name=PrefixedName("robot"))
    camera_body = Body(name=PrefixedName("head_camera"))
    with world.modify_world():
        world.add_body(world_body)
        world.add_body(robot_body)
        world.add_body(camera_body)
        world.add_connection(FixedConnection(parent=world_body, child=robot_body))
        world.add_connection(FixedConnection(parent=robot_body, child=camera_body))

    belief_state.set_camera_pose(
        world_frame="map",
        camera_frame="head_camera",
        world_T_camera=HomogeneousTransformationMatrix.from_xyz_rpy(x=2.0),
    )

    assert camera_body.parent_connection.parent is robot_body
    np.testing.assert_allclose(
        world.compute_forward_kinematics_np(world_body, camera_body), np.eye(4)
    )


# %% other processes


def test_updates_of_other_processes_are_applied_when_a_run_starts(
    synchronized_worlds: SynchronizedWorlds,
) -> None:
    belief_state = synchronized_worlds.belief_state
    other_body = Body(name=PrefixedName("other_body"))
    with synchronized_worlds.other_world.modify_world():
        synchronized_worlds.other_world.add_body(other_body)
    assert wait_until(lambda: len(belief_state.synchronizer.missed_messages) > 0)
    assert belief_state.world.find_world_entity_with_id(other_body.id) is None

    belief_state.start_run()

    assert belief_state.world.find_world_entity_with_id(other_body.id) is not None


def test_run_forgets_objects_another_process_removed(
    synchronized_worlds: SynchronizedWorlds,
) -> None:
    belief_state = synchronized_worlds.belief_state
    world = belief_state.world
    world_body = Body(name=PrefixedName("map"))
    with world.modify_world():
        world.add_body(world_body)
    connection = add_object(
        world,
        world_body,
        HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=world_body),
    )
    object_body = connection.child
    belief_state.tracked_objects[object_body.id] = ObjectBeliefState(body=object_body)
    other_world = synchronized_worlds.other_world
    assert wait_until(
        lambda: other_world.find_world_entity_with_id(object_body.id) is not None
    )

    other_object_body = other_world.get_kinematic_structure_entity_by_id(object_body.id)
    with other_world.modify_world():
        other_world.remove_connection(other_object_body.parent_connection)
        other_world.remove_kinematic_structure_entity(other_object_body)
    assert wait_until(lambda: len(belief_state.synchronizer.missed_messages) > 0)

    belief_state.start_run()

    assert belief_state.tracked_objects == {}


def test_changes_of_a_run_reach_other_processes_up_to_their_position(
    synchronized_worlds: SynchronizedWorlds,
) -> None:
    belief_state = synchronized_worlds.belief_state
    world = belief_state.world
    world_body = Body(name=PrefixedName("map"))
    with world.modify_world():
        world.add_body(world_body)
    snapshot = belief_state.start_run()
    connection = add_object(
        world,
        world_body,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.3, y=0.2, reference_frame=world_body
        ),
    )

    update = belief_state.changes_since(snapshot)

    assert wait_until(
        lambda: synchronized_worlds.other_synchronizer.has_applied(update.position)
    )
    other_world = synchronized_worlds.other_world
    np.testing.assert_allclose(
        other_world.compute_forward_kinematics_np(
            other_world.get_kinematic_structure_entity_by_id(world_body.id),
            other_world.get_kinematic_structure_entity_by_id(connection.child.id),
        ),
        world.compute_forward_kinematics_np(world_body, connection.child),
        atol=1e-9,
    )
