from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS
from robokudo.exceptions import SharedWorldReplacementError
from robokudo.perception_belief_state_context import PerceptionBeliefStateContext

from ._object_hypotheses import make_object_hypothesis

WORLD_FRAME = "map"
"""
Frame the camera pose is given in.
"""

CAMERA_FRAME = "camera"
"""
Frame of the camera.
"""

SYNCHRONIZATION_TIMEOUT = timedelta(seconds=5)
"""
How long a change may take to travel between synchronized worlds.
"""


def wait_until(condition: Callable[[], bool]) -> bool:
    """
    Wait for ``condition`` to hold, at most :data:`SYNCHRONIZATION_TIMEOUT`.

    :param condition: The condition to wait for.
    :return: Whether the condition held in time.
    """
    deadline = time.monotonic() + SYNCHRONIZATION_TIMEOUT.total_seconds()
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def camera_pose(x: float = 0.0) -> HomogeneousTransformationMatrix:
    """
    :param x: Position of the camera along the world x axis.
    :return: A camera pose looking along the world frame axes.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(x=x)


def cas_seen_from(
    context: PerceptionBeliefStateContext,
    world_T_camera: HomogeneousTransformationMatrix,
) -> CAS:
    """
    A CAS whose camera is placed at ``world_T_camera`` in the world of ``context``.

    :param context: The context the camera is placed in.
    :param world_T_camera: Pose of the camera in the world frame.
    :return: The CAS.
    """
    cas = CAS()
    cas.world_frame = WORLD_FRAME
    cas.camera_frame = CAMERA_FRAME
    cas.camera_to_world_transform = context.place_camera(
        WORLD_FRAME, CAMERA_FRAME, world_T_camera
    )
    return cas


# %% blackboard


class TestContextOnBlackboard:
    def test_creates_and_stores_a_context_when_none_is_stored(self) -> None:
        context = PerceptionBeliefStateContext.from_blackboard()

        assert PerceptionBeliefStateContext.from_blackboard() is context
        assert not context.is_synchronized

    def test_returns_the_stored_context(self) -> None:
        context = PerceptionBeliefStateContext()
        context.store_on_blackboard()

        assert PerceptionBeliefStateContext.from_blackboard() is context


# %% world replacement


class TestWorldReplacement:
    def test_replaced_world_drops_the_beliefs_held_in_the_previous_one(self) -> None:
        context = PerceptionBeliefStateContext()
        context.add_object_hypothesis_as_belief_state(make_object_hypothesis(), CAS())
        new_world = World()

        context.replace_world(new_world)

        assert context.world is new_world
        assert context.object_belief_states == {}


# %% camera placement


class TestCameraPlacement:
    def test_adds_missing_frames_and_moves_the_camera(self) -> None:
        context = PerceptionBeliefStateContext()
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=2.0, z=3.0
        )

        bound_world_T_camera = context.place_camera(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        world_body = context.world.get_body_by_name(WORLD_FRAME)
        camera_body = context.world.get_body_by_name(CAMERA_FRAME)
        assert context.world.root is world_body
        assert bound_world_T_camera.reference_frame is world_body
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_moves_a_placed_camera_again(self) -> None:
        context = PerceptionBeliefStateContext()
        context.place_camera(WORLD_FRAME, CAMERA_FRAME, camera_pose(x=1.0))
        world_T_camera = camera_pose(x=2.0)

        context.place_camera(WORLD_FRAME, CAMERA_FRAME, world_T_camera)

        camera_body = context.world.get_body_by_name(CAMERA_FRAME)
        assert np.allclose(camera_body.global_transform.to_np(), world_T_camera.to_np())

    def test_leaves_a_camera_located_by_another_kinematic_structure_in_place(
        self,
    ) -> None:
        world = World()
        world_body = Body(name=PrefixedName(WORLD_FRAME))
        head_body = Body(name=PrefixedName("head"))
        camera_body = Body(name=PrefixedName(CAMERA_FRAME))
        with world.modify_world():
            world.add_body(world_body)
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=world_body, child=head_body, world=world
                )
            )
            world.add_connection(FixedConnection(parent=head_body, child=camera_body))
        head_body.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                z=1.5, reference_frame=world_body
            )
        )
        camera_pose_before = camera_body.global_transform.to_np()
        context = PerceptionBeliefStateContext(world=world)
        world_T_camera = camera_pose(x=4.0)

        bound_world_T_camera = context.place_camera(
            WORLD_FRAME, CAMERA_FRAME, world_T_camera
        )

        assert np.allclose(camera_body.global_transform.to_np(), camera_pose_before)
        assert bound_world_T_camera.child_frame is camera_body
        assert np.allclose(bound_world_T_camera.to_np(), world_T_camera.to_np())


# %% object beliefs


class TestObjectBeliefs:
    def test_new_belief_is_a_box_at_the_world_pose_of_its_hypothesis(self) -> None:
        context = PerceptionBeliefStateContext()
        cas = cas_seen_from(context, camera_pose(x=1.0))
        hypothesis = make_object_hypothesis(translation=(0.0, 0.5, 2.0))

        object_belief = context.add_object_hypothesis_as_belief_state(hypothesis, cas)

        bounding_box = object_belief.latest_bbox_3d
        assert context.object_belief_states == {object_belief.uuid: object_belief}
        assert object_belief.latest_hypothesis is hypothesis
        assert np.allclose(
            object_belief.body.global_transform.position.to_np()[:3],
            np.array([1.0, 0.0, 0.0]) + np.array(bounding_box.pose.translation),
        )
        assert [shape.scale for shape in object_belief.body.visual.shapes] == [
            Scale(bounding_box.x_length, bounding_box.y_length, bounding_box.z_length)
        ]

    def test_belief_without_world_frame_is_kept_outside_the_world(self) -> None:
        context = PerceptionBeliefStateContext()

        object_belief = context.add_object_hypothesis_as_belief_state(
            make_object_hypothesis(), CAS()
        )

        assert context.object_belief_states == {object_belief.uuid: object_belief}
        assert object_belief.body not in context.world.bodies

    def test_updated_belief_moves_to_and_takes_the_shape_of_the_new_hypothesis(
        self,
    ) -> None:
        context = PerceptionBeliefStateContext()
        cas = cas_seen_from(context, camera_pose())
        object_belief = context.add_object_hypothesis_as_belief_state(
            make_object_hypothesis(), cas
        )
        hypothesis = make_object_hypothesis(
            translation=(0.3, 0.0, 1.0), extent=(0.4, 0.5, 0.6)
        )

        context.update_belief_state_with_object_hypothesis(
            object_belief, hypothesis, cas
        )

        bounding_box = object_belief.latest_bbox_3d
        assert object_belief.latest_hypothesis is hypothesis
        assert np.allclose(
            object_belief.body.global_transform.position.to_np()[:3],
            bounding_box.pose.translation,
        )
        assert [shape.scale for shape in object_belief.body.visual.shapes] == [
            Scale(bounding_box.x_length, bounding_box.y_length, bounding_box.z_length)
        ]


# %% synchronization with a shared world


@dataclass
class SharedWorld:
    """
    The world another process serves and keeps synchronized, such as the world of the
    process executing robot actions.
    """

    world: World
    """
    The served world.
    """

    server: FetchWorldServer
    """
    Serves the world to processes that join.
    """

    synchronizer: WorldSynchronizer
    """
    Keeps the served world in sync with the processes that joined.
    """

    def close(self) -> None:
        """
        Stop serving and synchronizing the world.
        """
        self.synchronizer.close()
        self.server.close()


@pytest.fixture
def shared_world(spinning_node: Node) -> Iterator[SharedWorld]:
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName(WORLD_FRAME)))
    served_world = SharedWorld(
        world=world,
        server=FetchWorldServer(node=spinning_node, world=world),
        synchronizer=WorldSynchronizer(node=spinning_node, _world=world),
    )
    yield served_world
    served_world.close()


@pytest.fixture
def synchronized_context(
    spinning_node: Node, shared_world: SharedWorld
) -> Iterator[PerceptionBeliefStateContext]:
    context = PerceptionBeliefStateContext.synchronized_with_shared_world(spinning_node)
    yield context
    context.close()


class TestSynchronizationWithSharedWorld:
    def test_starts_from_the_shared_world(
        self,
        shared_world: SharedWorld,
        synchronized_context: PerceptionBeliefStateContext,
    ) -> None:
        assert synchronized_context.is_synchronized
        assert [body.id for body in synchronized_context.world.bodies] == [
            body.id for body in shared_world.world.bodies
        ]

    def test_changes_of_the_shared_world_reach_perception(
        self,
        shared_world: SharedWorld,
        synchronized_context: PerceptionBeliefStateContext,
    ) -> None:
        table = Body(name=PrefixedName("table"))
        with shared_world.world.modify_world():
            shared_world.world.add_connection(
                Connection6DoF.create_with_dofs(
                    parent=shared_world.world.root,
                    child=table,
                    world=shared_world.world,
                )
            )

        assert wait_until(
            lambda: table.id in [body.id for body in synchronized_context.world.bodies]
        )

    def test_perceived_object_reaches_the_shared_world_at_its_pose(
        self,
        shared_world: SharedWorld,
        synchronized_context: PerceptionBeliefStateContext,
    ) -> None:
        cas = cas_seen_from(synchronized_context, camera_pose(x=1.0))

        object_belief = synchronized_context.add_object_hypothesis_as_belief_state(
            make_object_hypothesis(translation=(0.0, 0.5, 2.0)), cas
        )

        expected_pose = object_belief.body.global_transform.to_np()
        assert wait_until(
            lambda: object_belief.uuid
            in [body.id for body in shared_world.world.bodies]
            and np.allclose(
                shared_world.world.get_kinematic_structure_entity_by_id(
                    object_belief.uuid
                ).global_transform.to_np(),
                expected_pose,
            )
        )
        shared_body = shared_world.world.get_kinematic_structure_entity_by_id(
            object_belief.uuid
        )
        assert [shape.scale for shape in shared_body.visual.shapes] == [
            shape.scale for shape in object_belief.body.visual.shapes
        ]

    def test_shared_world_cannot_be_replaced(
        self, synchronized_context: PerceptionBeliefStateContext
    ) -> None:
        with pytest.raises(SharedWorldReplacementError):
            synchronized_context.replace_world(World())

    def test_closed_context_no_longer_shares_its_world(
        self, synchronized_context: PerceptionBeliefStateContext
    ) -> None:
        synchronized_context.close()

        assert not synchronized_context.is_synchronized
