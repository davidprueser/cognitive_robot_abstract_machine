"""
Tests for the example server of a shared world for RoboKudo to join.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from rclpy.node import Node
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.examples.serve_shared_world import SharedWorldServer

# %% served world


@pytest.fixture
def server(node: Node) -> Iterator[SharedWorldServer]:
    """
    :return: A server of a shared world, closed after the test.
    """
    served_world = SharedWorldServer(node=node)
    yield served_world
    served_world.close()


# %% tf tree


class TestTfTree:
    def test_body_named_like_a_frame_of_the_robot_is_published_under_another_frame(
        self, server: SharedWorldServer
    ) -> None:
        world = server.world
        camera = Body(name=PrefixedName("head_mount_kinect_rgb_optical_frame"))
        with world.modify_world():
            world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=world, parent=world.root, child=camera
                )
            )

        frame_names = server.marker_publisher.tf_publisher.frame_names
        assert frame_names.assign(camera) == f"{server.tf_frame_prefix}{camera.name}"
