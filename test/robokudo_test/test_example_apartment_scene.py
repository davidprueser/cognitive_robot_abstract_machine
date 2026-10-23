"""
Tests for the apartment the example robot perceives in, as it really is and as the robot
believes it to be.
"""

from __future__ import annotations

import numpy as np
import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.examples.apartment_scene import ApartmentScene
from robokudo.examples.descriptors.worlds.apartment_reality import WorldDescriptor

# %% worlds


@pytest.fixture(scope="module")
def reality() -> World:
    """
    :return: The apartment as it really is.
    """
    return WorldDescriptor().world


@pytest.fixture(scope="module")
def believed_apartment() -> World:
    """
    :return: A world of a robot with only its odometry frame, after the apartment was
        described in it.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        odometry = Body(name=PrefixedName("odom"))
        world.add_body(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=odometry, world=world)
        )
    ApartmentScene().describe(world, odometry_frame="odom")
    return world


def position_of(world: World, body_name: str) -> np.ndarray:
    """
    :param world: The world holding the body.
    :param body_name: The name of the body.
    :return: The position of the body relative to the root of ``world``.
    """
    return world.get_body_by_name(body_name).global_transform.to_np()[:3, 3]


# %% reality


class TestReality:
    def test_reality_is_rooted_where_the_world_of_the_robot_is(
        self, reality: World
    ) -> None:
        assert reality.root.name == PrefixedName("map")

    def test_every_object_stands_at_its_place(self, reality: World) -> None:
        for item in ApartmentScene().items:
            assert np.allclose(position_of(reality, item.name)[:2], [item.x, item.y])


# %% the apartment as the robot believes it to be


class TestBelievedApartment:
    def test_robot_is_localized_in_front_of_the_shelf(
        self, believed_apartment: World
    ) -> None:
        assert np.allclose(
            believed_apartment.get_body_by_name("odom").global_transform.to_np(),
            ApartmentScene().robot_pose.to_np(),
        )

    def test_known_objects_are_misplaced_by_the_description_error(
        self, believed_apartment: World, reality: World
    ) -> None:
        scene = ApartmentScene()
        for item in scene.items:
            if item.semantic_class is None:
                continue
            assert np.allclose(
                position_of(believed_apartment, item.name),
                position_of(reality, item.name) + [scene.description_error, 0.0, 0.0],
            )

    def test_known_objects_are_described_as_their_class(
        self, believed_apartment: World
    ) -> None:
        for item in ApartmentScene().items:
            if item.semantic_class is None:
                continue
            [annotation] = believed_apartment.get_semantic_annotations_by_type(
                item.semantic_class
            )
            assert annotation.root is believed_apartment.get_body_by_name(item.name)

    def test_objects_only_reality_holds_are_not_described(
        self, believed_apartment: World
    ) -> None:
        for item in ApartmentScene().items:
            if item.semantic_class is not None:
                continue
            assert believed_apartment.get_bodies_by_name(item.name) == []
