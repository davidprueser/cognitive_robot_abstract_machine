from __future__ import annotations
import numpy as np
import pytest

from robokudo.utils.region import (
    region_oriented_bounding_box,
    region_oriented_bounding_box_in_camera_coordinates,
    region_pose_annotation,
)
from robokudo.world_descriptor import BaseWorldDescriptor, RegionSpec
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Region
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


class TestUtilsRegion:
    @pytest.fixture
    def world_and_region(self) -> tuple[World, Region]:
        world_descriptor = BaseWorldDescriptor()
        root = world_descriptor.world.root
        region_specification = RegionSpec(
            name="test_region",
            box_scale=Scale(1.0, 2.0, 3.0),
            pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=3.0,
                pos_y=2.0,
                pos_z=1.0,
                quat_x=0.5,
                quat_y=0.5,
                quat_z=0.5,
                quat_w=0.5,
                reference_frame=root,
            ),
        )
        world_descriptor.build_regions(root, [region_specification])
        regions = world_descriptor.world.get_kinematic_structure_entity_by_type(Region)
        assert len(regions) == 1
        return world_descriptor.world, regions[0]

    def test_region_oriented_bounding_box(
        self, world_and_region: tuple[World, Region]
    ) -> None:
        world, region = world_and_region
        oriented_bounding_box = region_oriented_bounding_box(region, world=world)

        assert np.allclose(
            np.asarray(oriented_bounding_box.extent).reshape(-1)[:3], [1.0, 2.0, 3.0]
        )
        assert np.allclose(
            np.asarray(oriented_bounding_box.center).reshape(-1)[:3], [3.0, 2.0, 1.0]
        )
        assert np.allclose(
            oriented_bounding_box.R, [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

    def test_region_oriented_bounding_box_in_camera_coordinates(
        self, world_and_region: tuple[World, Region]
    ) -> None:
        world, region = world_and_region
        world_to_camera = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, 2.0],
                [0.0, -1.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        oriented_bounding_box = region_oriented_bounding_box_in_camera_coordinates(
            world=world, region=region, world_T_camera=world_to_camera
        )

        assert np.allclose(
            np.asarray(oriented_bounding_box.extent).reshape(-1)[:3], [1.0, 2.0, 3.0]
        )
        assert np.allclose(
            np.asarray(oriented_bounding_box.center).reshape(-1)[:3], [2.0, -1.0, 1.0]
        )
        assert np.allclose(
            oriented_bounding_box.R,
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]],
        )

    def test_region_pose_annotation(
        self, world_and_region: tuple[World, Region]
    ) -> None:
        _, region = world_and_region
        pose_annotation = region_pose_annotation(region)

        assert np.allclose(
            np.asarray(pose_annotation.translation).reshape(-1)[:3], [3.0, 2.0, 1.0]
        )
        # q and -q encode the same rotation, compare absolute values for robustness.
        assert np.allclose(
            np.abs(np.asarray(pose_annotation.rotation).reshape(-1)[:4]),
            [0.5, 0.5, 0.5, 0.5],
        )
