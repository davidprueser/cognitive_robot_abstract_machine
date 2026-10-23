from __future__ import annotations

import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_semdt_raytracer import (
    SemDTRayTracerCameraConfig,
)
from robokudo.descriptors.camera_configs.config_semdt_raytracer_robot_camera import (
    SemDTRayTracerRobotCameraConfig,
)
from robokudo.io.semdt_raytracer_camera_interface import (
    SemDTRayTracerCameraInterface,
    SemDTRayTracerRobotCameraInterface,
)

from ._perception_world import FrameName

# %% camera placement


class TestRenderingCameraPlacement:
    def test_rendering_camera_is_placed_in_the_world_perception_perceives_into(
        self,
    ) -> None:
        camera_config = SemDTRayTracerCameraConfig(
            world_descriptor_name="world_semdt_raytracer_cylinders", resolution=16
        )
        world = World()
        cas = CAS()

        SemDTRayTracerCameraInterface(camera_config).set_data(cas, world)

        camera_to_world_transform = cas.camera_to_world_transform
        assert camera_to_world_transform.child_frame is world.get_body_by_name(
            camera_config.camera_frame
        )
        assert np.allclose(
            camera_to_world_transform.to_np(),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                x=camera_config.camera_x,
                y=camera_config.camera_y,
                z=camera_config.camera_z,
                roll=camera_config.camera_roll,
                pitch=camera_config.camera_pitch,
                yaw=camera_config.camera_yaw,
            ).to_np(),
        )


# %% rendering from the robot camera


def world_with_robot_camera_at(
    world_T_camera: HomogeneousTransformationMatrix,
) -> World:
    """
    :param world_T_camera: The pose of the robot camera in the world frame.
    :return: A world whose camera body, as the robot that carries it would hold it, is
        at ``world_T_camera``.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName(FrameName.WORLD))
        camera = Body(name=PrefixedName(FrameName.CAMERA))
        world.add_body(root)
        world.add_connection(
            Connection6DoF.create_with_dofs(parent=root, child=camera, world=world)
        )
    camera.parent_connection.origin = HomogeneousTransformationMatrix(
        data=world_T_camera.to_np(), reference_frame=root
    )
    return world


class TestRenderingFromTheRobotCamera:
    def test_scene_is_rendered_from_where_the_world_holds_the_robot_camera(
        self,
    ) -> None:
        fixed_camera_config = SemDTRayTracerCameraConfig(resolution=16, camera_x=-1.0)
        world_T_camera = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=fixed_camera_config.camera_x,
            y=fixed_camera_config.camera_y,
            z=fixed_camera_config.camera_z,
            roll=fixed_camera_config.camera_roll,
            pitch=fixed_camera_config.camera_pitch,
            yaw=fixed_camera_config.camera_yaw,
        )
        fixed_camera_cas = CAS()
        SemDTRayTracerCameraInterface(fixed_camera_config).set_data(
            fixed_camera_cas, World()
        )
        robot_camera_cas = CAS()

        SemDTRayTracerRobotCameraInterface(
            SemDTRayTracerRobotCameraConfig(
                resolution=16, camera_frame=FrameName.CAMERA
            )
        ).set_data(robot_camera_cas, world_with_robot_camera_at(world_T_camera))

        assert np.array_equal(
            robot_camera_cas.get(CASViews.DEPTH_IMAGE),
            fixed_camera_cas.get(CASViews.DEPTH_IMAGE),
        )
