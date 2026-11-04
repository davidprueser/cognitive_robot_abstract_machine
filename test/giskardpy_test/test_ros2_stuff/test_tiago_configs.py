from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    WorldWithTiagoConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import DifferentialDrive

# %% speed limits of the base


def test_the_drive_takes_the_speed_limits_of_the_world_config():
    config = WorldWithTiagoConfig(urdf=load_xacro(Tiago.get_ros_file_path()))

    with config.world.modify_world():
        config.setup_world()

    drive = config.world.get_connections_by_type(DifferentialDrive)[0]
    assert drive.x_velocity.limits.upper.velocity == config.translation_velocity_limit
    assert drive.yaw.limits.upper.velocity == config.rotation_velocity_limit
