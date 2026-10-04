"""The simulator as the car: gym + LiDAR / actuation / odometry models + the car's mux and frames."""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from polimi_sim.common.gym_config import (
    build_gym_config,
    load_lidar,
    load_mount,
    load_yaml,
    write_gym_config,
)

# ideal:=true: the models pass everything through, to compare against the normal run.
# Safety behaviour (mux and VESC timeouts, steering clamp) and the 40 Hz scan rate stay.
IDEAL_LIDAR: dict[str, bool | int | float] = {'effects_enabled': False, 'blocks': 1, 'publish_delay': 0.0}
IDEAL_ACTUATION: dict[str, float] = {
    'command_delay': 0.0, 'servo_delay': 0.0, 'steer_offset': 0.0,
    'servo_rate': 1000.0, 'accel_max': 1000.0, 'brake_max': 1000.0,
}
IDEAL_ODOM: dict[str, float] = {'speed_scale_error': 0.0, 'speed_noise_std': 0.0, 'speed_deadband': 0.0}


def _setup(context: LaunchContext) -> list[LaunchDescriptionEntity]:
    def arg(name: str) -> str:
        return os.path.expanduser(LaunchConfiguration(name).perform(context))

    ideal = arg('ideal').lower() in ('true', '1')
    if arg('ideal').lower() not in ('true', '1', 'false', '0'):
        raise RuntimeError("ideal must be 'true' or 'false'")
    # Wheel odometry still drifts from tyre slip: in ideal mode draw everything on the truth.
    map_to_odom = 'truth' if ideal and arg('debug_map_to_odom') == 'off' else arg('debug_map_to_odom')

    # Independent files: which map, which laser, where the laser sits, and the gym's common part.
    config_dir = arg('config_dir')
    map_file = os.path.join(config_dir, 'maps', arg('map') + '.yaml')
    lidar_file = os.path.join(config_dir, 'lidar', arg('lidar') + '.yaml')
    mount_file = os.path.join(config_dir, 'car', 'laser_mount.yaml')
    gym_file = os.path.join(config_dir, 'sim', 'gym.yaml')
    for path in (map_file, lidar_file, mount_file, gym_file):
        if not os.path.isfile(path):
            raise RuntimeError(f'{path} not found')
    mount = load_mount(mount_file)
    gym_config = write_gym_config(
        build_gym_config(load_yaml(gym_file), load_yaml(map_file), load_lidar(lidar_file), mount),
        f"{arg('map')}_{arg('lidar')}",
    )

    gym = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py'
        )),
        launch_arguments={'config': gym_config, 'open_foxglove': 'false'}.items(),
    )
    lidar_model = Node(
        package='polimi_sim', executable='lidar_model', name='lidar_model', output='screen',
        parameters=[lidar_file, {
            'sim_config': gym_config,
            'mount_x': mount['x'],
            'mount_y': mount['y'],
            'mount_yaw': mount['yaw'],
        }, IDEAL_LIDAR if ideal else {}],
    )
    actuation_model = Node(
        package='polimi_sim', executable='actuation_model', name='actuation_model', output='screen',
        parameters=[arg('actuation_params'), IDEAL_ACTUATION if ideal else {}],
    )
    odom_model = Node(
        package='polimi_sim', executable='odom_model', name='odom_model', output='screen',
        parameters=[arg('odom_params'), {'debug_map_to_odom': map_to_odom}, IDEAL_ODOM if ideal else {}],
    )
    teleop_bridge = Node(
        package='polimi_sim', executable='teleop_bridge', name='teleop_bridge', output='screen',
        parameters=[arg('teleop_params')],
    )
    # The car's own mux and config, started as in f1tenth_stack's bringup. That remapping
    # matches nothing: the mux publishes /ackermann_cmd, which ackermann_to_vesc reads on the car.
    ackermann_mux = Node(
        package='ackermann_mux', executable='ackermann_mux', name='ackermann_mux',
        parameters=[os.path.join(get_package_share_directory('f1tenth_stack'), 'config', 'mux.yaml')],
        remappings=[('ackermann_cmd_out', 'ackermann_drive')],
    )
    static_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='static_baselink_to_laser',
        arguments=[
            '--x', str(mount['x']), '--y', str(mount['y']), '--z', str(mount['z']),
            '--yaw', str(mount['yaw']), '--roll', str(mount['roll']),
            '--frame-id', 'base_link', '--child-frame-id', 'laser',
        ],
    )
    return [gym, lidar_model, actuation_model, odom_model, teleop_bridge, ackermann_mux, static_tf]


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(get_package_share_directory('polimi_sim'), 'config')
    arguments = [
        ('map', 'spielberg', 'Map: a file name in <config_dir>/maps (spielberg | levine).'),
        ('lidar', 'sl450', 'Laser: a file name in <config_dir>/lidar.'),
        ('config_dir', '~/ws/config', 'Our config directory (ros2_ws/config).'),
        ('actuation_params', os.path.join(config, 'actuation_model.yaml'), 'actuation_model parameters.'),
        ('odom_params', os.path.join(config, 'odom_model.yaml'), 'odom_model parameters.'),
        ('teleop_params', os.path.join(config, 'teleop_bridge.yaml'), 'teleop_bridge parameters.'),
        ('ideal', 'false', 'true: no sensor or actuator imperfections, and debug_map_to_odom truth.'),
        ('debug_map_to_odom', 'off', 'map -> odom for debugging: off | static | truth. off with SLAM.'),
    ]
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in arguments],
        OpaqueFunction(function=_setup),
    ])
