"""The simulator as the car: gym + LiDAR / actuation / odometry models + the car's mux and frames."""
from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context: LaunchContext) -> list[LaunchDescriptionEntity]:
    def arg(name: str) -> str:
        return os.path.expanduser(LaunchConfiguration(name).perform(context))

    sim_config = arg('sim_config') or os.path.join(arg('config_dir'), 'sim', arg('map') + '.yaml')
    laser_mount = arg('laser_mount') or os.path.join(arg('config_dir'), 'car', 'laser_mount.yaml')
    for path in (sim_config, laser_mount):
        if not os.path.isfile(path):
            raise RuntimeError(f'{path} not found')
    with open(laser_mount) as mount_file:
        mount = {k: float(v) for k, v in yaml.safe_load(mount_file)['base_link_to_laser'].items()}

    gym = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py'
        )),
        launch_arguments={'config': sim_config, 'open_foxglove': 'false'}.items(),
    )
    lidar_model = Node(
        package='polimi_sim', executable='lidar_model', name='lidar_model', output='screen',
        parameters=[arg('lidar_params'), {
            'sim_config': sim_config,
            'mount_x': mount['x'],
            'mount_y': mount['y'],
            'mount_yaw': mount['yaw'],
        }],
    )
    actuation_model = Node(
        package='polimi_sim', executable='actuation_model', name='actuation_model', output='screen',
        parameters=[arg('actuation_params')],
    )
    odom_model = Node(
        package='polimi_sim', executable='odom_model', name='odom_model', output='screen',
        parameters=[arg('odom_params'), {
            'debug_map_to_odom': ParameterValue(
                LaunchConfiguration('debug_map_to_odom'), value_type=bool
            ),
        }],
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
        ('map', 'levine', 'Gym config name in <config_dir>/sim: levine | spielberg.'),
        ('sim_config', '', 'Gym config path. Overrides map.'),
        ('config_dir', '~/ws/config', 'Our config directory (ros2_ws/config).'),
        ('laser_mount', '', 'LiDAR mount file. Default: <config_dir>/car/laser_mount.yaml.'),
        ('lidar_params', os.path.join(config, 'lidar_model.yaml'), 'lidar_model parameters.'),
        ('actuation_params', os.path.join(config, 'actuation_model.yaml'), 'actuation_model parameters.'),
        ('odom_params', os.path.join(config, 'odom_model.yaml'), 'odom_model parameters.'),
        ('teleop_params', os.path.join(config, 'teleop_bridge.yaml'), 'teleop_bridge parameters.'),
        ('debug_map_to_odom', 'false', 'Static map -> odom at the true start pose. Never with SLAM.'),
    ]
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in arguments],
        OpaqueFunction(function=_setup),
    ])
