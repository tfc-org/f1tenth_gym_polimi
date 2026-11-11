"""The simulator as the car: gym + LiDAR / actuation / odometry models + the car's mux and frames."""
from __future__ import annotations

import os

import launch.logging
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_entity import LaunchDescriptionEntity
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap

from polimi_sim.common.gym_config import (
    build_gym_config,
    load_body,
    load_lidar,
    load_mount,
    load_yaml,
    resolve_relative,
    write_gym_config,
)
from polimi_sim.common.overlay import report, resolve
from polimi_sim.common.vehicle import vehicle_preset

# ideal:=true: the models pass everything through, to compare against the normal run.
# Safety behaviour (mux and VESC timeouts, steering clamp) and the 40 Hz scan rate stay.
IDEAL_LIDAR: dict[str, bool | int | float] = {'effects_enabled': False, 'blocks': 1, 'publish_delay': 0.0}
IDEAL_ACTUATION: dict[str, float] = {
    'command_delay': 0.0, 'servo_delay': 0.0, 'steer_offset': 0.0,
    'servo_rate': 1000.0, 'accel_max': 1000.0, 'brake_max': 1000.0,
}
IDEAL_ODOM: dict[str, float] = {'speed_scale_error': 0.0, 'speed_noise_std': 0.0, 'speed_deadband': 0.0}
# /map and /initialpose belong to the stack (SLAM, AMCL). Each rule names its node, so it never
# reaches the foxglove_bridge that the gym's launch starts.
GYM_REMAPS: tuple[tuple[str, str], ...] = (
    ('bridge:/initialpose', '/sim/initialpose'),
    ('map_server:/map', '/sim/map'),
)


def with_gym_remaps(gym: LaunchDescriptionEntity) -> GroupAction:
    """The gym's include, scoped with the node-scoped remaps of GYM_REMAPS."""
    return GroupAction([*(SetRemap(src=src, dst=dst) for src, dst in GYM_REMAPS), gym])


def _setup(context: LaunchContext) -> list[LaunchDescriptionEntity]:
    def arg(name: str) -> str:
        return os.path.expanduser(LaunchConfiguration(name).perform(context))

    ideal = arg('ideal').lower() in ('true', '1')
    if arg('ideal').lower() not in ('true', '1', 'false', '0'):
        raise RuntimeError("ideal must be 'true' or 'false'")
    # Wheel odometry still drifts from tyre slip: in ideal mode draw everything on the truth.
    map_to_odom = 'truth' if ideal and arg('debug_map_to_odom') == 'off' else arg('debug_map_to_odom')

    # Independent files: which map, which laser, where the laser sits, the gym's common part and
    # the models. Each comes from config_dir if it has it, else from the package's defaults.
    config_dir = arg('config_dir')
    world_source = os.path.join(arg('maps_dir'), 'src', f"{arg('map')}.yaml")
    if config_dir and os.path.isfile(world_source):
        # A room world: regenerate it at every start (fast and byte-identical), so an edit needs no extra step.
        from polimi_sim.world.car import load_car
        from polimi_sim.world.generate import generate

        try:
            for line in generate(world_source, config_dir, load_car(config_dir)):
                launch.logging.get_logger('make_world').info(line)
        except ValueError as ex:
            raise RuntimeError(f'make_world {world_source}: {ex}') from ex
    defaults = os.path.join(get_package_share_directory('polimi_sim'), 'config', 'defaults')
    relative = {
        'map': f"maps/{arg('map')}.yaml", 'lidar': f"lidar/{arg('lidar')}.yaml",
        'mount': 'car/laser_mount.yaml', 'body': 'car/body.yaml', 'gym': 'sim/gym.yaml',
        'actuation': 'actuation_model.yaml', 'odom': 'odom_model.yaml', 'teleop': 'teleop_bridge.yaml',
    }
    overrides = {'actuation': arg('actuation_params'), 'odom': arg('odom_params'), 'teleop': arg('teleop_params')}
    try:
        files = {k: resolve(rel, config_dir, defaults) for k, rel in relative.items() if not overrides.get(k)}
    except FileNotFoundError as ex:
        raise RuntimeError(str(ex)) from ex
    info, warnings = report(list(files.values()), config_dir)
    logger = launch.logging.get_logger('sim_car')
    for line in info:
        logger.info(line)
    for line in warnings:
        logger.warning(line)
    for key, path in overrides.items():
        if path:
            logger.info(f'config: {relative[key]} <- {path} (argument)')
    path_of = {**{k: f.path for k, f in files.items()}, **{k: v for k, v in overrides.items() if v}}
    map_file, lidar_file, mount_file, gym_file = (path_of[k] for k in ('map', 'lidar', 'mount', 'gym'))
    mount = load_mount(mount_file)
    body = load_body(path_of['body'])
    map_dir = os.path.dirname(map_file)
    world = {k: resolve_relative(str(v), map_dir) for k, v in load_yaml(map_file).get('world', {}).items()}
    gym_params = build_gym_config(load_yaml(gym_file), load_yaml(map_file), load_lidar(lidar_file), mount, map_dir)
    preset = vehicle_preset(str(gym_params['bridge']['ros__parameters']['vehicle_params']))
    gym_config = write_gym_config(gym_params, f"{arg('map')}_{arg('lidar')}")

    gym = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge_launch.py'
        )),
        launch_arguments={
            'config': gym_config, 'open_foxglove': 'false',
            'foxglove_layout': os.path.join(
                get_package_share_directory('polimi_sim'), 'config', 'foxglove', 'polimi_sim.json'
            ),
        }.items(),
    )
    lidar_model = Node(
        package='polimi_sim', executable='lidar_model', name='lidar_model', output='screen',
        parameters=[lidar_file, {
            'sim_config': gym_config,
            'lidar_map': str(world.get('room_lidar', '')),
            'mount_x': mount['x'],
            'mount_y': mount['y'],
            'mount_yaw': mount['yaw'],
        }, IDEAL_LIDAR if ideal else {}],
    )
    actuation_model = Node(
        package='polimi_sim', executable='actuation_model', name='actuation_model', output='screen',
        parameters=[path_of['actuation'], {
            'max_steer_left': body['steer_max_left'], 'max_steer_right': body['steer_max_right'],
        }, IDEAL_ACTUATION if ideal else {}],
    )
    odom_model = Node(
        package='polimi_sim', executable='odom_model', name='odom_model', output='screen',
        parameters=[path_of['odom'], {'debug_map_to_odom': map_to_odom, 'wheelbase': preset.lf + preset.lr}, IDEAL_ODOM if ideal else {}],
    )
    teleop_bridge = Node(
        package='polimi_sim', executable='teleop_bridge', name='teleop_bridge', output='screen',
        parameters=[path_of['teleop']],
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
    return [with_gym_remaps(gym), lidar_model, actuation_model, odom_model, teleop_bridge, ackermann_mux, static_tf]


def generate_launch_description() -> LaunchDescription:
    arguments = [
        ('map', 'spielberg', 'Map: a file name in maps/ (defaults: spielberg | levine).'),
        ('lidar', 'sl450', 'Laser: a file name in lidar/.'),
        ('config_dir', '', "A car's config directory; each file it lacks comes from the defaults."),
        ('maps_dir', '~/ws/maps', 'Room worlds: <maps_dir>/src/<map>.yaml is regenerated into config_dir at start.'),
        ('actuation_params', '', 'actuation_model parameter file; empty = actuation_model.yaml from the overlay.'),
        ('odom_params', '', 'odom_model parameter file; empty = odom_model.yaml from the overlay.'),
        ('teleop_params', '', 'teleop_bridge parameter file; empty = teleop_bridge.yaml from the overlay.'),
        ('ideal', 'false', 'true: no sensor or actuator imperfections, and debug_map_to_odom truth.'),
        ('debug_map_to_odom', 'off', 'map -> odom for debugging: off | static | truth. off with SLAM.'),
    ]
    return LaunchDescription([
        *[DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in arguments],
        OpaqueFunction(function=_setup),
    ])
