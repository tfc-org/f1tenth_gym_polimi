"""The gym's remaps name their node, stay inside the group, and polimi_sim starts no foxglove_bridge."""
import importlib.util
import math
import pathlib
from types import ModuleType

import pytest

pytest.importorskip('launch_ros')

from launch import LaunchContext  # noqa: E402
from launch.actions import LogInfo, PopLaunchConfigurations  # noqa: E402
from launch_ros.actions import SetRemap  # noqa: E402

LAUNCH_FILE = pathlib.Path(__file__).parents[2] / 'launch' / 'sim_car.launch.py'


def _launch_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('sim_car_launch', LAUNCH_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_gym_remaps_name_their_node_and_stay_in_the_group() -> None:
    gym = LogInfo(msg='the gym include')
    entities = _launch_module().with_gym_remaps(gym).get_sub_entities()
    context = LaunchContext()
    for entity in entities:
        if isinstance(entity, SetRemap):
            entity.execute(context)
    assert context.launch_configurations['ros_remaps'] == [
        ('bridge:/initialpose', '/sim/initialpose'),
        ('map_server:/map', '/sim/map'),
    ]
    assert gym in entities
    assert isinstance(entities[-1], PopLaunchConfigurations)


def test_polimi_sim_starts_no_foxglove_bridge_of_its_own() -> None:
    assert "package='foxglove_bridge'" not in LAUNCH_FILE.read_text()


def test_the_camera_tfs_use_the_mount_and_the_optical_rotation_the_drawing_uses() -> None:
    mount = {'x': 0.2, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': 0.2618, 'yaw': 0.0}
    mount_args, optical_args = _launch_module().camera_tf_arguments(mount, (0.0, -0.02375, 0.0))
    assert mount_args == ['--x', '0.2', '--y', '0.0', '--z', '0.2', '--roll', '0.0', '--pitch', '0.2618', '--yaw', '0.0',
                          '--frame-id', 'base_link', '--child-frame-id', 'camera_link']
    half_pi = str(-math.pi / 2)
    assert optical_args == ['--x', '0.0', '--y', '-0.02375', '--z', '0.0', '--roll', half_pi, '--pitch', '0.0', '--yaw', half_pi,
                            '--frame-id', 'camera_link', '--child-frame-id', 'camera_color_optical_frame']
