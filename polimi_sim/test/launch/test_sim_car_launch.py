"""The gym's remaps name their node, stay inside the group, and polimi_sim starts no foxglove_bridge."""
import importlib.util
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
