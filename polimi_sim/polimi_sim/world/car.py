"""The car's numbers for make_world, from a config directory over the package's defaults."""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory

from polimi_sim.common.gym_config import load_body, load_mount, load_yaml
from polimi_sim.common.overlay import resolve
from polimi_sim.common.vehicle import vehicle_preset
from polimi_sim.world.generate import Car


def load_car(config_dir: str) -> Car:
    defaults = os.path.join(get_package_share_directory('polimi_sim'), 'config', 'defaults')
    body = load_body(resolve('car/body.yaml', config_dir, defaults).path)
    mount = load_mount(resolve('car/laser_mount.yaml', config_dir, defaults).path)
    gym = load_yaml(resolve('sim/gym.yaml', config_dir, defaults).path)
    preset = vehicle_preset(str(gym['bridge']['ros__parameters']['vehicle_params']))
    return Car(preset.lf + preset.lr, preset.s_max, body['steer_max_left'], body['steer_max_right'], mount['z'])
