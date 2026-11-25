"""The gym's vehicle preset, as the bridge picks it from vehicle_params: lf, lr and the steering limit."""
from __future__ import annotations

from typing import NamedTuple


class Preset(NamedTuple):
    lf: float  # m, CoG to front axle
    lr: float  # m, CoG to rear axle (base_link)
    s_max: float  # rad


def vehicle_preset(name: str) -> Preset:
    from f1tenth_gym.envs.dynamic_models import (
        get_f1fifth_vehicle_parameters,
        get_f1tenth_vehicle_parameters,
        get_fullscale_vehicle_parameters,
    )

    getters = {
        'f1tenth': get_f1tenth_vehicle_parameters,
        'fullscale': get_fullscale_vehicle_parameters,
        'f1fifth': get_f1fifth_vehicle_parameters,
    }
    if name not in getters:
        raise ValueError(f'vehicle_params {name!r} is not one of {sorted(getters)}')
    vehicle = getters[name]()
    return Preset(float(vehicle.lf), float(vehicle.lr), float(vehicle.s_max))
