"""Ray casting on the gym's map, with the gym's own ray caster."""
from __future__ import annotations

import numpy as np
import yaml
from f1tenth_gym.envs.lidar.laser_models import ScanSimulator2D, get_scan
from f1tenth_gym.envs.track import Track

from polimi_sim.common.se2 import Pose2D
from polimi_sim.lidar.effects import FloatArray, ScanGeometry


def load_gym_track(sim_config: str) -> Track:
    """The map a gym config loads, resolved and loaded like gym_bridge does."""
    # Private helpers of the pinned f1tenth_gym_ros: reused so both sides see the same map.
    from f1tenth_gym_ros.gym_bridge import _load_track_from_yaml, _resolve_map_yaml_path

    with open(sim_config) as config_file:
        params = yaml.safe_load(config_file)['bridge']['ros__parameters']
    map_path = str(params['map_path'])
    scale = float(params.get('scale', 1.0))

    yaml_path = _resolve_map_yaml_path(map_path)
    if yaml_path is None:
        return Track.from_track_name(map_path, track_scale=scale)
    try:
        return Track.from_track_path(yaml_path, track_scale=scale)
    except (ValueError, FileNotFoundError) as ex:
        if isinstance(ex, FileNotFoundError) or 'centerline' in str(ex) or 'raceline' in str(ex):
            track: Track = _load_track_from_yaml(yaml_path, scale)[0]
            return track
        raise


class BlockCaster:
    """Casts any contiguous block of a scan's beams from a given laser pose."""

    def __init__(self, track: Track, geometry: ScanGeometry, range_max: float) -> None:
        self._geometry: ScanGeometry = geometry
        self._sim: ScanSimulator2D = ScanSimulator2D(
            geometry.num_beams,
            geometry.angle_max - geometry.angle_min,
            angle_min=geometry.angle_min,
            angle_max=geometry.angle_max,
            std_dev=0.0,
            min_range=0.0,
            max_range=range_max,
        )
        self._sim.set_map(track)

    def cast(self, laser: Pose2D, start: int, stop: int) -> FloatArray:
        sim = self._sim
        ranges: FloatArray = get_scan(
            np.array([laser.x, laser.y, laser.yaw], dtype=np.float64),
            sim.theta_dis,
            self._geometry.angle_min + start * self._geometry.angle_increment,
            stop - start,
            sim.theta_index_increment,
            sim.sines,
            sim.cosines,
            sim.eps,
            sim.orig_x,
            sim.orig_y,
            sim.orig_c,
            sim.orig_s,
            sim.map_height,
            sim.map_width,
            sim.map_resolution,
            sim.dt,
            sim.max_range,
        )
        return ranges
