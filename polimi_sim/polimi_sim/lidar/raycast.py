"""Ray casting on the gym's map, with the gym's own ray caster."""
from __future__ import annotations

import pathlib

import numpy as np
import yaml
from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D
from f1tenth_gym.envs.track import Track

from polimi_sim.common.se2 import Pose2D
from polimi_sim.lidar.effects import FloatArray, ScanGeometry, block_bounds


def load_gym_track(sim_config: str) -> Track:
    """The map a gym config loads, resolved and loaded like gym_bridge does."""
    # Private helpers of the pinned f1tenth_gym_ros: reused so both sides see the same map.
    from f1tenth_gym_ros.gym_bridge import _resolve_map_yaml_path

    with open(sim_config) as config_file:
        params = yaml.safe_load(config_file)['bridge']['ros__parameters']
    map_path = str(params['map_path'])
    scale = float(params.get('scale', 1.0))

    yaml_path = _resolve_map_yaml_path(map_path)
    if yaml_path is None:
        return Track.from_track_name(map_path, track_scale=scale)
    return load_map_track(yaml_path, scale)


def load_map_track(yaml_path: pathlib.Path, scale: float = 1.0) -> Track:
    """A map_server yaml loaded like gym_bridge does, with or without a centre line."""
    from f1tenth_gym_ros.gym_bridge import _load_track_from_yaml

    try:
        return Track.from_track_path(yaml_path, track_scale=scale)
    except (ValueError, FileNotFoundError) as ex:
        if isinstance(ex, FileNotFoundError) or 'centerline' in str(ex) or 'raceline' in str(ex):
            track: Track = _load_track_from_yaml(yaml_path, scale)[0]
            return track
        raise


class BlockCaster:
    """Casts the fixed blocks of a scan's beams from a given laser pose, one gym scanner per block."""

    def __init__(self, track: Track, geometry: ScanGeometry, range_max: float, blocks: int) -> None:
        self._sims: dict[tuple[int, int], SegmentScanSimulator2D] = {}
        for start, stop in block_bounds(geometry.num_beams, blocks):
            sim = SegmentScanSimulator2D(
                stop - start,
                (stop - 1 - start) * geometry.angle_increment,
                angle_min=geometry.angle_min + start * geometry.angle_increment,
                angle_max=geometry.angle_min + (stop - 1) * geometry.angle_increment,
                std_dev=0.0,
                min_range=0.0,
                max_range=range_max,
            )
            sim.set_map(track)
            sim.scan(np.zeros(3))
            self._sims[(start, stop)] = sim

    def cast(self, laser: Pose2D, start: int, stop: int) -> FloatArray:
        ranges: FloatArray = self._sims[(start, stop)].scan(np.array([laser.x, laser.y, laser.yaw]))
        return ranges
