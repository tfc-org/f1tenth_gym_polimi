"""Needs f1tenth_gym and f1tenth_gym_ros (sim images only)."""
import os
import time

import numpy as np
import pytest
import yaml

pytest.importorskip('f1tenth_gym')
pytest.importorskip('f1tenth_gym_ros')

from polimi_sim.common.gym_config import (  # noqa: E402
    build_gym_config,
    load_lidar,
    load_mount,
    load_yaml,
)
from polimi_sim.common.se2 import Pose2D  # noqa: E402
from polimi_sim.lidar.effects import distorted_scan, scan_geometry  # noqa: E402
from polimi_sim.lidar.raycast import BlockCaster, load_gym_track  # noqa: E402

CONFIG = os.path.expanduser('~/ws/config')
pytestmark = pytest.mark.skipif(not os.path.isdir(CONFIG), reason='no ros2_ws/config')

GEOMETRY = scan_geometry(40.0, 270.0, 0.2)
MOUNT = Pose2D(0.27, 0.0, 0.0)
START = Pose2D(-12.0, 0.0, 0.0)


@pytest.fixture(scope='module')
def caster(tmp_path_factory: pytest.TempPathFactory) -> BlockCaster:
    gym_config = build_gym_config(
        load_yaml(f'{CONFIG}/sim/gym.yaml'),
        load_yaml(f'{CONFIG}/maps/levine.yaml'),
        load_lidar(f'{CONFIG}/lidar/sl450.yaml'),
        load_mount(f'{CONFIG}/car/laser_mount.yaml'),
    )
    path = tmp_path_factory.mktemp('gym') / 'gym.yaml'
    path.write_text(yaml.safe_dump(gym_config))
    return BlockCaster(load_gym_track(str(path)), GEOMETRY, 25.0)


def test_blocks_match_a_single_cast(caster: BlockCaster) -> None:
    plain = caster.cast(Pose2D(START.x + MOUNT.x, START.y, 0.0), 0, GEOMETRY.num_beams)
    blocks = distorted_scan(caster.cast, lambda t: START, MOUNT, GEOMETRY, 8, 0.0)
    assert plain.shape == (1351,) and 0.2 < plain.min() and plain.max() <= 25.0
    # The gym looks beam angles up in a table: a block start can round to the next entry.
    assert np.mean(np.abs(blocks - plain) > 1e-6) < 0.02


def test_cast_budget(caster: BlockCaster) -> None:
    distorted_scan(caster.cast, lambda t: START, MOUNT, GEOMETRY, 8, 0.0)  # numba compile
    runs = 200
    t0 = time.perf_counter()
    for i in range(runs):
        pose = Pose2D(START.x + 0.01 * i, 0.0, 0.0)
        distorted_scan(caster.cast, lambda t, pose=pose: pose, MOUNT, GEOMETRY, 8, 0.0)
    per_scan_ms = (time.perf_counter() - t0) / runs * 1e3
    print(f'\n8-block scan of 1351 beams: {per_scan_ms:.3f} ms')
    assert per_scan_ms < 5.0
