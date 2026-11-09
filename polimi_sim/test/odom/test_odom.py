import math

import numpy as np
import pytest

from polimi_sim.odom.model import OdomConfig, OdomState, WheelOdometry

DT = 0.02
QUIET = dict(speed_scale_error=0.0, speed_noise_std=0.0)


def test_straight_line_shows_the_scale_error() -> None:
    odom = WheelOdometry(OdomConfig(speed_scale_error=0.02, speed_noise_std=0.0), np.random.default_rng(1))
    for _ in range(500):
        state = odom.update(DT, 2.0, 0.0)
    assert state.x == pytest.approx(2.0 * 10.0 * 1.02)
    assert (state.y, state.yaw) == (0.0, 0.0)
    assert state.speed == pytest.approx(2.04)


def test_constant_steering_gives_the_bicycle_circle() -> None:
    wheelbase, steer, speed = 0.33, 0.2, 1.5
    radius = wheelbase / math.tan(steer)
    odom = WheelOdometry(OdomConfig(wheelbase=wheelbase, **QUIET), np.random.default_rng(1))
    dt = 0.001
    points = []
    for _ in range(round(2.0 * math.pi * radius / speed / dt)):
        s = odom.update(dt, speed, steer)
        points.append((s.x, s.y))
    assert s.yaw_rate == pytest.approx(speed / radius)
    distance = np.hypot(np.array(points)[:, 0], np.array(points)[:, 1] - radius)
    np.testing.assert_allclose(distance, radius, rtol=2e-3)
    assert math.hypot(s.x, s.y) < 0.02  # back at the start after one lap


def test_deadband_keeps_a_parked_car_still() -> None:
    odom = WheelOdometry(OdomConfig(speed_noise_std=0.01), np.random.default_rng(1))
    for _ in range(1000):
        state = odom.update(DT, 0.0, 0.3)
    assert (state.x, state.y, state.yaw, state.speed) == (0.0, 0.0, 0.0, 0.0)


def test_noise_is_seeded() -> None:
    def run(seed: int) -> float:
        odom = WheelOdometry(OdomConfig(), np.random.default_rng(seed))
        for _ in range(200):
            state = odom.update(DT, 3.0, 0.1)
        return state.x
    assert run(5) == run(5)
    assert run(5) != run(6)


def test_reset_goes_back_to_the_origin() -> None:
    odom = WheelOdometry(OdomConfig(**QUIET), np.random.default_rng(1))
    for _ in range(100):
        odom.update(DT, 2.0, 0.2)
    assert odom.state.x != 0.0 and odom.state.yaw != 0.0
    odom.reset()
    assert odom.state == OdomState()


def test_config_is_validated() -> None:
    with pytest.raises(ValueError):
        OdomConfig(wheelbase=0.0)
