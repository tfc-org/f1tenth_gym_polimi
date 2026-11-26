import math

import pytest

from polimi_sim.teleop.conversion import twist_to_ackermann

ARGS = dict(wheelbase=0.33, max_steer=0.4189, min_speed=0.5)


def test_steering_angle_mode_clamps() -> None:
    assert twist_to_ackermann(1.0, 0.2, 'steering_angle', **ARGS) == (1.0, 0.2)
    assert twist_to_ackermann(-1.0, -1.0, 'steering_angle', **ARGS) == (-1.0, -0.4189)


def test_yaw_rate_mode() -> None:
    speed, steer = twist_to_ackermann(2.0, 1.0, 'yaw_rate', **ARGS)
    assert speed == 2.0 and steer == pytest.approx(math.atan(0.33 / 2.0))
    assert twist_to_ackermann(-2.0, 1.0, 'yaw_rate', **ARGS)[1] == pytest.approx(-math.atan(0.33 / 2.0))
    assert twist_to_ackermann(0.0, 0.5, 'yaw_rate', **ARGS)[1] == pytest.approx(math.atan(0.33 * 0.5 / 0.5))


def test_unknown_mode() -> None:
    with pytest.raises(ValueError):
        twist_to_ackermann(1.0, 0.0, 'nope', **ARGS)


@pytest.mark.parametrize('linear_x, angular_z', [(1.0, math.nan), (math.nan, 0.1), (1.0, math.inf)])
def test_a_non_finite_command_stops_instead_of_full_lock(linear_x: float, angular_z: float) -> None:
    for mode in ('steering_angle', 'yaw_rate'):
        assert twist_to_ackermann(linear_x, angular_z, mode, 0.33, 0.4189, 0.5) == (0.0, 0.0)
