"""Twist -> Ackermann conversion for teleop. No ROS imports."""
from __future__ import annotations

import math

ANGULAR_MODES: tuple[str, ...] = ('steering_angle', 'yaw_rate')


def twist_to_ackermann(
    linear_x: float,
    angular_z: float,
    angular_mode: str,
    wheelbase: float,
    max_steer: float,
    min_speed: float,
) -> tuple[float, float]:
    """Return (speed, steering angle).

    steering_angle: angular_z is the steering angle in rad.
    yaw_rate: angular_z is a yaw rate; steering = atan(L * w / v), with |v| held
    at `min_speed` or more so a slow car doesn't snap to full lock.
    """
    if angular_mode == 'steering_angle':
        steer = angular_z
    elif angular_mode == 'yaw_rate':
        speed = math.copysign(max(abs(linear_x), min_speed), linear_x)
        steer = math.atan(wheelbase * angular_z / speed)
    else:
        raise ValueError(f'angular_mode must be one of {ANGULAR_MODES}, got {angular_mode!r}')
    return linear_x, max(-max_steer, min(max_steer, steer))
