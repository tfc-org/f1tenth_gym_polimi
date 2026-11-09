"""VESC-like wheel odometry: speed from the motor, yaw rate from the steering command.

Mirrors vesc_ackermann's vesc_to_odom (kinematic bicycle, Euler steps). No ROS imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OdomConfig:
    wheelbase: float = 0.3302         # m
    speed_scale_error: float = 0.02   # measured = true * (1 + this)
    speed_noise_std: float = 0.02     # m/s
    speed_deadband: float = 0.05      # m/s, below this the VESC odometry reads 0

    def __post_init__(self) -> None:
        if self.wheelbase <= 0.0:
            raise ValueError('wheelbase must be > 0')
        if self.speed_noise_std < 0.0 or self.speed_deadband < 0.0:
            raise ValueError('speed_noise_std and speed_deadband must be >= 0')
        if self.speed_scale_error <= -1.0:
            raise ValueError('speed_scale_error must be > -1')


@dataclass(frozen=True)
class OdomState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0


class WheelOdometry:
    def __init__(self, config: OdomConfig, rng: np.random.Generator) -> None:
        self.config: OdomConfig = config
        self._rng: np.random.Generator = rng
        self.state: OdomState = OdomState()

    def reset(self) -> None:
        """Back to the origin, as after a restart of the VESC odometry."""
        self.state = OdomState()

    def update(self, dt: float, true_speed: float, steering_command: float) -> OdomState:
        """Advance by dt seconds with the true longitudinal speed and the commanded steering."""
        cfg = self.config
        speed = true_speed * (1.0 + cfg.speed_scale_error)
        if cfg.speed_noise_std > 0.0:
            speed += float(self._rng.normal(0.0, cfg.speed_noise_std))
        if abs(speed) < cfg.speed_deadband:
            speed = 0.0
        yaw_rate = speed * math.tan(steering_command) / cfg.wheelbase

        s = self.state
        yaw = s.yaw + yaw_rate * dt
        self.state = OdomState(
            x=s.x + speed * math.cos(s.yaw) * dt,
            y=s.y + speed * math.sin(s.yaw) * dt,
            yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
            speed=speed,
            yaw_rate=yaw_rate,
        )
        return self.state
