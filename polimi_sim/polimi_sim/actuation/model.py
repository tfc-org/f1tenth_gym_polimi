"""Actuation model between the mux and the gym: delays, timeout, servo and speed limits.

All of it is time-based (seconds). No ROS imports.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')
_EPS = 1e-9


@dataclass(frozen=True)
class ActuationConfig:
    command_delay: float = 0.02   # s, command -> VESC
    vesc_timeout: float = 0.5     # s without commands -> speed 0
    steer_offset: float = 0.0     # rad, trim
    servo_delay: float = 0.03     # s, dead time of the servo
    servo_rate: float = 3.2       # rad/s
    max_steer: float = 0.4189     # rad
    accel_max: float = 4.0        # m/s^2
    brake_max: float = 6.0        # m/s^2

    def __post_init__(self) -> None:
        for name in ('command_delay', 'vesc_timeout', 'servo_delay'):
            if getattr(self, name) < 0.0:
                raise ValueError(f'{name} must be >= 0')
        for name in ('servo_rate', 'max_steer', 'accel_max', 'brake_max'):
            if getattr(self, name) <= 0.0:
                raise ValueError(f'{name} must be > 0')


@dataclass(frozen=True)
class ActuationOutput:
    speed: float             # m/s, to the gym
    steering_angle: float    # rad, to the gym
    steering_command: float  # rad, as sent to the VESC (before the servo dynamics)


class DelayLine(Generic[T]):
    """Values that become visible `delay` seconds after they were pushed."""

    def __init__(self) -> None:
        self._pending: deque[tuple[float, T]] = deque()
        self._current: tuple[float, T] | None = None

    def push(self, stamp: float, value: T) -> None:
        self._pending.append((stamp, value))

    def get(self, now: float, delay: float) -> tuple[float, T] | None:
        """Newest value pushed at or before `now - delay`, with the time it became visible."""
        while self._pending and self._pending[0][0] + delay <= now + _EPS:
            stamp, value = self._pending.popleft()
            self._current = (stamp + delay, value)
        return self._current


def _approach(value: float, target: float, max_step: float) -> float:
    return value + max(-max_step, min(max_step, target - value))


class ActuationModel:
    """Feed commands with `command`, then call `step` at a fixed rate."""

    def __init__(self, config: ActuationConfig) -> None:
        self.config: ActuationConfig = config
        self._commands: DelayLine[tuple[float, float]] = DelayLine()
        self._servo: DelayLine[float] = DelayLine()
        self._last_step: float | None = None
        self._speed: float = 0.0
        self._steer: float = 0.0

    def command(self, stamp: float, speed: float, steering_angle: float) -> None:
        if math.isfinite(speed) and math.isfinite(steering_angle):
            self._commands.push(stamp, (speed, steering_angle))

    def step(self, now: float) -> ActuationOutput:
        cfg = self.config
        dt = 0.0 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now

        target_speed, steer = 0.0, 0.0
        received = self._commands.get(now, cfg.command_delay)
        if received is not None:
            arrival, (speed, steer) = received
            if now - arrival <= cfg.vesc_timeout + _EPS:
                target_speed = speed

        steering_command = max(-cfg.max_steer, min(cfg.max_steer, steer + cfg.steer_offset))
        self._servo.push(now, steering_command)
        delayed = self._servo.get(now, cfg.servo_delay)
        if delayed is not None:
            self._steer = _approach(self._steer, delayed[1], cfg.servo_rate * dt)

        self._speed = self._ramp_speed(target_speed, dt)
        return ActuationOutput(self._speed, self._steer, steering_command)

    def _ramp_speed(self, target: float, dt: float) -> float:
        """Speeding up is limited by accel_max, slowing down by brake_max."""
        cfg = self.config
        speed = self._speed
        slowing = speed * target < 0.0 or abs(target) < abs(speed)
        if not slowing:
            return _approach(speed, target, cfg.accel_max * dt)
        stop_at = target if speed * target > 0.0 else 0.0
        return _approach(speed, stop_at, cfg.brake_max * dt)
