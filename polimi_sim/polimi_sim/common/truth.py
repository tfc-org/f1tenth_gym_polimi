"""Jumps in the gym's truth: teleports (a reset) and contacts (a push out of a wall). No ROS imports."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from polimi_sim.common.se2 import Pose2D, wrap_angle

ContactEvent = Literal['start', 'end']


@dataclass(frozen=True)
class TeleportConfig:
    v_max: float = 20.0  # m/s, the gym's f1tenth preset
    yaw_rate_max: float = 30.0  # rad/s, about v_max * tan(s_max) / wheelbase
    position_margin: float = 0.05  # m
    heading_margin: float = 0.1  # rad

    def __post_init__(self) -> None:
        if min(self.v_max, self.yaw_rate_max) <= 0.0 or min(self.position_margin, self.heading_margin) < 0.0:
            raise ValueError('v_max and yaw_rate_max must be > 0, the margins >= 0')


class TeleportDetector:
    """A truth pose further from the previous one than the car can move in the time between them."""

    def __init__(self, config: TeleportConfig | None = None) -> None:
        self._config: TeleportConfig = config or TeleportConfig()
        self._last: tuple[float, Pose2D] | None = None

    def update(self, stamp: float, pose: Pose2D) -> bool:
        """Feed the next truth pose; True when it jumped from the previous one."""
        previous, self._last = self._last, (stamp, pose)
        if previous is None:
            return False
        dt = max(stamp - previous[0], 0.0)
        c = self._config
        moved = math.hypot(pose.x - previous[1].x, pose.y - previous[1].y)
        turned = abs(wrap_angle(pose.yaw - previous[1].yaw))
        return moved > c.v_max * dt + c.position_margin or turned > c.yaw_rate_max * dt + c.heading_margin


class ContactDebounce:
    """One contact per push: it starts on a rising edge of the collision flag and ends once the flag
    has stayed False for `debounce` seconds, so a flag that chatters during one push is one contact."""

    def __init__(self, debounce: float = 0.1) -> None:
        if debounce < 0.0:
            raise ValueError('debounce must be >= 0')
        self._debounce: float = debounce
        self._last_true: float = -math.inf
        self.in_contact: bool = False

    def update(self, stamp: float, colliding: bool) -> ContactEvent | None:
        """Feed the next collision flag; 'start' or 'end' when a contact begins or ends."""
        if colliding:
            self._last_true = stamp
            if not self.in_contact:
                self.in_contact = True
                return 'start'
            return None
        if self.in_contact and stamp - self._last_true >= self._debounce:
            self.in_contact = False
            return 'end'
        return None
