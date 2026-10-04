"""SE(2) poses: composition, interpolation and a time-indexed buffer. No ROS imports."""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Twist2D:
    """Velocity in the body frame."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """Pose of b, given in the frame of a, in the frame a is given in."""
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return Pose2D(a.x + c * b.x - s * b.y, a.y + s * b.x + c * b.y, wrap_angle(a.yaw + b.yaw))


def interpolate(a: Pose2D, b: Pose2D, alpha: float) -> Pose2D:
    """Linear in position, shortest arc in yaw. alpha = 0 gives a, 1 gives b."""
    return Pose2D(
        a.x + alpha * (b.x - a.x),
        a.y + alpha * (b.y - a.y),
        wrap_angle(a.yaw + alpha * wrap_angle(b.yaw - a.yaw)),
    )


def extrapolate(pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D:
    """Constant body twist for dt seconds (first order)."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return Pose2D(
        pose.x + (c * twist.vx - s * twist.vy) * dt,
        pose.y + (s * twist.vx + c * twist.vy) * dt,
        wrap_angle(pose.yaw + twist.wz * dt),
    )


class PoseBuffer:
    """Poses indexed by time (seconds), kept for `horizon` seconds.

    `sample` interpolates between stored poses. Past the newest one it extrapolates
    with that pose's twist, for at most `max_extrapolation` seconds, then holds.
    """

    def __init__(self, horizon: float, max_extrapolation: float) -> None:
        if horizon <= 0.0 or max_extrapolation < 0.0:
            raise ValueError('horizon must be > 0 and max_extrapolation >= 0')
        self._horizon: float = horizon
        self._max_extrapolation: float = max_extrapolation
        self._stamps: list[float] = []
        self._poses: list[Pose2D] = []
        self._twists: list[Twist2D] = []

    def __len__(self) -> int:
        return len(self._stamps)

    def clear(self) -> None:
        self._stamps.clear()
        self._poses.clear()
        self._twists.clear()

    def newest(self) -> Pose2D | None:
        return self._poses[-1] if self._poses else None

    def add(self, stamp: float, pose: Pose2D, twist: Twist2D) -> bool:
        """Append a pose. Stamps must increase: an older or equal stamp is dropped."""
        if self._stamps and stamp <= self._stamps[-1]:
            return False
        self._stamps.append(stamp)
        self._poses.append(pose)
        self._twists.append(twist)
        keep_from = bisect_right(self._stamps, stamp - self._horizon)
        keep_from = min(keep_from, len(self._stamps) - 1)
        if keep_from > 0:
            del self._stamps[:keep_from]
            del self._poses[:keep_from]
            del self._twists[:keep_from]
        return True

    def sample(self, stamp: float) -> Pose2D | None:
        if not self._stamps:
            return None
        if stamp <= self._stamps[0]:
            return self._poses[0]
        if stamp >= self._stamps[-1]:
            dt = min(stamp - self._stamps[-1], self._max_extrapolation)
            return extrapolate(self._poses[-1], self._twists[-1], dt)
        i = bisect_right(self._stamps, stamp)
        t0, t1 = self._stamps[i - 1], self._stamps[i]
        return interpolate(self._poses[i - 1], self._poses[i], (stamp - t0) / (t1 - t0))
