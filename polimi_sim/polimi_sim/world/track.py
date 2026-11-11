"""A world's track: the centre line from straights and arcs, its checks and its tape lines. No ROS imports."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

CLOSE_TOLERANCE = 0.01  # m
CLOSE_HEADING_TOLERANCE = math.radians(0.5)


@dataclass(frozen=True)
class Arc:
    radius: float  # m
    angle: float  # rad, positive turns left


@dataclass(frozen=True)
class Straight:
    length: float  # m


@dataclass(frozen=True)
class CenterLine:
    """Points every `step` metres along a closed loop, with the heading and the arc length at each."""

    xy: FloatArray  # (n, 2), the first point not repeated at the end
    heading: FloatArray  # (n,) rad
    s: FloatArray  # (n,) m
    length: float  # m, the whole loop

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """x, y and heading at arc length s (wrapped onto the loop)."""
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side='right')) - 1
        j = (i + 1) % len(self.s)
        span = (self.s[j] if j else self.length) - self.s[i]
        a = (s - self.s[i]) / span if span > 0.0 else 0.0
        x, y = self.xy[i] + a * (self.xy[j] - self.xy[i])
        dh = math.atan2(math.sin(self.heading[j] - self.heading[i]), math.cos(self.heading[j] - self.heading[i]))
        return float(x), float(y), float(self.heading[i] + a * dh)

    def offset(self, d: float) -> FloatArray:
        """The line at signed distance d from the centre line, positive to the left."""
        normal = np.column_stack((-np.sin(self.heading), np.cos(self.heading)))
        return self.xy + d * normal


def parse_primitives(items: list[dict[str, Any]]) -> list[Straight | Arc]:
    """`{straight: length}` or `{arc: {radius, angle}}` (angle in degrees, positive left)."""
    primitives: list[Straight | Arc] = []
    for item in items:
        if set(item) == {'straight'} and float(item['straight']) > 0.0:
            primitives.append(Straight(float(item['straight'])))
        elif set(item) == {'arc'} and float(item['arc']['radius']) > 0.0 and float(item['arc']['angle']) != 0.0:
            primitives.append(Arc(float(item['arc']['radius']), math.radians(float(item['arc']['angle']))))
        else:
            raise ValueError(f'track primitive {item}: expected {{straight: length}} or {{arc: {{radius, angle}}}}')
    return primitives


def centerline(
    primitives: list[Straight | Arc], origin: tuple[float, float, float], step: float = 0.05
) -> CenterLine:
    """Chain the primitives from origin (x, y, heading); fails with the gap if they don't close."""
    x, y, h = origin
    xs, ys, hs, ss = [x], [y], [h], [0.0]
    s = 0.0
    for p in primitives:
        length = p.length if isinstance(p, Straight) else p.radius * abs(p.angle)
        n = max(1, math.ceil(length / step))
        x0, y0, h0 = x, y, h
        for k in range(1, n + 1):
            u = length * k / n
            if isinstance(p, Straight):
                x, y = x0 + u * math.cos(h0), y0 + u * math.sin(h0)
            else:
                side = math.copysign(1.0, p.angle)
                cx, cy = x0 - side * p.radius * math.sin(h0), y0 + side * p.radius * math.cos(h0)
                h = h0 + side * u / p.radius
                x, y = cx + side * p.radius * math.sin(h), cy - side * p.radius * math.cos(h)
            xs.append(x)
            ys.append(y)
            hs.append(h)
            ss.append(s + u)
        s += length
    gap = math.hypot(x - origin[0], y - origin[1])
    turn = math.atan2(math.sin(h - origin[2]), math.cos(h - origin[2]))
    if gap > CLOSE_TOLERANCE or abs(turn) > CLOSE_HEADING_TOLERANCE:
        raise ValueError(
            f'the track does not close: it ends at ({x:.3f}, {y:.3f}) heading {math.degrees(h):.1f} deg, '
            f'{gap:.3f} m and {math.degrees(turn):.1f} deg from its start'
        )
    return CenterLine(np.column_stack((xs[:-1], ys[:-1])), np.array(hs[:-1]), np.array(ss[:-1]), s)


def min_radius(wheelbase: float, s_max: float, steer_max_side: float) -> float:
    """The tightest circle the rear axle can drive on one side."""
    return wheelbase / math.tan(min(s_max, steer_max_side))


def tightest_corners(primitives: list[Straight | Arc]) -> dict[str, float]:
    """The smallest arc radius turning left and turning right (inf when the track has none)."""
    tightest = {'left': math.inf, 'right': math.inf}
    for p in primitives:
        if isinstance(p, Arc):
            side = 'left' if p.angle > 0.0 else 'right'
            tightest[side] = min(tightest[side], p.radius)
    return tightest
