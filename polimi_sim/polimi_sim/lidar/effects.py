"""Rotating-LiDAR model: timing fields, motion distortion and range effects. No ROS imports."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from polimi_sim.common.se2 import Pose2D, compose

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# cast(laser pose in the map, first beam, one past the last beam) -> ranges of those beams
CastFn = Callable[[Pose2D, int, int], FloatArray]
# pose_at(time in seconds) -> base_link pose in the map
PoseFn = Callable[[float], Pose2D]


@dataclass(frozen=True)
class ScanGeometry:
    num_beams: int
    angle_min: float        # rad
    angle_max: float        # rad
    angle_increment: float  # rad
    time_increment: float   # s between beams
    scan_time: float        # s per revolution
    sweep_time: float       # s from the first to the last beam


def scan_geometry(rate_hz: float, fov_deg: float, resolution_deg: float) -> ScanGeometry:
    """Fields of a sensor that spins at `rate_hz` and fires every `resolution_deg`."""
    if rate_hz <= 0.0 or resolution_deg <= 0.0 or not 0.0 < fov_deg <= 360.0:
        raise ValueError('rate_hz and resolution_deg must be > 0, fov_deg in (0, 360]')
    beams_per_revolution = round(360.0 / resolution_deg)
    num_beams = round(fov_deg / resolution_deg) + 1
    scan_time = 1.0 / rate_hz
    time_increment = scan_time / beams_per_revolution
    half_fov = math.radians(fov_deg) / 2.0
    return ScanGeometry(
        num_beams=num_beams,
        angle_min=-half_fov,
        angle_max=half_fov,
        angle_increment=math.radians(resolution_deg),
        time_increment=time_increment,
        scan_time=scan_time,
        sweep_time=(num_beams - 1) * time_increment,
    )


def block_bounds(num_beams: int, blocks: int) -> list[tuple[int, int]]:
    """Split the beams into `blocks` contiguous [start, stop) ranges of similar size."""
    if blocks < 1:
        raise ValueError('blocks must be >= 1')
    edges = np.linspace(0, num_beams, min(blocks, num_beams) + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def block_center_time(geometry: ScanGeometry, start: int, stop: int) -> float:
    """Time of the middle of a block, from the first beam of the scan."""
    return 0.5 * (start + stop - 1) * geometry.time_increment


def distorted_scan(
    cast: CastFn,
    pose_at: PoseFn,
    mount: Pose2D,
    geometry: ScanGeometry,
    blocks: int,
    t_start: float,
) -> FloatArray:
    """Ray-cast each block from where the laser was when that block was measured."""
    ranges = np.empty(geometry.num_beams, dtype=np.float64)
    for start, stop in block_bounds(geometry.num_beams, blocks):
        base = pose_at(t_start + block_center_time(geometry, start, stop))
        ranges[start:stop] = cast(compose(base, mount), start, stop)
    return ranges


@dataclass(frozen=True)
class EffectsConfig:
    enabled: bool = True
    range_min: float = 0.05
    range_max: float = 25.0
    no_return_value: float = math.inf
    noise_std: float = 0.01               # m; 0 = off
    dropout_prob: float = 0.002           # 0 = off
    grazing_max_deg: float = 80.0
    grazing_dropout_prob: float = 0.5     # 0 = off
    reflectivity_range_max: float = 15.0  # m; 0 = off
    mixed_pixel_jump: float = 0.3         # m
    mixed_pixel_prob: float = 0.3         # 0 = off

    def __post_init__(self) -> None:
        for name in ('dropout_prob', 'grazing_dropout_prob', 'mixed_pixel_prob'):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f'{name} must be in [0, 1]')
        if self.noise_std < 0.0 or self.reflectivity_range_max < 0.0:
            raise ValueError('noise_std and reflectivity_range_max must be >= 0')
        if self.mixed_pixel_jump <= 0.0 or not 0.0 < self.grazing_max_deg <= 90.0:
            raise ValueError('mixed_pixel_jump must be > 0, grazing_max_deg in (0, 90]')
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError('need 0 <= range_min < range_max')


def valid_returns(
    ranges: FloatArray, range_min: float, range_max: float, reflectivity_range_max: float = 0.0
) -> BoolArray:
    """Beams that hit something the sensor can see."""
    valid: BoolArray = np.isfinite(ranges) & (ranges >= range_min) & (ranges < range_max)
    if reflectivity_range_max > 0.0:
        valid &= ranges <= reflectivity_range_max
    return valid


def depth_jumps(ranges: FloatArray, jump: float) -> BoolArray:
    """jumps[i] is True when beams i and i + 1 are more than `jump` apart."""
    out: BoolArray = np.abs(np.diff(ranges)) > jump
    return out


def next_to_jump(jumps: BoolArray) -> BoolArray:
    """Per beam: True when it sits on either side of a depth jump."""
    out = np.zeros(jumps.size + 1, dtype=np.bool_)
    out[:-1] |= jumps
    out[1:] |= jumps
    return out


def incidence_angles(ranges: FloatArray, angle_increment: float) -> FloatArray:
    """Angle between each beam and the surface normal (rad), from neighbouring beams.

    In polar coordinates a surface r(phi) is hit at tan(incidence) = |dr/dphi| / r.
    """
    slope = np.gradient(ranges) / angle_increment
    out: FloatArray = np.arctan2(np.abs(slope), ranges)
    return out


def grazing_beams(
    ranges: FloatArray, angle_increment: float, grazing_max_deg: float, jump: float
) -> BoolArray:
    """Beams hitting a surface beyond `grazing_max_deg`, away from depth jumps."""
    if ranges.size < 3:
        return np.zeros(ranges.size, dtype=np.bool_)
    steep = incidence_angles(ranges, angle_increment) > math.radians(grazing_max_deg)
    out: BoolArray = steep & ~next_to_jump(depth_jumps(ranges, jump))
    return out


def mix_pixels(
    ranges: FloatArray, valid: BoolArray, jump: float, prob: float, rng: np.random.Generator
) -> FloatArray:
    """At a depth jump the far beam can land between foreground and background."""
    out: FloatArray = ranges.copy()
    edges = np.flatnonzero(depth_jumps(ranges, jump) & valid[:-1] & valid[1:])
    if edges.size == 0 or prob <= 0.0:
        return out
    edges = edges[rng.random(edges.size) < prob]
    left, right = ranges[edges], ranges[edges + 1]
    far = np.where(left > right, edges, edges + 1)
    out[far] = rng.uniform(np.minimum(left, right), np.maximum(left, right))
    return out


def apply_effects(
    clean: FloatArray, angle_increment: float, config: EffectsConfig, rng: np.random.Generator
) -> NDArray[np.float32]:
    """Turn ideal ranges into what the sensor reports."""
    cfg = config
    if not cfg.enabled:
        valid = valid_returns(clean, cfg.range_min, cfg.range_max)
        return np.where(valid, clean, cfg.no_return_value).astype(np.float32)

    valid = valid_returns(clean, cfg.range_min, cfg.range_max, cfg.reflectivity_range_max)
    lost = ~valid
    if cfg.grazing_dropout_prob > 0.0:
        grazing = grazing_beams(clean, angle_increment, cfg.grazing_max_deg, cfg.mixed_pixel_jump)
        lost |= grazing & (rng.random(clean.size) < cfg.grazing_dropout_prob)
    if cfg.dropout_prob > 0.0:
        lost |= rng.random(clean.size) < cfg.dropout_prob

    ranges = mix_pixels(clean, valid, cfg.mixed_pixel_jump, cfg.mixed_pixel_prob, rng)
    if cfg.noise_std > 0.0:
        ranges = ranges + rng.normal(0.0, cfg.noise_std, size=ranges.size)
        lost |= (ranges < cfg.range_min) | (ranges >= cfg.range_max)
    return np.where(lost, cfg.no_return_value, ranges).astype(np.float32)
