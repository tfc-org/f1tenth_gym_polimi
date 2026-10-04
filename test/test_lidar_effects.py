import math

import numpy as np
import pytest

from polimi_sim.lidar_effects import (
    CastFn,
    EffectsConfig,
    FloatArray,
    apply_effects,
    block_bounds,
    block_center_time,
    distorted_scan,
    grazing_beams,
    incidence_angles,
    mix_pixels,
    scan_geometry,
    valid_returns,
)
from polimi_sim.se2 import Pose2D

SL450 = scan_geometry(40.0, 270.0, 0.2)
INC = SL450.angle_increment
ALL_OFF = EffectsConfig(
    noise_std=0.0, dropout_prob=0.0, grazing_dropout_prob=0.0,
    reflectivity_range_max=0.0, mixed_pixel_prob=0.0,
)


def angles() -> FloatArray:
    return SL450.angle_min + np.arange(SL450.num_beams) * INC


def wall_ahead(distance: float) -> CastFn:
    """Caster for a wall at x = distance (laser looking along +x when yaw = 0)."""
    def cast(laser: Pose2D, start: int, stop: int) -> FloatArray:
        beam = laser.yaw + angles()[start:stop]
        with np.errstate(divide='ignore'):
            ranges = (distance - laser.x) / np.cos(beam)
        return np.where((np.cos(beam) > 1e-9) & (ranges < 25.0), ranges, 25.0)
    return cast


@pytest.mark.parametrize('rate', [15.0, 20.0, 25.0, 30.0, 40.0])
def test_timing_fields(rate: float) -> None:
    g = scan_geometry(rate, 270.0, 0.2)
    assert g.num_beams == 1351
    assert g.angle_min == pytest.approx(-math.radians(135.0))
    assert g.angle_min + (g.num_beams - 1) * g.angle_increment == pytest.approx(g.angle_max)
    assert g.scan_time == pytest.approx(1.0 / rate)
    assert g.time_increment == pytest.approx(1.0 / (rate * 1800.0))
    assert g.sweep_time == pytest.approx(0.75 / rate)


def test_sweep_is_18_75_ms_at_40_hz() -> None:
    assert SL450.sweep_time == pytest.approx(0.01875)


def test_blocks_cover_every_beam_once_and_times_increase() -> None:
    bounds = block_bounds(SL450.num_beams, 8)
    assert len(bounds) == 8 and bounds[0][0] == 0 and bounds[-1][1] == SL450.num_beams
    assert all(a[1] == b[0] for a, b in zip(bounds[:-1], bounds[1:], strict=True))
    times = [block_center_time(SL450, a, b) for a, b in bounds]
    assert all(t0 < t1 for t0, t1 in zip(times[:-1], times[1:], strict=True))
    assert 0.0 < times[0] and times[-1] < SL450.sweep_time
    assert block_bounds(3, 8) == [(0, 1), (1, 2), (2, 3)]


def test_static_car_equals_plain_cast() -> None:
    cast = wall_ahead(5.0)
    mount = Pose2D(0.27, 0.0, 0.0)
    plain = cast(mount, 0, SL450.num_beams)
    blocks = distorted_scan(cast, lambda t: Pose2D(0.0, 0.0, 0.0), mount, SL450, 8, 3.0)
    np.testing.assert_array_equal(blocks, plain)
    np.testing.assert_allclose(apply_effects(plain, INC, ALL_OFF, np.random.default_rng(1))[plain < 25.0],
                               plain[plain < 25.0], rtol=1e-6)


def test_motion_distortion_shifts_blocks_by_the_distance_travelled() -> None:
    speed, t_start, mount = 8.0, 2.0, Pose2D(0.0, 0.0, 0.0)
    cast = wall_ahead(5.0)
    static = cast(mount, 0, SL450.num_beams)
    moving = distorted_scan(cast, lambda t: Pose2D(speed * (t - t_start), 0.0, 0.0), mount, SL450, 8, t_start)
    bounds = block_bounds(SL450.num_beams, 8)
    cos = np.cos(angles())
    for start, stop in bounds[2:6]:  # blocks that see the wall
        travelled = speed * block_center_time(SL450, start, stop)
        np.testing.assert_allclose((static - moving)[start:stop], travelled / cos[start:stop], rtol=1e-9)
    first, last = bounds[2], bounds[5]
    shift = (static - moving)[last[0]] * cos[last[0]] - (static - moving)[first[0]] * cos[first[0]]
    assert shift == pytest.approx(speed * (block_center_time(SL450, *last) - block_center_time(SL450, *first)))


def test_valid_returns() -> None:
    ranges = np.array([0.01, 1.0, 16.0, 25.0, np.inf])
    np.testing.assert_array_equal(valid_returns(ranges, 0.05, 25.0), [False, True, True, False, False])
    np.testing.assert_array_equal(valid_returns(ranges, 0.05, 25.0, 15.0), [False, True, False, False, False])


def test_effects_disabled_only_marks_no_returns() -> None:
    clean = np.array([1.0, 2.0, 25.0, 20.0])
    off = EffectsConfig(enabled=False, no_return_value=0.0)
    np.testing.assert_allclose(apply_effects(clean, INC, off, np.random.default_rng(1)), [1.0, 2.0, 0.0, 20.0])


def test_noise_only_changes_values() -> None:
    clean = np.full(20000, 5.0)
    cfg = EffectsConfig(noise_std=0.02, dropout_prob=0.0, grazing_dropout_prob=0.0, mixed_pixel_prob=0.0)
    out = apply_effects(clean, INC, cfg, np.random.default_rng(1))
    assert np.all(np.isfinite(out))
    assert float(np.mean(out)) == pytest.approx(5.0, abs=1e-3)
    assert float(np.std(out)) == pytest.approx(0.02, rel=0.05)


def test_dropouts_only_remove_beams() -> None:
    clean = np.full(50000, 5.0)
    cfg = EffectsConfig(noise_std=0.0, dropout_prob=0.01, grazing_dropout_prob=0.0, mixed_pixel_prob=0.0)
    out = apply_effects(clean, INC, cfg, np.random.default_rng(1))
    lost = np.isinf(out)
    assert float(np.mean(lost)) == pytest.approx(0.01, rel=0.2)
    np.testing.assert_array_equal(out[~lost], 5.0)


def test_reflectivity_limit() -> None:
    clean = np.array([5.0, 14.9, 15.1, 24.0])
    cfg = EffectsConfig(noise_std=0.0, dropout_prob=0.0, grazing_dropout_prob=0.0, mixed_pixel_prob=0.0)
    np.testing.assert_allclose(apply_effects(clean, INC, cfg, np.random.default_rng(1)), [5.0, 14.9, np.inf, np.inf])


def test_incidence_angle_on_a_flat_wall() -> None:
    a = angles()[300:1051]  # +-75 deg
    ranges = 2.0 / np.cos(a)
    np.testing.assert_allclose(incidence_angles(ranges, INC)[1:-1], np.abs(a)[1:-1], atol=2e-3)


def test_grazing_only_hits_steep_beams() -> None:
    a = angles()[250:1101]  # +-85 deg
    ranges = 0.5 / np.cos(a)
    grazing = grazing_beams(ranges, INC, 80.0, 0.3)
    assert not grazing[np.abs(a) < math.radians(79.0)].any()
    assert grazing[(np.abs(a) > math.radians(81.0)) & (np.abs(a) < math.radians(84.0))].all()
    cfg = EffectsConfig(noise_std=0.0, dropout_prob=0.0, grazing_dropout_prob=1.0,
                        reflectivity_range_max=0.0, mixed_pixel_prob=0.0)
    out = apply_effects(ranges, INC, cfg, np.random.default_rng(1))
    np.testing.assert_array_equal(np.isinf(out), grazing)


def test_mixed_pixels_only_touch_the_far_beam_at_an_edge() -> None:
    clean = np.concatenate([np.full(10, 1.0), np.full(10, 3.0), np.full(10, 1.5)])
    valid = np.ones(clean.size, dtype=np.bool_)
    out = mix_pixels(clean, valid, 0.3, 1.0, np.random.default_rng(1))
    changed = np.flatnonzero(out != clean)
    np.testing.assert_array_equal(changed, [10, 19])
    assert 1.0 < out[10] < 3.0 and 1.5 < out[19] < 3.0
    np.testing.assert_array_equal(mix_pixels(clean, valid, 0.3, 0.0, np.random.default_rng(1)), clean)


def test_same_seed_same_scan() -> None:
    clean = 3.0 / np.cos(angles()[400:900])
    a = apply_effects(clean, INC, EffectsConfig(), np.random.default_rng(7))
    b = apply_effects(clean, INC, EffectsConfig(), np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_config_is_validated() -> None:
    with pytest.raises(ValueError):
        EffectsConfig(dropout_prob=1.5)
    with pytest.raises(ValueError):
        EffectsConfig(range_min=30.0)
