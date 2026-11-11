import math

import numpy as np
import pytest

from polimi_sim.eval.monitor import MonitorConfig, TrackMonitor
from polimi_sim.world.track import CenterLine, centerline, parse_primitives

LOOP = [
    {'straight': 4.0}, {'arc': {'radius': 2.0, 'angle': 90}}, {'arc': {'radius': 1.5, 'angle': 90}},
    {'straight': 0.75}, {'arc': {'radius': 1.5, 'angle': 30}}, {'arc': {'radius': 1.5, 'angle': -60}},
    {'arc': {'radius': 1.5, 'angle': 30}}, {'straight': 0.75}, {'arc': {'radius': 1.75, 'angle': 180}},
]
LINE: CenterLine = centerline(parse_primitives(LOOP), (0.0, 0.0, 0.0))
CONFIG = MonitorConfig(width=1.0, tape_width=0.05, track_width=0.281, lr=0.17145)
START = 2.0
DT = 0.01


def _rear_axle(s: float, lateral: float = 0.0) -> tuple[float, float, float]:
    """The rear-axle pose that puts the CoG on the centre line (or `lateral` to its left) at s."""
    x, y, h = LINE.pose_at(s)
    x, y = x - lateral * math.sin(h), y + lateral * math.cos(h)
    return x - CONFIG.lr * math.cos(h), y - CONFIG.lr * math.sin(h), h


def _drive(monitor: TrackMonitor, s_values: np.ndarray, t0: float = 0.0, lateral: float = 0.0) -> list:
    statuses = []
    for i, s in enumerate(s_values):
        st = monitor.update(t0 + i * DT, *_rear_axle(float(s), lateral), 3.0)
        statuses.append((st.progress, st.on_track, st.lateral_offset, list(st.finished)))
    return statuses


def test_one_lap_along_the_centre_line_counts_once_and_is_valid() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    statuses = _drive(monitor, START + np.arange(0.0, LINE.length + 1.0, 0.03))
    laps = [r for *_, finished in statuses for r in finished]
    assert len(laps) == 1 and laps[0].valid and laps[0].time == pytest.approx(LINE.length / 0.03 * DT, rel=0.01)
    assert laps[0].max_offset < 0.01


def test_progress_never_jumps_through_the_hairpin_and_the_chicane() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    progress = [p for p, *_ in _drive(monitor, START + np.arange(0.0, LINE.length - 0.05, 0.03))]
    steps = np.diff(progress)
    assert steps.min() > 0.0 and steps.max() < 0.06


def test_the_offset_is_positive_to_the_left_and_off_track_past_the_rule() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    limit = CONFIG.off_track_offset  # 0.5 + 0.05 + 0.1405
    assert limit == pytest.approx(0.6905)
    (_, on_left, offset_left, _), = _drive(monitor, np.array([START]), lateral=0.3)
    assert offset_left == pytest.approx(0.3, abs=1e-3) and on_left
    (_, on_out, offset_out, _), = _drive(monitor, np.array([START + 0.01]), t0=DT, lateral=-(limit + 0.02))
    assert offset_out == pytest.approx(-(limit + 0.02), abs=1e-3) and not on_out


def test_a_wider_car_goes_off_track_sooner() -> None:
    wide = MonitorConfig(width=1.0, tape_width=0.05, track_width=0.4, lr=CONFIG.lr)
    on = [TrackMonitor(LINE.xy, START, cfg).update(0.0, *_rear_axle(START, 0.7), 0.0).on_track for cfg in (CONFIG, wide)]
    assert on == [False, True]


def test_back_and_forth_over_the_start_line_counts_no_lap() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    wiggle = START + 0.5 * np.sin(np.linspace(0.0, 20.0 * math.pi, 2000))
    assert not any(finished for *_, finished in _drive(monitor, wiggle))


def test_a_shortcut_across_the_infield_counts_no_lap() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    half = START + np.arange(0.0, LINE.length / 2.0, 0.03)
    statuses = _drive(monitor, half)
    # jump straight back to just before the start line, as if cut across, then finish
    statuses += _drive(monitor, START + np.arange(LINE.length - 0.5, LINE.length + 0.5, 0.03), t0=len(half) * DT)
    assert not any(finished for *_, finished in statuses)


def test_a_contact_makes_the_lap_invalid() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    _drive(monitor, START + np.arange(0.0, 3.0, 0.03))
    monitor.contact()
    statuses = _drive(monitor, START + np.arange(3.0, LINE.length + 0.5, 0.03), t0=1.0)
    laps = [r for *_, finished in statuses for r in finished]
    assert len(laps) == 1 and not laps[0].valid


def test_after_a_reset_the_partial_lap_is_not_counted_but_the_next_one_is() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    _drive(monitor, START + np.arange(0.0, 2.0, 0.03))
    monitor.jump()  # teleported to mid-lap
    mid = START + LINE.length / 2.0
    statuses = _drive(monitor, mid + np.arange(0.0, 1.5 * LINE.length + 0.5, 0.03), t0=1.0)
    laps = [r for *_, finished in statuses for r in finished]
    assert len(laps) == 1 and laps[0].valid


def test_the_lap_clock_starts_when_the_car_moves_not_when_it_settles() -> None:
    monitor = TrackMonitor(LINE.xy, START, CONFIG)
    for i in range(500):  # parked for 5 s, creeping a few millimetres as the gym settles
        monitor.update(i * DT, *_rear_axle(START + 0.00004 * i), 0.0)
    laps = [r for *_, finished in _drive(monitor, START + 0.02 + np.arange(0.0, LINE.length, 0.03), t0=5.0)
            for r in finished]
    assert len(laps) == 1 and laps[0].time == pytest.approx(LINE.length / 0.03 * DT, rel=0.01)
