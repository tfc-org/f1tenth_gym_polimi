"""track_monitor's logic: progress, laps, lateral offset and off-track status from the truth. No ROS imports."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
MOVING = 0.1  # m/s: the lap clock starts once the car moves faster than this


@dataclass(frozen=True)
class MonitorConfig:
    width: float  # m, between the inner edges of the two tapes
    tape_width: float  # m
    track_width: float  # m, between the wheel centres (car/body.yaml)
    lr: float  # m, rear axle (the truth pose) to the CoG
    checkpoints: int = 10
    search_window: float = 1.0  # m, plus the distance travelled since the last pose

    @property
    def off_track_offset(self) -> float:
        """The CoG offset past which, for a car aligned with the track, all four wheels are past a tape."""
        return self.width / 2.0 + self.tape_width + self.track_width / 2.0


@dataclass(frozen=True)
class LapRecord:
    lap: int
    time: float  # s
    valid: bool  # no contact and no truth jump during the lap
    max_offset: float  # m, the largest |lateral offset| of the CoG
    time_off_track: float  # s


@dataclass
class Status:
    progress: float = 0.0  # m along the lap, from the start line
    lateral_offset: float = 0.0  # m, CoG, positive to the left of the centre line
    on_track: bool = True
    lap: int = 0  # completed laps
    lap_time: float = 0.0  # s, of the lap in progress
    finished: list[LapRecord] = field(default_factory=list)


class CenterLineProjector:
    """Arc length and signed offset of a point on a closed polyline, searched near the last answer."""

    def __init__(self, xy: FloatArray) -> None:
        self.a = xy
        self.d = np.roll(xy, -1, axis=0) - xy
        self.seg = np.hypot(self.d[:, 0], self.d[:, 1])
        self.s0 = np.concatenate(([0.0], np.cumsum(self.seg)[:-1]))
        self.length = float(self.seg.sum())

    def project(self, x: float, y: float, near: float | None, window: float) -> tuple[float, float]:
        """(arc length, signed offset) of the nearest point, among segments within `window` of `near`."""
        if near is None:
            idx = np.arange(len(self.seg))
        else:
            ds = (self.s0 - near + self.length / 2.0) % self.length - self.length / 2.0
            idx = np.flatnonzero((ds > -window - self.seg) & (ds < window))
        rel = np.array([x, y]) - self.a[idx]
        t = np.clip(np.einsum('ij,ij->i', rel, self.d[idx]) / self.seg[idx] ** 2, 0.0, 1.0)
        foot = rel - t[:, None] * self.d[idx]
        dist = np.hypot(foot[:, 0], foot[:, 1])
        k = int(np.argmin(dist))
        i = idx[k]
        side = math.copysign(1.0, float(self.d[i, 0] * rel[k, 1] - self.d[i, 1] * rel[k, 0]))
        return float(self.s0[i] + t[k] * self.seg[i]) % self.length, side * float(dist[k])


class TrackMonitor:
    """A lap runs from the start line to the start line; it counts only if every checkpoint
    (evenly spaced along it) was passed in order, so a shortcut or a back-and-forth never counts."""

    def __init__(self, centerline: FloatArray, start_s: float, config: MonitorConfig) -> None:
        self._config = config
        self._line = CenterLineProjector(centerline)
        self._length = self._line.length
        self._start_s = start_s
        self.status = Status()
        self._s: float | None = None  # arc length at the last pose; None until (re)acquired
        self._last: tuple[float, float, float] | None = None  # stamp, x, y of the CoG
        self._u = 0.0  # progress since the start line, unwrapped: k * length is a start-line crossing
        self._lap_start_u = 0.0
        self._lap_start_t: float | None = None
        self._next_checkpoint = 1
        self._valid = True
        self._max_offset = 0.0
        self._off_time = 0.0

    @property
    def length(self) -> float:
        return self._length

    def _new_lap(self, start_u: float, stamp: float | None) -> None:
        self._lap_start_u, self._lap_start_t = start_u, stamp
        self._next_checkpoint, self._valid, self._max_offset, self._off_time = 1, True, 0.0, 0.0

    def jump(self) -> None:
        """The truth teleported (a reset): find the car again; the lap restarts at the next start line."""
        self._s, self._last = None, None

    def contact(self) -> None:
        """A contact started: the lap in progress is invalid."""
        self._valid = False

    def update(self, stamp: float, x: float, y: float, yaw: float, speed: float) -> Status:
        """Feed the truth pose (rear axle) and speed; returns the status, with the laps finished on this step."""
        c, st, length = self._config, self.status, self._length
        st.finished = []
        cx, cy = x + c.lr * math.cos(yaw), y + c.lr * math.sin(yaw)
        dt = 0.0 if self._last is None else stamp - self._last[0]
        travelled = 0.0 if self._last is None else math.hypot(cx - self._last[1], cy - self._last[2])
        continuous = self._s is not None
        s, offset = self._line.project(cx, cy, self._s, c.search_window + travelled)
        if continuous and abs(offset) > c.width:
            continuous = False  # lost, e.g. cut across the infield: search the whole loop
            s, offset = self._line.project(cx, cy, None, 0.0)

        if self._s is None:  # (re)acquired: place the car, its lap restarts from the line behind it
            u = (s - self._start_s) % length
            u = 0.0 if u > length - 0.05 else u  # on the start line, just behind it by rounding
            self._u = math.floor(self._u / length) * length + u
            self._new_lap(math.floor(self._u / length + 1e-9) * length, None)
        else:
            before = self._u
            self._u += (s - self._s + length / 2.0) % length - length / 2.0
            if self._lap_start_t is None and abs(speed) > MOVING:
                self._lap_start_t = stamp - dt  # the lap clock starts when the car first moves
            spacing = length / c.checkpoints
            # Only a checkpoint crossed on a continuous step counts; one skipped by a jump never will.
            while (continuous and self._next_checkpoint < c.checkpoints
                   and before < self._lap_start_u + self._next_checkpoint * spacing <= self._u):
                self._next_checkpoint += 1
        self._s, self._last = s, (stamp, cx, cy)

        st.lateral_offset = offset
        st.on_track = abs(offset) <= c.off_track_offset
        self._max_offset = max(self._max_offset, abs(offset))
        self._off_time += 0.0 if st.on_track else dt
        if self._u - self._lap_start_u >= length:
            if self._next_checkpoint == c.checkpoints:
                st.lap += 1
                started = self._lap_start_t if self._lap_start_t is not None else stamp
                st.finished.append(LapRecord(st.lap, stamp - started, self._valid, self._max_offset, self._off_time))
            self._new_lap(self._lap_start_u + length, stamp)
        st.progress = self._u - self._lap_start_u
        st.lap_time = 0.0 if self._lap_start_t is None else stamp - self._lap_start_t
        return st
