"""The camera file (profiles, intrinsics, plumb_bob distortion), the mount and the camera's pose. No ROS imports."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from polimi_sim.common.gym_config import load_yaml

FloatArray = npt.NDArray[np.float64]
MOUNT_KEYS: tuple[str, ...] = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
BODY_TO_OPTICAL: FloatArray = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
OPTICAL_RPY: tuple[float, float, float] = (-math.pi / 2, 0.0, -math.pi / 2)


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole and plumb_bob distortion (d = k1 k2 p1 p2 k3), as in camera_info."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    d: tuple[float, ...]

    @property
    def k(self) -> FloatArray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points: FloatArray) -> FloatArray:
        """Optical-frame points (n, 3), z > 0, to undistorted pixels (n, 2)."""
        uv: FloatArray = points[:, :2] / points[:, 2:3] * (self.fx, self.fy) + (self.cx, self.cy)
        return uv


@dataclass(frozen=True)
class Camera:
    """One profile of a camera file; color_in_link is camera_link -> colour sensor in m."""

    profile: str
    rate_hz: float
    intrinsics: Intrinsics
    color_in_link: tuple[float, float, float]


def load_camera(path: str, profile: str = '') -> Camera:
    """One profile of a camera file; empty profile = the file's `profile`."""
    content = load_yaml(path)
    name = profile or str(content.get('profile', ''))
    profiles: dict[str, Any] = content.get('profiles', {})
    if name not in profiles:
        raise ValueError(f'{path}: no profile {name!r} (has {", ".join(profiles) or "none"})')
    p = profiles[name]
    if p.get('distortion_model', 'plumb_bob') != 'plumb_bob':
        raise ValueError(f'{path}: only plumb_bob distortion is drawn (rational_polynomial is deferred, O-29)')
    d = tuple(float(v) for v in p.get('d', [0.0] * 5))
    if len(d) != 5:
        raise ValueError(f'{path}: plumb_bob takes 5 coefficients, got {len(d)}')
    intrinsics = Intrinsics(int(p['width']), int(p['height']), float(p['fx']), float(p['fy']),
                            float(p['cx']), float(p['cy']), d)
    offset = content.get('color_in_camera_link', {})
    return Camera(name, float(p['rate_hz']), intrinsics,
                  (float(offset.get('x', 0.0)), float(offset.get('y', 0.0)), float(offset.get('z', 0.0))))


def load_camera_mount(path: str) -> dict[str, float]:
    """base_link -> camera_link: x, y, z, roll, pitch, yaw."""
    mount = load_yaml(path).get('base_link_to_camera_link', {})
    missing = [k for k in MOUNT_KEYS if k not in mount]
    if missing:
        raise ValueError(f'{path} is missing {", ".join(missing)}')
    return {k: float(mount[k]) for k in MOUNT_KEYS}


def rotation_rpy(roll: float, pitch: float, yaw: float) -> FloatArray:
    """R = Rz(yaw) Ry(pitch) Rx(roll), the ROS fixed-axis convention."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    r: FloatArray = rz @ ry @ rx
    return r


@dataclass(frozen=True)
class OpticalMount:
    """The colour optical frame in base_link: rotation (columns are the optical axes) and position."""

    rotation: FloatArray
    position: FloatArray


def optical_mount(mount: dict[str, float], color_in_link: tuple[float, float, float]) -> OpticalMount:
    """base_link -> camera_link (the mount) -> colour sensor -> optical axes (x right, y down, z forward)."""
    r_link = rotation_rpy(mount['roll'], mount['pitch'], mount['yaw'])
    position = np.array([mount['x'], mount['y'], mount['z']]) + r_link @ np.asarray(color_in_link)
    return OpticalMount(r_link @ BODY_TO_OPTICAL, position)


def camera_in_world(x: float, y: float, yaw: float, mount: OpticalMount) -> tuple[FloatArray, FloatArray]:
    """(R, C) of the optical frame in the world, for base_link at (x, y, yaw) on the floor."""
    r_base = rotation_rpy(0.0, 0.0, yaw)
    return r_base @ mount.rotation, np.array([x, y, 0.0]) + r_base @ mount.position


def distortion_maps(intrinsics: Intrinsics) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None:
    """cv2.remap maps that turn an undistorted drawing into the distorted image, or None without distortion.

    remap reads, for each output (distorted) pixel, the source (undistorted) pixel: undistortPoints on the
    distorted pixel grid. initUndistortRectifyMap gives the opposite direction.
    """
    if not any(intrinsics.d):
        return None
    w, h = intrinsics.width, intrinsics.height
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = np.stack((u.ravel(), v.ravel()), axis=1).reshape(-1, 1, 2)
    k = intrinsics.k
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
    ideal = cv2.undistortPoints(grid, k, np.asarray(intrinsics.d), None, None, k, criteria).reshape(h, w, 2)
    return ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32)
