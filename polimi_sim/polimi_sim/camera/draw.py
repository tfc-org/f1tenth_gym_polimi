"""Draws a scene as the camera sees it: clip to the view, project, fill far to near with OpenCV. No ROS imports."""
from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

from polimi_sim.camera.intrinsics import FloatArray, Intrinsics, distortion_maps
from polimi_sim.camera.scene import Polygon, Scene

NEAR = 0.05
MARGIN = 64
SHIFT = 4
Image = npt.NDArray[np.uint8]


def shade(rgb: tuple[int, int, int], distance: float) -> tuple[int, int, int]:
    """Simple distance shading: 100 % close, 75 % far."""
    f = 0.75 + 0.25 * math.exp(-distance / 6.0)
    return int(rgb[0] * f), int(rgb[1] * f), int(rgb[2] * f)


def clip(points: FloatArray, planes: FloatArray) -> FloatArray:
    """Sutherland-Hodgman against planes (m, 4), inside where n . p + d >= 0. Fewer than 3 points: nothing left."""
    side = points @ planes[:, :3].T + planes[:, 3]
    if (side >= 0.0).all():
        return points
    if (side < 0.0).all(axis=0).any():
        return points[:0]
    for k in np.flatnonzero((side < 0.0).any(axis=0)):
        if len(points) < 3:
            break
        s = points @ planes[k, :3] + planes[k, 3]
        out = []
        for i in range(len(points)):
            j = (i + 1) % len(points)
            if s[i] >= 0.0:
                out.append(points[i])
            if (s[i] >= 0.0) != (s[j] >= 0.0):
                out.append(points[i] + (points[j] - points[i]) * (s[i] / (s[i] - s[j])))
        points = np.array(out) if out else points[:0]
    return points if len(points) >= 3 else points[:0]


class Renderer:
    """Draws one Scene through one camera; everything that doesn't change per frame is precomputed."""

    def __init__(self, scene: Scene, intrinsics: Intrinsics) -> None:
        k = intrinsics
        self._k = k
        self._scene = scene
        self._planes = np.array([
            [0.0, 0.0, 1.0, -NEAR],
            [1.0, 0.0, (k.cx + MARGIN) / k.fx, 0.0],
            [-1.0, 0.0, (k.width - 1 + MARGIN - k.cx) / k.fx, 0.0],
            [0.0, 1.0, (k.cy + MARGIN) / k.fy, 0.0],
            [0.0, -1.0, (k.height - 1 + MARGIN - k.cy) / k.fy, 0.0],
        ])
        self._maps = distortion_maps(k)
        self._centres = np.array([p.points.mean(axis=0) for p in scene.solids]).reshape(-1, 3)
        self._normals = np.array([p.normal if p.normal is not None else np.zeros(3) for p in scene.solids]).reshape(-1, 3)
        self._two_sided = np.array([p.normal is None for p in scene.solids], dtype=bool)
        self._canvas: Image = np.empty((k.height, k.width, 3), np.uint8)

    def _pixels(self, polygon: Polygon, r: FloatArray, c: FloatArray) -> npt.NDArray[np.int32] | None:
        cam = clip((polygon.points - c) @ r, self._planes)
        if len(cam) == 0:
            return None
        return np.round(self._k.project(cam) * (1 << SHIFT)).astype(np.int32)

    def render(self, r: FloatArray, c: FloatArray) -> Image:
        """The frame for the optical frame at rotation r (columns: its axes in the world) and position c.

        The returned array is reused by the next call: copy it to keep it.
        """
        img = self._canvas
        img[:] = self._scene.background
        for layer in (self._scene.floor, self._scene.tape):
            by_colour: dict[tuple[int, int, int], list[npt.NDArray[np.int32]]] = {}
            for polygon in layer:
                px = self._pixels(polygon, r, c)
                if px is not None:
                    by_colour.setdefault(polygon.rgb, []).append(px)
            for rgb, pts in by_colour.items():
                cv2.fillPoly(img, pts, rgb, cv2.LINE_AA, SHIFT)
        if len(self._centres):
            facing = self._two_sided | (np.einsum('ij,ij->i', self._normals, c - self._centres) > 0.0)
            distance = np.linalg.norm(self._centres - c, axis=1)
            for i in sorted(np.flatnonzero(facing), key=lambda i: -distance[i]):
                polygon = self._scene.solids[i]
                px = self._pixels(polygon, r, c)
                if px is not None:
                    cv2.fillPoly(img, [px], shade(polygon.rgb, float(distance[i])), cv2.LINE_AA, SHIFT)
        if self._maps is not None:
            return np.asarray(cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR), dtype=np.uint8)
        return img
