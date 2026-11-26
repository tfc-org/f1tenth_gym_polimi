"""The world as flat 3D polygons for the camera, built once from the generated scene.yaml and tape.yaml. No ROS imports."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polimi_sim.camera.intrinsics import FloatArray
from polimi_sim.common.gym_config import load_yaml

CEILING_RGB: tuple[int, int, int] = (200, 200, 205)
CHUNK_LENGTH = 1.0
PANEL_LENGTH = 1.0


@dataclass(frozen=True)
class Polygon:
    """A planar polygon in the world (n, 3) with its colour; `normal` points to the side it is seen from, or None for both."""

    points: FloatArray
    rgb: tuple[int, int, int]
    normal: FloatArray | None = None


@dataclass
class Scene:
    """Drawn in this order: background (the ceiling), floor, tape, then the solids far to near."""

    background: tuple[int, int, int] = CEILING_RGB
    floor: list[Polygon] = field(default_factory=list)
    tape: list[Polygon] = field(default_factory=list)
    solids: list[Polygon] = field(default_factory=list)


def _rgb(value: list[int]) -> tuple[int, int, int]:
    r, g, b = (int(v) for v in value)
    return r, g, b


def _counter_clockwise(xy: FloatArray) -> FloatArray:
    x, y = xy[:, 0], xy[:, 1]
    area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return xy if area > 0.0 else xy[::-1]


def _on_floor(xy: FloatArray, z: float = 0.0) -> FloatArray:
    return np.column_stack((xy, np.full(len(xy), z)))


def tape_chunks(line: FloatArray, width: float, chunk_length: float = CHUNK_LENGTH) -> list[FloatArray]:
    """A closed tape strip around the points `line` (its centre) as quads chains of at most chunk_length.

    Neighbouring chunks share their end edge and never overlap: one fillPoly with overlapping polygons
    leaves the overlap as a hole.
    """
    tangent = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left, right = line + normal * width / 2.0, line - normal * width / 2.0
    step = float(np.mean(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    per_chunk = max(1, math.floor(chunk_length / step))
    n = len(line)
    chunks = []
    for start in range(0, n, per_chunk):
        idx = np.arange(start, min(start + per_chunk, n) + 1) % n
        chunks.append(np.vstack((left[idx], right[idx][::-1])))
    return chunks


def _panels(a: FloatArray, b: FloatArray, height: float, normal: FloatArray, rgb: tuple[int, int, int]) -> list[Polygon]:
    count = max(1, math.ceil(float(np.linalg.norm(b - a)) / PANEL_LENGTH))
    out = []
    for i in range(count):
        p0, p1 = a + (b - a) * i / count, a + (b - a) * (i + 1) / count
        out.append(Polygon(np.array([[*p0, 0.0], [*p1, 0.0], [*p1, height], [*p0, height]]), rgb, normal))
    return out


def _side_normal(a: FloatArray, b: FloatArray, outward: bool) -> FloatArray:
    """Of the edge a -> b of a counter-clockwise outline: outward points right of the edge, inward left."""
    dx, dy = b - a
    n = np.array([dy, -dx, 0.0]) / math.hypot(dx, dy)
    return n if outward else -n


def build_scene(scene: dict[str, Any], tape: dict[str, Any]) -> Scene:
    """The camera's polygons: the room's floor and wall panels, the tape chunks, every object's sides and top."""
    out = Scene()
    walls = scene['walls']
    outline = _counter_clockwise(np.asarray(walls['outline'], dtype=float))
    out.floor.append(Polygon(_on_floor(outline), _rgb(scene['floor']['rgb'])))
    for line in tape['lines']:
        for chunk in tape_chunks(np.asarray(line, dtype=float), float(tape['tape_width'])):
            out.tape.append(Polygon(_on_floor(chunk), _rgb(tape['rgb'])))
    wall_rgb, wall_height = _rgb(walls['rgb']), float(walls['height'])
    for a, b in zip(outline, np.roll(outline, -1, axis=0), strict=True):
        out.solids += _panels(a, b, wall_height, _side_normal(a, b, outward=False), wall_rgb)
    for obj in scene.get('objects', []):
        footprint = _counter_clockwise(np.asarray(obj['polygon'], dtype=float))
        height, rgb = float(obj['height']), _rgb(obj['rgb'])
        for a, b in zip(footprint, np.roll(footprint, -1, axis=0), strict=True):
            normal = _side_normal(a, b, outward=True)
            out.solids.append(Polygon(np.array([[*a, 0.0], [*b, 0.0], [*b, height], [*a, height]]), rgb, normal))
        out.solids.append(Polygon(_on_floor(footprint, height), rgb, np.array([0.0, 0.0, 1.0])))
    return out


def load_scene(scene_path: str, tape_path: str) -> Scene:
    """build_scene from the files make_world writes (the world: block's scene and tape)."""
    return build_scene(load_yaml(scene_path), load_yaml(tape_path))
