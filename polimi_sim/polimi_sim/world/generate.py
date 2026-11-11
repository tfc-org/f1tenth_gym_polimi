"""make_world: one world source file -> the maps, the camera scene and the truth the simulator needs.

Writes into <maps>/<world>/ (the source is <maps>/src/<world>.yaml) and <config>/maps/<world>.yaml.
The output is byte-for-byte the same for the same inputs, so regenerating at every start leaves
git clean. No ROS imports.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import yaml

from polimi_sim.world.track import CenterLine, centerline, min_radius, parse_primitives, tightest_corners

RADIUS_MARGIN = 0.2  # m, on top of the car's tightest circle
MAP_THRESHOLDS: dict[str, Any] = {'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}
FREE, OCCUPIED = 254, 0


@dataclass(frozen=True)
class Car:
    """What the checks and the LiDAR slice need from the car."""

    wheelbase: float  # m, lf + lr of the gym preset
    s_max: float  # rad, the gym preset's steering limit
    steer_max_left: float  # rad, car/body.yaml
    steer_max_right: float  # rad
    laser_z: float  # m, car/laser_mount.yaml


def _r(value: float) -> float:
    return round(float(value), 4) + 0.0  # + 0.0 turns -0.0 into 0.0


def _points(xy: np.ndarray) -> list[list[float]]:
    return [[_r(x), _r(y)] for x, y in xy]


def _polygon(obj: dict[str, Any]) -> np.ndarray:
    """A box (x, y, yaw in degrees, length, width) or a prism (polygon) as an (n, 2) array."""
    if 'polygon' in obj:
        return np.asarray(obj['polygon'], dtype=float)
    half_l, half_w = float(obj['length']) / 2.0, float(obj['width']) / 2.0
    yaw = math.radians(float(obj.get('yaw', 0.0)))
    corners = np.array([[half_l, half_w], [-half_l, half_w], [-half_l, -half_w], [half_l, -half_w]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return corners @ rot.T + np.array([float(obj['x']), float(obj['y'])])


class Grid:
    """An occupancy image over the room, origin at its lower-left corner, row 0 at the top."""

    def __init__(self, outline: np.ndarray, resolution: float, margin_cells: int = 2) -> None:
        self.resolution = resolution
        self.origin = outline.min(axis=0) - margin_cells * resolution
        size = np.ceil((outline.max(axis=0) - outline.min(axis=0)) / resolution).astype(int) + 2 * margin_cells
        self.width, self.height = int(size[0]), int(size[1])

    def _pixels(self, xy: np.ndarray) -> np.ndarray:
        u = (xy[:, 0] - self.origin[0]) / self.resolution - 0.5
        v = self.height - (xy[:, 1] - self.origin[1]) / self.resolution - 0.5
        return np.round(np.column_stack((u, v)) * 256.0).astype(np.int32)  # 8 bits of sub-pixel

    def image(self, outline: np.ndarray, obstacles: list[np.ndarray]) -> np.ndarray:
        """Outside the room and every obstacle occupied; a cell any part of an obstacle covers is occupied."""
        img = np.full((self.height, self.width), OCCUPIED, dtype=np.uint8)
        cv2.fillPoly(img, [self._pixels(outline)], FREE, lineType=cv2.LINE_8, shift=8)
        cv2.polylines(img, [self._pixels(outline)], True, OCCUPIED, 1, lineType=cv2.LINE_8, shift=8)  # the walls
        for polygon in obstacles:  # fillPoly also fills every cell the edge touches, so thin legs stay
            cv2.fillPoly(img, [self._pixels(polygon)], OCCUPIED, lineType=cv2.LINE_8, shift=8)
        return img

    def map_yaml(self, image: str) -> dict[str, Any]:
        return {'image': image, 'resolution': self.resolution,
                'origin': [_r(self.origin[0]), _r(self.origin[1]), 0.0], **MAP_THRESHOLDS}


def _dump(path: str, header: str, content: dict[str, Any]) -> None:
    with open(path, 'w') as out:
        out.write(header)
        yaml.safe_dump(content, out, sort_keys=False, default_flow_style=None, width=120)


def _rel(path: str, start: str) -> str:
    return './' + os.path.relpath(path, start)


def check_track(primitives: list[Any], car: Car, allow_tight: bool) -> list[str]:
    """The turning-radius check on each side; fails unless allow_tight, then only warns."""
    tightest = tightest_corners(primitives)
    limits = {'left': car.steer_max_left, 'right': car.steer_max_right}
    lines, failures = [], []
    for side, radius in tightest.items():
        need = min_radius(car.wheelbase, car.s_max, limits[side]) + RADIUS_MARGIN
        lines.append(f'tightest {side} corner: R {radius:.2f} m, needs >= {need:.2f} m (R_min + {RADIUS_MARGIN} m)')
        if radius < need:
            failures.append(f'the {side} corner of radius {radius:.2f} m is tighter than {need:.2f} m')
    if failures and not allow_tight:
        raise ValueError('; '.join(failures) + ' (--allow-tight lets it through)')
    return lines + [f'WARNING: {f}' for f in failures]


def generate(source_path: str, config_dir: str, car: Car, allow_tight: bool = False) -> list[str]:
    """Write the world's files; returns the report. Raises ValueError when a hard check fails."""
    with open(source_path) as source_file:
        src: dict[str, Any] = yaml.safe_load(source_file)
    name = os.path.splitext(os.path.basename(source_path))[0]
    maps_dir = os.path.dirname(os.path.dirname(os.path.abspath(source_path)))
    out_dir = os.path.join(maps_dir, name)
    config_maps = os.path.join(config_dir, 'maps')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(config_maps, exist_ok=True)

    room, track_src = src['room'], src['track']
    outline = np.asarray(room['outline'], dtype=float)
    objects = src.get('objects', [])
    polygons = [_polygon(obj) for obj in objects]
    width, tape_width = float(track_src['width']), float(track_src.get('tape_width', 0.05))
    primitives = parse_primitives(track_src['primitives'])
    report = check_track(primitives, car, allow_tight)
    line: CenterLine = centerline(primitives, tuple(float(v) for v in track_src['origin'][:2]) + (
        math.radians(float(track_src['origin'][2])),))

    grid = Grid(outline, float(src.get('resolution', 0.05)))
    lidar_obstacles = [p for p, obj in zip(polygons, objects, strict=True) if float(obj['height']) > car.laser_z]
    cv2.imwrite(os.path.join(out_dir, 'room.png'), grid.image(outline, polygons))
    cv2.imwrite(os.path.join(out_dir, 'room_lidar.png'), grid.image(outline, lidar_obstacles))
    header = f'# Generated by make_world from {os.path.relpath(source_path, maps_dir)}. Do not edit: edit the source.\n'
    _dump(os.path.join(out_dir, 'room.yaml'), header, grid.map_yaml('room.png'))
    _dump(os.path.join(out_dir, 'room_lidar.yaml'), header, grid.map_yaml('room_lidar.png'))

    edge = width / 2.0 + tape_width / 2.0  # centre of each tape strip
    _dump(os.path.join(out_dir, 'tape.yaml'), header, {
        'frame': 'sim_world', 'width': width, 'tape_width': tape_width, 'rgb': list(track_src['rgb']),
        'length': _r(line.length), 'centerline': _points(line.xy),
        'lines': [_points(line.offset(edge)), _points(line.offset(-edge))],
    })
    _dump(os.path.join(out_dir, 'scene.yaml'), header, {
        'frame': 'sim_world', 'floor': {'rgb': list(src['floor']['rgb'])},
        'walls': {'outline': _points(outline), 'height': float(room['wall_height']), 'rgb': list(room['rgb'])},
        'objects': [{'polygon': _points(p), 'height': float(o['height']), 'rgb': list(o['rgb'])}
                    for p, o in zip(polygons, objects, strict=True)],
        'tape': {'rgb': list(track_src['rgb']), 'width': tape_width},
    })
    with open(os.path.join(out_dir, 'centerline.csv'), 'w') as csv:
        csv.write('# x_m, y_m, w_right_m, w_left_m\n')
        for x, y in line.xy:
            csv.write(f'{x:.4f}, {y:.4f}, {width / 2.0:.4f}, {width / 2.0:.4f}\n')

    sx, sy, stheta = line.pose_at(float(src['start']['s']))
    _dump(os.path.join(config_maps, f'{name}.yaml'), header.replace(
        'Do not edit', 'Do not edit; regenerated at every start'), {
        'map_path': _rel(os.path.join(out_dir, 'room.yaml'), config_maps),
        'sx': _r(sx), 'sy': _r(sy), 'stheta': _r(stheta),
        'world': {key: _rel(os.path.join(out_dir, f'{key}.yaml'), config_maps)
                  for key in ('room_lidar', 'scene', 'tape')},
    })
    tape_length = sum(float(np.sum(np.hypot(*np.diff(np.vstack((pts, pts[:1])), axis=0).T)))
                      for pts in (line.offset(edge), line.offset(-edge)))
    return report + [
        f'lap {line.length:.2f} m, tape {tape_length:.1f} m for both lines',
        f'start (rear axle): x {sx:.3f}, y {sy:.3f}, yaw {stheta:.3f} rad',
        f'wrote {out_dir}/ and {os.path.join(config_maps, name + ".yaml")}',
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='make_world', description=__doc__.splitlines()[0])
    parser.add_argument('source', help='<maps>/src/<world>.yaml')
    parser.add_argument('--config-dir', default=os.path.expanduser('~/ws/config'),
                        help="the car's config directory; car/ files it lacks come from the defaults")
    parser.add_argument('--allow-tight', action='store_true', help='warn instead of failing on tight corners')
    args = parser.parse_args(argv)
    from polimi_sim.world.car import load_car  # needs f1tenth_gym and the package's defaults

    try:
        for line in generate(args.source, args.config_dir, load_car(args.config_dir), args.allow_tight):
            print(line)
    except (ValueError, KeyError, FileNotFoundError) as ex:
        print(f'make_world: {ex}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
