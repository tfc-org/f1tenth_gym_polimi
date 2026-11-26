"""render_frame: one camera frame of a world at a pose, without ROS; --footprint reports what the mount sees.

The world must be generated (make_world, or simcar <world> once). Writes the colour image as a PNG.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

from polimi_sim.camera.draw import Renderer
from polimi_sim.camera.intrinsics import (
    FloatArray,
    Intrinsics,
    OpticalMount,
    camera_in_world,
    load_camera,
    load_camera_mount,
    optical_mount,
)
from polimi_sim.camera.scene import load_scene
from polimi_sim.common.gym_config import load_yaml, resolve_relative

STEP = 0.01
REACH = 6.0


def in_view(intrinsics: Intrinsics, mount: OpticalMount, points: FloatArray) -> np.ndarray:
    """Which base_link points (n, 3) project inside the image."""
    cam = (points - mount.position) @ mount.rotation
    ahead = cam[:, 2] > 1e-6
    uv = intrinsics.project(np.where(ahead[:, None], cam, 1.0))
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= intrinsics.width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= intrinsics.height - 1)
    result: np.ndarray = ahead & inside
    return result


def floor_hit(intrinsics: Intrinsics, mount: OpticalMount, u: float, v: float) -> FloatArray | None:
    """Where the ray of pixel (u, v) meets the floor, in base_link; None above the horizon."""
    k = intrinsics
    ray = mount.rotation @ np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
    if ray[2] >= -1e-9:
        return None
    hit: FloatArray = mount.position - ray * mount.position[2] / ray[2]
    return hit


def first_pair_in_view(intrinsics: Intrinsics, mount: OpticalMount, left: FloatArray, right: FloatArray) -> int | None:
    """Index of the first sample at which both lines' points are in view."""
    both = np.flatnonzero(in_view(intrinsics, mount, left) & in_view(intrinsics, mount, right))
    return int(both[0]) if len(both) else None


def footprint(intrinsics: Intrinsics, mount: OpticalMount, edge: float, corners: dict[str, float]) -> list[str]:
    """The floor the camera sees, and how far ahead both tape lines (at +-edge from the centre line) come into view."""
    k = intrinsics
    near = floor_hit(k, mount, k.cx, k.height - 1)
    far = floor_hit(k, mount, k.cx, 0)
    lines = []
    if near is None:
        lines.append('floor: not in view (the camera looks above the horizon)')
    else:
        far_text = f'{far[0]:.2f} m' if far is not None else 'the horizon'
        lines.append(f'floor: from {near[0]:.2f} m ahead of base_link to {far_text}')
        y = np.linspace(-10.0, 10.0, 4001)
        for x in (float(near[0]) + STEP, 1.0, 2.0):
            seen = y[in_view(k, mount, np.column_stack((np.full_like(y, x), y, np.zeros_like(y))))]
            if x > near[0] and len(seen):
                lines.append(f'  at {x:.2f} m: {seen.max() - seen.min():.2f} m wide ({seen.max():.2f} left to {seen.min():.2f} right)')
    s = np.arange(0.0, REACH, STEP)
    zero = np.zeros_like(s)
    i = first_pair_in_view(k, mount, np.column_stack((s, zero + edge, zero)), np.column_stack((s, zero - edge, zero)))
    lines.append(f'straight: both lines in view from {s[i]:.2f} m ahead' if i is not None else
                 f'straight: the two lines are never in view together within {REACH:.0f} m')
    for side, radius in corners.items():
        sign = 1.0 if side == 'left' else -1.0
        phi = s / radius
        seen = []
        for name, d in (('inner', -edge), ('outer', edge)):
            ring = np.column_stack(((radius + d) * np.sin(phi), sign * (radius - (radius + d) * np.cos(phi)), zero))
            visible = np.flatnonzero(in_view(k, mount, ring))
            seen.append(f'{name} line from {s[visible[0]]:.2f} m along it' if len(visible) else f'{name} line never in view')
        lines.append(f'tightest {side} corner (R {radius:.2f} m), car on the centre line: {", ".join(seen)}')
    return lines


def _defaults_dir() -> str:
    from ament_index_python.packages import get_package_share_directory

    return os.path.join(get_package_share_directory('polimi_sim'), 'config', 'defaults')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='render_frame', description=__doc__.splitlines()[0])
    parser.add_argument('world', help='a generated world: <config-dir>/maps/<world>.yaml')
    parser.add_argument('--pose', nargs=3, type=float, metavar=('X', 'Y', 'YAW'), help='rear axle in the world, yaw in rad')
    parser.add_argument('--profile', default='', help="the camera file's profile (default: its own)")
    parser.add_argument('--out', default='frame.png', help='colour PNG to write')
    parser.add_argument('--footprint', action='store_true', help='print the floor the camera sees and where both lines come into view')
    parser.add_argument('--camera', default='gemini_335l', help='a file name in camera/')
    parser.add_argument('--config-dir', default=os.path.expanduser('~/ws/config'), help="the car's config directory")
    parser.add_argument('--maps-dir', default=os.path.expanduser('~/ws/maps'), help='where <world> has its source, for --footprint')
    args = parser.parse_args(argv)
    if args.pose is None and not args.footprint:
        parser.error('give --pose, --footprint or both')
    from polimi_sim.common.overlay import resolve

    try:
        defaults = _defaults_dir()
        camera = load_camera(resolve(f'camera/{args.camera}.yaml', args.config_dir, defaults).path, args.profile)
        mount = optical_mount(load_camera_mount(resolve('car/camera_mount.yaml', args.config_dir, defaults).path),
                              camera.color_in_link)
        map_file = os.path.join(args.config_dir, 'maps', f'{args.world}.yaml')
        world = {key: resolve_relative(str(v), os.path.dirname(map_file)) for key, v in load_yaml(map_file).get('world', {}).items()}
        if 'scene' not in world:
            raise ValueError(f'{map_file} has no world: block (render_frame draws room worlds only)')
        if args.footprint:
            from polimi_sim.world.track import parse_primitives, tightest_corners

            tape = load_yaml(world['tape'])
            source = os.path.join(args.maps_dir, 'src', f'{args.world}.yaml')
            corners = tightest_corners(parse_primitives(load_yaml(source)['track']['primitives'])) if os.path.isfile(source) else {}
            for line in footprint(camera.intrinsics, mount, float(tape['width']) / 2 + float(tape['tape_width']) / 2, corners):
                print(line)
        if args.pose is not None:
            img = Renderer(load_scene(world['scene'], world['tape']), camera.intrinsics).render(
                *camera_in_world(args.pose[0], args.pose[1], args.pose[2], mount))
            if not cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
                raise ValueError(f'cannot write {args.out}')
            print(f'wrote {args.out} ({camera.profile}, pose {args.pose[0]:.2f} {args.pose[1]:.2f} {math.degrees(args.pose[2]):.1f} deg)')
    except (ValueError, KeyError, FileNotFoundError) as ex:
        print(f'render_frame: {ex}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
