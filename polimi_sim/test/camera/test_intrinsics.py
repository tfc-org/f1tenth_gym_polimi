import math
import pathlib

import cv2
import numpy as np
import pytest

from polimi_sim.camera.intrinsics import (
    BODY_TO_OPTICAL,
    OPTICAL_RPY,
    Intrinsics,
    camera_in_world,
    distortion_maps,
    load_camera,
    load_camera_mount,
    optical_mount,
    rotation_rpy,
)

DEFAULTS = pathlib.Path(__file__).parents[2] / 'config' / 'defaults'
CAMERA = str(DEFAULTS / 'camera' / 'gemini_335l.yaml')
MOUNT = str(DEFAULTS / 'car' / 'camera_mount.yaml')
DISTORTED = Intrinsics(1280, 720, 596.8, 596.8, 639.5, 359.5, (-0.12, 0.03, 0.001, -0.0015, -0.004))


def test_the_default_profile_is_the_720p_crop_of_the_94_degree_sensor() -> None:
    camera = load_camera(CAMERA)
    k = camera.intrinsics
    assert (camera.profile, k.width, k.height, camera.rate_hz) == ('720p30', 1280, 720, 30.0)
    assert math.degrees(2 * math.atan(k.width / 2 / k.fx)) == pytest.approx(94.0, abs=0.05)
    assert math.degrees(2 * math.atan(k.height / 2 / k.fy)) == pytest.approx(62.2, abs=0.1)
    assert not any(k.d)


def test_an_unknown_profile_names_the_ones_there_are() -> None:
    with pytest.raises(ValueError, match="no profile '720p60' \\(has 720p30\\)"):
        load_camera(CAMERA, '720p60')


def test_the_mount_needs_every_key(tmp_path: pathlib.Path) -> None:
    assert load_camera_mount(MOUNT)['pitch'] == pytest.approx(math.radians(15.0), abs=1e-4)
    path = tmp_path / 'mount.yaml'
    path.write_text('base_link_to_camera_link: {x: 0.2, y: 0.0, z: 0.2}\n')
    with pytest.raises(ValueError, match='roll, pitch, yaw'):
        load_camera_mount(str(path))


def test_no_distortion_needs_no_remap() -> None:
    assert distortion_maps(load_camera(CAMERA).intrinsics) is None


def test_the_maps_take_each_distorted_pixel_to_its_ideal_source() -> None:
    maps = distortion_maps(DISTORTED)
    assert maps is not None
    k = DISTORTED
    for u, v in [(0, 0), (1279, 719), (100, 600), (640, 360), (1200, 40)]:
        ideal_u, ideal_v = maps[0][v, u], maps[1][v, u]
        ray = np.array([[(ideal_u - k.cx) / k.fx, (ideal_v - k.cy) / k.fy, 1.0]])
        back, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), k.k, np.asarray(k.d))
        assert back.ravel() == pytest.approx([u, v], abs=0.05)


def test_the_optical_axis_follows_the_mount_pitch_and_the_car_yaw() -> None:
    mount = {'x': 0.2, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': math.radians(15.0), 'yaw': 0.0}
    r, c = camera_in_world(1.0, 2.0, math.pi / 2, optical_mount(mount, (0.0, -0.02, 0.0)))
    assert c == pytest.approx([1.02, 2.2, 0.2])
    forward, right, down = r[:, 2], r[:, 0], r[:, 1]
    assert forward == pytest.approx([0.0, math.cos(math.radians(15.0)), -math.sin(math.radians(15.0))])
    assert right == pytest.approx([1.0, 0.0, 0.0])
    assert down[2] < 0.0


def test_the_optical_tf_rotation_is_the_one_the_drawing_uses() -> None:
    assert rotation_rpy(*OPTICAL_RPY) == pytest.approx(BODY_TO_OPTICAL)
