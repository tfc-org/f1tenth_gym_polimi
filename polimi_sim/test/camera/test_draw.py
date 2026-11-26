import math

import numpy as np
import pytest

from polimi_sim.camera.draw import Renderer, shade
from polimi_sim.camera.intrinsics import Intrinsics, camera_in_world, optical_mount
from polimi_sim.camera.scene import Polygon, Scene

K = Intrinsics(1280, 720, 596.8, 596.8, 639.5, 359.5, (0.0,) * 5)
PITCH = math.radians(15.0)
MOUNT = optical_mount({'x': 0.2, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': PITCH, 'yaw': 0.0}, (0.0, 0.0, 0.0))
FLOOR, TAPE, BOX, WALL = (190, 190, 190), (20, 20, 20), (200, 60, 40), (90, 90, 200)
UP = np.array([0.0, 0.0, 1.0])


def _square(x: float, y: float, half: float, z: float = 0.0) -> np.ndarray:
    return np.array([[x - half, y - half, z], [x + half, y - half, z], [x + half, y + half, z], [x - half, y + half, z]])


def _box(x: float, y: float, half: float, height: float) -> list[Polygon]:
    c = _square(x, y, half)[:, :2]
    faces = []
    for a, b in zip(c, np.roll(c, -1, axis=0), strict=True):
        n = np.array([b[1] - a[1], a[0] - b[0], 0.0])
        faces.append(Polygon(np.array([[*a, 0.0], [*b, 0.0], [*b, height], [*a, height]]), BOX, n / np.linalg.norm(n)))
    faces.append(Polygon(_square(x, y, half, height), BOX, UP))
    return faces


def _floor() -> Polygon:
    return Polygon(np.array([[-5.0, -20.0, 0.0], [30.0, -20.0, 0.0], [30.0, 20.0, 0.0], [-5.0, 20.0, 0.0]]), FLOOR)


def _expected_pixel(point: tuple[float, float, float]) -> tuple[int, int]:
    """Independent of the code: the camera at (0.2, 0, 0.2), pitched down, base_link at the origin facing +x."""
    d = np.asarray(point) - np.array([0.2, 0.0, 0.2])
    z = np.array([math.cos(PITCH), 0.0, -math.sin(PITCH)])
    x = np.array([0.0, -1.0, 0.0])
    y = np.array([-math.sin(PITCH), 0.0, -math.cos(PITCH)])
    return round(K.cx + K.fx * d @ x / (d @ z)), round(K.cy + K.fy * d @ y / (d @ z))


def _render(scene: Scene) -> np.ndarray:
    return Renderer(scene, K).render(*camera_in_world(0.0, 0.0, 0.0, MOUNT)).copy()


def test_a_floor_point_lands_on_its_pixel() -> None:
    scene = Scene(floor=[_floor()], tape=[Polygon(_square(2.0, 0.0, 0.08), TAPE), Polygon(_square(1.5, 0.6, 0.08), TAPE)])
    img = _render(scene)
    for point in [(2.0, 0.0, 0.0), (1.5, 0.6, 0.0)]:
        u, v = _expected_pixel(point)
        assert tuple(img[v, u]) == TAPE
        assert tuple(img[v + 25, u]) == FLOOR
    assert _expected_pixel((2.0, 0.0, 0.0))[1] == pytest.approx(268.6, abs=1)
    assert _expected_pixel((1.5, 0.6, 0.0))[0] < K.cx


def test_a_floor_through_the_near_plane_and_the_sides_fills_completely() -> None:
    img = _render(Scene(floor=[_floor()]))
    horizon = K.cy - K.fy * math.tan(PITCH)
    assert (img[int(horizon) + 20:] == FLOOR).all()
    assert (img[: int(horizon) - 5] == Scene().background).all()


def test_a_box_in_front_hides_the_tape_and_the_wall_behind_it() -> None:
    wall = Polygon(np.array([[5.0, -3.0, 0.0], [5.0, 3.0, 0.0], [5.0, 3.0, 2.5], [5.0, -3.0, 2.5]]), WALL, np.array([-1.0, 0.0, 0.0]))
    scene = Scene(floor=[_floor()], tape=[Polygon(_square(3.0, 0.0, 0.05), TAPE)], solids=[*_box(2.0, 0.0, 0.2, 0.5), wall])
    img = _render(scene)
    front = shade(BOX, float(np.linalg.norm(np.array([1.8, 0.0, 0.25]) - [0.2, 0.0, 0.2])))
    u, v = _expected_pixel((3.0, 0.0, 0.0))
    assert tuple(img[v, u]) == front
    u, v = _expected_pixel((1.8, 0.0, 0.45))
    assert tuple(img[v, u]) == front


def test_the_top_of_a_low_box_shows_with_the_wall_behind_above_it() -> None:
    wall = Polygon(np.array([[5.0, -3.0, 0.0], [5.0, 3.0, 0.0], [5.0, 3.0, 2.5], [5.0, -3.0, 2.5]]), WALL, np.array([-1.0, 0.0, 0.0]))
    scene = Scene(floor=[_floor()], solids=[wall, *_box(1.5, 0.0, 0.2, 0.08)])
    img = _render(scene)
    top = img[_expected_pixel((1.5, 0.0, 0.08))[::-1]]
    assert tuple(top) == shade(BOX, float(np.linalg.norm(np.array([1.5, 0.0, 0.08]) - [0.2, 0.0, 0.2])))
    u, v = _expected_pixel((5.0, 0.0, 0.5))
    assert tuple(img[v, u]) == shade(WALL, float(np.linalg.norm(np.array([5.0, 0.0, 1.25]) - [0.2, 0.0, 0.2])))


def test_a_face_seen_from_behind_is_not_drawn() -> None:
    wall = Polygon(np.array([[3.0, -3.0, 0.0], [3.0, 3.0, 0.0], [3.0, 3.0, 2.5], [3.0, -3.0, 2.5]]), WALL, np.array([1.0, 0.0, 0.0]))
    img = _render(Scene(floor=[_floor()], solids=[wall]))
    u, v = _expected_pixel((3.0, 0.0, 0.5))
    assert tuple(img[v, u]) == Scene().background
