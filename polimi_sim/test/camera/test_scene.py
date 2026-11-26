import math

import numpy as np
import pytest

from polimi_sim.camera.scene import build_scene, tape_chunks


def _area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))) / 2.0


def _circle(radius: float, step: float = 0.05) -> np.ndarray:
    a = np.arange(0.0, 2 * math.pi, step / radius)
    return np.column_stack((radius * np.cos(a), radius * np.sin(a)))


SCENE = {
    'floor': {'rgb': [190, 190, 190]},
    'walls': {'outline': [[0, 0], [0, 8], [10, 8], [10, 0]], 'height': 2.5, 'rgb': [225, 225, 220]},
    'objects': [{'polygon': [[4, 3], [5, 3], [5, 4], [4, 4]], 'height': 0.5, 'rgb': [60, 110, 60]}],
}
TAPE = {'tape_width': 0.05, 'rgb': [20, 20, 20], 'lines': [_circle(1.5).tolist(), _circle(2.55).tolist()]}


def test_tape_chunks_share_their_end_edge_and_never_overlap() -> None:
    line, width = _circle(2.0), 0.05
    chunks = tape_chunks(line, width)
    half = [len(c) // 2 for c in chunks]
    for (a, ha), b in zip(zip(chunks, half, strict=True), chunks[1:] + chunks[:1], strict=True):
        assert a[ha - 1] == pytest.approx(b[0])
        assert a[ha] == pytest.approx(b[-1])
        assert float(np.sum(np.linalg.norm(np.diff(a[:ha], axis=0), axis=1))) <= 1.0 + 1e-9
    annulus = math.pi * ((2.0 + width / 2) ** 2 - (2.0 - width / 2) ** 2)
    assert sum(_area(c) for c in chunks) == pytest.approx(annulus, rel=0.01)


def test_walls_are_panels_of_at_most_a_metre_facing_into_the_room() -> None:
    scene = build_scene(SCENE, TAPE)
    walls = [p for p in scene.solids if p.points[:, 2].max() == pytest.approx(2.5)]
    assert len(walls) == 10 + 8 + 10 + 8
    for wall in walls:
        assert np.linalg.norm(wall.points[1] - wall.points[0]) <= 1.0 + 1e-9
        assert wall.normal is not None
        assert np.dot(wall.normal, np.array([5.0, 4.0, 1.25]) - wall.points.mean(axis=0)) > 0.0


def test_an_object_has_outward_sides_and_a_top() -> None:
    scene = build_scene(SCENE, TAPE)
    box = [p for p in scene.solids if p.rgb == (60, 110, 60)]
    assert len(box) == 5
    for face in box:
        assert face.normal is not None
        assert np.dot(face.normal, face.points.mean(axis=0) - np.array([4.5, 3.5, 0.25])) > 0.0
    assert len(scene.floor) == 1
    assert len(scene.tape) == sum(len(tape_chunks(np.asarray(line), 0.05)) for line in TAPE['lines'])
