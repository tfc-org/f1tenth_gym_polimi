import math

import pytest

from polimi_sim.camera.intrinsics import Intrinsics, optical_mount
from polimi_sim.camera.render_frame import floor_hit, footprint

K = Intrinsics(1280, 720, 596.8, 596.8, 639.5, 359.5, (0.0,) * 5)
MOUNT = optical_mount({'x': 0.2, 'y': 0.0, 'z': 0.2, 'roll': 0.0, 'pitch': math.radians(15.0), 'yaw': 0.0}, (0.0, 0.0, 0.0))


def test_the_bottom_row_meets_the_floor_where_the_lower_edge_of_the_view_does() -> None:
    hit = floor_hit(K, MOUNT, K.cx, K.height - 1)
    assert hit is not None
    below = math.radians(15.0) + math.atan((K.height - 1 - K.cy) / K.fy)
    assert hit[0] == pytest.approx(0.2 + 0.2 / math.tan(below), abs=1e-6)
    assert hit[1] == pytest.approx(0.0, abs=1e-9) and hit[2] == pytest.approx(0.0, abs=1e-9)
    assert floor_hit(K, MOUNT, K.cx, 0) is None


def test_the_footprint_reports_the_floor_and_where_both_lines_come_into_view() -> None:
    lines = footprint(K, MOUNT, 0.525, {'left': 1.5, 'right': 1.5})
    assert lines[0] == 'floor: from 0.39 m ahead of base_link to the horizon'
    straight = next(line for line in lines if line.startswith('straight'))
    assert float(straight.split('from ')[1].split(' m')[0]) > 0.39
    assert sum(line.startswith('tightest') for line in lines) == 2
