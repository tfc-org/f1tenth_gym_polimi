import copy
import math
import os
import pathlib

import cv2
import numpy as np
import pytest
import yaml

from polimi_sim.common.gym_config import resolve_relative
from polimi_sim.world.generate import FREE, OCCUPIED, Car, generate
from polimi_sim.world.track import centerline, min_radius, parse_primitives

CAR = Car(wheelbase=0.3302, s_max=0.4189, steer_max_left=0.3135, steer_max_right=0.2634, laser_z=0.11)
LOOP = [
    {'straight': 4.0}, {'arc': {'radius': 2.0, 'angle': 90}}, {'arc': {'radius': 1.5, 'angle': 90}},
    {'straight': 0.75}, {'arc': {'radius': 1.5, 'angle': 30}}, {'arc': {'radius': 1.5, 'angle': -60}},
    {'arc': {'radius': 1.5, 'angle': 30}}, {'straight': 0.75}, {'arc': {'radius': 1.75, 'angle': 180}},
]
SOURCE = {
    'room': {'outline': [[0, 0], [10, 0], [10, 8], [0, 8]], 'wall_height': 2.5, 'rgb': [230, 230, 225]},
    'floor': {'rgb': [190, 190, 190]},
    'track': {'origin': [2.875, 2.25, 0.0], 'primitives': LOOP, 'width': 1.0, 'tape_width': 0.05, 'rgb': [20, 20, 20]},
    'start': {'s': 2.0},
    'resolution': 0.05,
    'objects': [
        {'x': 4.5, 'y': 4.0, 'length': 0.4, 'width': 0.4, 'height': 0.08, 'rgb': [200, 60, 40]},
        {'x': 6.01, 'y': 4.01, 'length': 0.02, 'width': 0.02, 'height': 0.5, 'rgb': [60, 60, 60]},  # misses every cell centre
    ],
}


def _write(tmp_path: pathlib.Path, source: dict) -> tuple[str, str]:
    (tmp_path / 'maps' / 'src').mkdir(parents=True, exist_ok=True)
    path = tmp_path / 'maps' / 'src' / 'demo.yaml'
    path.write_text(yaml.safe_dump(source))
    return str(path), str(tmp_path / 'config')


def _cell(world: pathlib.Path, image: str, x: float, y: float) -> int:
    meta = yaml.safe_load((world / image.replace('.png', '.yaml')).read_text())
    img = cv2.imread(str(world / image), cv2.IMREAD_GRAYSCALE)
    col = int((x - meta['origin'][0]) / meta['resolution'])
    row = img.shape[0] - 1 - int((y - meta['origin'][1]) / meta['resolution'])
    return int(img[row, col])


def test_the_demo_loop_closes_and_an_open_chain_fails_with_its_gap() -> None:
    line = centerline(parse_primitives(LOOP), (0.0, 0.0, 0.0))
    assert line.length == pytest.approx(5.5 + 4.5 * math.pi)  # straights, then 2.0, 1.5, the chicane and 1.75 m arcs
    with pytest.raises(ValueError, match=r'does not close: .* 0\.500 m'):
        centerline(parse_primitives([*LOOP[:-1], {'arc': {'radius': 1.75, 'angle': 180}}, {'straight': 0.5}]),
                   (0.0, 0.0, 0.0))


def test_the_minimum_radius_differs_left_and_right() -> None:
    assert min_radius(CAR.wheelbase, CAR.s_max, CAR.steer_max_left) == pytest.approx(1.02, abs=0.01)
    assert min_radius(CAR.wheelbase, CAR.s_max, CAR.steer_max_right) == pytest.approx(1.22, abs=0.01)


def test_a_corner_tighter_than_the_car_fails_unless_allowed(tmp_path: pathlib.Path) -> None:
    tight = copy.deepcopy(SOURCE)
    tight['track']['primitives'][5] = {'arc': {'radius': 1.3, 'angle': -60}}  # right turn, needs 1.42 m
    source, config = _write(tmp_path, tight)
    with pytest.raises(ValueError, match='right corner of radius 1.30'):
        generate(source, config, CAR)
    left = copy.deepcopy(SOURCE)
    left['track']['primitives'][2] = {'arc': {'radius': 1.2, 'angle': 90}}  # left turn, needs 1.22 m
    source, config = _write(tmp_path, left)
    with pytest.raises(ValueError, match='left corner of radius 1.20'):
        generate(source, config, CAR)


def test_the_maps_and_the_lidar_slice(tmp_path: pathlib.Path) -> None:
    source, config = _write(tmp_path, SOURCE)
    generate(source, config, CAR)
    world = tmp_path / 'maps' / 'demo'
    for image in ('room.png', 'room_lidar.png'):
        values = set(np.unique(cv2.imread(str(world / image), cv2.IMREAD_GRAYSCALE)))
        assert values == {FREE, OCCUPIED}, image  # no grey
        assert _cell(world, image, -0.05, 4.0) == OCCUPIED  # outside the room
        assert _cell(world, image, 2.0, 1.0) == FREE
        assert _cell(world, image, 6.01, 4.01) == OCCUPIED  # a 2 cm post still fills its cell
    assert _cell(world, 'room.png', 4.5, 4.0) == OCCUPIED  # the low box collides
    assert _cell(world, 'room_lidar.png', 4.5, 4.0) == FREE  # but is under the laser


def test_the_config_names_the_world_and_nothing_turns_on_the_gyms_laps(tmp_path: pathlib.Path) -> None:
    source, config = _write(tmp_path, SOURCE)
    generate(source, config, CAR)
    map_file = pathlib.Path(config) / 'maps' / 'demo.yaml'
    entry = yaml.safe_load(map_file.read_text())
    for path in (entry['map_path'], *entry['world'].values()):
        assert path.startswith('./') and os.path.isfile(resolve_relative(path, str(map_file.parent)))
    world = tmp_path / 'maps' / 'demo'
    assert (world / 'centerline.csv').is_file()
    assert not (world / 'room_centerline.csv').exists() and not (world / 'room_raceline.csv').exists()
    assert not {'centerline', 'raceline'} & set(yaml.safe_load((world / 'room.yaml').read_text()))


def test_the_start_puts_the_rear_axle_on_the_centre_line(tmp_path: pathlib.Path) -> None:
    source, config = _write(tmp_path, SOURCE)
    generate(source, config, CAR)
    entry = yaml.safe_load((pathlib.Path(config) / 'maps' / 'demo.yaml').read_text())
    assert (entry['sx'], entry['sy'], entry['stheta']) == pytest.approx((2.875 + 2.0, 2.25, 0.0), abs=1e-3)


def test_regenerating_writes_the_same_bytes(tmp_path: pathlib.Path) -> None:
    source, config = _write(tmp_path, SOURCE)
    generate(source, config, CAR)
    files = sorted(p for p in tmp_path.rglob('*') if p.is_file() and 'src' not in p.parts)
    first = {p: p.read_bytes() for p in files}
    generate(source, config, CAR)
    assert {p: p.read_bytes() for p in files} == first
