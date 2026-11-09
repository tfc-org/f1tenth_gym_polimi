import copy

import pytest
import yaml

from polimi_sim.common.gym_config import build_gym_config, load_lidar, load_mount, write_gym_config

BASE = {'bridge': {'ros__parameters': {'ego_drive_topic': 'sim/drive', 'scale': 1.0, 'kb_teleop': False}}}
MAP = {'map_path': 'maps/Spielberg', 'sx': 0.0, 'sy': 0.0, 'stheta': 0.26}
LIDAR = {'fov_deg': 270.0, 'resolution_deg': 0.2, 'range_min': 0.05, 'range_max': 25.0, 'noise_std': 0.01}
MOUNT = {'x': 0.27, 'y': 0.0, 'z': 0.11, 'yaw': 0.0, 'roll': 0.0}


def test_map_lidar_and_mount_end_up_in_the_gym_config() -> None:
    base = copy.deepcopy(BASE)
    params = build_gym_config(base, MAP, LIDAR, MOUNT)['bridge']['ros__parameters']
    assert base == BASE  # inputs untouched
    assert params['ego_drive_topic'] == 'sim/drive'
    assert (params['map_path'], params['sx'], params['stheta']) == ('maps/Spielberg', 0.0, 0.26)
    assert params['scan_num_beams'] == 1351 and isinstance(params['scan_num_beams'], int)
    assert (params['scan_angle_min'], params['scan_angle_max']) == (-135.0, 135.0)
    assert (params['scan_range_min'], params['scan_range_max']) == (0.05, 25.0)
    assert params['lidar_base_link_to_lidar_tf'] == [0.27, 0.0, 0.0]
    assert params['lidar_enabled'] is True and params['lidar_noise_std'] == 0.0


def test_map_and_lidar_are_independent() -> None:
    other_map = dict(MAP, map_path='maps/levine', sx=-12.0)
    other_lidar = dict(LIDAR, fov_deg=360.0, resolution_deg=0.5)
    a = build_gym_config(BASE, other_map, LIDAR, MOUNT)['bridge']['ros__parameters']
    b = build_gym_config(BASE, MAP, other_lidar, MOUNT)['bridge']['ros__parameters']
    assert a['scan_num_beams'] == 1351 and a['map_path'] == 'maps/levine'
    assert b['scan_num_beams'] == 721 and b['map_path'] == 'maps/Spielberg'


def test_a_map_file_can_override_any_gym_parameter() -> None:
    params = build_gym_config(BASE, dict(MAP, scale=2.0), LIDAR, MOUNT)['bridge']['ros__parameters']
    assert params['scale'] == 2.0


def test_missing_keys_are_reported() -> None:
    with pytest.raises(ValueError, match='stheta'):
        build_gym_config(BASE, {'map_path': 'x', 'sx': 0.0, 'sy': 0.0}, LIDAR, MOUNT)
    with pytest.raises(ValueError, match='fov_deg'):
        build_gym_config(BASE, MAP, {'resolution_deg': 0.2, 'range_min': 0.05, 'range_max': 25.0}, MOUNT)


def test_files_round_trip(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    lidar_file, mount_file = tmp_path / 'lidar.yaml', tmp_path / 'mount.yaml'
    lidar_file.write_text(yaml.safe_dump({'/**': {'ros__parameters': LIDAR}}))
    mount_file.write_text(yaml.safe_dump({'base_link_to_laser': MOUNT}))
    assert load_lidar(str(lidar_file)) == LIDAR
    assert load_mount(str(mount_file)) == MOUNT
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    path = write_gym_config(build_gym_config(BASE, MAP, LIDAR, MOUNT), 'spielberg_sl450')
    assert path.endswith('polimi_sim/gym_spielberg_sl450.yaml')
    with open(path) as written:
        assert yaml.safe_load(written)['bridge']['ros__parameters']['scan_num_beams'] == 1351
