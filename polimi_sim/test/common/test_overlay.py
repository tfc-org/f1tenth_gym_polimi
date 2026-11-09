import os
from pathlib import Path

import pytest

from polimi_sim.common.overlay import DEFAULT_LABEL, resolve, report

DEFAULTS = Path(__file__).resolve().parents[2] / 'config' / 'defaults'


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[str, str]:
    defaults, own = tmp_path / 'defaults', tmp_path / 'own'
    for rel in ('car/laser_mount.yaml', 'sim/gym.yaml'):
        (defaults / rel).parent.mkdir(parents=True, exist_ok=True)
        (defaults / rel).write_text('default')
    (own / 'car').mkdir(parents=True)
    (own / 'car/laser_mount.yaml').write_text('own')
    return str(own), str(defaults)


def test_config_dir_wins_over_the_default(dirs: tuple[str, str]) -> None:
    own, defaults = dirs
    assert resolve('car/laser_mount.yaml', own, defaults) == (
        'car/laser_mount.yaml', os.path.join(own, 'car/laser_mount.yaml'), True)


def test_missing_file_falls_back_with_a_warning(dirs: tuple[str, str]) -> None:
    own, defaults = dirs
    files = [resolve(rel, own, defaults) for rel in ('car/laser_mount.yaml', 'sim/gym.yaml')]
    info, warnings = report(files, own)
    assert files[1].path == os.path.join(defaults, 'sim/gym.yaml') and not files[1].from_config_dir
    assert info == [f'config: car/laser_mount.yaml <- {own}', 'config: sim/gym.yaml <- defaults',
                    f'config: 1 from {own}, 1 defaults']
    assert warnings == [f'config: sim/gym.yaml not in {own}, using the default']


def test_empty_config_dir_uses_only_defaults(dirs: tuple[str, str]) -> None:
    _, defaults = dirs
    files = [resolve(rel, '', defaults) for rel in ('car/laser_mount.yaml', 'sim/gym.yaml')]
    assert not any(f.from_config_dir for f in files)
    assert report(files, '') == (['config: car/laser_mount.yaml <- defaults', 'config: sim/gym.yaml <- defaults',
                                  'config: 2 defaults'], [])


def test_missing_everywhere_raises(dirs: tuple[str, str]) -> None:
    own, defaults = dirs
    with pytest.raises(FileNotFoundError):
        resolve('maps/nowhere.yaml', own, defaults)


def test_every_default_file_carries_the_label() -> None:
    files = sorted(DEFAULTS.rglob('*.yaml'))
    assert len(files) == 8
    for path in files:
        assert path.read_text().splitlines()[0] == DEFAULT_LABEL, path
