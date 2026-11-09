from __future__ import annotations

import os
from typing import NamedTuple

DEFAULT_LABEL = '# Default: stock f1tenth / datasheet values, not a measured car. Pass config_dir:= for a real car.'


class Resolved(NamedTuple):
    relative: str
    path: str
    from_config_dir: bool


def resolve(relative: str, config_dir: str, defaults_dir: str) -> Resolved:
    """<config_dir>/<relative> if that file exists, otherwise <defaults_dir>/<relative>."""
    if config_dir:
        own = os.path.join(config_dir, relative)
        if os.path.isfile(own):
            return Resolved(relative, own, True)
    default = os.path.join(defaults_dir, relative)
    if not os.path.isfile(default):
        raise FileNotFoundError(f'{relative} is in neither {config_dir or "(no config_dir)"} nor {defaults_dir}')
    return Resolved(relative, default, False)


def report(files: list[Resolved], config_dir: str) -> tuple[list[str], list[str]]:
    """Info lines (one per file, then a summary) and a warning per file config_dir lacks."""
    info = [f'config: {f.relative} <- {config_dir if f.from_config_dir else "defaults"}' for f in files]
    own = sum(f.from_config_dir for f in files)
    info.append(f'config: {own} from {config_dir}, {len(files) - own} defaults' if config_dir
                else f'config: {len(files)} defaults')
    warnings = [f'config: {f.relative} not in {config_dir}, using the default'
                for f in files if config_dir and not f.from_config_dir]
    return info, warnings
