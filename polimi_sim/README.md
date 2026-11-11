# polimi_sim

The simulator as the car: f1tenth_gym_ros (the package next to this folder) plus models of the car's LiDAR, actuation and wheel odometry, the car's own command mux and its frames. The stack talks to it through the same topics and frames as on the real car.

The design (each model, its parameters, the frames, the known limits) is in `TDD_Gym`, in [f1tenth_polimi](https://github.com/tfc-org/f1tenth_polimi) under `docs/software/PDF_Documentation/sim/`.

## What the workspace needs

- This repo (branch `dev-jazzy-polimi`): `f1tenth_gym_ros` and `polimi_sim`. colcon and rosdep don't look inside a package, so name this folder explicitly: `colcon build --base-paths src src/f1tenth_gym_ros/polimi_sim`, and the same paths for `rosdep install --from-paths`.
- [f1tenth_system](https://github.com/tfc-org/f1tenth_system_polimi), branch `jazzy-polimi`: `polimi_sim` drives through the car's `ackermann_mux` with `f1tenth_stack`'s `mux.yaml`. `ackermann_mux` has no Jazzy apt release, so it is built from source there.
- f1tenth_gym `dev-jax`, installed with pip (Python 3.12, ROS 2 Jazzy). Set `JAX_PLATFORMS=cpu` when there is no GPU.

## Run

```bash
ros2 launch polimi_sim sim_car.launch.py                                # defaults: spielberg, sl450
ros2 launch polimi_sim sim_car.launch.py map:=levine ideal:=true        # no imperfections
ros2 launch polimi_sim sim_car.launch.py config_dir:=$HOME/ws/config    # a real car's configs
```

## Configs

`config/defaults/` holds every file the launch needs, at stock f1tenth and datasheet values, never a measured car; each file says so on its first line:

| File | What |
|---|---|
| `sim/gym.yaml` | the bridge's common parameters |
| `maps/spielberg.yaml`, `maps/levine.yaml` | map and start pose (rear axle) |
| `lidar/sl450.yaml` | the laser model |
| `car/laser_mount.yaml` | `base_link` to `laser` |
| `actuation_model.yaml`, `odom_model.yaml`, `teleop_bridge.yaml` | the other models |

With `config_dir:=<dir>`, each file is taken from `<dir>/<same path>` if it exists, else from the defaults. The launch logs where each one came from:

```
config: car/laser_mount.yaml <- /home/me/ws/config
config: sim/gym.yaml <- defaults
config: 2 from /home/me/ws/config, 5 defaults
```

It warns when `config_dir` lacks one of the measured car's files (`car/`, `lidar/`); the other defaults are expected. `actuation_params:=`, `odom_params:=` and `teleop_params:=` take a model file directly.

## Truth frame and the stack's topics

- The gym's truth is in `sim_world` (`truth_frame` in `sim/gym.yaml`, a parameter of this fork's bridge): the TF `sim_world -> ego_racecar/base_link`, `/ego_racecar/odom` and the gym's map. `map` stays free for SLAM and AMCL.
- Two gym topics move out of the stack's way: the bridge's `/initialpose` to `/sim/initialpose` (a pose there teleports the car) and map_server's `/map` to `/sim/map`. Each remap names its node (`bridge:/initialpose`), because the gym's launch also starts `foxglove_bridge`, which an unscoped remap would reach too.
- `odom_model`'s `debug_map_to_odom` (`static`, `truth`) publishes `map -> odom` under its `map_frame`, plus a static identity `sim_world -> map`, so the stack's tree and the truth line up.
- Foxglove: open `config/foxglove/polimi_sim.json` (fixed frame `sim_world`, `/sim/map`, the pose tool on `/sim/initialpose`); the launch logs its path.

## Tests

```bash
python3 -m pytest test -q
```

`test/lidar/test_raycast.py` needs f1tenth_gym and f1tenth_gym_ros, and `test/launch/` needs `launch_ros`, so they run only in a simulator image. Lint and types: `ruff check .` and `mypy` from this folder (`pyproject.toml`).
