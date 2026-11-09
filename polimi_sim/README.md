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

It also warns for each file `config_dir` lacks. `actuation_params:=`, `odom_params:=` and `teleop_params:=` take a model file directly.

## Tests

```bash
python3 -m pytest test -q
```

`test/lidar/test_raycast.py` needs f1tenth_gym and f1tenth_gym_ros, so it runs only in a simulator image.
