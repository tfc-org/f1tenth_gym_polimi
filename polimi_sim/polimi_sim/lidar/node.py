"""lidar_model: an SL450-like /scan from the gym's ground-truth pose and map."""
from __future__ import annotations

from array import array
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import LaserScan

from polimi_sim.common.ros_utils import RelativeClock, declare, with_updates, yaw_from_quaternion
from polimi_sim.common.se2 import Pose2D, PoseBuffer, Twist2D
from polimi_sim.lidar.effects import (
    EffectsConfig,
    ScanGeometry,
    apply_effects,
    distorted_scan,
    scan_geometry,
)
from polimi_sim.lidar.raycast import BlockCaster, load_gym_track

SL450_RATES_HZ: tuple[float, ...] = (15.0, 20.0, 25.0, 30.0, 40.0)

EFFECT_DEFAULTS: dict[str, Any] = {
    'effects_enabled': True,
    'range_min': 0.05,
    'range_max': 25.0,
    'no_return_value': 'inf',
    'noise_std': 0.01,
    'dropout_prob': 0.002,
    'grazing_max_deg': 80.0,
    'grazing_dropout_prob': 0.5,
    'reflectivity_range_max': 15.0,
    'mixed_pixel_jump': 0.3,
    'mixed_pixel_prob': 0.3,
}


def effects_config(values: dict[str, Any]) -> EffectsConfig:
    fields = {k: v for k, v in values.items() if k not in ('effects_enabled', 'no_return_value')}
    return EffectsConfig(
        enabled=bool(values['effects_enabled']),
        no_return_value=float(values['no_return_value']),
        **fields,
    )


class LidarModelNode(Node):
    def __init__(self) -> None:
        super().__init__('lidar_model')
        p = declare(self, {
            'sim_config': '',
            'truth_odom_topic': 'ego_racecar/odom',
            'scan_topic': 'scan',
            'frame_id': 'laser',
            'mount_x': 0.27,
            'mount_y': 0.0,
            'mount_yaw': 0.0,
            'rate_hz': 40.0,
            'fov_deg': 270.0,
            'resolution_deg': 0.2,
            'blocks': 8,
            'publish_delay': 0.002,
            'stamp_at': 'start',
            'seed': 0,
        })
        effects = declare(self, EFFECT_DEFAULTS)

        if not p['sim_config']:
            raise ValueError('sim_config (path of the gym config) is required')
        if p['stamp_at'] not in ('start', 'end'):
            raise ValueError("stamp_at must be 'start' or 'end'")
        if p['blocks'] < 1 or p['publish_delay'] < 0.0:
            raise ValueError('blocks must be >= 1 and publish_delay >= 0')
        if p['rate_hz'] not in SL450_RATES_HZ:
            self.get_logger().warning(f"rate_hz {p['rate_hz']} is not an SL450 rate {SL450_RATES_HZ}")

        self._geometry: ScanGeometry = scan_geometry(p['rate_hz'], p['fov_deg'], p['resolution_deg'])
        self._effects: EffectsConfig = effects_config(effects)
        self._mount: Pose2D = Pose2D(p['mount_x'], p['mount_y'], p['mount_yaw'])
        self._blocks: int = p['blocks']
        self._publish_delay: float = p['publish_delay']
        self._stamp_at_start: bool = p['stamp_at'] == 'start'
        self._frame_id: str = p['frame_id']
        self._rng: np.random.Generator = np.random.default_rng(p['seed'] or None)

        self._caster: BlockCaster = BlockCaster(
            load_gym_track(p['sim_config']), self._geometry, self._effects.range_max
        )
        self._rel_clock: RelativeClock = RelativeClock(self)
        # Extrapolation covers the age of the newest truth pose (physics step 10 ms + publish 4 ms).
        self._poses: PoseBuffer = PoseBuffer(horizon=0.5, max_extrapolation=0.03)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._scan_pub = self.create_publisher(LaserScan, p['scan_topic'], 10)
        self._odom_sub = self.create_subscription(Odometry, p['truth_odom_topic'], self._on_truth, 50)
        self._timer = self.create_timer(self._geometry.scan_time, self._on_timer)
        g = self._geometry
        self.get_logger().info(
            f'{g.num_beams} beams at {p["rate_hz"]} Hz, sweep {g.sweep_time * 1e3:.2f} ms, '
            f'{self._blocks} blocks, map from {p["sim_config"]}'
        )

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        try:
            config = effects_config(with_updates(self, EFFECT_DEFAULTS, params))
            if config.range_max != self._effects.range_max:
                raise ValueError('range_max cannot change at run time')
        except ValueError as ex:
            return SetParametersResult(successful=False, reason=str(ex))
        self._effects = config
        return SetParametersResult(successful=True)

    def _on_truth(self, msg: Odometry) -> None:
        pose = Pose2D(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(msg.pose.pose.orientation),
        )
        # The bridge republishes each physics step several times: keep the first of each.
        if pose == self._poses.newest():
            return
        twist = Twist2D(msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z)
        self._poses.add(self._rel_clock.from_msg(msg.header.stamp), pose, twist)

    def _pose_at(self, stamp: float) -> Pose2D:
        pose = self._poses.sample(stamp)
        assert pose is not None  # _on_timer returns early on an empty buffer
        return pose

    def _on_timer(self) -> None:
        if len(self._poses) == 0:
            return
        g = self._geometry
        t_end = self._rel_clock.now() - self._publish_delay
        t_start = t_end - g.sweep_time
        clean = distorted_scan(self._caster.cast, self._pose_at, self._mount, g, self._blocks, t_start)

        scan = LaserScan()
        scan.header.stamp = self._rel_clock.to_msg(t_start if self._stamp_at_start else t_end)
        scan.header.frame_id = self._frame_id
        scan.angle_min = g.angle_min
        scan.angle_max = g.angle_max
        scan.angle_increment = g.angle_increment
        scan.time_increment = g.time_increment
        scan.scan_time = g.scan_time
        scan.range_min = self._effects.range_min
        scan.range_max = self._effects.range_max
        ranges = apply_effects(clean, g.angle_increment, self._effects, self._rng)
        scan.ranges = array('f', ranges.tobytes())  # rclpy takes array('f') for float32[], not numpy
        self._scan_pub.publish(scan)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LidarModelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()  # no rclpy.shutdown(): after a signal the context is already down


if __name__ == '__main__':
    main()
