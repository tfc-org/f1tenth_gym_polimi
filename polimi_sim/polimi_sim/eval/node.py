"""track_monitor: progress, laps, lateral offset, off-track status and simulator health from the truth."""
from __future__ import annotations

import os
import tempfile
import time
from typing import TextIO

import numpy as np
import rclpy
import yaml
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int32

from polimi_sim.common.ros_utils import RelativeClock, declare, yaw_from_quaternion
from polimi_sim.common.se2 import Pose2D
from polimi_sim.common.truth import ContactDebounce, TeleportDetector
from polimi_sim.eval.monitor import MOVING, LapRecord, MonitorConfig, TrackMonitor

CSV_HEADER = 'lap,time_s,valid,max_offset_m,time_off_track_s\n'


class TrackMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('track_monitor')
        p = declare(self, {
            'tape': '',
            'world': 'world',
            'lr': 0.17145,
            'track_width': 0.281,
            'checkpoints': 10,
            'search_window': 1.0,
            'contact_debounce': 0.3,
            'truth_odom_topic': 'ego_racecar/odom',
            'collision_topic': 'ego_racecar/collision',
            'log_dir': '~/ws/runs',
        })
        if not p['tape']:
            raise ValueError('tape (path of the world tape.yaml) is required')
        with open(p['tape']) as tape_file:
            tape = yaml.safe_load(tape_file)
        config = MonitorConfig(
            width=float(tape['width']), tape_width=float(tape['tape_width']),
            track_width=p['track_width'], lr=p['lr'], checkpoints=p['checkpoints'], search_window=p['search_window'],
        )
        self._monitor = TrackMonitor(np.asarray(tape['centerline'], dtype=float), float(tape['start_s']), config)
        self._teleports = TeleportDetector()
        self._contacts = ContactDebounce(p['contact_debounce'])
        self._rel_clock = RelativeClock(self)
        self._world: str = p['world']
        self._log_dir: str = os.path.expanduser(p['log_dir'])
        self._csv: TextIO | None = None
        self._collisions = 0
        self._last_pose: Pose2D | None = None
        self._odom_msgs = 0
        self._distinct_moving = 0
        self._moving_time = 0.0
        self._last_stamp: float | None = None

        self._pubs = {name: self.create_publisher(kind, f'/sim/eval/{name}', 10) for name, kind in (
            ('on_track', Bool), ('lateral_offset', Float64), ('progress', Float64),
            ('lap', Int32), ('lap_time', Float64), ('collisions', Int32))}
        self._physics_rate = self.create_publisher(Float64, '/sim/health/physics_rate', 10)
        self._odom_rate = self.create_publisher(Float64, '/sim/health/odom_rate', 10)
        self.create_subscription(Odometry, p['truth_odom_topic'], self._on_truth, 50)
        self.create_subscription(Bool, p['collision_topic'], self._on_collision, 50)
        self.create_timer(1.0, self._on_health)
        self.get_logger().info(
            f'lap {self._monitor.length:.2f} m, off track past {config.off_track_offset:.3f} m from the centre line'
        )

    def _on_collision(self, msg: Bool) -> None:
        if self._contacts.update(self._rel_clock.now(), msg.data) == 'start':
            self._collisions += 1
            self._monitor.contact()
            self.get_logger().info(f'contact {self._collisions}: lap invalid')

    def _on_truth(self, msg: Odometry) -> None:
        self._odom_msgs += 1
        pose = Pose2D(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_from_quaternion(msg.pose.pose.orientation))
        if pose == self._last_pose:
            return  # the bridge republishes each physics step
        stamp = self._rel_clock.from_msg(msg.header.stamp)
        if self._last_stamp is not None and abs(msg.twist.twist.linear.x) > MOVING:
            self._distinct_moving += 1
            self._moving_time += stamp - self._last_stamp
        self._last_pose, self._last_stamp = pose, stamp
        if self._teleports.update(stamp, pose):
            self._monitor.jump()
            self.get_logger().info('truth jumped (a reset): the lap restarts at the next start line')
        st = self._monitor.update(stamp, pose.x, pose.y, pose.yaw, msg.twist.twist.linear.x)
        for record in st.finished:
            self._write_lap(record)
        values = (Bool(data=st.on_track), Float64(data=st.lateral_offset), Float64(data=st.progress),
                  Int32(data=st.lap), Float64(data=st.lap_time), Int32(data=self._collisions))
        for pub, value in zip(self._pubs.values(), values, strict=True):
            pub.publish(value)

    def _on_health(self) -> None:
        self._odom_rate.publish(Float64(data=float(self._odom_msgs)))
        self._odom_msgs = 0
        if self._moving_time > 0.5:
            rate = self._distinct_moving / self._moving_time
            self._physics_rate.publish(Float64(data=rate))
            if rate < 95.0:
                self.get_logger().warning(f'physics at {rate:.0f} Hz, expected 100: the gym is starved')
        self._distinct_moving, self._moving_time = 0, 0.0

    def _write_lap(self, lap: LapRecord) -> None:
        if self._csv is None:
            directory = self._log_dir
            if not (os.path.isdir(directory) and os.access(directory, os.W_OK)):
                directory = os.path.join(tempfile.gettempdir(), 'polimi_sim')
                os.makedirs(directory, exist_ok=True)
                self.get_logger().warning(f'{self._log_dir} is not writable: laps go to {directory}')
            path = os.path.join(directory, f'laps_{self._world}_{time.strftime("%Y%m%d-%H%M%S")}.csv')
            self._csv = open(path, 'w')
            self._csv.write(CSV_HEADER)
            self.get_logger().info(f'laps -> {path}')
        self._csv.write(f'{lap.lap},{lap.time:.3f},{str(lap.valid).lower()},{lap.max_offset:.3f},{lap.time_off_track:.3f}\n')
        self._csv.flush()
        self.get_logger().info(f'lap {lap.lap}: {lap.time:.2f} s, {"valid" if lap.valid else "invalid"}')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrackMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
