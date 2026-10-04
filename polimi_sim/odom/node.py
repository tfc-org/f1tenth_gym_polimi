"""odom_model: VESC-like /odom and TF odom -> base_link from the gym's true speed."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from polimi_sim.common.ros_utils import (
    RelativeClock,
    declare,
    quaternion_from_yaw,
    with_updates,
    yaw_from_quaternion,
)
from polimi_sim.common.se2 import Pose2D, compose, inverse
from polimi_sim.odom.model import OdomConfig, WheelOdometry

MAP_TO_ODOM_MODES: tuple[str, ...] = ('off', 'static', 'truth')


class OdomModelNode(Node):
    def __init__(self) -> None:
        super().__init__('odom_model')
        p = declare(self, {
            'truth_odom_topic': 'ego_racecar/odom',
            'steering_command_topic': 'sim/steering_command',
            'odom_topic': 'odom',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'publish_tf': True,
            'rate_hz': 50.0,
            'seed': 0,
            'debug_map_to_odom': 'off',
            # x, y, z, roll, pitch, yaw. Defaults are vesc_to_odom's (it leaves the twist at 0).
            'pose_covariance_diagonal': [0.2, 0.2, 0.0, 0.0, 0.0, 0.4],
            'twist_covariance_diagonal': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        self._tunable: tuple[str, ...] = tuple(asdict(OdomConfig()))
        config = OdomConfig(**declare(self, asdict(OdomConfig())))
        if p['rate_hz'] <= 0.0:
            raise ValueError('rate_hz must be > 0')
        if p['debug_map_to_odom'] not in MAP_TO_ODOM_MODES:
            raise ValueError(f'debug_map_to_odom must be one of {MAP_TO_ODOM_MODES}')
        for name in ('pose_covariance_diagonal', 'twist_covariance_diagonal'):
            if len(p[name]) != 6:
                raise ValueError(f'{name} needs 6 values')

        self._odometry: WheelOdometry = WheelOdometry(config, np.random.default_rng(p['seed'] or None))
        self._rel_clock: RelativeClock = RelativeClock(self)
        self._odom_frame: str = p['odom_frame']
        self._base_frame: str = p['base_frame']
        self._pose_covariance: list[float] = list(p['pose_covariance_diagonal'])
        self._twist_covariance: list[float] = list(p['twist_covariance_diagonal'])
        self._true_speed: float | None = None
        self._pin_static: bool = True
        self._true_pose: Pose2D = Pose2D(0.0, 0.0, 0.0)
        self._map_frame: str = 'map'
        self._track_truth: bool = p['debug_map_to_odom'] == 'truth'
        self._steering_command: float | None = None
        self._last_update: float | None = None

        self._tf: TransformBroadcaster | None = (
            TransformBroadcaster(self) if p['publish_tf'] or self._track_truth else None
        )
        self._publish_odom_tf: bool = p['publish_tf']
        self._static_tf: StaticTransformBroadcaster | None = (
            StaticTransformBroadcaster(self) if p['debug_map_to_odom'] == 'static' else None
        )

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._odom_pub = self.create_publisher(Odometry, p['odom_topic'], 10)
        self._truth_sub = self.create_subscription(Odometry, p['truth_odom_topic'], self._on_truth, 10)
        self._steer_sub = self.create_subscription(
            Float64, p['steering_command_topic'], self._on_steering_command, 10
        )
        self._reset_srv = self.create_service(Empty, '~/reset', self._on_reset)
        self._timer = self.create_timer(1.0 / p['rate_hz'], self._on_timer)

    def _on_reset(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        """Zero the odometry, e.g. after the car was moved by hand with /initialpose."""
        self._odometry.reset()
        self._pin_static = True
        self.get_logger().info('odometry reset to the origin')
        return response

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        try:
            self._odometry.config = OdomConfig(**with_updates(self, self._tunable, params))
        except ValueError as ex:
            return SetParametersResult(successful=False, reason=str(ex))
        return SetParametersResult(successful=True)

    def _on_truth(self, msg: Odometry) -> None:
        if self._pin_static and self._static_tf is not None:
            # Odometry starts at 0: pin odom on the true pose of that moment so both can be overlaid.
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = msg.header.frame_id
            tf.child_frame_id = self._odom_frame
            tf.transform.translation.x = msg.pose.pose.position.x
            tf.transform.translation.y = msg.pose.pose.position.y
            tf.transform.rotation = msg.pose.pose.orientation
            self._static_tf.sendTransform(tf)
        self._pin_static = False
        self._true_speed = msg.twist.twist.linear.x
        self._map_frame = msg.header.frame_id
        self._true_pose = Pose2D(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(msg.pose.pose.orientation),
        )

    def _on_steering_command(self, msg: Float64) -> None:
        self._steering_command = msg.data

    def _on_timer(self) -> None:
        # Like vesc_to_odom: nothing until the VESC reports and a servo command was seen.
        if self._true_speed is None or self._steering_command is None:
            return
        now = self._rel_clock.now()
        dt = 0.0 if self._last_update is None else now - self._last_update
        self._last_update = now
        state = self._odometry.update(dt, self._true_speed, self._steering_command)
        stamp = self._rel_clock.to_msg(now)
        orientation = quaternion_from_yaw(state.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = state.x
        odom.pose.pose.position.y = state.y
        odom.pose.pose.orientation = orientation
        odom.twist.twist.linear.x = state.speed
        odom.twist.twist.angular.z = state.yaw_rate
        for i in range(6):
            odom.pose.covariance[7 * i] = self._pose_covariance[i]
            odom.twist.covariance[7 * i] = self._twist_covariance[i]
        self._odom_pub.publish(odom)

        if self._tf is not None and self._publish_odom_tf:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = state.x
            tf.transform.translation.y = state.y
            tf.transform.rotation = orientation
            self._tf.sendTransform(tf)

        if self._tf is not None and self._track_truth:
            # A perfect localizer: map -> odom that puts base_link on the true pose.
            correction = compose(self._true_pose, inverse(Pose2D(state.x, state.y, state.yaw)))
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self._map_frame
            tf.child_frame_id = self._odom_frame
            tf.transform.translation.x = correction.x
            tf.transform.translation.y = correction.y
            tf.transform.rotation = quaternion_from_yaw(correction.yaw)
            self._tf.sendTransform(tf)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = OdomModelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()  # no rclpy.shutdown(): after a signal the context is already down


if __name__ == '__main__':
    main()
