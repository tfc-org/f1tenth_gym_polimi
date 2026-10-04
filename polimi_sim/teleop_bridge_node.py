"""teleop_bridge: /cmd_vel (Twist) -> /teleop (Ackermann), the mux's high-priority input."""
from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter

from polimi_sim.ros_utils import declare, with_updates
from polimi_sim.teleop import ANGULAR_MODES, twist_to_ackermann

TUNABLE: tuple[str, ...] = ('angular_mode', 'wheelbase', 'max_steer', 'min_speed')


class TeleopBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('teleop_bridge')
        p = declare(self, {
            'input_topic': 'cmd_vel',
            'output_topic': 'teleop',
            'angular_mode': 'steering_angle',
            'wheelbase': 0.3302,
            'max_steer': 0.4189,
            'min_speed': 0.5,
        })
        self._validate(p)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._pub = self.create_publisher(AckermannDriveStamped, p['output_topic'], 10)
        self._sub = self.create_subscription(Twist, p['input_topic'], self._on_twist, 10)

    @staticmethod
    def _validate(values: dict[str, object]) -> None:
        if values['angular_mode'] not in ANGULAR_MODES:
            raise ValueError(f'angular_mode must be one of {ANGULAR_MODES}')
        for name in ('wheelbase', 'max_steer', 'min_speed'):
            value = values[name]
            if not isinstance(value, float) or value <= 0.0:
                raise ValueError(f'{name} must be a float > 0')

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        try:
            self._validate(with_updates(self, TUNABLE, params))
        except ValueError as ex:
            return SetParametersResult(successful=False, reason=str(ex))
        return SetParametersResult(successful=True)

    def _on_twist(self, msg: Twist) -> None:
        # One message out per message in, never a republish: when the operator
        # lets go, the mux input times out and autonomy takes over again.
        speed, steer = twist_to_ackermann(
            msg.linear.x,
            msg.angular.z,
            self.get_parameter('angular_mode').value,
            self.get_parameter('wheelbase').value,
            self.get_parameter('max_steer').value,
            self.get_parameter('min_speed').value,
        )
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.speed = speed
        out.drive.steering_angle = steer
        self._pub.publish(out)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TeleopBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
