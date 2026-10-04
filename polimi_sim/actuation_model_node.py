"""actuation_model: mux output -> gym drive command, through VESC and servo limits."""
from __future__ import annotations

from dataclasses import asdict

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64

from polimi_sim.actuation import ActuationConfig, ActuationModel
from polimi_sim.ros_utils import RelativeClock, declare, with_updates


class ActuationModelNode(Node):
    def __init__(self) -> None:
        super().__init__('actuation_model')
        p = declare(self, {
            'input_topic': 'ackermann_cmd',
            'output_topic': 'sim/drive',
            'steering_command_topic': 'sim/steering_command',
            'rate_hz': 100.0,
        })
        self._tunable: tuple[str, ...] = tuple(asdict(ActuationConfig()))
        config = ActuationConfig(**declare(self, asdict(ActuationConfig())))
        if p['rate_hz'] <= 0.0:
            raise ValueError('rate_hz must be > 0')

        self._model: ActuationModel = ActuationModel(config)
        self._rel_clock: RelativeClock = RelativeClock(self)

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._drive_pub = self.create_publisher(AckermannDriveStamped, p['output_topic'], 10)
        self._steer_pub = self.create_publisher(Float64, p['steering_command_topic'], 10)
        self._sub = self.create_subscription(
            AckermannDriveStamped, p['input_topic'], self._on_command, 10
        )
        self._timer = self.create_timer(1.0 / p['rate_hz'], self._on_timer)

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        try:
            self._model.config = ActuationConfig(**with_updates(self, self._tunable, params))
        except ValueError as ex:
            return SetParametersResult(successful=False, reason=str(ex))
        return SetParametersResult(successful=True)

    def _on_command(self, msg: AckermannDriveStamped) -> None:
        self._model.command(self._rel_clock.now(), msg.drive.speed, msg.drive.steering_angle)

    def _on_timer(self) -> None:
        now = self._rel_clock.now()
        out = self._model.step(now)
        drive = AckermannDriveStamped()
        drive.header.stamp = self._rel_clock.to_msg(now)
        drive.drive.speed = out.speed
        drive.drive.steering_angle = out.steering_angle
        self._drive_pub.publish(drive)
        self._steer_pub.publish(Float64(data=out.steering_command))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActuationModelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
