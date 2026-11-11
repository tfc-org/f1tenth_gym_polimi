"""sim_reset: put the car somewhere. /sim/reset (start pose) and /sim/reset_pose (rear axle, truth frame)
publish on the gym's /sim/initialpose, then zero /odom, as after moving the car by hand."""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Empty

from polimi_sim.common.ros_utils import declare, quaternion_from_yaw


class SimResetNode(Node):
    def __init__(self) -> None:
        super().__init__('sim_reset')
        p = declare(self, {
            'sx': 0.0, 'sy': 0.0, 'stheta': 0.0,
            'truth_frame': 'sim_world',
            'initialpose_topic': '/sim/initialpose',
            'odom_reset_service': '/odom_model/reset',
        })
        self._start = Pose()
        self._start.position.x, self._start.position.y = p['sx'], p['sy']
        self._start.orientation = quaternion_from_yaw(p['stheta'])
        self._frame: str = p['truth_frame']
        self._pub = self.create_publisher(PoseWithCovarianceStamped, p['initialpose_topic'], 10)
        self._odom_reset = self.create_client(Empty, p['odom_reset_service'])
        self.create_service(Empty, '/sim/reset', self._on_reset)
        self.create_subscription(PoseStamped, '/sim/reset_pose', self._on_reset_pose, 10)

    def _put(self, pose: Pose) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame
        msg.pose.pose = pose
        self._pub.publish(msg)
        if self._odom_reset.service_is_ready():
            self._odom_reset.call_async(Empty.Request())
        else:
            self.get_logger().warning('odom_model/reset is not available: /odom was not zeroed')
        p = pose.position
        self.get_logger().info(f'car put at ({p.x:.2f}, {p.y:.2f})')

    def _on_reset(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        self._put(self._start)
        return response

    def _on_reset_pose(self, msg: PoseStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id != self._frame:
            self.get_logger().warning(f'reset_pose in {msg.header.frame_id!r}, taken as {self._frame!r}')
        self._put(msg.pose)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SimResetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
