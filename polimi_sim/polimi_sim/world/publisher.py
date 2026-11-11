"""world_publisher: the tape as markers on /sim/truth/tape, so Foxglove draws the track under the car."""
from __future__ import annotations

import rclpy
import yaml
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from polimi_sim.common.ros_utils import declare


class WorldPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__('world_publisher')
        p = declare(self, {'tape': '', 'truth_frame': 'sim_world'})
        if not p['tape']:
            raise ValueError('tape (path of the world tape.yaml) is required')
        with open(p['tape']) as tape_file:
            tape = yaml.safe_load(tape_file)
        r, g, b = (float(c) / 255.0 for c in tape['rgb'])
        markers = MarkerArray()
        for i, line in enumerate(tape['lines']):
            marker = Marker()
            marker.header.frame_id = p['truth_frame']
            marker.ns, marker.id, marker.type = 'tape', i, Marker.LINE_STRIP
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(tape['tape_width'])
            marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            marker.points = [Point(x=float(x), y=float(y), z=0.001) for x, y in [*line, line[0]]]  # closed
            markers.markers.append(marker)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(MarkerArray, '/sim/truth/tape', latched)
        self._pub.publish(markers)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = WorldPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
