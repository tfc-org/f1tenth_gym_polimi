"""Small helpers shared by the polimi_sim nodes."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time


class RelativeClock:
    """Node time as float seconds since the node started (keeps float64 precise)."""

    def __init__(self, node: Node) -> None:
        self._node: Node = node
        self._origin_ns: int = node.get_clock().now().nanoseconds

    def now(self) -> float:
        return (self._node.get_clock().now().nanoseconds - self._origin_ns) * 1e-9

    def from_msg(self, stamp: TimeMsg) -> float:
        return (Time.from_msg(stamp).nanoseconds - self._origin_ns) * 1e-9

    def to_msg(self, seconds: float) -> TimeMsg:
        stamp: TimeMsg = Time(nanoseconds=self._origin_ns + round(seconds * 1e9)).to_msg()
        return stamp


def yaw_from_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quaternion_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def declare(node: Node, defaults: dict[str, Any]) -> dict[str, Any]:
    """Declare typed parameters and return their values."""
    return {name: node.declare_parameter(name, default).value for name, default in defaults.items()}


def with_updates(node: Node, names: Iterable[str], updates: list[Parameter]) -> dict[str, Any]:
    """Current values of `names`, overridden by the parameters being set."""
    values = {name: node.get_parameter(name).value for name in names}
    values.update({p.name: p.value for p in updates if p.name in values})
    return values
