"""camera_model: the Gemini 335L colour stream drawn from the world at the truth pose, on the driver's topics."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from array import array

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from polimi_sim.camera.draw import Renderer
from polimi_sim.camera.intrinsics import camera_in_world, load_camera, optical_mount
from polimi_sim.camera.scene import load_scene
from polimi_sim.common.ros_utils import RelativeClock, declare, with_updates, yaw_from_quaternion
from polimi_sim.common.se2 import Pose2D, PoseBuffer, Twist2D
from polimi_sim.common.truth import TeleportDetector


def summary(values_ms: list[float]) -> dict[str, float]:
    a = np.asarray(values_ms)
    return {'mean': round(float(a.mean()), 3), 'p99': round(float(np.percentile(a, 99)), 3), 'max': round(float(a.max()), 3)}


class CameraModelNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_model')
        p = declare(self, {
            'scene': '',
            'tape': '',
            'camera_file': '',
            'profile': '',
            'mount_x': 0.2,
            'mount_y': 0.0,
            'mount_z': 0.2,
            'mount_roll': 0.0,
            'mount_pitch': 0.2618,
            'mount_yaw': 0.0,
            'truth_odom_topic': 'ego_racecar/odom',
            'image_topic': 'camera/color/image_raw',
            'info_topic': 'camera/color/camera_info',
            'frame_id': 'camera_color_optical_frame',
            'preview_rate_hz': 10.0,
            'preview_scale': 0.5,
            'preview_quality': 80,
            'opencv_threads': 2,
            'world': 'world',
            'benchmark_dir': '~/ws/runs/benchmarks',
            'benchmark_after': 60.0,
        })
        if not (p['scene'] and p['tape'] and p['camera_file']):
            raise ValueError('scene, tape and camera_file are required: the launch takes them from the world and the overlay')
        cv2.setNumThreads(p['opencv_threads'])
        camera = load_camera(p['camera_file'], p['profile'])
        k = camera.intrinsics
        mount = {key: float(p[f'mount_{key}']) for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
        self._camera = camera
        self._mount = optical_mount(mount, camera.color_in_link)
        self._renderer = Renderer(load_scene(p['scene'], p['tape']), k)
        self._period = 1.0 / camera.rate_hz
        self._preview_period = 1.0 / p['preview_rate_hz'] if p['preview_rate_hz'] > 0.0 else float('inf')
        self._preview_scale: float = p['preview_scale']
        self._preview_quality: int = p['preview_quality']
        self._frame_id: str = p['frame_id']
        self._world: str = p['world']
        self._benchmark_dir: str = os.path.expanduser(p['benchmark_dir'])
        self._benchmark_after: float = p['benchmark_after']

        self._lock = threading.Lock()
        self._poses = PoseBuffer(horizon=0.5, max_extrapolation=0.03)
        self._teleports = TeleportDetector()
        self._rel_clock = RelativeClock(self)
        self._last_slot = -1
        self._last_preview = -float('inf')
        self._first_stamp: float | None = None
        self._draw_ms: list[float] = []
        self._frame_ms: list[float] = []
        self._benchmark_done = self._benchmark_after <= 0.0

        self._image = Image(height=k.height, width=k.width, encoding='rgb8', is_bigendian=0, step=k.width * 3)
        self._image.header.frame_id = self._frame_id
        self._info = CameraInfo(height=k.height, width=k.width, distortion_model='plumb_bob', d=list(k.d),
                                k=[k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0],
                                r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                p=[k.fx, 0.0, k.cx, 0.0, 0.0, k.fy, k.cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        self._info.header.frame_id = self._frame_id
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._image_pub = self.create_publisher(Image, p['image_topic'], 10)
        self._info_pub = self.create_publisher(CameraInfo, p['info_topic'], 10)
        self._preview_pub = self.create_publisher(CompressedImage, p['image_topic'] + '/compressed', 10)
        self.create_subscription(Odometry, p['truth_odom_topic'], self._on_truth, 50,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(self._period / 4.0, self._on_timer, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f'{camera.profile}: {k.width}x{k.height} rgb8 at {camera.rate_hz:g} Hz on {p["image_topic"]}, '
            f'preview on {p["image_topic"]}/compressed while subscribed'
        )

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        try:
            with_updates(self, (), params)
        except ValueError as ex:
            return SetParametersResult(successful=False, reason=str(ex))
        return SetParametersResult(successful=True)

    def _on_truth(self, msg: Odometry) -> None:
        pose = Pose2D(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_from_quaternion(msg.pose.pose.orientation))
        twist = Twist2D(msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z)
        stamp = self._rel_clock.from_msg(msg.header.stamp)
        with self._lock:
            if pose == self._poses.newest():
                return
            if self._teleports.update(stamp, pose):
                self._poses.clear()
            self._poses.add(stamp, pose, twist)

    def _on_timer(self) -> None:
        """Frames on a fixed grid of the profile's period: each stamp is slot / rate, drawn at the truth pose of that instant."""
        slot = int(self._rel_clock.now() / self._period)
        if slot <= self._last_slot:
            return
        self._last_slot = slot
        stamp = slot * self._period
        with self._lock:
            pose = self._poses.sample(stamp)
        if pose is None:
            return
        start = time.perf_counter()
        img = self._renderer.render(*camera_in_world(pose.x, pose.y, pose.yaw, self._mount))
        drawn = time.perf_counter()
        header_stamp = self._rel_clock.to_msg(stamp)
        data = array('B')
        data.frombytes(memoryview(img).cast('B'))
        self._image.data = data
        self._image.header.stamp = header_stamp
        self._info.header.stamp = header_stamp
        self._image_pub.publish(self._image)
        self._info_pub.publish(self._info)
        if stamp - self._last_preview >= self._preview_period - 1e-6 and self._preview_pub.get_subscription_count() > 0:
            self._last_preview = stamp
            self._publish_preview(img, header_stamp)
        done = time.perf_counter()
        self._account(stamp, (drawn - start) * 1e3, (done - start) * 1e3)

    def _publish_preview(self, img: np.ndarray, stamp: TimeMsg) -> None:
        small = cv2.resize(img, None, fx=self._preview_scale, fy=self._preview_scale, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self._preview_quality])
        if ok:
            msg = CompressedImage(format='jpeg', data=jpeg.tobytes())
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            self._preview_pub.publish(msg)

    def _account(self, stamp: float, draw_ms: float, frame_ms: float) -> None:
        if frame_ms > self._period * 1e3:
            self.get_logger().warning(f'frame took {frame_ms:.1f} ms, over the {self._period * 1e3:.1f} ms period',
                                      throttle_duration_sec=5.0)
        if self._benchmark_done:
            return
        if self._first_stamp is None:
            self._first_stamp = stamp
        self._draw_ms.append(draw_ms)
        self._frame_ms.append(frame_ms)
        if stamp - self._first_stamp >= self._benchmark_after:
            self._benchmark_done = True
            self._write_benchmark(stamp - self._first_stamp)

    def _write_benchmark(self, seconds: float) -> None:
        k = self._camera.intrinsics
        result: dict[str, object] = {
            'world': self._world, 'profile': self._camera.profile, 'width': k.width, 'height': k.height,
            'rate_hz': self._camera.rate_hz, 'frames': len(self._frame_ms), 'seconds': round(seconds, 2),
            'draw_ms': summary(self._draw_ms), 'frame_ms': summary(self._frame_ms),
            'over_period': sum(ms > self._period * 1e3 for ms in self._frame_ms), 'cpus': os.cpu_count(),
        }
        directory = self._benchmark_dir
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError:
            pass
        if not (os.path.isdir(directory) and os.access(directory, os.W_OK)):
            directory = os.path.join(tempfile.gettempdir(), 'polimi_sim')
            os.makedirs(directory, exist_ok=True)
            self.get_logger().warning(f'{self._benchmark_dir} is not writable: the benchmark goes to {directory}')
        path = os.path.join(directory, f'camera_{self._world}_{self._camera.profile}_{time.strftime("%Y%m%d-%H%M%S")}.json')
        with open(path, 'w') as out:
            json.dump(result, out, indent=2)
        f = summary(self._frame_ms)
        self.get_logger().info(f'benchmark -> {path}: {len(self._frame_ms)} frames, draw and publish mean {f["mean"]:.1f} ms, '
                               f'p99 {f["p99"]:.1f} ms, max {f["max"]:.1f} ms')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraModelNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
