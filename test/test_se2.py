import math

import pytest

from polimi_sim.se2 import (
    Pose2D,
    PoseBuffer,
    Twist2D,
    compose,
    extrapolate,
    interpolate,
    inverse,
    wrap_angle,
)


def test_wrap_angle() -> None:
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi / 2.0 - 2.0 * math.pi) == pytest.approx(-math.pi / 2.0)


def test_compose_rotates_then_translates() -> None:
    laser = compose(Pose2D(1.0, 2.0, math.pi / 2.0), Pose2D(0.27, 0.0, 0.0))
    assert (laser.x, laser.y, laser.yaw) == pytest.approx((1.0, 2.27, math.pi / 2.0))


def test_inverse_undoes_compose() -> None:
    a = Pose2D(1.5, -2.0, 2.5)
    for pose in (compose(a, inverse(a)), compose(inverse(a), a)):
        assert (pose.x, pose.y, pose.yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    truth, odom = Pose2D(3.0, 4.0, 1.0), Pose2D(0.5, 0.2, -0.3)
    back = compose(compose(truth, inverse(odom)), odom)  # map -> odom, then odom -> base_link
    assert (back.x, back.y, back.yaw) == pytest.approx((truth.x, truth.y, truth.yaw))


def test_interpolate_position_and_ends() -> None:
    a, b = Pose2D(0.0, 0.0, 0.0), Pose2D(2.0, -4.0, 1.0)
    assert interpolate(a, b, 0.0) == a
    mid = interpolate(a, b, 0.25)
    assert (mid.x, mid.y, mid.yaw) == pytest.approx((0.5, -1.0, 0.25))


def test_interpolate_yaw_takes_the_short_way_across_pi() -> None:
    a, b = Pose2D(0.0, 0.0, math.radians(170.0)), Pose2D(0.0, 0.0, math.radians(-170.0))
    assert abs(interpolate(a, b, 0.5).yaw) == pytest.approx(math.pi)
    assert interpolate(a, b, 0.25).yaw == pytest.approx(math.radians(175.0))


def test_extrapolate_uses_body_twist() -> None:
    pose = extrapolate(Pose2D(0.0, 0.0, math.pi / 2.0), Twist2D(2.0, 0.0, 1.0), 0.1)
    assert (pose.x, pose.y, pose.yaw) == pytest.approx((0.0, 0.2, math.pi / 2.0 + 0.1))


def test_buffer_interpolates_and_clamps() -> None:
    buf = PoseBuffer(horizon=1.0, max_extrapolation=0.02)
    assert buf.sample(0.0) is None
    buf.add(0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(1.0))
    buf.add(0.1, Pose2D(0.1, 0.0, 0.0), Twist2D(1.0))
    before, inside = buf.sample(-1.0), buf.sample(0.05)
    assert before == Pose2D(0.0, 0.0, 0.0)
    assert inside is not None and inside.x == pytest.approx(0.05)


def test_buffer_extrapolation_is_capped() -> None:
    buf = PoseBuffer(horizon=1.0, max_extrapolation=0.02)
    buf.add(0.0, Pose2D(0.0, 0.0, 0.0), Twist2D(1.0))
    near, far = buf.sample(0.01), buf.sample(5.0)
    assert near is not None and near.x == pytest.approx(0.01)
    assert far is not None and far.x == pytest.approx(0.02)


def test_buffer_drops_old_and_out_of_order() -> None:
    buf = PoseBuffer(horizon=0.55, max_extrapolation=0.0)
    for i in range(20):
        assert buf.add(0.1 * i, Pose2D(float(i), 0.0, 0.0), Twist2D())
    assert not buf.add(1.0, Pose2D(99.0, 0.0, 0.0), Twist2D())
    assert len(buf) == 6  # 1.4 .. 1.9
    oldest = buf.sample(0.0)
    assert oldest is not None and oldest.x == pytest.approx(14.0)
