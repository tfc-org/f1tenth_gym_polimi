import pytest

from polimi_sim.actuation.model import ActuationConfig, ActuationModel, DelayLine

DT = 0.01
FAST = dict(servo_rate=1e6, accel_max=1e6, brake_max=1e6)


def run(model: ActuationModel, t0: float, t1: float) -> list:
    steps = round((t1 - t0) / DT)
    return [model.step(t0 + i * DT) for i in range(steps + 1)]


def test_delay_line() -> None:
    line: DelayLine[int] = DelayLine()
    assert line.get(0.0, 0.1) is None
    line.push(0.0, 1)
    line.push(0.05, 2)
    assert line.get(0.09, 0.1) is None
    assert line.get(0.1, 0.1) == (pytest.approx(0.1), 1)
    assert line.get(0.2, 0.1) == (pytest.approx(0.15), 2)


def test_command_delay_is_exact_to_the_step() -> None:
    model = ActuationModel(ActuationConfig(command_delay=0.02, servo_delay=0.0, **FAST))
    model.step(0.0)
    model.command(0.0, 2.0, 0.1)
    assert model.step(0.01).speed == 0.0
    out = model.step(0.02)
    assert out.speed == pytest.approx(2.0) and out.steering_command == pytest.approx(0.1)


def test_timeout_stops_the_car_and_holds_the_steering() -> None:
    model = ActuationModel(ActuationConfig(command_delay=0.0, vesc_timeout=0.5, servo_delay=0.0, **FAST))
    model.command(0.0, 3.0, 0.2)
    outs = run(model, 0.0, 0.6)
    assert outs[50].speed == pytest.approx(3.0)   # t = 0.50, still within the timeout
    assert outs[51].speed == 0.0                  # t = 0.51
    assert outs[60].steering_angle == pytest.approx(0.2)


def test_no_command_means_no_motion() -> None:
    out = run(ActuationModel(ActuationConfig()), 0.0, 0.2)[-1]
    assert (out.speed, out.steering_angle, out.steering_command) == (0.0, 0.0, 0.0)


def test_servo_delay_and_slew_limit() -> None:
    cfg = ActuationConfig(command_delay=0.0, servo_delay=0.03, servo_rate=2.0, accel_max=1e6, brake_max=1e6)
    model = ActuationModel(cfg)
    model.step(0.0)
    model.command(0.0, 1.0, 0.4)
    outs = run(model, 0.01, 0.4)
    assert outs[0].steering_command == pytest.approx(0.4)   # sent to the VESC at once
    assert outs[2].steering_angle == 0.0                    # t = 0.03, servo has not reacted
    steers = [o.steering_angle for o in outs]
    assert max(b - a for a, b in zip(steers[:-1], steers[1:], strict=True)) <= 2.0 * DT + 1e-9
    assert steers[5] == pytest.approx(2.0 * 0.03)           # t = 0.06: moving since 0.04, 3 steps at 2 rad/s
    assert steers[-1] == pytest.approx(0.4)


def test_offset_applied_once_and_clamped() -> None:
    cfg = ActuationConfig(command_delay=0.0, servo_delay=0.0, steer_offset=0.05, max_steer=0.4, **FAST)
    model = ActuationModel(cfg)
    model.command(0.0, 0.0, 0.1)
    outs = run(model, 0.0, 0.1)
    assert outs[-1].steering_command == pytest.approx(0.15)
    assert outs[-1].steering_angle == pytest.approx(0.15)
    model.command(0.1, 0.0, 0.39)
    assert model.step(0.11).steering_command == pytest.approx(0.4)


def test_accel_and_brake_limits() -> None:
    cfg = ActuationConfig(command_delay=0.0, vesc_timeout=10.0, accel_max=4.0, brake_max=6.0)
    model = ActuationModel(cfg)
    model.step(0.0)
    model.command(0.0, 2.0, 0.0)
    up = [o.speed for o in run(model, 0.01, 1.0)]
    assert max(b - a for a, b in zip(up[:-1], up[1:], strict=True)) <= 4.0 * DT + 1e-9
    assert up[24] == pytest.approx(1.0) and up[-1] == pytest.approx(2.0)   # 0.25 s at 4 m/s^2
    model.command(1.0, 0.0, 0.0)
    down = [o.speed for o in run(model, 1.01, 2.0)]
    assert min(b - a for a, b in zip(down[:-1], down[1:], strict=True)) >= -6.0 * DT - 1e-9
    assert down[9] == pytest.approx(2.0 - 6.0 * 0.1) and down[-1] == 0.0


def test_reversing_brakes_to_zero_then_accelerates() -> None:
    cfg = ActuationConfig(command_delay=0.0, vesc_timeout=10.0, accel_max=4.0, brake_max=6.0)
    model = ActuationModel(cfg)
    model.command(0.0, 3.0, 0.0)
    run(model, 0.0, 1.0)
    model.command(1.0, -1.0, 0.0)
    speeds = [o.speed for o in run(model, 1.01, 2.0)]
    assert speeds[49] == pytest.approx(0.0, abs=1e-9)       # 3 m/s at 6 m/s^2 = 0.5 s
    assert speeds[59] == pytest.approx(-0.4)                # then 4 m/s^2
    assert speeds[-1] == pytest.approx(-1.0)


def test_config_is_validated() -> None:
    with pytest.raises(ValueError):
        ActuationConfig(servo_rate=0.0)
    with pytest.raises(ValueError):
        ActuationConfig(command_delay=-0.1)
