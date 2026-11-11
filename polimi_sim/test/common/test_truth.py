import math

from polimi_sim.common.se2 import Pose2D
from polimi_sim.common.truth import ContactDebounce, TeleportDetector

STEP = 0.01  # s, one physics step


def test_the_first_pose_is_not_a_teleport() -> None:
    assert TeleportDetector().update(0.0, Pose2D(5.0, 5.0, 1.0)) is False


def test_a_fast_but_continuous_pose_is_not_a_teleport() -> None:
    detector = TeleportDetector()
    jumps = [detector.update(i * STEP, Pose2D(15.0 * i * STEP, 0.0, 0.0)) for i in range(100)]
    assert not any(jumps)


def test_a_reset_is_a_teleport_once() -> None:
    detector = TeleportDetector()
    detector.update(0.0, Pose2D(0.0, 0.0, 0.0))
    assert detector.update(STEP, Pose2D(3.0, 1.0, 0.0)) is True
    assert detector.update(2 * STEP, Pose2D(3.0, 1.0, 0.0)) is False


def test_a_heading_flip_in_place_is_a_teleport() -> None:
    detector = TeleportDetector()
    detector.update(0.0, Pose2D(1.0, 1.0, 0.0))
    assert detector.update(STEP, Pose2D(1.0, 1.0, math.pi)) is True


def test_the_same_pose_after_a_long_gap_is_not_a_teleport() -> None:
    detector = TeleportDetector()
    detector.update(0.0, Pose2D(1.0, 1.0, 3.1))
    assert detector.update(5.0, Pose2D(1.0, 1.0, -3.1)) is False


def test_a_collision_is_one_contact_with_a_start_and_an_end() -> None:
    contact = ContactDebounce(debounce=0.1)
    events = [contact.update(t * 0.004, t < 50) for t in range(200)]
    assert [e for e in events if e] == ['start', 'end']
    assert 74 <= events.index('end') <= 75  # 0.1 s after the last True sample at 49


def test_a_chattering_flag_is_one_contact() -> None:
    contact = ContactDebounce(debounce=0.1)
    flags = [True, True, False, True, False, False, True] * 20 + [False] * 50
    events = [contact.update(i * 0.004, flag) for i, flag in enumerate(flags)]
    assert [e for e in events if e] == ['start', 'end']


def test_two_pushes_apart_are_two_contacts() -> None:
    contact = ContactDebounce(debounce=0.1)
    flags = [True] * 10 + [False] * 50 + [True] * 10 + [False] * 50
    events = [contact.update(i * 0.004, flag) for i, flag in enumerate(flags)]
    assert [e for e in events if e] == ['start', 'end', 'start', 'end']
