import math

from polimi_sim.world.track import centerline, parse_primitives


def test_the_heading_is_wrapped_on_a_clockwise_loop() -> None:
    loop = [{'straight': 2.0}, {'arc': {'radius': 1.0, 'angle': -180}}, {'straight': 2.0}, {'arc': {'radius': 1.0, 'angle': -180}}]
    line = centerline(parse_primitives(loop), (0.0, 0.0, 0.0))
    headings = [line.pose_at(s)[2] for s in [i * line.length / 200 for i in range(200)]]
    assert all(-math.pi <= h <= math.pi for h in headings)
