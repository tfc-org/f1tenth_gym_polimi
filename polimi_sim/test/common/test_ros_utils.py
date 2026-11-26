import pytest

pytest.importorskip('rclpy')

from rclpy.parameter import Parameter  # noqa: E402

from polimi_sim.common.ros_utils import with_updates  # noqa: E402


class _Node:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get_parameter(self, name: str) -> Parameter:
        return Parameter(name, value=self._values[name])


def test_a_tunable_parameter_is_merged() -> None:
    node = _Node({'noise_std': 0.01, 'dropout_prob': 0.002})
    assert with_updates(node, ('noise_std', 'dropout_prob'), [Parameter('noise_std', value=0.02)]) == {
        'noise_std': 0.02, 'dropout_prob': 0.002}


def test_a_parameter_read_only_at_start_is_refused() -> None:
    node = _Node({'noise_std': 0.01})
    with pytest.raises(ValueError, match='fixed at start, restart the node to change: blocks, rate_hz'):
        with_updates(node, ('noise_std',), [Parameter('rate_hz', value=20.0), Parameter('blocks', value=1)])
