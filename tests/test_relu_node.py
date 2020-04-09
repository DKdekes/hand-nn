import pytest
import numpy as np
from hand.node.ReluNode import ReluNode
np.random.seed(1)


@pytest.fixture
def relu_node():
    return ReluNode(3)


def test_compute(relu_node):
    x = [0.2, 0.4, 0.6]
    expected = 0.6739353958396248
    res = relu_node.compute(x)
    assert expected == res


