import pytest
import numpy as np
np.random.seed(1)
from hand.node.SigmoidNode import SigmoidNode


@pytest.fixture
def sigmoid_node():
    return SigmoidNode(1)


def test_compute(sigmoid_node):
    x = 1
    expected = 0.7571921198123589
    res = sigmoid_node.compute(x)
    print(res)
    assert expected == res, 'bad single value node propagation'
