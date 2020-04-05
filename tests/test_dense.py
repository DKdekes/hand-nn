import pytest
from numpy import random
random.seed(1)
from hand.layer import DenseLayer


@pytest.fixture
def dense_layer():
    return DenseLayer.DenseLayer(1, 1)


def test_compute(dense_layer):
    print()
    x = 1
    print([node.w for node in dense_layer.nodes])
    output = dense_layer.compute(x)
    print(output)
